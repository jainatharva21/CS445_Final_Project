import collections
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.draw import disk


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name="log.txt"):
        self.loss_list, self.cpk_dir = [], log_dir
        self.visualizations_dir = os.path.join(log_dir, "train-vis")
        os.makedirs(self.visualizations_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, log_file_name), "a")
        self.zfill_num, self.visualizer = zfill_num, Visualizer(**visualizer_params)
        self.checkpoint_freq, self.epoch = checkpoint_freq, 0
        self.best_loss, self.names = float("inf"), None

    def log_scores(self, loss_names):
        m = np.array(self.loss_list).mean(0)
        line = str(self.epoch).zfill(self.zfill_num) + ") " + "; ".join("%s - %.5f" % pair for pair in zip(loss_names, m))
        print(line, file=self.log_file); self.log_file.flush(); self.loss_list = []

    def visualize_rec(self, inp, out):
        path = os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num))
        imageio.imwrite(path, self.visualizer.visualize(inp["driving"], inp["source"], out))

    def save_cpk(self, emergent=False):
        sd = {k: v.state_dict() for k, v in self.models.items()}
        sd["epoch"] = self.epoch
        pth = os.path.join(self.cpk_dir, "%s-checkpoint.pth.tar" % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(pth) and emergent):
            torch.save(sd, pth)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None):
        loc = None if torch.cuda.is_available() else "cpu"
        ck = torch.load(checkpoint_path, map_location=loc)
        if generator is not None:
            generator.load_state_dict(ck["generator"])
        if kp_detector is not None:
            kp_detector.load_state_dict(ck["kp_detector"])
        if discriminator is not None:
            try:
                discriminator.load_state_dict(ck["discriminator"])
            except Exception:
                print("No discriminator checkpoint; discriminator re-initialized randomly")
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(ck["optimizer_generator"])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(ck["optimizer_discriminator"])
            except RuntimeError:
                print("No discriminator optimizer dict; skipping")
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(ck["optimizer_kp_detector"])
        return ck["epoch"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if "models" in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch, self.models = epoch, models
        if (epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap="gist_rainbow"):
        self.kp_size, self.draw_border = kp_size, draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        im = np.copy(image)
        sp = np.array(im.shape[:2][::-1])[np.newaxis]
        kp = sp * (kp_array + 1) / 2
        for i, p in enumerate(kp):
            rr, cc = disk((p[1], p[0]), self.kp_size, shape=im.shape[:2])
            im[rr, cc] = np.array(self.colormap(i / len(kp)))[:3]
        return im

    def create_image_column_with_kp(self, images, kp):
        return self.create_image_column(np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)]))

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images); images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        cols = []
        for a in args:
            cols.append(self.create_image_column_with_kp(a[0], a[1]) if isinstance(a, tuple) else self.create_image_column(a))
        return np.concatenate(cols, axis=1)

    def visualize(self, driving, source, out):
        parts = []
        src = source.data.cpu()
        kps = out["kp_source"]["value"].data.cpu().numpy()
        src = np.transpose(src, (0, 2, 3, 1))
        parts.append((src, kps))

        if "transformed_frame" in out:
            tf = np.transpose(out["transformed_frame"].data.cpu().numpy(), (0, 2, 3, 1))
            parts.append((tf, out["transformed_kp"]["value"].data.cpu().numpy()))

        drv = np.transpose(driving.data.cpu().numpy(), (0, 2, 3, 1))
        parts.append((drv, out["kp_driving"]["value"].data.cpu().numpy()))

        if "deformed" in out:
            parts.append(np.transpose(out["deformed"].data.cpu().numpy(), (0, 2, 3, 1)))

        pred = np.transpose(out["prediction"].data.cpu().numpy(), (0, 2, 3, 1))
        if "kp_norm" in out:
            parts.append((pred, out["kp_norm"]["value"].data.cpu().numpy()))
        parts.append(pred)

        if "occlusion_map" in out:
            om = out["occlusion_map"].data.cpu().repeat(1, 3, 1, 1)
            om = np.transpose(F.interpolate(om, size=src.shape[1:3]).numpy(), (0, 2, 3, 1))
            parts.append(om)

        if "sparse_deformed" in out:
            fm, k = [], out["sparse_deformed"].shape[1]
            for i in range(k):
                im = out["sparse_deformed"][:, i].data.cpu()
                im = F.interpolate(im, size=src.shape[1:3])
                m = F.interpolate(out["mask"][:, i : i + 1].data.cpu().repeat(1, 3, 1, 1), size=src.shape[1:3])
                im = np.transpose(im.numpy(), (0, 2, 3, 1))
                m = np.transpose(m.numpy(), (0, 2, 3, 1))
                c = (0, 0, 0) if i == 0 else np.array(self.colormap((i - 1) / (k - 1)))[:3]
                c = np.array(c).reshape((1, 1, 1, 3))
                parts += [im, m * c] if i else [im, m]
                fm.append(m * c)
            parts.append(sum(fm))

        grid = self.create_image_grid(*parts)
        return (255 * grid).astype(np.uint8)
