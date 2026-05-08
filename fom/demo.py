"""CLI: animate a driving video from a checkpoint (standalone)."""
import os
import sys
from argparse import ArgumentParser
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import imageio
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
import ffmpeg
from skimage import img_as_ubyte
from skimage.transform import resize

from .services.inference import normalize_kp
from .modules.generator import OcclusionAwareGenerator
from .modules.keypoint_detector import KPDetector
from .sync_batchnorm import DataParallelWithCallback


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        cfg = yaml.full_load(f)
    mp = cfg["model_params"]
    g = OcclusionAwareGenerator(**mp["generator_params"], **mp["common_params"])
    k = KPDetector(**mp["kp_detector_params"], **mp["common_params"])
    if not cpu:
        g.cuda(); k.cuda()
    ck = torch.load(checkpoint_path, map_location="cpu" if cpu else None)
    g.load_state_dict(ck["generator"])
    k.load_state_dict(ck["kp_detector"])
    if not cpu:
        g = DataParallelWithCallback(g)
        k = DataParallelWithCallback(k)
    g.eval(); k.eval()
    return g, k


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    out = []
    dev = not cpu
    src = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if dev:
        src = src.cuda()
    drv = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
    kps = kp_detector(src)
    kd0 = kp_detector(drv[:, :, 0])
    with torch.no_grad():
        for fi in tqdm(range(drv.shape[2])):
            df = drv[:, :, fi]
            if dev:
                df = df.cuda()
            kd = kp_detector(df)
            kn = normalize_kp(kps, kd, kd0, use_relative_movement=relative, use_relative_jacobian=relative,
                               adapt_movement_scale=adapt_movement_scale)
            pred = generator(src, kp_source=kps, kp_driving=kn)["prediction"].data.cpu().numpy().transpose(0, 2, 3, 1)[0]
            out.append(pred)
    return out


def find_best_frame(source, driving, cpu=False):
    import face_alignment
    from scipy.spatial import ConvexHull

    def nk(p):
        p = p - p.mean(axis=0, keepdims=True)
        a = np.sqrt(ConvexHull(p[:, :2]).volume)
        p[:, :2] /= a
        return p

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device="cpu" if cpu else "cuda")
    k0 = nk(fa.get_landmarks(255 * source)[0])
    best, best_i = float("inf"), 0
    for i, im in tqdm(enumerate(driving)):
        kd = nk(fa.get_landmarks(255 * im)[0])
        s = (np.abs(k0 - kd) ** 2).sum()
        if s < best:
            best, best_i = s, i
    return best_i


def main(argv=None):
    if sys.version_info[0] < 3:
        raise SystemExit("Python 3 required")
    ap = ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="vox-cpk.pth.tar")
    ap.add_argument("--source_image", default="sup-mat/source.png")
    ap.add_argument("--driving_video", default="driving.mp4")
    ap.add_argument("--result_video", default="result.mp4")
    ap.add_argument("--relative", dest="relative", action="store_true")
    ap.add_argument("--adapt_scale", dest="adapt_scale", action="store_true")
    ap.add_argument("--find_best_frame", dest="find_best_frame", action="store_true")
    ap.add_argument("--best_frame", type=int, default=None)
    ap.add_argument("--cpu", dest="cpu", action="store_true")
    ap.add_argument("--audio", dest="audio", action="store_true")
    ap.set_defaults(relative=False, adapt_scale=False)
    opt = ap.parse_args(argv)

    src_img = resize(imageio.imread(opt.source_image), (256, 256))[..., :3]
    rdr = imageio.get_reader(opt.driving_video)
    fps = rdr.get_meta_data()["fps"]
    frames = []
    try:
        for fr in rdr:
            frames.append(resize(fr, (256, 256))[..., :3])
    except RuntimeError:
        pass
    rdr.close()

    gen, kpd = load_checkpoints(opt.config, opt.checkpoint, cpu=opt.cpu)

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(src_img, frames, cpu=opt.cpu)
        print("Best frame:", i)
        fwd, bkw = frames[i:], frames[: i + 1][::-1]
        pf = make_animation(src_img, fwd, gen, kpd, opt.relative, opt.adapt_scale, opt.cpu)
        pb = make_animation(src_img, bkw, gen, kpd, opt.relative, opt.adapt_scale, opt.cpu)
        seq = pb[::-1] + pf[1:]
    else:
        seq = make_animation(src_img, frames, gen, kpd, opt.relative, opt.adapt_scale, opt.cpu)

    imageio.mimsave(opt.result_video, [img_as_ubyte(x) for x in seq], fps=fps)

    if opt.audio:
        try:
            suf = splitext(opt.result_video)[1]
            with NamedTemporaryFile(suffix=suf, delete=False) as tmp:
                tpath = tmp.name
            ffmpeg.output(
                ffmpeg.input(opt.result_video).video, ffmpeg.input(opt.driving_video).audio, tpath, c="copy"
            ).run()
            with open(tpath, "rb") as fr, open(opt.result_video, "wb") as fw:
                copyfileobj(fr, fw)
            os.remove(tpath)
        except ffmpeg.Error:
            print("Could not mux audio.")


if __name__ == "__main__":
    main()
