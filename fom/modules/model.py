from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from torchvision import models

from .util import AntiAliasInterpolation2d, make_coordinate_grid


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        feats = models.vgg19(pretrained=True).features
        bnds = [0, 2, 7, 12, 21, 30]
        self.slices = nn.ModuleList(nn.Sequential(*[feats[j] for j in range(bnds[i], bnds[i + 1])]) for i in range(5))
        self.register_buffer("mean", torch.tensor(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), dtype=torch.float32))
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        ys = []
        for s in self.slices:
            X = s(X)
            ys.append(X)
        return ys


class ImagePyramide(nn.Module):
    def __init__(self, scales, num_channels):
        super().__init__()
        self.downs = nn.ModuleDict({str(sc).replace(".", "-"): AntiAliasInterpolation2d(num_channels, sc) for sc in scales})

    def forward(self, x):
        return {("prediction_" + k.replace("-", ".")): mod(x) for k, mod in self.downs.items()}


class Transform:
    def __init__(self, bs, **kwargs):
        n = torch.normal(0, kwargs["sigma_affine"] * torch.ones(bs, 2, 3))
        self.theta = n + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs
        if "sigma_tps" in kwargs and "points_tps" in kwargs:
            self.tps = True
            p = kwargs["points_tps"]
            self.control_points = make_coordinate_grid((p, p), n.type()).unsqueeze(0)
            self.control_params = torch.normal(
                0, kwargs["sigma_tps"] * torch.ones(bs, 1, p ** 2))
        else:
            self.tps = False

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).to(coordinates.device).unsqueeze(1)
        y = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        y = y.squeeze(-1)
        if not self.tps:
            return y
        cp = self.control_points.type(coordinates.type()).to(coordinates.device)
        pr = self.control_params.type(coordinates.type()).to(coordinates.device)
        d = (coordinates.reshape(coordinates.shape[0], -1, 1, 2) - cp.view(1, 1, -1, 2)).abs().sum(-1)
        return y + (d ** 2 * torch.log(d + 1e-6) * pr).sum(2).view(self.bs, coordinates.shape[1], 1)

    def transform_frame(self, frame):
        g = make_coordinate_grid(frame.shape[2:], frame.type()).unsqueeze(0).to(frame.device)
        g = g.view(1, frame.shape[2] * frame.shape[3], 2)
        return F.grid_sample(frame, self.warp_coordinates(g).view(self.bs, *frame.shape[2:], 2), padding_mode="reflection")

    def jacobian(self, coordinates):
        nc = self.warp_coordinates(coordinates)
        return torch.cat([
            grad(nc[..., 0].sum(), coordinates, create_graph=True)[0].unsqueeze(-2),
            grad(nc[..., 1].sum(), coordinates, create_graph=True)[0].unsqueeze(-2),
        ], dim=-2)


def detach_kp(kp):
    return {k: v.detach() for k, v in kp.items()}


class GeneratorFullModel(nn.Module):
    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super().__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.loss_weights = train_params["loss_weights"]
        self.scales = train_params["scales"]
        self.disc_scales = discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        need_vgg = sum(self.loss_weights["perceptual"]) != 0
        self.vgg = Vgg19() if need_vgg else None
        if need_vgg and torch.cuda.is_available():
            self.vgg = self.vgg.cuda()

    def forward(self, x):
        ks, kd = self.kp_extractor(x["source"]), self.kp_extractor(x["driving"])
        gen = self.generator(x["source"], kp_source=ks, kp_driving=kd)
        gen.update({"kp_source": ks, "kp_driving": kd})
        lv, lw = {}, self.loss_weights
        preg, pdriv = self.pyramid(gen["prediction"]), self.pyramid(x["driving"])

        if self.vgg:
            vt = preg["prediction_" + str(self.scales[0])].new_tensor(0.0)
            for sc in self.scales:
                xs, ys = self.vgg(preg["prediction_" + str(sc)]), self.vgg(pdriv["prediction_" + str(sc)])
                for i, ww in enumerate(lw["perceptual"]):
                    vt = vt + ww * (xs[i] - ys[i].detach()).abs().mean()
            lv["perceptual"] = vt

        if lw["generator_gan"] != 0:
            dk = detach_kp(kd)
            dg, dr = self.discriminator(preg, kp=dk), self.discriminator(pdriv, kp=dk)
            gt = preg["prediction_" + str(self.scales[0])].new_tensor(0.0)
            for s in self.disc_scales:
                k = "prediction_map_%s" % s
                gt = gt + lw["generator_gan"] * ((1 - dg[k]) ** 2).mean()
            lv["gen_gan"] = gt
            if sum(lw["feature_matching"]) != 0:
                ft = preg["prediction_" + str(self.scales[0])].new_tensor(0.0)
                for s in self.disc_scales:
                    fk = "feature_maps_%s" % s
                    for i, (a, bb) in enumerate(zip(dr[fk], dg[fk])):
                        wfm = lw["feature_matching"][i]
                        if wfm:
                            ft = ft + wfm * (a - bb).abs().mean()
                lv["feature_matching"] = ft

        if lw["equivariance_value"] != 0 or lw["equivariance_jacobian"] != 0:
            tr = Transform(x["driving"].shape[0], **self.train_params["transform_params"])
            tx = tr.transform_frame(x["driving"])
            tk = self.kp_extractor(tx)
            gen["transformed_frame"], gen["transformed_kp"] = tx, tk
            if lw["equivariance_value"] != 0:
                lv["equivariance_value"] = lw["equivariance_value"] * (
                    kd["value"] - tr.warp_coordinates(tk["value"])).abs().mean()
            if lw["equivariance_jacobian"] != 0:
                jt = torch.matmul(tr.jacobian(tk["value"]), tk["jacobian"])
                val = torch.matmul(torch.inverse(kd["jacobian"]), jt)
                eye = torch.eye(2, device=val.device, dtype=val.dtype).view(1, 1, 2, 2)
                lv["equivariance_jacobian"] = lw["equivariance_jacobian"] * (eye - val).abs().mean()

        return lv, gen


class DiscriminatorFullModel(nn.Module):
    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super().__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.scales = discriminator.scales
        self.wd = train_params["loss_weights"]["discriminator_gan"]
        self.pyramid = ImagePyramide(train_params["scales"], generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

    def forward(self, x, generated):
        preg, pdriv = self.pyramid(generated["prediction"].detach()), self.pyramid(x["driving"])
        dk = detach_kp(generated["kp_driving"])
        dg = self.discriminator(preg, kp=dk)
        dr = self.discriminator(pdriv, kp=dk)
        vt = preg["prediction_" + str(self.scales[0])].new_tensor(0.0)
        for s in self.scales:
            k = "prediction_map_%s" % s
            vt += self.wd * ((1 - dr[k]) ** 2 + dg[k] ** 2).mean()
        return {"disc_gan": vt}
