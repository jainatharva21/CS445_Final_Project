import json
import time
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import torch


class Logger:
    """Scalars to log.txt + metrics.jsonl, image grids and checkpoints to subdirs."""

    def __init__(self, log_dir, log_freq=100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.log_dir / 'checkpoints'
        self.image_dir = self.log_dir / 'images'
        self.ckpt_dir.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        self.log_freq = log_freq

        self.log_file = open(self.log_dir / 'log.txt', 'a')
        self.metrics_file = open(self.log_dir / 'metrics.jsonl', 'a')
        self.buffer = {}
        self.buffer_count = 0
        self.start_time = time.time()

    def log_scalars(self, scalars, step):
        for k, v in scalars.items():
            if torch.is_tensor(v):
                v = v.item()
            self.buffer.setdefault(k, []).append(float(v))
        self.buffer_count += 1
        if self.buffer_count >= self.log_freq:
            self._flush(step)

    def _flush(self, step):
        if not self.buffer:
            return
        avg = {k: float(np.mean(v)) for k, v in self.buffer.items()}
        elapsed = time.time() - self.start_time
        msg = f'[step {step:>7d} | {elapsed:>6.0f}s] ' + '  '.join(f'{k}={v:.4f}' for k, v in avg.items())
        print(msg, flush=True)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.metrics_file.write(json.dumps({'step': step, 'time': elapsed, **avg}) + '\n')
        self.metrics_file.flush()
        self.buffer.clear()
        self.buffer_count = 0

    def log_images(self, images, step, name='sample'):
        if torch.is_tensor(images):
            images = images.detach().cpu().numpy()
        arr = np.transpose(images, (0, 2, 3, 1))
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        grid = np.concatenate(list(arr), axis=1)
        path = self.image_dir / f'{name}_{step:07d}.png'
        iio.imwrite(path, grid)
        return path

    def save_checkpoint(self, state, step, is_best=False):
        payload = {'step': step, **state}
        path = self.ckpt_dir / f'ckpt_{step:07d}.pth'
        torch.save(payload, path)
        torch.save(payload, self.ckpt_dir / 'latest.pth')
        if is_best:
            torch.save(payload, self.ckpt_dir / 'best.pth')
        return path

    def load_checkpoint(self, path=None, map_location='cpu'):
        if path is None:
            path = self.ckpt_dir / 'latest.pth'
        path = Path(path)
        if not path.exists():
            return None
        return torch.load(path, map_location=map_location, weights_only=False)

    def close(self):
        if self.buffer:
            self._flush(-1)
        self.log_file.close()
        self.metrics_file.close()
