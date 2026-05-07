import os
import glob
import numpy as np
import imageio.v3 as iio
import torch
from skimage import img_as_float32
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset

from augmentation import AllAugmentationTransform

VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.gif', '.mkv', '.webm')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def _list_frames(folder):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f'*{ext}')))
    return sorted(paths)


def _to_rgb_float(video, target_hw):
    if video.ndim == 3:
        video = np.repeat(video[..., None], 3, axis=-1)
    if video.shape[-1] == 4:
        video = video[..., :3]
    video = img_as_float32(video)
    if video.shape[1:3] != target_hw:
        video = np.stack([resize(f, target_hw, anti_aliasing=True) for f in video]).astype(np.float32)
    return video


def read_video(path, frame_shape=(256, 256, 3)):
    target_hw = (frame_shape[0], frame_shape[1])
    if os.path.isdir(path):
        frame_paths = _list_frames(path)
        if not frame_paths:
            raise ValueError(f'No frames in {path}')
        video = np.stack([imread(p) for p in frame_paths], axis=0)
    elif path.lower().endswith(VIDEO_EXTS):
        video = iio.imread(path)
    else:
        raise ValueError(f'Unrecognized path: {path}')
    return _to_rgb_float(video, target_hw)


class FramesDataset(Dataset):
    """Videos sampled as (source, driving) pairs for training, or full sequences for eval.

    Layout expected:
        root_dir/train/<video_id>/  (folder of frames)   OR  root_dir/train/<video_id>.mp4
        root_dir/test/...

    With id_sampling=True, video IDs are split on '#' (VoxCeleb convention:
    `<person>#<clip>`) and __getitem__ returns a pair from one randomly-chosen
    clip per identity.
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True,
                 id_sampling=False, augmentation_params=None, sampling_mode='pair'):
        if sampling_mode not in {'pair', 'full'}:
            raise ValueError("sampling_mode must be 'pair' or 'full'")
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.is_train = is_train
        self.id_sampling = id_sampling and is_train
        self.sampling_mode = sampling_mode

        split = 'train' if is_train else 'test'
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f'{split_dir} not found. Expected: {root_dir}/{{train,test}}/<video_id>/'
            )
        entries = sorted(e for e in os.listdir(split_dir) if not e.startswith('.'))
        if not entries:
            raise RuntimeError(f'No videos in {split_dir}')

        if self.id_sampling:
            self.id_to_videos = {}
            for e in entries:
                ident = e.split('#')[0]
                self.id_to_videos.setdefault(ident, []).append(os.path.join(split_dir, e))
            self.identities = sorted(self.id_to_videos.keys())
        self.videos = [os.path.join(split_dir, e) for e in entries]

        self.transform = (
            AllAugmentationTransform(**augmentation_params)
            if (is_train and augmentation_params) else None
        )

    def __len__(self):
        return len(self.identities) if self.id_sampling else len(self.videos)

    def _load_pair(self, path):
        target_hw = (self.frame_shape[0], self.frame_shape[1])
        if os.path.isdir(path):
            frames = _list_frames(path)
            if not frames:
                raise RuntimeError(f'Empty frame folder: {path}')
            idxs = np.random.choice(len(frames), size=2, replace=True)
            video = np.stack([imread(frames[i]) for i in idxs], axis=0)
            return _to_rgb_float(video, target_hw)
        video = read_video(path, self.frame_shape)
        return video[np.random.choice(video.shape[0], size=2, replace=True)]

    def __getitem__(self, idx):
        if self.sampling_mode == 'full':
            path = self.videos[idx]
            video = read_video(path, self.frame_shape)
            return {
                'video': torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2))),
                'name': os.path.basename(path),
            }

        if self.id_sampling:
            ident = self.identities[idx]
            path = np.random.choice(self.id_to_videos[ident])
            name = ident
        else:
            path = self.videos[idx]
            name = os.path.basename(path)

        video = self._load_pair(path)
        if self.transform is not None:
            video = self.transform(video)
        source = torch.from_numpy(np.ascontiguousarray(video[0].transpose(2, 0, 1)))
        driving = torch.from_numpy(np.ascontiguousarray(video[1].transpose(2, 0, 1)))
        return {'source': source, 'driving': driving, 'name': name}


class PairedDataset(Dataset):
    """For animation eval: source from one video, driving from another."""

    def __init__(self, initial_dataset, num_pairs=50, seed=0):
        self.initial_dataset = initial_dataset
        rng = np.random.RandomState(seed)
        n = len(initial_dataset)
        self.pairs = [(int(rng.randint(n)), int(rng.randint(n))) for _ in range(num_pairs)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        s, d = self.initial_dataset[i], self.initial_dataset[j]
        return {
            'source_video': s['video'], 'source_name': s['name'],
            'driving_video': d['video'], 'driving_name': d['name'],
        }
