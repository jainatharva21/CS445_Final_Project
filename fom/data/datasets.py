"""
PyTorch datasets for paired-frame training and full-clip evaluation.

``FramesDataset`` selects random frame pairs during training or returns entire clips
for reconstruction / pairing. Indexing rules mirror the MGIF layout expected by the
YAML configs without changing tensor shapes consumed by the models.
"""
from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from skimage import io, img_as_float32
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .spatial_augmentation import ClipAugmentationPipeline
from .video_io import read_video


class FramesDataset(Dataset):
    """Clips under ``root_dir`` with optional train-time augmentation."""

    def __init__(
        self,
        root_dir: str,
        frame_shape: tuple[int, int, int] = (256, 256, 3),
        id_sampling: bool = False,
        is_train: bool = True,
        random_seed: int = 0,
        pairs_list: str | None = None,
        augmentation_params: dict[str, Any] | None = None,
    ):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.is_train = is_train

        top_level = os.listdir(root_dir)
        train_dir = os.path.join(root_dir, "train")

        if os.path.isdir(train_dir):
            assert os.path.isdir(os.path.join(root_dir, "test"))
            print("Use predefined train-test split.")
            if id_sampling:
                train_ids = {os.path.basename(entry).split("#")[0] for entry in os.listdir(train_dir)}
            else:
                train_ids = os.listdir(train_dir)
            train_ids = list(train_ids)
            _test_ids = os.listdir(os.path.join(root_dir, "test"))
            self.root_dir = os.path.join(root_dir, "train" if is_train else "test")
        else:
            print("Use random train-test split.")
            train_ids, _test_ids = train_test_split(top_level, random_state=random_seed, test_size=0.2)

        self.videos = list(train_ids if is_train else _test_ids)
        self.transform = (
            ClipAugmentationPipeline(**augmentation_params) if is_train and augmentation_params else None
        )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_key = self.videos[idx]
        if self.is_train and self.id_sampling:
            clip_path = np.random.choice(glob.glob(os.path.join(self.root_dir, clip_key + "*.mp4")))
        else:
            clip_path = os.path.join(self.root_dir, clip_key)
        display_name = os.path.basename(clip_path)

        if self.is_train and os.path.isdir(clip_path):
            frame_files = sorted(os.listdir(clip_path))
            pair_idx = np.sort(np.random.choice(len(frame_files), 2, replace=True))
            video_array = np.stack(
                [img_as_float32(io.imread(os.path.join(clip_path, frame_files[i]))) for i in pair_idx]
            )
        else:
            video_array = read_video(clip_path, frame_shape=self.frame_shape)
            frame_count = len(video_array)
            if self.is_train:
                chosen = np.sort(np.random.choice(frame_count, 2, replace=True))
                video_array = video_array[chosen]
            else:
                video_array = video_array[np.arange(frame_count)]

        if self.transform:
            video_array = self.transform(list(video_array))

        if self.is_train:
            source_hwc, driving_hwc = map(np.asarray, (video_array[0], video_array[1]))
            return {
                "driving": driving_hwc.transpose(2, 0, 1).astype(np.float32),
                "source": source_hwc.transpose(2, 0, 1).astype(np.float32),
                "name": display_name,
            }

        clip_chw = np.asarray(video_array, dtype=np.float32).transpose(3, 0, 1, 2)
        return {"video": clip_chw, "name": display_name}


class DatasetRepeater(Dataset):
    """Repeats an underlying dataset index range (``num_repeats`` > 1 in YAML)."""

    def __init__(self, dataset: Dataset, num_repeats: int = 100):
        self._base = dataset
        self._repeats = num_repeats

    def __len__(self) -> int:
        return len(self._base) * self._repeats

    def __getitem__(self, idx: int) -> Any:
        return self._base[idx % len(self._base)]


class PairedDataset(Dataset):
    """Pairs driving and source clips for cross-identity animation."""

    def __init__(self, initial_dataset: FramesDataset, number_of_pairs: int, seed: int = 0):
        np.random.seed(seed)
        self.initial_dataset = initial_dataset
        plist = initial_dataset.pairs_list
        video_keys = initial_dataset.videos

        if plist is None:
            side = min(number_of_pairs, len(initial_dataset))
            grid = np.mgrid[:side, :side].reshape(2, -1).T
            take = min(len(grid), number_of_pairs)
            self.pairs = grid[np.random.choice(len(grid), take, replace=False)]
        else:
            key_to_index = {name: i for i, name in enumerate(video_keys)}
            table = pd.read_csv(plist)
            mask = np.logical_and(table["source"].isin(video_keys), table["driving"].isin(video_keys))
            table = table[mask]
            take = min(len(table), number_of_pairs)
            self.pairs = [
                (key_to_index[table.iloc[i]["driving"]], key_to_index[table.iloc[i]["source"]]) for i in range(take)
            ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        driving_idx, source_idx = self.pairs[idx]
        driving_batch = {f"driving_{k}": v for k, v in self.initial_dataset[driving_idx].items()}
        source_batch = {f"source_{k}": v for k, v in self.initial_dataset[source_idx].items()}
        return {**driving_batch, **source_batch}
