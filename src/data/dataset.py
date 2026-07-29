import random
from pathlib import Path, PureWindowsPath

import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    def __init__(self, split_csv, cfg, transform=None):
        self.data = pd.read_csv(split_csv)
        self.root = Path(cfg.paths.data.processed_dir)
        self.transform = transform

        self.num_frames = cfg.preprocessing.frame_count
        self.sampling = cfg.preprocessing.get("sampling", "uniform")

        # Use the same random augmentation for all frames in a video
        # to preserve temporal consistency
        self.consistent_aug = cfg.preprocessing.get("consistent_augment", True)

    def __len__(self):
        return len(self.data)

    def _sample_indices(self, total):
        if total <= self.num_frames:
            return list(range(total))

        if self.sampling == "uniform":
            step = total / self.num_frames
            return [int(i * step) for i in range(self.num_frames)]

        if self.sampling == "random":
            return sorted(random.sample(range(total), self.num_frames))

        return list(range(self.num_frames))

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Accept both Windows and POSIX-style paths stored in the CSV
        rel_path = PureWindowsPath(str(row["video_path"])).as_posix()
        video_dir = self.root / rel_path
        label = int(row["label"])

        frame_paths = sorted(video_dir.glob("*.jpg"))
        if len(frame_paths) == 0:
            raise RuntimeError(f"No frame found in {video_dir}")

        indices = self._sample_indices(len(frame_paths))

        # Fix the random seed per video so that all frames receive
        # the same augmentation sequence
        seed = random.randint(0, 2**31 - 1) if self.consistent_aug else None

        frames = []
        for i in indices:
            img = cv2.imread(str(frame_paths[i]))
            if img is None:
                raise RuntimeError(f"Cannot read frame {frame_paths[i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)

            frames.append(img)

        # Repeat the last frame until the required sequence length is reached
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames)

        return {
            "frames": frames,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": str(row["video_id"]),
        }
