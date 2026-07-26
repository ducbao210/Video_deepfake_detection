from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):

    def __init__(self, split_csv, cfg, transform=None):

        self.data = pd.read_csv(split_csv)

        self.root = Path(cfg.paths.data.processed_dir)

        self.transform = transform

        self.num_frames = cfg.preprocessing.frame_count
        self.sampling = cfg.preprocessing.get("sampling", "uniform")

    def __len__(self):
        return len(self.data)

    def _sample_indices(self, total):

        if total <= self.num_frames:
            return list(range(total))

        if self.sampling == "uniform":

            step = total / self.num_frames

            return [int(i * step) for i in range(self.num_frames)]

        return list(range(self.num_frames))

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        video_dir = self.root / row["video_path"]

        label = int(row["label"])

        frame_paths = sorted(video_dir.glob("*.jpg"))

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frame found in {video_dir}")

        indices = self._sample_indices(len(frame_paths))

        frames = []

        for i in indices:

            img = cv2.imread(str(frame_paths[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                img = self.transform(img)

            frames.append(img)

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames)

        middle = len(frames) // 2

        image = frames[middle]

        return {
            "frames": frames,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": row["video_id"],
        }
