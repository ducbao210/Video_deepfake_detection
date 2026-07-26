import sys
import random
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger


def build_records(video_paths, label, processed_dir):
    records = []

    for video in video_paths:
        num_frames = len(list(video.glob("*.jpg")))

        if num_frames == 0:
            continue

        records.append(
            {
                "video_path": str(video.relative_to(processed_dir)),
                "video_id": video.name,
                "label": label,
                "num_frames": num_frames,
            }
        )

    return records


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    logger = get_logger(cfg.logging, log_file="split_dataset.log")

    random.seed(cfg.seed)

    processed_dir = Path(cfg.paths.data.processed_dir)
    split_dir = Path(cfg.paths.data.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    classes = {
        "original": ("DFD_original_sequences", 0),
        "manipulated": ("DFD_manipulated_sequences", 1),
    }

    train_data = []
    val_data = []
    test_data = []

    train_video = 0
    val_video = 0
    test_video = 0

    for class_name, (folder_name, label) in classes.items():

        class_dir = processed_dir / folder_name

        video_paths = sorted(
            [p for p in class_dir.iterdir() if p.is_dir()],
            key=lambda x: x.name,
        )

        random.shuffle(video_paths)

        total = len(video_paths)

        train_end = int(total * cfg.split.train)
        val_end = train_end + int(total * cfg.split.val)

        train_videos = video_paths[:train_end]
        val_videos = video_paths[train_end:val_end]
        test_videos = video_paths[val_end:]

        train_video += len(train_videos)
        val_video += len(val_videos)
        test_video += len(test_videos)

        train_data.extend(build_records(train_videos, label, processed_dir))
        val_data.extend(build_records(val_videos, label, processed_dir))
        test_data.extend(build_records(test_videos, label, processed_dir))

    pd.DataFrame(train_data).to_csv(cfg.paths.data.train_csv, index=False)
    pd.DataFrame(val_data).to_csv(cfg.paths.data.val_csv, index=False)
    pd.DataFrame(test_data).to_csv(cfg.paths.data.test_csv, index=False)

    logger.info("========== SUMMARY ==========")
    logger.info(f"Train : {train_video} videos")
    logger.info(f"Val   : {val_video} videos")
    logger.info(f"Test  : {test_video} videos")
    logger.info("=============================")


if __name__ == "__main__":
    main()
