import os
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger


def extract_frames_from_video(video_path, output_dir, frame_count, frame_size):
    """
    Hàm trích xuất frame cốt lõi, không phụ thuộc vào OmegaConf/DictConfig.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False

    os.makedirs(output_dir, exist_ok=True)

    target_num_frames = min(frame_count, total_frames)

    frame_indices = set(
        np.linspace(
            0,
            total_frames - 1,
            target_num_frames,
            dtype=int,
        )
    )

    current_frame = 0
    extracted_count = 0
    saved_idx = 0

    # Read sequentially instead of using cap.set()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame in frame_indices:
            frame = cv2.resize(frame, frame_size)

            save_path = os.path.join(
                output_dir,
                f"frame_{saved_idx:04d}.jpg",
            )

            cv2.imwrite(save_path, frame)

            saved_idx += 1
            extracted_count += 1
            frame_indices.remove(current_frame)

            if extracted_count >= target_num_frames:
                break

        current_frame += 1

    cap.release()
    return True


def process_video(video_path, raw_dir, processed_dir, frame_count, image_size):
    """
    Worker cho ProcessPool.
    Lưu ý: Các tham số truyền vào (đặc biệt qua multiprocessing) nên là dữ liệu nguyên thủy (primitive types).
    """
    video_path = Path(video_path)
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    relative_path = video_path.relative_to(raw_dir)
    output_dir = processed_dir / relative_path.parent / relative_path.stem

    success = extract_frames_from_video(
        video_path=video_path,
        output_dir=str(output_dir),
        frame_count=frame_count,
        frame_size=image_size,
    )

    return str(video_path), success


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    logger = get_logger(
        cfg.logging,
        name="ExtractFrames",
        log_file="extract_frames.log",
    )

    logger.info("Extracting frames...")

    raw_dir = Path(cfg.paths.data.raw_dir)
    processed_dir = Path(cfg.paths.data.processed_dir)

    frame_count = cfg.preprocessing.frame_count
    image_size = tuple(cfg.preprocessing.image_size)

    num_workers = cfg.preprocessing.get(
        "num_workers",
        min(4, os.cpu_count()),
    )

    video_paths = sorted(
        glob.glob(
            str(raw_dir / "**" / "*.mp4"),
            recursive=True,
        )
    )

    logger.info(f"Found {len(video_paths)} videos.")
    logger.info(f"Using {num_workers} worker processes.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_video,
                str(video_path),
                str(raw_dir),
                str(processed_dir),
                frame_count,
                image_size,
            )
            for video_path in video_paths
        ]

        failed = []

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing videos",
            dynamic_ncols=True,
            miniters=1,
        ):
            try:
                video_path, success = future.result()

                if not success:
                    failed.append(video_path)

            except Exception as e:
                failed.append(str(e))

    logger.info("\nDone.")

    if failed:
        logger.info(f"Failed: {len(failed)} videos")
        for item in failed:
            logger.info(item)


if __name__ == "__main__":
    main()
