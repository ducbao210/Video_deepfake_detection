#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger
from src.data.transforms import get_val_transforms
from src.models import build_model


def extract_frames_memory(video_path, frame_count, image_size):
    """
    Extract a fixed number of frames from a video using uniform sampling.
    Returns a list of RGB NumPy arrays.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Invalid video or no frames were found.")
    target_num_frames = min(frame_count, total_frames)
    frame_indices = set(np.linspace(0, total_frames - 1, target_num_frames, dtype=int))

    frames = []
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_indices:
            frame = cv2.resize(frame, tuple(image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_indices.remove(current_frame)
            if len(frames) >= target_num_frames:
                break
        current_frame += 1

    cap.release()

    # Pad with the last frame if fewer than the required number of frames were extracted
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())

    return frames


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger(cfg.logging, name=cfg.experiment_name, log_file="inference.log")

    video_path = cfg.inference.video_path
    if not video_path or not Path(video_path).exists():
        logger.error(f"Invalid video path: {video_path}")
        return

    device = torch.device(
        cfg.device
        if cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Initialize the model
    model = build_model(cfg).to(device)
    checkpoint_path = cfg.inference.checkpoint

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        logger.info(
            f"Checkpoint not found locally at {ckpt_path}. Attempting to download from Hugging Face..."
        )

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()

        try:
            downloaded_path = hf_hub_download(
                repo_id=cfg.huggingface.repo_id,
                filename=f"{cfg.huggingface.path_in_repo}/best.pth",
                token=cfg.huggingface.token,
            )
            ckpt_path = Path(downloaded_path)
            logger.info(f"Download complete! Checkpoint used: {ckpt_path}")
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            return

    logger.info(f"Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Process the video (extract frames and apply transforms)
    logger.info(f"Analyzing video: {video_path}")
    frame_count = cfg.preprocessing.frame_count
    image_size = cfg.preprocessing.image_size

    raw_frames = extract_frames_memory(video_path, frame_count, image_size)
    val_transforms = get_val_transforms(cfg)

    transformed_frames = [val_transforms(frame) for frame in raw_frames]

    # Format the tensor as (B, T, C, H, W)
    input_tensor = torch.stack(transformed_frames).unsqueeze(0).to(device)

    # Run inference
    threshold = cfg.inference.threshold
    with torch.inference_mode():
        logits = model(input_tensor)
        # Use the probability of class 1 (fake/manipulated)
        prob = torch.softmax(logits, dim=1)[:, 1].item()

    is_fake = prob >= threshold
    label_str = "MANIPULATED (FAKE)" if is_fake else "ORIGINAL (REAL)"

    logger.info("========== INFERENCE RESULTS ==========")
    logger.info(f"Video:      {Path(video_path).name}")
    logger.info(f"Prediction: {label_str}")
    logger.info(f"Confidence: {prob * 100:.2f}% (Fake probability)")
    logger.info("======================================")


if __name__ == "__main__":
    main()
