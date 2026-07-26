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
    Trích xuất một số lượng frame cố định (uniform sampling) từ video.
    Trả về danh sách các numpy array (RGB).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video tại đường dẫn: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video không hợp lệ hoặc không có khung hình.")

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

    # Padding frame cuối nếu chưa đủ số lượng yêu cầu
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())

    return frames


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger(cfg.logging, log_file="inference.log")

    video_path = cfg.inference.video_path
    if not video_path or not Path(video_path).exists():
        logger.error(f"Đường dẫn video không hợp lệ: {video_path}")
        return

    device = torch.device(
        cfg.device
        if cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1. Khởi tạo Model
    model = build_model(cfg).to(device)
    checkpoint_path = cfg.inference.checkpoint

    logger.info(f"Đang tải trọng số từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Xử lý Video (Trích xuất & Transform Frame)
    logger.info(f"Đang phân tích video: {video_path}")
    frame_count = cfg.preprocessing.frame_count
    image_size = cfg.preprocessing.image_size

    raw_frames = extract_frames_memory(video_path, frame_count, image_size)
    val_transforms = get_val_transforms(cfg)

    transformed_frames = [val_transforms(frame) for frame in raw_frames]

    # Định dạng tensor thành (B, T, C, H, W)
    input_tensor = torch.stack(transformed_frames).unsqueeze(0).to(device)

    # 3. Chạy Inference
    threshold = cfg.inference.threshold
    with torch.inference_mode():
        logits = model(input_tensor)
        # Lấy class index 1 (fake/manipulated)
        prob = torch.softmax(logits, dim=1)[:, 1].item()

    is_fake = prob >= threshold
    label_str = "MANIPULATED (FAKE)" if is_fake else "ORIGINAL (REAL)"

    logger.info("========== KẾT QUẢ SUY LUẬN ==========")
    logger.info(f"Video:     {Path(video_path).name}")
    logger.info(f"Dự đoán:   {label_str}")
    logger.info(f"Độ tự tin: {prob * 100:.2f}% (Tỷ lệ giả mạo)")
    logger.info("======================================")


if __name__ == "__main__":
    main()
