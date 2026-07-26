import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, seed_everything
from src.data.dataset import DeepfakeDataset
from src.data.transforms import get_val_transforms
from src.models import build_model
from src.training.engine import evaluate


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger(cfg.logging, log_file="evaluate.log")
    seed_everything(cfg.seed)

    device = torch.device(
        cfg.device
        if cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1. Chuẩn bị Test Dataset
    val_transforms = get_val_transforms(cfg)
    test_dataset = DeepfakeDataset(
        cfg.paths.data.test_csv, cfg, transform=val_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    # 2. Khởi tạo Model & Load Checkpoint
    model = build_model(cfg).to(device)
    checkpoint_path = cfg.inference.checkpoint

    logger.info(f"Đang tải checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss().to(device)

    # 3. Tiến hành đánh giá (Evaluation)
    logger.info("Đang chạy dự đoán trên tập Test...")
    metrics = evaluate(model, test_loader, criterion, device)

    # 4. Hiển thị kết quả
    logger.info("========== KẾT QUẢ ĐÁNH GIÁ (TEST SET) ==========")
    logger.info(f"Loss:      {metrics['loss']:.4f}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"AUC:       {metrics['auc']:.4f}")
    logger.info("=================================================")


if __name__ == "__main__":
    main()
