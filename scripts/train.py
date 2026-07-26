import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Cấu hình đường dẫn gốc để import thư mục src
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, seed_everything
from src.data import DeepfakeDataset, get_train_transforms, get_val_transforms
from src.models import build_model
from src.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Khởi tạo logger và set seed
    logger = get_logger(cfg.logging, log_file="train.log")
    logger.info(f"Bắt đầu quy trình huấn luyện cho: {cfg.experiment_name}")
    seed_everything(cfg.seed)

    # 1. Chuẩn bị Dataset & DataLoader
    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    train_dataset = DeepfakeDataset(
        cfg.paths.data.train_csv, cfg, transform=train_transforms
    )
    val_dataset = DeepfakeDataset(cfg.paths.data.val_csv, cfg, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers,
    )

    # 2. Khởi tạo Model
    model = build_model(cfg)

    # 3. Thiết lập Loss, Optimizer và Scheduler
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.training.scheduler.get("min_lr", 1e-6),
    )

    # 4. Bắt đầu huấn luyện qua Engine Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
