#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, seed_everything, seed_worker
from src.data import DeepfakeDataset, get_train_transforms, get_val_transforms
from src.models import build_model
from src.training.trainer import Trainer
from src.training import build_optimizer, build_scheduler


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize the logger and set the random seed
    logger = get_logger(cfg.logging, name=cfg.experiment_name, log_file="train.log")

    logger.info(f"Starting training for experiment: {cfg.experiment_name}")
    seed_everything(cfg.seed)

    # Prepare the datasets and data loaders
    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    train_dataset = DeepfakeDataset(
        cfg.paths.data.train_csv, cfg, transform=train_transforms
    )
    val_dataset = DeepfakeDataset(cfg.paths.data.val_csv, cfg, transform=val_transforms)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=(
            cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0
        ),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers
        and cfg.dataloader.num_workers > 0,
    )

    # Initialize the model
    model = build_model(cfg)

    num_params = sum(p.numel() for p in model.parameters())
    model_name_upper = cfg.model.name.upper()
    logger.info(f"MODEL: {model_name_upper}")
    logger.info(f"TOTAL PARAMETERS: {num_params:,}")

    # Configure the loss function, optimizer, and learning rate scheduler
    train_labels = train_dataset.data["label"].to_numpy()
    class_counts = np.bincount(train_labels, minlength=cfg.model.classifier.num_classes)
    class_weights = len(train_labels) / (
        len(class_counts) * np.maximum(class_counts, 1)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    logger.info(f"Training set class distribution: {class_counts.tolist()}")

    logger.info(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.training.get("label_smoothing", 0.0),
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Start training using the Trainer engine
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
