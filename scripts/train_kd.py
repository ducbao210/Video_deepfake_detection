import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DeepfakeDataset, get_train_transforms, get_val_transforms
from src.models import build_model
from src.training.kd_trainer import KDTrainer
from src.training.optim import build_optimizer, build_scheduler
from src.utils import get_logger, seed_everything, seed_worker


def _load_submodel(cfg, model_name, checkpoint_path, device, logger):
    """
    Build a model by name and optionally load its checkpoint.
    Merge the corresponding model configuration into a copy of the base config.
    """
    model_cfg = OmegaConf.load(ROOT / "configs" / "model" / f"{model_name}.yaml")

    # Chuyển đổi cfg sang dict để gỡ bỏ giới hạn struct mode
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    unfrozen_cfg = OmegaConf.create(cfg_dict)

    merged = OmegaConf.merge(unfrozen_cfg, model_cfg)

    model = build_model(merged)

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info(f"Loading weights for '{model_name.capitalize()}' from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.info(
            f"'{model_name.capitalize()}' initialized from a pretrained backbone (no checkpoint provided)"
        )

    return model.to(device)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger(cfg.logging, name=cfg.experiment_name, log_file="kd_train.log")
    seed_everything(cfg.seed)

    device = torch.device(
        cfg.device
        if cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if "kd" not in cfg:
        raise ValueError(
            "Missing KD configuration. Run with:\n"
            "  python scripts/train_kd.py +model=kd_config training=kd_training"
        )

    # Dataset & DataLoader
    train_dataset = DeepfakeDataset(
        cfg.paths.data.train_csv, cfg, transform=get_train_transforms(cfg)
    )
    val_dataset = DeepfakeDataset(
        cfg.paths.data.val_csv, cfg, transform=get_val_transforms(cfg)
    )

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

    # Teacher (frozen) & Student
    teacher = _load_submodel(
        cfg, cfg.kd.teacher.model, cfg.kd.teacher.get("checkpoint"), device, logger
    )
    student = _load_submodel(
        cfg, cfg.kd.student.model, cfg.kd.student.get("checkpoint"), device, logger
    )

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    logger.info(
        f"Teacher {cfg.kd.teacher.model}: {n_teacher/1e6:.1f}M params | "
        f"Student {cfg.kd.student.model}: {n_student/1e6:.1f}M params "
        f"(compression ratio: {n_teacher/n_student:.2f}x)"
    )

    # Loss with class weights
    train_labels = train_dataset.data["label"].to_numpy()
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        len(train_labels) / (len(class_counts) * np.maximum(class_counts, 1)),
        dtype=torch.float32,
    )
    logger.info(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.training.get("label_smoothing", 0.0),
    )

    # Optimize the student model only
    optimizer = build_optimizer(student, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = KDTrainer(
        student_model=student,
        teacher_model=teacher,
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
