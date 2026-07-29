# src/training/optim.py
import math

import torch


def build_optimizer(model, cfg):
    """Build an optimizer based on cfg.training.optimizer.name."""
    name = cfg.training.optimizer.get("name", "adamw").lower()
    lr = float(cfg.training.learning_rate)
    wd = float(cfg.training.weight_decay)

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=wd,
            momentum=cfg.training.optimizer.get("momentum", 0.9),
            nesterov=True,
        )

    raise ValueError(f"Unsupported optimizer '{name}'. Available: adamw, adam, sgd")


def build_scheduler(optimizer, cfg):
    """
    Build a learning rate scheduler with optional warmup.
    The scheduler is stepped once per epoch, matching the training loop in Trainer.fit().
    """
    name = cfg.training.scheduler.get("name", "cosine").lower()
    total_epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    base_lr = float(cfg.training.learning_rate)
    min_lr = float(cfg.training.scheduler.get("min_lr", 0.0))
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    if name == "none":
        return None

    def lr_lambda(epoch):
        # Warmup phase: linearly increase the learning rate
        # from 1/warmup_epochs to the base learning rate.
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, progress))

        if name == "cosine":
            return min_ratio + (1.0 - min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        if name == "linear":
            return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
        if name == "step":
            step_size = cfg.training.scheduler.get("step_size", 10)
            gamma = cfg.training.scheduler.get("gamma", 0.1)
            return gamma ** ((epoch - warmup_epochs) // step_size)
        if name == "constant":
            return 1.0

        raise ValueError(
            f"Unsupported scheduler '{name}'. "
            f"Available: cosine, linear, step, constant, none"
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
