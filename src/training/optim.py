# src/training/optim.py
import math
import torch


def build_optimizer(model, cfg):
    name = cfg.training.optimizer.get("name", "adamw").lower()
    base_lr = float(cfg.training.learning_rate)
    wd = float(cfg.training.weight_decay)
    
    backbone_lr_ratio = float(cfg.training.get("backbone_lr_ratio", 0.1))
    backbone_lr = base_lr * backbone_lr_ratio

    head_keywords = ["classifier", "lstm", "attention", "head"]
    no_decay_keywords = ["bias", "norm"]

    param_groups = [
        {"params": [], "lr": base_lr, "weight_decay": wd, "name": "head_decay"},
        {"params": [], "lr": base_lr, "weight_decay": 0.0, "name": "head_no_decay"},
        {"params": [], "lr": backbone_lr, "weight_decay": wd, "name": "backbone_decay"},
        {"params": [], "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_no_decay"},
    ]

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = any(kw in param_name.lower() for kw in head_keywords)
        is_no_decay = (param.ndim <= 1) or any(kw in param_name.lower() for kw in no_decay_keywords)

        if is_head and not is_no_decay:
            param_groups[0]["params"].append(param)
        elif is_head and is_no_decay:
            param_groups[1]["params"].append(param)
        elif not is_head and not is_no_decay:
            param_groups[2]["params"].append(param)
        else:
            param_groups[3]["params"].append(param)

    if name == "adamw":
        return torch.optim.AdamW(param_groups)
    if name == "adam":
        return torch.optim.Adam(param_groups)
    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
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
