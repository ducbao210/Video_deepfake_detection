import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from src.utils import AverageMeter, calculate_metrics


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler=None,
    max_norm=0.0,
    accum_steps=1,
):
    model.train()
    loss_meter = AverageMeter()

    preds_list = []
    labels_list = []

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc="Training", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if scaler is not None:
            with autocast(device_type=device.type):
                outputs = model(frames)
                loss = criterion(outputs, labels) / accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()

        is_update_step = (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader)
        if is_update_step:
            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item() * accum_steps, frames.size(0))

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds_list.extend(probs.detach().cpu().numpy())
        labels_list.extend(labels.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    metrics = calculate_metrics(labels_list, preds_list)
    metrics["loss"] = loss_meter.avg
    return metrics


def train_one_epoch_kd(
    student,
    teacher,
    dataloader,
    criterion,
    optimizer,
    device,
    temperature,
    hard_weight,
    soft_weight,
    scaler=None,
    max_norm=0.0,
    accum_steps=1,
):
    student.train()
    teacher.eval()

    loss_meter = AverageMeter()
    preds_list = []
    labels_list = []

    kd_criterion = nn.KLDivLoss(reduction="batchmean")

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="KD Training", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if scaler is not None:
            with autocast(device_type=device.type):
                with torch.no_grad():
                    teacher_logits = teacher(frames)

                student_logits = student(frames)

                hard_loss = criterion(student_logits, labels)

                soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                soft_loss = kd_criterion(student_log_probs, soft_targets) * (
                    temperature**2
                )

                loss = (hard_weight * hard_loss + soft_weight * soft_loss) / accum_steps
        else:
            with torch.no_grad():
                teacher_logits = teacher(frames)

            student_logits = student(frames)

            hard_loss = criterion(student_logits, labels)
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            soft_loss = kd_criterion(student_log_probs, soft_targets) * (temperature**2)

            loss = (hard_weight * hard_loss + soft_weight * soft_loss) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_update_step = (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader)

        if is_update_step:
            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item() * accum_steps, frames.size(0))

        probs = torch.softmax(student_logits, dim=1)[:, 1]
        preds_list.extend(probs.detach().cpu().numpy())
        labels_list.extend(labels.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    metrics = calculate_metrics(labels_list, preds_list)
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()

    preds_list = []
    labels_list = []

    pbar = tqdm(dataloader, desc="Evaluating", dynamic_ncols=True)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss_meter.update(loss.item(), frames.size(0))

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds_list.extend(probs.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    metrics = calculate_metrics(labels_list, preds_list)
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.inference_mode()
def predict(model, dataloader, device):
    """
    Run inference and return predictions together with their corresponding video IDs.
    """
    model.eval()

    preds_list = []
    video_ids_list = []

    pbar = tqdm(dataloader, desc="Predicting", dynamic_ncols=True)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)

        outputs = model(frames)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds_list.extend(probs.cpu().numpy())

        if "video_id" in batch:
            video_ids_list.extend(batch["video_id"])

    return video_ids_list, preds_list
