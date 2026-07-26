# src/training/engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.utils import AverageMeter, calculate_metrics


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, scaler=None, max_norm=0.0
):
    model.train()
    loss_meter = AverageMeter()

    preds_list = []
    labels_list = []

    pbar = tqdm(dataloader, desc="Training", dynamic_ncols=True)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Training
        if scaler is not None:
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # Cập nhật metrics
        loss_meter.update(loss.item(), frames.size(0))

        # Lấy xác suất của class 1 (manipulated/fake) để tính AUC
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
):
    student.train()
    teacher.eval()  # Teacher luôn ở chế độ eval

    loss_meter = AverageMeter()
    preds_list = []
    labels_list = []

    kd_criterion = nn.KLDivLoss(reduction="batchmean")

    pbar = tqdm(dataloader, desc="KD Training", dynamic_ncols=True)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                with torch.no_grad():
                    teacher_logits = teacher(frames)

                student_logits = student(frames)

                # Hard loss: Cross Entropy gốc
                hard_loss = criterion(student_logits, labels)

                # Soft loss: KL Divergence (có scale theo temperature^2)
                soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                soft_loss = kd_criterion(student_log_probs, soft_targets) * (
                    temperature**2
                )

                # Total loss
                loss = hard_weight * hard_loss + soft_weight * soft_loss

            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                teacher_logits = teacher(frames)

            student_logits = student(frames)

            hard_loss = criterion(student_logits, labels)
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            soft_loss = kd_criterion(student_log_probs, soft_targets) * (temperature**2)

            loss = hard_weight * hard_loss + soft_weight * soft_loss

            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
            optimizer.step()

        loss_meter.update(loss.item(), frames.size(0))

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
    Hàm thực hiện inference và trả về kết quả dự đoán cùng video_id.
    """
    model.eval()

    preds_list = []
    video_ids_list = []

    pbar = tqdm(dataloader, desc="Predicting", dynamic_ncols=True)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)

        outputs = model(frames)

        # Lấy xác suất của class 1 (manipulated/fake)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds_list.extend(probs.cpu().numpy())

        # Trích xuất video_id nếu có trong batch
        if "video_id" in batch:
            video_ids_list.extend(batch["video_id"])

    return video_ids_list, preds_list
