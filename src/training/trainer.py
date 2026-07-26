# src/training/trainer.py
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.training import train_one_epoch, evaluate, predict as engine_predict
from src.utils import get_logger
from src.utils import extract_epoch_from_filename


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler, cfg
    ):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
            if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = cfg.training.epochs
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if cfg.training.get("mixed_precision", False)
            else None
        )
        self.max_norm = cfg.training.get("gradient_clip", 0.0)

        # Cấu hình logging
        self.output_dir = Path(cfg.output_dir)
        self.checkpoint_dir = self.output_dir / cfg.logging.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")

        self.logger = get_logger(cfg.logging, log_file="train.log")
        self.writer = (
            SummaryWriter(log_dir=self.output_dir / cfg.logging.tensorboard_dir)
            if cfg.logging.tensorboard
            else None
        )

        # Cấu hình Early Stopping
        self.patience = cfg.callbacks.early_stopping.patience
        self.monitor = cfg.callbacks.early_stopping.monitor
        self.mode = cfg.callbacks.early_stopping.mode

        self.best_metric = float("-inf") if self.mode == "max" else float("inf")
        self.patience_counter = 0

        # ==========================================
        # CƠ CHẾ RESUME TRAINING
        # ==========================================
        self.start_epoch = 1
        resume_path = cfg.training.get("resume_checkpoint", None)

        if resume_path and os.path.isfile(resume_path):
            self.logger.info(f"🔄 Đang tải checkpoint từ: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Extract epoch từ tên file theo yêu cầu
            extracted_epoch = extract_epoch_from_filename(resume_path)
            self.start_epoch = extracted_epoch + 1

            # (Tùy chọn) Phục hồi lại best_metric nếu có trong checkpoint
            if "metrics" in checkpoint and self.monitor in checkpoint["metrics"]:
                self.best_metric = checkpoint["metrics"][self.monitor]

            self.logger.info(
                f"✅ Đã tải thành công! Tiếp tục training từ epoch {self.start_epoch}"
            )
        else:
            self.logger.info(
                "▶️ Không có checkpoint (hoặc đường dẫn không hợp lệ). Bắt đầu train từ đầu."
            )

    def fit(self):
        self.logger.info(
            f"Bắt đầu huấn luyện {self.cfg.experiment_name} trên {self.device}..."
        )

        # CẬP NHẬT: Vòng lặp bắt đầu từ self.start_epoch thay vì 1
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info(f"\n[{'='*20} Epoch {epoch}/{self.epochs} {'='*20}]")

            # Huấn luyện
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.scaler,
                self.max_norm,
            )

            # Đánh giá
            val_metrics = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

            if self.scheduler:
                self.scheduler.step()

            self._log_epoch_results(epoch, train_metrics, val_metrics)

            # Kiểm tra Early Stopping và lưu checkpoint
            val_monitor = val_metrics[self.monitor]
            is_best = (
                (val_monitor > self.best_metric)
                if self.mode == "max"
                else (val_monitor < self.best_metric)
            )

            if is_best:
                self.logger.info(
                    f"🏆 {self.monitor} cải thiện từ {self.best_metric:.4f} -> {val_monitor:.4f}"
                )
                self.best_metric = val_monitor
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"⚠️ {self.monitor} không cải thiện. Patience: {self.patience_counter}/{self.patience}"
                )

            # CẬP NHẬT: Lưu file checkpoint theo từng epoch
            if self.cfg.logging.get("save_last", True):
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            if self.patience_counter >= self.patience:
                self.logger.info(f"🛑 Kích hoạt Early stopping tại epoch {epoch}!")
                break

        if self.writer:
            self.writer.close()
        self.logger.info("Hoàn tất huấn luyện!")

    def _log_epoch_results(self, epoch, train_metrics, val_metrics):
        log_str = (
            f"Train - Loss: {train_metrics['loss']:.4f} - Acc: {train_metrics['accuracy']:.4f} - AUC: {train_metrics['auc']:.4f} | "
            f"Val - Loss: {val_metrics['loss']:.4f} - Acc: {val_metrics['accuracy']:.4f} - AUC: {val_metrics['auc']:.4f}"
        )
        self.logger.info(log_str)

        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            # Lấy learning rate hiện tại
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Train/learning_rate", current_lr, epoch)

    def _save_checkpoint(self, epoch, metrics, is_best=False, filename=None):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }

        if is_best:
            # File 1: Best checkpoint
            save_path = self.checkpoint_dir / f"best_{self.run_id}.pth"
            torch.save(state, save_path)
            self.logger.info(f"Lưu best model checkpoint tại: {save_path}")
        else:
            # File 2: Checkpoint đánh số epoch theo yêu cầu (ví dụ: checkpoint_epoch_15.pth)
            actual_filename = (
                filename if filename else f"checkpoint_epoch_{epoch}_{self.run_id}.pth"
            )
            save_path = self.checkpoint_dir / actual_filename
            torch.save(state, save_path)

    def predict(self, test_loader):
        """
        Thực hiện inference trên tập dữ liệu kiểm tra.
        """
        self.logger.info(f"Bắt đầu dự đoán trên {self.device}...")

        video_ids, preds = engine_predict(self.model, test_loader, self.device)

        self.logger.info(f"Hoàn tất dự đoán cho {len(video_ids)} videos!")
        return video_ids, preds
