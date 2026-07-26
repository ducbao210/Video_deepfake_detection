# src/training/kd_trainer.py
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


from src.training import train_one_epoch_kd, evaluate, predict as engine_predict
from src.utils import get_logger
from src.utils import extract_epoch_from_filename  # Import hàm trích xuất epoch


class KDTrainer:
    def __init__(
        self,
        student_model,
        teacher_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        cfg,
    ):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
            if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.student = student_model.to(self.device)
        self.teacher = teacher_model.to(self.device)

        # Đóng băng toàn bộ trọng số của teacher[cite: 5]
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # KD Configs từ kd_training.yaml[cite: 5]
        self.temperature = cfg.training.get("temperature", 4.0)
        self.hard_weight = cfg.training.get("hard_loss_weight", 0.3)
        self.soft_weight = cfg.training.get("soft_loss_weight", 0.7)

        self.epochs = cfg.training.epochs
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if cfg.training.get("mixed_precision", False)
            else None
        )
        self.max_norm = cfg.training.get("gradient_clip", 0.0)

        # Logging & Checkpoint[cite: 5]
        self.output_dir = Path(cfg.output_dir)
        self.checkpoint_dir = self.output_dir / cfg.logging.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")

        self.logger = get_logger(cfg.logging, log_file="kd_train.log")
        self.writer = (
            SummaryWriter(log_dir=self.output_dir / cfg.logging.tensorboard_dir)
            if cfg.logging.tensorboard
            else None
        )

        # Early Stopping[cite: 5]
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

            self.student.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Extract epoch từ tên file theo yêu cầu
            extracted_epoch = extract_epoch_from_filename(resume_path)
            self.start_epoch = extracted_epoch + 1

            # (Tùy chọn) Phục hồi lại best_metric nếu có trong checkpoint
            if "metrics" in checkpoint and self.monitor in checkpoint["metrics"]:
                self.best_metric = checkpoint["metrics"][self.monitor]

            self.logger.info(
                f"✅ Đã tải thành công! Tiếp tục training KD từ epoch {self.start_epoch}"
            )
        else:
            self.logger.info(
                "▶️ Không có checkpoint (hoặc đường dẫn không hợp lệ). Bắt đầu train từ đầu."
            )

    def fit(self):
        self.logger.info(f"Bắt đầu Knowledge Distillation trên {self.device}...")
        self.logger.info(
            f"Teacher: {self.teacher.__class__.__name__} -> Student: {self.student.__class__.__name__}"
        )
        self.logger.info(
            f"T={self.temperature}, Alpha(Hard)={self.hard_weight}, Soft={self.soft_weight}"
        )

        # CẬP NHẬT: Vòng lặp bắt đầu từ self.start_epoch thay vì 1
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info(f"\n[{'='*20} Epoch {epoch}/{self.epochs} {'='*20}]")

            # Huấn luyện KD[cite: 5]
            train_metrics = train_one_epoch_kd(
                self.student,
                self.teacher,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.temperature,
                self.hard_weight,
                self.soft_weight,
                self.scaler,
                self.max_norm,
            )

            # Đánh giá Student[cite: 5]
            val_metrics = evaluate(
                self.student, self.val_loader, self.criterion, self.device
            )

            if self.scheduler:
                self.scheduler.step()

            self._log_epoch_results(epoch, train_metrics, val_metrics)

            # Early Stopping check[cite: 5]
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

            # CẬP NHẬT: Bỏ filename cố định "kd_last.pth" để hàm _save_checkpoint tự động gán số epoch
            if self.cfg.logging.get("save_last", True):
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            if self.patience_counter >= self.patience:
                self.logger.info(f"🛑 Kích hoạt Early stopping tại epoch {epoch}!")
                break

        if self.writer:
            self.writer.close()
        self.logger.info("Hoàn tất chắt lọc tri thức (KD)!")

    def _log_epoch_results(self, epoch, train_metrics, val_metrics):
        log_str = (
            f"Train - Loss: {train_metrics['loss']:.4f} - Acc: {train_metrics['accuracy']:.4f} - AUC: {train_metrics['auc']:.4f} | "
            f"Val - Loss: {val_metrics['loss']:.4f} - Acc: {val_metrics['accuracy']:.4f} - AUC: {val_metrics['auc']:.4f}"
        )
        self.logger.info(log_str)

        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"KD_Train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"KD_Val/{k}", v, epoch)
            self.writer.add_scalar(
                "KD_Train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch
            )

    def _save_checkpoint(self, epoch, metrics, is_best=False, filename=None):
        state = {
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),  # CẬP NHẬT: Lưu self.student thay vì self.model
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }

        if is_best:
            # Lưu file tốt nhất mang theo run_id
            save_path = self.checkpoint_dir / f"kd_best_{self.run_id}.pth"
            torch.save(state, save_path)
            self.logger.info(f"Lưu best model checkpoint tại: {save_path}")
        else:
            # Lưu file đánh số epoch mang theo run_id
            actual_filename = (
                filename
                if filename
                else f"kd_checkpoint_epoch_{epoch}_{self.run_id}.pth"
            )
            save_path = self.checkpoint_dir / actual_filename
            torch.save(state, save_path)

    def predict(self, test_loader):
        """
        Thực hiện inference trên tập dữ liệu kiểm tra bằng mô hình Student.
        """
        self.logger.info(f"Bắt đầu dự đoán bằng mô hình Student trên {self.device}...")

        # Lưu ý: truyền self.student để thực hiện dự đoán
        video_ids, preds = engine_predict(self.student, test_loader, self.device)

        self.logger.info(f"Hoàn tất dự đoán cho {len(video_ids)} videos!")
        return video_ids, preds
