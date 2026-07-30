# src/training/kd_trainer.py
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
from src.utils import save_history, plot_learning_curves, get_logger
from src.training import train_one_epoch_kd, evaluate, predict as engine_predict


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

        # Freeze all teacher model parameters
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # KD settings from kd_training.yaml
        self.temperature = cfg.training.get("temperature", 4.0)
        self.hard_weight = cfg.training.get("hard_loss_weight", 0.3)
        self.soft_weight = cfg.training.get("soft_loss_weight", 0.7)

        self.epochs = cfg.training.epochs
        use_amp = cfg.training.get("mixed_precision", False)
        if use_amp and self.device.type != "cuda":
            use_amp = False

        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None
        self.max_norm = cfg.training.get("gradient_clip", 0.0)

        # Logging & checkpointing
        self.output_dir = Path(cfg.output_dir)
        self.log_dir = Path(cfg.logging.log_dir)
        self.checkpoint_dir = Path(cfg.logging.checkpoint_dir)
        self.tensorboard_dir = Path(cfg.logging.tensorboard_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        cfg.logging.log_dir = str(self.log_dir)

        self.logger = get_logger(
            cfg.logging,
            name=cfg.experiment_name,
            log_file="kd_train.log",
        )

        self.writer = (
            SummaryWriter(log_dir=self.tensorboard_dir)
            if cfg.logging.tensorboard
            else None
        )

        # Early stopping configuration
        self.patience = cfg.callbacks.early_stopping.patience
        self.monitor = cfg.callbacks.early_stopping.monitor
        self.monitor_key = self.monitor.removeprefix("val_").removeprefix("train_")
        self.mode = cfg.callbacks.early_stopping.mode

        self.best_metric = float("-inf") if self.mode == "max" else float("inf")
        self.patience_counter = 0

        # History
        self.history = defaultdict(list)

        # Resume training
        self.start_epoch = 1
        resume_path = cfg.training.get("resume_checkpoint", None)

        if resume_path and os.path.isfile(resume_path):

            self.logger.info(f"Loading checkpoint from: {resume_path}")
            checkpoint = torch.load(
                resume_path, map_location=self.device, weights_only=False
            )

            self.student.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if (
                self.scheduler is not None
                and checkpoint.get("scheduler_state_dict") is not None
            ):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if (
                self.scaler is not None
                and checkpoint.get("scaler_state_dict") is not None
            ):
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            self.best_metric = checkpoint.get("best_metric", self.best_metric)
            self.patience_counter = checkpoint.get("patience_counter", 0)

            last_epoch = checkpoint["epoch"]
            self.start_epoch = last_epoch + 1

            self.logger.info(
                f"Checkpoint loaded successfully. Resuming KD training from epoch {self.start_epoch}."
            )
        else:
            self.logger.info(
                "No valid checkpoint found. Starting training from scratch."
            )

    def fit(self):
        self.logger.info(f"Starting Knowledge Distillation on {self.device}...")
        self.logger.info(
            f"Teacher: {self.teacher.__class__.__name__} -> Student: {self.student.__class__.__name__}"
        )
        self.logger.info(
            f"T={self.temperature}, Alpha(Hard)={self.hard_weight}, Soft={self.soft_weight}"
        )

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info(f"\n[{'='*20} Epoch {epoch}/{self.epochs} {'='*20}]")

            # Train the student with knowledge distillation.
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
                accum_steps=self.cfg.dataloader.get("accum_steps", 1),
            )

            # Evaluate the student model
            val_metrics = evaluate(
                self.student, self.val_loader, self.criterion, self.device
            )

            if self.scheduler:
                self.scheduler.step()

            self._log_epoch_results(epoch, train_metrics, val_metrics)

            self.history["epoch"].append(epoch)
            for k, v in train_metrics.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_metrics.items():
                self.history[f"val_{k}"].append(v)
            # Early stopping check.
            if self.monitor_key not in val_metrics:
                raise KeyError(
                    f"callbacks.early_stopping.monitor='{self.monitor}' does not match "
                    f"the available metrics: {sorted(val_metrics)}. "
                    f"Please use one of: {sorted('val_' + k for k in val_metrics)}"
                )
            val_monitor = val_metrics[self.monitor_key]
            is_best = (
                (val_monitor > self.best_metric)
                if self.mode == "max"
                else (val_monitor < self.best_metric)
            )

            if is_best:
                self.logger.info(
                    f"🏆 {self.monitor} improved from {self.best_metric:.4f} to {val_monitor:.4f}"
                )
                self.best_metric = val_monitor
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"{self.monitor} did not improve. Patience: {self.patience_counter}/{self.patience}"
                )

            if self.cfg.logging.get("save_last", True):
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}!")
                break

        if self.writer:
            self.writer.close()
        if self.cfg.huggingface.get("enabled", False) and self.cfg.logging.get(
            "save_last", True
        ):
            self._upload_checkpoint(self.checkpoint_dir / "last.pth")

        save_history(self.history, self.output_dir / "history.json")
        plot_learning_curves(
            self.history,
            save_path=self.output_dir / "plots" / "learning_curves.png",
            metrics=("loss", "auc", "f1"),
        )
        self.logger.info("Knowledge Distillation training completed.")

    def _log_epoch_results(self, epoch, train_metrics, val_metrics):
        train_str = " - ".join(
            [f"{k.capitalize()}: {v:.4f}" for k, v in train_metrics.items()]
        )
        val_str = " - ".join(
            [f"{k.capitalize()}: {v:.4f}" for k, v in val_metrics.items()]
        )

        self.logger.info(f"Train - {train_str} | Val - {val_str}")

        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"KD_Train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"KD_Val/{k}", v, epoch)
            self.writer.add_scalar(
                "KD_Train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch
            )

    def _save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": (self.scaler.state_dict() if self.scaler else None),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
            "config": self.cfg,
        }

        if is_best:
            save_path = self.checkpoint_dir / "best.pth"
            self.logger.info(f"Saving best checkpoint: {save_path}")
        else:
            save_path = self.checkpoint_dir / "last.pth"
        torch.save(state, save_path)
        # only upload new best ckpt
        if is_best and self.cfg.huggingface.get("enabled", False):
            self._upload_checkpoint(save_path)

    def _upload_checkpoint(self, checkpoint_path):
        from huggingface_hub import HfApi
        from dotenv import load_dotenv
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()

        load_dotenv(".env")
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            self.logger.warning("HF_TOKEN is not set. Skipping upload to Hugging Face.")
            return

        api = HfApi(token=hf_token)

        try:
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=f"{self.cfg.huggingface.path_in_repo}/{checkpoint_path.name}",
                repo_id=self.cfg.huggingface.repo_id,
                repo_type="model",
                token=hf_token,
            )
            self.logger.info(f"Uploaded {checkpoint_path.name} to Hugging Face.")
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")

    def predict(self, test_loader):
        """
        Run inference on the test set using the student model.
        """
        self.logger.info(
            f"Running inference with the student model on {self.device}..."
        )
        video_ids, preds = engine_predict(self.student, test_loader, self.device)
        self.logger.info(f"Inference completed for {len(video_ids)} videos.")
        return video_ids, preds
