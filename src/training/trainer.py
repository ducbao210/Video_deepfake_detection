import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
from src.utils import save_history, plot_learning_curves, get_logger
from src.training import train_one_epoch, evaluate, predict as engine_predict


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
        use_amp = cfg.training.get("mixed_precision", False)
        if use_amp and self.device.type != "cuda":
            use_amp = False

        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None
        self.max_norm = cfg.training.get("gradient_clip", 0.0)

        # Logging configuration
        self.output_dir = Path(cfg.output_dir)

        # Use the checkpoint directory directly from the config
        self.checkpoint_dir = Path(cfg.logging.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            cfg.logging, name=cfg.experiment_name, log_file="train.log"
        )
        self.writer = (
            SummaryWriter(log_dir=cfg.logging.tensorboard_dir)
            if cfg.logging.tensorboard
            else None
        )

        # Early stpping configuration
        self.patience = cfg.callbacks.early_stopping.patience
        self.monitor = cfg.callbacks.early_stopping.monitor
        self.monitor_key = self.monitor.removeprefix("val_").removeprefix("train_")
        self.mode = cfg.callbacks.early_stopping.mode

        self.best_metric = float("-inf") if self.mode == "max" else float("inf")
        self.patience_counter = 0

        # history
        self.history = defaultdict(list)

        # resume training
        self.start_epoch = 1
        resume_path = cfg.training.get("resume_checkpoint", None)

        if resume_path and os.path.isfile(resume_path):
            self.logger.info(f"Loading checkpoint from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
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

            # Use the epoch stored in the checkpoint; fall back to the filename if needed
            last_epoch = checkpoint["epoch"]
            self.start_epoch = last_epoch + 1

            self.logger.info(
                f"Checkpoint loaded successfully! Resuming training from epoch {self.start_epoch}"
            )
        else:
            self.logger.info(
                "No valid checkpoint found. Starting training from scratch."
            )

    def fit(self):
        self.logger.info(
            f"Starting training for {self.cfg.experiment_name} on {self.device}..."
        )

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info(f"\n[{'='*20} Epoch {epoch}/{self.epochs} {'='*20}]")

            # training
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.scaler,
                self.max_norm,
                accum_steps=self.cfg.dataloader.get("accum_steps", 1),
            )

            # evaluation
            val_metrics = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

            if self.scheduler:
                self.scheduler.step()

            self.history["epoch"].append(epoch)
            for k, v in train_metrics.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_metrics.items():
                self.history[f"val_{k}"].append(v)

            self._log_epoch_results(epoch, train_metrics, val_metrics)

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
                    f"🏆 {self.monitor} cải thiện từ {self.best_metric:.4f} -> {val_monitor:.4f}"
                )
                self.best_metric = val_monitor
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"{self.monitor} did not improve. Patience: {self.patience_counter}/{self.patience}"
                )

            # Save the latest checkpoint after every epoch
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
        self.logger.info("Training completed!")

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
                self.writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Train/learning_rate", current_lr, epoch)

    def _save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
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
            self.logger.warning("HF_TOKEN not found. Skipping upload to Hugging Face.")
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
        Run inference on the test dataset.
        """

        self.logger.info(f"Running inference on {self.device}...")
        video_ids, preds = engine_predict(self.model, test_loader, self.device)
        self.logger.info(f"Inference completed for {len(video_ids)} videos!")
        return video_ids, preds
