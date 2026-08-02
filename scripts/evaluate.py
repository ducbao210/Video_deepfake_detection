import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import (
    get_logger,
    seed_everything,
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
)
from src.data.dataset import DeepfakeDataset
from src.data.transforms import get_val_transforms
from src.models import build_model
from src.training.engine import predict


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger(cfg.logging, name=cfg.experiment_name, log_file="evaluate.log")
    seed_everything(cfg.seed)

    device = torch.device(
        cfg.device
        if cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Prepare datasets
    val_transforms = get_val_transforms(cfg)
    test_dataset = DeepfakeDataset(
        cfg.paths.data.test_csv, cfg, transform=val_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    # Load model weights
    model = build_model(cfg).to(device)
    checkpoint_path = cfg.inference.checkpoint

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    # Pick the best threshold on validation data
    val_dataset = DeepfakeDataset(
        cfg.paths.data.val_csv, cfg, transform=get_val_transforms(cfg)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )
    with torch.inference_mode():
        _, val_probs = predict(model, val_loader, device)
        val_probs = np.asarray(val_probs)
        val_labels = val_dataset.data["label"].to_numpy()

        candidates = np.linspace(0.05, 0.95, 19)
        best_threshold = max(
            candidates,
            key=lambda t: f1_score(val_labels, (val_probs >= t).astype(int)),
        )
        logger.info(f"Optimal F1 threshold on the validation set: {best_threshold:.2f}")
        # Evaluate on the test set
        _, test_probs = predict(model, test_loader, device)
        test_probs = np.asarray(test_probs)
        test_labels = test_dataset.data["label"].to_numpy()

    metrics = calculate_metrics(test_labels, test_probs, threshold=best_threshold)
    cm = confusion_matrix(
        test_labels, (test_probs >= best_threshold).astype(int), labels=[1, 0]
    )
    report_dir = Path(cfg.inference.output_path) / "eval_report"
    plot_confusion_matrix(cm, save_path=report_dir / "confusion_matrix.png")
    plot_roc_curve(test_labels, test_probs, save_path=report_dir / "roc_curve.png")
    plot_pr_curve(test_labels, test_probs, save_path=report_dir / "pr_curve.png")
    logger.info(f"Evaluating plots saved at: {report_dir}")

    # Log results

    logger.info("========== TEST SET EVALUATION RESULTS ==========")

    logger.info(f"Threshold: {best_threshold:.2f} (selected on the validation set)")
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "ap",
    ):
        if name in metrics:
            logger.info(f"{name:18s}: {metrics[name]:.4f}")
    logger.info(f"CONFUSION MATRIX [[TP FN][FP TN]]:\n{cm}")
    logger.info("=================================================")


if __name__ == "__main__":
    main()
