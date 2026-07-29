import sys
from functools import lru_cache
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.transforms import get_val_transforms
from src.models import build_model


class ModelBundle:
    def __init__(self, model, transform, cfg, device):
        self.model = model
        self.transform = transform
        self.cfg = cfg
        self.device = device


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    import os

    config_dir = ROOT / "configs"
    model_name = os.getenv("MODEL_NAME", "convnext")
    checkpoint = os.getenv(
        "CHECKPOINT_PATH", str(ROOT / "checkpoints" / model_name / "best.pth")
    )

    # Merge the configuration files similarly to Hydra, without using @hydra.main
    parts = [
        OmegaConf.load(config_dir / "paths/default.yaml"),
        OmegaConf.load(config_dir / "dataset/default.yaml"),
        OmegaConf.load(config_dir / "preprocessing/default.yaml"),
        OmegaConf.load(config_dir / "preprocessing/transforms/default.yaml"),
        OmegaConf.load(config_dir / f"model/{model_name}.yaml"),
        OmegaConf.load(config_dir / "inference/default.yaml"),
    ]
    cfg = OmegaConf.merge(*parts)
    cfg.inference.checkpoint = checkpoint

    device_env = os.getenv("DEVICE", "auto")
    if device_env == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_env)
    model = build_model(cfg)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Please set the CHECKPOINT_PATH environment variable correctly."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    return ModelBundle(model, get_val_transforms(cfg), cfg, device)
