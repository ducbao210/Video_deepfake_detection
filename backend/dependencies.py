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
        "CHECKPOINT_PATH",
        str(ROOT / "outputs" / "checkpoints" / model_name / "best.pth"),
    )

    # Merge the configuration files similarly to Hydra, without using @hydra.main
    parts = [
        OmegaConf.load(config_dir / "paths/default.yaml"),
        OmegaConf.load(config_dir / "dataset/default.yaml"),
        OmegaConf.load(config_dir / "preprocessing/default.yaml"),
        OmegaConf.load(config_dir / "preprocessing/transforms/default.yaml"),
        OmegaConf.load(config_dir / f"model/{model_name}.yaml"),
        OmegaConf.load(config_dir / "inference/default.yaml"),
        OmegaConf.load(config_dir / "huggingface/default.yaml"),  # required for cfg.huggingface.* access
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
        print(
            f"Checkpoint not found locally at {ckpt_path}. Attempting to download from Hugging Face..."
        )

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import enable_progress_bars

        # Bật hiển thị thanh tiến trình (progress bar)
        enable_progress_bars()

        try:
            downloaded_path = hf_hub_download(
                repo_id=cfg.huggingface.repo_id,
                filename=f"{cfg.huggingface.path_in_repo}/best.pth",
            )
            ckpt_path = Path(downloaded_path)
            print(f"Download complete! Checkpoint saved at: {ckpt_path}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to download checkpoint: {e}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    return ModelBundle(model, get_val_transforms(cfg), cfg, device)
