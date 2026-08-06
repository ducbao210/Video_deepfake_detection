import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


class ModelBundle:
    def __init__(self, model, transform, cfg, device):
        self.model = model
        self.transform = transform
        self.cfg = cfg
        self.device = device


class ModelManager:
    """Manages multiple model bundles with lazy loading and caching."""

    def __init__(self):
        self._bundles: Dict[str, ModelBundle] = {}

    def get_available_models(self) -> list:
        """Return list of available model names by reading model.name from config files."""
        model_dir = ROOT / "configs" / "model"
        models = []
        for p in sorted(model_dir.glob("*.yaml")):
            try:
                cfg = OmegaConf.load(p)
                name = cfg.get("model", {}).get("name")
                if name and name not in models:
                    models.append(name)
            except Exception:
                name = p.stem
                if name not in models:
                    models.append(name)
        return models

    def get_model_bundle(self, model_name: str) -> ModelBundle:
        """Get or create a model bundle for the specified model."""
        if model_name in self._bundles:
            return self._bundles[model_name]

        bundle = self._load_model(model_name)
        self._bundles[model_name] = bundle
        return bundle

    def _find_config_for_model(self, model_name: str) -> Path:
        """Find the config file that contains the given model name."""
        model_dir = ROOT / "configs" / "model"
        for p in sorted(model_dir.glob("*.yaml")):
            try:
                cfg = OmegaConf.load(p)
                if cfg.get("model", {}).get("name") == model_name:
                    return p
            except Exception:
                continue
        direct_path = model_dir / f"{model_name}.yaml"
        if direct_path.exists():
            return direct_path
        raise FileNotFoundError(f"No config file found for model '{model_name}'")

    def _load_model(self, model_name: str) -> ModelBundle:
        """Load a model bundle."""
        import os
        import torch

        from src.data.transforms import get_val_transforms
        from src.models import build_model

        checkpoint = os.getenv(
            "CHECKPOINT_PATH",
            str(ROOT / "outputs" / model_name / "checkpoints" / "best.pth"),
        )
        cfg = load_backend_config(model_name, checkpoint)

        device_env = os.getenv("DEVICE", "auto")
        if device_env == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_env)

        if "architecture" in cfg.model and cfg.model.architecture.get("pretrained", False):
            cfg.model.architecture.pretrained = False
        elif "cnn" in cfg.model and cfg.model.cnn.get("pretrained", False):
            cfg.model.cnn.pretrained = False

        model = build_model(cfg)

        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            print(
                f"Checkpoint not found locally at {ckpt_path}. "
                f"Attempting to download from Hugging Face..."
            )

            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import enable_progress_bars

            enable_progress_bars()

            try:
                downloaded_path = hf_hub_download(
                    repo_id=cfg.huggingface.repo_id,
                    filename=f"{cfg.huggingface.path_in_repo}/best.pth",
                    token=cfg.huggingface.token,
                    local_dir=ckpt_path.parent,
                )
                downloaded_path = Path(downloaded_path)

                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                if downloaded_path.resolve() != ckpt_path.resolve():
                    shutil.copy2(downloaded_path, ckpt_path)

                print(f"Download complete! Checkpoint saved at: {ckpt_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to download checkpoint: {e}\n"
                    f"Run `python scripts/train.py model={model_name}` to train it yourself, "
                    f"or check HF_TOKEN in .env if the repo is private."
                )

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()

        return ModelBundle(model, get_val_transforms(cfg), cfg, device)


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def load_backend_config(model_name: str, checkpoint: str):
    """Load config by finding the correct yaml file for the model name."""
    cfg = load_config(config_dir=str(ROOT / "configs"), model_name=model_name)

    if not cfg.get("experiment_name"):
        cfg.experiment_name = model_name

    if not cfg.huggingface.get("path_in_repo"):
        cfg.huggingface.path_in_repo = f"checkpoints/{cfg.experiment_name}"

    cfg.inference.checkpoint = checkpoint
    return cfg


def get_model_bundle() -> ModelBundle:
    """Get the default model bundle (backward compatibility)."""
    import os

    model_name = os.getenv("MODEL_NAME", "convnext")
    return get_model_manager().get_model_bundle(model_name)
