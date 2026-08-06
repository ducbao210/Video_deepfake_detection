from pathlib import Path

from omegaconf import OmegaConf


def load_config(config_dir="configs", model_name=None):
    """
    Load and merge project configuration files in a Hydra-like manner.

    Args:
        config_dir: Configuration directory.
        model_name: Model configuration to load
            (e.g., convnext, hybrid_bilstm, video_swin).

    Returns:
        OmegaConf configuration object.
    """

    config_dir = Path(config_dir)

    # Load the main configuration file.
    base_cfg = OmegaConf.load(config_dir / "config.yaml")

    # If no model is specified, use the one defined in config.yaml.
    if model_name is None:
        for item in base_cfg.defaults:
            if isinstance(item, dict) and "model" in item:
                model_name = item["model"]
                break

    configs = [base_cfg]

    # Paths
    configs.append(OmegaConf.load(config_dir / "paths/default.yaml"))

    # Dataset
    configs.append(OmegaConf.load(config_dir / "dataset/default.yaml"))

    # DataLoader
    configs.append(OmegaConf.load(config_dir / "dataloader/default.yaml"))

    # Preprocessing
    configs.append(OmegaConf.load(config_dir / "preprocessing/default.yaml"))
    configs.append(OmegaConf.load(config_dir / "preprocessing/transforms/default.yaml"))

    # Load only the selected model configuration.
    configs.append(OmegaConf.load(config_dir / f"model/{model_name}.yaml"))

    # Training
    configs.append(OmegaConf.load(config_dir / "training/default.yaml"))

    # Inference
    configs.append(OmegaConf.load(config_dir / "inference/default.yaml"))

    # Callbacks
    configs.append(OmegaConf.load(config_dir / "callbacks/default.yaml"))

    # Logging
    configs.append(OmegaConf.load(config_dir / "logging/default.yaml"))

    # Hugging Face
    configs.append(OmegaConf.load(config_dir / "huggingface/default.yaml"))

    # Merge all configuration files.
    cfg = OmegaConf.merge(*configs)

    # Resolve interpolations (e.g., ${...}).
    OmegaConf.resolve(cfg)

    return cfg
