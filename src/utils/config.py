from pathlib import Path
from omegaconf import OmegaConf


def load_config(config_dir="configs", model_name=None):
    """
    Load config theo kiểu Hydra.

    Args:
        config_dir: thư mục configs
        model_name: convnext / hybrid_bilstm / video_swin

    Returns:
        OmegaConf config
    """

    config_dir = Path(config_dir)

    # Load config chính
    base_cfg = OmegaConf.load(config_dir / "config.yaml")

    # Nếu không truyền model_name
    # lấy từ defaults trong config.yaml
    if model_name is None:
        for item in base_cfg.defaults:
            if isinstance(item, dict) and "model" in item:
                model_name = item["model"]
                break

    configs = []

    # paths
    configs.append(OmegaConf.load(config_dir / "paths/default.yaml"))

    # dataset
    configs.append(OmegaConf.load(config_dir / "dataset/default.yaml"))

    # dataloader
    configs.append(OmegaConf.load(config_dir / "dataloader/default.yaml"))

    # preprocessing
    configs.append(OmegaConf.load(config_dir / "preprocessing/default.yaml"))

    configs.append(OmegaConf.load(config_dir / "preprocessing/transforms/default.yaml"))

    # ==========================
    # MODEL CHỌN 1 CÁI DUY NHẤT
    # ==========================
    configs.append(OmegaConf.load(config_dir / f"model/{model_name}.yaml"))

    # training
    configs.append(OmegaConf.load(config_dir / "training/default.yaml"))

    # inference
    configs.append(OmegaConf.load(config_dir / "inference/default.yaml"))

    # callbacks
    configs.append(OmegaConf.load(config_dir / "callbacks/default.yaml"))

    # logging
    configs.append(OmegaConf.load(config_dir / "logging/default.yaml"))

    # merge
    cfg = OmegaConf.merge(*configs)

    OmegaConf.resolve(cfg)

    return cfg
