from pathlib import Path
import logging
import sys

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "timm",
    "PIL",
    "filelock",
)


def get_logger(log_cfg, name="DeepfakeDetection", log_file="train.log"):
    """
    Create logger.

    Args:
        log_cfg: cfg.logging
        name (str): Logger name.
        log_file (str): Log file name.

    Returns:
        logging.Logger
    """
    for noisy_name in _NOISY_LOGGERS:
        logging.getLogger(noisy_name).setLevel(logging.WARNING)

    log_dir = Path(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name.upper())

    # Prevent adding duplicate handlers
    if logger.handlers:
        return logger

    level = getattr(logging, log_cfg.level.upper(), logging.INFO)

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File
    file_handler = logging.FileHandler(
        log_dir / log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
