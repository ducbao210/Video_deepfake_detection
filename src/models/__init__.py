from .convnext import ConvNeXtDetector
from .hybrid_convnext_bilstm import HybridConvNeXtBiLSTM
from .video_swin import VideoSwinDetector
from .timesformer import TimeSformerDetector

MODEL_REGISTRY = {
    "convnext": ConvNeXtDetector,
    "hybrid_bilstm": HybridConvNeXtBiLSTM,
    "video_swin": VideoSwinDetector,
    "timesformer": TimeSformerDetector,
}


def build_model(cfg):
    model_name = cfg.model.name

    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return model_cls(cfg.model)


__all__ = ["build_model", "MODEL_REGISTRY"]
