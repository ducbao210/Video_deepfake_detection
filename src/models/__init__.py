import importlib

MODEL_REGISTRY = {
    "convnext": (".convnext", "ConvNeXtDetector"),
    "hybrid_bilstm": (".hybrid_convnext_bilstm", "HybridConvNeXtBiLSTM"),
    "video_swin": (".video_swin", "VideoSwinDetector"),
    "timesformer": (".timesformer", "TimeSformerDetector"),
}


def build_model(cfg):
    model_name = cfg.model.name

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    module_path, class_name = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path, package=__name__)
    model_cls = getattr(module, class_name)

    return model_cls(cfg.model)


__all__ = ["build_model", "MODEL_REGISTRY"]
