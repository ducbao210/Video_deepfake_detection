import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn as nn
from transformers import TimesformerConfig, TimesformerModel


class TimeSformerDetector(nn.Module):
    """
    Input : (B, T, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, model_config):
        super().__init__()
        arch = model_config.architecture
        cls_cfg = model_config.classifier

        pretrained_id = arch.get(
            "pretrained_model", "facebook/timesformer-base-finetuned-k400"
        )
        num_frames = int(arch.get("num_frames", 8))
        image_size = int(arch.get("image_size", 224))

        if arch.get("pretrained", True):
            # Allow loading with different frame or image sizes.
            self.encoder = TimesformerModel.from_pretrained(
                pretrained_id,
                num_frames=num_frames,
                image_size=image_size,
                ignore_mismatched_sizes=True,
            )
        else:
            config = TimesformerConfig(num_frames=num_frames, image_size=image_size)
            self.encoder = TimesformerModel(config)

        if arch.get("freeze_backbone", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Re-enable temporal-position parameters
            for name, param in self.encoder.named_parameters():
                if "time_embeddings" in name or "temporal" in name:
                    param.requires_grad = True

            for param in self.encoder.encoder.layer[-1].parameters():
                param.requires_grad = True

            for param in self.encoder.layernorm.parameters():
                param.requires_grad = True

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(cls_cfg.dropout),
            nn.Linear(hidden_size, cls_cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        expected_frames = self.encoder.config.num_frames
        assert (
            T == expected_frames
        ), f"Input has {T} frames, but the model expects {expected_frames}."

        outputs = self.encoder(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)
