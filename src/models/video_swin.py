import torch
import torch.nn as nn
from torchvision.models.video import (
    swin3d_b,
    swin3d_s,
    swin3d_t,
    Swin3D_B_Weights,
    Swin3D_S_Weights,
    Swin3D_T_Weights,
)

_BUILDERS = {
    "swin3d_t": (swin3d_t, Swin3D_T_Weights.KINETICS400_V1),
    "swin3d_s": (swin3d_s, Swin3D_S_Weights.KINETICS400_V1),
    "swin3d_b": (swin3d_b, Swin3D_B_Weights.KINETICS400_V1),
}


class VideoSwinDetector(nn.Module):
    """
    Video Swin Transformer with 3D shifted-window attention
    across both spatial and temporal dimensions.

    Input : (B, T, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, model_config):
        super().__init__()

        arch = model_config.architecture
        cls_cfg = model_config.classifier

        backbone = arch.get("backbone", "swin3d_t")
        if backbone not in _BUILDERS:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Available backbones: {list(_BUILDERS)}"
            )

        builder, weights = _BUILDERS[backbone]
        self.encoder = builder(
            weights=weights if arch.get("pretrained", True) else None
        )

        in_features = self.encoder.head.in_features
        self.encoder.head = nn.Identity()

        if arch.get("freeze_backbone", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.encoder.features[-1].parameters():
                param.requires_grad = True

            for param in self.encoder.norm.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(cls_cfg.dropout),
            nn.Linear(in_features, cls_cfg.num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        features = self.encoder(x)
        return self.classifier(features)
