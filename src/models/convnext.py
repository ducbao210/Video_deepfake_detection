import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import timm


class ConvNeXtDetector(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        # Read model settings directly from the model configuration
        arch = model_config.architecture
        cls_cfg = model_config.classifier

        self.encoder = timm.create_model(
            model_name=arch.backbone,
            pretrained=arch.pretrained,
            num_classes=0,
        )

        # Freeze the backbone if specified in the configuration
        if arch.get("freeze_backbone", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Unfreeze the final stage
            for param in self.encoder.stages[-1].parameters():
                param.requires_grad = True

        in_features = self.encoder.num_features

        # Configure the chunk size to reduce GPU memory usage.
        self.chunk_size = arch.get("chunk_size", 64)
        self.use_grad_checkpoint = arch.get("grad_checkpoint", False)

        self.classifier = nn.Sequential(
            nn.Dropout(cls_cfg.dropout),
            nn.Linear(in_features, cls_cfg.num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        features_list = []
        for i in range(0, x.size(0), self.chunk_size):
            chunk = x[i : i + self.chunk_size]

            if self.training and self.use_grad_checkpoint:
                # Skip storing intermediate activations and recompute them during the backward pass
                feat = cp.checkpoint(self.encoder, chunk, use_reentrant=False)
            else:
                feat = self.encoder(chunk)

            features_list.append(feat)

        features = torch.cat(features_list, dim=0).reshape(B, T, -1)
        video_features = features.mean(dim=1)

        logits = self.classifier(video_features)

        return logits
