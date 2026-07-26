import torch
import torch.nn as nn
import timm


class VideoSwinDetector(nn.Module):
    def __init__(self, model_config):  # Đổi tên biến thành model_config cho rõ ràng
        super().__init__()

        # Gọi trực tiếp từ model_config
        arch = model_config.architecture
        cls_cfg = model_config.classifier

        self.encoder = timm.create_model(
            model_name=arch.backbone,
            pretrained=arch.pretrained,
            num_classes=0,
        )

        in_features = self.encoder.num_features

        # Thêm chunk_size vào để chống OOM
        self.chunk_size = model_config.get("chunk_size", 64)

        self.classifier = nn.Sequential(
            nn.Dropout(cls_cfg.dropout),
            nn.Linear(in_features, cls_cfg.num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)

        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # (B,T,C,H,W) -> (B*T,C,H,W)
        x = x.reshape(B * T, C, H, W)

        # ---------------------------
        # Áp dụng Chunking (Tránh OOM) thay vì đẩy trực tiếp tựa như features = self.encoder(x)
        # ---------------------------
        features_list = []
        for i in range(0, x.size(0), self.chunk_size):
            chunk = x[i : i + self.chunk_size]
            features_list.append(self.encoder(chunk))

        features = torch.cat(features_list, dim=0)

        # (B*T,F) -> (B,T,F)
        features = features.reshape(B, T, -1)

        # Temporal Average Pooling
        video_features = features.mean(dim=1)

        logits = self.classifier(video_features)

        return logits
