import torch
import torch.nn as nn
import timm


class HybridConvNeXtBiLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        cnn_cfg = cfg.cnn
        lstm_cfg = cfg.lstm
        cls_cfg = cfg.classifier

        # ---------------------------
        # ConvNeXt Backbone
        # ---------------------------
        self.encoder = timm.create_model(
            model_name=cnn_cfg.backbone,
            pretrained=cnn_cfg.pretrained,
            num_classes=0,
        )

        if cnn_cfg.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        feature_dim = self.encoder.num_features

        # ---------------------------
        # BiLSTM
        # ---------------------------
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_cfg.hidden_size,
            num_layers=lstm_cfg.num_layers,
            batch_first=True,
            bidirectional=lstm_cfg.bidirectional,
            dropout=lstm_cfg.dropout if lstm_cfg.num_layers > 1 else 0.0,
        )

        lstm_out_dim = (
            lstm_cfg.hidden_size * 2 if lstm_cfg.bidirectional else lstm_cfg.hidden_size
        )

        # ---------------------------
        # Temporal Attention Layer
        # ---------------------------
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        # ---------------------------
        # Classifier
        # ---------------------------
        self.pooling = cfg.sequence.pooling

        self.classifier = nn.Sequential(
            nn.Dropout(cls_cfg.dropout),
            nn.Linear(lstm_out_dim, lstm_cfg.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_cfg.dropout),
            nn.Linear(lstm_cfg.hidden_size, cls_cfg.num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (B, T, C, H, W)

        Returns:
            logits: Tensor (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # (B,T,C,H,W) -> (B*T,C,H,W)
        x = x.reshape(B * T, C, H, W)

        # ---------------------------
        # Chunked CNN Feature Extraction (Chống OOM)
        # ---------------------------
        chunk_size = 64  # Bạn có thể giảm xuống 32 nếu GPU vẫn báo hết bộ nhớ
        features_list = []
        for i in range(0, x.size(0), chunk_size):
            chunk = x[i : i + chunk_size]
            features_list.append(self.encoder(chunk))

        features = torch.cat(features_list, dim=0)

        # (B*T,F) -> (B,T,F)
        features = features.reshape(B, T, -1)

        # ---------------------------
        # Temporal Modeling (BiLSTM)
        # ---------------------------
        lstm_out, _ = self.lstm(features)

        # ---------------------------
        # Temporal Pooling
        # ---------------------------
        if self.pooling == "mean":
            video_feature = lstm_out.mean(dim=1)

        elif self.pooling == "max":
            video_feature, _ = lstm_out.max(dim=1)

        elif self.pooling == "last":
            video_feature = lstm_out[:, -1]

        elif self.pooling == "attention":
            # 1. Tính trọng số Attention cho từng frame
            attn_weights = self.attention(lstm_out)  # (B, T, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)  # Chuẩn hóa trọng số

            # 2. Áp dụng trọng số lên output của BiLSTM
            weighted_out = lstm_out * attn_weights  # (B, T, lstm_out_dim)

            # 3. Tính tổng lại để ra feature đại diện cho cả video
            video_feature = weighted_out.sum(dim=1)  # (B, lstm_out_dim)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Final classification
        logits = self.classifier(video_feature)

        return logits
