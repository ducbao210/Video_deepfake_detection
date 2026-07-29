import torch
import torch.nn as nn
import timm
import torch.utils.checkpoint as cp


class HybridConvNeXtBiLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        cnn_cfg = cfg.cnn
        lstm_cfg = cfg.lstm
        cls_cfg = cfg.classifier

        self.encoder = timm.create_model(
            model_name=cnn_cfg.backbone,
            pretrained=cnn_cfg.pretrained,
            num_classes=0,
        )

        if cnn_cfg.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        feature_dim = self.encoder.num_features
        self.chunk_size = cnn_cfg.get("chunk_size", 64)
        self.use_grad_checkpoint = cnn_cfg.get("grad_checkpoint", False)

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

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

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
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Classification logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        features_list = []

        for i in range(0, x.size(0), self.chunk_size):
            chunk = x[i : i + self.chunk_size]

            if self.training and self.use_grad_checkpoint:
                feat = cp.checkpoint(
                    self.encoder,
                    chunk,
                    use_reentrant=False,
                )
            else:
                feat = self.encoder(chunk)

            features_list.append(feat)

        features = torch.cat(features_list, dim=0)
        features = features.reshape(B, T, -1)

        lstm_out, _ = self.lstm(features)

        if self.pooling == "mean":
            video_feature = lstm_out.mean(dim=1)

        elif self.pooling == "max":
            video_feature, _ = lstm_out.max(dim=1)

        elif self.pooling == "last":
            video_feature = lstm_out[:, -1]

        elif self.pooling == "attention":
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)

            weighted_out = lstm_out * attn_weights
            video_feature = weighted_out.sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.classifier(video_feature)

        return logits
