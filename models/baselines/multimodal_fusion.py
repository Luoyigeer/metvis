"""
Naive multimodal baselines (image + meteorology).
- Early fusion: concat image feat with LSTM-encoded meteo sequence.
- Late fusion: project image feat and LSTM-encoded meteo then sum.
"""
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as tv_models

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class _ImageBackbone(nn.Module):
    """ResNet50 backbone that returns pooled features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = 2048

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.features(img).flatten(1)


class _MeteoLSTMEncoder(nn.Module):
    """BiLSTM encoder over meteo sequences with attention pooling."""

    def __init__(self,
                 d_in: int = config.NUM_METEO_FEATURES + 1,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(d_in, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_q = nn.Linear(hidden_dim * 2, 1)
        self.out_dim = hidden_dim * 2

    def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(seq_feat)
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn_q(lstm_out), dim=1)
        context = (attn_w * lstm_out).sum(dim=1)
        return context


class EarlyFusionBaseline(nn.Module):
    """Early fusion via concatenation of image features and meteo LSTM embedding."""

    def __init__(self,
                 num_classes: int = config.NUM_VIS_CLASSES,
                 pretrained: bool = True,
                 hidden_dim: int = 256,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2):
        super().__init__()
        self.backbone = _ImageBackbone(pretrained=pretrained)
        self.meteo_encoder = _MeteoLSTMEncoder(
            d_in=config.NUM_METEO_FEATURES + 1,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
        )
        self.meteo_dim = self.meteo_encoder.out_dim
        self.fuse = nn.Sequential(
            nn.Linear(self.backbone.feat_dim + self.meteo_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, img: torch.Tensor, seq_feat: torch.Tensor, **kwargs):
        img_feat = self.backbone(img)
        meteo_feat = self.meteo_encoder(seq_feat)
        fused = self.fuse(torch.cat([img_feat, meteo_feat], dim=-1))
        return {
            "feat": fused,
            "cls_logits": self.cls_head(fused),
            "vis_pred": self.reg_head(fused),
        }


class LateFusionBaseline(nn.Module):
    """Late fusion by projecting image and meteo LSTM embedding then summing."""

    def __init__(self,
                 num_classes: int = config.NUM_VIS_CLASSES,
                 pretrained: bool = True,
                 fusion_dim: int = 256,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2):
        super().__init__()
        self.backbone = _ImageBackbone(pretrained=pretrained)
        self.meteo_encoder = _MeteoLSTMEncoder(
            d_in=config.NUM_METEO_FEATURES + 1,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
        )
        self.meteo_dim = self.meteo_encoder.out_dim
        self.img_proj = nn.Sequential(
            nn.Linear(self.backbone.feat_dim, fusion_dim),
            nn.ReLU(),
        )
        self.meteo_proj = nn.Sequential(
            nn.Linear(self.meteo_dim, fusion_dim),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(fusion_dim, num_classes)
        self.reg_head = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Sigmoid())

    def forward(self, img: torch.Tensor, seq_feat: torch.Tensor, **kwargs):
        img_feat = self.img_proj(self.backbone(img))
        meteo_feat = self.meteo_proj(self.meteo_encoder(seq_feat))
        fused = img_feat + meteo_feat
        return {
            "feat": fused,
            "cls_logits": self.cls_head(fused),
            "vis_pred": self.reg_head(fused),
        }
