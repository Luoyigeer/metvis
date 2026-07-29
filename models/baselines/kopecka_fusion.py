# models/baselines/kopecka_fusion.py
"""
Kopecka et al. "Estimation of atmospheric visibility by deep learning model
using multimodal dataset", Knowledge-Based Systems, 2025

核心设计 (按原论文):
  - 图像分支: EfficientNetV2M (本实现替换为 ResNet50，与统一协议一致)
  - 气象分支: MLP 处理当前时刻气象传感器数据 (非时序)
  - 融合: 特征拼接后接全连接层

原论文气象特征: 温度、湿度、压力、风速、降水量等
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class MeteoMLP(nn.Module):
    """
    气象数据编码器 - 全连接网络
    原论文: 处理当前时刻的传感器读数
    """

    def __init__(self,
                 d_in: int = config.NUM_METEO_FEATURES + 1,  # 气象特征 + 历史能见度
                 hidden_dims: list = [128, 64, 32],
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = d_in
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hd

        self.encoder = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1] if hidden_dims else d_in

    def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
        """
        seq_feat: [B, T, d_in] 气象时序数据
        原论文使用当前时刻，取最后一个时间步
        """
        current_meteo = seq_feat[:, -1, :]  # [B, d_in]
        return self.encoder(current_meteo)


class KopeckaVisibility(nn.Module):
    """
    Kopecka et al. 多模态能见度估计网络

    统一协议适配:
      - 图像编码器: ResNet50 (与其他图像基线一致)
      - 气象编码器: MLP (按原论文)
      - 融合: 拼接 + 全连接
    """

    def __init__(self,
                 num_classes: int = config.NUM_VIS_CLASSES,
                 img_feat_dim: int = 2048,
                 meteo_hidden: list = [128, 64, 32],
                 fusion_dim: int = 256,
                 pretrained: bool = True):
        super().__init__()

        # 图像编码器 (ResNet50，与统一协议一致)
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        self.img_backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.img_pool = nn.AdaptiveAvgPool2d(1)

        # 图像投影
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # 气象编码器 (原论文 MLP)
        d_in = config.NUM_METEO_FEATURES + 1
        self.meteo_encoder = MeteoMLP(d_in, meteo_hidden, dropout=0.3)

        # 气象投影
        self.meteo_proj = nn.Sequential(
            nn.Linear(self.meteo_encoder.out_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
        )

        # 输出头
        self.cls_head = nn.Linear(fusion_dim // 2, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, seq_feat: torch.Tensor, **kwargs):
        # 图像特征
        img_feat = self.img_pool(self.img_backbone(img)).flatten(1)
        img_emb = self.img_proj(img_feat)

        # 气象特征 (取当前时刻，按原论文)
        meteo_emb = self.meteo_proj(self.meteo_encoder(seq_feat))

        # 拼接融合
        fused = torch.cat([img_emb, meteo_emb], dim=-1)
        fused = self.fusion(fused)

        return {
            "feat": fused,
            "cls_logits": self.cls_head(fused),
            "vis_pred": self.reg_head(fused),
        }