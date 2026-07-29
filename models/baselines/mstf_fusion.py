# models/baselines/mstf_fusion.py
"""
MSTF-Net: Multi-modal Spatio-Temporal Fusion Network for Visibility Estimation
参考: Bu et al., "Road Visibility Estimation Based on Multimodal Fusion of
       Time-Series Images and Meteorological Data",
       Journal of Computing in Civil Engineering, Jan 2026

原论文核心设计:
  - 时序图像序列 (10帧) + 气象时序数据 (9类气象特征)
  - 空间特征提取模块 (Spatial Feature Extraction)
  - 时间特征提取模块 (Temporal Feature Extraction)
  - 多模态融合模块 (Multi-modal Fusion)

本实现适配统一实验协议:
  - 图像编码器: ResNet50 (替换原文可能使用的其他 backbone)
  - 时序气象编码: LSTM (按原文)
  - 融合: 特征拼接 + 全连接 (按原文)

原文链接: https://doi.org/10.1061/(ASCE)CP.1943-5487.0001234 (示例)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class SpatialFeatureExtractor(nn.Module):
    """
    空间特征提取模块
    原文: 使用 CNN 提取单帧图像的空间特征
    本实现: ResNet50 (与统一协议一致)
    """

    def __init__(self, feat_dim: int = 512, pretrained: bool = True):
        super().__init__()
        # 使用 ResNet50 作为 backbone (与统一协议一致)
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        # 去掉最后的全连接层和池化层
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 投影到固定维度
        self.proj = nn.Sequential(
            nn.Linear(2048, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.feat_dim = feat_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, 3, H, W]
        返回: [B, feat_dim]
        """
        feat = self.pool(self.features(img)).flatten(1)  # [B, 2048]
        return self.proj(feat)


class TemporalFeatureExtractor(nn.Module):
    """
    时间特征提取模块 (气象时序)
    原文: 使用 LSTM 编码气象时序数据
    """

    def __init__(self,
                 d_in: int = config.NUM_METEO_FEATURES + 1,  # 9类气象 + 能见度历史
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.feat_dim = hidden_dim

    def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
        """
        seq_feat: [B, T, d_in]
        返回: [B, feat_dim]
        """
        lstm_out, (hn, cn) = self.lstm(seq_feat)  # [B, T, hidden*2]
        # 取最后时间步的输出 (或使用注意力池化，原文使用最后时间步)
        last_out = lstm_out[:, -1, :]  # [B, hidden*2]
        return self.proj(last_out)


class MultiModalFusion(nn.Module):
    """
    多模态融合模块
    原文: 特征拼接后接全连接层进行融合
    """

    def __init__(self,
                 spatial_dim: int,
                 temporal_dim: int,
                 fusion_dim: int = 256,
                 num_classes: int = config.NUM_VIS_CLASSES):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Linear(fusion_dim // 2, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.feat_dim = fusion_dim // 2

    def forward(self, spatial_feat: torch.Tensor, temporal_feat: torch.Tensor) -> dict:
        fused = torch.cat([spatial_feat, temporal_feat], dim=-1)
        fused = self.fusion(fused)

        return {
            "feat": fused,
            "cls_logits": self.cls_head(fused),
            "vis_pred": self.reg_head(fused),
        }


class MSTFVisibility(nn.Module):
    """
    MSTF-Net: 多模态时空融合能见度估计网络

    原论文特点:
      1. 使用时序图像序列 (10帧) + 气象时序数据
      2. 空间特征提取 (CNN) + 时间特征提取 (LSTM)
      3. 多模态融合 (拼接 + 全连接)

    统一实验协议适配:
      - 图像使用单帧 (因对比基线均使用单帧)
      - 图像编码器: ResNet50
      - 气象编码器: LSTM (2层, 双向)
      - 融合: 拼接 + 全连接
    """

    def __init__(self,
                 num_classes: int = config.NUM_VIS_CLASSES,
                 spatial_dim: int = 512,
                 temporal_hidden: int = 128,
                 fusion_dim: int = 256,
                 pretrained: bool = True):
        super().__init__()

        # 空间特征提取 (图像)
        self.spatial_extractor = SpatialFeatureExtractor(spatial_dim, pretrained)

        # 时间特征提取 (气象时序)
        d_in = config.NUM_METEO_FEATURES + 1
        self.temporal_extractor = TemporalFeatureExtractor(
            d_in=d_in,
            hidden_dim=temporal_hidden,
            num_layers=2,
            dropout=0.3,
        )

        # 多模态融合
        self.fusion = MultiModalFusion(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_hidden,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
        )

    def forward(self, img: torch.Tensor, seq_feat: torch.Tensor, **kwargs):
        """
        img: [B, 3, H, W] 单帧图像 (统一协议)
        seq_feat: [B, T, d_in] 气象时序数据
        """
        # 空间特征
        spatial_feat = self.spatial_extractor(img)  # [B, spatial_dim]

        # 时间特征 (气象时序)
        temporal_feat = self.temporal_extractor(seq_feat)  # [B, temporal_hidden]

        # 融合输出
        return self.fusion(spatial_feat, temporal_feat)