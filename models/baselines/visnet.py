"""
VisNet 复现
参考: Meteorological Visibility Estimation using cameras (能见度专用CNN)
核心设计:
  - 多尺度特征提取 (类似FPN)
  - 雾感知注意力模块 (Haze-Aware Attention)
  - 序数分类头 (Ordinal Classification) + 回归头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class HazeAwareAttention(nn.Module):
    """
    雾感知通道注意力
    通过暗通道响应加权特征图，增强雾气相关区域
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = self.avg_pool(x).view(B, C)
        mx  = self.max_pool(x).view(B, C)
        att = self.fc(torch.cat([avg, mx], dim=1)).view(B, C, 1, 1)
        return x * att


class MultiScaleFPN(nn.Module):
    """
    轻量FPN: 融合ResNet多层特征
    层级: layer2(512), layer3(1024), layer4(2048) -> 统一投影到256
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.lat2 = nn.Conv2d(512,  out_channels, 1)
        self.lat3 = nn.Conv2d(1024, out_channels, 1)
        self.lat4 = nn.Conv2d(2048, out_channels, 1)

        self.merge3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.merge2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_channels * 3   # 3层拼接

    def forward(self, c2, c3, c4):
        p4 = self.lat4(c4)
        p3 = self.merge3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest'))
        p2 = self.merge2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest'))

        f4 = self.pool(p4).flatten(1)
        f3 = self.pool(p3).flatten(1)
        f2 = self.pool(p2).flatten(1)
        return torch.cat([f4, f3, f2], dim=1)   # [B, 256*3]


class OrdinalHead(nn.Module):
    """
    序数分类头: 输出K-1个二元决策 P(y > k)，累积得概率分布
    参考: Li et al., Ordinal Regression with Multiple Output CNN (CVPR 2015)
    """
    def __init__(self, in_dim: int, num_classes: int = config.NUM_VIS_CLASSES):
        super().__init__()
        self.K = num_classes
        # K-1 个二分类器共享前置层
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.ordinal_fc = nn.Linear(128, num_classes - 1)

    def forward(self, feat: torch.Tensor):
        """
        返回:
            logits_ord: [B, K-1]  每个阈值的logit P(y > k)
            cls_logits: [B, K]    转换为K类分布 (用于CE损失)
        """
        h = self.shared(feat)
        logits_ord = self.ordinal_fc(h)  # [B, K-1]

        # 转为累积概率: p_k = sigmoid(logit_k)
        prob_exceed = torch.sigmoid(logits_ord)   # [B, K-1] P(y > k)

        # 类别概率: P(y=0)=1-P(y>0), P(y=k)=P(y>k-1)-P(y>k), P(y=K-1)=P(y>K-2)
        B = feat.shape[0]
        ones  = torch.ones(B, 1, device=feat.device)
        zeros = torch.zeros(B, 1, device=feat.device)
        # 拼接边界: [1, p0, p1, ..., p_{K-2}, 0]
        cum = torch.cat([ones, prob_exceed, zeros], dim=1)  # [B, K+1]
        cls_probs = cum[:, :-1] - cum[:, 1:]                # [B, K]
        cls_probs = cls_probs.clamp(1e-6, 1.0)

        # log概率作为logits (与CrossEntropyLoss兼容)
        cls_logits = torch.log(cls_probs)
        return logits_ord, cls_logits


class VisNet(nn.Module):
    """
    VisNet: 能见度专用多尺度+序数回归网络
    backbone: ResNet50 (pretrained)
    """
    def __init__(self, num_classes: int = config.NUM_VIS_CLASSES,
                 pretrained: bool = True, fpn_channels: int = 256):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        # 提取中间层
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 ch
        self.layer2 = backbone.layer2   # 512 ch
        self.layer3 = backbone.layer3   # 1024 ch
        self.layer4 = backbone.layer4   # 2048 ch

        # 雾感知注意力 (对layer2/3/4各加一个)
        self.att2 = HazeAwareAttention(512)
        self.att3 = HazeAwareAttention(1024)
        self.att4 = HazeAwareAttention(2048)

        # 多尺度FPN
        self.fpn = MultiScaleFPN(fpn_channels)
        feat_dim = self.fpn.out_dim   # 256*3 = 768

        # 特征投影
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.feat_dim = 256

        # 序数分类头
        self.ordinal_head = OrdinalHead(256, num_classes)

        # 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, img, **kwargs):
        x = self.layer0(img)
        x = self.layer1(x)
        c2 = self.att2(self.layer2(x))
        c3 = self.att3(self.layer3(c2))
        c4 = self.att4(self.layer4(c3))

        fpn_feat = self.fpn(c2, c3, c4)
        feat = self.proj(fpn_feat)

        logits_ord, cls_logits = self.ordinal_head(feat)
        vis_pred = self.reg_head(feat)

        return {
            "feat":        feat,
            "cls_logits":  cls_logits,    # [B, K] log概率
            "logits_ord":  logits_ord,    # [B, K-1] 序数logit (用于BCE损失)
            "vis_pred":    vis_pred,
        }
