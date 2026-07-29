"""DenseNet-121 能见度基线"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
import sys, os
import config

class DenseNetVis(nn.Module):
    """
    DenseNet121 + 能见度分类/回归头
    参考: Huang et al., Densely Connected Convolutional Networks, CVPR 2017
    """
    def __init__(self, num_classes: int = config.NUM_VIS_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.densenet121(weights=weights)
        self.features = backbone.features          # -> [B, 1024, 7, 7]
        self.feat_dim = 1024
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cls_head = nn.Sequential(
            nn.Linear(self.feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(self.feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, img, **kwargs):
        f = self.pool(torch.relu(self.features(img))).flatten(1)  # [B, 1024]
        return {
            "feat":       f,
            "cls_logits": self.cls_head(f),
            "vis_pred":   self.reg_head(f),
        }


class ResNetVis(nn.Module):
    """
    ResNet50 + 能见度分类/回归头
    参考: He et al., Deep Residual Learning for Image Recognition, CVPR 2016
    使用 ImageNet 预训练权重，去掉最后全连接层，接能见度专用头
    """
    def __init__(self, num_classes: int = config.NUM_VIS_CLASSES,
                 pretrained: bool = True,
                 variant: str = "resnet50"):
        super().__init__()
        if variant == "resnet50":
            weights  = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet50(weights=weights)
            feat_dim = 2048
        elif variant == "resnet101":
            weights  = tv_models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet101(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"不支持的 ResNet 变体: {variant}")

        # 去掉 avgpool 和 fc，保留特征提取部分
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, img, **kwargs):
        f = self.pool(self.backbone(img)).flatten(1)   # [B, feat_dim]
        return {
            "feat":       f,
            "cls_logits": self.cls_head(f),
            "vis_pred":   self.reg_head(f),
        }
