"""
ResNet 系列基线模型
支持: ResNet18 / ResNet34 / ResNet50 / ResNet101
统一接口: forward(img) -> {feat, cls_logits, vis_pred}
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

_WEIGHTS = {
    "resnet18":  (tv_models.resnet18,  tv_models.ResNet18_Weights.IMAGENET1K_V1,  512),
    "resnet34":  (tv_models.resnet34,  tv_models.ResNet34_Weights.IMAGENET1K_V1,  512),
    "resnet50":  (tv_models.resnet50,  tv_models.ResNet50_Weights.IMAGENET1K_V1,  2048),
    "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.IMAGENET1K_V1, 2048),
}


class ResNet(nn.Module):
    """
    ResNet + 能见度分类/回归头

    参考:
      He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    """
    def __init__(self,
                 variant:     str = "resnet50",
                 num_classes: int = config.NUM_VIS_CLASSES,
                 pretrained:  bool = True):
        super().__init__()
        assert variant in _WEIGHTS, f"不支持的variant: {variant}，可选: {list(_WEIGHTS)}"

        build_fn, weights_cls, feat_dim = _WEIGHTS[variant]
        weights   = weights_cls if pretrained else None
        backbone  = build_fn(weights=weights)

        # 去掉最后的全连接层，保留 GAP 之前的特征提取部分
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = feat_dim

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        # 回归头（归一化能见度 → [0,1]）
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, **kwargs):
        feat = self.features(img).flatten(1)   # [B, feat_dim]
        return {
            "feat":       feat,
            "cls_logits": self.cls_head(feat),  # [B, K]
            "vis_pred":   self.reg_head(feat),  # [B, 1]
        }


ResNetVis = ResNet  # 兼容别名
