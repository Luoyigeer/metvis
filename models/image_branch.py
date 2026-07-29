"""
图像分支: CNN (ResNet50) + ViT + TCAM 时间条件自适应
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.edl import EvidentialClsHead, EvidentialRegHead
from models.tcam import TCAM, make_delta_hours, get_tcam_cond_dim


class CNNBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class ViTBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = tv_models.vit_b_16(weights=weights)
        self.patch_embed = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        x = self.encoder(torch.cat([cls, x], dim=1))
        return x[:, 0]


class AuxImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class ImageBranch(nn.Module):
    def __init__(self, feat_dim=config.IMAGE_FEAT_DIM,
                 num_classes=config.NUM_VIS_CLASSES, pretrained=True,
                 use_tcam: bool = True,
                 delta_periods=None):
        super().__init__()
        self.cnn = CNNBranch(pretrained)
        self.vit = ViTBranch(pretrained)
        self.depth_enc = AuxImageEncoder(128)
        self.trans_enc = AuxImageEncoder(128)

        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768 + 128 + 128, 1024),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, feat_dim),
            nn.LayerNorm(feat_dim), nn.GELU(),
        )

        self.tcam = TCAM(
            feat_dim=feat_dim,
            cond_dim=get_tcam_cond_dim(delta_periods),
            hidden=config.TCAM_HIDDEN_DIM,
            num_layers=config.TCAM_NUM_LAYERS,
            delta_periods=delta_periods,
            use_tcam=use_tcam,
        )

        self.cls_head = EvidentialClsHead(feat_dim, num_classes)
        self.reg_head = EvidentialRegHead(feat_dim, hidden_dim=64)

    def forward(self, img, depth, trans,
                cur_time_feat=None, delta_hours=None):
        cat = torch.cat([
            self.cnn(img), self.vit(img),
            self.depth_enc(depth), self.trans_enc(trans)
        ], dim=1)
        cat = torch.nan_to_num(cat, nan=0.0, posinf=0.0, neginf=0.0)
        feat = torch.nan_to_num(self.fusion(cat), nan=0.0, posinf=0.0, neginf=0.0)

        if cur_time_feat is None:
            cur_time_feat = torch.zeros(feat.shape[0], 10, device=feat.device)
        if delta_hours is None:
            delta_hours = torch.zeros(feat.shape[0], device=feat.device)

        tcam_out = self.tcam(feat, cur_time_feat, delta_hours)
        feat = torch.nan_to_num(tcam_out["feat"], nan=0.0, posinf=0.0, neginf=0.0)
        return feat, self.cls_head(feat), self.reg_head(feat), tcam_out
