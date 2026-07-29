"""CLIP 能见度基线"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import config


def _load_clip_visual(pretrained: bool = True):
    """
    尝试加载openai/clip, 失败则回退到torchvision ViT-B/16
    返回 (visual_encoder, feat_dim, preprocess_or_None)
    """
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        visual = model.visual
        feat_dim = 512
        print("[CLIP] 使用 openai/clip ViT-B/32")
        return visual, feat_dim, preprocess
    except ImportError:
        import torchvision.models as tv_models
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = tv_models.vit_b_16(weights=weights)
        # 包装成统一接口
        class ViTWrapper(nn.Module):
            def __init__(self, vit):
                super().__init__()
                self.patch_embed = vit.conv_proj
                self.class_token = vit.class_token
                self.encoder = vit.encoder
            def forward(self, x):
                B = x.shape[0]
                x = self.patch_embed(x).flatten(2).transpose(1,2)
                cls = self.class_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.encoder(x)
                return x[:, 0]   # CLS token
        print("[CLIP] openai/clip未安装, 回退到torchvision ViT-B/16")
        return ViTWrapper(vit), 768, None


# ----------------------------------------------------------------
# CLIP + 线性头
# ----------------------------------------------------------------
class CLIPVis(nn.Module):
    """
    CLIP视觉编码器 + 线性分类/回归头
    视觉编码器默认冻结, 只训练头部
    """
    def __init__(self, num_classes: int = config.NUM_VIS_CLASSES,
                 pretrained: bool = True, freeze_visual: bool = True):
        super().__init__()
        self.visual, feat_dim, self.preprocess = _load_clip_visual(pretrained)

        if freeze_visual:
            for p in self.visual.parameters():
                p.requires_grad = False

        self.feat_dim = feat_dim
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(256, num_classes)
        self.reg_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, img, **kwargs):
        with torch.set_grad_enabled(not all(
            not p.requires_grad for p in self.visual.parameters()
        )):
            feat = self.visual(img)
            if hasattr(feat, 'float'):
                feat = feat.float()

        h = self.adapter(feat)
        return {
            "feat":       feat,
            "cls_logits": self.cls_head(h),
            "vis_pred":   self.reg_head(h),
        }


# ----------------------------------------------------------------
# OrdinalCLIP
# ----------------------------------------------------------------
class RankPromptEmbedding(nn.Module):
    """
    可学习的排名感知文本提示
    每个能见度等级对应一个可学习embedding, 施加单调约束(soft)
    参考: OrdinalCLIP §3.2 Rank Prompt Learning
    """
    def __init__(self, num_classes: int, embed_dim: int, context_len: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # context_len个可学习token × num_classes个等级
        self.ctx = nn.Parameter(
            torch.randn(num_classes, context_len, embed_dim) * 0.02
        )
        # 等级锚点: 线性初始化使初始顺序单调
        anchor = torch.linspace(-1, 1, num_classes).unsqueeze(-1)  # [K,1]
        self.register_buffer("anchor", anchor)

        self.proj = nn.Sequential(
            nn.Linear(context_len * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self):
        """
        返回 [K, embed_dim] 每个等级的提示表示
        施加软单调约束: 通过cumsum使表示在某维度单调递增
        """
        B, L, D = self.ctx.shape
        flat = self.ctx.view(B, L * D)
        prompts = self.proj(flat)           # [K, D]

        # 软单调正则: 对最后一维的均值排序约束由损失函数处理(见OrdinalCLIPLoss)
        return prompts                      # [K, D]


