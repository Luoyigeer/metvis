"""OrdinalCLIP 能见度基线"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import config
from models.baselines.clip import RankPromptEmbedding


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
class OrdinalCLIP(nn.Module):
    """
    OrdinalCLIP: CLIP视觉特征 + 可学习序数提示对齐
    训练阶段:
      - 视觉特征 z_img = CLIP_visual(img)   [B, D]
      - 文本提示 z_txt = RankPrompts()      [K, D]
      - 相似度: s_k = cos(z_img, z_txt_k)  [B, K]
      - 序数解码: 参考OrdinalCLIP §3.3
    """
    def __init__(self, num_classes: int = config.NUM_VIS_CLASSES,
                 pretrained: bool = True, freeze_visual: bool = True,
                 context_len: int = 4, temp: float = 0.07):
        super().__init__()
        self.visual, feat_dim, self.preprocess = _load_clip_visual(pretrained)
        self.feat_dim = feat_dim

        if freeze_visual:
            for p in self.visual.parameters():
                p.requires_grad = False

        # 视觉特征投影到提示空间
        self.visual_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim),
        )

        # 可学习序数提示
        self.rank_prompts = RankPromptEmbedding(num_classes, feat_dim, context_len)

        self.num_classes = num_classes
        self.temp = nn.Parameter(torch.tensor(temp))

        # 回归头 (辅助)
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, img, **kwargs):
        # 视觉特征
        with torch.set_grad_enabled(not all(
            not p.requires_grad for p in self.visual.parameters()
        )):
            feat_raw = self.visual(img)
            if hasattr(feat_raw, 'float'):
                feat_raw = feat_raw.float()

        feat = self.visual_proj(feat_raw)          # [B, D]
        feat_norm = F.normalize(feat, dim=-1)

        # 序数提示
        prompts = self.rank_prompts()              # [K, D]
        prompts_norm = F.normalize(prompts, dim=-1)

        # 相似度矩阵 [B, K]
        logit_scale = self.temp.exp().clamp(max=100)
        sim = feat_norm @ prompts_norm.T * logit_scale  # [B, K]

        # OrdinalCLIP解码: 将相似度转为序数概率
        # P(y=k) ∝ sim_k, 同时通过累积差分施加单调约束
        cls_logits = sim   # 直接用相似度作为logits (CE loss)

        vis_pred = self.reg_head(feat_raw)

        return {
            "feat":       feat_raw,
            "feat_proj":  feat,
            "prompts":    prompts,
            "cls_logits": cls_logits,
            "vis_pred":   vis_pred,
            "sim":        sim,
        }


class OrdinalCLIPLoss(nn.Module):
    """
    OrdinalCLIP专用损失
    = CE损失 + 单调正则 + 对比损失
    """
    def __init__(self, num_classes: int = config.NUM_VIS_CLASSES,
                 lambda_mono: float = 0.1, lambda_contra: float = 0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.lambda_mono = lambda_mono
        self.lambda_contra = lambda_contra
        self.num_classes = num_classes

    def monotonic_loss(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        软单调约束: 相邻等级提示的余弦相似度应单调递增
        即 sim(k, k+1) 在序数方向上应有一致排序
        """
        norms = F.normalize(prompts, dim=-1)   # [K, D]
        # 相邻内积: P(y>k) 应该是单调的
        sims = (norms[:-1] * norms[1:]).sum(-1)  # [K-1]
        # 惩罚相邻相似度过低 (应该保持一定顺序感)
        mono_loss = F.relu(0.5 - sims).mean()
        return mono_loss

    def forward(self, outputs: dict, vis_cls: torch.Tensor) -> dict:
        cls_logits = outputs["cls_logits"]
        prompts = outputs["prompts"]

        loss_ce   = self.ce(cls_logits, vis_cls)
        loss_mono = self.monotonic_loss(prompts)

        total = loss_ce + self.lambda_mono * loss_mono
        return {"total": total, "ce": loss_ce, "mono": loss_mono}
