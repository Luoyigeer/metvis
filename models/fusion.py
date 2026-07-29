"""
融合模块 (方向二: 不确定性引导门控, UGG)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.edl import EvidentialClsHead, EvidentialRegHead


def _default_ugg_mask():
    return dict(getattr(config, "DEFAULT_UGG_MASK",
                        {"u_img": True, "u_ts": True, "avail": True, "feat_sim": True}))


class UncertaintyGuidedGate(nn.Module):
    def __init__(self, seq_len: int = config.SEQ_LEN,
                 epistemic_clip: float = config.UGG_EPISTEMIC_CLIP,
                 gate_input_mask: dict = None):
        super().__init__()
        self.seq_len = max(1, int(seq_len))
        self.epistemic_clip = epistemic_clip
        self.gate_input_mask = gate_input_mask or _default_ugg_mask()

        self.gate_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate_net[0].weight)
        nn.init.constant_(self.gate_net[0].bias[0], 0.0)
        nn.init.constant_(self.gate_net[0].bias[1], 0.0)
        nn.init.constant_(self.gate_net[0].bias[2], -2.0)
        nn.init.constant_(self.gate_net[0].bias[3], 0.0)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.constant_(self.gate_net[2].bias, -1.0)

    def forward(self,
                img_feat: torch.Tensor,
                ts_feat: torch.Tensor,
                available_steps: torch.Tensor,
                u_img: torch.Tensor,
                u_ts_epistemic: torch.Tensor) -> torch.Tensor:

        img_feat = torch.nan_to_num(img_feat, nan=0.0, posinf=0.0, neginf=0.0)
        ts_feat = torch.nan_to_num(ts_feat, nan=0.0, posinf=0.0, neginf=0.0)
        u_img = torch.nan_to_num(u_img, nan=0.0, posinf=0.0, neginf=0.0)
        u_ts_epistemic = torch.nan_to_num(u_ts_epistemic, nan=0.0, posinf=0.0, neginf=0.0)

        avail_norm = (available_steps / self.seq_len).clamp(0, 1)
        u_img_n = u_img.clamp(0, 1)
        u_ts_n = (u_ts_epistemic / self.epistemic_clip).clamp(0, 1)

        img_n = F.normalize(img_feat, dim=-1)
        ts_n = F.normalize(ts_feat, dim=-1)
        if img_n.shape[1] != ts_n.shape[1]:
            feat_sim = (img_n * F.pad(ts_n, (0, img_n.shape[1] - ts_n.shape[1]))).sum(dim=-1)
        else:
            feat_sim = (img_n * ts_n).sum(dim=-1)
        feat_sim_n = (feat_sim + 1.0) / 2.0

        mask = self.gate_input_mask
        if not mask.get("u_img", True):
            u_img_n = torch.zeros_like(u_img_n)
        if not mask.get("u_ts", True):
            u_ts_n = torch.zeros_like(u_ts_n)
        if not mask.get("feat_sim", True):
            feat_sim_n = torch.full_like(feat_sim_n, 0.5)

        avail_for_gate = avail_norm
        avail_mult = avail_norm.unsqueeze(-1)
        if not mask.get("avail", True):
            avail_for_gate = torch.ones_like(avail_norm)
            avail_mult = torch.ones_like(avail_norm).unsqueeze(-1)

        gate_in = torch.stack([u_img_n, u_ts_n, avail_for_gate, feat_sim_n], dim=-1)
        alpha = self.gate_net(gate_in)
        alpha = alpha * avail_mult
        return alpha.clamp(config.INFER_ALPHA_MIN, config.INFER_ALPHA_MAX)


class LegacyGate(nn.Module):
    def __init__(self, img_dim, ts_dim, seq_len=config.SEQ_LEN):
        super().__init__()
        self.seq_len = max(1, int(seq_len))
        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + ts_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_feat, ts_feat, available_steps,
                u_img=None, u_ts_epistemic=None):
        avail_norm = (available_steps / self.seq_len).unsqueeze(-1)
        gate_in = torch.cat([img_feat, ts_feat, avail_norm], dim=-1)
        alpha = self.gate_net(gate_in) * avail_norm
        return alpha.clamp(config.INFER_ALPHA_MIN, config.INFER_ALPHA_MAX)


class FusionHead(nn.Module):
    def __init__(self, img_dim=config.IMAGE_FEAT_DIM,
                 ts_dim=config.TIME_FEAT_DIM,
                 fused_dim=config.FUSED_DIM,
                 num_classes=config.NUM_VIS_CLASSES,
                 use_unc_gate: bool = None,
                 gate_input_mask: dict = None):
        super().__init__()
        use_unc_gate = config.USE_UNC_GATE if use_unc_gate is None else use_unc_gate
        self.use_unc_gate = use_unc_gate

        if use_unc_gate:
            self.gate = UncertaintyGuidedGate(gate_input_mask=gate_input_mask)
        else:
            self.gate = LegacyGate(img_dim, ts_dim)

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU(),
        )
        self.ts_proj = nn.Sequential(
            nn.Linear(ts_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU(),
        )
        ts_lin = self.ts_proj[0]
        if isinstance(ts_lin, nn.Linear):
            gain = float(getattr(config, "STAGE3_TS_PROJ_INIT_GAIN", 0.01))
            nn.init.xavier_uniform_(ts_lin.weight, gain=gain)
            nn.init.zeros_(ts_lin.bias)
        self.post = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim), nn.GELU(), nn.Dropout(0.1),
        )
        self.reg_proj = nn.Linear(fused_dim, img_dim)
        self.cls_head = EvidentialClsHead(fused_dim, num_classes)
        self.reg_head = EvidentialRegHead(img_dim, hidden_dim=64)

    def forward(self, img_feat, ts_feat, available_steps,
                u_img=None, u_ts_epistemic=None, force_img_only=False,
                alpha_scale=1.0, blend_alpha_scale=None):
        if u_img is None:
            u_img = torch.zeros(img_feat.shape[0], device=img_feat.device)
        if u_ts_epistemic is None:
            u_ts_epistemic = torch.zeros(img_feat.shape[0], device=img_feat.device)

        if self.use_unc_gate:
            alpha_raw = self.gate(
                img_feat, ts_feat, available_steps,
                u_img.squeeze(-1) if u_img.dim() > 1 else u_img,
                u_ts_epistemic.squeeze(-1) if u_ts_epistemic.dim() > 1 else u_ts_epistemic,
            )
        else:
            alpha_raw = self.gate(img_feat, ts_feat, available_steps)
        gate_scale = float(alpha_scale)
        blend_scale = float(blend_alpha_scale if blend_alpha_scale is not None else alpha_scale)
        alpha = alpha_raw * gate_scale
        alpha_blend = alpha_raw * blend_scale

        img_p = self.img_proj(img_feat)
        ts_p = self.ts_proj(ts_feat)
        fused = self.post((1 - alpha_blend) * img_p + alpha_blend * ts_p)
        cls_out = self.cls_head(fused)
        reg_feat = self.reg_proj(fused)
        reg_out = self.reg_head(reg_feat)
        return fused, cls_out, reg_out, alpha, alpha_raw
