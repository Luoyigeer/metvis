"""
完整能见度估计模型 (UGG + TVKD + TCAM 版)
"""
import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.image_branch import ImageBranch
from models.temporal_branch import TimesNetBranch
from models.fusion import FusionHead


def resolve_ugg_uncertainties(img_cls_out, ts_vis_reg_out, ugg_unc_mode: str):
    """根据消融模式选择传入 UGG 的不确定性信号。"""
    u_img_cls = img_cls_out["uncertainty"].squeeze(-1)
    u_img_cls = torch.nan_to_num(u_img_cls, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)

    u_ts_reg = ts_vis_reg_out["epistemic"].squeeze(-1).float()
    u_ts_reg = torch.nan_to_num(
        u_ts_reg, nan=0.0, posinf=config.UGG_EPISTEMIC_CLIP, neginf=0.0
    ).clamp(max=config.UGG_EPISTEMIC_CLIP)

    mode = ugg_unc_mode or getattr(config, "UGG_UNC_MODE_DEFAULT", "default")
    zero = torch.zeros_like(u_img_cls)

    if mode == "cls_only":
        return u_img_cls, zero
    if mode == "reg_only":
        return zero, u_ts_reg
    if mode == "swap":
        return u_ts_reg, u_img_cls
    return u_img_cls, u_ts_reg


class VisibilityModel(nn.Module):

    def __init__(self, d_in=config.NUM_METEO_FEATURES + 1, pretrained=True,
                 use_unc_gate: bool = None,
                 gate_input_mask: dict = None,
                 ugg_unc_mode: str = None,
                 use_tcam: bool = True,
                 delta_periods=None):
        super().__init__()
        self.ugg_unc_mode = ugg_unc_mode or getattr(
            config, "UGG_UNC_MODE_DEFAULT", "default"
        )
        self.img_branch = ImageBranch(
            feat_dim=config.IMAGE_FEAT_DIM,
            num_classes=config.NUM_VIS_CLASSES,
            pretrained=pretrained,
            use_tcam=use_tcam,
            delta_periods=delta_periods,
        )
        self.ts_branch = TimesNetBranch(d_in=d_in, feat_dim=config.TIME_FEAT_DIM)
        self.fusion = FusionHead(
            img_dim=config.IMAGE_FEAT_DIM,
            ts_dim=config.TIME_FEAT_DIM,
            fused_dim=config.FUSED_DIM,
            num_classes=config.NUM_VIS_CLASSES,
            use_unc_gate=use_unc_gate,
            gate_input_mask=gate_input_mask,
        )

    def forward(self, img, depth, trans,
                seq_feat, time_feat_seq, cur_time_feat, available_steps,
                delta_hours=None, force_img_only=False, alpha_scale=1.0,
                blend_alpha_scale=None):
        B = img.shape[0]
        if delta_hours is None:
            delta_hours = torch.zeros(B, device=img.device)

        img_feat, img_cls_out, img_reg_out, tcam_out = self.img_branch(
            img, depth, trans,
            cur_time_feat=cur_time_feat,
            delta_hours=delta_hours,
        )

        ts_feat, ts_vis_reg_out, ts_future_reg_out = self.ts_branch(
            seq_feat, time_feat_seq, cur_time_feat
        )
        ts_feat = torch.nan_to_num(ts_feat, nan=0.0, posinf=0.0, neginf=0.0)

        u_img, u_ts_epistemic = resolve_ugg_uncertainties(
            img_cls_out, ts_vis_reg_out, self.ugg_unc_mode
        )

        fused_feat, fused_cls_out, fused_reg_out, gate_alpha, gate_alpha_raw = (
            self.fusion(
                img_feat, ts_feat, available_steps,
                u_img=u_img,
                u_ts_epistemic=u_ts_epistemic,
                force_img_only=force_img_only,
                alpha_scale=alpha_scale,
                blend_alpha_scale=blend_alpha_scale,
            )
        )


        return {
            "img_feat": img_feat,
            "img_cls_out": img_cls_out,
            "img_reg_out": img_reg_out,
            "tcam_gamma": tcam_out["gamma"],
            "tcam_beta": tcam_out["beta"],
            "ts_feat": ts_feat,
            "ts_vis_reg_out": ts_vis_reg_out,
            "ts_future_reg_out": ts_future_reg_out,
            "fused_feat": fused_feat,
            "fused_cls_out": fused_cls_out,
            "fused_reg_out": fused_reg_out,
            "gate_alpha": gate_alpha,
            "gate_alpha_raw": gate_alpha_raw,
            "gate_u_img": u_img,
            "gate_u_ts": u_ts_epistemic,
            "delta_hours": delta_hours,
            "fused_cls_logits": fused_cls_out["cls_logits"],
            "fused_vis_pred": fused_reg_out["vis_pred"],
            "img_cls_logits": img_cls_out["cls_logits"],
            "img_vis_pred": img_reg_out["vis_pred"],
            "ts_vis_pred": ts_vis_reg_out["vis_pred"],
            "ts_future_pred": ts_future_reg_out["vis_pred"],
        }
