"""
消融实验共享基础设施：三阶段训练、评估、结果保存。
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset_factory import build_all_loaders, build_loader
from data.image_datasets import TimeAligner
from data.noaa_dataset import NOAADataProcessor
from data.sampler import compute_class_weights, get_class_counts
from evaluate import compute_metrics
from losses import NTXentLoss, TemporalVisualKDLoss, UGGRegularizer, TCAMLoss
from models.edl import EDLClassificationLoss, EDLRegressionLoss
from models.visibility_model import VisibilityModel
from utils import load_checkpoint, to_device_batch

STAGE2_CKPT = config.STAGE2_CKPT
FUSION_TRANSITION_EPOCHS = 3
FUSION_DISTILL_EPOCHS = 10  # 默认模式 freeze 结束后仍用 img→fuse 蒸馏若干 epoch


def _stage3_train_force_img_only(epoch: int, freeze_epochs: int) -> bool:
    """默认模式仅 freeze 期强制 img-only eval bypass；transition 起走 alpha schedule。"""
    if epoch <= 0:
        return True
    return epoch <= freeze_epochs


def _stage3_transition_alpha(epoch: int, freeze_epochs: int, trans_hi: float) -> float:
    """freeze 结束后 transition 期线性升 alpha（0.05 → trans_hi）。"""
    freeze_end = freeze_epochs
    transition_end = freeze_epochs + FUSION_TRANSITION_EPOCHS
    if epoch <= freeze_end:
        return 0.0
    if epoch > transition_end:
        return trans_hi
    first_trans = freeze_end + 1
    denom = max(FUSION_TRANSITION_EPOCHS - 1, 1)
    t = (epoch - first_trans) / denom
    t = min(max(t, 0.0), 1.0)
    return 0.05 + t * (trans_hi - 0.05)


def _stage3_alpha_schedule(epoch: int, freeze_epochs: int,
                           dual_freeze: bool = False) -> float:
    """dual_freeze / 默认模式：freeze→transition 渐开→UGG 渐升。"""
    if epoch <= 0:
        return 0.0
    freeze_end = freeze_epochs
    transition_end = freeze_epochs + FUSION_TRANSITION_EPOCHS
    trans_hi = float(
        getattr(config, "STAGE3_DEFAULT_TRANSITION_ALPHA", None)
        or getattr(config, "STAGE3_DUAL_TRANSITION_ALPHA", 0.15)
    )
    if epoch <= freeze_end:
        return 0.0
    if epoch <= transition_end:
        return _stage3_transition_alpha(epoch, freeze_epochs, trans_hi)
    if dual_freeze:
        ramp_epochs = max(1, int(getattr(config, "STAGE3_DUAL_ALPHA_RAMP_EPOCHS", 10)))
        ugg_lo = 0.2
        first_ugg = transition_end + 1
        denom = max(ramp_epochs - 1, 1)
        t = (epoch - first_ugg) / denom
        t = min(max(t, 0.0), 1.0)
        return ugg_lo + t * (1.0 - ugg_lo)
    ramp_epochs = max(1, int(getattr(config, "STAGE3_UGG_RAMP_EPOCHS", 5)))
    first_ugg = transition_end + 1
    denom = max(ramp_epochs - 1, 1)
    t = (epoch - first_ugg) / denom
    t = min(max(t, 0.0), 1.0)
    return trans_hi + t * (1.0 - trans_hi)


def _stage3_blend_alpha_scale(alpha_scale: float) -> float:
    """与 train_epoch 一致：freeze 期用 min_blend 保持 gate 在计算图。"""
    min_blend = float(getattr(config, "STAGE3_TRAIN_MIN_ALPHA", 0.02))
    return max(float(alpha_scale), min_blend)


def _stage3_eval_mode(epoch: int, freeze_epochs: int, head=None,
                      for_test: bool = False, force_img_only=None,
                      dual_freeze: bool = False, alpha_scale_override=None) -> dict:
    """
    统一 Stage3 验证/测试前向。
    eval 与 train 共用 blend_alpha_scale，不再在 eval 时整路替换为图像头。
    """
    if for_test:
        ramp = 1.0
        return {
            "force_img_only": False, "use_fused": True,
            "alpha_scale": ramp, "blend_alpha_scale": ramp,
            "dual_freeze": dual_freeze,
        }
    if head == "img":
        return {
            "force_img_only": True, "use_fused": False,
            "alpha_scale": 0.0, "blend_alpha_scale": 0.0,
            "dual_freeze": dual_freeze,
        }
    if alpha_scale_override is not None:
        ramp = float(alpha_scale_override)
        return {
            "force_img_only": False, "use_fused": True,
            "alpha_scale": ramp,
            "blend_alpha_scale": _stage3_blend_alpha_scale(ramp),
            "dual_freeze": dual_freeze,
        }
    if dual_freeze:
        ramp = _stage3_alpha_schedule(epoch, freeze_epochs, dual_freeze=True)
        return {
            "force_img_only": False, "use_fused": True,
            "alpha_scale": ramp,
            "blend_alpha_scale": _stage3_blend_alpha_scale(ramp),
            "dual_freeze": True,
        }
    if head == "fused" and force_img_only is not None:
        foi = bool(force_img_only)
        ramp = 0.0 if foi else _stage3_alpha_schedule(epoch, freeze_epochs, dual_freeze=False)
        return {
            "force_img_only": foi, "use_fused": True,
            "alpha_scale": ramp if not foi else 0.0,
            "blend_alpha_scale": 0.0 if foi else _stage3_blend_alpha_scale(ramp),
            "dual_freeze": False,
        }
    ramp = _stage3_alpha_schedule(epoch, freeze_epochs, dual_freeze=False)
    foi = ramp <= 0.0 and _stage3_train_force_img_only(epoch, freeze_epochs)
    return {
        "force_img_only": foi, "use_fused": True,
        "alpha_scale": ramp,
        "blend_alpha_scale": 0.0 if foi else _stage3_blend_alpha_scale(ramp),
        "dual_freeze": False,
    }


def _stage3_use_fuse_distill(epoch: int, freeze: bool, fusion_transition: bool,
                             dual_freeze: bool, freeze_fusion_epochs: int) -> bool:
    """dual_freeze：仅首个 freeze epoch；默认：freeze+transition+post 全程蒸馏。"""
    if dual_freeze:
        max_ep = int(getattr(config, "STAGE3_DUAL_DISTILL_MAX_EPOCH", 1))
        return freeze and epoch <= max_ep
    distill_end = (
        freeze_fusion_epochs + FUSION_TRANSITION_EPOCHS + FUSION_DISTILL_EPOCHS
    )
    return epoch <= distill_end


def _refresh_edl_cls_out(cls_out: dict, num_classes: int = None) -> None:
    """alpha 修复后同步重算 prob / uncertainty / cls_logits。"""
    K = num_classes or config.NUM_VIS_CLASSES
    alpha = cls_out["alpha"]
    S = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    prob = alpha / S
    prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = (float(K) / S).clamp(max=1.0)
    uncertainty = torch.nan_to_num(uncertainty, nan=1.0, posinf=1.0, neginf=1.0)
    cls_out["S"] = S
    cls_out["prob"] = prob
    cls_out["uncertainty"] = uncertainty
    cls_out["cls_logits"] = torch.log(prob + 1e-8)


def _ensure_loss_grad(total: torch.Tensor, params) -> torch.Tensor:
    """当 loss 标量无梯度时，挂接可训练参数零乘子以保持计算图。"""
    total = total.float()
    if total.requires_grad:
        return total
    trainable = [p for p in params if p.requires_grad]
    if not trainable:
        return total
    anchor = trainable[0].reshape(-1)[0]
    return total + anchor * 0.0


def _apply_stage3_total_floor(total: torch.Tensor, fuse_sup: torch.Tensor,
                              dual_freeze: bool) -> torch.Tensor:
    """fuse_sup 近零或 dual_freeze 时加独立下限，避免 tot 打印为 0。"""
    floor = float(getattr(config, "STAGE3_MIN_TOTAL_FLOOR", 1e-3))
    fuse_val = float(fuse_sup.detach().item()) if torch.is_tensor(fuse_sup) else float(fuse_sup)
    if dual_freeze or fuse_val < 1e-8:
        total = total + total.new_tensor(floor)
    return total


def _sanitize_loss_scalar(loss: torch.Tensor, max_val: float = None) -> torch.Tensor:
    """NaN/Inf 消毒，避免整 epoch batch 被 skip。"""
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    if max_val is not None:
        loss = loss.clamp(max=float(max_val))
    return loss


def _init_layernorm_neutral(module: nn.Module) -> None:
    """LayerNorm 初始为恒等缩放，减轻 warm-start 后特征漂移。"""
    if isinstance(module, nn.LayerNorm):
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.zero_()


def _resolve_stage2_ckpt(dataset_name: str, suffix: str = None) -> str:
    path = config.stage2_ckpt_path(dataset_name, suffix=suffix)
    if os.path.exists(path):
        return path
    if suffix is None and os.path.exists(STAGE2_CKPT):
        return STAGE2_CKPT
    return path


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_stage3_freeze_state(model, epoch: int, freeze_fusion_epochs: int,
                            dual_freeze: bool = False):
    """Freeze image branch always; optionally freeze ts; train fusion heads early."""
    freeze_gate = epoch <= freeze_fusion_epochs
    if hasattr(model, "img_branch"):
        for p in model.img_branch.parameters():
            p.requires_grad = False
    if dual_freeze and hasattr(model, "ts_branch"):
        for p in model.ts_branch.parameters():
            p.requires_grad = False
    fusion = getattr(model, "fusion", None)
    if fusion is not None:
        for name, p in fusion.named_parameters():
            p.requires_grad = not (freeze_gate and name.startswith("gate."))


class TimeReconHead(nn.Module):
    def __init__(self, in_dim=config.TIME_FEAT_DIM, time_emb_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(),
            nn.Linear(32, time_emb_dim), nn.Tanh(),
        )

    def forward(self, ts_feat):
        return self.net(ts_feat)


def _stage3_ts_cls_teacher_logits(outputs: dict, device) -> torch.Tensor:
    """时序分支无分类头时，由回归 gamma 推导软分类 logits 作为 KD teacher。"""
    ts_out = outputs.get("ts_vis_reg_out")
    if ts_out is None:
        return None
    if "cls_logits" in ts_out:
        return ts_out["cls_logits"].detach()
    gamma = ts_out["gamma"].squeeze(-1).detach()
    if gamma.numel() == 0:
        return None
    max_vis = float(getattr(config, "MAX_VIS", 50000.0))
    vis_m = gamma.clamp(0.0, 1.0) * max_vis
    bins = getattr(config, "VIS_BINS", [0, 200, 500, 1000, 2000, 5000, 50000])
    centers = [(bins[i] + bins[i + 1]) / 2.0 for i in range(config.NUM_VIS_CLASSES)]
    centers_t = torch.tensor(centers, device=device, dtype=vis_m.dtype)
    dist = (vis_m.unsqueeze(-1) - centers_t.unsqueeze(0)).abs()
    soft = torch.softmax(-dist / 500.0, dim=-1)
    return torch.log(soft + 1e-8)


def _stage3_vis_tgt_norm(vis_val: torch.Tensor) -> torch.Tensor:
    """统一回归目标到 [0,1]；仅当真米制（>1）时才除 MAX_VIS。"""
    max_vis = float(getattr(config, "MAX_VIS", 50000.0))
    valid = vis_val >= 0
    if valid.any() and vis_val[valid].max() > 1.0:
        return (vis_val / max_vis).clamp(0.0, 1.0)
    return vis_val.clamp(0.0, 1.0)


def _stage3_vis_val_to_meter_np(vis_val_np, max_vis: float):
    """eval 用：与训练 _stage3_vis_tgt_norm 互逆的米制转换。"""
    if getattr(config, "USE_METER_LABELS", False) and float(vis_val_np.max()) > 1.0:
        return vis_val_np
    return vis_val_np * max_vis


def _stage3_reg_sample_weights(vis_cls: torch.Tensor, mask: torch.Tensor,
                               device) -> torch.Tensor:
    """对 0–3 类回归样本返回更高 per-sample 权重（均值归一化）。"""
    n = int(mask.sum().item())
    if n == 0:
        return torch.ones(0, device=device)
    w = torch.ones(n, device=device)
    low_w = float(getattr(config, "STAGE3_LOW_VIS_REG_WEIGHT", 1.0))
    low_max = int(getattr(config, "STAGE3_LOW_VIS_MAX_CLS", 3))
    cls_sub = vis_cls[mask]
    w[cls_sub <= low_max] = low_w
    return w / w.mean().clamp(min=1e-8)


def _stage3_weighted_edl_component(edl_reg, ro, mask, vis_cls, vis_tgt, key="reg"):
    """低/高能见度子集分别算 EDL 分量（reg 恒正 / total 含 NIG）后加权合并。"""
    low_max = int(getattr(config, "STAGE3_LOW_VIS_MAX_CLS", 3))
    low_w = float(getattr(config, "STAGE3_LOW_VIS_REG_WEIGHT", 1.0))
    idx = mask.nonzero(as_tuple=True)[0]
    cls_m = vis_cls[mask]
    low_local = cls_m <= low_max
    high_local = ~low_local
    zero = ro["gamma"].sum() * 0.0
    parts, weights = [], []
    for local_m, w in ((low_local, low_w), (high_local, 1.0)):
        if not local_m.any():
            continue
        sub_idx = idx[local_m]
        parts.append(edl_reg(
            ro["gamma"][sub_idx], ro["nu"][sub_idx],
            ro["alpha"][sub_idx], ro["beta"][sub_idx],
            vis_tgt[local_m],
        )[key])
        weights.append(w * int(local_m.sum().item()))
    if not parts:
        return zero
    if len(parts) == 1:
        return parts[0]
    total_w = sum(weights)
    return sum(p * w for p, w in zip(parts, weights)) / max(total_w, 1e-8)


def _stage3_fuse_reg_positive(edl_reg, ro, mask, vis_cls, vis_tgt, device):
    """仅 EDL 恒正 evidence-reg 项，供 dual_freeze mixed total 使用。"""
    return _stage3_weighted_edl_component(
        edl_reg, ro, mask, vis_cls, vis_tgt, key="reg",
    )


def _clamp_stage3_total_nonnegative(total: torch.Tensor) -> torch.Tensor:
    floor = float(getattr(config, "STAGE3_MIN_TOTAL_FLOOR", 1e-3))
    return total.clamp(min=floor)


class Stage3Loss(nn.Module):
    def __init__(self, w_recon=0.1, class_weights=None, align_weight=1.0):
        super().__init__()
        self.w_recon = w_recon
        self.align_weight = align_weight
        self.class_weights = class_weights
        self.use_meter = getattr(config, "USE_METER_LABELS", True)
        self.edl_cls = EDLClassificationLoss(
            num_classes=config.NUM_VIS_CLASSES,
            annealing_start=config.EDL_ANNEAL_START,
            annealing_step=config.EDL_ANNEAL_STEP,
        )
        self.edl_reg = EDLRegressionLoss(coeff=config.EDL_REG_COEFF)
        self.contra = NTXentLoss()
        self.mse = nn.MSELoss()
        self.tvkd = TemporalVisualKDLoss()
        self.ugg_reg = UGGRegularizer()
        self.tcam_loss = TCAMLoss()

    def forward(self, outputs, time_recon_out, batch, epoch=0, freeze=False,
                fusion_transition=False, dual_freeze=False,
                freeze_fusion_epochs=5, alpha_ramp=0.0):
        device = batch["vis_cls"].device
        vis_cls = batch["vis_cls"]
        vis_val = batch["vis_val"]
        cur_time = batch["cur_time_feat"]
        zero = outputs["img_feat"].sum() * 0.0

        valid_cls = vis_cls >= 0
        valid_reg = vis_val >= 0
        avail_steps = batch.get(
            "available_steps",
            torch.zeros(vis_cls.shape[0], device=device),
        )
        valid_contra = avail_steps > 0

        def _finite_rows(x: torch.Tensor) -> torch.Tensor:
            return torch.isfinite(x).all(dim=1)

        vis_val_safe = _stage3_vis_tgt_norm(vis_val)

        L_img_cls = zero.clone()
        if valid_cls.any():
            img_alpha = outputs["img_cls_out"]["alpha"]
            img_mask = valid_cls & _finite_rows(img_alpha)
            if img_mask.any():
                L_img_cls = self.edl_cls(
                    img_alpha[img_mask], vis_cls[img_mask], epoch,
                    class_weights=self.class_weights)["total"]

        L_img_reg = zero.clone()
        if valid_reg.any():
            ro = outputs["img_reg_out"]
            reg_mask = (
                valid_reg
                & torch.isfinite(ro["gamma"]).squeeze(-1)
                & torch.isfinite(ro["nu"]).squeeze(-1)
                & torch.isfinite(ro["alpha"]).squeeze(-1)
                & torch.isfinite(ro["beta"]).squeeze(-1)
                & torch.isfinite(vis_val_safe)
            )
            if reg_mask.any():
                L_img_reg = self.edl_reg(
                    ro["gamma"][reg_mask], ro["nu"][reg_mask],
                    ro["alpha"][reg_mask], ro["beta"][reg_mask],
                    vis_val_safe[reg_mask])["total"]

        L_fuse_cls = zero.clone()
        L_fuse_reg = zero.clone()
        L_fuse_reg_pos = zero.clone()
        L_fuse_reg_nig = zero.clone()
        L_fuse_sup_cls = zero.clone()
        L_fuse_sup_reg = zero.clone()
        fuse_mask = torch.zeros_like(valid_cls, dtype=torch.bool)
        reg_mask2 = torch.zeros_like(valid_reg, dtype=torch.bool)
        if valid_cls.any():
            fuse_alpha = outputs["fused_cls_out"]["alpha"]
            fuse_mask = valid_cls & _finite_rows(fuse_alpha)
            if fuse_mask.any():
                L_fuse_cls = self.edl_cls(
                    fuse_alpha[fuse_mask], vis_cls[fuse_mask], epoch,
                    class_weights=self.class_weights)["total"]
                raw_logits = outputs["fused_cls_out"].get(
                    "raw_logits", outputs["fused_cls_out"]["cls_logits"],
                )[fuse_mask]
                cw = self.class_weights
                if cw is not None:
                    cw = cw.to(raw_logits.device)
                label_smooth = float(getattr(config, "STAGE3_LABEL_SMOOTH", 0.05))
                L_fuse_sup_cls = F.cross_entropy(
                    raw_logits, vis_cls[fuse_mask], weight=cw,
                    label_smoothing=label_smooth,
                )
        if valid_reg.any():
            ro2 = outputs["fused_reg_out"]
            reg_mask2 = (
                valid_reg
                & torch.isfinite(ro2["gamma"]).squeeze(-1)
                & torch.isfinite(ro2["nu"]).squeeze(-1)
                & torch.isfinite(ro2["alpha"]).squeeze(-1)
                & torch.isfinite(ro2["beta"]).squeeze(-1)
                & torch.isfinite(vis_val_safe)
            )
            if reg_mask2.any():
                tgt = vis_val_safe[reg_mask2]
                if dual_freeze and getattr(config, "STAGE3_FUSE_REG_USE_EDL", True):
                    use_nig = getattr(config, "STAGE3_FUSE_REG_EDL_USE_NIG", False)
                    L_fuse_reg_pos = _stage3_fuse_reg_positive(
                        self.edl_reg, ro2, reg_mask2, vis_cls, tgt, device,
                    )
                    L_fuse_reg = (
                        _stage3_weighted_edl_component(
                            self.edl_reg, ro2, reg_mask2, vis_cls, tgt, key="total",
                        )
                        if use_nig
                        else L_fuse_reg_pos
                    )
                    with torch.no_grad():
                        nig_parts = []
                        idx = reg_mask2.nonzero(as_tuple=True)[0]
                        for i in idx:
                            d = self.edl_reg(
                                ro2["gamma"][i:i + 1], ro2["nu"][i:i + 1],
                                ro2["alpha"][i:i + 1], ro2["beta"][i:i + 1],
                                tgt[i:i + 1],
                            )
                            nig_parts.append(d["nig"])
                        if nig_parts:
                            L_fuse_reg_nig = torch.stack(nig_parts).mean()
                else:
                    L_fuse_reg = self.edl_reg(
                        ro2["gamma"][reg_mask2], ro2["nu"][reg_mask2],
                        ro2["alpha"][reg_mask2], ro2["beta"][reg_mask2],
                        tgt,
                    )["total"]
                    L_fuse_reg_pos = L_fuse_reg
                gamma = ro2["gamma"].squeeze(-1)[reg_mask2]
                sw = _stage3_reg_sample_weights(vis_cls, reg_mask2, device)
                L_fuse_sup_reg = (
                    F.smooth_l1_loss(gamma, tgt, reduction="none") * sw
                ).mean()

        fuse_edl = getattr(config, "STAGE3_FUSE_EDL_SCALE", 0.1) * (
            config.LOSS_CLS_WEIGHT * L_fuse_cls + config.LOSS_REG_WEIGHT * L_fuse_reg
        )
        w_sup_cls = float(getattr(config, "STAGE3_SUP_CLS_WEIGHT", 1.0))
        w_sup_reg = float(getattr(config, "STAGE3_SUP_REG_WEIGHT", 1.0))
        if dual_freeze and float(alpha_ramp) > 0.0:
            w_sup_reg *= float(getattr(config, "STAGE3_SUP_REG_WEIGHT_RAMP", 2.0))
        if dual_freeze:
            w_smooth = float(getattr(config, "STAGE3_SUP_REG_SMOOTH_WEIGHT", 0.1)) * w_sup_reg
            fuse_sup = w_sup_cls * L_fuse_sup_cls + w_smooth * L_fuse_sup_reg
        else:
            fuse_sup = w_sup_cls * L_fuse_sup_cls + w_sup_reg * L_fuse_sup_reg

        L_contra = zero.clone()
        if valid_contra.sum() >= 2:
            z_img = outputs["img_feat"][valid_contra]
            z_ts = outputs["ts_feat"][valid_contra]
            if freeze or dual_freeze:
                z_img = z_img.detach()
            if dual_freeze:
                z_ts = z_ts.detach()
            L_contra = self.contra(z_img, z_ts)

        L_time_recon = zero.clone()
        if valid_contra.any() and avail_steps.max().item() > 0:
            recon_f = time_recon_out.float()
            cur_f = cur_time.detach().float()
            recon_mask = (
                valid_contra
                & torch.isfinite(recon_f).all(dim=1)
                & torch.isfinite(cur_f).all(dim=1)
            )
            if recon_mask.any():
                L_time_recon = self.mse(recon_f[recon_mask], cur_f[recon_mask])

        L_ts_reg = zero.clone()
        if valid_contra.any() and valid_reg.any():
            combined = valid_contra & valid_reg & torch.isfinite(vis_val_safe)
            if combined.any():
                ts = outputs["ts_vis_reg_out"]
                combined = combined & (
                    torch.isfinite(ts["gamma"]).squeeze(-1)
                    & torch.isfinite(ts["nu"]).squeeze(-1)
                    & torch.isfinite(ts["alpha"]).squeeze(-1)
                    & torch.isfinite(ts["beta"]).squeeze(-1)
                )
                if combined.any():
                    L_ts_reg = self.edl_reg(
                        ts["gamma"][combined], ts["nu"][combined],
                        ts["alpha"][combined], ts["beta"][combined],
                        vis_val_safe[combined])["total"]

        L_align = zero.clone()
        if freeze and not dual_freeze and valid_contra.any() and valid_reg.any():
            align_mask = valid_contra & valid_reg
            if align_mask.any():
                img_gamma = outputs["img_reg_out"]["gamma"][align_mask].squeeze(-1).detach()
                ts_gamma = outputs["ts_vis_reg_out"]["gamma"][align_mask].squeeze(-1)
                L_align = self.mse(ts_gamma, img_gamma)

        L_kd = zero.clone()
        kd_dict = {"total": L_kd, "lambda": 0.0}
        if not freeze and not dual_freeze and getattr(config, "USE_KD", True) and valid_contra.sum() >= 2:
            ts_teacher_logits = _stage3_ts_cls_teacher_logits(outputs, device)
            if ts_teacher_logits is not None:
                img_student_logits = outputs["img_cls_out"]["cls_logits"]
                kd_dict = self.tvkd(img_student_logits, ts_teacher_logits, epoch=epoch)
                L_kd = kd_dict["total"]

        ugg_dict = self.ugg_reg(outputs["gate_alpha"], outputs["gate_u_img"], avail_steps)
        L_gate_reg = ugg_dict["total"]
        tcam_dict = self.tcam_loss(outputs, batch)
        L_tcam = tcam_dict["total"]

        L_fuse_distill = zero.clone()
        use_fuse_distill = _stage3_use_fuse_distill(
            epoch, freeze, fusion_transition, dual_freeze, freeze_fusion_epochs,
        )
        if use_fuse_distill:
            if valid_cls.any():
                fuse_prob = outputs["fused_cls_out"]["prob"]
                img_prob = outputs["img_cls_out"]["prob"].detach()
                mask = valid_cls & _finite_rows(fuse_prob) & _finite_rows(img_prob)
                if mask.any():
                    L_fuse_distill = self.mse(fuse_prob[mask], img_prob[mask])
            if valid_reg.any():
                fuse_g = outputs["fused_reg_out"]["gamma"].squeeze(-1)
                img_g = outputs["img_reg_out"]["gamma"].squeeze(-1).detach()
                mask = (
                    valid_reg
                    & torch.isfinite(fuse_g)
                    & torch.isfinite(img_g)
                )
                if mask.any():
                    L_fuse_distill = L_fuse_distill + self.mse(fuse_g[mask], img_g[mask])

        L_gate_anchor = zero.clone()
        L_blend_anchor = zero.clone()
        L_blend_reg_anchor = zero.clone()
        L_gamma_distill = zero.clone()
        L_gate_prewarm = zero.clone()
        fuse_img_prob_mse = zero.clone()
        if valid_cls.any():
            fuse_prob_diag = outputs["fused_cls_out"]["prob"]
            img_prob_diag = outputs["img_cls_out"]["prob"].detach()
            diag_mask = valid_cls & _finite_rows(fuse_prob_diag) & _finite_rows(img_prob_diag)
            if diag_mask.any():
                fuse_img_prob_mse = self.mse(fuse_prob_diag[diag_mask], img_prob_diag[diag_mask])
        if dual_freeze:
            ga_raw = outputs["gate_alpha_raw"].squeeze(-1)
            L_gate_prewarm = (ga_raw ** 2).mean()
        if float(alpha_ramp) > 0.0:
            gate_a = outputs["gate_alpha_raw"].squeeze(-1)
            L_gate_anchor = (gate_a ** 2).mean() * float(alpha_ramp)
            if valid_cls.any():
                fuse_prob = outputs["fused_cls_out"]["prob"]
                img_prob = outputs["img_cls_out"]["prob"].detach()
                mask = valid_cls & _finite_rows(fuse_prob) & _finite_rows(img_prob)
                if mask.any():
                    L_blend_anchor = (
                        self.mse(fuse_prob[mask], img_prob[mask]) * float(alpha_ramp)
                    )
            if valid_reg.any():
                fuse_g = outputs["fused_reg_out"]["gamma"].squeeze(-1)
                img_g = outputs["img_reg_out"]["gamma"].squeeze(-1).detach()
                reg_mask_b = (
                    valid_reg
                    & torch.isfinite(fuse_g)
                    & torch.isfinite(img_g)
                )
                if reg_mask_b.any():
                    sw_b = _stage3_reg_sample_weights(vis_cls, reg_mask_b, device)
                    diff = (fuse_g[reg_mask_b] - img_g[reg_mask_b]) ** 2
                    blend_scale = (
                        float(alpha_ramp)
                        if getattr(config, "STAGE3_BLEND_REG_USE_RAMP", False)
                        else 1.0
                    )
                    L_blend_reg_anchor = (diff * sw_b).mean() * blend_scale
        if float(alpha_ramp) > 0.0 and valid_reg.any():
            fuse_gd = outputs["fused_reg_out"]["gamma"].squeeze(-1)
            img_gd = outputs["img_reg_out"]["gamma"].squeeze(-1).detach()
            gd_mask = (
                valid_reg
                & torch.isfinite(fuse_gd)
                & torch.isfinite(img_gd)
            )
            if gd_mask.any():
                sw_gd = _stage3_reg_sample_weights(vis_cls, gd_mask, device)
                L_gamma_distill = (
                    (fuse_gd[gd_mask] - img_gd[gd_mask]) ** 2 * sw_gd
                ).mean()

        w_prewarm = float(getattr(config, "STAGE3_GATE_PREWARM_WEIGHT", 0.05))
        w_contra = float(config.LOSS_CONTRA_WEIGHT)
        if float(alpha_ramp) > 0.0:
            w_contra = float(getattr(config, "STAGE3_CONTRA_WEIGHT_RAMP", 0.05))
        w_reg_edl = (
            float(getattr(config, "STAGE3_FUSE_REG_EDL_WEIGHT", 1.0))
            if dual_freeze and getattr(config, "STAGE3_FUSE_REG_USE_EDL", True)
            else 0.0
        )
        w_gamma_d = float(getattr(config, "STAGE3_GAMMA_DISTILL_WEIGHT", 2.0))

        fuse_sup = _sanitize_loss_scalar(fuse_sup)
        L_fuse_distill = _sanitize_loss_scalar(L_fuse_distill)
        L_gate_prewarm = _sanitize_loss_scalar(L_gate_prewarm)
        L_gate_anchor = _sanitize_loss_scalar(L_gate_anchor)
        L_blend_anchor = _sanitize_loss_scalar(L_blend_anchor)
        L_blend_reg_anchor = _sanitize_loss_scalar(L_blend_reg_anchor)
        L_gamma_distill = _sanitize_loss_scalar(L_gamma_distill)
        L_fuse_reg_pos = _sanitize_loss_scalar(L_fuse_reg_pos)
        L_gate_reg = _sanitize_loss_scalar(L_gate_reg)
        L_contra = _sanitize_loss_scalar(L_contra, max_val=float(getattr(config, "STAGE3_CONTRA_MAX", 10.0)))

        if freeze:
            if dual_freeze:
                total = (
                    fuse_sup
                    + w_contra * L_contra
                    + L_fuse_distill
                    + w_prewarm * L_gate_prewarm
                )
                if float(alpha_ramp) > 0.0:
                    total = (
                        total + L_gate_reg
                        + w_reg_edl * L_fuse_reg_pos
                        + w_gamma_d * L_gamma_distill
                    )
                    w_blend_r_f = getattr(config, "STAGE3_BLEND_REG_ANCHOR_WEIGHT", 1.0)
                    w_blend_a_f = getattr(config, "STAGE3_BLEND_ANCHOR_WEIGHT", 0.5)
                    w_gate_a_f = getattr(config, "STAGE3_GATE_ANCHOR_WEIGHT", 0.1)
                    total = (
                        total
                        + w_gate_a_f * L_gate_anchor
                        + w_blend_a_f * L_blend_anchor
                        + w_blend_r_f * L_blend_reg_anchor
                    )
                total = _apply_stage3_total_floor(total, fuse_sup, dual_freeze=True)
                total = _sanitize_loss_scalar(total)
                total = _clamp_stage3_total_nonnegative(total)
            else:
                total = (
                    self.align_weight * L_align
                    + config.LOSS_CONTRA_WEIGHT * L_contra
                    + config.LOSS_TS_WEIGHT * L_ts_reg
                    + self.w_recon * L_time_recon
                    + L_fuse_distill
                    + fuse_sup
                )
                if float(alpha_ramp) > 0.0:
                    w_gate_a = getattr(config, "STAGE3_GATE_ANCHOR_WEIGHT", 0.1)
                    w_blend_a = getattr(config, "STAGE3_BLEND_ANCHOR_WEIGHT", 0.5)
                    w_blend_r = getattr(config, "STAGE3_BLEND_REG_ANCHOR_WEIGHT", 1.0)
                    total = (
                        total
                        + w_gate_a * L_gate_anchor
                        + w_blend_a * L_blend_anchor
                        + w_blend_r * L_blend_reg_anchor
                        + w_gamma_d * L_gamma_distill
                    )
        elif dual_freeze:
            w_gate_a = getattr(config, "STAGE3_GATE_ANCHOR_WEIGHT", 0.1)
            w_blend_a = getattr(config, "STAGE3_BLEND_ANCHOR_WEIGHT", 0.5)
            w_blend_r = getattr(config, "STAGE3_BLEND_REG_ANCHOR_WEIGHT", 1.0)
            total = (
                fuse_sup
                + w_reg_edl * L_fuse_reg_pos
                + w_contra * L_contra
                + L_gate_reg + L_tcam
                + w_gate_a * L_gate_anchor
                + w_blend_a * L_blend_anchor
                + w_blend_r * L_blend_reg_anchor
                + w_gamma_d * L_gamma_distill
                + w_prewarm * L_gate_prewarm
            )
            total = _apply_stage3_total_floor(total, fuse_sup, dual_freeze=True)
            total = _sanitize_loss_scalar(total)
            total = _clamp_stage3_total_nonnegative(total)
        else:
            total = (
                config.LOSS_CLS_WEIGHT * L_img_cls * 0.5
                + config.LOSS_REG_WEIGHT * L_img_reg * 0.5
                + fuse_sup
                + w_contra * L_contra
                + self.w_recon * L_time_recon
                + config.LOSS_TS_WEIGHT * L_ts_reg
                + L_kd + L_gate_reg + L_tcam
            )
            if use_fuse_distill:
                total = total + L_fuse_distill
            if float(alpha_ramp) > 0.0:
                w_gate_a = getattr(config, "STAGE3_GATE_ANCHOR_WEIGHT", 0.1)
                w_blend_a = getattr(config, "STAGE3_BLEND_ANCHOR_WEIGHT", 0.5)
                w_blend_r = getattr(config, "STAGE3_BLEND_REG_ANCHOR_WEIGHT", 1.0)
                total = (
                    total
                    + w_gate_a * L_gate_anchor
                    + w_blend_a * L_blend_anchor
                    + w_blend_r * L_blend_reg_anchor
                    + w_gamma_d * L_gamma_distill
                )
            total = _apply_stage3_total_floor(total, fuse_sup, dual_freeze=False)
            total = _sanitize_loss_scalar(total)

        return {
            "total": total, "img_cls": L_img_cls, "img_reg": L_img_reg,
            "fuse_cls": L_fuse_cls, "fuse_reg": L_fuse_reg,
            "fuse_sup_cls": L_fuse_sup_cls, "fuse_sup_reg": L_fuse_sup_reg,
            "fuse_distill": L_fuse_distill, "fuse_edl": fuse_edl,
            "gate_anchor": L_gate_anchor, "blend_anchor": L_blend_anchor,
            "blend_reg_anchor": L_blend_reg_anchor,
            "gamma_distill": L_gamma_distill,
            "fuse_reg_edl": L_fuse_reg_pos,
            "reg_nig": L_fuse_reg_nig,
            "gate_prewarm": L_gate_prewarm,
            "contra": L_contra, "time_recon": L_time_recon,
            "ts_reg": L_ts_reg, "align": L_align, "kd": L_kd,
            "kd_lambda": kd_dict["lambda"], "gate_reg": L_gate_reg, "tcam": L_tcam,
        }


def is_better_val_mae_ckpt(val_mae, mae_0_3, best_mae, best_mae_0_3, tie_tol):
    """全局 MAE 为主；接近时以 0-3 类 MAE 决胜（与 Stage2 一致）。"""
    if val_mae < 0:
        return False
    if best_mae < 0:
        return True
    if val_mae < best_mae - tie_tol:
        return True
    if abs(val_mae - best_mae) <= tie_tol:
        if mae_0_3 >= 0 and (best_mae_0_3 < 0 or mae_0_3 < best_mae_0_3):
            return True
    return False


def _load_noaa_pretrained(model, device):
    if os.path.exists(config.NOAA_PRETRAIN_CKPT):
        ckpt = load_checkpoint(config.NOAA_PRETRAIN_CKPT, map_location=device)
        state = ckpt.get("model_state", ckpt)
        key_w = "input_proj.weight"
        key_b = "input_proj.bias"
        if key_w in state:
            cur_w = model.ts_branch.state_dict().get(key_w)
            if cur_w is not None and state[key_w].shape != cur_w.shape:
                old_w = state[key_w]
                out_dim, in_dim = cur_w.shape
                old_out, old_in = old_w.shape
                if old_out == out_dim:
                    if old_in < in_dim:
                        pad = torch.zeros(
                            (out_dim, in_dim - old_in),
                            device=old_w.device, dtype=old_w.dtype,
                        )
                        state[key_w] = torch.cat([old_w, pad], dim=1)
                    else:
                        state[key_w] = old_w[:, :in_dim]
        if key_b in state:
            cur_b = model.ts_branch.state_dict().get(key_b)
            if cur_b is not None and state[key_b].shape != cur_b.shape:
                state.pop(key_b, None)
        skipped, missing, _ = safe_load_state_dict(model.ts_branch, state)
        print(f"[权重] 时序分支 ← NOAA预训练 "
              f"(加载={len(state)-len(skipped)}, 跳过={len(skipped)}, 缺失={len(missing)})")
    else:
        print(f"[警告] 未找到NOAA预训练: {config.NOAA_PRETRAIN_CKPT}")


def _init_fusion_from_img_branch(model, img_state: dict):
    """Warm-start fusion heads/projector from stage-2 image branch weights."""
    fusion = getattr(model, "fusion", None)
    if fusion is None or not img_state:
        return

    # Preserve img_feat structure through img_proj (512 -> 256 partial identity).
    proj = fusion.img_proj[0]
    if isinstance(proj, nn.Linear) and proj.weight.shape == (config.FUSED_DIM, config.IMAGE_FEAT_DIM):
        with torch.no_grad():
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            for i in range(min(config.FUSED_DIM, config.IMAGE_FEAT_DIM)):
                proj.weight[i, i] = 1.0
    _init_layernorm_neutral(fusion.img_proj[1])

    # post 近恒等，使 alpha=0 时融合路径 ≈ 图像投影特征
    post_lin = fusion.post[0]
    if isinstance(post_lin, nn.Linear) and post_lin.in_features == post_lin.out_features:
        with torch.no_grad():
            nn.init.eye_(post_lin.weight)
            nn.init.zeros_(post_lin.bias)
    _init_layernorm_neutral(fusion.post[1])

    # cls 首层：复用图像分支前 FUSED_DIM 维权重
    src_w = img_state.get("cls_head.net.0.weight")
    src_b = img_state.get("cls_head.net.0.bias")
    if src_w is not None and hasattr(fusion.cls_head, "net"):
        dst_lin = fusion.cls_head.net[0]
        if isinstance(dst_lin, nn.Linear) and src_w.shape[0] == dst_lin.out_features:
            with torch.no_grad():
                dst_lin.weight.data = src_w[:, :dst_lin.in_features].clone()
                if src_b is not None:
                    dst_lin.bias.data = src_b.clone()

    # reg_proj: fused -> img_dim 部分恒等（后补零维）
    rp = fusion.reg_proj
    if isinstance(rp, nn.Linear) and rp.out_features == config.IMAGE_FEAT_DIM:
        with torch.no_grad():
            nn.init.zeros_(rp.weight)
            nn.init.zeros_(rp.bias)
            for i in range(min(rp.in_features, rp.out_features)):
                rp.weight[i, i] = 1.0

    # Copy classifier output layer (128 -> K) when shapes match.
    cls_dst = fusion.cls_head.state_dict()
    copied_cls = 0
    for key in ("net.3.weight", "net.3.bias"):
        src_key = f"cls_head.{key}"
        if src_key in img_state and key in cls_dst and cls_dst[key].shape == img_state[src_key].shape:
            cls_dst[key] = img_state[src_key]
            copied_cls += 1
    if copied_cls:
        fusion.cls_head.load_state_dict(cls_dst)

    # Copy regression head only when hidden dims match (512 -> shared -> heads).
    reg_dst = fusion.reg_head.state_dict()
    copied_reg = 0
    for key, val in reg_dst.items():
        src_key = f"reg_head.{key}"
        if src_key in img_state and val.shape == img_state[src_key].shape:
            reg_dst[key] = img_state[src_key]
            copied_reg += 1
    if copied_reg:
        fusion.reg_head.load_state_dict(reg_dst)
    print(f"[权重] 融合头 warm-start ← 图像分支 "
          f"(cls层={copied_cls}, reg参数={copied_reg}, "
          f"img_proj/post/reg_proj=partial-identity)")


def load_pretrained_weights(model, device, dataset_name,
                            load_noaa_pretrain: bool = True,
                            load_stage2: bool = True,
                            stage2_path: str = None):
    if load_noaa_pretrain:
        _load_noaa_pretrained(model, device)
    else:
        print("[ablation] temporal branch random init; skip NOAA pretrain")

    if stage2_path is None:
        stage2_path = _resolve_stage2_ckpt(dataset_name)
    if load_stage2 and os.path.exists(stage2_path):
        ckpt2 = load_checkpoint(stage2_path, map_location=device)
        img_state = ckpt2["img_branch_state"]
        skipped, missing, _ = safe_load_state_dict(model.img_branch, img_state)
        n_loaded = len(img_state) - len(skipped)
        print(f"[权重] 图像分支 ← 阶段二 ({stage2_path}) "
              f"(加载={n_loaded}, 跳过={len(skipped)}, 缺失={len(missing)})")
        if skipped:
            preview = ", ".join(skipped[:4])
            if len(skipped) > 4:
                preview += "..."
            print(f"  跳过键(形状不匹配): {preview}")
        _init_fusion_from_img_branch(model, img_state)
    elif load_stage2:
        print(f"[警告] 未找到阶段二权重: {config.stage2_ckpt_path(dataset_name)}")


def build_stage3_model(device, dataset_name="frosi", d_in=None,
                       load_noaa_pretrain=True, load_stage2=True,
                       model_kwargs=None, pretrained=True,
                       stage2_path: str = None):
    d_in = d_in or (config.NUM_METEO_FEATURES + 1)
    kwargs = dict(model_kwargs or {})
    model = VisibilityModel(d_in=d_in, pretrained=pretrained, **kwargs).to(device)
    load_pretrained_weights(
        model, device, dataset_name,
        load_noaa_pretrain=load_noaa_pretrain,
        load_stage2=load_stage2,
        stage2_path=stage2_path,
    )
    return model


def build_aligner(dataset_name: str, no_time_align: bool = False):
    if no_time_align:
        return None
    processor = NOAADataProcessor()
    processor.load()
    return TimeAligner(processor)


def inject_delta_hours_batch(batch, aligner, prob=None,
                             min_h=None, max_h=None):
    """训练时随机注入 delta_hours 与 future_vis。"""
    if aligner is None:
        return batch
    prob = prob if prob is not None else getattr(config, "STAGE3_DELTA_INJECT_PROB", 0.5)
    min_h = min_h if min_h is not None else getattr(config, "STAGE3_DELTA_MIN_H", 0.5)
    max_h = max_h if max_h is not None else getattr(config, "STAGE3_DELTA_MAX_H", 6.0)

    B = batch["img"].shape[0]
    device = batch["img"].device
    delta_h = batch.get("delta_hours", torch.zeros(B, device=device)).clone()
    future_vis = batch.get("future_vis", torch.full((B,), -1.0, device=device)).clone()

    if prob <= 0:
        batch["delta_hours"] = delta_h
        batch["future_vis"] = future_vis
        return batch

    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        batch["delta_hours"] = delta_h
        batch["future_vis"] = future_vis
        return batch

    import pandas as pd

    for i in torch.where(mask)[0].tolist():
        dh = float(np.random.uniform(min_h, max_h))
        delta_h[i] = dh
        station_id = None
        if "align_station_id" in batch:
            sids = batch["align_station_id"]
            sid = sids[i] if isinstance(sids, (list, tuple)) else sids[i]
            if isinstance(sid, str) and sid:
                station_id = sid
        query_dt = None
        if "sample_datetime" in batch:
            dts = batch["sample_datetime"]
            dt_raw = dts[i] if isinstance(dts, (list, tuple)) else dts
            if isinstance(dt_raw, str) and dt_raw:
                query_dt = pd.Timestamp(dt_raw)
        if query_dt is not None and station_id:
            fut = aligner.get_future_vis(query_dt, dh, station_id=station_id)
            if fut >= 0:
                from data.noaa_dataset import normalize_vis
                future_vis[i] = normalize_vis(fut)

    batch["delta_hours"] = delta_h
    batch["future_vis"] = future_vis
    return batch


def _stage3_clip_params(model, recon_head, criterion, dual_freeze):
    """Parameters that may receive gradients in Stage3 train_epoch."""
    if dual_freeze:
        modules = [getattr(model, "fusion", None), criterion.contra]
    else:
        modules = [model, recon_head, criterion.contra]
    params = []
    seen = set()
    for mod in modules:
        if mod is None:
            continue
        for p in mod.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def train_epoch(model, recon_head, loader, optimizers, scaler, device, criterion,
                epoch, accum=4, freeze=False, freeze_fusion_epochs=5,
                amp_enabled=True, aligner=None, inject_delta=True,
                dual_freeze=False):
    model.train()
    criterion.train()
    if not dual_freeze:
        recon_head.train()
    fusion_transition = (
        (not freeze) and (epoch <= freeze_fusion_epochs + FUSION_TRANSITION_EPOCHS)
    )
    if dual_freeze:
        force_img_only = False
        alpha_scale = _stage3_alpha_schedule(
            epoch, freeze_fusion_epochs, dual_freeze=True,
        )
        min_blend = float(getattr(config, "STAGE3_TRAIN_MIN_ALPHA", 0.02))
        blend_alpha_scale = max(alpha_scale, min_blend)
    else:
        force_img_only = False
        alpha_scale = _stage3_alpha_schedule(
            epoch, freeze_fusion_epochs, dual_freeze=False,
        )
        min_blend = float(getattr(config, "STAGE3_TRAIN_MIN_ALPHA", 0.02))
        blend_alpha_scale = max(alpha_scale, min_blend)
    effective_ramp = float(alpha_scale)
    sums = {k: 0.0 for k in [
        "total", "img_cls", "img_reg", "fuse_cls", "fuse_reg",
        "fuse_sup_cls", "fuse_sup_reg", "fuse_distill", "fuse_edl",
        "gate_anchor", "blend_anchor", "blend_reg_anchor", "gamma_distill",
        "fuse_reg_edl", "gate_prewarm",
        "contra", "time_recon", "ts_reg", "kd", "kd_lambda", "gate_reg", "tcam",
    ]}
    n = 0
    n_skip = 0
    n_batches = 0
    for opt in optimizers:
        opt.zero_grad()

    for step, raw in enumerate(loader):
        b = to_device_batch(raw, device)
        if inject_delta:
            b = inject_delta_hours_batch(b, aligner)
        with autocast(enabled=amp_enabled):
            delta_h = b.get("delta_hours", torch.zeros(b["img"].shape[0], device=device))
            outputs = model(
                b["img"], b["depth"], b["trans"],
                b["seq_feat"], b["time_feat_seq"],
                b["cur_time_feat"], b["available_steps"],
                delta_hours=delta_h,
                force_img_only=force_img_only,
                alpha_scale=alpha_scale,
                blend_alpha_scale=blend_alpha_scale,
            )
            for head in ("img_cls_out", "fused_cls_out"):
                outputs[head]["alpha"] = torch.nan_to_num(
                    outputs[head]["alpha"], nan=1.0, posinf=1.0, neginf=1.0
                )
                _refresh_edl_cls_out(outputs[head])
            for head in ("img_reg_out", "fused_reg_out", "ts_vis_reg_out"):
                for k in ("gamma", "nu", "alpha", "beta"):
                    outputs[head][k] = torch.nan_to_num(
                        outputs[head][k], nan=0.0, posinf=0.0, neginf=0.0
                    )
            if dual_freeze:
                recon = torch.zeros(
                    b["vis_cls"].shape[0], 10, device=device, dtype=torch.float32,
                )
            else:
                ts_feat = torch.nan_to_num(
                    outputs["ts_feat"].float(), nan=0.0, posinf=0.0, neginf=0.0,
                )
                recon = recon_head(ts_feat)
            ld = criterion(
                outputs, recon, b, epoch, freeze=freeze,
                fusion_transition=fusion_transition, dual_freeze=dual_freeze,
                freeze_fusion_epochs=freeze_fusion_epochs,
                alpha_ramp=effective_ramp,
            )

        clip_params = _stage3_clip_params(model, recon_head, criterion, dual_freeze)
        loss = _ensure_loss_grad((ld["total"] / accum).float(), clip_params)

        n_batches += 1
        if not torch.isfinite(ld["total"]):
            n_skip += 1
            for opt in optimizers:
                opt.zero_grad()
            continue

        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            for opt in optimizers:
                scaler.unscale_(opt)
            if clip_params:
                nn.utils.clip_grad_norm_(clip_params, config.GRAD_CLIP)
            for opt in optimizers:
                if any(
                    p.grad is not None
                    for g in opt.param_groups for p in g.get("params", [])
                ):
                    scaler.step(opt)
                opt.zero_grad()
            scaler.update()

        bs = len(b["vis_cls"])
        for k in sums:
            if k in ld:
                v = ld[k]
                sums[k] += (v.item() if torch.is_tensor(v) else float(v)) * bs
        n += bs

    if n_batches > 0 and n_batches % accum != 0:
        for opt in optimizers:
            scaler.unscale_(opt)
        clip_params = _stage3_clip_params(model, recon_head, criterion, dual_freeze)
        if clip_params:
            nn.utils.clip_grad_norm_(clip_params, config.GRAD_CLIP)
        for opt in optimizers:
            if any(
                p.grad is not None
                for g in opt.param_groups for p in g.get("params", [])
            ):
                scaler.step(opt)
            opt.zero_grad()
        scaler.update()

    return {k: v / max(n, 1) for k, v in sums.items()}, n_skip, n_batches


@torch.no_grad()
def eval_epoch(model, loader, device, epoch=0, freeze_epochs=5,
               delta_hours_override=None, variant_name=None,
               force_img_only=None, head=None, for_test=False,
               alpha_scale=None, dual_freeze=False,
               alpha_scale_override=None):
    model.eval()
    pred_cls_all, gt_cls_all = [], []
    pred_val_all, gt_val_all = [], []
    gate_alpha_sum, gate_alpha_n = 0.0, 0
    max_vis = float(getattr(config, "MAX_VIS", 50000.0))
    use_ablation_fwd = variant_name is not None
    mode = _stage3_eval_mode(
        epoch, freeze_epochs, head=head, for_test=for_test,
        force_img_only=force_img_only, dual_freeze=dual_freeze,
        alpha_scale_override=alpha_scale_override,
    )
    if force_img_only is None:
        force_img_only = mode["force_img_only"]
    use_fused = mode["use_fused"]
    if alpha_scale is None:
        alpha_scale = mode["alpha_scale"]
    blend_alpha_scale = mode.get(
        "blend_alpha_scale",
        _stage3_blend_alpha_scale(alpha_scale),
    )

    for bi, raw in enumerate(loader):
        b = to_device_batch(raw, device)
        B = b["img"].shape[0]
        if delta_hours_override is not None:
            delta_h = torch.full((B,), float(delta_hours_override), device=device)
            b = dict(b)
            b["delta_hours"] = delta_h
        if use_ablation_fwd:
            out = ablation_forward(model, b, variant_name, device)
        else:
            delta_h = b.get("delta_hours", torch.zeros(B, device=device))
            out = model(
                b["img"], b["depth"], b["trans"],
                b["seq_feat"], b["time_feat_seq"],
                b["cur_time_feat"], b["available_steps"],
                delta_hours=delta_h,
                force_img_only=force_img_only,
                alpha_scale=alpha_scale,
                blend_alpha_scale=blend_alpha_scale,
            )
        gt_cls = b["vis_cls"].cpu().numpy()
        if use_fused:
            prob = torch.nan_to_num(
                out["fused_cls_out"]["prob"], nan=0.0, posinf=0.0, neginf=0.0,
            )
            pred_val_norm = out["fused_reg_out"]["gamma"].squeeze(-1)
            head_used = "fused_img_only" if force_img_only else "fused"
        else:
            prob = torch.nan_to_num(
                out["img_cls_out"]["prob"], nan=0.0, posinf=0.0, neginf=0.0,
            )
            pred_val_norm = out["img_reg_out"]["gamma"].squeeze(-1)
            head_used = "img"
        pred_cls = prob.argmax(-1).cpu().numpy()
        gate_alpha_sum += out["gate_alpha"].float().sum().item()
        gate_alpha_n += out["gate_alpha"].numel()
        pred_val = torch.nan_to_num(pred_val_norm, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy() * max_vis
        gt_raw = b["vis_val"].cpu().numpy()
        gt_val = _stage3_vis_val_to_meter_np(gt_raw, max_vis)
        pred_cls_all.extend(pred_cls.tolist())
        gt_cls_all.extend(gt_cls.tolist())
        pred_val_all.extend(pred_val.tolist())
        gt_val_all.extend(gt_val.tolist())

    gt_val_arr = np.array(gt_val_all)
    pred_val_arr = np.array(pred_val_all)
    gt_cls_arr = np.array(gt_cls_all)
    pred_cls_arr = np.array(pred_cls_all)
    valid = (gt_val_arr >= 0) & (gt_cls_arr >= 0)
    if valid.any():
        metrics = compute_metrics(
            gt_val_arr[valid], pred_val_arr[valid],
            gt_cls_arr[valid], pred_cls_arr[valid],
        )
    else:
        metrics = compute_metrics(gt_val_arr, pred_val_arr, gt_cls_arr, pred_cls_arr)
    if gate_alpha_n > 0:
        metrics["gate_alpha_mean"] = gate_alpha_sum / gate_alpha_n
    return metrics


def eval_current(model, loader, device, variant_name=None):
    return eval_epoch(
        model, loader, device, epoch=999, freeze_epochs=0,
        variant_name=variant_name, for_test=True,
    )


def safe_load_state_dict(model, state):
    model_state = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return skipped, missing, unexpected


def _sanitize_ckpt_suffix(variant_name: str, ckpt_suffix=None) -> str:
    raw = ckpt_suffix if ckpt_suffix is not None else variant_name
    return str(raw).replace("/", "_").replace("\\", "_")


def train_stage3(
    variant_name: str,
    dataset_name: str,
    aligner,
    device,
    epochs=None,
    lr=None,
    batch_size=None,
    accum=4,
    freeze_fusion=None,
    w_recon=0.1,
    ts_lr_ratio=0.05,
    model_kwargs=None,
    load_noaa_pretrain=True,
    load_stage2=True,
    use_kd=True,
    contra_weight=None,
    ckpt_suffix=None,
    inject_delta=True,
    skip_align_check=True,
    model=None,
    dual_freeze=None,
    stage2_path: str = None,
):
    """完整三阶段联合微调，供各消融 runner 调用。默认 dual_freeze 与主模型一致。"""
    set_seed()
    epochs = epochs if epochs is not None else int(
        getattr(config, "MAIN_STAGE3_EPOCHS", 15)
    )
    freeze_fusion = freeze_fusion if freeze_fusion is not None else int(
        getattr(config, "MAIN_FREEZE_FUSION", 5)
    )
    if dual_freeze is None:
        dual_freeze = bool(getattr(config, "MAIN_DUAL_FREEZE", True))
    lr = lr or config.MAIN_LR
    batch_size = batch_size or config.BATCH_SIZE
    config.USE_METER_LABELS = True

    orig_kd = config.USE_KD
    orig_contra = config.LOSS_CONTRA_WEIGHT
    config.USE_KD = use_kd
    if contra_weight is not None:
        config.LOSS_CONTRA_WEIGHT = contra_weight

    train_loader, val_loader, test_loader = build_all_loaders(
        dataset_name, aligner, batch_size
    )
    class_weights = None
    if config.CLASS_WEIGHT_MODE != "none":
        counts = get_class_counts(train_loader.dataset)
        class_weights = compute_class_weights(counts, mode=config.CLASS_WEIGHT_MODE)

    if model is None:
        model = build_stage3_model(
            device, dataset_name,
            load_noaa_pretrain=load_noaa_pretrain,
            load_stage2=load_stage2,
            model_kwargs=model_kwargs,
            stage2_path=stage2_path,
        )
    else:
        model = model.to(device)
    recon_hd = TimeReconHead(config.TIME_FEAT_DIM, 10).to(device)
    criterion = Stage3Loss(w_recon=w_recon, class_weights=class_weights).to(device)

    if dual_freeze:
        opt_fusion = AdamW(
            model.fusion.parameters(), lr=lr, weight_decay=config.MAIN_WEIGHT_DECAY,
        )
        opt_align = AdamW(
            criterion.contra.parameters(), lr=lr, weight_decay=config.MAIN_WEIGHT_DECAY,
        )
        optimizers = [opt_fusion, opt_align]
        schedulers = [
            CosineAnnealingLR(opt_fusion, T_max=epochs, eta_min=lr * 0.01),
            CosineAnnealingLR(opt_align, T_max=epochs, eta_min=lr * 0.01),
        ]
    else:
        img_params = (
            list(model.img_branch.parameters())
            + list(getattr(model, "fusion", nn.Module()).parameters())
        )
        ts_params = (
            list(getattr(model, "ts_branch", nn.Module()).parameters())
            if hasattr(model, "ts_branch") else []
        )
        recon_params = list(recon_hd.parameters())
        opt_img = AdamW(img_params, lr=lr, weight_decay=config.MAIN_WEIGHT_DECAY)
        opt_ts = AdamW(ts_params, lr=lr * ts_lr_ratio, weight_decay=config.MAIN_WEIGHT_DECAY)
        opt_recon = AdamW(recon_params, lr=lr, weight_decay=config.MAIN_WEIGHT_DECAY)
        opt_align = AdamW(
            criterion.contra.parameters(), lr=lr, weight_decay=config.MAIN_WEIGHT_DECAY,
        )
        optimizers = [opt_img, opt_ts, opt_recon, opt_align]
        schedulers = [
            CosineAnnealingLR(opt_img, T_max=epochs, eta_min=lr * 0.01),
            CosineAnnealingLR(opt_ts, T_max=epochs, eta_min=lr * ts_lr_ratio * 0.01),
            CosineAnnealingLR(opt_recon, T_max=epochs, eta_min=lr * 0.01),
            CosineAnnealingLR(opt_align, T_max=epochs, eta_min=lr * 0.01),
        ]

    scaler = GradScaler()

    suffix = _sanitize_ckpt_suffix(variant_name, ckpt_suffix)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"ablation_{suffix}_{dataset_name}.pth")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_mae = -1.0
    best_mae_0_3 = -1.0
    best_ckpt_epoch = 0
    tie_tol = getattr(config, "STAGE3_BEST_MAE_TIE_TOL", 50.0)
    ugg_start = freeze_fusion + FUSION_TRANSITION_EPOCHS

    print(f"[训练] 变体={variant_name} 数据集={dataset_name} epochs={epochs}")

    for epoch in range(1, epochs + 1):
        freeze = epoch <= freeze_fusion
        set_stage3_freeze_state(
            model, epoch, freeze_fusion, dual_freeze=dual_freeze,
        )
        ramp = _stage3_alpha_schedule(epoch, freeze_fusion, dual_freeze=dual_freeze)
        L, n_skip, n_batches = train_epoch(
            model, recon_hd, train_loader,
            optimizers, scaler, device, criterion,
            epoch, accum=accum, freeze=freeze, freeze_fusion_epochs=freeze_fusion,
            aligner=aligner, inject_delta=inject_delta, dual_freeze=dual_freeze,
        )
        M = eval_epoch(
            model, val_loader, device, epoch=epoch,
            freeze_epochs=freeze_fusion, dual_freeze=dual_freeze,
        )
        for sch in schedulers:
            sch.step()
        unique_pred = int(M.get("unique_pred_classes", config.NUM_VIS_CLASSES))
        in_ugg = epoch > ugg_start
        if epoch == freeze_fusion:
            M_img = eval_epoch(
                model, val_loader, device, epoch=epoch,
                freeze_epochs=freeze_fusion, head="img",
            )
            M_fused_io = eval_epoch(
                model, val_loader, device, epoch=epoch,
                freeze_epochs=freeze_fusion, force_img_only=True, head="fused",
                dual_freeze=dual_freeze,
            )
            print(
                f"  [诊断] E{epoch} freeze末: img_ACC={M_img['acc']:.4f} "
                f"fused_io_ACC={M_fused_io['acc']:.4f}"
            )
        mae_0_3 = M.get("mae_cls_0_3", -1)
        mae_0_3_s = f"{mae_0_3:.1f}" if mae_0_3 >= 0 else "—"
        if dual_freeze:
            print(
                f"  E{epoch:3d} tot={L['total']:.4f} sup_r={L.get('fuse_sup_reg', 0):.6f} "
                f"ACC={M['acc']:.4f} MAE={M['mae']:.1f} MAE_0-3={mae_0_3_s} "
                f"a={ramp:.2f} skip={n_skip}/{n_batches}"
            )
        else:
            print(
                f"  E{epoch:3d} loss={L['total']:.4f} "
                f"ACC={M['acc']:.4f} MAE={M['mae']:.1f} "
                f"a={ramp:.2f} [{time.time():.0f}]"
            )
        val_mae = M["mae"]
        can_save = unique_pred >= 2 if in_ugg else True
        if not can_save:
            print(
                f"  [警告] 跳过保存 best: 融合预测仅 {unique_pred} 个类别 "
                f"(分布={M.get('pred_cls_dist', [])})"
            )
        elif is_better_val_mae_ckpt(val_mae, mae_0_3, best_mae, best_mae_0_3, tie_tol):
            best_mae = val_mae
            best_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_mae_0_3
            best_ckpt_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "variant": variant_name,
                "epoch": epoch,
                "alpha_ramp": float(ramp),
                "dual_freeze": dual_freeze,
                "metrics": M,
            }, ckpt_path)

    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        safe_load_state_dict(model, ckpt["model_state"])
        best_ckpt_epoch = int(ckpt.get("epoch", best_ckpt_epoch))
        best_alpha = float(ckpt.get("alpha_ramp", _stage3_alpha_schedule(
            best_ckpt_epoch, freeze_fusion, dual_freeze=dual_freeze,
        )))
    else:
        best_alpha = _stage3_alpha_schedule(
            best_ckpt_epoch or epochs, freeze_fusion, dual_freeze=dual_freeze,
        )
    test_M = eval_epoch(
        model, test_loader, device,
        epoch=best_ckpt_epoch or epochs,
        freeze_epochs=freeze_fusion,
        dual_freeze=dual_freeze,
        alpha_scale_override=best_alpha,
    )

    config.USE_KD = orig_kd
    config.LOSS_CONTRA_WEIGHT = orig_contra
    return test_M, ckpt_path


def ablation_forward(model, batch, variant_name: str, device: torch.device) -> dict:
    from ablation_routing import resolve_variant_name, is_structural_variant
    delta_h = batch.get(
        "delta_hours",
        torch.zeros(batch["img"].shape[0], device=device),
    )
    if is_structural_variant(variant_name) and resolve_variant_name(variant_name) == "w/o_TS":
        return model(
            batch["img"], batch["depth"], batch["trans"],
            None, None, batch["cur_time_feat"], batch["available_steps"],
            delta_hours=delta_h,
        )
    return model(
        batch["img"], batch["depth"], batch["trans"],
        batch["seq_feat"], batch["time_feat_seq"],
        batch["cur_time_feat"], batch["available_steps"],
        delta_hours=delta_h,
    )


def _load_stage2_img_weights(model, device, dataset_name: str = "frosi"):
    stage2_path = _resolve_stage2_ckpt(dataset_name)
    if not os.path.exists(stage2_path):
        return
    ckpt = load_checkpoint(stage2_path, map_location=device)
    state = ckpt.get("img_branch_state", {})
    targets = []
    if hasattr(model, "img_branch"):
        targets.append(model.img_branch)
    if hasattr(model, "model") and hasattr(model.model, "img_branch"):
        targets.append(model.model.img_branch)
    for t in targets:
        skipped, missing, _ = safe_load_state_dict(t, state)
        if skipped:
            print(f"[权重] 结构消融图像分支跳过 {len(skipped)} 个形状不匹配键")


def train_structural_ablation(
    variant_name: str,
    dataset_name: str,
    aligner,
    device,
    epochs=15,
    lr=None,
    batch_size=None,
    accum=4,
    contra_weight=None,
    ckpt_suffix=None,
):
    """结构消融（w/o_TS, w/o_Gate）：简化单阶段训练，仅加载 Stage2 图像权重。

    与 Full 等变体的 train_stage3() 三阶段协议不同，仅作结构验证用途；
    对比实验时请结合文档说明训练协议差异。
    """
    from models.ablations import build_ablation_model, _resolve_name
    from losses import VisibilityLoss

    set_seed()
    lr = lr or config.MAIN_LR
    batch_size = batch_size or config.BATCH_SIZE
    resolved = _resolve_name(variant_name)

    orig_contra = config.LOSS_CONTRA_WEIGHT
    if contra_weight is not None:
        config.LOSS_CONTRA_WEIGHT = contra_weight

    train_loader, val_loader, test_loader = build_all_loaders(
        dataset_name, aligner, batch_size
    )
    model = build_ablation_model(resolved).to(device)
    _load_stage2_img_weights(model, device, dataset_name)

    criterion = VisibilityLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.MAIN_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler()

    suffix = _sanitize_ckpt_suffix(variant_name, ckpt_suffix)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"ablation_{suffix}_{dataset_name}.pth")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_mae = -1.0
    best_mae_0_3 = -1.0
    tie_tol = getattr(config, "STAGE3_BEST_MAE_TIE_TOL", 50.0)

    print(f"[结构消融] 变体={resolved} 模型={type(model).__name__}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        optimizer.zero_grad()
        for step, raw in enumerate(train_loader):
            batch = to_device_batch(raw, device)
            with autocast():
                out = ablation_forward(model, batch, variant_name, device)
                ld = criterion(out, batch, epoch)
                loss = ld["total"] / accum
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += ld["total"].item() * len(batch["vis_cls"])
            n += len(batch["vis_cls"])
        scheduler.step()
        M = eval_epoch(
            model, val_loader, device, epoch=999, freeze_epochs=0,
            variant_name=variant_name,
        )
        print(f"  E{epoch:3d} loss={total_loss/max(n,1):.4f} ACC={M['acc']:.4f} MAE={M['mae']:.1f}")
        val_mae = M["mae"]
        mae_0_3 = M.get("mae_cls_0_3", -1)
        if is_better_val_mae_ckpt(val_mae, mae_0_3, best_mae, best_mae_0_3, tie_tol):
            best_mae = val_mae
            best_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_mae_0_3
            torch.save({
                "model_state": model.state_dict(),
                "variant": resolved,
                "epoch": epoch,
                "alpha_ramp": 1.0,
            }, ckpt_path)

    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        safe_load_state_dict(model, ckpt["model_state"])
        best_ckpt_epoch = int(ckpt.get("epoch", epochs))
        best_alpha = float(ckpt.get("alpha_ramp", 1.0))
    else:
        best_ckpt_epoch = epochs
        best_alpha = 1.0
    test_M = eval_epoch(
        model, test_loader, device,
        epoch=best_ckpt_epoch,
        freeze_epochs=0,
        variant_name=variant_name,
        alpha_scale_override=best_alpha,
    )
    config.LOSS_CONTRA_WEIGHT = orig_contra
    return test_M, ckpt_path


def save_ablation_result(results_path: str, dataset: str, variant: str, metrics: dict):
    from experiments.experiment_protocol import metrics_to_result_dict
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    data = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault(dataset, {})[variant] = metrics_to_result_dict(metrics)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[保存] {results_path} -> {dataset}/{variant}")
