"""
对比/消融实验与主模型共享的数据、评估、选模协议。

主模型基准：train_stage3_joint --dual_freeze --freeze_fusion 5 --epochs 15
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset_factory import build_all_loaders
from data.sampler import get_class_counts, compute_class_weights
from experiments.ablation_common import (
    is_better_val_mae_ckpt,
    _stage3_vis_tgt_norm,
    _stage3_vis_val_to_meter_np,
)

MAIN_FREEZE_FUSION = int(getattr(config, "MAIN_FREEZE_FUSION", 5))
MAIN_STAGE3_EPOCHS = int(getattr(config, "MAIN_STAGE3_EPOCHS", 15))
MAIN_DUAL_FREEZE = bool(getattr(config, "MAIN_DUAL_FREEZE", True))
BASELINE_BEST_MAE_TIE_TOL = float(
    getattr(config, "BASELINE_BEST_MAE_TIE_TOL", None)
    or getattr(config, "STAGE3_BEST_MAE_TIE_TOL", 50.0)
)


def setup_meter_labels():
    config.USE_METER_LABELS = True


def build_protocol_loaders(dataset_name, aligner, batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    return build_all_loaders(dataset_name, aligner, batch_size)


def build_protocol_class_weights(train_loader, device):
    if config.CLASS_WEIGHT_MODE == "none":
        return None
    counts = get_class_counts(train_loader.dataset)
    return compute_class_weights(
        counts, mode=config.CLASS_WEIGHT_MODE, device=device,
    )


def select_best_ckpt(val_mae, mae_0_3, best_mae, best_mae_0_3, tie_tol=None):
    tol = tie_tol if tie_tol is not None else BASELINE_BEST_MAE_TIE_TOL
    return is_better_val_mae_ckpt(val_mae, mae_0_3, best_mae, best_mae_0_3, tol)


def tgt_vis_norm(vis_val):
    return _stage3_vis_tgt_norm(vis_val)


def gt_vis_to_meter_np(vis_val_np, max_vis=None):
    max_vis = float(max_vis or getattr(config, "MAX_VIS", 50000.0))
    return _stage3_vis_val_to_meter_np(vis_val_np, max_vis)


def pred_norm_to_meter_np(pred_norm_np, max_vis=None):
    max_vis = float(max_vis or getattr(config, "MAX_VIS", 50000.0))
    return pred_norm_np * max_vis


def metrics_to_result_dict(metrics):
    """统一 JSON 保存字段（基线 / 消融）。"""
    return {
        "acc": round(float(metrics.get("acc", -1)), 4),
        "recall": round(float(metrics.get("recall", -1)), 4),
        "mae": round(float(metrics.get("mae", -1)), 3),
        "mse": round(float(metrics.get("mse", -1)), 3),
        "mae_cls_0_3": round(float(metrics.get("mae_cls_0_3", -1)), 3),
        "mae_cls_4_5": round(float(metrics.get("mae_cls_4_5", -1)), 3),
        "per_class_recall": metrics.get("per_class_recall", []),
    }
