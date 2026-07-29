"""
阶段二: 图像分支独立训练
- 时序分支完全冻结
- 图像分支直接接 EDL 分类/回归头
- 输入: 带雾图像 + 深度图 + 透射图
- 监督: 能见度等级 (EDL分类) + 能见度数值 (EDL回归)

用法:
  python train_stage2_image.py --dataset frosi --epochs 15
  python train_stage2_image.py --dataset frosi --epochs 10 \\
      --delta-periods 2 6 12 24 --ckpt-suffix TCAM_wo_1h \\
      --init-from checkpoints/stage2_image_frosi.pth
"""
import os, sys, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset_factory import build_all_loaders
from data.sampler import get_class_counts, compute_class_weights
from models.image_branch import ImageBranch
from models.edl import EDLClassificationLoss, EDLRegressionLoss
from evaluate import compute_metrics
from experiments.ablation_common import (
    _stage3_vis_tgt_norm, _stage3_vis_val_to_meter_np, safe_load_state_dict,
)
from utils import to_device_batch, load_checkpoint


def _is_better_stage2_ckpt(val_mae, mae_0_3, best_mae, best_mae_0_3, tie_tol):
    """全局 MAE 为主；接近时以 0-3 类 MAE 决胜。"""
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


def set_seed(seed=config.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------
# 独立图像模型 (只有图像分支 + 两个EDL头, 无时序/融合)
# 阶段三会把训好的权重迁移到完整模型
# ----------------------------------------------------------------
class ImageOnlyModel(nn.Module):
    def __init__(self, pretrained=True, delta_periods=None):
        super().__init__()
        self.img_branch = ImageBranch(
            feat_dim=config.IMAGE_FEAT_DIM,
            num_classes=config.NUM_VIS_CLASSES,
            pretrained=pretrained,
            delta_periods=delta_periods,
        )

    def forward(self, img, depth, trans, cur_time_feat=None, delta_hours=None):
        feat, cls_out, reg_out, _tcam_out = self.img_branch(
            img, depth, trans,
            cur_time_feat=cur_time_feat,
            delta_hours=delta_hours,
        )
        return {"feat": feat, "cls_out": cls_out, "reg_out": reg_out}


def get_loss_fn():
    return (
        EDLClassificationLoss(
            num_classes=config.NUM_VIS_CLASSES,
            annealing_start=config.EDL_ANNEAL_START,
            annealing_step=config.EDL_ANNEAL_STEP,
        ),
        EDLRegressionLoss(coeff=config.EDL_REG_COEFF),
    )


def train_epoch(model, loader, optimizer, scaler, device, edl_cls, edl_reg, epoch,
                accum=4, class_weights=None):
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad()

    for step, raw in enumerate(loader):
        b = to_device_batch(raw, device)

        with autocast():
            out = model(
                b["img"], b["depth"], b["trans"],
                cur_time_feat=b.get("cur_time_feat"),
                delta_hours=b.get("delta_hours"),
            )
            loss = torch.tensor(0.0, device=device)

            valid_cls = b["vis_cls"] >= 0
            if valid_cls.any():
                d = edl_cls(
                    out["cls_out"]["alpha"][valid_cls],
                    b["vis_cls"][valid_cls],
                    epoch,
                    class_weights=class_weights,
                )
                loss = loss + 0.1 * d["total"]

            valid_reg = b["vis_val"] >= 0
            if valid_reg.any():
                ro = out["reg_out"]
                vis_val_norm = _stage3_vis_tgt_norm(b["vis_val"])[valid_reg]
                d2 = edl_reg(
                    ro["gamma"][valid_reg],
                    ro["nu"][valid_reg],
                    ro["alpha"][valid_reg],
                    ro["beta"][valid_reg],
                    vis_val_norm,
                )
                loss = loss + d2["total"]

            loss = loss / accum

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

        total_loss += loss.item() * accum * len(b["vis_cls"])
        n += len(b["vis_cls"])

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    pred_cls_all, gt_cls_all = [], []
    pred_val_all, gt_val_all = [], []
    max_vis = float(getattr(config, "MAX_VIS", 50000.0))

    for raw in loader:
        b = to_device_batch(raw, device)
        out = model(
            b["img"], b["depth"], b["trans"],
            cur_time_feat=b.get("cur_time_feat"),
            delta_hours=b.get("delta_hours"),
        )

        pred_cls = out["cls_out"]["prob"].argmax(-1).cpu().numpy()
        gt_cls = b["vis_cls"].cpu().numpy()

        pred_val_norm = out["reg_out"]["gamma"].squeeze(-1).cpu().numpy()
        pred_val = pred_val_norm * max_vis

        gt_raw = b["vis_val"].cpu().numpy()
        gt_val = _stage3_vis_val_to_meter_np(gt_raw, max_vis)

        pred_cls_all.extend(pred_cls.tolist())
        gt_cls_all.extend(gt_cls.tolist())
        pred_val_all.extend(pred_val.tolist())
        gt_val_all.extend(gt_val.tolist())

    return compute_metrics(
        np.array(gt_val_all), np.array(pred_val_all),
        np.array(gt_cls_all), np.array(pred_cls_all),
    )


def train_stage2_image(
    dataset_name: str,
    epochs: int,
    device,
    delta_periods=None,
    ckpt_path: str = None,
    init_ckpt_path: str = None,
    lr: float = 1e-4,
    batch_size: int = None,
    accum: int = 4,
    mae_tie_tol: float = None,
    log_suffix: str = "",
):
    """训练图像分支并保存 checkpoint；供 CLI 与 TCAM 消融 runner 调用。"""
    config.USE_METER_LABELS = True
    batch_size = batch_size or config.BATCH_SIZE
    mae_tie_tol = mae_tie_tol if mae_tie_tol is not None else config.STAGE2_BEST_MAE_TIE_TOL
    ckpt_path = ckpt_path or config.stage2_ckpt_path(dataset_name)

    periods_label = delta_periods if delta_periods is not None else "default"
    print(f"[阶段二-图像训练] 数据集={dataset_name}  periods={periods_label}  "
          f"设备={device}{log_suffix}")

    train_loader, val_loader, test_loader = build_all_loaders(
        dataset_name, aligner=None, batch_size=batch_size,
    )

    class_weights = None
    if config.CLASS_WEIGHT_MODE != "none":
        counts = get_class_counts(train_loader.dataset)
        class_weights = compute_class_weights(
            counts, mode=config.CLASS_WEIGHT_MODE, device=device,
        )

    model = ImageOnlyModel(pretrained=True, delta_periods=delta_periods).to(device)

    if init_ckpt_path and os.path.exists(init_ckpt_path):
        ckpt_init = load_checkpoint(init_ckpt_path, map_location=device)
        img_state = ckpt_init.get("img_branch_state", {})
        skipped, missing, _ = safe_load_state_dict(model.img_branch, img_state)
        n_loaded = len(img_state) - len(skipped)
        print(f"[阶段二] warm-start ← {init_ckpt_path} "
              f"(加载={n_loaded}, 跳过={len(skipped)}, 缺失={len(missing)})")
        if skipped:
            preview = ", ".join(skipped[:4])
            if len(skipped) > 4:
                preview += "..."
            print(f"  跳过键(形状不匹配): {preview}")

    edl_cls, edl_reg = get_loss_fn()
    optimizer = AdamW(model.parameters(), lr=lr,
                      weight_decay=config.MAIN_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler()

    log_name = f"stage2_image{log_suffix.replace(' ', '_')}.csv"
    log_path = os.path.join(config.LOG_DIR, log_name)
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_acc,val_recall,val_mae,val_mse,"
                "val_mae_0_3,val_mae_4_5\n")

    best_mae = -1.0
    best_mae_0_3 = -1.0
    print("[阶段二] 开始训练 (时序分支冻结)")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(
            model, train_loader, optimizer, scaler,
            device, edl_cls, edl_reg, epoch, accum,
            class_weights=class_weights,
        )
        metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        val_mae = metrics.get("mae", -1.0)
        mae_0_3 = metrics.get("mae_cls_0_3", -1.0)
        mae_4_5 = metrics.get("mae_cls_4_5", -1.0)
        mae_0_3_s = f"{mae_0_3:.1f}" if mae_0_3 >= 0 else "—"
        mae_4_5_s = f"{mae_4_5:.1f}" if mae_4_5 >= 0 else "—"
        print(f"  E{epoch:3d} loss={tr_loss:.4f}  "
              f"ACC={metrics['acc']:.4f}  Recall={metrics['recall']:.4f}  "
              f"MAE={metrics['mae']:.1f}  MAE_0-3={mae_0_3_s}  "
              f"MAE_4-5={mae_4_5_s}  [{time.time()-t0:.1f}s]")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{metrics['acc']:.4f},"
                    f"{metrics['recall']:.4f},{metrics['mae']:.3f},"
                    f"{metrics['mse']:.3f},{mae_0_3:.3f},{mae_4_5:.3f}\n")

        if _is_better_stage2_ckpt(
            val_mae, mae_0_3, best_mae, best_mae_0_3, mae_tie_tol,
        ):
            best_mae = val_mae
            best_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_mae_0_3
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "img_branch_state": model.img_branch.state_dict(),
                "metrics": metrics,
                "dataset": dataset_name,
                "delta_periods": delta_periods,
            }, ckpt_path)
            mae_0_3_best_s = (
                f"{best_mae_0_3:.1f}m" if best_mae_0_3 >= 0 else "—"
            )
            print(f"    -> 保存最佳图像分支 (MAE={best_mae:.1f}m, "
                  f"MAE_0-3={mae_0_3_best_s})")

    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.img_branch.load_state_dict(ckpt["img_branch_state"])
        test_m = eval_epoch(model, test_loader, device)
        test_mae_0_3 = test_m.get("mae_cls_0_3", -1.0)
        test_mae_4_5 = test_m.get("mae_cls_4_5", -1.0)
        test_mae_0_3_s = f"{test_mae_0_3:.1f}" if test_mae_0_3 >= 0 else "—"
        test_mae_4_5_s = f"{test_mae_4_5:.1f}" if test_mae_4_5 >= 0 else "—"
        print(f"\n[阶段二-测试] ACC={test_m['acc']:.4f}  "
              f"Recall={test_m['recall']:.4f}  MAE={test_m['mae']:.1f}  "
              f"MAE_0-3={test_mae_0_3_s}  MAE_4-5={test_mae_4_5_s}")
        print(f"图像分支权重已保存: {ckpt_path}")
    else:
        print(f"[警告] 未保存任何 checkpoint: {ckpt_path}")

    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="frosi",
                        choices=["frosi"])
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--accum",      type=int,   default=4)
    parser.add_argument("--no_cuda",    action="store_true")
    parser.add_argument("--mae_tie_tol", type=float,
                        default=config.STAGE2_BEST_MAE_TIE_TOL,
                        help="全局 MAE 接近时的决胜阈值（米）")
    parser.add_argument("--delta-periods", nargs="+", type=float, default=None,
                        help="TCAM 周期列表，如 1 2 6 12 24")
    parser.add_argument("--ckpt-suffix", default=None,
                        help="checkpoint 变体后缀，如 TCAM_wo_1h")
    parser.add_argument("--init-from", default=None,
                        help="warm-start 来源 checkpoint（如 full Stage2）")
    args = parser.parse_args()

    set_seed()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    ckpt_path = config.stage2_ckpt_path(args.dataset, suffix=args.ckpt_suffix)
    log_suffix = f"_{args.ckpt_suffix}" if args.ckpt_suffix else ""

    train_stage2_image(
        dataset_name=args.dataset,
        epochs=args.epochs,
        device=device,
        delta_periods=args.delta_periods,
        ckpt_path=ckpt_path,
        init_ckpt_path=args.init_from,
        lr=args.lr,
        batch_size=args.batch_size,
        accum=args.accum,
        mae_tie_tol=args.mae_tie_tol,
        log_suffix=log_suffix,
    )


if __name__ == "__main__":
    main()
