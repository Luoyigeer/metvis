"""
阶段三: 联合微调 (UGG + TVKD 版)
"""
import os
import sys
import time
import argparse

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset_factory import build_all_loaders
from data.noaa_dataset import NOAADataProcessor
from data.image_datasets import TimeAligner
from data.sampler import get_class_counts, compute_class_weights
from experiments.ablation_common import (
    Stage3Loss, TimeReconHead, build_stage3_model, build_aligner,
    train_epoch, eval_epoch, safe_load_state_dict, set_seed, set_stage3_freeze_state,
    is_better_val_mae_ckpt, FUSION_TRANSITION_EPOCHS,
    _stage3_alpha_schedule,
)
from experiments.save_result import save_main_result
from models.visibility_model import VisibilityModel
from utils import load_checkpoint

STAGE3_CKPT = config.STAGE3_CKPT


def build_stage3_ckpt_path(args) -> str:
    suffix = []
    if getattr(args, "dual_freeze", False):
        suffix.append("dualFreeze")
    if getattr(args, "no_unc_gate", False):
        suffix.append("noUGG")
    if getattr(args, "no_kd", False):
        suffix.append("noKD")
    if getattr(args, "no_time_align", False):
        suffix.append("noAlign")
    if getattr(args, "no_noaa_pretrain", False):
        suffix.append("noNOAA")
    if getattr(args, "dataset", None):
        suffix.append(str(args.dataset))
    name = "stage3_joint"
    if suffix:
        name += "_" + "_".join(suffix)
    return os.path.join(config.CHECKPOINT_DIR, f"{name}.pth")


def build_stage3_dual_ckpt_paths(args):
    """dual_freeze 分阶段 ckpt：_io（UGG 前）与 _ugg（UGG 阶段）。"""
    base = build_stage3_ckpt_path(args)
    root, ext = os.path.splitext(base)
    return root + "_io" + ext, root + "_ugg" + ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="frosi",
                        choices=["frosi"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=config.MAIN_LR)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--w_recon", type=float, default=0.1)
    parser.add_argument("--ts_lr_ratio", type=float, default=0.05)
    parser.add_argument("--freeze_fusion", type=int, default=5)
    parser.add_argument("--dual_freeze", action="store_true",
                        help="冻结图像+时序双塔，仅训融合与 NT-Xent 投影对齐")
    parser.add_argument("--no_unc_gate", action="store_true")
    parser.add_argument("--no_kd", action="store_true")
    parser.add_argument("--no_noaa_pretrain", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_time_align", action="store_true")
    parser.add_argument("--skip_align_check", action="store_true")
    parser.add_argument("--mae_tie_tol", type=float,
                        default=config.STAGE3_BEST_MAE_TIE_TOL,
                        help="全局 MAE 接近时的决胜阈值（米）")
    args = parser.parse_args()

    config.USE_METER_LABELS = True
    if args.no_unc_gate:
        config.USE_UNC_GATE = False
        print("[消融] 禁用 UGG")
    if args.no_kd:
        config.USE_KD = False
        print("[消融] 禁用 TVKD")

    set_seed()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    mode_name = "dual_freeze" if args.dual_freeze else "默认联合微调"
    print(f"[阶段三] 数据集={args.dataset}  设备={device}  模式={mode_name}")

    aligner = build_aligner(args.dataset, no_time_align=args.no_time_align)

    if not args.skip_align_check and aligner is not None:
        from data.dataset_factory import build_loader
        _check_loader = build_loader(args.dataset, "train", aligner, batch_size=16, num_workers=0)
        _b = next(iter(_check_loader))
        _avail = _b["available_steps"]
        print(f"[诊断] 时序对齐率: {(_avail > 0).float().mean().item()*100:.1f}%")
        del _check_loader, _b

    train_loader, val_loader, test_loader = build_all_loaders(
        args.dataset, aligner, args.batch_size
    )

    class_weights = None
    if config.CLASS_WEIGHT_MODE != "none":
        counts = get_class_counts(train_loader.dataset)
        class_weights = compute_class_weights(counts, mode=config.CLASS_WEIGHT_MODE)

    model_kwargs = {}
    if args.no_unc_gate:
        model_kwargs["use_unc_gate"] = False

    model = build_stage3_model(
        device, args.dataset,
        load_noaa_pretrain=not args.no_noaa_pretrain,
        load_stage2=True,
        model_kwargs=model_kwargs or None,
    )
    recon_hd = TimeReconHead(config.TIME_FEAT_DIM, 10).to(device)

    criterion = Stage3Loss(w_recon=args.w_recon, class_weights=class_weights).to(device)
    schedulers = []

    if args.dual_freeze:
        opt_fusion = AdamW(
            model.fusion.parameters(), lr=args.lr, weight_decay=config.MAIN_WEIGHT_DECAY,
        )
        opt_align = AdamW(
            criterion.contra.parameters(), lr=args.lr, weight_decay=config.MAIN_WEIGHT_DECAY,
        )
        optimizers = [opt_fusion, opt_align]
        schedulers = [
            CosineAnnealingLR(opt_fusion, T_max=args.epochs, eta_min=args.lr * 0.01),
            CosineAnnealingLR(opt_align, T_max=args.epochs, eta_min=args.lr * 0.01),
        ]
    else:
        img_params = (
            list(model.img_branch.parameters()) + list(model.fusion.parameters())
        )
        ts_params = list(model.ts_branch.parameters())
        recon_params = list(recon_hd.parameters())
        opt_img = AdamW(img_params, lr=args.lr, weight_decay=config.MAIN_WEIGHT_DECAY)
        opt_ts = AdamW(
            ts_params, lr=args.lr * args.ts_lr_ratio,
            weight_decay=config.MAIN_WEIGHT_DECAY,
        )
        opt_recon = AdamW(recon_params, lr=args.lr, weight_decay=config.MAIN_WEIGHT_DECAY)
        opt_align = AdamW(
            criterion.contra.parameters(), lr=args.lr, weight_decay=config.MAIN_WEIGHT_DECAY,
        )
        optimizers = [opt_img, opt_ts, opt_recon, opt_align]
        schedulers = [
            CosineAnnealingLR(opt_img, T_max=args.epochs, eta_min=args.lr * 0.01),
            CosineAnnealingLR(opt_ts, T_max=args.epochs,
                              eta_min=args.lr * args.ts_lr_ratio * 0.01),
            CosineAnnealingLR(opt_recon, T_max=args.epochs, eta_min=args.lr * 0.01),
            CosineAnnealingLR(opt_align, T_max=args.epochs, eta_min=args.lr * 0.01),
        ]

    scaler = GradScaler()
    ckpt_path = build_stage3_ckpt_path(args)
    if args.dual_freeze:
        ckpt_io_path, ckpt_ugg_path = build_stage3_dual_ckpt_paths(args)
    best_mae = -1.0
    best_mae_0_3 = -1.0
    best_io_mae = -1.0
    best_io_mae_0_3 = -1.0
    best_ugg_mae = -1.0
    best_ugg_mae_0_3 = -1.0
    best_ckpt_epoch = -1
    best_ckpt_ugg = False

    M0 = eval_epoch(
        model, val_loader, device, epoch=0,
        freeze_epochs=args.freeze_fusion, head="img",
    )
    print(f"[诊断] 训练前验证(图像头): ACC={M0['acc']:.4f} MAE={M0['mae']:.1f}")

    ugg_start = args.freeze_fusion + FUSION_TRANSITION_EPOCHS

    for epoch in range(1, args.epochs + 1):
        freeze = epoch <= args.freeze_fusion
        set_stage3_freeze_state(
            model, epoch, args.freeze_fusion, dual_freeze=args.dual_freeze,
        )
        t0 = time.time()
        L, n_skip, n_batches = train_epoch(
            model, recon_hd, train_loader, optimizers,
            scaler, device, criterion, epoch, args.accum, freeze=freeze,
            freeze_fusion_epochs=args.freeze_fusion,
            aligner=aligner, inject_delta=True, dual_freeze=args.dual_freeze,
        )
        M = eval_epoch(
            model, val_loader, device, epoch=epoch,
            freeze_epochs=args.freeze_fusion, dual_freeze=args.dual_freeze,
        )
        if epoch == args.freeze_fusion:
            M_img_ref = eval_epoch(
                model, val_loader, device, epoch=epoch,
                freeze_epochs=args.freeze_fusion, head="img",
            )
            M_fused = eval_epoch(
                model, val_loader, device, epoch=epoch,
                freeze_epochs=args.freeze_fusion,
                force_img_only=True,
                head="fused",
            )
            print(
                f"  [诊断] 融合头就绪: img_ACC={M_img_ref['acc']:.4f} "
                f"fused_ACC={M_fused['acc']:.4f} fused_MAE={M_fused['mae']:.1f}"
            )
        if epoch == ugg_start and args.dual_freeze:
            M_preflight = eval_epoch(
                model, val_loader, device, epoch=epoch,
                freeze_epochs=args.freeze_fusion, dual_freeze=True,
                alpha_scale_override=0.1,
            )
            ramp_now = _stage3_alpha_schedule(
                epoch, args.freeze_fusion, dual_freeze=True,
            )
            gate_mean = M_preflight.get("gate_alpha_mean", -1.0)
            gate_s = f"{gate_mean:.4f}" if gate_mean >= 0 else "—"
            print(
                f"  [诊断] UGG 预检(E{epoch} ramp={ramp_now:.2f}): "
                f"alpha=0.1 ACC={M_preflight['acc']:.4f} "
                f"MAE={M_preflight['mae']:.1f} gate_alpha_mean={gate_s}"
            )
        if epoch == ugg_start + 1:
            M_img = eval_epoch(
                model, val_loader, device, epoch=epoch,
                freeze_epochs=args.freeze_fusion, head="img",
            )
            print(
                f"  [诊断] UGG 全开: img_ACC={M_img['acc']:.4f} "
                f"fused_UGG_ACC={M['acc']:.4f} fused_MAE={M['mae']:.1f}"
            )
        for sch in schedulers:
            sch.step()
        if args.dual_freeze:
            ramp = _stage3_alpha_schedule(
                epoch, args.freeze_fusion, dual_freeze=True,
            )
            head_tag = f"fused-a{ramp:.2f}"
        else:
            ramp = 0.0
            head_tag = "fused" if epoch > ugg_start else "fused-io"
        mae_0_3 = M.get("mae_cls_0_3", -1)
        mae_0_3_s = f"{mae_0_3:.1f}" if mae_0_3 >= 0 else "—"
        print(
            f"  E{epoch:3d} tot={L['total']:.4f} fuse_d={L['fuse_distill']:.6f} "
            f"sup_c={L['fuse_sup_cls']:.6f} sup_r={L['fuse_sup_reg']:.6f} "
            f"reg_edl={L.get('fuse_reg_edl', 0):.6f} "
            f"contra={L['contra']:.6f} kd={L['kd']:.3f} gate={L['gate_reg']:.3f} "
            f"gate_pw={L.get('gate_prewarm', 0):.4f} gate_a={L.get('gate_anchor', 0):.4f} "
            f"blend_a={L.get('blend_anchor', 0):.4f} blend_r={L.get('blend_reg_anchor', 0):.4f} "
            f"gamma_d={L.get('gamma_distill', 0):.4f} skip={n_skip}/{n_batches} "
            f"[{head_tag}] ACC={M['acc']:.4f} MAE={M['mae']:.1f} "
            f"MAE_0-3={mae_0_3_s} [{time.time()-t0:.1f}s]"
        )
        val_mae = M["mae"]
        unique_pred = int(M.get("unique_pred_classes", config.NUM_VIS_CLASSES))
        in_ugg_phase = epoch > ugg_start
        can_save = unique_pred >= 2 if in_ugg_phase else True
        if not can_save:
            print(
                f"  [警告] 跳过保存 best: 融合预测仅 {unique_pred} 个类别 "
                f"(分布={M.get('pred_cls_dist', [])})"
            )
        elif is_better_val_mae_ckpt(
            val_mae, mae_0_3,
            best_ugg_mae if (args.dual_freeze and in_ugg_phase)
            else best_io_mae if args.dual_freeze
            else best_mae,
            best_ugg_mae_0_3 if (args.dual_freeze and in_ugg_phase)
            else best_io_mae_0_3 if args.dual_freeze
            else best_mae_0_3,
            args.mae_tie_tol,
        ):
            if args.dual_freeze and in_ugg_phase:
                best_ugg_mae = val_mae
                best_ugg_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_ugg_mae_0_3
                save_path = ckpt_ugg_path
                phase_tag = "UGG"
            elif args.dual_freeze:
                best_io_mae = val_mae
                best_io_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_io_mae_0_3
                save_path = ckpt_io_path
                phase_tag = "IO"
            else:
                best_mae = val_mae
                best_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_mae_0_3
                save_path = ckpt_path
                phase_tag = ""
            best_mae = val_mae
            best_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_mae_0_3
            best_ckpt_epoch = epoch
            best_ckpt_ugg = in_ugg_phase
            save_dict = {
                "epoch": epoch, "model_state": model.state_dict(),
                "metrics": M, "args": vars(args),
                "is_ugg_phase": in_ugg_phase,
                "alpha_ramp": float(ramp) if args.dual_freeze else None,
            }
            if not args.dual_freeze:
                save_dict["recon_state"] = recon_hd.state_dict()
            torch.save(save_dict, save_path)
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "metrics": M}, config.INFER_CHECKPOINT)
            mae_0_3_best_s = (
                f"{best_mae_0_3:.1f}m" if best_mae_0_3 >= 0 else "—"
            )
            phase_s = f" ({phase_tag})" if phase_tag else ""
            print(f"    -> 保存最佳融合模型{phase_s} (MAE={best_mae:.1f}m, "
                  f"MAE_0-3={mae_0_3_best_s})")

    test_ckpt_path = ckpt_path
    if args.dual_freeze:
        if os.path.exists(ckpt_ugg_path):
            test_ckpt_path = ckpt_ugg_path
        elif os.path.exists(ckpt_io_path):
            test_ckpt_path = ckpt_io_path
            print(
                "\n[警告] 无 UGG 阶段 best ckpt，回退至 IO 阶段 ckpt；"
                "测试使用 ckpt 保存的 alpha_ramp"
            )
    if os.path.exists(test_ckpt_path):
        ckpt = load_checkpoint(test_ckpt_path, map_location=device)
        safe_load_state_dict(model, ckpt["model_state"])
        ckpt_epoch = int(ckpt.get("epoch", best_ckpt_epoch))
        test_alpha = float(ckpt["alpha_ramp"]) if ckpt.get("alpha_ramp") is not None else _stage3_alpha_schedule(
            ckpt_epoch, args.freeze_fusion, dual_freeze=args.dual_freeze,
        )
        test_M = eval_epoch(
            model, test_loader, device,
            epoch=ckpt_epoch,
            freeze_epochs=args.freeze_fusion,
            dual_freeze=args.dual_freeze,
            alpha_scale_override=test_alpha,
        )
        print(f"\n[测试] ACC={test_M['acc']:.4f} MAE={test_M['mae']:.1f} "
              f"(alpha={test_alpha:.2f})")
        save_main_result(args.dataset, test_M)


if __name__ == "__main__":
    main()
