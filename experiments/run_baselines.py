"""
Baseline training/evaluation on FROSI.

Image baselines: ResNet / DenseNet / AlexNet / VisNet / CLIP / OrdinalCLIP
Temporal baselines: LSTM / Transformer / TimesNet (skipped on FROSI — no real capture time)
Multimodal baselines: MM_EarlyFusion / MM_LateFusion

Usage:
  python experiments/run_baselines.py --dataset frosi
  python experiments/run_baselines.py --dataset frosi --models ResNet DenseNet
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluate import compute_metrics
from models.losses import CombinedLoss
from experiments.ablation_common import build_aligner, is_better_val_mae_ckpt
from experiments.experiment_protocol import (
    setup_meter_labels,
    build_protocol_loaders,
    build_protocol_class_weights,
    gt_vis_to_meter_np,
    tgt_vis_norm,
    metrics_to_result_dict,
    BASELINE_BEST_MAE_TIE_TOL,
    MAIN_STAGE3_EPOCHS,
)
from utils import to_device_batch, load_checkpoint

RESULTS_DIR = os.path.join(config.LOG_DIR, "baseline_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_MODELS   = ["ResNet", "DenseNet", "AlexNet", "VisNet", "CLIP", "OrdinalCLIP"]
TEMPORAL_MODELS= ["LSTM", "Transformer", "TimesNet"]
MULTIMODAL_MODELS = ["MM_EarlyFusion", "MM_LateFusion", "MSTFNet", "Kopecka"]
ALL_MODELS     = IMAGE_MODELS + TEMPORAL_MODELS + MULTIMODAL_MODELS

# FROSI 不做时序基线（无时间信息）
FROSI_SKIP_TS  = True

# 缓存 MAX_VIS 避免重复读取
_MAX_VIS = float(getattr(config, "MAX_VIS", 50000.0))


def _to_meter_pred(pred_norm: torch.Tensor) -> torch.Tensor:
    """将模型输出的归一化预测值 [0,1] 转换为米制"""
    return pred_norm * _MAX_VIS


def _json_safe_metrics(obj):
    """Recursively replace NaN/Inf and numpy scalars with JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _json_safe_metrics(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_metrics(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ================================================================
# 模型工厂
# ================================================================
def build_model(name: str, d_in: int = config.NUM_METEO_FEATURES + 1):
    if name == "ResNet":
        from models.baselines.resnet import ResNet
        return ResNet(variant="resnet50")
    elif name == "DenseNet":
        from models.baselines.densenet import DenseNetVis
        return DenseNetVis()
    elif name == "AlexNet":
        from models.baselines.alexnet import AlexNetVis
        return AlexNetVis()
    elif name == "VisNet":
        from models.baselines.visnet import VisNet
        return VisNet()
    elif name == "CLIP":
        from models.baselines.clip import CLIPVis
        return CLIPVis()
    elif name == "OrdinalCLIP":
        from models.baselines.ordinalclip import OrdinalCLIP
        return OrdinalCLIP()
    elif name == "LSTM":
        from models.baselines.lstm import LSTMVis
        return LSTMVis(d_in=d_in)
    elif name == "Transformer":
        from models.baselines.transformer import TransformerVis
        return TransformerVis(d_in=d_in)
    elif name == "TimesNet":
        from models.temporal_branch import TimesNetBranch as TimesNetBaseline
        return TimesNetBaseline(d_in=d_in)
    elif name == "MM_EarlyFusion":
        from models.baselines.multimodal_fusion import EarlyFusionBaseline
        return EarlyFusionBaseline()
    elif name == "MM_LateFusion":
        from models.baselines.multimodal_fusion import LateFusionBaseline
        return LateFusionBaseline()
    elif name == "MSTFNet":
        from models.baselines.mstf_fusion import MSTFVisibility
        return MSTFVisibility()
    elif name == "Kopecka":
        from models.baselines.kopecka_fusion import KopeckaVisibility
        return KopeckaVisibility()
    else:
        raise ValueError(f"未知模型: {name}")


# ================================================================
# 前向接口统一
# ================================================================
def vis_to_class(vis_m: float) -> int:
    """将能见度值（米）转换为等级"""
    bins = config.VIS_BINS
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if low <= vis_m < high:
            return i
    return len(bins) - 2


def is_ts_model(name):
    return name in TEMPORAL_MODELS

def is_mm_model(name): 
    return name in MULTIMODAL_MODELS

def forward_model(model, batch, name, device):
    """统一前向传播，返回 cls_logits 和 vis_pred（归一化值）"""
    if is_ts_model(name):
        # 时序模型：输入气象时序，输出预测值
        feat, vis_reg, _ = model(
            batch["seq_feat"], batch["time_feat_seq"], batch["cur_time_feat"]
        )
        # vis_reg 可能是 dict 或 tensor
        if isinstance(vis_reg, dict):
            pred_val = vis_reg["vis_pred"]
        else:
            pred_val = vis_reg
        # 时序模型没有分类头，从预测值反推等级
        pred_val_norm = pred_val.squeeze(-1).clamp(0.0, 1.0)
        pred_meter = _to_meter_pred(pred_val_norm)
        # 计算预测等级
        pred_cls_list = [vis_to_class(float(v)) for v in pred_meter.cpu()]
        cls_logits = torch.zeros(len(pred_cls_list), config.NUM_VIS_CLASSES, device=device)
        cls_logits.scatter_(1, torch.tensor(pred_cls_list, device=device).unsqueeze(1), 10.0)
        return {"cls_logits": cls_logits, "vis_pred": pred_val_norm}
    
    elif is_mm_model(name):
        # 多模态模型：输入图像 + 气象时序
        out = model(batch["img"], batch["seq_feat"])
        return {"cls_logits": out["cls_logits"], "vis_pred": out["vis_pred"]}
    
    else:
        # 图像模型：只输入图像
        out = model(batch["img"])
        return {"cls_logits": out["cls_logits"], "vis_pred": out["vis_pred"]}


def get_loss_with_focal(out, batch, loss_fn):
    """Focal + SmoothL1；回归在 [0,1] 归一化空间监督。"""
    cls_targets = batch["vis_cls"]
    valid_reg = batch["vis_val"] >= 0
    if valid_reg.any():
        pred_reg = out["vis_pred"].squeeze(-1)
        gt_reg = tgt_vis_norm(batch["vis_val"])
    else:
        pred_reg = out["vis_pred"].squeeze(-1)
        gt_reg = batch["vis_val"]

    return loss_fn(
        cls_logits=out["cls_logits"],
        cls_targets=cls_targets,
        pred_vis=pred_reg,
        gt_vis=gt_reg,
    )


# ================================================================
# 训练 / 评估
# ================================================================
def train_one_epoch(model, loader, optimizer, scaler, device, name, loss_fn, accum=4):
    model.train()
    total = 0
    n = 0
    optimizer.zero_grad()
    
    for step, raw in enumerate(loader):
        b = to_device_batch(raw, device)
        
        with autocast():
            out = forward_model(model, b, name, device)
            loss = get_loss_with_focal(out, b, loss_fn) / accum
        
        if not torch.isfinite(loss * accum):
            optimizer.zero_grad()
            continue
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total += loss.item() * accum * len(b["vis_cls"])
        n += len(b["vis_cls"])
    
    return total / max(n, 1)



@torch.no_grad()
def eval_one_epoch(model, loader, device, name):
    model.eval()
    pred_cls = []
    gt_cls = []
    pred_val = []
    gt_val = []
    
    for raw in loader:
        b = to_device_batch(raw, device)
        out = forward_model(model, b, name, device)
        
        # 分类预测
        pred_cls.extend(out["cls_logits"].argmax(-1).cpu().numpy().tolist())
        gt_cls.extend(b["vis_cls"].cpu().numpy().tolist())
        
        # 回归预测：模型输出是归一化的，转回米制
        pred_norm = out["vis_pred"].squeeze(-1).cpu().numpy()
        pred_meter = pred_norm * _MAX_VIS
        pred_val.extend(pred_meter.tolist())
        
        # 真值转米制（与主模型 eval 一致）
        gt_meter = gt_vis_to_meter_np(b["vis_val"].cpu().numpy())
        gt_val.extend(gt_meter.tolist())
    
    gt_val_arr = np.array(gt_val)
    pred_val_arr = np.array(pred_val)
    gt_cls_arr = np.array(gt_cls)
    pred_cls_arr = np.array(pred_cls)
    
    # 过滤无效样本
    valid = (gt_val_arr >= 0) & (gt_cls_arr >= 0)
    if valid.any():
        return compute_metrics(
            gt_val_arr[valid], pred_val_arr[valid],
            gt_cls_arr[valid], pred_cls_arr[valid],
        )
    return compute_metrics(gt_val_arr, pred_val_arr, gt_cls_arr, pred_cls_arr)


# ================================================================
# 单个基线运行
# ================================================================
def run_one(model_name, dataset_name, epochs, lr, batch_size, device, no_time_align=False):
    print(f"\n{'='*55}")
    print(f"  {model_name}  ×  {dataset_name.upper()}")
    print(f"{'='*55}")
    
    # FROSI 不做时序基线，但多模态模型会使用仅图像模式
    if dataset_name == "frosi" and is_ts_model(model_name):
        print("  [跳过] FROSI 无时间信息，不评估时序模型")
        return {"acc": -1, "recall": -1, "mae": -1, "mse": -1}
    
    setup_meter_labels()

    needs_align = (is_ts_model(model_name) or is_mm_model(model_name)) and not no_time_align
    if dataset_name == "frosi":
        aligner = None
    elif needs_align:
        aligner = build_aligner(dataset_name)
    else:
        aligner = build_aligner(dataset_name, no_time_align=True)

    train_loader, val_loader, test_loader = build_protocol_loaders(
        dataset_name, aligner, batch_size,
    )
    print(
        f"  [数据] train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    class_weights = build_protocol_class_weights(train_loader, device)
    if class_weights is not None:
        print(f"[类别权重] mode={config.CLASS_WEIGHT_MODE} {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.ones(config.NUM_VIS_CLASSES, device=device)
    
    # 构建模型
    d_in = config.NUM_METEO_FEATURES + 1
    model = build_model(model_name, d_in).to(device)
    
    # 构建损失函数（使用 Focal Loss）
    loss_fn = CombinedLoss(
        cls_weight=1.0,      # 分类权重
        reg_weight=1.0,      # 回归权重
        focal_gamma=2.0,     # Focal Loss gamma 参数（2.0 是标准值）
        label_smoothing=0.05
    )
    loss_fn.set_class_weights(class_weights)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler()

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"baseline_{model_name}_{dataset_name}.pth")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_mae = -1.0
    best_mae_0_3 = -1.0
    tie_tol = BASELINE_BEST_MAE_TIE_TOL

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, scaler, device, model_name, loss_fn)
        m = eval_one_epoch(model, val_loader, device, model_name)
        sch.step()

        mae_0_3 = m.get("mae_cls_0_3", -1)
        mae_0_3_s = f"{mae_0_3:.1f}" if mae_0_3 >= 0 else "—"
        if "per_class_recall" in m and m["per_class_recall"]:
            per_cls_recall_str = " ".join([f"{x:.3f}" for x in m["per_class_recall"]])
            print(f"  E{epoch:3d} loss={loss:.4f}  "
                  f"ACC={m['acc']:.4f}  Recall={m['recall']:.4f}  "
                  f"MAE={m['mae']:.1f}m MAE_0-3={mae_0_3_s}  "
                  f"[cls_recall={per_cls_recall_str}]")
        else:
            print(f"  E{epoch:3d} loss={loss:.4f}  "
                  f"ACC={m['acc']:.4f}  Recall={m['recall']:.4f}  "
                  f"MAE={m['mae']:.1f}m MAE_0-3={mae_0_3_s} MSE={m['mse']:.1f}  "
                  f"[{time.time()-t0:.1f}s]")

        val_mae = m["mae"]
        if is_better_val_mae_ckpt(val_mae, mae_0_3, best_mae, best_mae_0_3, tie_tol):
            best_mae = val_mae
            best_mae_0_3 = mae_0_3 if mae_0_3 >= 0 else best_mae_0_3
            torch.save({"model_state": model.state_dict(), "metrics": m}, ckpt_path)
    
    # 测试集评估
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    else:
        print(f"  [警告] 未找到最优权重文件，使用当前模型评估测试集")
    
    tm = eval_one_epoch(model, test_loader, device, model_name)
    
    # 打印详细结果
    print(f"  [TEST] ACC={tm['acc']:.4f}  Recall={tm['recall']:.4f}  "
          f"MAE={tm['mae']:.1f}m MSE={tm['mse']:.1f} "
          f"MAE_0-3={tm.get('mae_cls_0_3', -1):.1f}")
    if "per_class_recall" in tm and tm["per_class_recall"]:
        print(f"  [TEST] 每类召回率: {[f'{x:.3f}' for x in tm['per_class_recall']]}")

    return metrics_to_result_dict(tm)

# ================================================================
# 主入口
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="frosi",
                        choices=["frosi"])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--epochs", type=int, default=MAIN_STAGE3_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_time_align", action="store_true",
                        help="禁用多模态基线的时间对齐，使用全零气象输入")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[设备] {device}")
    
    datasets = [args.dataset]
    
    all_results = {}
    for ds in datasets:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {ds.upper()}")
        print(f"{'#'*60}")
        all_results[ds] = {}
        
        for m in args.models:
            out = run_one(m, ds, args.epochs, args.lr, args.batch_size, device,
                          no_time_align=args.no_time_align)
            all_results[ds][m] = out
    
    # 保存结果
    ds_tag = "_".join(datasets)
    out_path = os.path.join(RESULTS_DIR, f"baseline_{ds_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe_metrics(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n[完成] 结果已保存: {out_path}")
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("对比实验汇总  (ACC / Recall / MAE(m) / MSE)")
    print("=" * 80)
    
    for ds, res in all_results.items():
        print(f"\n数据集: {ds.upper()}")
        print(f"{'模型':<20} {'ACC':>8} {'Recall':>8} {'MAE(m)':>10} {'MSE(m2)':>10}")
        print("-" * 60)
        for mn, m in res.items():
            acc = f"{m['acc']:.4f}" if m['acc'] >= 0 else "—"
            recall = f"{m['recall']:.4f}" if m['recall'] >= 0 else "—"
            mae = f"{m['mae']:.1f}" if m['mae'] >= 0 else "—"
            mse = f"{m['mse']:.1f}" if m['mse'] >= 0 else "—"
            print(f"{mn:<20} {acc:>8} {recall:>8} {mae:>10} {mse:>10}")
        print()


if __name__ == "__main__":
    main()