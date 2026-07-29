"""
NOAA预训练脚本
只训练时序分支 (TimesNetBranch) 在多站点NOAA气象数据上
使结果模型具备气象->能见度的先验知识
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.noaa_dataset import NOAADataProcessor, TIME_FEAT_DIM, vis_to_class
from data.meteorological_dataset import NOAASequenceDataset
from models.temporal_branch import TimesNetBranch
from losses import NOAAPretrainLoss
from evaluate import compute_metrics
from utils import to_device_batch


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(d_in: int) -> TimesNetBranch:
    return TimesNetBranch(d_in=d_in)


class NOAAPretrainWrapper(nn.Module):
    """包装TimesNetBranch,提供统一输出dict接口"""

    def __init__(self, ts_branch: TimesNetBranch):
        super().__init__()
        self.ts = ts_branch

    def forward(self, batch: dict) -> dict:
        feat, vis_reg_out, future_reg_out = self.ts(
            batch["seq_feat"],
            batch["time_feat_seq"],
            batch["cur_time_feat"],
        )
        return {
            # EDL接口: NOAAPretrainLoss 需要完整的 reg_out dict
            "ts_vis_reg_out":    vis_reg_out,
            "ts_future_reg_out": future_reg_out,
            # 兼容旧评估代码
            "ts_vis_pred":       vis_reg_out["vis_pred"],
            "ts_future_pred":    future_reg_out["vis_pred"],
        }


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    n = n_skip = 0
    for batch in loader:
        batch = to_device_batch(batch, device)
        outputs = model(batch)
        loss_dict = criterion(outputs, batch)
        loss = loss_dict["total"]

        # 跳过 NaN/inf loss 的 batch，避免参数污染
        if not torch.isfinite(loss):
            n_skip += 1
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        # 检测梯度 NaN
        grad_nan = any(
            p.grad is not None and
            (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters() if p.requires_grad
        )
        if not grad_nan:
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
        else:
            n_skip += 1
            optimizer.zero_grad()

        total_loss += loss.item() * len(batch["target_vis"])
        n += len(batch["target_vis"])

    if n_skip > 0:
        print(f"  [警告] 跳过 {n_skip} 个NaN/inf batch")
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred_cls, all_gt_cls = [], []
    all_pred_val, all_gt_val = [], []
    n = 0

    max_vis = float(getattr(config, "MAX_VIS", 50000.0))
    use_meter = bool(getattr(config, "USE_METER_LABELS", False))

    for batch in loader:
        batch = to_device_batch(batch, device)
        outputs = model(batch)
        loss_dict = criterion(outputs, batch)
        total_loss += loss_dict["total"].item() * len(batch["target_vis"])
        n += len(batch["target_vis"])

        # 收集预测 (从归一化值反推等级)
        pred_vis = outputs["ts_vis_pred"].squeeze(-1).cpu().numpy() * max_vis
        gt_vis   = batch["target_vis"].cpu().numpy()
        if not use_meter:
            gt_vis = gt_vis * max_vis

        pred_cls = [vis_to_class(v) for v in pred_vis]
        gt_cls   = batch["target_cls"].cpu().numpy().tolist()

        all_pred_cls.extend(pred_cls)
        all_gt_cls.extend(gt_cls)
        all_pred_val.extend(pred_vis.tolist())
        all_gt_val.extend(gt_vis.tolist())

    metrics = compute_metrics(
        np.array(all_gt_val), np.array(all_pred_val),
        np.array(all_gt_cls), np.array(all_pred_cls),
    )
    return total_loss / max(n, 1), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NOAA_PRETRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.NOAA_LR)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE * 4)
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--max_samples", type=int, default=500_000,
                        help="每个split最大样本数，-1=不限制 (默认50万)")
    parser.add_argument("--num_workers", type=int, default=-1,
                        help="DataLoader workers数，-1=自动(Windows用0)")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[预训练] 设备: {device}")

    # 加载NOAA数据
    print("[预训练] 加载NOAA数据...")
    processor = NOAADataProcessor()
    processor.load(force_reload=args.force_reload)

    train_ds = NOAASequenceDataset(processor, split="train",
                                   max_samples=args.max_samples)
    val_ds   = NOAASequenceDataset(processor, split="val",
                                   max_samples=max(1, args.max_samples // 5)
                                   if args.max_samples > 0 else -1)

    _nw = config.NUM_WORKERS if args.num_workers < 0 else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=_nw, pin_memory=config.PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=_nw, pin_memory=config.PIN_MEMORY)

    # 计算输入维度
    sample = train_ds[0]
    d_in = sample["seq_feat"].shape[-1]
    print(f"[预训练] 时序输入维度: {d_in}")

    ts_branch = build_model(d_in)
    model = NOAAPretrainWrapper(ts_branch).to(device)
    # 把 Dataset 计算好的类别权重传给损失函数
    class_weights = torch.tensor(
        train_ds.class_weights, dtype=torch.float32
    ).to(device)
    criterion = NOAAPretrainLoss(class_weights=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.NOAA_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_mae = float('inf')
    log_path = os.path.join(config.LOG_DIR, "noaa_pretrain.txt")

    print("[预训练] 开始训练...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} | "
              f"ACC={metrics['acc']:.4f} Recall={metrics['recall']:.4f} "
              f"MAE={metrics['mae']:.1f} MSE={metrics['mse']:.1f} | "
              f"{elapsed:.1f}s")

        # 保存日志
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},"
                    f"{metrics['acc']:.4f},{metrics['recall']:.4f},"
                    f"{metrics['mae']:.3f},{metrics['mse']:.3f}\n")

        # 保存最佳
        if metrics["mae"] < best_mae or best_mae < 0:
            best_mae = metrics["mae"]
            ckpt = {
                "epoch": epoch,
                "model_state": ts_branch.state_dict(),
                "metrics": metrics,
                "d_in": d_in,
            }
            torch.save(ckpt, config.NOAA_PRETRAIN_CKPT)
            print(f"  -> 保存最佳预训练模型 (MAE={best_mae:.1f}m)")

    print(f"[预训练完成] 最佳MAE: {best_mae:.1f}m")
    print(f"模型保存至: {config.NOAA_PRETRAIN_CKPT}")


if __name__ == "__main__":
    main()
