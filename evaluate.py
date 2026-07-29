# evaluate.py
"""
评估指标计算模块

提供能见度估计任务的评估指标：
- 分类指标：Accuracy, Recall (macro), F1-Score, Kappa
- 回归指标：MAE, MSE, RMSE, R²
- 每类召回率：用于分析类别不平衡问题
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score,
    cohen_kappa_score,
    mean_absolute_error, 
    mean_squared_error,
    r2_score
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _empty_bin_metrics():
    return {"mae": -1.0, "mse": -1.0, "n": 0}


def _mae_for_classes(gt_val, pred_val, gt_cls, cls_ids):
    mask = np.isin(gt_cls, cls_ids)
    if not mask.any():
        return _empty_bin_metrics()
    return {
        "mae": float(mean_absolute_error(gt_val[mask], pred_val[mask])),
        "mse": float(mean_squared_error(gt_val[mask], pred_val[mask])),
        "n": int(mask.sum()),
    }


def _format_bin_mae(mae: float, n: int) -> str:
    if n <= 0 or mae < 0:
        return "—"
    return f"{mae:.1f}m"


def compute_metrics(gt_val: np.ndarray, pred_val: np.ndarray,
                    gt_cls: np.ndarray, pred_cls: np.ndarray,
                    return_per_class: bool = True) -> dict:
    """
    计算能见度估计任务的评估指标

    Args:
        gt_val: 真值能见度（米制），shape (N,)
        pred_val: 预测能见度（米制），shape (N,)
        gt_cls: 真值等级 (0-5)，shape (N,)
        pred_cls: 预测等级 (0-5)，shape (N,)
        return_per_class: 是否返回每类召回率

    Returns:
        dict: 包含以下指标
            - acc: 分类准确率
            - recall: Macro 召回率
            - f1: Macro F1 分数
            - kappa: Cohen's Kappa 系数
            - mae: 平均绝对误差（米）
            - mse: 均方误差（米²）
            - rmse: 均方根误差（米）
            - r2: 决定系数 R²
            - per_class_recall: 每类召回率列表 (K,)
            - per_class_f1: 每类 F1 分数列表 (K,)
            - class_counts: 各类样本数量 (K,)
            - mae_cls_0_3 / mse_cls_0_3 / n_cls_0_3: 0-3 类分档回归
            - mae_cls_4_5 / mse_cls_4_5 / n_cls_4_5: 4-5 类分档回归
    """
    # 确保输入是 numpy 数组
    gt_val = np.array(gt_val).flatten()
    pred_val = np.array(pred_val).flatten()
    gt_cls = np.array(gt_cls).flatten()
    pred_cls = np.array(pred_cls).flatten()
    
    # 过滤无效样本（-1 表示无标注）
    valid_mask = (gt_cls >= 0) & (gt_val >= 0)
    
    if not valid_mask.any():
        # 没有有效样本，返回默认值
        K = config.NUM_VIS_CLASSES
        return {
            "acc": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "kappa": 0.0,
            "mae": -1.0,
            "mse": -1.0,
            "rmse": -1.0,
            "r2": -1.0,
            "per_class_recall": [0.0] * K,
            "per_class_f1": [0.0] * K,
            "class_counts": [0] * K,
            "n_samples": 0,
            "mae_cls_0_3": -1.0,
            "mse_cls_0_3": -1.0,
            "n_cls_0_3": 0,
            "mae_cls_4_5": -1.0,
            "mse_cls_4_5": -1.0,
            "n_cls_4_5": 0,
        }
    
    # 过滤后的数据
    gt_val_valid = gt_val[valid_mask]
    pred_val_valid = pred_val[valid_mask]
    gt_cls_valid = gt_cls[valid_mask]
    pred_cls_valid = pred_cls[valid_mask]
    
    n_samples = len(gt_cls_valid)
    K = config.NUM_VIS_CLASSES
    
    # ========== 分类指标 ==========
    # 准确率
    acc = accuracy_score(gt_cls_valid, pred_cls_valid)
    
    # Macro 召回率（各类别平均，不考虑样本不平衡）
    recall = recall_score(gt_cls_valid, pred_cls_valid, 
                          average='macro', 
                          labels=list(range(K)),
                          zero_division=0)
    
    # Macro F1 分数
    f1 = f1_score(gt_cls_valid, pred_cls_valid, 
                  average='macro', 
                  labels=list(range(K)),
                  zero_division=0)
    
    # Cohen's Kappa（考虑随机一致性的系数）
    kappa = cohen_kappa_score(gt_cls_valid, pred_cls_valid)
    
    # ========== 每类召回率和 F1 ==========
    per_class_recall = None
    per_class_f1 = None
    class_counts = None
    
    if return_per_class:
        per_class_recall = recall_score(gt_cls_valid, pred_cls_valid,
                                        labels=list(range(K)),
                                        average=None,
                                        zero_division=0)
        per_class_f1 = f1_score(gt_cls_valid, pred_cls_valid,
                                labels=list(range(K)),
                                average=None,
                                zero_division=0)
        
        # 统计各类样本数量
        class_counts = [np.sum(gt_cls_valid == i) for i in range(K)]
    
    # ========== 回归指标 ==========
    # MAE (米)
    mae = mean_absolute_error(gt_val_valid, pred_val_valid)
    
    # MSE (米²)
    mse = mean_squared_error(gt_val_valid, pred_val_valid)
    
    # RMSE (米)
    rmse = np.sqrt(mse)
    
    # R² 决定系数（可能为负数，表示比简单均值还差）
    r2 = r2_score(gt_val_valid, pred_val_valid)

    # ========== 分档回归指标 ==========
    low_bin = _mae_for_classes(
        gt_val_valid, pred_val_valid, gt_cls_valid,
        config.VIS_MAE_LOW_CLASSES,
    )
    high_bin = _mae_for_classes(
        gt_val_valid, pred_val_valid, gt_cls_valid,
        config.VIS_MAE_HIGH_CLASSES,
    )
    
    # ========== 额外诊断指标 ==========
    # 相对绝对误差 (Relative Absolute Error) - 百分比
    # 避免除以零
    gt_mean = np.mean(gt_val_valid)
    if gt_mean > 0:
        rae = np.mean(np.abs(pred_val_valid - gt_val_valid) / gt_val_valid) * 100
    else:
        rae = -1.0
    
    # 预测值分布统计（用于调试）
    pred_cls_dist = [np.sum(pred_cls_valid == i) for i in range(K)]
    
    # 判断模型是否退化到只预测多数类
    # 如果预测类别只有1-2种，说明模型退化了
    unique_pred = len(np.unique(pred_cls_valid))
    is_degenerate = unique_pred <= 2 and K > 2
    
    # ========== 打印详细报告 ==========
    print(f"\n[评估报告] 样本数={n_samples:,}")
    print(f"  分类指标:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Macro Recall: {recall:.4f}")
    print(f"    Macro F1: {f1:.4f}")
    print(f"    Cohen's Kappa: {kappa:.4f}")
    print(f"  回归指标:")
    print(f"    MAE: {mae:.2f}m  |  RMSE: {rmse:.2f}m")
    print(f"    MSE: {mse:.2f}m²  |  R²: {r2:.4f}")
    print(f"  分档回归:")
    print(f"    0-3类 MAE={_format_bin_mae(low_bin['mae'], low_bin['n'])}  "
          f"(n={low_bin['n']:,})  |  "
          f"4-5类 MAE={_format_bin_mae(high_bin['mae'], high_bin['n'])}  "
          f"(n={high_bin['n']:,})")
    
    if per_class_recall is not None:
        print(f"  每类召回率:")
        vis_class_names = getattr(config, 'VIS_CLASS_NAMES', 
                                   [f'C{i}' for i in range(K)])
        for i in range(K):
            count = class_counts[i] if class_counts else 0
            print(f"    {vis_class_names[i]:<12}: "
                  f"召回={per_class_recall[i]:.4f}  "
                  f"F1={per_class_f1[i]:.4f}  "
                  f"样本数={count:,}")
    
    if is_degenerate:
        print(f"  ⚠️ 警告: 模型退化! 只预测了 {unique_pred}/{K} 个类别")
        print(f"    预测分布: {pred_cls_dist}")
    
    # 构建返回字典
    result = {
        # 分类指标
        "acc": float(acc),
        "recall": float(recall),
        "f1": float(f1),
        "kappa": float(kappa),
        
        # 回归指标
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mae_cls_0_3": low_bin["mae"],
        "mse_cls_0_3": low_bin["mse"],
        "n_cls_0_3": low_bin["n"],
        "mae_cls_4_5": high_bin["mae"],
        "mse_cls_4_5": high_bin["mse"],
        "n_cls_4_5": high_bin["n"],
        
        # 诊断指标
        "rae": float(rae),
        "is_degenerate": bool(is_degenerate),
        "unique_pred_classes": int(unique_pred),
        "n_samples": int(n_samples),
        
        # 详细分布
        "per_class_recall": per_class_recall.tolist() if per_class_recall is not None else [],
        "per_class_f1": per_class_f1.tolist() if per_class_f1 is not None else [],
        "class_counts": class_counts if class_counts is not None else [],
        "pred_cls_dist": pred_cls_dist,
    }
    
    return result


def compute_metrics_simple(gt_val, pred_val, gt_cls, pred_cls):
    """
    简化版指标计算（兼容旧代码）
    返回最基本的 acc, recall, mae, mse, per_class_recall
    """
    result = compute_metrics(gt_val, pred_val, gt_cls, pred_cls, return_per_class=True)
    return {
        "acc": result["acc"],
        "recall": result["recall"],
        "mae": result["mae"],
        "mse": result["mse"],
        "per_class_recall": result["per_class_recall"],
        "f1": result["f1"],
        "kappa": result["kappa"],
    }


def print_detailed_results(results_dict: dict, dataset_name: str = ""):
    """
    打印多个模型的详细对比结果
    
    Args:
        results_dict: {model_name: metrics_dict}
        dataset_name: 数据集名称
    """
    if dataset_name:
        print(f"\n{'='*80}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*80}")
    
    K = config.NUM_VIS_CLASSES
    vis_class_names = getattr(config, 'VIS_CLASS_NAMES', 
                               [f'C{i}' for i in range(K)])
    
    # 表头
    print(f"\n{'模型':<20} {'ACC':>8} {'Recall':>8} {'F1':>8} {'Kappa':>8} "
          f"{'MAE(m)':>10} {'RMSE(m)':>10}")
    print("-" * 80)
    
    for model_name, metrics in results_dict.items():
        acc = f"{metrics['acc']:.4f}" if metrics['acc'] >= 0 else "—"
        recall = f"{metrics['recall']:.4f}" if metrics['recall'] >= 0 else "—"
        f1 = f"{metrics.get('f1', 0):.4f}" if metrics.get('f1', -1) >= 0 else "—"
        kappa = f"{metrics.get('kappa', 0):.4f}" if metrics.get('kappa', -1) >= 0 else "—"
        mae = f"{metrics['mae']:.1f}" if metrics['mae'] >= 0 else "—"
        rmse = f"{metrics.get('rmse', 0):.1f}" if metrics.get('rmse', -1) >= 0 else "—"
        
        print(f"{model_name:<20} {acc:>8} {recall:>8} {f1:>8} {kappa:>8} "
              f"{mae:>10} {rmse:>10}")
    
    # 打印每类召回率（如果第一个模型有）
    first_model = next(iter(results_dict.values()))
    if "per_class_recall" in first_model and first_model["per_class_recall"]:
        print(f"\n每类召回率详情:")
        print(f"{'模型':<20}", end="")
        for i, name in enumerate(vis_class_names):
            print(f"{name:>10}", end="")
        print()
        print("-" * (20 + 10 * K))
        
        for model_name, metrics in results_dict.items():
            if "per_class_recall" in metrics and metrics["per_class_recall"]:
                print(f"{model_name:<20}", end="")
                for r in metrics["per_class_recall"][:K]:
                    print(f"{r:>10.4f}", end="")
                print()


if __name__ == "__main__":
    # 测试代码
    print("测试 evaluate.py...")
    
    # 模拟数据
    np.random.seed(42)
    n = 1000
    gt_cls = np.random.randint(0, 6, n)
    pred_cls = gt_cls.copy()
    # 加入一些错误
    noise_idx = np.random.choice(n, 100, replace=False)
    pred_cls[noise_idx] = np.random.randint(0, 6, 100)
    
    gt_val = np.random.uniform(50, 50000, n)
    pred_val = gt_val + np.random.normal(0, 1000, n)
    pred_val = np.clip(pred_val, 0, 50000)
    
    metrics = compute_metrics(gt_val, pred_val, gt_cls, pred_cls)
    print("\n测试结果:")
    for k, v in metrics.items():
        if k not in ["per_class_recall", "per_class_f1", "class_counts", "pred_cls_dist"]:
            print(f"  {k}: {v}")