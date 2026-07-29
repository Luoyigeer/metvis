"""
Class-balanced sampler and loss-weight helpers for imbalanced visibility labels.

Provides:
  compute_class_weights()
  BalancedBatchSampler
  build_balanced_loader()
"""
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ----------------------------------------------------------------
# 统计数据集各类别数量
# ----------------------------------------------------------------
def get_class_counts(dataset: Dataset, label_key: str = "vis_cls") -> np.ndarray:
    """
    统计 dataset 各等级样本数。
    返回长度为 NUM_VIS_CLASSES 的 numpy 数组。
    直接读 df["vis_cls"]（FROSIDataset 等均有 self.df）
    """
    K = config.NUM_VIS_CLASSES
    counts = np.zeros(K, dtype=np.int64)

    if hasattr(dataset, "df") and "vis_cls" in dataset.df.columns:
        for c in dataset.df["vis_cls"]:
            if 0 <= c < K:
                counts[c] += 1
        return counts

    # 兜底：逐样本读取（慢，仅当 df 不可用时）
    for i in range(len(dataset)):
        item = dataset[i]
        c = int(item[label_key].item()) if hasattr(item[label_key], "item") else int(item[label_key])
        if 0 <= c < K:
            counts[c] += 1
    return counts


# ----------------------------------------------------------------
# 类别权重（用于 CE 损失）
# ----------------------------------------------------------------
def compute_class_weights(counts: np.ndarray,
                          mode: str = config.CLASS_WEIGHT_MODE,
                          device=None) -> torch.Tensor:
    """
    根据类别频率计算损失权重。

    mode:
      "inv_freq"  : w_k = N / (K * n_k)      激进，稀有类权重极高
      "sqrt_inv"  : w_k = sqrt(N / (K * n_k)) mild inverse-frequency weighting
      "none"      : 全1，不加权

    返回: FloatTensor [K]，0样本的类权重设为最大权重
    """
    K = len(counts)
    N = counts.sum()

    if mode == "none" or N == 0:
        w = torch.ones(K, dtype=torch.float32)
        return w.to(device) if device else w

    weights = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if counts[k] > 0:
            if mode == "sqrt_inv":
                weights[k] = math.sqrt(N / (K * counts[k]))
            else:  # inv_freq
                weights[k] = N / (K * counts[k])
        # counts[k]==0: 暂设0，后面填充

    # 0样本类设为最大权重（避免除零影响，实际上不会有对应样本）
    max_w = weights[weights > 0].max() if (weights > 0).any() else 1.0
    weights[weights == 0] = max_w

    # 归一化到均值=1（保持总损失量级不变）
    weights = weights / weights.mean()

    print(f"[ClassWeights/{mode}] " +
          " ".join(f"{config.VIS_CLASS_NAMES[k]}:{weights[k]:.2f}" for k in range(K)))

    w = torch.tensor(weights, dtype=torch.float32)
    return w.to(device) if device else w


# ----------------------------------------------------------------
# 平衡采样器（WeightedRandomSampler）
# ----------------------------------------------------------------
def build_sample_weights(dataset: Dataset,
                         samples_per_class: int = config.BALANCED_SAMPLES_PER_CLASS,
                         max_class5_ratio: float = config.MAX_CLASS5_RATIO
                         ) -> np.ndarray:
    """
    为每个样本计算采样权重，使每类在一个 epoch 内被采样约 samples_per_class 次。

    samples_per_class:
      > 0: per-class target count (fixed; useful for extreme imbalance)
      -1 : 各类采样到与最大类相同数量（完全均衡）

    max_class5_ratio:
      等级5的采样权重被额外缩减，使其占总样本比例不超过此值
    """
    K = config.NUM_VIS_CLASSES
    counts = np.zeros(K, dtype=np.int64)
    labels = []

    if hasattr(dataset, "df") and "vis_cls" in dataset.df.columns:
        labels = dataset.df["vis_cls"].tolist()
        for c in labels:
            if 0 <= c < K:
                counts[c] += 1
    else:
        for i in range(len(dataset)):
            item = dataset[i]
            c = int(item["vis_cls"].item())
            labels.append(c)
            if 0 <= c < K:
                counts[c] += 1

    if samples_per_class == -1:
        target = int(counts[counts > 0].max())
    else:
        target = samples_per_class

    # 每类采样权重 = target / count（实现均衡）
    class_w = np.zeros(K, dtype=np.float64)
    for k in range(K):
        class_w[k] = target / counts[k] if counts[k] > 0 else 0.0

    # 对等级5额外惩罚（控制其在batch中的比例）
    if max_class5_ratio < 1.0 and counts[K-1] > 0:
        # 当前等级5自然比例
        natural_ratio = counts[K-1] / counts.sum()
        if natural_ratio > max_class5_ratio:
            # 缩减等级5权重使其采样比例降到 max_class5_ratio
            scale = max_class5_ratio / natural_ratio
            class_w[K-1] *= scale

    # 赋给每个样本
    sample_weights = np.array(
        [class_w[c] if 0 <= c < K else 0.0 for c in labels],
        dtype=np.float64
    )

    # 统计实际采样分布（理论值）
    total_w = sample_weights.sum()
    if total_w > 0:
        print("[SampleWeights] 理论采样分布:")
        for k in range(K):
            mask = np.array(labels) == k
            class_total_w = sample_weights[mask].sum()
            ratio = class_total_w / total_w * 100
            n_samples = int(target) if counts[k] > 0 else 0
            print(f"  {config.VIS_CLASS_NAMES[k]:<12} "
                  f"原始={counts[k]:>7,}  "
                  f"采样比={ratio:>5.1f}%  "
                  f"目标={n_samples:>5}")

    return sample_weights


def build_balanced_loader(dataset: Dataset,
                          batch_size: int = config.BATCH_SIZE,
                          num_workers: int = config.NUM_WORKERS,
                          samples_per_class: int = config.BALANCED_SAMPLES_PER_CLASS,
                          max_class5_ratio: float = config.MAX_CLASS5_RATIO,
                          ) -> DataLoader:
    """
    构建使用 WeightedRandomSampler 的平衡 DataLoader。
    替代 shuffle=True 的普通 DataLoader 用于训练集。
    """
    sample_w = build_sample_weights(dataset, samples_per_class, max_class5_ratio)
    # epoch 内总采样次数 = K * samples_per_class（均衡时）
    K         = config.NUM_VIS_CLASSES
    num_samples = K * (samples_per_class if samples_per_class > 0 else
                       int(get_class_counts(dataset).max()))

    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_w).float(),
        num_samples = num_samples,
        replacement = True,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = config.PIN_MEMORY,
        drop_last   = True,
    )


def build_mild_sample_weights(dataset: Dataset,
                              power: float = config.IMAGE_MILD_POWER,
                              ) -> np.ndarray:
    """
    轻度平衡采样权重：在自然分布与完全均衡之间插值。

    每样本权重 w_k = (N / (K * n_k))^(1 - power)
      power=1.0 → w_k=1，采样比例 ∝ n_k（自然分布）
      power=0.0 → w_k ∝ 1/n_k，各类采样比例接近均等
      power=0.5 → 轻度抬稀有类，但不强制完全均衡
    """
    K = config.NUM_VIS_CLASSES
    counts = get_class_counts(dataset)
    N = counts.sum()
    labels = []

    if hasattr(dataset, "df") and "vis_cls" in dataset.df.columns:
        labels = dataset.df["vis_cls"].tolist()
    else:
        for i in range(len(dataset)):
            item = dataset[i]
            labels.append(int(item["vis_cls"].item()))

    class_w = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if counts[k] > 0:
            class_w[k] = (N / (K * counts[k])) ** (1.0 - power)

    sample_weights = np.array(
        [class_w[c] if 0 <= c < K else 0.0 for c in labels],
        dtype=np.float64,
    )
    total_w = sample_weights.sum()
    if total_w > 0:
        print("[SampleWeights/mild] 理论采样分布:")
        for k in range(K):
            mask = np.array(labels) == k
            if not mask.any():
                continue
            class_total_w = sample_weights[mask].sum()
            ratio = class_total_w / total_w * 100
            print(f"  {config.VIS_CLASS_NAMES[k]:<12} "
                  f"原始={counts[k]:>7,}  "
                  f"采样比={ratio:>5.1f}%")

    return sample_weights


def build_mild_loader(dataset: Dataset,
                      batch_size: int = config.BATCH_SIZE,
                      num_workers: int = config.NUM_WORKERS,
                      power: float = config.IMAGE_MILD_POWER,
                      ) -> DataLoader:
    """轻度平衡 DataLoader：epoch 长度 ≈ 数据集大小，分布介于自然与均衡之间。"""
    sample_w = build_mild_sample_weights(dataset, power=power)
    num_samples = len(dataset)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w).float(),
        num_samples=num_samples,
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
