"""
工具函数: 图像预处理、深度图/透射图生成接口、可视化等
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import cv2
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.noaa_dataset import vis_to_class

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_DATASET_NAMES = ("frosi",)


def load_checkpoint(path, map_location=None):
    """Load a trusted local checkpoint (PyTorch 2.6+ defaults weights_only=True)."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def to_device_batch(raw, device):
    """Move tensor batch fields to device; keep string/list metadata unchanged."""
    batch = dict(raw)
    for k, v in raw.items():
        if hasattr(v, "to"):
            batch[k] = v.to(device)
    return batch


# ================================================================
# Image preprocessing helpers
# ================================================================

def estimate_dark_channel(img_bgr: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Dark-channel prior map in [0, 1]."""
    img = img_bgr.astype(np.float32) / 255.0
    dark = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(dark, kernel)
    return dark


def estimate_transmission(img_bgr: np.ndarray, omega: float = 0.95,
                           patch_size: int = 15) -> np.ndarray:
    """Transmission map from dark-channel prior, values in [0, 1]."""
    img = img_bgr.astype(np.float32) / 255.0
    dark = estimate_dark_channel(img_bgr, patch_size)
    flat = dark.flatten()
    n_top = max(int(len(flat) * 0.001), 1)
    idx = np.argsort(flat)[-n_top:]
    A = np.max(img.reshape(-1, 3)[idx // img.shape[1], :], axis=0)
    A = np.clip(A, 0.1, 1.0)
    trans = 1 - omega * np.min(img / A, axis=2)
    trans = np.clip(trans, 0.1, 1.0)
    return trans.astype(np.float32)


def generate_depth_proxy(img_bgr: np.ndarray) -> np.ndarray:
    """Sobel-gradient depth proxy in [0, 1]."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    if grad.max() > 0:
        grad = grad / grad.max()
    return grad.astype(np.float32)


def get_dataset_root(dataset_name: str, image_root: Optional[str] = None) -> str:
    """Return dataset image root used for aux-map relative paths."""
    if image_root:
        return os.path.abspath(image_root)
    if dataset_name == "frosi":
        return os.path.abspath(config.FROSI_ROOT)
    raise ValueError(f"Unknown dataset: {dataset_name}. Available: {_DATASET_NAMES}")


def _rel_image_key(img_path: str, image_root: str) -> str:
    """图像相对路径（保留子目录结构，统一 .png 后缀）。"""
    img_abs = os.path.abspath(img_path)
    root_abs = os.path.abspath(image_root)
    try:
        rel = os.path.relpath(img_abs, root_abs)
    except ValueError:
        rel = os.path.basename(img_abs)
    rel = rel.replace("\\", "/")
    stem, _ = os.path.splitext(rel)
    return stem + ".png"


def aux_cache_paths(dataset_name: str, img_path: str,
                    image_root: Optional[str] = None) -> Tuple[str, str]:
    """返回 (depth_cache_path, trans_cache_path)。"""
    root = get_dataset_root(dataset_name, image_root=image_root)
    rel_png = _rel_image_key(img_path, root)
    depth_dir = str(Path(config.AUX_CACHE_ROOT) / dataset_name / "depth")
    trans_dir = str(Path(config.AUX_CACHE_ROOT) / dataset_name / "trans")
    return (
        str(Path(depth_dir) / rel_png),
        str(Path(trans_dir) / rel_png),
    )


def rgb_path_to_bgr_array(img_path: str) -> np.ndarray:
    """读取 RGB 图像并转为 BGR uint8，供 DCP / 深度代理使用。"""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image
    rgb = np.array(Image.open(img_path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_aux_maps(img_path: str, depth_path: str, trans_path: str) -> bool:
    """从 RGB 原图生成 depth/trans 并写入缓存路径。"""
    img_bgr = rgb_path_to_bgr_array(img_path)
    depth = generate_depth_proxy(img_bgr)
    trans = estimate_transmission(img_bgr)
    os.makedirs(os.path.dirname(depth_path), exist_ok=True)
    os.makedirs(os.path.dirname(trans_path), exist_ok=True)
    ok_d = cv2.imwrite(depth_path, (depth * 255).astype(np.uint8))
    ok_t = cv2.imwrite(trans_path, (trans * 255).astype(np.uint8))
    return bool(ok_d and ok_t)


def preprocess_dataset_aux_maps(dataset_name: str,
                                image_root: Optional[str] = None,
                                skip_existing: bool = True) -> dict:
    """为指定数据集批量生成 depth/trans 缓存（镜像相对路径）。"""
    root = get_dataset_root(dataset_name, image_root=image_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"图像目录不存在: {root}")

    img_paths = [
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in _IMAGE_EXTS
    ]
    stats = {"total": len(img_paths), "written": 0, "skipped": 0, "failed": 0}

    for i, img_path in enumerate(img_paths):
        img_str = str(img_path)
        depth_path, trans_path = aux_cache_paths(dataset_name, img_str, image_root=root)
        if skip_existing and os.path.exists(depth_path) and os.path.exists(trans_path):
            stats["skipped"] += 1
            continue
        try:
            if write_aux_maps(img_str, depth_path, trans_path):
                stats["written"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"  [失败] {img_str}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  [{dataset_name}] {i+1}/{len(img_paths)} "
                  f"写={stats['written']} 跳={stats['skipped']} 败={stats['failed']}")

    print(f"[aux预处理-{dataset_name}] 完成: {stats}")
    return stats


def load_aux_maps(img_path: str, dataset_name: str, aux_tf,
                  image_root: Optional[str] = None,
                  allow_generate: bool = False):
    """
    加载 depth/trans 张量。训练时 allow_generate=False，缺失则报错。
    推理时可 allow_generate=True 对单张图即时生成并写入缓存。
    """
    from PIL import Image

    depth_path, trans_path = aux_cache_paths(
        dataset_name, img_path, image_root=image_root,
    )
    if not (os.path.exists(depth_path) and os.path.exists(trans_path)):
        if allow_generate:
            write_aux_maps(img_path, depth_path, trans_path)
        else:
            raise FileNotFoundError(
                f"缺少 depth/trans 缓存: {depth_path}\n"
                f"请先运行: python scripts/preprocess_aux_maps.py --dataset {dataset_name}"
            )

    depth = aux_tf(Image.open(depth_path).convert("L"))
    trans = aux_tf(Image.open(trans_path).convert("L"))
    return depth, trans


def preprocess_images(image_dir: str,
                      depth_dir: str,
                      trans_dir: str,
                      skip_existing: bool = True):
    """
    批量预处理（扁平 stem 命名，兼容旧接口）。
    新代码请使用 preprocess_dataset_aux_maps。
    """
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    img_paths = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in _IMAGE_EXTS]
    print(f"[预处理] 找到 {len(img_paths)} 张图像")

    for i, img_path in enumerate(img_paths):
        stem = img_path.stem
        depth_path = os.path.join(depth_dir, stem + ".png")
        trans_path = os.path.join(trans_dir, stem + ".png")

        if skip_existing and os.path.exists(depth_path) and os.path.exists(trans_path):
            continue

        try:
            write_aux_maps(str(img_path), depth_path, trans_path)
        except Exception as e:
            print(f"  [警告] 无法处理: {img_path}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(img_paths)}")

    print("[预处理] 完成")


# ================================================================
# Learning-rate schedule: Warmup + Cosine
# ================================================================

class WarmupCosineScheduler:
    """Linear warmup + cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.opt = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base_lr = base_lr

    def step(self, epoch: int):
        import math
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr


# ================================================================
# GPU显存监控
# ================================================================

def print_gpu_memory():
    """打印当前GPU显存占用(仅CUDA)"""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[GPU] 已用={alloc:.2f}GB  预留={reserved:.2f}GB  总量={total:.2f}GB")
    except Exception:
        pass


