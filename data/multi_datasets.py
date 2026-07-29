"""
FROSI dataset loader for the MetVis release package.

Expected layout:
  dataset/FROSI/Fog/{50,100,150,200,250,300,400}/*.png

Legacy fallback:
  dataset/FROSI/images/ + labels.csv  (filename, visibility_m)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.noaa_dataset import vis_to_class, normalize_vis, encode_time_features
from data.image_datasets import build_transforms, build_aux_transforms, TimeAligner
from utils import load_aux_maps


def _build_sample(img_path: str, dt: pd.Timestamp,
                  vis_cls: int, vis_val_raw: float,
                  img_tf, aligner: TimeAligner,
                  dataset_name: str,
                  aux_tf,
                  image_root: str = None,
                  station_id: str = None,
                  preferred_station_id: str = None,
                  spatial_dist_km: float = -1.0,
                  dist_band: str = "unknown") -> dict:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to read image: {img_path}") from e
    img = img_tf(img)

    depth, trans = load_aux_maps(
        img_path, dataset_name, aux_tf,
        image_root=image_root, allow_generate=False,
    )

    if aligner is not None:
        seq_feat, time_feat_seq, avail = aligner.get_sequence(
            dt, station_id=station_id, preferred_station_id=preferred_station_id,
        )
        align_station_id = preferred_station_id or getattr(aligner, "_last_matched_station", None) or station_id
        prior_map = aligner.processor.get_vis_prior()
        prior_key = (pd.Timestamp(dt).hour, pd.Timestamp(dt).month)
        prior_vis_m = float(prior_map.get(prior_key, -1.0))
        prior_norm = normalize_vis(prior_vis_m) if prior_vis_m >= 0 else -1.0
        from data.image_datasets import _time_prior_prob
        prior_prob = _time_prior_prob(prior_norm)
    else:
        D = config.NUM_METEO_FEATURES + 1
        from data.noaa_dataset import TIME_FEAT_DIM as TFD
        seq_feat = np.zeros((config.SEQ_LEN, D), dtype=np.float32)
        time_feat_seq = np.zeros((config.SEQ_LEN, TFD), dtype=np.float32)
        avail = 0
        prior_norm = -1.0
        prior_prob = np.zeros(config.NUM_VIS_CLASSES, dtype=np.float32)
        align_station_id = preferred_station_id or station_id

    cur_time_feat = encode_time_features(dt)
    vis_val = normalize_vis(vis_val_raw) if vis_val_raw >= 0 else -1.0

    return {
        "img": img,
        "depth": depth,
        "trans": trans,
        "seq_feat": torch.tensor(seq_feat, dtype=torch.float32),
        "time_feat_seq": torch.tensor(time_feat_seq, dtype=torch.float32),
        "cur_time_feat": torch.tensor(cur_time_feat, dtype=torch.float32),
        "vis_cls": torch.tensor(vis_cls, dtype=torch.long),
        "vis_val": torch.tensor(vis_val, dtype=torch.float32),
        "time_prior_vis": torch.tensor(prior_norm, dtype=torch.float32),
        "time_prior_prob": torch.tensor(prior_prob, dtype=torch.float32),
        "future_vis": torch.tensor(-1.0, dtype=torch.float32),
        "available_steps": torch.tensor(float(avail), dtype=torch.float32),
        "delta_hours": torch.tensor(0.0, dtype=torch.float32),
        "sample_datetime": str(dt),
        "align_station_id": align_station_id or "",
        "spatial_dist_km": torch.tensor(float(spatial_dist_km), dtype=torch.float32),
        "dist_band": dist_band,
    }


def _random_split_fallback(df: pd.DataFrame, split: str,
                           train_ratio: float, val_ratio: float,
                           test_ratio: float, seed: int) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    if split == "train":
        return df.iloc[:n_train].reset_index(drop=True)
    if split == "val":
        return df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    return df.iloc[n_train + n_val:].reset_index(drop=True)


def _split_df_stratified(df: pd.DataFrame, split: str,
                         train_ratio: float, val_ratio: float,
                         seed: int = None,
                         stratify_col: str = "vis_m") -> pd.DataFrame:
    seed = config.SEED if seed is None else seed
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}"
        )

    if len(df) < 3 or stratify_col not in df.columns:
        return _random_split_fallback(
            df, split, train_ratio, val_ratio, test_ratio, seed,
        )

    y = df[stratify_col]
    if y.value_counts().min() < 2:
        warnings.warn(
            f"FROSI {stratify_col} has a bin with < 2 samples; falling back to random split",
            stacklevel=2,
        )
        return _random_split_fallback(
            df, split, train_ratio, val_ratio, test_ratio, seed,
        )

    try:
        train_val, test_df = train_test_split(
            df, test_size=test_ratio, stratify=y, random_state=seed,
        )
        val_rel = val_ratio / max(train_ratio + val_ratio, 1e-8)
        train_df, val_df = train_test_split(
            train_val, test_size=val_rel, stratify=train_val[stratify_col],
            random_state=seed,
        )
    except ValueError:
        warnings.warn("FROSI stratify failed; falling back to random split", stacklevel=2)
        return _random_split_fallback(
            df, split, train_ratio, val_ratio, test_ratio, seed,
        )

    if split == "train":
        return train_df.reset_index(drop=True)
    if split == "val":
        return val_df.reset_index(drop=True)
    return test_df.reset_index(drop=True)


def _print_frosi_split_dist(split: str, part_df: pd.DataFrame, total: int,
                            vis_dirs: list) -> None:
    vis_parts = "  ".join(
        f"{int(v)}m={int((part_df['vis_m'] == v).sum())}" for v in vis_dirs
    )
    cls_counts = part_df["vis_cls"].value_counts().sort_index()
    cls_parts = "  ".join(f"cls{int(k)}={int(v)}" for k, v in cls_counts.items())
    print(
        f"[FROSI-{split}] {len(part_df):,} samples  (total {total:,}, "
        f"vis: {vis_parts}; class: {cls_parts})"
    )


class FROSIDataset(Dataset):
    """
    FROSI synthetic fog dataset.

    Visibility label = Fog subdirectory name (meters).
    Datetime is fixed to 2020-01-01 00:00 (no real capture time).
    Stage-3 may attach NOAA sequences via TimeAligner at that timestamp.
    """
    _VIS_DIRS = [50, 100, 150, 200, 250, 300, 400]
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, split: str = "train",
                 train_ratio: float = None,
                 val_ratio: float = None,
                 aligner: TimeAligner = None,
                 transform=None,
                 root: str = None):
        self.root = root or config.FROSI_ROOT
        self.aligner = aligner
        self.img_tf = transform or build_transforms(split)
        self.aux_tf = build_aux_transforms()
        train_ratio = config.FROSI_TRAIN_RATIO if train_ratio is None else train_ratio
        val_ratio = config.FROSI_VAL_RATIO if val_ratio is None else val_ratio

        records = self._scan()
        if not records:
            raise RuntimeError(
                f"FROSI dataset empty or wrong path: {self.root}\n"
                f"Expected: {{root}}/Fog/50/*.png  Fog/100/*.png  ..."
            )

        df = pd.DataFrame(records).reset_index(drop=True)
        self.df = _split_df_stratified(
            df, split, train_ratio, val_ratio,
            seed=config.FROSI_SPLIT_SEED,
            stratify_col="vis_m",
        )
        _print_frosi_split_dist(split, self.df, len(df), self._VIS_DIRS)

    def _scan(self) -> list:
        fog_dir = Path(self.root) / "Fog"
        if not fog_dir.exists():
            return self._scan_legacy()

        records = []
        for vis_m in self._VIS_DIRS:
            vis_dir = fog_dir / str(vis_m)
            if not vis_dir.exists():
                continue
            vis_cls = vis_to_class(float(vis_m))
            for img_path in sorted(vis_dir.iterdir()):
                if img_path.suffix.lower() in self._IMG_EXTS:
                    records.append({
                        "filepath": str(img_path),
                        "vis_m": float(vis_m),
                        "vis_cls": vis_cls,
                        "datetime": pd.Timestamp("2020-01-01 00:00"),
                    })
        return records

    def _scan_legacy(self) -> list:
        import csv
        label_path = os.path.join(self.root, "labels.csv")
        img_dir = os.path.join(self.root, "images")
        if not os.path.exists(label_path):
            return []
        records = []
        with open(label_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get("filename", "").strip()
                vis = float(row.get("visibility_m", row.get("vis_m", -1)) or -1)
                if fn and vis >= 0:
                    records.append({
                        "filepath": os.path.join(img_dir, fn),
                        "vis_m": vis,
                        "vis_cls": vis_to_class(vis),
                        "datetime": pd.Timestamp("2020-01-01 00:00"),
                    })
        return records

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return _build_sample(
            img_path=row["filepath"],
            dt=pd.Timestamp(row["datetime"]),
            vis_cls=int(row["vis_cls"]),
            vis_val_raw=float(row["vis_m"]),
            img_tf=self.img_tf,
            aligner=self.aligner,
            dataset_name="frosi",
            aux_tf=self.aux_tf,
            image_root=self.root,
        )
