"""
Image transforms and NOAA TimeAligner for the FROSI release package.
"""
import os, re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.meteorological_dataset import (
    vis_to_class, normalize_vis, encode_time_features, TIME_FEAT_DIM
)
from data import noaa_dataset as noaa_ds
from data.noaa_dataset import NOAADataProcessor

_MILES_TO_METERS = 1609.34


def build_transforms(split="train"):
    if split == "train":
        return T.Compose([
            T.Resize(config.IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(config.IMAGE_MEAN, config.IMAGE_STD),
        ])
    return T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(config.IMAGE_MEAN, config.IMAGE_STD),
    ])


def build_aux_transforms():
    return T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])

def _time_prior_prob(vis_norm: float) -> np.ndarray:
    """将时间先验的均值能见度转换为软标签分布 (K,)."""
    K = config.NUM_VIS_CLASSES
    bins = config.VIS_BINS
    max_vis = 50000.0
    if vis_norm < 0:
        return np.zeros(K, dtype=np.float32)
    vis_m = float(np.clip(vis_norm, 0.0, 1.0)) * max_vis
    centers = []
    for i in range(K):
        lo = bins[i]
        hi = bins[i + 1]
        if hi == float("inf"):
            hi = max_vis
        centers.append((lo + hi) * 0.5)
    centers = np.array(centers, dtype=np.float32) / max_vis
    x = vis_m / max_vis
    temp = max(float(config.TIME_PRIOR_TEMP), 1e-3)
    logits = -((centers - x) ** 2) / (2 * temp * temp)
    logits = logits - logits.max()
    prob = np.exp(logits)
    prob = prob / (prob.sum() + 1e-8)
    return prob.astype(np.float32)


# ----------------------------------------------------------------
# 时间软对齐 (跨站点容错)
# ----------------------------------------------------------------
class TimeAligner:
    """
    跨站点时间对齐：优先站点匹配，失败则全站点时间回退。
    """

    def __init__(self, processor: NOAADataProcessor,
                 seq_len: int = config.SEQ_LEN,
                 max_time_gap_hours: float = 2.0):
        self.seq_len     = int(seq_len)
        # allow config override for cyclic alignment
        self.max_gap     = pd.Timedelta(
            hours=float(getattr(config, "TIME_ALIGN_MAX_GAP_HOURS", max_time_gap_hours))
        )
        self.processor   = processor
        self.scaler      = processor.scaler
        self.meteo_feats = [f for f in processor.features if f != "VIS_DISTANCE"]
        self.station_dfs: dict[str, pd.DataFrame] = {}
        self._build_index()
        self._printed_miss = False
        self._scaler_ready = self._check_scaler_ready()
        self._best_station_cache: dict[tuple[int, int], str] = {}
        # Cache config
        self._cache_enable = bool(getattr(config, "TIME_ALIGN_CACHE_ENABLE", False))
        self._cache_dir = getattr(config, "TIME_ALIGN_CACHE_DIR", None)
        self._cache_max_items = int(getattr(config, "TIME_ALIGN_CACHE_MAX_ITEMS", 0) or 0)
        self._cache_bucket_minutes = int(getattr(config, "TIME_ALIGN_CACHE_BUCKET_MINUTES", 60))
        self._cache_disk = bool(getattr(config, "TIME_ALIGN_CACHE_DISK", False))
        self._cache_mem = {}
        self._cache_order = []
        self._last_matched_station = None
        if self._cache_enable and self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
        print("[TimeAligner] cache_enable=", self._cache_enable, "cache_dir=", self._cache_dir)

    def _check_scaler_ready(self) -> bool:
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.scaler)
            return True
        except Exception:
            return False

    def _build_index(self):
        if self.processor.df is not None:
            for sid, sdf in self.processor.df.groupby("STATION"):
                self.station_dfs[sid] = sdf.set_index("datetime").sort_index()
        else:
            self._lazy_station_ids = list(self.processor.station_index.keys())
            self._loaded: dict[str, pd.DataFrame] = {}

    def _get_station_df(self, station_id: str) -> pd.DataFrame:
        if station_id in self.station_dfs:
            return self.station_dfs[station_id]
        if hasattr(self, "_lazy_station_ids"):
            if station_id in self._loaded:
                return self._loaded[station_id]
            if len(self._loaded) >= 4:
                old = next(iter(self._loaded))
                del self._loaded[old]
            sdf = self.processor.load_station(station_id)
            if len(sdf) > 0:
                sdf = sdf.set_index("datetime").sort_index()
            self._loaded[station_id] = sdf
            return sdf
        return pd.DataFrame()

    def _bucket_time(self, query_time: pd.Timestamp) -> pd.Timestamp:
        bucket = max(1, self._cache_bucket_minutes)
        ts = pd.Timestamp(query_time)
        minute = (ts.minute // bucket) * bucket
        return ts.replace(minute=minute, second=0, microsecond=0)

    def _cache_key(self, query_time: pd.Timestamp, station_id: str = None) -> str:
        ts = self._bucket_time(query_time)
        sid = station_id or "__any__"
        # Include seq_len to avoid reusing cached sequences across settings.
        return f"{sid}_T{self.seq_len}_{ts.strftime('%Y%m%d%H%M')}"

    def _cache_path(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"{key}.npz")

    def _cache_get(self, key: str):
        if not self._cache_enable:
            return None
        if key in self._cache_mem:
            return self._cache_mem[key]
        if self._cache_disk and self._cache_dir:
            path = self._cache_path(key)
            if os.path.exists(path):
                try:
                    data = np.load(path, allow_pickle=False)
                    seq_feat = data["seq_feat"]
                    time_feat_seq = data["time_feat_seq"]
                    avail = int(data["avail"])
                    return (seq_feat, time_feat_seq, avail)
                except Exception:
                    return None
        return None

    def _cache_put(self, key: str, seq_feat: np.ndarray,
                   time_feat_seq: np.ndarray, avail: int):
        if not self._cache_enable:
            return
        if self._cache_max_items > 0 and key not in self._cache_mem:
            self._cache_order.append(key)
            if len(self._cache_order) > self._cache_max_items:
                old = self._cache_order.pop(0)
                self._cache_mem.pop(old, None)
        self._cache_mem[key] = (seq_feat, time_feat_seq, int(avail))
        if self._cache_disk and self._cache_dir:
            path = self._cache_path(key)
            if not os.path.exists(path):
                try:
                    np.savez_compressed(path,
                                        seq_feat=seq_feat,
                                        time_feat_seq=time_feat_seq,
                                        avail=np.array([int(avail)], dtype=np.int32))
                except Exception:
                    pass

    def get_sequence(self, query_time: pd.Timestamp, station_id: str = None,
                     preferred_station_id: str = None):
        cache_key = None
        if self._cache_enable:
            sid_key = preferred_station_id or station_id or "__any__"
            cache_key = self._cache_key(query_time, sid_key)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._last_matched_station = sid_key if sid_key != "__any__" else None
                return cached

        # 空间最近站优先
        if preferred_station_id:
            sdf = self._get_station_df(preferred_station_id)
            if len(sdf) > 0:
                seq, tf, avail = self._extract_seq(sdf, query_time)
                if avail > 0:
                    self._last_matched_station = preferred_station_id
                    if cache_key is not None:
                        self._cache_put(cache_key, seq, tf, avail)
                    return seq, tf, avail

        if station_id:
            if station_id in (self.station_dfs if self.station_dfs else getattr(self, "_lazy_station_ids", [])):
                sdf = self._get_station_df(station_id)
                seq, tf, avail = self._extract_seq(sdf, query_time)
                if cache_key is not None:
                    self._cache_put(cache_key, seq, tf, avail)
                return seq, tf, avail

        all_stations = (
            list(self.station_dfs.keys())
            if self.station_dfs
            else getattr(self, "_lazy_station_ids", [])
        )
        # 随机抽样部分站点，避免全站点扫描（全局回退路径）
        limit = int(getattr(config, "TIME_ALIGN_RANDOM_STATIONS", 0) or 0)
        if limit > 0 and len(all_stations) > limit:
            all_stations = list(np.random.choice(all_stations, size=limit, replace=False))
        if not all_stations:
            seq = self._zero_seq()
            tf = self._build_time_feat(query_time)
            if cache_key is not None:
                self._cache_put(cache_key, seq, tf, 0)
            return seq, tf, 0

        # Cache best station by (month, hour) to avoid full scan each time.
        ts = pd.Timestamp(query_time)
        cache_best_key = (int(ts.month), int(ts.hour))
        cached_sid = self._best_station_cache.get(cache_best_key)
        if cached_sid is not None:
            sdf = self._get_station_df(cached_sid)
            if len(sdf) > 0:
                seq, tf, avail = self._extract_seq(sdf, query_time)
                if avail > 0:
                    if cache_key is not None:
                        self._cache_put(cache_key, seq, tf, avail)
                    return seq, tf, avail
            # Cached station is no longer usable, drop it.
            self._best_station_cache.pop(cache_best_key, None)

        best = None
        best_abs_dt = None
        best_sid = None
        for sid in all_stations:
            sdf = self._get_station_df(sid)
            if len(sdf) == 0:
                continue
            seq, tf, avail = self._extract_seq(sdf, query_time)
            if avail <= 0:
                continue
            # Estimate nearest time gap using cyclic delta if enabled.
            dt, _ = self._cyclic_delta_seconds(query_time, sdf.index)
            if len(dt) == 0:
                continue
            min_abs_dt = float(np.min(np.abs(dt)))
            if best_abs_dt is None or min_abs_dt < best_abs_dt:
                best_abs_dt = min_abs_dt
                best = (seq, tf, avail)
                best_sid = sid

        if best is not None:
            if best_sid is not None:
                self._best_station_cache[cache_best_key] = best_sid
            if cache_key is not None:
                self._cache_put(cache_key, best[0], best[1], best[2])
            return best

        seq = self._zero_seq()
        tf = self._build_time_feat(query_time)
        if cache_key is not None:
            self._cache_put(cache_key, seq, tf, 0)
        return seq, tf, 0

    def encode_cyclic_time(self, ts: pd.Timestamp):
        doy = ts.dayofyear
        sec = ts.hour * 3600 + ts.minute * 60 + ts.second
        return doy * 86400 + sec

    def _cyclic_delta_seconds(self, query_time: pd.Timestamp, times: pd.DatetimeIndex):
        # Map all timestamps onto a single-year cycle if enabled.
        if not getattr(config, "TIME_ALIGN_CYCLIC", False):
            dt = (query_time - times).total_seconds()
            return dt, dt >= 0

        query_scalar = self.encode_cyclic_time(query_time)
        time_scalar = times.map(self.encode_cyclic_time)
        dt = query_scalar - time_scalar

        year_seconds = float(getattr(config, "TIME_ALIGN_YEAR_SECONDS", 365 * 24 * 3600))
        if year_seconds > 0:
            dt = (dt + year_seconds / 2.0) % year_seconds - year_seconds / 2.0
        return dt, np.ones(len(times), dtype=bool)

    def temporal_kernel(self, dt_seconds):
        sigma = float(getattr(config, "TIME_ALIGN_SIGMA_SECONDS", config.TIME_KERNEL_SIGMA))
        return np.exp(-(dt_seconds ** 2) / (2 * sigma * sigma))

    def _extract_seq(self, sdf: pd.DataFrame, query_time: pd.Timestamp):
        if self.seq_len <= 0:
            return self._zero_seq(), self._build_time_feat(query_time), 0
        if len(sdf) == 0:
            if getattr(config, "TIME_ALIGN_TIME_ONLY_FALLBACK", False):
                return self._time_only_seq(query_time)
            return self._zero_seq(), self._build_time_feat(query_time), 0

        # 所有时间
        times = sdf.index

        dt, mask = self._cyclic_delta_seconds(query_time, times)
        if mask.sum() == 0:
            if getattr(config, "TIME_ALIGN_TIME_ONLY_FALLBACK", False):
                return self._time_only_seq(query_time)
            return self._zero_seq(), self._build_time_feat(query_time), 0

        valid_df = sdf.iloc[np.where(mask)[0]]

        # 核心：时间权重
        sigma = float(getattr(config, "TIME_ALIGN_SIGMA_SECONDS", config.TIME_KERNEL_SIGMA))
        weights = np.exp(-(dt ** 2) / (2 * sigma * sigma))

        # 取最近 seq_len 个
        idx = np.argsort(np.abs(dt))[:self.seq_len]

        dt = dt[idx]
        weights = weights[idx]
        past = valid_df.iloc[idx]

        # gap 判断
        if np.min(np.abs(dt)) > self.max_gap.total_seconds():
            if getattr(config, "TIME_ALIGN_TIME_ONLY_FALLBACK", False):
                return self._time_only_seq(query_time)
            return self._zero_seq(), self._build_time_feat(query_time), 0

        avail = int((weights > 1e-3).sum())
        if avail < self.seq_len:
            pad_rows = past.iloc[np.zeros(self.seq_len - avail, dtype=int)]
            past = pd.concat([pad_rows, past])

        if self.meteo_feats:
            meteo = past[[c for c in self.meteo_feats if c in past.columns]].values.astype(np.float32)
            col_means = np.nanmean(meteo, axis=0, keepdims=True)
            col_means = np.nan_to_num(col_means, nan=0.0)
            meteo = np.where(np.isnan(meteo), col_means, meteo)
            if meteo.shape[1] < len(self.meteo_feats):
                meteo = np.pad(meteo, ((0,0),(0,len(self.meteo_feats)-meteo.shape[1])))
            if self._scaler_ready:
                meteo = self.scaler.transform(meteo)
            else:
                # Fallback: avoid NotFittedError, keep raw meteo values
                meteo = np.nan_to_num(meteo, nan=0.0)
        else:
            meteo = np.zeros((self.seq_len, 1), dtype=np.float32)

        vis_col = past["VIS_DISTANCE"].values.astype(np.float32) \
            if "VIS_DISTANCE" in past.columns \
            else np.zeros(self.seq_len, dtype=np.float32)
        vis_norm = noaa_ds.normalize_vis(np.nan_to_num(vis_col, nan=0.0)).reshape(-1, 1)
        seq_feat = np.concatenate([meteo, vis_norm], axis=1).astype(np.float32)

        if len(past.index) == 0:
            time_feat_seq = self._build_time_feat(query_time)
        else:
            time_feat_seq = np.stack([
                noaa_ds.encode_time_features(pd.Timestamp(ts)) for ts in past.index
            ], axis=0).astype(np.float32)

        if getattr(config, "TIME_ALIGN_DEBUG", False):
            # One-time lightweight stats to verify seq_len and availability.
            if not hasattr(self, "_debug_samples"):
                self._debug_samples = 0
                self._debug_avail = []
            if self._debug_samples < int(getattr(config, "TIME_ALIGN_DEBUG_MAX_SAMPLES", 2000)):
                self._debug_samples += 1
                self._debug_avail.append(int(avail))
                if self._debug_samples == int(getattr(config, "TIME_ALIGN_DEBUG_MAX_SAMPLES", 2000)):
                    arr = np.array(self._debug_avail, dtype=np.int32)
                    if arr.size > 0:
                        msg = (
                            f"[TimeAligner][debug] seq_len={self.seq_len} "
                            f"avail_mean={arr.mean():.2f} "
                            f"p0={np.percentile(arr,0)} p25={np.percentile(arr,25)} "
                            f"p50={np.percentile(arr,50)} p75={np.percentile(arr,75)} p100={np.percentile(arr,100)}"
                        )
                        print(msg)

        return seq_feat, time_feat_seq, avail

    def get_future_vis(self, query_time, delta_hours: float,
                       station_id: str = None) -> float:
        """查询 t+Δt 时刻的能见度（米），失败返回 -1。"""
        if station_id is None:
            station_id = self._last_matched_station
        if not station_id:
            return -1.0
        future_time = pd.Timestamp(query_time) + pd.Timedelta(hours=float(delta_hours))
        sdf = self._get_station_df(station_id)
        if len(sdf) == 0:
            return -1.0
        dt, mask = self._cyclic_delta_seconds(future_time, sdf.index)
        if len(dt) == 0 or mask.sum() == 0:
            return -1.0
        idx = int(np.argmin(np.abs(dt)))
        if float(np.abs(dt[idx])) > self.max_gap.total_seconds():
            return -1.0
        if "VIS_DISTANCE" not in sdf.columns:
            return -1.0
        vis = float(sdf.iloc[idx]["VIS_DISTANCE"])
        return vis if np.isfinite(vis) and vis >= 0 else -1.0

    def _time_only_seq(self, query_time: pd.Timestamp):
        """Fallback: use capture-time features only (no meteo history)."""
        D = len(self.meteo_feats) + 1
        seq_feat = np.zeros((self.seq_len, D), dtype=np.float32)
        time_feat_seq = self._build_time_feat(query_time)
        avail = 0 if self.seq_len <= 0 else 1
        return seq_feat, time_feat_seq, avail

    def _zero_seq(self):
        D = len(self.meteo_feats) + 1
        return np.zeros((self.seq_len, D), dtype=np.float32)

    def _build_time_feat(self, query_time: pd.Timestamp):
        tf = noaa_ds.encode_time_features(query_time)
        return np.tile(tf, (self.seq_len, 1)).astype(np.float32)
