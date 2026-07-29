"""
NOAA气象数据处理模块 (流式加载版)
- 两遍扫描: 第一遍拟合Scaler, 第二遍构建站点级numpy缓存
- 每个站点独立缓存为 .npz 文件, 按需加载
- 彻底避免全量 pd.concat OOM

缓存目录结构:
  cache/noaa_stations/
    {station_id}.npz   每站点数组缓存
    scaler.pkl         全局StandardScaler
    index.json         站点列表 + 行数索引
"""
import os
import glob
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ----------------------------------------------------------------
# 工具函数 (不变)
# ----------------------------------------------------------------
def vis_to_class(vis_m: float) -> int:
    bins = config.VIS_BINS
    for i in range(len(bins) - 1):
        if bins[i] <= vis_m < bins[i + 1]:
            return i
    return len(bins) - 2


def normalize_vis(vis_m, max_vis: float = 50000.0):
    if isinstance(vis_m, np.ndarray):
        return np.clip(vis_m / max_vis, 0.0, 1.0).astype(np.float32)
    return float(min(float(vis_m) / max_vis, 1.0))


def encode_time_features(dt: pd.Timestamp) -> np.ndarray:
    """Encode time features to a fixed-length (10,) float32 vector.

    Robust to inputs like None/NaT, numpy datetime64, strings, and other
    pandas/numpy scalar types.

    Returns all-zeros if timestamp can't be parsed.
    """
    try:
        ts = pd.Timestamp(dt)
    except Exception:
        return np.zeros(TIME_FEAT_DIM, dtype=np.float32)

    if pd.isna(ts):
        return np.zeros(TIME_FEAT_DIM, dtype=np.float32)

    h = int(ts.hour)
    m = int(ts.month)
    dow = int(ts.dayofweek)

    # clamp just in case we ever get out-of-range values
    h = max(0, min(23, h))
    m = max(1, min(12, m))
    dow = max(0, min(6, dow))

    season_oh = np.zeros(4, dtype=np.float32)
    season_idx = int(((m - 1) // 3) % 4)
    season_oh[season_idx] = 1.0

    return np.array([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        *season_oh,
    ], dtype=np.float32)


def encode_time_array(datetimes: np.ndarray) -> np.ndarray:
    """批量时间编码, 返回 [N, 10]"""
    return np.stack([
        encode_time_features(pd.Timestamp(ts)) for ts in datetimes
    ], axis=0).astype(np.float32)


TIME_FEAT_DIM = 10


# ----------------------------------------------------------------
# 单站点 CSV 解析 (复用 noaa_preprocess 的字段逻辑)
# ----------------------------------------------------------------
_MISS_VIS = 999999
_MISS_TMP = 9999
_MISS_WND_DIR = 999
_MISS_WND_SPD = 9999
_MISS_CIG = 99999
_MISS_SLP = 99999


def _col0(series: pd.Series) -> pd.Series:
    """取逗号分隔的第0个子字段"""
    return series.str.split(",", expand=True).iloc[:, 0]


def _col3(series: pd.Series) -> pd.Series:
    return series.str.split(",", expand=True).iloc[:, 3]


def _parse_station_raw(path: str, station_id: str,
                       features: list) -> pd.DataFrame:
    """
    解析原始 NOAA ISD CSV → 标准 DataFrame
    支持两种格式:
      (A) 预处理后格式 (noaa_preprocess.py 输出): 列名即 VIS_DISTANCE 等
      (B) 原始 ISD 格式: 列名为 WND/CIG/VIS/TMP/DEW/SLP
    """
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return None
    df.columns = [c.strip().strip('"') for c in df.columns]

    # 时间解析
    date_col = "DATE" if "DATE" in df.columns else None
    if date_col is None:
        return None
    df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if len(df) == 0:
        return None

    out = pd.DataFrame({"datetime": df["datetime"]})
    out["STATION"] = station_id

    # ---- 格式判断 ----
    # 格式A: noaa_preprocess.py 输出，标准列名直接对应（有任意一个即认定）
    _PREPROCESSED_MARKERS = {
        "VIS_DISTANCE", "TEMPERATURE", "DEW_POINT",
        "WIND_DIRECTION", "WIND_SPEED", "CLOUD_HEIGHT", "SEA_LEVEL_PRESSURE"
    }
    is_preprocessed = bool(_PREPROCESSED_MARKERS & set(df.columns))

    if is_preprocessed:
        for col in features:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                out[col] = np.nan
        return out

    # ---- 格式B: 原始 ISD 编码格式 ----
    feat_map = {
        "VIS_DISTANCE":       _parse_vis_col,
        "TEMPERATURE":        _parse_tmp_col,
        "DEW_POINT":          _parse_dew_col,
        "WIND_DIRECTION":     _parse_wnd_dir_col,
        "WIND_SPEED":         _parse_wnd_spd_col,
        "CLOUD_HEIGHT":       _parse_cig_col,
        "SEA_LEVEL_PRESSURE": _parse_slp_col,
    }
    for col in features:
        fn = feat_map.get(col)
        if fn:
            out[col] = fn(df)
        else:
            out[col] = np.nan
    return out


def _parse_vis_col(df):
    if "VIS" not in df.columns: return np.nan
    raw = pd.to_numeric(_col0(df["VIS"]), errors="coerce")
    raw[raw == _MISS_VIS] = np.nan
    raw[(raw < 0) | (raw > 160934)] = np.nan
    return raw

def _parse_tmp_col(df):
    if "TMP" not in df.columns: return np.nan
    raw = pd.to_numeric(_col0(df["TMP"]), errors="coerce")
    raw[raw.abs() == _MISS_TMP] = np.nan
    return raw / 10.0

def _parse_dew_col(df):
    if "DEW" not in df.columns: return np.nan
    raw = pd.to_numeric(_col0(df["DEW"]), errors="coerce")
    raw[raw.abs() == _MISS_TMP] = np.nan
    return raw / 10.0

def _parse_wnd_dir_col(df):
    if "WND" not in df.columns: return np.nan
    raw = pd.to_numeric(_col0(df["WND"]), errors="coerce")
    raw[raw == _MISS_WND_DIR] = np.nan
    raw[(raw < 0) | (raw > 360)] = np.nan
    return raw

def _parse_wnd_spd_col(df):
    if "WND" not in df.columns: return np.nan
    raw = pd.to_numeric(_col3(df["WND"]), errors="coerce")
    raw[raw == _MISS_WND_SPD] = np.nan
    raw = raw / 10.0
    raw[(raw < 0) | (raw > 120)] = np.nan
    return raw

def _parse_cig_col(df):
    if "CIG" not in df.columns: return np.nan
    raw = pd.to_numeric(_col0(df["CIG"]), errors="coerce")
    raw[raw == _MISS_CIG] = np.nan
    return raw

def _parse_slp_col(df):
    if "SLP" not in df.columns: return np.nan
    raw = pd.to_numeric(_col0(df["SLP"]), errors="coerce")
    raw[raw == _MISS_SLP] = np.nan
    raw = raw / 10.0
    raw[(raw < 870) | (raw > 1084)] = np.nan
    return raw


# ----------------------------------------------------------------
# 站点级缺失值填充
# ----------------------------------------------------------------
def _interpolate_station(df: pd.DataFrame, features: list,
                         max_gap: int = 6,
                         station_id: str = "") -> pd.DataFrame:
    """线性插值, 最大连续填补 max_gap 小时"""
    # 兼容 datetime 既是普通列也可能已是索引的情况
    if "datetime" in df.columns:
        df = df.set_index("datetime").sort_index()
    elif df.index.name == "datetime":
        df = df.sort_index()
    else:
        # 尝试把第一个 datetime-like 索引/列作为时间轴
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.set_index(col).sort_index()
                df.index.name = "datetime"
                break
        else:
            raise KeyError("找不到 datetime 列，无法插值")

    # 去重 (同一时刻多行取均值)
    num_cols = [c for c in features if c in df.columns]
    df_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df_num = df_num.resample("h").mean()
    df_num = df_num.reindex(
        pd.date_range(df_num.index.min(), df_num.index.max(), freq="h")
    )

    # pandas 2.x: limit 不能大于序列长度，动态计算安全上界
    safe_limit = max(1, min(max_gap, max(1, len(df_num) - 1)))

    # 风向循环插值
    if "WIND_DIRECTION" in df_num.columns:
        wd_rad = np.deg2rad(df_num["WIND_DIRECTION"])
        sin_wd = np.sin(wd_rad).interpolate(
            method="linear", limit=safe_limit, limit_direction="both")
        cos_wd = np.cos(wd_rad).interpolate(
            method="linear", limit=safe_limit, limit_direction="both")
        df_num["WIND_DIRECTION"] = np.rad2deg(np.arctan2(sin_wd, cos_wd)) % 360

    # 气象特征（非VIS）用更宽松的插值窗口（最多48h），减少缺失
    # VIS_DISTANCE 严格使用 safe_limit，避免过度插值引入噪声
    meteo_others = [c for c in num_cols
                    if c not in ("WIND_DIRECTION", "VIS_DISTANCE")]
    if meteo_others:
        meteo_limit = max(1, min(48, max(1, len(df_num) - 1)))
        df_num[meteo_others] = df_num[meteo_others].interpolate(
            method="linear", limit=meteo_limit, limit_direction="both")

    if "VIS_DISTANCE" in df_num.columns:
        df_num[["VIS_DISTANCE"]] = df_num[["VIS_DISTANCE"]].interpolate(
            method="linear", limit=safe_limit, limit_direction="both")

    # 优先用传入的 station_id（文件名），其次用数据列（可能为空字符串）
    _sid = station_id or (df["STATION"].iloc[0] if "STATION" in df.columns else "")
    df_num["STATION"] = _sid
    df_num = df_num.reset_index()
    # reset_index 后时间列名可能是 "datetime" 或原索引名，统一重命名
    if df_num.columns[0] != "datetime":
        df_num = df_num.rename(columns={df_num.columns[0]: "datetime"})
    return df_num


# ----------------------------------------------------------------
# NOAADataProcessor: 流式两遍扫描
# ----------------------------------------------------------------
class NOAADataProcessor:
    """
    流式加载 NOAA 数据:
      第一遍: 逐站点解析 → 在线拟合 StandardScaler (IncrementalPCA 思路)
      第二遍: 逐站点处理 → 写入 .npz 缓存, 不在内存保留整体 df

    self.df = None  (不再保留全量 DataFrame)
    通过 self.station_index 知道有哪些站点
    """

    def __init__(self,
                 noaa_root:  str  = config.NOAA_ROOT,
                 features:   list = config.NOAA_FEATURES,
                 cache_dir:  str  = os.path.join(config.CACHE_DIR, "noaa_stations"),
                 max_gap_h:  int  = 6,
                 drop_no_vis:bool = False):
        self.noaa_root   = noaa_root
        self.features    = features
        self.cache_dir   = cache_dir
        self.max_gap_h   = max_gap_h
        self.drop_no_vis = drop_no_vis
        self.meteo_feats = [f for f in features if f != "VIS_DISTANCE"]

        self.scaler        = StandardScaler()
        self.station_index = {}   # {station_id: npz_path}
        self.vis_prior     = {}   # {(hour, month): mean_vis}
        self.df            = None # 兼容接口, 实际不保留全量df

        os.makedirs(cache_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 公共入口
    # ----------------------------------------------------------
    def load(self, force_reload: bool = False) -> "NOAADataProcessor":
        index_path  = os.path.join(self.cache_dir, "index.json")
        scaler_path = os.path.join(self.cache_dir, "scaler.pkl")
        prior_path  = os.path.join(self.cache_dir, "vis_prior.pkl")

        if not force_reload and os.path.exists(index_path) and os.path.exists(scaler_path):
            print(f"[NOAAProcessor] 从缓存加载: {self.cache_dir}")
            with open(index_path)      as f: self.station_index = json.load(f)
            with open(scaler_path, "rb") as f: self.scaler = pickle.load(f)
            if os.path.exists(prior_path):
                with open(prior_path, "rb") as f: self.vis_prior = pickle.load(f)

            # ---- 自动修复: 扫描缓存目录，把 index.json 遗漏的 .npz 补入 ----
            n_before = len(self.station_index)
            all_npz  = glob.glob(os.path.join(self.cache_dir, "*.npz"))
            added = 0
            for npz_path in all_npz:
                sid = os.path.splitext(os.path.basename(npz_path))[0]
                if sid not in self.station_index:
                    self.station_index[sid] = npz_path
                    added += 1

            # 同时移除 index.json 里路径已不存在的条目
            removed = 0
            dead = [sid for sid, p in self.station_index.items()
                    if not os.path.exists(p)]
            for sid in dead:
                del self.station_index[sid]
                removed += 1

            if added > 0 or removed > 0:
                print(f"[NOAAProcessor] 索引自动修复: +{added} 补入  -{removed} 清理  "
                      f"({n_before} → {len(self.station_index)} 站点)")
                # 回写修复后的 index.json
                with open(index_path, "w") as f:
                    json.dump(self.station_index, f)

            print(f"[NOAAProcessor] {len(self.station_index)} 个站点已加载")

            if not self.is_scaler_fitted():
                print("[NOAAProcessor] 警告: Scaler 未拟合，尝试从缓存重建")
                if self._fit_scaler_from_cache():
                    with open(scaler_path, "wb") as f:
                        pickle.dump(self.scaler, f)

            self._build_compat_df_stub()
            return self

        # allow index-only cache and rebuild scaler if missing
        if not force_reload and os.path.exists(index_path) and not os.path.exists(scaler_path):
            print(f"[NOAAProcessor] 仅索引缓存存在: {self.cache_dir}")
            with open(index_path) as f:
                self.station_index = json.load(f)
            if self._fit_scaler_from_cache():
                with open(scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)
            self._build_compat_df_stub()
            return self

        # ---- 首次构建或强制重建 ----
        all_csv = self._collect_csv_files()
        if not all_csv:
            raise RuntimeError(
                f"未找到任何CSV文件, 请检查路径: {self.noaa_root}\n"
                f"期望结构: {self.noaa_root}/2017/*.csv  2018/*.csv ..."
            )
        print(f"[NOAAProcessor] 找到 {len(all_csv)} 个站点文件, 开始处理...")

        # ---- 第一遍: 在线拟合 Scaler ----
        # 若 scaler 已存在（上次中断后保留），直接复用，跳过第一遍
        if not force_reload and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("[NOAAProcessor] 复用已有 Scaler，跳过第一遍扫描")
        else:
            self._fit_scaler_online(all_csv)  # all_csv 现在是 station_groups
            # 提前保存 Scaler，中断后可复用
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

        # ---- 第二遍: 构建站点 npz 缓存 ----
        # 断点续跑: 已有 .npz 的站点直接加入索引，跳过重新生成
        n_ok = n_skip = n_new = 0
        prior_accum = {}

        for station_id, csv_paths in all_csv:
            npz_path = os.path.join(self.cache_dir, f"{station_id}.npz")
            if not force_reload and os.path.exists(npz_path):
                self.station_index[station_id] = npz_path
                n_ok += 1
                continue

            result = self._process_one_station(
                csv_paths, station_id, npz_path, prior_accum
            )
            if result:
                self.station_index[station_id] = npz_path
                n_ok += 1; n_new += 1
            else:
                n_skip += 1

            # 每处理 1000 个新站点就中间保存一次索引，防止中断丢失
            if n_new > 0 and n_new % 1000 == 0:
                with open(index_path, "w") as f:
                    json.dump(self.station_index, f)

        # 计算 vis_prior
        self.vis_prior = {k: float(np.mean(v))
                          for k, v in prior_accum.items() if v}

        # 最终保存
        with open(index_path,  "w") as f: json.dump(self.station_index, f)
        with open(scaler_path, "wb") as f: pickle.dump(self.scaler, f)
        with open(prior_path,  "wb") as f: pickle.dump(self.vis_prior, f)

        print(f"[NOAAProcessor] 完成: {n_ok} 站点 "
              f"(新建 {n_new}, 已有 {n_ok - n_new}), 跳过 {n_skip}")
        self._build_compat_df_stub()
        return self

    # ----------------------------------------------------------
    # 内部: 收集文件列表
    # ----------------------------------------------------------
    def _collect_csv_files(self):
        """
        按站点ID分组，返回 {station_id: [csv_path_year1, csv_path_year2, ...]}
        同一站点多年数据合并处理，生成一个跨年连续的 npz
        """
        station_files: dict = {}   # {sid: [path_sorted_by_year]}
        for year_dir in sorted(glob.glob(os.path.join(self.noaa_root, "*"))):
            if not os.path.isdir(year_dir):
                continue
            for csv_file in sorted(glob.glob(os.path.join(year_dir, "*.csv"))):
                sid = os.path.splitext(os.path.basename(csv_file))[0]
                if sid not in station_files:
                    station_files[sid] = []
                station_files[sid].append(csv_file)

        # Flat layout: {root}/{station_id}.csv
        for csv_file in sorted(glob.glob(os.path.join(self.noaa_root, "*.csv"))):
            basename = os.path.basename(csv_file)
            if basename.lower() in ("stations_meta.csv",):
                continue
            sid = os.path.splitext(basename)[0]
            if sid not in station_files:
                station_files[sid] = []
            if csv_file not in station_files[sid]:
                station_files[sid].append(csv_file)

        # 返回 list of (station_id, [csv_paths])，年份已排序
        return [(sid, paths) for sid, paths in station_files.items()]

    # ----------------------------------------------------------
    # 内部: 在线拟合 Scaler (逐站点累积均值/方差)
    # ----------------------------------------------------------
    def _fit_scaler_online(self, station_groups):
        """
        station_groups: list of (station_id, [csv_paths])
        每个站点只处理一次（跨年合并后取样），避免重复计数
        """
        if not self.meteo_feats:
            return
        n_stations = len(station_groups)
        print(f"[NOAAProcessor] 第一遍: 拟合 Scaler ({n_stations} 个唯一站点)...")
        from sklearn.preprocessing import StandardScaler as SS
        sc = SS()
        fitted = False
        for i, (station_id, csv_paths) in enumerate(station_groups):
            try:
                # 只取第一个年份文件做 Scaler 估计（代表性样本，节省时间）
                # 若需更精确可改为全部年份，但会慢 8x
                df = _parse_station_raw(csv_paths[0], station_id, self.features)
                if df is None or len(df) == 0:
                    continue
                vals = df[self.meteo_feats].apply(
                    pd.to_numeric, errors="coerce"
                ).dropna(how="all").values.astype(np.float32)
                if len(vals) == 0:
                    continue
                mask = ~np.isnan(vals).any(axis=1)
                vals = vals[mask]
                if len(vals) == 0:
                    continue
                sc.partial_fit(vals)
                fitted = True
            except Exception:
                continue
            if (i + 1) % 500 == 0:
                print(f"  Scaler: {i+1}/{n_stations}")
        if fitted:
            # 修复 scale_=0 的特征（常数列），设为1.0避免除零 → inf
            if hasattr(sc, 'scale_') and sc.scale_ is not None:
                zero_mask = sc.scale_ < 1e-6
                if zero_mask.any():
                    import warnings
                    feats_zero = [f for f, z in zip(self.meteo_feats, zero_mask) if z]
                    warnings.warn(f"Scaler: {feats_zero} 的 scale 接近0，已设为1.0（常数列）")
                    sc.scale_[zero_mask] = 1.0
                    sc.var_[zero_mask]   = 1.0
            self.scaler = sc
        print(f"  Scaler 拟合完成 (基于 {n_stations} 个站点的首年数据)")

    # ----------------------------------------------------------
    # 内部: 处理单个站点并写入 npz
    # ----------------------------------------------------------
    def _process_one_station(self, csv_paths, station_id,
                             npz_path, prior_accum) -> bool:
        """
        csv_paths: 该站点所有年份的 CSV 路径列表（已按年份排序）
        多年数据 concat 后作为完整时间序列处理
        """
        try:
            # 逐年解析后合并，保证时间连续性
            year_dfs = []
            for csv_path in csv_paths:
                df_y = _parse_station_raw(csv_path, station_id, self.features)
                if df_y is not None and len(df_y) > 0:
                    year_dfs.append(df_y)
            if not year_dfs:
                return False

            # 按时间排序合并，去掉年份边界的重复时间戳
            df = pd.concat(year_dfs, ignore_index=True)
            df = df.sort_values("datetime").drop_duplicates(
                subset=["datetime"]
            ).reset_index(drop=True)

            df = _interpolate_station(df, self.features,
                                       self.max_gap_h, station_id=station_id)
            if df is None or len(df) == 0:
                return False

            # VIS 全缺失处理
            vis_col = df.get("VIS_DISTANCE",
                             pd.Series(np.nan, index=df.index))
            if isinstance(vis_col, pd.Series):
                vis_valid = vis_col.notna().sum()
            else:
                vis_valid = 0

            if vis_valid == 0 and self.drop_no_vis:
                return False

            # 能见度先验统计（向量化，避免 iterrows 在大站点时卡顿）
            if "VIS_DISTANCE" in df.columns and "datetime" in df.columns:
                vis_rows = df[df["VIS_DISTANCE"].notna()].copy()
                if len(vis_rows) > 0:
                    vis_rows["_h"]  = vis_rows["datetime"].dt.hour
                    vis_rows["_mo"] = vis_rows["datetime"].dt.month
                    for (h, mo), grp in vis_rows.groupby(["_h", "_mo"]):
                        key = (int(h), int(mo))
                        if key not in prior_accum:
                            prior_accum[key] = []
                        prior_accum[key].extend(grp["VIS_DISTANCE"].tolist())

            # 转为 numpy 数组存储
            datetimes = df["datetime"].values.astype("datetime64[ns]")
            timestamps = datetimes.astype(np.int64)   # unix ns → int64

            feat_arrays = {}
            for col in self.features:
                if col in df.columns:
                    feat_arrays[col] = df[col].values.astype(np.float32)
                else:
                    feat_arrays[col] = np.full(len(df), np.nan, dtype=np.float32)

            np.savez_compressed(
                npz_path,
                timestamps=timestamps,
                **feat_arrays,
            )
            return True

        except Exception as e:
            return False

    # ----------------------------------------------------------
    # 兼容旧接口: 构建 df 占位符 (不加载实际数据)
    # ----------------------------------------------------------
    def _build_compat_df_stub(self):
        """
        旧代码通过 processor.df.groupby("STATION") 遍历数据。
        新版中 self.df = None，NOAASequenceDataset 已改为
        通过 station_index 直接读 npz，不再依赖 self.df。
        此方法保持向后兼容, 仅供 get_vis_prior 等方法使用。
        """
        self.df = None   # 明确标记: 不再保留全量 df

    # ----------------------------------------------------------
    # 读取单个站点 npz → DataFrame
    # ----------------------------------------------------------
    def load_station(self, station_id: str) -> pd.DataFrame:
        """按需加载单个站点数据, 返回 DataFrame"""
        npz_path = self.station_index.get(station_id)
        if npz_path is None or not os.path.exists(npz_path):
            return pd.DataFrame()
        data = np.load(npz_path, allow_pickle=False)
        df = pd.DataFrame()
        df["datetime"] = pd.to_datetime(
            data["timestamps"].astype("datetime64[ns]")
        )
        df["STATION"] = station_id
        for col in self.features:
            if col in data:
                df[col] = data[col]
            else:
                df[col] = np.nan
        return df

    # ----------------------------------------------------------
    # 能见度先验 (推理阶段使用)
    # ----------------------------------------------------------
    def get_vis_prior(self, station_id: str = None) -> dict:
        return self.vis_prior

    # ----------------------------------------------------------
    # 兼容 TimeAligner 需要的多站点数据访问
    # ----------------------------------------------------------
    def iter_stations(self):
        """迭代所有站点: yield (station_id, df)"""
        for sid in self.station_index:
            df = self.load_station(sid)
            if len(df) > 0:
                yield sid, df

    def is_scaler_fitted(self) -> bool:
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.scaler)
            return True
        except Exception:
            return False

    def _fit_scaler_from_cache(self, max_stations: int = 200) -> bool:
        if not self.meteo_feats or not self.station_index:
            return False
        from sklearn.preprocessing import StandardScaler as SS
        sc = SS()
        fitted = False
        for i, npz_path in enumerate(self.station_index.values()):
            try:
                data = np.load(npz_path, allow_pickle=False, mmap_mode="r")
                cols = [data[f] for f in self.meteo_feats if f in data]
                if not cols:
                    continue
                vals = np.stack(cols, axis=1).astype(np.float32)
                if len(vals) == 0:
                    continue
                step = max(1, len(vals) // 5000)
                vals = vals[::step]
                mask = ~np.isnan(vals).any(axis=1)
                vals = vals[mask]
                if len(vals) == 0:
                    continue
                sc.partial_fit(vals)
                fitted = True
            except Exception:
                continue
            if max_stations and (i + 1) >= max_stations:
                break

        if fitted:
            if hasattr(sc, "scale_") and sc.scale_ is not None:
                zero_mask = sc.scale_ < 1e-6
                if zero_mask.any():
                    sc.scale_[zero_mask] = 1.0
                    sc.var_[zero_mask]   = 1.0
            self.scaler = sc
        return fitted
