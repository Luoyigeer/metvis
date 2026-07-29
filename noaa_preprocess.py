"""
NOAA ISD (Integrated Surface Database) 批量预处理脚本

将原始 NOAA ISD 格式 CSV 转换为 noaa_dataset.py 可直接读取的标准格式。

原始字段编码规则 (NOAA ISD 文档):
  WND : direction(°), quality, obs_type, speed(×0.1 m/s), speed_quality
        缺失值: direction=999, speed=9999
  CIG : ceiling_height(m), quality, determination, cavok
        缺失值: 99999
  VIS : distance(m), quality, variability, variability_quality
        缺失值: 999999
  TMP : signed_temp(×0.1 °C), quality
        缺失值: +9999
  DEW : signed_dewpoint(×0.1 °C), quality
        缺失值: +9999
  SLP : sea_level_pressure(×0.1 hPa), quality
        缺失值: 99999

输出列 (与 noaa_dataset.py 中 config.NOAA_FEATURES 一一对应):
  STATION, DATE(datetime字符串), LATITUDE, LONGITUDE, ELEVATION, NAME,
  VIS_DISTANCE, TEMPERATURE, DEW_POINT,
  WIND_DIRECTION, WIND_SPEED, CLOUD_HEIGHT, SEA_LEVEL_PRESSURE

处理策略:
  1. 逐站点解析, 单位换算, 缺失值统一置 NaN
  2. 站点内时间排序 + 线性插值 (最大连续插值 6 小时)
  3. 若某特征在整个站点全缺失 → 跳过该特征列 (不影响其他特征)
  4. VIS_DISTANCE 全缺失的站点 → 可选保留(用于气象预训练)或丢弃
  5. 输出到 <output_dir>/<year>/<station_id>.csv, 与原目录结构相同

用法:
  # 处理所有年份，输出到 data/noaa_processed/
  python noaa_preprocess.py --input_dir data/noaa --output_dir data/noaa_processed

  # 只处理 2019-2022，跳过 VIS 全缺失站点
  python noaa_preprocess.py --input_dir data/noaa --output_dir data/noaa_processed \\
      --years 2019 2020 2021 2022 --drop_no_vis

  # 快速验证（每年只处理前 5 个站点）
  python noaa_preprocess.py --input_dir data/noaa --output_dir data/noaa_processed \\
      --dry_run --max_stations 5
"""

import os
import sys
import glob
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        items = list(iterable)
        total = len(items)
        for i, item in enumerate(items):
            if total > 0 and i % max(1, total // 10) == 0:
                print(f"    {desc} {i}/{total}", flush=True)
            yield item

warnings.filterwarnings("ignore")

# ================================================================
# 字段解析函数
# ================================================================

def _split_field(series: pd.Series, idx: int) -> pd.Series:
    """按逗号分割，取第 idx 个子字段，返回字符串 Series"""
    return series.str.split(",", expand=True).iloc[:, idx]


def parse_vis(vis_col: pd.Series) -> pd.Series:
    """
    VIS 字段 → 能见度(米)
    格式: distance,quality,variability,variability_quality
    缺失: distance == 999999
    """
    raw = pd.to_numeric(_split_field(vis_col, 0), errors="coerce")
    raw[raw == 999999] = np.nan
    # NOAA ISD 能见度单位直接是米，无需换算
    # 有效范围 0~160934m (100 miles)，超出视为异常
    raw[(raw < 0) | (raw > 160934)] = np.nan
    return raw


def parse_tmp(tmp_col: pd.Series) -> pd.Series:
    """
    TMP / DEW 字段 → 温度(°C)
    格式: signed_value(×0.1°C),quality
    缺失: value == +9999 or -9999 (即数值 9999 或 -9999)
    """
    raw = pd.to_numeric(_split_field(tmp_col, 0), errors="coerce")
    raw[raw.abs() == 9999] = np.nan
    return raw / 10.0   # ×0.1°C → °C


def parse_wnd_direction(wnd_col: pd.Series) -> pd.Series:
    """
    WND 字段 → 风向(°)
    格式: direction,quality,obs_type,speed,speed_quality
    缺失: direction == 999
    有效范围: 0~360°
    """
    raw = pd.to_numeric(_split_field(wnd_col, 0), errors="coerce")
    raw[raw == 999] = np.nan
    raw[(raw < 0) | (raw > 360)] = np.nan
    return raw


def parse_wnd_speed(wnd_col: pd.Series) -> pd.Series:
    """
    WND 字段 → 风速(m/s)
    格式: direction,quality,obs_type,speed(×0.1 m/s),speed_quality
    缺失: speed == 9999
    """
    raw = pd.to_numeric(_split_field(wnd_col, 3), errors="coerce")
    raw[raw == 9999] = np.nan
    raw = raw / 10.0   # ×0.1 m/s → m/s
    raw[(raw < 0) | (raw > 120)] = np.nan   # 120 m/s 为合理上限
    return raw


def parse_cig(cig_col: pd.Series) -> pd.Series:
    """
    CIG 字段 → 云底高度(m)
    格式: height,quality,determination,cavok
    缺失: height == 99999
    22000m = "无云/无法测量" 特殊值，保留（表示晴天）
    """
    raw = pd.to_numeric(_split_field(cig_col, 0), errors="coerce")
    raw[raw == 99999] = np.nan
    raw[(raw < 0) & (raw != 22000)] = np.nan
    return raw


def parse_slp(slp_col: pd.Series) -> pd.Series:
    """
    SLP 字段 → 海平面气压(hPa)
    格式: value(×0.1 hPa),quality
    缺失: value == 99999
    有效范围: 870~1084 hPa
    """
    raw = pd.to_numeric(_split_field(slp_col, 0), errors="coerce")
    raw[raw == 99999] = np.nan
    raw = raw / 10.0   # ×0.1 hPa → hPa
    raw[(raw < 870) | (raw > 1084)] = np.nan
    return raw


# ================================================================
# 单站点解析
# ================================================================

def parse_station_file(path: str) -> Optional[pd.DataFrame]:
    """
    解析单个站点 CSV → 标准 DataFrame
    返回列: STATION, DATE, LATITUDE, LONGITUDE, ELEVATION, NAME,
            VIS_DISTANCE, TEMPERATURE, DEW_POINT,
            WIND_DIRECTION, WIND_SPEED, CLOUD_HEIGHT, SEA_LEVEL_PRESSURE
    失败返回 None
    """
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception as e:
        return None

    df.columns = [c.strip().strip('"') for c in df.columns]

    # ---- 时间 ----
    if "DATE" not in df.columns:
        return None
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    if len(df) == 0:
        return None

    # ---- 站点元信息 ----
    station_id = str(df["STATION"].iloc[0]).strip() if "STATION" in df.columns \
        else Path(path).stem
    lat  = pd.to_numeric(df["LATITUDE"].iloc[0],  errors="coerce") if "LATITUDE"  in df.columns else np.nan
    lon  = pd.to_numeric(df["LONGITUDE"].iloc[0], errors="coerce") if "LONGITUDE" in df.columns else np.nan
    elev = pd.to_numeric(df["ELEVATION"].iloc[0], errors="coerce") if "ELEVATION" in df.columns else np.nan
    name = df["NAME"].iloc[0].strip() if "NAME" in df.columns else ""

    # ---- 解析气象字段 ----
    out = pd.DataFrame()
    out["STATION"]           = station_id
    out["DATE"]              = df["DATE"]
    out["LATITUDE"]          = lat
    out["LONGITUDE"]         = lon
    out["ELEVATION"]         = elev
    out["NAME"]              = name

    out["VIS_DISTANCE"]      = parse_vis(df["VIS"])        if "VIS" in df.columns \
                                else np.nan
    out["TEMPERATURE"]       = parse_tmp(df["TMP"])        if "TMP" in df.columns \
                                else np.nan
    out["DEW_POINT"]         = parse_tmp(df["DEW"])        if "DEW" in df.columns \
                                else np.nan
    out["WIND_DIRECTION"]    = parse_wnd_direction(df["WND"]) if "WND" in df.columns \
                                else np.nan
    out["WIND_SPEED"]        = parse_wnd_speed(df["WND"])  if "WND" in df.columns \
                                else np.nan
    out["CLOUD_HEIGHT"]      = parse_cig(df["CIG"])        if "CIG" in df.columns \
                                else np.nan
    out["SEA_LEVEL_PRESSURE"]= parse_slp(df["SLP"])        if "SLP" in df.columns \
                                else np.nan

    out = out.sort_values("DATE").reset_index(drop=True)
    return out


# ================================================================
# 缺失值插值
# ================================================================

METEO_COLS = [
    "VIS_DISTANCE", "TEMPERATURE", "DEW_POINT",
    "WIND_DIRECTION", "WIND_SPEED", "CLOUD_HEIGHT", "SEA_LEVEL_PRESSURE"
]

def interpolate_station(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    """
    对单站点数据进行时间线性插值
    - 先将 DATE 设为索引，resample 到整点小时（补全缺失时刻）
    - 对每列做线性插值，最大连续插值窗口 = max_gap_hours
    - 风向使用循环插值（0°/360° 边界处理）
    """
    df = df.copy()
    df = df.set_index("DATE")

    # 去重（同一时刻多条记录取均值）
    numeric_cols = [c for c in METEO_COLS if c in df.columns]
    meta_cols    = ["STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME"]

    # 元信息列取第一条
    meta = df[meta_cols].iloc[[0]].reindex(
        pd.date_range(df.index.min(), df.index.max(), freq="h")
    ).ffill()

    # 数值列取均值后 resample
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    num_df = num_df.resample("h").mean()     # 整点小时重采样
    num_df = num_df.reindex(
        pd.date_range(num_df.index.min(), num_df.index.max(), freq="h")
    )

    # 风向循环插值 (sin/cos 分量插值后还原)
    if "WIND_DIRECTION" in num_df.columns:
        wd_rad  = np.deg2rad(num_df["WIND_DIRECTION"])
        sin_wd  = np.sin(wd_rad).interpolate(
            method="linear", limit=max_gap_hours, limit_direction="both"
        )
        cos_wd  = np.cos(wd_rad).interpolate(
            method="linear", limit=max_gap_hours, limit_direction="both"
        )
        num_df["WIND_DIRECTION"] = np.rad2deg(np.arctan2(sin_wd, cos_wd)) % 360

    # 其余列线性插值
    other_cols = [c for c in numeric_cols if c != "WIND_DIRECTION"]
    num_df[other_cols] = num_df[other_cols].interpolate(
        method="linear", limit=max_gap_hours, limit_direction="both"
    )

    # 合并
    result = num_df.copy()
    result["STATION"]   = df["STATION"].iloc[0]
    result["LATITUDE"]  = lat = df["LATITUDE"].iloc[0]  if "LATITUDE"  in df.columns else np.nan
    result["LONGITUDE"] = df["LONGITUDE"].iloc[0] if "LONGITUDE" in df.columns else np.nan
    result["ELEVATION"] = df["ELEVATION"].iloc[0] if "ELEVATION" in df.columns else np.nan
    result["NAME"]      = df["NAME"].iloc[0]       if "NAME"      in df.columns else ""
    result = result.reset_index().rename(columns={"index": "DATE"})

    return result


# ================================================================
# 批量处理主函数
# ================================================================

def process_year(year_dir: str, output_year_dir: str,
                 drop_no_vis: bool = False,
                 max_gap_hours: int = 6,
                 max_stations: Optional[int] = None) -> dict:
    """
    处理单个年份目录下的所有站点 CSV
    返回统计信息字典
    """
    os.makedirs(output_year_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(year_dir, "*.csv")))
    if max_stations:
        csv_files = csv_files[:max_stations]

    stats = {
        "total":         len(csv_files),
        "success":       0,
        "skip_no_data":  0,
        "skip_no_vis":   0,
        "skip_error":    0,
        "vis_available": 0,   # VIS 非全缺失的站点数
    }

    for csv_file in tqdm(csv_files, desc=f"  {os.path.basename(year_dir)}", leave=False):
        station_id = Path(csv_file).stem
        out_path   = os.path.join(output_year_dir, f"{station_id}.csv")

        # 跳过已处理的文件
        if os.path.exists(out_path):
            stats["success"] += 1
            continue

        try:
            df = parse_station_file(csv_file)
            if df is None or len(df) == 0:
                stats["skip_no_data"] += 1
                continue

            # VIS 全缺失处理
            vis_valid = df["VIS_DISTANCE"].notna().sum()
            if vis_valid == 0:
                stats["skip_no_vis"] += 1
                if drop_no_vis:
                    continue   # 丢弃此站点
                # 保留 (其他气象特征可用于预训练)
            else:
                stats["vis_available"] += 1

            # 插值
            df = interpolate_station(df, max_gap_hours=max_gap_hours)

            # 输出列顺序与 noaa_dataset.py 期望完全一致
            out_cols = [
                "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
                "VIS_DISTANCE", "TEMPERATURE", "DEW_POINT",
                "WIND_DIRECTION", "WIND_SPEED", "CLOUD_HEIGHT", "SEA_LEVEL_PRESSURE"
            ]
            df = df[[c for c in out_cols if c in df.columns]]

            # DATE 格式化为字符串（noaa_dataset 用 pd.to_datetime 解析）
            df["DATE"] = df["DATE"].dt.strftime("%Y/%m/%d %H:%M")

            df.to_csv(out_path, index=False)
            stats["success"] += 1

        except Exception as e:
            stats["skip_error"] += 1
            # uncomment for debug:
            # import traceback; traceback.print_exc()

    return stats


def run_all(input_dir: str, output_dir: str,
            years: Optional[list] = None,
            drop_no_vis: bool = False,
            max_gap_hours: int = 6,
            max_stations: Optional[int] = None,
            dry_run: bool = False):
    """
    遍历所有年份目录批量处理
    """
    year_dirs = sorted(glob.glob(os.path.join(input_dir, "*")))
    year_dirs = [d for d in year_dirs if os.path.isdir(d)]

    if years:
        year_dirs = [d for d in year_dirs
                     if os.path.basename(d) in [str(y) for y in years]]

    if not year_dirs:
        print(f"[错误] 未找到年份目录: {input_dir}")
        print("期望结构: input_dir/2017/*.csv  input_dir/2018/*.csv  ...")
        sys.exit(1)

    print(f"[批量处理] 共 {len(year_dirs)} 个年份: "
          f"{[os.path.basename(d) for d in year_dirs]}")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print(f"  丢弃VIS全缺失站点: {drop_no_vis}")
    print(f"  最大插值窗口: {max_gap_hours}h")
    if dry_run:
        print(f"  [DRY RUN] 每年最多处理 {max_stations or 5} 个站点")
        max_stations = max_stations or 5

    total_stats = {
        "total": 0, "success": 0,
        "skip_no_data": 0, "skip_no_vis": 0,
        "skip_error": 0, "vis_available": 0
    }

    for year_dir in year_dirs:
        year = os.path.basename(year_dir)
        out_year_dir = os.path.join(output_dir, year)
        stats = process_year(
            year_dir, out_year_dir,
            drop_no_vis=drop_no_vis,
            max_gap_hours=max_gap_hours,
            max_stations=max_stations,
        )
        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"  [{year}] 成功={stats['success']}  "
              f"跳过(无数据)={stats['skip_no_data']}  "
              f"跳过(无VIS)={stats['skip_no_vis']}  "
              f"错误={stats['skip_error']}  "
              f"有效VIS站点={stats['vis_available']}")

    print("\n" + "=" * 60)
    print("处理完成汇总")
    print("=" * 60)
    print(f"  总站点文件数:       {total_stats['total']}")
    print(f"  成功输出:           {total_stats['success']}")
    print(f"  跳过(空数据):       {total_stats['skip_no_data']}")
    print(f"  跳过(VIS全缺失):    {total_stats['skip_no_vis']}")
    print(f"  解析错误:           {total_stats['skip_error']}")
    print(f"  含有效VIS的站点:    {total_stats['vis_available']}")
    print(f"  输出目录:           {output_dir}")

    # 验证输出可被 noaa_dataset 读取
    _verify_output(output_dir)


# ================================================================
# 验证输出格式
# ================================================================

def _verify_output(output_dir: str, n_check: int = 3):
    """随机抽查 n 个输出文件，验证格式与 noaa_dataset 兼容"""
    all_files = glob.glob(os.path.join(output_dir, "*", "*.csv"))
    if not all_files:
        return

    import random
    samples = random.sample(all_files, min(n_check, len(all_files)))

    expected_cols = {
        "STATION", "DATE", "VIS_DISTANCE", "TEMPERATURE", "DEW_POINT",
        "WIND_DIRECTION", "WIND_SPEED", "CLOUD_HEIGHT", "SEA_LEVEL_PRESSURE"
    }
    print("\n[验证] 抽查输出文件格式...")
    for path in samples:
        df = pd.read_csv(path, nrows=5)
        missing = expected_cols - set(df.columns)
        vis_ok  = "VIS_DISTANCE" in df.columns
        date_ok = pd.to_datetime(df["DATE"].iloc[0], errors="coerce") is not pd.NaT

        status = "✓" if not missing and date_ok else "✗"
        print(f"  {status} {Path(path).parent.name}/{Path(path).name}  "
              f"行数约={len(pd.read_csv(path))}  "
              f"缺列={missing if missing else '无'}  "
              f"DATE可解析={date_ok}")


# ================================================================
# 辅助: 单文件快速验证
# ================================================================

def preview_file(path: str, n: int = 5):
    """预览单个原始文件的解析结果"""
    print(f"\n=== 预览: {path} ===")
    df = parse_station_file(path)
    if df is None:
        print("解析失败")
        return
    df_interp = interpolate_station(df)
    print(f"原始行数: {len(df)}  插值后行数: {len(df_interp)}")
    print(f"时间范围: {df_interp['DATE'].min()} ~ {df_interp['DATE'].max()}")
    print("\n各特征非缺失率:")
    for col in METEO_COLS:
        if col in df_interp.columns:
            pct = df_interp[col].notna().mean() * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {col:<25} [{bar}] {pct:5.1f}%")
    print(f"\n前 {n} 行:")
    print(df_interp.head(n).to_string(index=False))


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOAA ISD 批量预处理 → noaa_dataset.py 兼容格式"
    )
    parser.add_argument("--input_dir",  required=True,
                        help="原始NOAA数据根目录, 包含 2017/2018/... 子目录")
    parser.add_argument("--output_dir", required=True,
                        help="处理后数据输出目录 (保持相同年份子目录结构)")
    parser.add_argument("--years",      nargs="+", type=int, default=None,
                        help="只处理指定年份, 例: --years 2019 2020 2021")
    parser.add_argument("--drop_no_vis", action="store_true",
                        help="丢弃 VIS_DISTANCE 全缺失的站点 (默认保留)")
    parser.add_argument("--max_gap_hours", type=int, default=6,
                        help="线性插值最大连续填补小时数 (默认6)")
    parser.add_argument("--max_stations",  type=int, default=None,
                        help="每年最多处理N个站点 (用于调试)")
    parser.add_argument("--dry_run", action="store_true",
                        help="快速验证模式: 每年只处理前5个站点")
    parser.add_argument("--preview",  default=None,
                        help="预览单个原始CSV文件的解析结果后退出")
    args = parser.parse_args()

    if args.preview:
        preview_file(args.preview)
        return

    run_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        years=args.years,
        drop_no_vis=args.drop_no_vis,
        max_gap_hours=args.max_gap_hours,
        max_stations=args.max_stations,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
