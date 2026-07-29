"""
对比实验结果汇总 (ACC/Recall + MAE/MSE 版)

指标说明:
  ACC   (%)    分类准确率
  Recall(%)    各类别召回率宏平均
  MAE   (m)    平均绝对误差，回归头输出
  MSE   (m²)   均方误差，回归头输出

消融分析附加:
  per_class_recall  各等级召回率（写入 CSV，不进 LaTeX 主表）

用法:
  python experiments/compare_all.py
  python experiments/compare_all.py --latex --per_class
"""
import os, sys, json, argparse, csv, math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RESULTS_DIR  = os.path.join(config.LOG_DIR, "baseline_results")
ABLATION_DIR = os.path.join(config.LOG_DIR, "ablation_results")
OUT_DIR      = os.path.join(config.LOG_DIR, "summary")
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
# 模型分组 & 显示名称
# ================================================================
IMAGE_MODELS    = ["AlexNet", "DenseNet", "ResNet", "VisNet", "CLIP", "OrdinalCLIP"]
TEMPORAL_MODELS = ["LSTM", "Transformer", "TimesNet"]
MULTIMODAL_MODELS = ["MM_EarlyFusion", "MM_LateFusion"]
NIGHT_BASELINES = ["ZeroDCE_ResNet", "CycleGAN_ResNet"]
ABLATION_MODELS = ["w/o_TS", "w/o_Gate", "w/o_Contra",
                   "w/o_UGG", "w/o_KD", "w/o_UGG_KD", "w/o_TCAM", "Full"]

EXTENDED_ABLATION_GROUPS = {
    "UGG Input": ["UGG_Full", "UGG_wo_u_img", "UGG_wo_u_ts", "UGG_wo_avail", "UGG_wo_feat_sim"],
    "TCAM Period": ["TCAM_full", "TCAM_wo_1h", "TCAM_wo_6h", "TCAM_wo_12h", "TCAM_wo_24h",
                    "TCAM_short", "TCAM_long", "TCAM_1_6_24"],
    "EDL Uncertainty": ["EDL_default", "EDL_cls_only", "EDL_reg_only", "EDL_swap"],
    "NOAA Pretrain": ["w_NOAA_pretrain", "w/o_NOAA_pretrain"],
}

DISPLAY_NAMES = {
    "AlexNet":     "AlexNet",
    "DenseNet":    "DenseNet-121",
    "ResNet":      "ResNet-50",
    "VisNet":      "VisNet",
    "CLIP":        "CLIP",
    "OrdinalCLIP": "OrdinalCLIP",
    "MM_EarlyFusion": "MM-EarlyFusion",
    "MM_LateFusion":  "MM-LateFusion",
    "ZeroDCE_ResNet": "Zero-DCE → ResNet-50",
    "CycleGAN_ResNet": "CycleGAN(Night→Day) → ResNet-50",
    "LSTM":        "BiLSTM",
    "Transformer": "Transformer",
    "TimesNet":    "TimesNet (TS-only)",
    "w/o_TS":      r"Ours w/o $\mathcal{T}$",
    "w/o_Gate":    "Ours w/o Gate",
    "w/o_Contra":  r"Ours w/o $\mathcal{L}_{c}$",
    "w/o_UGG":     "Ours w/o UGG",
    "w/o_KD":      "Ours w/o KD",
    "w/o_TCAM":    "Ours w/o TCAM",
    "Full":        r"\textbf{Ours (Full)}",
}
DISPLAY_NAMES_MD = {k: (v.replace(r"\textbf{","**").replace("}","**")
                          .replace(r"$\mathcal{T}$","TS")
                          .replace(r"$\mathcal{L}_{c}$","Lc"))
                    for k, v in DISPLAY_NAMES.items()}

DATASET_NAMES = {"frosi": "FROSI"}
ALL_DATASETS  = list(DATASET_NAMES.keys())


# ================================================================
# 数据加载
# ================================================================
def _load(path):
    if not os.path.exists(path): return {}
    with open(path, encoding="utf-8") as f: return json.load(f)

def load_all_results() -> dict:
    master = {ds: {} for ds in ALL_DATASETS}
    baseline_glob = [
        os.path.join(RESULTS_DIR, f"baseline_{ds}.json") for ds in ALL_DATASETS
    ] + [os.path.join(RESULTS_DIR, "baseline_all.json")]
    for src_path in baseline_glob + [
        os.path.join(ABLATION_DIR, "ablation_results.json"),
    ]:
        for ds, d in _load(src_path).items():
            master.setdefault(ds.lower(), {}).update(d)

    extra_sources = [
        "ugg_input_ablation.json",
        "tcam_period_ablation.json",
        "edl_unc_ablation.json",
        "noaa_pretrain_ablation.json",
        "available_steps_eval.json",
    ]
    for fname in extra_sources:
        path = os.path.join(ABLATION_DIR, fname)
        for ds, d in _load(path).items():
            master.setdefault(ds.lower(), {}).update(d)

    main_path = os.path.join(config.LOG_DIR, "main_results.json")
    if os.path.exists(main_path):
        with open(main_path, encoding="utf-8") as f: lst = json.load(f)
        for r in lst:
            ds = r.get("dataset","").lower()
            model_key = r.get("model", "Full")
            if model_key == "ours":
                model_key = "Full"
            master.setdefault(ds, {})[model_key] = {
                "acc": r.get("acc", float("nan")),
                "recall": r.get("recall", float("nan")),
                "mae": r.get("mae", float("nan")),
                "mse": r.get("mse", float("nan")),
                "mae_cls_0_3": r.get("mae_cls_0_3", float("nan")),
                "mae_cls_4_5": r.get("mae_cls_4_5", float("nan")),
                "per_class_recall": r.get("per_class_recall", []),
            }
    return master


# ================================================================
# 格式化
# ================================================================
def _nan(v):
    try: return math.isnan(float(v))
    except: return True

def _fmt_acc(v):
    # ACC 存储为 [0,1]，显示为百分比
    return f"{float(v)*100:.2f}" if not _nan(v) and float(v) >= 0 else "-"

def _fmt_recall(v):
    return f"{float(v)*100:.2f}" if not _nan(v) and float(v) >= 0 else "-"

def _fmt_mae(v):
    return f"{float(v):.1f}" if not _nan(v) and float(v) >= 0 else "-"

def _fmt_mse(v):
    return f"{float(v):.1f}" if not _nan(v) and float(v) >= 0 else "-"

def _fmt_mae03(v):
    return f"{float(v):.1f}" if not _nan(v) and float(v) >= 0 else "-"


def _find_best(master, datasets, metric, higher):
    bests = {}
    for ds in datasets:
        vals = [float(v.get(metric, float("nan")))
                for v in master.get(ds,{}).values()
                if not _nan(v.get(metric, float("nan"))) and float(v.get(metric,0)) >= 0]
        if vals: bests[ds] = max(vals) if higher else min(vals)
    result = {}
    for ds in datasets:
        for mk, v in master.get(ds,{}).items():
            val = v.get(metric, float("nan"))
            result[(ds,mk)] = (not _nan(val) and ds in bests
                               and abs(float(val)-bests[ds]) < 1e-9)
    return result


def render_extended_markdown(master, dataset="frosi"):
    lines = [f"## Extended Ablations ({DATASET_NAMES.get(dataset, dataset)})\n"]
    for grp, models in EXTENDED_ABLATION_GROUPS.items():
        lines.append(f"### {grp}\n")
        lines.append("| Model | ACC(%) | Recall(%) | MAE(m) | MAE_0-3(m) | MSE(m²) |")
        lines.append("|:------|-------:|---------:|-------:|-----------:|--------:|")
        for mk in models:
            d = master.get(dataset, {}).get(mk, {})
            lines.append(
                f"| {mk} | {_fmt_acc(d.get('acc'))} | {_fmt_recall(d.get('recall'))} | "
                f"{_fmt_mae(d.get('mae'))} | {_fmt_mae03(d.get('mae_cls_0_3'))} | "
                f"{_fmt_mse(d.get('mse'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_horizon_table(dataset="frosi"):
    path = os.path.join(ABLATION_DIR, "horizon_eval.json")
    data = _load(path).get(dataset, {})
    if not data:
        return ""
    lines = [f"## Horizon Prediction ({DATASET_NAMES.get(dataset, dataset)})\n",
             "| Horizon | n | ACC(%) | MAE(m) |",
             "|:--------|--:|-------:|-------:|"]
    for key in sorted(data.keys(), key=lambda k: data[k].get("horizon_h", 0)):
        d = data[key]
        lines.append(
            f"| {d.get('horizon_h', key)} h | {d.get('n', '-')} | "
            f"{_fmt_acc(d.get('acc'))} | {_fmt_mae(d.get('mae'))} |"
        )
    return "\n".join(lines) + "\n"


def render_spatial_table(dataset="frosi"):
    path = os.path.join(ABLATION_DIR, "spatial_distance_eval.json")
    data = _load(path).get(dataset, {})
    if not data:
        return ""
    lines = [f"## Spatial Distance Bands ({DATASET_NAMES.get(dataset, dataset)})\n",
             "| Band | n | ACC(%) | MAE(m) |",
             "|:-----|--:|-------:|-------:|"]
    for band in ["<3km", "3-6km", ">6km", "unknown"]:
        if band not in data:
            continue
        d = data[band]
        lines.append(
            f"| {band} | {d.get('n', '-')} | {_fmt_acc(d.get('acc'))} | {_fmt_mae(d.get('mae'))} |"
        )
    return "\n".join(lines) + "\n"


def render_horizon_latex(dataset="frosi"):
    path = os.path.join(ABLATION_DIR, "horizon_eval.json")
    data = _load(path).get(dataset, {})
    if not data:
        return ""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{Short-term visibility prediction on {DATASET_NAMES.get(dataset, dataset)}.}}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"$\Delta t$ (h) & $n$ & ACC$\uparrow$ & MAE$\downarrow$ (m) \\", r"\midrule",
    ]
    for key in sorted(data.keys(), key=lambda k: data[k].get("horizon_h", 0)):
        d = data[key]
        lines.append(
            f"{d.get('horizon_h', 0):.1f} & {d.get('n', 0)} & "
            f"{_fmt_acc(d.get('acc'))} & {_fmt_mae(d.get('mae'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ================================================================
# Markdown
# ================================================================
def render_markdown(master, datasets):
    hdr = "| 模型 |"
    sep = "|:-----|"
    for ds in datasets:
        hdr += f" {DATASET_NAMES.get(ds,ds)} ACC↑(%) | Recall↑(%) | MAE↓(m) | MAE_0-3↓(m) | MSE↓(m²) |"
        sep += " :---: | :---: | :---: | :---: | :---: |"
    lines = [hdr, sep]

    def row(mk):
        name = DISPLAY_NAMES_MD.get(mk, mk)
        r = f"| {name} |"
        for ds in datasets:
            d = master.get(ds,{}).get(mk,{})
            r += (f" {_fmt_acc(d.get('acc',float('nan')))} |"
                  f" {_fmt_recall(d.get('recall',float('nan')))} |"
                  f" {_fmt_mae(d.get('mae',float('nan')))} |"
                  f" {_fmt_mae03(d.get('mae_cls_0_3',float('nan')))} |"
                  f" {_fmt_mse(d.get('mse',float('nan')))} |")
        lines.append(r)

    lines.append("| **图像基线** |" + " | | | | |" * len(datasets))
    for m in IMAGE_MODELS:    row(m)
    lines.append("| **时序基线** |" + " | | | | |" * len(datasets))
    for m in TEMPORAL_MODELS: row(m)
    lines.append("| **多模态基线** |" + " | | | | |" * len(datasets))
    for m in MULTIMODAL_MODELS: row(m)
    lines.append("| **夜间对比基线** |" + " | | | | |" * len(datasets))
    for m in NIGHT_BASELINES: row(m)
    lines.append("| **消融实验** |" + " | | | | |" * len(datasets))
    for m in ABLATION_MODELS: row(m)
    return "\n".join(lines)


# ================================================================
# CSV（含逐类 Recall）
# ================================================================
def render_csv(master, datasets):
    header = ["Model", "Type"]
    for ds in datasets:
        dn = DATASET_NAMES.get(ds,ds)
        header += [f"{dn}_ACC(%)", f"{dn}_Recall(%)", f"{dn}_MAE(m)",
                   f"{dn}_MAE_0-3(m)", f"{dn}_MSE(m2)"]
    # 附加逐类 Recall 列
    for ds in datasets:
        dn = DATASET_NAMES.get(ds,ds)
        for cls_name in config.VIS_CLASS_NAMES:
            header.append(f"{dn}_Recall_{cls_name}")

    rows = [header]
    for grp, mlist in [("Image",IMAGE_MODELS),
                       ("Temporal",TEMPORAL_MODELS),
                       ("Multimodal",MULTIMODAL_MODELS),
                       ("Night",NIGHT_BASELINES),
                       ("Ablation",ABLATION_MODELS)]:
        for mk in mlist:
            row = [DISPLAY_NAMES_MD.get(mk,mk), grp]
            for ds in datasets:
                d = master.get(ds,{}).get(mk,{})
                row += [_fmt_acc(d.get("acc",float("nan"))),
                        _fmt_recall(d.get("recall",float("nan"))),
                        _fmt_mae(d.get("mae",float("nan"))),
                        _fmt_mae03(d.get("mae_cls_0_3", float("nan"))),
                        _fmt_mse(d.get("mse",float("nan")))]
            # 逐类 Recall
            for ds in datasets:
                d   = master.get(ds,{}).get(mk,{})
                pcr = d.get("per_class_recall",[])
                for i in range(config.NUM_VIS_CLASSES):
                    v = pcr[i] if i < len(pcr) else float("nan")
                    row.append(f"{v*100:.2f}" if not _nan(v) else "-")
            rows.append(row)
    return rows


# ================================================================
# LaTeX（三线表，最优值加粗）
# ================================================================
def render_latex(master, datasets):
    n_ds = len(datasets)
    b_acc = _find_best(master, datasets, "acc", higher=True)
    b_rec = _find_best(master, datasets, "recall", higher=True)
    b_mae = _find_best(master, datasets, "mae", higher=False)
    b_mse = _find_best(master, datasets, "mse", higher=False)

    def _lv(raw, fmt_fn, ds, mk, bd):
        s = fmt_fn(raw)
        return rf"\textbf{{{s}}}" if s != "-" and bd.get((ds,mk),False) else s

    lines = [
        r"% \usepackage{booktabs,multirow}",
        r"\begin{table*}[htbp]",
        r"\centering",
        (r"\caption{Visibility estimation comparison. "
         r"ACC/Recall are classification metrics. "
         r"MAE and MSE are in meters. "
         r"\textbf{Bold} = best per dataset per metric.}"),
        r"\label{tab:comparison}",
        rf"\begin{{tabular}}{{ll{'rrrr'*n_ds}}}",
        r"\toprule",
    ]

    # 数据集 multicolumn
    ds_row = r"\multicolumn{2}{l}{\textbf{Method}}"
    cmidrules = []
    for i, ds in enumerate(datasets):
        col_s = 3 + i*4; col_e = col_s+3
        ds_row += rf" & \multicolumn{{4}}{{c}}{{\textbf{{{DATASET_NAMES.get(ds,ds)}}}}}"
        cmidrules.append(rf"\cmidrule(lr){{{col_s}-{col_e}}}")
    lines.append(ds_row + r" \\")
    lines.extend(cmidrules)

    # 指标行
    m_row = r"& &"
    for i in range(n_ds):
        m_row += r" ACC$\uparrow$ & Recall$\uparrow$ & MAE$\downarrow$ & MSE$\downarrow$"
        if i < n_ds-1: m_row += " &"
    lines += [m_row + r" \\", r"\midrule"]

    def grp(label, mlist):
        lines.append(rf"\multicolumn{{{2+n_ds*4}}}{{l}}{{\textit{{{label}}}}} \\")
        for mk in mlist:
            disp = DISPLAY_NAMES.get(mk, mk)
            r = rf"~~{disp} & &"
            for i, ds in enumerate(datasets):
                d = master.get(ds,{}).get(mk,{})
                acc_s = _lv(d.get("acc",float("nan")), _fmt_acc, ds, mk, b_acc)
                rec_s = _lv(d.get("recall",float("nan")), _fmt_recall, ds, mk, b_rec)
                mae_s = _lv(d.get("mae",float("nan")), _fmt_mae, ds, mk, b_mae)
                mse_s = _lv(d.get("mse",float("nan")), _fmt_mse, ds, mk, b_mse)
                r += f" {acc_s} & {rec_s} & {mae_s} & {mse_s}"
                if i < n_ds-1: r += " &"
            lines.append(r + r" \\")

    grp("Image-based Baselines",    IMAGE_MODELS)
    lines.append(r"\midrule")
    grp("Temporal-based Baselines", TEMPORAL_MODELS)
    lines.append(r"\midrule")
    grp("Multimodal Baselines",     MULTIMODAL_MODELS)
    lines.append(r"\midrule")
    grp("Nighttime Baselines",      NIGHT_BASELINES)
    lines.append(r"\midrule")
    grp("Ablation Study",           ABLATION_MODELS)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ================================================================
# 逐类 Recall 打印（消融分析专用）
# ================================================================
def print_per_class_recall(master, datasets):
    print("\n" + "="*90)
    print("逐类 Recall（消融分析附加指标，%）")
    print("="*90)
    for ds in datasets:
        print(f"\n▶ {DATASET_NAMES.get(ds,ds)}")
        col_w = 11
        hdr   = f"{'模型':<26}" + "".join(f"{n:>{col_w}}" for n in config.VIS_CLASS_NAMES)
        print(hdr); print("-"*len(hdr))
        for mk in IMAGE_MODELS + TEMPORAL_MODELS + MULTIMODAL_MODELS + ABLATION_MODELS:
            pcr = master.get(ds,{}).get(mk,{}).get("per_class_recall",[])
            if pcr:
                vals = "".join(
                    f"{v*100:>{col_w}.2f}" if not _nan(v) else f"{'N/A':>{col_w}}"
                    for v in pcr
                )
                print(f"{DISPLAY_NAMES_MD.get(mk,mk):<26}{vals}")


# ================================================================
# 主入口
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="all")
    parser.add_argument("--latex",     action="store_true")
    parser.add_argument("--per_class", action="store_true",
                        help="打印逐类Recall（消融分析用）")
    args = parser.parse_args()

    master = load_all_results()
    if all(not v for v in master.values()):
        print("[警告] 未找到任何实验结果")
        return

    datasets = ALL_DATASETS if args.dataset=="all" else [args.dataset]

    print("\n" + "="*80)
    print(" 能见度估计对比实验  指标: ACC/Recall(%) / MAE(m) / MAE_0-3(m) / MSE(m²)")
    print(" 协议: dual_freeze 15ep (Full) | dataset_factory + 米制 GT (baselines/ablations)")
    print("="*80)
    print(render_markdown(master, datasets))

    if args.per_class:
        print_per_class_recall(master, datasets)

    csv_path = os.path.join(OUT_DIR, "results_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(render_csv(master, datasets))
    print(f"\n[已保存] CSV -> {csv_path}")

    md_path = os.path.join(OUT_DIR, "results_summary.md")
    ext_ds = datasets[0] if len(datasets) == 1 else "frosi"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 能见度估计对比实验\n\n")
        f.write(
            "> 指标: ACC/Recall(%) / MAE(m) / MAE_0-3(m) / MSE(m²)\n"
            "> 协议: 主模型/消融 Full = dual_freeze 15ep; 基线/消融共享 dataset_factory + 米制 eval\n\n"
        )
        f.write(render_markdown(master, datasets))
        f.write("\n\n")
        f.write(render_extended_markdown(master, ext_ds))
        f.write(render_horizon_table(ext_ds))
        f.write(render_spatial_table(ext_ds))
    print(f"[已保存] Markdown -> {md_path}")

    if args.latex:
        latex_str = render_latex(master, datasets)
        tex_path = os.path.join(OUT_DIR, "results_latex.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"[已保存] LaTeX -> {tex_path}")
        h_tex = render_horizon_latex(ext_ds)
        if h_tex:
            h_path = os.path.join(OUT_DIR, "horizon_latex.tex")
            with open(h_path, "w", encoding="utf-8") as f:
                f.write(h_tex)
            print(f"[已保存] Horizon LaTeX -> {h_path}")

    print(render_extended_markdown(master, ext_ds))
    print(render_horizon_table(ext_ds))
    print(render_spatial_table(ext_ds))


if __name__ == "__main__":
    main()
