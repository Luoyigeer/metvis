"""
主模型结果保存工具
在 train_main.py 测试集评估后调用，将结果写入 logs/main_results.json
供 compare_all.py 汇总使用
"""
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def save_main_result(dataset_name: str, metrics: dict):
    """
    保存主模型在某数据集上的测试结果
    dataset_name: 'frosi'
    metrics: compute_metrics() 的输出 {acc, recall, mae, mse, per_class_recall}
    """
    out_path = os.path.join(config.LOG_DIR, "main_results.json")

    # 加载已有结果
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    # 去重更新（model 键与 compare_all 消融表 "Full" 对齐）
    record = {
        "model":           "Full",
        "dataset":         dataset_name,
        "acc":             round(float(metrics.get("acc", -1)), 4),
        "recall":          round(float(metrics.get("recall", -1)), 4),
        "mae":             round(float(metrics.get("mae", -1)), 3),
        "mse":             round(float(metrics.get("mse", -1)), 3),
        "mae_cls_0_3":     round(float(metrics.get("mae_cls_0_3", -1)), 3),
        "mae_cls_4_5":     round(float(metrics.get("mae_cls_4_5", -1)), 3),
        "per_class_recall": [round(float(v), 4) for v in
                              metrics.get("per_class_recall", [])],
    }
    merged = {(r["model"], r["dataset"]): r for r in existing}
    merged[("Full", dataset_name)] = record
    if ("ours", dataset_name) in merged:
        del merged[("ours", dataset_name)]

    with open(out_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2, ensure_ascii=False)
    print(f"[结果已保存] {out_path} <- dataset={dataset_name} "
          f"ACC={record['acc']} Recall={record['recall']}")
