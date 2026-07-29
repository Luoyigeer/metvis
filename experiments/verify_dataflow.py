"""
Smoke checks for the FROSI release package.

Usage:
  python experiments/verify_dataflow.py
"""
import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_imports():
    results = {}
    for name, stmt in [
        ("ablation_common", "from experiments.ablation_common import build_stage3_model, build_aligner, train_stage3"),
        ("visibility_model", "from models.visibility_model import VisibilityModel"),
        ("ablations", "from models.ablations import build_ablation_model, ABLATION_VARIANTS"),
        ("train_stage3_joint", "import train_stage3_joint"),
        ("compare_all", "from experiments.compare_all import load_all_results"),
        ("frosi_dataset", "from data.multi_datasets import FROSIDataset"),
        ("dataset_factory", "from data.dataset_factory import DATASET_REGISTRY"),
    ]:
        try:
            exec(stmt, globals())
            results[name] = "OK"
        except Exception as e:
            results[name] = f"FAIL: {e}"
    return results


def check_registry():
    from data.dataset_factory import DATASET_REGISTRY
    keys = list(DATASET_REGISTRY.keys())
    ok = keys == ["frosi"]
    return {"registry": keys, "frosi_only": ok}


def check_paths():
    import config
    return {
        "frosi_root": config.FROSI_ROOT,
        "frosi_exists": os.path.isdir(config.FROSI_ROOT),
        "fog_dirs": {
            str(v): os.path.isdir(os.path.join(config.FROSI_FOG_DIR, str(v)))
            for v in config.FROSI_VIS_DIRS
        },
        "noaa_root": config.NOAA_ROOT,
        "noaa_exists": os.path.isdir(config.NOAA_ROOT),
        "noaa_pretrain_ckpt": config.NOAA_PRETRAIN_CKPT,
        "noaa_pretrain_exists": os.path.isfile(config.NOAA_PRETRAIN_CKPT),
    }


def check_model_forward():
    import torch
    import config
    from models.visibility_model import VisibilityModel

    m = VisibilityModel(pretrained=False)
    B = 2
    out = m(
        torch.randn(B, 3, *config.IMAGE_SIZE),
        torch.zeros(B, 1, *config.IMAGE_SIZE),
        torch.zeros(B, 1, *config.IMAGE_SIZE),
        torch.randn(B, config.SEQ_LEN, config.NUM_METEO_FEATURES + 1),
        torch.randn(B, config.SEQ_LEN, 10),
        torch.randn(B, 10),
        torch.tensor([12.0, 0.0]),
        delta_hours=torch.tensor([0.0, 2.0]),
    )
    return {"forward_keys": sorted(out.keys())}


def main():
    print("=== import checks ===")
    for k, v in check_imports().items():
        print(f"  {k}: {v}")

    print("\n=== registry ===")
    print(" ", check_registry())

    print("\n=== paths ===")
    for k, v in check_paths().items():
        print(f"  {k}: {v}")

    print("\n=== model forward ===")
    try:
        print(" ", check_model_forward())
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
