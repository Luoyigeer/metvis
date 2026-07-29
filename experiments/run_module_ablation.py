"""
模块级消融实验（修复版，默认走三阶段训练）
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from experiments.ablation_common import (
    build_aligner, train_stage3, train_structural_ablation, save_ablation_result,
)
from ablation_routing import get_use_kd, is_structural_variant, ABLATION_VARIANTS
from models.ablations import get_contra_weight

RESULTS_PATH = os.path.join(config.LOG_DIR, "ablation_results", "ablation_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="frosi",
                        choices=["frosi"])
    parser.add_argument("--variants", nargs="+", default=ABLATION_VARIANTS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=config.MAIN_LR)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    import torch
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    datasets = [args.dataset]

    for ds in datasets:
        aligner = build_aligner(ds)
        for variant in args.variants:
            if is_structural_variant(variant):
                metrics, _ = train_structural_ablation(
                    variant_name=variant,
                    dataset_name=ds,
                    aligner=aligner,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    contra_weight=get_contra_weight(variant),
                    ckpt_suffix=variant,
                )
            else:
                model_kwargs = {}
                if variant in ("w_o_TCAM", "w/o_TCAM"):
                    model_kwargs["use_tcam"] = False
                if variant in ("w_o_UGG", "w/o_UGG", "w_o_UGG_KD", "w/o_UGG_KD"):
                    model_kwargs["use_unc_gate"] = False
                metrics, _ = train_stage3(
                    variant_name=variant,
                    dataset_name=ds,
                    aligner=aligner,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    model_kwargs=model_kwargs or None,
                    use_kd=get_use_kd(variant),
                    contra_weight=get_contra_weight(variant),
                    ckpt_suffix=variant,
                )
            save_ablation_result(RESULTS_PATH, ds, variant, metrics)


if __name__ == "__main__":
    main()
