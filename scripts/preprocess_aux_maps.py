"""
Precompute depth / transmission aux-map caches.

Usage:
  python scripts/preprocess_aux_maps.py --dataset frosi
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import preprocess_dataset_aux_maps, get_dataset_root

DATASETS = ("frosi",)


def main():
    parser = argparse.ArgumentParser(description="Build depth/trans offline caches")
    parser.add_argument(
        "--dataset",
        default="frosi",
        choices=list(DATASETS),
        help="Target dataset",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help="Override default image root",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Rewrite existing caches",
    )
    args = parser.parse_args()

    name = args.dataset
    root = get_dataset_root(name, image_root=args.image_root)
    print(f"\n=== {name}  root={root}  cache={config.AUX_CACHE_ROOT} ===")
    preprocess_dataset_aux_maps(
        name,
        image_root=root,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
