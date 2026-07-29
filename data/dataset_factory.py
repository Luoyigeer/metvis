"""
Dataset factory for the FROSI release package.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from torch.utils.data import DataLoader
from data.multi_datasets import FROSIDataset
from data.image_datasets import TimeAligner
from data.sampler import build_balanced_loader, build_mild_loader

DATASET_REGISTRY = {
    "frosi": FROSIDataset,
}


def build_loader(dataset_name: str, split: str,
                 aligner: TimeAligner = None,
                 batch_size: int = config.BATCH_SIZE,
                 num_workers: int = config.NUM_WORKERS) -> DataLoader:
    """
    dataset_name: 'frosi'
    split: 'train' | 'val' | 'test'
    """
    shuffle = (split == "train")

    if dataset_name in DATASET_REGISTRY:
        ds = DATASET_REGISTRY[dataset_name](split=split, aligner=aligner)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}"
        )

    if split == "train":
        sampler_mode = config.SAMPLER_MODE
        if sampler_mode == "balanced":
            return build_balanced_loader(
                ds, batch_size=batch_size, num_workers=num_workers,
            )
        if sampler_mode == "mild":
            return build_mild_loader(
                ds, batch_size=batch_size, num_workers=num_workers,
            )

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
        drop_last=shuffle,
    )


def build_all_loaders(dataset_name: str, aligner: TimeAligner = None,
                      batch_size: int = config.BATCH_SIZE):
    """Return (train_loader, val_loader, test_loader)."""
    return (
        build_loader(dataset_name, "train", aligner, batch_size),
        build_loader(dataset_name, "val", aligner, batch_size),
        build_loader(dataset_name, "test", aligner, batch_size),
    )
