"""
One-click smoke demo for reviewers.

Requires:
  - checkpoints/noaa_pretrain.pth
  - at least one image under dataset/FROSI/Fog/

Does NOT require a full MetVis checkpoint or NOAA CSV files.
This only verifies that the package loads data and runs a forward
(and optionally one backward) step — it does not reproduce paper metrics.

Usage:
  python demo.py
  python demo.py --forward-only
  python demo.py --image dataset/FROSI/Fog/100/example.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset_factory import build_loader
from data.image_datasets import build_transforms, build_aux_transforms
from data.noaa_dataset import encode_time_features, TIME_FEAT_DIM
from experiments.ablation_common import _load_noaa_pretrained
from models.visibility_model import VisibilityModel
from models.edl import EDLClassificationLoss, EDLRegressionLoss
from utils import to_device_batch, preprocess_dataset_aux_maps, load_aux_maps
import pandas as pd
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_frosi_images(root: str | None = None) -> list[Path]:
    fog = Path(root or config.FROSI_ROOT) / "Fog"
    if not fog.is_dir():
        return []
    paths = []
    for p in sorted(fog.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def require_inputs(image: str | None) -> Path:
    ckpt = Path(config.NOAA_PRETRAIN_CKPT)
    if not ckpt.is_file():
        raise SystemExit(
            f"[ERROR] Missing NOAA pretrain weights.\n"
            f"  Place the file at: {ckpt.as_posix()}\n"
            f"  (This is the only required checkpoint for the smoke demo.)"
        )

    if image:
        img_path = Path(image)
        if not img_path.is_file():
            raise SystemExit(f"[ERROR] Image not found: {img_path}")
        return img_path

    images = find_frosi_images()
    if not images:
        raise SystemExit(
            f"[ERROR] No FROSI sample images found under "
            f"{(Path(config.FROSI_ROOT) / 'Fog').as_posix()}\n"
            f"  Expected layout: dataset/FROSI/Fog/{{50,100,...,400}}/*.png\n"
            f"  A small sample subset is enough for this smoke test."
        )
    return images[0]


def ensure_aux_maps() -> None:
    root = config.FROSI_ROOT
    if not os.path.isdir(root):
        return
    print(f"[demo] Ensuring depth/trans caches for {root} ...")
    try:
        stats = preprocess_dataset_aux_maps("frosi", image_root=root, skip_existing=True)
        print(f"[demo] aux maps: {stats}")
    except Exception as e:
        print(f"[demo] WARNING: aux preprocess failed ({e}); "
              f"will try on-the-fly generation if needed.")


def build_single_image_batch(image_path: Path, device: torch.device) -> dict:
    """Build a batch=1 dict compatible with VisibilityModel.forward."""
    img_tf = build_transforms("test")
    aux_tf = build_aux_transforms()
    img = img_tf(Image.open(image_path).convert("RGB"))
    try:
        depth, trans = load_aux_maps(
            str(image_path), "frosi", aux_tf,
            image_root=config.FROSI_ROOT, allow_generate=True,
        )
    except Exception:
        depth = torch.zeros(1, *config.IMAGE_SIZE)
        trans = torch.zeros(1, *config.IMAGE_SIZE)

    dt = pd.Timestamp("2020-01-01 00:00")
    D = config.NUM_METEO_FEATURES + 1
    batch = {
        "img": img.unsqueeze(0),
        "depth": depth.unsqueeze(0) if depth.dim() == 3 else depth,
        "trans": trans.unsqueeze(0) if trans.dim() == 3 else trans,
        "seq_feat": torch.zeros(1, config.SEQ_LEN, D),
        "time_feat_seq": torch.zeros(1, config.SEQ_LEN, TIME_FEAT_DIM),
        "cur_time_feat": torch.tensor(encode_time_features(dt)).unsqueeze(0),
        "available_steps": torch.tensor([0.0]),
        "delta_hours": torch.tensor([0.0]),
        "vis_cls": torch.tensor([2], dtype=torch.long),
        "vis_val": torch.tensor([100.0 / config.MAX_VIS], dtype=torch.float32),
    }
    # normalize depth/trans shapes to [B,1,H,W]
    for k in ("depth", "trans"):
        t = batch[k]
        if t.dim() == 3:
            batch[k] = t.unsqueeze(0)
        elif t.dim() == 2:
            batch[k] = t.unsqueeze(0).unsqueeze(0)
    return to_device_batch(batch, device)


def get_loader_batch(device: torch.device, batch_size: int = 1) -> dict:
    # Monkeypatch aux loading to allow on-the-fly generation for tiny samples
    import utils as utils_mod
    import data.multi_datasets as md

    _orig = utils_mod.load_aux_maps

    def _load_allow(*args, **kwargs):
        kwargs["allow_generate"] = True
        return _orig(*args, **kwargs)

    utils_mod.load_aux_maps = _load_allow
    md.load_aux_maps = _load_allow
    try:
        loader = build_loader("frosi", "train", aligner=None, batch_size=batch_size)
        raw = next(iter(loader))
        return to_device_batch(raw, device)
    finally:
        utils_mod.load_aux_maps = _orig
        md.load_aux_maps = _orig


def run_forward(model: VisibilityModel, batch: dict) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(
            batch["img"], batch["depth"], batch["trans"],
            batch["seq_feat"], batch["time_feat_seq"], batch["cur_time_feat"],
            batch["available_steps"],
            delta_hours=batch.get("delta_hours"),
            force_img_only=True,
        )
    return out


def run_one_train_step(model: VisibilityModel, batch: dict, device: torch.device) -> float:
    """One Stage2-style update on the image EDL heads (full model, img path)."""
    model.train()
    edl_cls = EDLClassificationLoss(
        num_classes=config.NUM_VIS_CLASSES,
        annealing_start=config.EDL_ANNEAL_START,
        annealing_step=config.EDL_ANNEAL_STEP,
    )
    edl_reg = EDLRegressionLoss(coeff=config.EDL_REG_COEFF)
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=config.MAIN_WEIGHT_DECAY)
    opt.zero_grad()

    out = model(
        batch["img"], batch["depth"], batch["trans"],
        batch["seq_feat"], batch["time_feat_seq"], batch["cur_time_feat"],
        batch["available_steps"],
        delta_hours=batch.get("delta_hours"),
        force_img_only=True,
    )
    from experiments.ablation_common import _stage3_vis_tgt_norm

    loss = torch.tensor(0.0, device=device)
    valid_cls = batch["vis_cls"] >= 0
    if valid_cls.any():
        d = edl_cls(
            out["img_cls_out"]["alpha"][valid_cls],
            batch["vis_cls"][valid_cls],
            epoch=1,
        )
        loss = loss + 0.1 * d["total"]

    valid_reg = batch["vis_val"] >= 0
    if valid_reg.any():
        ro = out["img_reg_out"]
        vis_tgt = _stage3_vis_tgt_norm(batch["vis_val"])[valid_reg]
        d2 = edl_reg(
            ro["gamma"][valid_reg],
            ro["nu"][valid_reg],
            ro["alpha"][valid_reg],
            ro["beta"][valid_reg],
            vis_tgt,
        )
        loss = loss + d2["total"]

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="MetVis FROSI smoke demo for reviewers")
    parser.add_argument("--forward-only", action="store_true",
                        help="Skip the one-step backward pass")
    parser.add_argument("--image", default=None,
                        help="Optional single image path (skips DataLoader)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    sample = require_inputs(args.image)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("=" * 60)
    print("MetVis smoke demo (FROSI sample + NOAA pretrain only)")
    print("=" * 60)
    print(f"  device:     {device}")
    print(f"  NOAA ckpt:  {config.NOAA_PRETRAIN_CKPT}")
    print(f"  sample img: {sample}")
    print("  note: this does NOT reproduce paper metrics")
    print("=" * 60)

    ensure_aux_maps()

    print("[demo] Building VisibilityModel(pretrained=False) ...")
    model = VisibilityModel(pretrained=False).to(device)
    print("[demo] Loading NOAA temporal pretrain into ts_branch ...")
    if not os.path.isfile(config.NOAA_PRETRAIN_CKPT):
        raise SystemExit(f"[ERROR] Missing {config.NOAA_PRETRAIN_CKPT}")
    _load_noaa_pretrained(model, device)
    print("[demo] NOAA pretrain load attempted (see messages above).")

    if args.image:
        print("[demo] Building single-image batch ...")
        batch = build_single_image_batch(Path(args.image), device)
    else:
        print("[demo] Loading one FROSI mini-batch (aligner=None) ...")
        try:
            batch = get_loader_batch(device, batch_size=max(1, args.batch_size))
        except Exception as e:
            print(f"[demo] DataLoader failed ({e}); falling back to single image.")
            batch = build_single_image_batch(sample, device)

    print("[demo] Forward pass ...")
    out = run_forward(model, batch)
    keys = sorted(out.keys())
    print(f"  output keys ({len(keys)}): {', '.join(keys[:12])}"
          + (" ..." if len(keys) > 12 else ""))
    if "gate_alpha" in out:
        print(f"  gate_alpha: {out['gate_alpha'].detach().cpu().view(-1).tolist()}")
    if "fused_cls_out" in out and "evidence" in out["fused_cls_out"]:
        ev = out["fused_cls_out"]["evidence"]
        print(f"  fused cls evidence shape: {tuple(ev.shape)}")
    elif "img_cls_out" in out and "evidence" in out["img_cls_out"]:
        ev = out["img_cls_out"]["evidence"]
        print(f"  img cls evidence shape: {tuple(ev.shape)}")

    if not args.forward_only:
        print("[demo] One training step (image EDL) ...")
        loss = run_one_train_step(model, batch, device)
        print(f"  train step loss: {loss:.6f}")

    print("=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
