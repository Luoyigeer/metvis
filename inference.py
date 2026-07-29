"""
推理脚本 — 支持 --delta_hours TCAM 目标时刻预测
"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.noaa_dataset import NOAADataProcessor, vis_to_class, normalize_vis, encode_time_features
from data.image_datasets import TimeAligner, build_transforms, build_aux_transforms
from models.visibility_model import VisibilityModel
from models.tcam import make_delta_hours
from utils import load_checkpoint, load_aux_maps


def denorm_vis(v_norm: float, max_vis: float = 50000.0) -> float:
    return v_norm * max_vis


def cls_to_range_str(cls_idx: int) -> str:
    names = config.VIS_CLASS_NAMES
    return names[cls_idx] if 0 <= cls_idx < len(names) else "未知"


class VisibilityInferencer:

    def __init__(self, ckpt_path: str = config.INFER_CHECKPOINT, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.processor = NOAADataProcessor()
        self.processor.load()
        self.aligner = TimeAligner(self.processor)
        self.vis_prior = self.processor.get_vis_prior()

        self.model = VisibilityModel(pretrained=False).to(self.device)
        ckpt = load_checkpoint(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()
        print(f"[推理] 模型已加载: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

        self.img_tf = build_transforms("test")
        self.aux_tf = build_aux_transforms()

    def _load_img(self, path: str, rgb: bool = True) -> torch.Tensor:
        if path is None:
            return torch.zeros(1, 1, *config.IMAGE_SIZE)
        img = Image.open(path)
        if rgb:
            return self.img_tf(img.convert("RGB")).unsqueeze(0)
        return self.aux_tf(img.convert("L")).unsqueeze(0)

    def _zero_aux(self) -> torch.Tensor:
        return torch.zeros(1, 1, *config.IMAGE_SIZE)

    def infer(self, image_path: str, query_time: datetime,
              depth_path: str = None, trans_path: str = None,
              station_id: str = None, delta_hours: float = 0.0,
              aux_dataset: str = "frosi") -> dict:
        ts = pd.Timestamp(query_time)
        img = self._load_img(image_path, rgb=True).to(self.device)

        if depth_path and trans_path:
            depth = self._load_img(depth_path, rgb=False).to(self.device)
            trans = self._load_img(trans_path, rgb=False).to(self.device)
        else:
            depth_t, trans_t = load_aux_maps(
                image_path, aux_dataset, self.aux_tf,
                allow_generate=True,
            )
            depth = depth_t.unsqueeze(0).to(self.device)
            trans = trans_t.unsqueeze(0).to(self.device)

        seq_feat, time_feat_seq, avail = self.aligner.get_sequence(ts, station_id)
        cur_time_feat = encode_time_features(ts)

        seq_feat_t = torch.tensor(seq_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        time_feat_seq_t = torch.tensor(time_feat_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        cur_time_feat_t = torch.tensor(cur_time_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        avail_t = torch.tensor([avail], dtype=torch.float32).to(self.device)
        delta_t = make_delta_hours(1, delta_hours, self.device)

        with torch.no_grad():
            outputs = self.model(
                img, depth, trans,
                seq_feat_t, time_feat_seq_t,
                cur_time_feat_t, avail_t,
                delta_hours=delta_t,
            )

        fused_cls = int(outputs["fused_cls_out"]["prob"].argmax(dim=-1).item())
        fused_vis = denorm_vis(outputs["fused_reg_out"]["gamma"].item())
        ts_future_vis = denorm_vis(outputs["ts_future_pred"].item())
        img_vis = denorm_vis(outputs["img_vis_pred"].item())
        alpha = float(outputs["gate_alpha"].item())

        h, m = ts.hour, ts.month
        prior_vis = self.vis_prior.get((h, m), self.vis_prior.get((h, 1), 5000.0))

        tcam_future_vis = fused_vis if delta_hours > 0 else None

        return {
            "vis_class": fused_cls,
            "vis_range": cls_to_range_str(fused_cls),
            "vis_value_m": fused_vis,
            "tcam_pred_m": tcam_future_vis,
            "ts_future_pred_m": ts_future_vis,
            "img_vis_m": img_vis,
            "gate_alpha": alpha,
            "available_steps": avail,
            "prior_vis_m": prior_vis,
            "delta_hours": delta_hours,
        }


def main():
    parser = argparse.ArgumentParser(description="能见度推理")
    parser.add_argument("--image", required=True)
    parser.add_argument("--time", required=True, help="格式: '2023/6/15 14:00'")
    parser.add_argument("--delta_hours", type=float, default=0.0,
                        help="TCAM 目标时刻与拍摄时刻的小时差 (0=当前, >0=未来预测)")
    parser.add_argument("--depth", default=None)
    parser.add_argument("--trans", default=None)
    parser.add_argument("--station", default=None)
    parser.add_argument("--aux_dataset", default="frosi",
                        choices=["frosi"],
                        help="depth/trans 缓存数据集名（未指定 --depth/--trans 时使用）")
    parser.add_argument("--ckpt", default=config.INFER_CHECKPOINT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    inferencer = VisibilityInferencer(ckpt_path=args.ckpt, device=args.device)
    query_time = datetime.strptime(args.time, "%Y/%m/%d %H:%M")
    result = inferencer.infer(
        image_path=args.image, query_time=query_time,
        depth_path=args.depth, trans_path=args.trans,
        station_id=args.station, delta_hours=args.delta_hours,
        aux_dataset=args.aux_dataset,
    )

    print("\n" + "=" * 50)
    print("能见度估计结果")
    print("=" * 50)
    print(f"  Δt:          {result['delta_hours']} h")
    print(f"  等级:        {result['vis_class']} ({result['vis_range']})")
    print(f"  TCAM估计:    {result['vis_value_m']:.0f} m")
    print(f"  图像估计:    {result['img_vis_m']:.0f} m")
    print(f"  时序权重α:   {result['gate_alpha']:.3f}")
    print(f"  可用历史:    {result['available_steps']} 步")
    print(f"  统计先验:    {result['prior_vis_m']:.0f} m")
    print(f"  时序future:  {result['ts_future_pred_m']:.0f} m (固定步长头)")
    print("=" * 50)


if __name__ == "__main__":
    main()
