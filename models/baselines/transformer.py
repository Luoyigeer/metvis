"""Transformer 时序基线"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import config
from data.noaa_dataset import TIME_FEAT_DIM

class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerVis(nn.Module):
    """
    Transformer Encoder + CLS Token 池化
    参考: Vaswani et al., Attention Is All You Need, NeurIPS 2017
    适配时序气象数据 -> 能见度估计
    """
    def __init__(self,
                 d_in: int = config.NUM_METEO_FEATURES + 1,
                 time_emb_dim: int = TIME_FEAT_DIM,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 d_ff: int = 128,
                 feat_dim: int = config.TIME_FEAT_DIM,
                 dropout: float = 0.1,
                 seq_len: int = config.SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(d_in + time_emb_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        # 可学习CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 当前时刻时间编码融合
        self.cur_proj = nn.Linear(time_emb_dim, d_model)

        self.feat_proj = nn.Sequential(
            nn.Linear(d_model * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        self.vis_head = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.future_head = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, seq_feat, time_feat_seq, cur_time_feat):
        B = seq_feat.shape[0]
        x = torch.cat([seq_feat, time_feat_seq], dim=-1)   # [B, T, D+T_emb]
        x = self.input_proj(x)                             # [B, T, d_model]

        # 前置CLS token
        cls = self.cls_token.expand(B, -1, -1)             # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)                     # [B, T+1, d_model]
        x = self.pos_enc(x)

        x = self.encoder(x)                                # [B, T+1, d_model]
        x = self.norm(x)
        cls_out = x[:, 0]                                  # [B, d_model]

        cur_t = self.cur_proj(cur_time_feat)               # [B, d_model]
        combined = torch.cat([cls_out, cur_t], dim=-1)     # [B, 2*d_model]

        feat = self.feat_proj(combined)                    # [B, feat_dim]
        vis_pred    = self.vis_head(feat)
        future_pred = self.future_head(feat)
        return feat, vis_pred, future_pred
