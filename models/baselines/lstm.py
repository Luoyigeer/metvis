"""BiLSTM 时序基线"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import config
from data.noaa_dataset import TIME_FEAT_DIM

class LSTMVis(nn.Module):
    """
    双向LSTM + 注意力池化
    参考: Hochreiter & Schmidhuber, Long Short-Term Memory, Neural Computation 1997
    """
    def __init__(self,
                 d_in: int = config.NUM_METEO_FEATURES + 1,
                 time_emb_dim: int = TIME_FEAT_DIM,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 feat_dim: int = config.TIME_FEAT_DIM,
                 dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(d_in + time_emb_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2  # 双向

        # 自注意力池化 (对时间步加权)
        self.attn_q = nn.Linear(lstm_out_dim, 1)

        # 当前时刻时间编码融合
        self.cur_proj = nn.Linear(time_emb_dim, lstm_out_dim)

        self.feat_proj = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, feat_dim),
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
        """
        seq_feat:       [B, T, D_in]
        time_feat_seq:  [B, T, T_emb]
        cur_time_feat:  [B, T_emb]
        """
        x = torch.cat([seq_feat, time_feat_seq], dim=-1)  # [B, T, D+T_emb]
        x = self.input_proj(x)                            # [B, T, H]

        lstm_out, _ = self.lstm(x)                        # [B, T, 2H]

        # 注意力加权池化
        attn_w = torch.softmax(self.attn_q(lstm_out), dim=1)  # [B, T, 1]
        context = (attn_w * lstm_out).sum(dim=1)              # [B, 2H]

        cur_t = self.cur_proj(cur_time_feat)              # [B, 2H]
        combined = torch.cat([context, cur_t], dim=-1)    # [B, 4H]

        feat = self.feat_proj(combined)                   # [B, feat_dim]
        vis_pred    = self.vis_head(feat)
        future_pred = self.future_head(feat)
        return feat, vis_pred, future_pred


# ----------------------------------------------------------------
# Transformer 时序基线
# ----------------------------------------------------------------
