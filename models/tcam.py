"""
时间条件自适应模块 (Time-Conditioned Adaptive Module, TCAM)
"""
import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_delta_periods(periods=None):
    if periods is None:
        periods = getattr(config, "TCAM_DELTA_PERIODS", [1.0, 2.0, 6.0, 12.0, 24.0])
    return [float(p) for p in periods]


def get_tcam_cond_dim(periods=None) -> int:
    return 10 + len(get_delta_periods(periods)) * 2


# backward-compatible module-level defaults
DELTA_PERIODS_H = get_delta_periods()
DELTA_T_DIM = len(DELTA_PERIODS_H) * 2


def encode_delta_t(delta_hours: float, periods=None) -> np.ndarray:
    feats = []
    for T in get_delta_periods(periods):
        angle = 2 * np.pi * delta_hours / T
        feats.extend([np.sin(angle), np.cos(angle)])
    return np.array(feats, dtype=np.float32)


def encode_delta_t_batch(delta_hours_tensor: torch.Tensor,
                         periods=None) -> torch.Tensor:
    device = delta_hours_tensor.device
    period_list = get_delta_periods(periods)
    periods_t = torch.tensor(period_list, device=device, dtype=torch.float32)
    angles = 2 * np.pi * delta_hours_tensor.unsqueeze(1) / periods_t.unsqueeze(0)
    sin_enc = torch.sin(angles)
    cos_enc = torch.cos(angles)
    return torch.stack([sin_enc, cos_enc], dim=2).view(-1, len(period_list) * 2)


class FiLMGenerator(nn.Module):
    def __init__(self,
                 cond_dim: int,
                 hidden: int = config.TCAM_HIDDEN_DIM,
                 feat_dim: int = config.TCAM_FEAT_DIM):
        super().__init__()
        self.feat_dim = feat_dim
        self.shared = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden, feat_dim)
        self.beta_head = nn.Linear(hidden, feat_dim)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, cond: torch.Tensor):
        h = self.shared(cond)
        return self.gamma_head(h), self.beta_head(h)


class TCAMLayer(nn.Module):
    def __init__(self,
                 feat_dim: int = config.TCAM_FEAT_DIM,
                 cond_dim: int = config.TCAM_INPUT_DIM,
                 hidden: int = config.TCAM_HIDDEN_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.film_gen = FiLMGenerator(cond_dim, hidden, feat_dim)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_gen(cond)
        return gamma * self.norm(feat) + beta


class TCAM(nn.Module):
    def __init__(self,
                 feat_dim: int = config.TCAM_FEAT_DIM,
                 cond_dim: int = None,
                 hidden: int = config.TCAM_HIDDEN_DIM,
                 num_layers: int = config.TCAM_NUM_LAYERS,
                 delta_periods=None,
                 use_tcam: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.delta_periods = get_delta_periods(delta_periods)
        self.cond_dim = cond_dim or get_tcam_cond_dim(self.delta_periods)
        self.use_tcam = use_tcam
        self.layers = nn.ModuleList([
            TCAMLayer(feat_dim, self.cond_dim, hidden)
            for _ in range(num_layers)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(0.05),
        )

    def forward(self,
                feat: torch.Tensor,
                cur_time_feat: torch.Tensor,
                delta_hours: torch.Tensor) -> dict:
        if not self.use_tcam:
            return {"feat": feat, "gamma": None, "beta": None}

        delta_enc = encode_delta_t_batch(delta_hours, self.delta_periods)
        cond = torch.cat([cur_time_feat, delta_enc], dim=-1)
        x = feat
        last_gamma = last_beta = None
        for layer in self.layers:
            x_new = layer(x, cond)
            x = x + x_new
            last_gamma, last_beta = layer.film_gen(cond)
        x = self.ffn(x)
        return {"feat": x, "gamma": last_gamma, "beta": last_beta}


def make_delta_hours(batch_size: int, delta_h: float,
                     device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), delta_h,
                      dtype=torch.float32, device=device)
