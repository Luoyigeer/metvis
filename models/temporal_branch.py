"""
TimesNet 时序分支 (EDL版)
vis_head 替换为 EvidentialRegHead，输出认知/偶然不确定性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.noaa_dataset import TIME_FEAT_DIM
from models.edl import EvidentialRegHead


# ----------------------------------------------------------------
# Inception 多尺度卷积块
# ----------------------------------------------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_kernels=config.TIMESNET_N_KERNELS):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=2*i+1, padding=i)
            for i in range(num_kernels)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.stack([c(x) for c in self.convs], dim=-1).mean(-1)


# ----------------------------------------------------------------
# TimesBlock
# ----------------------------------------------------------------
class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff,
                 top_k=config.TIMESNET_TOP_K,
                 num_kernels=config.TIMESNET_N_KERNELS):
        super().__init__()
        self.seq_len = seq_len
        self.top_k   = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        res  = x
        # Run FFT in float32 with AMP disabled to avoid cuFFT fp16 size limits.
        orig_dtype = x.dtype
        with torch.cuda.amp.autocast(False):
            x_fp32 = x.float()
            xf   = torch.fft.rfft(x_fp32, dim=1)
        freq = xf.abs().mean(-1)
        freq[:, 0] = 0
        k_eff = min(self.top_k, freq.shape[1])
        if k_eff <= 0:
            return self.norm(res).to(orig_dtype)
        _, idx = torch.topk(freq, k_eff, dim=1)
        periods = T // (idx + 1).clamp(min=1)

        outs = []
        for i in range(k_eff):
            p   = int(periods[:, i].float().mean().item())
            p   = max(p, 1)
            pad = (p - T % p) % p
            xp  = F.pad(x_fp32, (0, 0, 0, pad))
            rows = (T + pad) // p
            x2d = xp.reshape(B, rows, p, D).permute(0, 3, 1, 2)
            x2d = self.conv(x2d)
            x1d = x2d.permute(0, 2, 3, 1).reshape(B, T + pad, D)[:, :T]
            outs.append(x1d)

        x_out = self.norm(torch.stack(outs, 0).mean(0) + res)
        return x_out.to(orig_dtype)


# ----------------------------------------------------------------
# TimesNetBranch  (EDL版)
# ----------------------------------------------------------------
class TimesNetBranch(nn.Module):
    """
    输入:  seq_feat[B,T,D_in], time_feat_seq[B,T,T_emb], cur_time_feat[B,T_emb]
    输出:
        feat:          [B, TIME_FEAT_DIM]
        vis_reg_out:   dict  EvidentialRegHead 输出 (当前能见度 + 不确定性)
        future_reg_out:dict  EvidentialRegHead 输出 (未来单步)
    """
    def __init__(self,
                 d_in=config.NUM_METEO_FEATURES + 1,
                 time_emb_dim=TIME_FEAT_DIM,
                 seq_len=config.SEQ_LEN,
                 d_model=config.TIMESNET_D_MODEL,
                 d_ff=config.TIMESNET_D_FF,
                 num_layers=config.TIMESNET_NUM_LAYERS,
                 top_k=config.TIMESNET_TOP_K,
                 num_kernels=config.TIMESNET_N_KERNELS,
                 feat_dim=config.TIME_FEAT_DIM):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_proj  = nn.Linear(d_in + time_emb_dim, d_model)
        self.layers      = nn.ModuleList([
            TimesBlock(max(1, self.seq_len), d_model, d_ff, top_k, num_kernels)
            for _ in range(num_layers)
        ])
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.cur_proj    = nn.Linear(time_emb_dim, d_model)
        self.feat_proj   = nn.Sequential(
            nn.Linear(d_model * 2, feat_dim),
            nn.LayerNorm(feat_dim), nn.GELU(),
        )
        # EDL 回归头: 当前 + 未来各一个
        self.vis_head    = EvidentialRegHead(feat_dim, hidden_dim=32)
        self.future_head = EvidentialRegHead(feat_dim, hidden_dim=32)

    def forward(self, seq_feat, time_feat_seq, cur_time_feat):
        if self.seq_len <= 0:
            # No temporal history: rely on current-time embedding only.
            B = cur_time_feat.shape[0]
            d_model = self.cur_proj.out_features
            x_pool = torch.zeros(B, d_model, device=cur_time_feat.device, dtype=cur_time_feat.dtype)
        else:
            x    = self.input_proj(torch.cat([seq_feat, time_feat_seq], dim=-1))
            for layer in self.layers:
                x = layer(x)
            x_pool  = self.pool(x.transpose(1, 2)).squeeze(-1)
        cur_t   = self.cur_proj(cur_time_feat)
        feat    = self.feat_proj(torch.cat([x_pool, cur_t], dim=-1))

        vis_reg_out    = self.vis_head(feat)
        future_reg_out = self.future_head(feat)
        return feat, vis_reg_out, future_reg_out
