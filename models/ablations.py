"""
消融实验模型变体

变体列表 (8个):
  Full         完整模型 (UGG + TVKD + TCAM + 对比损失)
  w/o_TS       去掉时序分支 (纯图像CNN+ViT)
  w/o_Gate     固定权重融合α=0.3，替代自适应门控
  w/o_Contra   对比损失权重置0 (结构不变)
  w/o_UGG      不确定性门控→退回available_steps门控 (方向二消融)
  w/o_KD       去掉时序→视觉知识蒸馏 (方向一消融)
  w/o_UGG_KD   同时去掉UGG和KD
  w/o_TCAM     去掉时间条件自适应 (方向三消融)

所有变体:
  - 统一前向签名: forward(img, depth, trans, seq_feat, time_feat_seq,
                          cur_time_feat, available_steps, delta_hours=None)
  - 统一输出 dict key，与 VisibilityLoss 和 eval_epoch 兼容
"""
import torch
import torch.nn as nn
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.image_branch import ImageBranch
from models.temporal_branch     import TimesNetBranch
from models.fusion       import FusionHead


# ================================================================
# 工具: 从 EDL 输出 dict 提取兼容字段
# ================================================================
def _cls_logits(cls_out: dict) -> torch.Tensor:
    """从 EvidentialClsHead 输出取 cls_logits"""
    return cls_out["cls_logits"]

def _vis_pred(reg_out: dict) -> torch.Tensor:
    """从 EvidentialRegHead 输出取 vis_pred"""
    return reg_out["vis_pred"]


# ================================================================
# 完整输出 dict 模板
# ================================================================
def _make_output(img_feat, img_cls_out, img_reg_out,
                 ts_feat, ts_vis_reg_out, ts_future_reg_out,
                 fused_feat, fused_cls_out, fused_reg_out,
                 gate_alpha,
                 gate_u_img=None, gate_u_ts=None,
                 tcam_gamma=None, tcam_beta=None,
                 delta_hours=None) -> dict:
    """构建统一格式的输出 dict，兼容 VisibilityLoss / eval_epoch"""
    B = img_feat.shape[0]
    dev = img_feat.device

    if gate_u_img is None:
        gate_u_img = torch.zeros(B, device=dev)
    if gate_u_ts is None:
        gate_u_ts = torch.zeros(B, device=dev)
    if delta_hours is None:
        delta_hours = torch.zeros(B, device=dev)

    return {
        # 图像分支
        "img_feat":           img_feat,
        "img_cls_out":        img_cls_out,
        "img_reg_out":        img_reg_out,
        # TCAM（可为 None）
        "tcam_gamma":         tcam_gamma,
        "tcam_beta":          tcam_beta,
        # 时序分支
        "ts_feat":            ts_feat,
        "ts_vis_reg_out":     ts_vis_reg_out,
        "ts_future_reg_out":  ts_future_reg_out,
        # 融合头
        "fused_feat":         fused_feat,
        "fused_cls_out":      fused_cls_out,
        "fused_reg_out":      fused_reg_out,
        "gate_alpha":         gate_alpha,
        # 门控信号
        "gate_u_img":         gate_u_img,
        "gate_u_ts":          gate_u_ts,
        "delta_hours":        delta_hours,
        # 兼容旧接口（eval_epoch 使用）
        "fused_cls_logits":   _cls_logits(fused_cls_out),
        "fused_vis_pred":     _vis_pred(fused_reg_out),
        "img_cls_logits":     _cls_logits(img_cls_out),
        "img_vis_pred":       _vis_pred(img_reg_out),
        "ts_vis_pred":        _vis_pred(ts_vis_reg_out),
        "ts_future_pred":     _vis_pred(ts_future_reg_out),
    }


def _zero_ts_outputs(B: int, device: torch.device) -> tuple:
    """当无时序分支时，返回零值时序输出"""
    from models.edl import EvidentialRegHead
    # 构造最小 EDL reg_out dict（保持接口兼容）
    dummy_feat  = torch.zeros(B, config.TIME_FEAT_DIM, device=device)
    dummy_reg_out = {
        "gamma":     torch.full((B, 1), 0.0, device=device),
        "nu":        torch.ones(B, 1, device=device),
        "alpha":     torch.full((B, 1), 2.0, device=device),
        "beta":      torch.ones(B, 1, device=device),
        "vis_pred":  torch.zeros(B, 1, device=device),
        "epistemic": torch.zeros(B, 1, device=device),
        "aleatoric": torch.zeros(B, 1, device=device),
        "total_unc": torch.zeros(B, 1, device=device),
    }
    return dummy_feat, dummy_reg_out, dummy_reg_out


# ================================================================
# A: w/o_TS — 纯图像，去掉时序分支
# ================================================================
class AblationNoTS(nn.Module):
    """
    消融A: 去掉时序分支，只使用 CNN+ViT 图像分支。
    gate_alpha 固定为 0，融合头直接输出图像特征的 EDL 预测。
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.img_branch = ImageBranch(
            feat_dim=config.IMAGE_FEAT_DIM,
            num_classes=config.NUM_VIS_CLASSES,
            pretrained=pretrained,
        )
        # 独立融合头（无时序输入，输入维度=IMAGE_FEAT_DIM）
        from models.edl import EvidentialClsHead, EvidentialRegHead
        self.fused_cls = EvidentialClsHead(config.IMAGE_FEAT_DIM, config.NUM_VIS_CLASSES)
        self.fused_reg = EvidentialRegHead(config.IMAGE_FEAT_DIM, hidden_dim=64)

    def forward(self, img, depth, trans,
                seq_feat=None, time_feat_seq=None,
                cur_time_feat=None, available_steps=None,
                delta_hours=None):
        B = img.shape[0]
        dev = img.device

        # 图像分支（返回4个值：feat, cls_out, reg_out, tcam_out）
        img_feat, img_cls_out, img_reg_out, tcam_out = self.img_branch(
            img, depth, trans,
            cur_time_feat=cur_time_feat if cur_time_feat is not None
                          else torch.zeros(B, 10, device=dev),
            delta_hours=delta_hours if delta_hours is not None
                        else torch.zeros(B, device=dev),
        )

        # 零时序输出
        ts_feat, ts_vis_reg, ts_fut_reg = _zero_ts_outputs(B, dev)

        # 融合 = 纯图像特征
        fused_cls_out = self.fused_cls(img_feat)
        fused_reg_out = self.fused_reg(img_feat)
        gate_alpha    = torch.zeros(B, 1, device=dev)

        return _make_output(
            img_feat, img_cls_out, img_reg_out,
            ts_feat, ts_vis_reg, ts_fut_reg,
            img_feat, fused_cls_out, fused_reg_out,
            gate_alpha,
            tcam_gamma=tcam_out.get("gamma"),
            tcam_beta =tcam_out.get("beta"),
            delta_hours=delta_hours if delta_hours is not None
                        else torch.zeros(B, device=dev),
        )


# ================================================================
# B: w/o_Gate — 固定权重融合 α=0.3
# ================================================================
class FixedWeightFusion(nn.Module):
    """固定 α=0.3 的加权融合，替代自适应门控"""
    def __init__(self, img_dim, ts_dim, fused_dim, num_classes,
                 fixed_alpha: float = 0.3):
        super().__init__()
        self.alpha    = fixed_alpha
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU(),
        )
        self.ts_proj  = nn.Sequential(
            nn.Linear(ts_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU(),
        )
        self.post = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.LayerNorm(fused_dim),
            nn.GELU(), nn.Dropout(0.1),
        )
        from models.edl import EvidentialClsHead, EvidentialRegHead
        self.cls_head = EvidentialClsHead(fused_dim, num_classes)
        self.reg_head = EvidentialRegHead(fused_dim, hidden_dim=64)

    def forward(self, img_feat, ts_feat, available_steps=None):
        fused = self.post(
            (1 - self.alpha) * self.img_proj(img_feat) +
            self.alpha        * self.ts_proj(ts_feat)
        )
        B = img_feat.shape[0]
        alpha_t = torch.full((B, 1), self.alpha, device=img_feat.device)
        return fused, self.cls_head(fused), self.reg_head(fused), alpha_t, alpha_t


class AblationNoGate(nn.Module):
    """消融B: 去掉自适应门控，改为固定权重融合 α=0.3"""
    def __init__(self, d_in=config.NUM_METEO_FEATURES + 1,
                 pretrained=True, fixed_alpha=0.3):
        super().__init__()
        self.img_branch = ImageBranch(
            feat_dim=config.IMAGE_FEAT_DIM,
            num_classes=config.NUM_VIS_CLASSES,
            pretrained=pretrained,
        )
        self.ts_branch = TimesNetBranch(d_in=d_in, feat_dim=config.TIME_FEAT_DIM)
        self.fusion    = FixedWeightFusion(
            img_dim=config.IMAGE_FEAT_DIM,
            ts_dim=config.TIME_FEAT_DIM,
            fused_dim=config.FUSED_DIM,
            num_classes=config.NUM_VIS_CLASSES,
            fixed_alpha=fixed_alpha,
        )

    def forward(self, img, depth, trans,
                seq_feat, time_feat_seq, cur_time_feat, available_steps,
                delta_hours=None):
        B   = img.shape[0]
        dev = img.device
        dh  = delta_hours if delta_hours is not None else torch.zeros(B, device=dev)
        ct  = cur_time_feat if cur_time_feat is not None else torch.zeros(B, 10, device=dev)

        img_feat, img_cls_out, img_reg_out, tcam_out = self.img_branch(
            img, depth, trans, cur_time_feat=ct, delta_hours=dh)

        # ts_branch 返回 (feat, vis_reg_out_dict, future_reg_out_dict)
        ts_feat, ts_vis_reg_out, ts_future_reg_out = self.ts_branch(
            seq_feat, time_feat_seq, cur_time_feat)

        fused_feat, fused_cls_out, fused_reg_out, gate_alpha, _gate_raw = self.fusion(
            img_feat, ts_feat, available_steps)

        return _make_output(
            img_feat, img_cls_out, img_reg_out,
            ts_feat, ts_vis_reg_out, ts_future_reg_out,
            fused_feat, fused_cls_out, fused_reg_out,
            gate_alpha,
            gate_u_img=img_cls_out["uncertainty"].squeeze(-1),
            gate_u_ts=ts_vis_reg_out["epistemic"].squeeze(-1),
            tcam_gamma=tcam_out.get("gamma"),
            tcam_beta=tcam_out.get("beta"),
            delta_hours=dh,
        )


class AblationNoUGG_KD(nn.Module):
    """同时去掉不确定性门控和知识蒸馏"""

    def __init__(self, d_in=config.NUM_METEO_FEATURES + 1, pretrained=True):
        super().__init__()
        from models.visibility_model import VisibilityModel
        self.model = VisibilityModel(
            d_in=d_in, pretrained=pretrained, use_unc_gate=False,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ================================================================
# C: w/o_Contra — 结构与完整模型相同，训练时对比损失权重=0
# ================================================================
class AblationNoContra(nn.Module):
    def __init__(self, d_in=config.NUM_METEO_FEATURES + 1, pretrained=True):
        super().__init__()
        from models.visibility_model import VisibilityModel
        self.model = VisibilityModel(d_in=d_in, pretrained=pretrained)

    def forward(self, img, depth, trans,
                seq_feat, time_feat_seq, cur_time_feat, available_steps,
                delta_hours=None):
        return self.model(img, depth, trans, seq_feat, time_feat_seq,
                          cur_time_feat, available_steps, delta_hours=delta_hours)


# ================================================================
# 方向一: w/o_KD — 关闭知识蒸馏
# ================================================================
class AblationNoKD(nn.Module):
    def __init__(self, d_in=config.NUM_METEO_FEATURES + 1, pretrained=True):
        super().__init__()
        from models.visibility_model import VisibilityModel
        self.model = VisibilityModel(d_in=d_in, pretrained=pretrained)

    def forward(self, img, depth, trans,
                seq_feat, time_feat_seq, cur_time_feat, available_steps,
                delta_hours=None):
        return self.model(img, depth, trans, seq_feat, time_feat_seq,
                          cur_time_feat, available_steps, delta_hours=delta_hours)


# ================================================================
# 方向二: w/o_UGG — 退回 available_steps 门控
# ================================================================
class AblationNoUGG(nn.Module):
    def __init__(self, d_in=config.NUM_METEO_FEATURES + 1, pretrained=True):
        super().__init__()
        from models.visibility_model import VisibilityModel
        self.model = VisibilityModel(
            d_in=d_in, pretrained=pretrained, use_unc_gate=False,
        )

    def forward(self, img, depth, trans,
                seq_feat, time_feat_seq, cur_time_feat, available_steps,
                delta_hours=None):
        return self.model(img, depth, trans, seq_feat, time_feat_seq,
                          cur_time_feat, available_steps, delta_hours=delta_hours)


class AblationNoTCAM(nn.Module):
    def __init__(self, d_in=config.NUM_METEO_FEATURES + 1, pretrained=True):
        super().__init__()
        from models.visibility_model import VisibilityModel
        self.model = VisibilityModel(
            d_in=d_in, pretrained=pretrained, use_tcam=False,
        )

    def forward(self, img, depth, trans,
                seq_feat, time_feat_seq, cur_time_feat, available_steps,
                delta_hours=None):
        return self.model(img, depth, trans, seq_feat, time_feat_seq,
                          cur_time_feat, available_steps, delta_hours=delta_hours)


# ================================================================
# 注册表 & 工厂函数
# ================================================================
ABLATION_CONFIGS = {
    # name: (model_class, extra_kwargs, contra_weight)
    "Full":         (None,             {},                  config.LOSS_CONTRA_WEIGHT),
    "w/o_TS":       (AblationNoTS,     {"pretrained": True}, 0.2),
    "w/o_Gate":     (AblationNoGate,   {"pretrained": True}, 0.2),
    "w/o_Contra":   (AblationNoContra, {"pretrained": True}, 0.0),
    "w/o_UGG":      (AblationNoUGG,    {"pretrained": True}, 0.2),
    "w/o_KD":       (AblationNoKD,     {"pretrained": True}, 0.2),
    "w/o_UGG_KD":   (AblationNoUGG_KD, {"pretrained": True}, 0.2),
    "w/o_TCAM":     (AblationNoTCAM,   {"pretrained": True}, 0.2),
}

from ablation_routing import (
    ABLATION_VARIANTS,
    STRUCTURAL_VARIANTS,
    _NO_KD_VARIANTS,
    resolve_variant_name,
    is_structural_variant as _routing_is_structural,
    get_use_kd as _routing_get_use_kd,
)


def _resolve_name(name: str) -> str:
    """将下划线变体名解析为规范的 '/' 形式，不存在则原样返回。"""
    return resolve_variant_name(name)


def build_ablation_model(name: str, d_in: int = config.NUM_METEO_FEATURES + 1,
                         model_kwargs: dict = None):
    name = _resolve_name(name)
    extra = dict(model_kwargs or {})
    if name == "Full":
        from models.visibility_model import VisibilityModel
        return VisibilityModel(d_in=d_in, pretrained=True, **extra)
    if name not in ABLATION_CONFIGS:
        raise ValueError(f"未知消融变体: {name}，可选: {list(ABLATION_CONFIGS)}")
    cls, kwargs, _ = ABLATION_CONFIGS[name]
    if hasattr(cls.__init__, '__code__') and \
       'd_in' in cls.__init__.__code__.co_varnames:
        kwargs = dict(kwargs, d_in=d_in)
    if extra and hasattr(cls, 'model'):
        # wrapper variants delegate to inner VisibilityModel via extra at train time
        pass
    return cls(**kwargs)


def get_contra_weight(name: str) -> float:
    """获取变体的对比损失权重"""
    name = _resolve_name(name)
    return ABLATION_CONFIGS.get(name, (None, {}, config.LOSS_CONTRA_WEIGHT))[2]


def get_use_kd(name: str) -> bool:
    """是否启用知识蒸馏"""
    return _routing_get_use_kd(name)


def is_structural_variant(name: str) -> bool:
    return _routing_is_structural(name)

