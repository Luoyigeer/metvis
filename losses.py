"""
损失函数模块 (UGG + TVKD 版)

新增:
  TemporalVisualKDLoss  方向一: 时序→视觉知识蒸馏
  UGGRegularizer        方向二: 门控正则损失 (可选, 约束门控行为)

完整损失组成 (VisibilityLoss / Stage3Loss 使用):
  L_cls      EDL分类 (图像+融合)
  L_reg      EDL回归 (图像+融合)
  L_ts       时序EDL回归
  L_contra   NT-Xent对比  [仅 available_steps>0 时计算]
  L_pred     单步预测
  L_kd       时序→视觉知识蒸馏  [仅 available_steps>0 时计算]
  L_gate_reg 门控不确定性正则    [仅 available_steps>0 时计算]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.edl import EDLClassificationLoss, EDLRegressionLoss


# ================================================================
# 方向三: TCAM 预测损失 + 光照自适应正则
# ================================================================
class TCAMLoss(nn.Module):
    """
    TCAM 两路损失:

    (A) 预测损失 (delta_hours > 0 的样本)
        当 batch 中有 delta_hours>0 的样本时，用对应的 future_vis 标注
        监督融合头输出。ground truth 来自数据增强:
        Stage3Loss 对每个 batch 随机选半数样本注入 delta_hours ∈ (0, pred_max]
        并把对应的 vis_val 向前推 delta_hours 小时（若无未来标注则用-1跳过）

    (B) 光照对比正则 (可选, 同一场景不同时刻特征一致性)
        暂时简化为: TCAM gamma 参数的 L2 范数惩罚（防止仿射变换过大）
        L_reg = mean(||gamma - 1||^2 + ||beta||^2)
    """
    def __init__(self,
                 w_pred:  float = config.TCAM_PRED_WEIGHT,
                 w_reg:   float = 1e-3):
        super().__init__()
        self.w_pred = w_pred
        self.w_reg  = w_reg
        self.reg    = nn.SmoothL1Loss()

    def forward(self, outputs: dict, batch: dict) -> dict:
        device    = outputs["fused_vis_pred"].device
        zero      = torch.tensor(0.0, device=device)

        delta_h   = batch.get("delta_hours", None)   # [B]
        future_vis = batch.get("future_vis", None)    # [B] 归一化

        # ---- (A) 预测损失 ----
        L_pred = zero.clone()
        if (delta_h is not None and future_vis is not None
                and config.USE_TCAM):
            pred_mask = (delta_h > 0) & (future_vis >= 0)
            if pred_mask.any():
                pred_fused = outputs["fused_reg_out"]["gamma"][pred_mask].squeeze(-1)
                gt_fut     = future_vis[pred_mask]
                valid = torch.isfinite(pred_fused) & torch.isfinite(gt_fut)
                if valid.any():
                    if getattr(config, "USE_METER_LABELS", False):
                        L_pred = self.reg(pred_fused[valid], gt_fut[valid])
                    else:
                        L_pred = self.reg(
                            pred_fused[valid],
                            gt_fut[valid].clamp(0.0, 1.0)
                        )

        # ---- (B) 仿射正则 ----
        L_reg = zero.clone()
        gamma = outputs.get("tcam_gamma")
        beta  = outputs.get("tcam_beta")
        if gamma is not None and beta is not None and config.USE_TCAM:
            gamma = torch.nan_to_num(gamma, nan=1.0, posinf=1.0, neginf=1.0)
            beta  = torch.nan_to_num(beta,  nan=0.0, posinf=0.0, neginf=0.0)
            L_reg = (
                (gamma - 1.0).pow(2).mean() +
                beta.pow(2).mean()
            )

        total = self.w_pred * L_pred + self.w_reg * L_reg
        return {"total": total, "pred": L_pred, "reg": L_reg}


# ================================================================
# NT-Xent 对比损失 (不变)
# ================================================================
class NTXentLoss(nn.Module):
    """
    NT-Xent 对比损失。
    img_feat(512维) 和 ts_feat(64维) 维度不同，通过各自的 MLP 投影头
    映射到公共空间 proj_dim(128维) 后计算相似度。
    投影头在 __init__ 中创建，参与 optimizer 优化。
    参考: Chen et al., SimCLR, ICML 2020
    """
    def __init__(self, temperature: float = config.CONTRA_TEMP,
                 img_dim:     int   = config.IMAGE_FEAT_DIM,   # 512
                 ts_dim:      int   = config.TIME_FEAT_DIM,    # 64
                 proj_dim:    int   = 128):
        super().__init__()
        self.temp = temperature
        # 两路投影头：各自映射到 proj_dim
        self.proj_img = nn.Sequential(
            nn.Linear(img_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )
        self.proj_ts = nn.Sequential(
            nn.Linear(ts_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )

    def forward(self, z_img: torch.Tensor, z_ts: torch.Tensor) -> torch.Tensor:
        """
        z_img: [B, img_dim]  图像特征（L2归一化前）
        z_ts:  [B, ts_dim]   时序特征（L2归一化前）
        """
        z_img = torch.nan_to_num(z_img, nan=0.0, posinf=0.0, neginf=0.0)
        z_ts  = torch.nan_to_num(z_ts,  nan=0.0, posinf=0.0, neginf=0.0)
        B = z_img.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z_img.device)

        # 投影 + L2 归一化
        h_img = F.normalize(self.proj_img(z_img), dim=-1)   # [B, proj_dim]
        h_ts  = F.normalize(self.proj_ts(z_ts),  dim=-1)    # [B, proj_dim]

        # NT-Xent: 同一样本的图像/时序特征互为正例
        z      = torch.cat([h_img, h_ts], dim=0)            # [2B, proj_dim]
        sim    = torch.mm(z, z.T).float() / self.temp       # [2B, 2B]
        sim    = torch.clamp(sim, min=-50.0, max=50.0)
        mask   = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim    = sim.masked_fill(mask, torch.finfo(sim.dtype).min)
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0,  B,  device=z.device),
        ])
        return F.cross_entropy(sim, labels)


# ================================================================
# 方向一: 时序→视觉知识蒸馏  (Temporal-Visual KD)
# ================================================================
class TemporalVisualKDLoss(nn.Module):
    """
    用时序分支的预测分布作为软标签，蒸馏图像分支的分类头。

    理论依据:
      时序分支经过 NOAA 预训练，掌握"某时刻/气象条件下能见度分布"的先验。
      当图像模糊（薄雾过渡区）或标注噪声较大时，时序提供的软标签
      比独热硬标签包含更多气候规律信息，帮助图像分支学到更平滑的决策边界。

    蒸馏方向 (config.KD_DIRECTION):
      "ts2img"        单向: 时序教师 → 图像学生
      "bidirectional" 双向: ts→img  +  img→ts (互蒸馏)

    权重退火:
      λ_kd(epoch) = KD_WEIGHT_MAX × min(1, epoch / KD_WARMUP_EPOCHS)
      前 KD_WARMUP_EPOCHS epoch 线性增大，让时序教师先充分预热。

    损失公式:
      L_kd = τ² × KL( softmax(p_student/τ) || softmax(p_teacher/τ) )
      τ²   补偿温度缩放对梯度幅度的影响 (Hinton et al. 2015)
    """

    def __init__(self,
                 tau:       float = config.KD_TEMP,
                 direction: str   = config.KD_DIRECTION):
        super().__init__()
        self.tau       = tau
        self.direction = direction

    def _kd_loss(self,
                 student_logits: torch.Tensor,   # [B, K]
                 teacher_logits: torch.Tensor,   # [B, K]
                 ) -> torch.Tensor:
        """
        单向 KL 蒸馏损失
        student_logits: 学生的 log_softmax 分类输出 (或 logits)
        teacher_logits: 教师的分类输出 (detach)
        """
        # 教师软标签 (stop_gradient)
        p_teacher = F.softmax(teacher_logits / self.tau, dim=-1).detach()  # [B, K]
        # 学生 log 概率
        log_p_student = F.log_softmax(student_logits / self.tau, dim=-1)   # [B, K]
        kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        return kd * (self.tau ** 2)   # 温度补偿

    def forward(self,
                img_cls_logits: torch.Tensor,    # [B, K] 图像分支分类 logits
                ts_cls_logits:  torch.Tensor,    # [B, K] 时序分支分类 logits
                epoch:          int = 0,
                ) -> dict:
        """
        返回 {"total": loss, "ts2img": ..., "img2ts": ..., "lambda": ...}
        """
        if not config.USE_KD:
            zero = torch.tensor(0.0, device=img_cls_logits.device)
            return {"total": zero, "ts2img": zero, "img2ts": zero, "lambda": 0.0}

        # 退火权重
        lam = config.KD_WEIGHT_MAX * min(
            1.0, epoch / max(config.KD_WARMUP_EPOCHS, 1)
        )

        loss_ts2img = self._kd_loss(img_cls_logits, ts_cls_logits)
        loss_img2ts = torch.tensor(0.0, device=img_cls_logits.device)

        if self.direction == "bidirectional":
            loss_img2ts = self._kd_loss(ts_cls_logits, img_cls_logits)

        total = lam * (loss_ts2img + loss_img2ts)

        return {
            "total":   total,
            "ts2img":  loss_ts2img,
            "img2ts":  loss_img2ts,
            "lambda":  lam,
        }


# ================================================================
# 方向二辅助: 门控正则损失 (UGG Regularizer)
# ================================================================
class UGGRegularizer(nn.Module):
    """
    对门控 alpha 施加软约束，引导门控学习符合预期的行为：

    约束1: 单调性约束 (Monotonicity)
      当图像不确定性高 u_img > threshold 且时序有数据 avail > 0 时，
      alpha 应该 > alpha_min，否则惩罚。
      L_mono = ReLU(α_min - alpha) × I[u_img > u_thresh, avail > 0]

    约束2: 稀疏性约束 (Sparsity)
      当时序无数据 avail = 0 时，alpha 应该接近 0。
      L_sparse = alpha × I[avail = 0]

    两个约束保证门控在极端情况下行为可控，
    同时不干预门控在一般情况下的自由学习。

    权重很小 (默认 1e-3)，作为软约束而非硬约束。
    """

    def __init__(self,
                 alpha_min: float = 0.1,
                 u_thresh:  float = 0.5,
                 w_mono:    float = 1e-3,
                 w_sparse:  float = 1e-3):
        super().__init__()
        self.alpha_min = alpha_min
        self.u_thresh  = u_thresh
        self.w_mono    = w_mono
        self.w_sparse  = w_sparse

    def forward(self,
                gate_alpha:      torch.Tensor,   # [B, 1]
                u_img:           torch.Tensor,   # [B]
                available_steps: torch.Tensor,   # [B]
                ) -> dict:

        gate_alpha = torch.nan_to_num(gate_alpha, nan=0.0, posinf=0.0, neginf=0.0)
        u_img      = torch.nan_to_num(u_img,      nan=0.0, posinf=0.0, neginf=0.0)

        alpha  = gate_alpha.squeeze(-1)          # [B]
        avail  = (available_steps > 0).float()   # [B] 0/1

        # 约束1: 图像不确定高 + 有时序数据 → alpha 不应过小
        high_unc = (u_img > self.u_thresh).float()
        L_mono   = (F.relu(self.alpha_min - alpha) * high_unc * avail).mean()

        # 约束2: 无时序数据 → alpha 应接近 0
        L_sparse = (alpha * (1.0 - avail)).mean()

        total = self.w_mono * L_mono + self.w_sparse * L_sparse
        return {"total": total, "mono": L_mono, "sparse": L_sparse}


# ================================================================
# 主训练损失 (VisibilityLoss: 用于 train_main.py)
# ================================================================
class VisibilityLoss(nn.Module):
    """
    完整多任务损失，集成 UGG 正则和 TVKD 蒸馏。
    epoch 参数驱动: EDL KL退火 + KD权重退火

    关键修改:
      对比损失 / KD / 门控正则 只在 batch 内有 available_steps>0
      的样本时才计算，避免零时序特征污染图像分支梯度。
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.edl_cls   = EDLClassificationLoss(
            num_classes=config.NUM_VIS_CLASSES,
            annealing_start=config.EDL_ANNEAL_START,
            annealing_step=config.EDL_ANNEAL_STEP,
        )
        self.edl_reg   = EDLRegressionLoss(coeff=config.EDL_REG_COEFF)
        self.contra    = NTXentLoss()
        self.tvkd      = TemporalVisualKDLoss()      # 方向一
        self.ugg_reg   = UGGRegularizer()            # 方向二辅助
        self.tcam_loss = TCAMLoss()                  # 方向三
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, outputs: dict, batch: dict, epoch: int = 0) -> dict:
        vis_cls    = batch["vis_cls"]
        vis_val    = batch["vis_val"]
        future_vis = batch.get("future_vis", None)
        device     = vis_cls.device
        zero       = torch.tensor(0.0, device=device)
        time_prior_prob = batch.get("time_prior_prob", None)
        avail_steps = batch.get("available_steps", None)

        valid_cls = vis_cls >= 0
        valid_reg = vis_val >= 0

        from experiments.ablation_common import _stage3_vis_tgt_norm
        vis_tgt = _stage3_vis_tgt_norm(vis_val) if valid_reg.any() else vis_val

        # 判断 batch 内哪些样本有真实时序数据
        if avail_steps is not None:
            valid_contra = avail_steps > 0      # [B] bool
        else:
            valid_contra = torch.ones(vis_cls.shape[0], dtype=torch.bool, device=device)

        # ---- EDL 分类 (图像 + 融合) ----
        loss_cls = zero.clone()
        if valid_cls.any():
            d1 = self.edl_cls(outputs["img_cls_out"]["alpha"][valid_cls],
                              vis_cls[valid_cls], epoch,
                              class_weights=self.class_weights)
            d2 = self.edl_cls(outputs["fused_cls_out"]["alpha"][valid_cls],
                              vis_cls[valid_cls], epoch,
                              class_weights=self.class_weights)
            loss_cls = (d1["total"] + d2["total"]) * 0.5

        # ---- EDL 回归 (图像 + 融合) ----
        loss_reg = zero.clone()
        if valid_reg.any():
            ro  = outputs["img_reg_out"]
            ro2 = outputs["fused_reg_out"]
            r1  = self.edl_reg(ro["gamma"][valid_reg],  ro["nu"][valid_reg],
                               ro["alpha"][valid_reg],  ro["beta"][valid_reg],
                               vis_tgt[valid_reg])["total"]
            r2  = self.edl_reg(ro2["gamma"][valid_reg], ro2["nu"][valid_reg],
                               ro2["alpha"][valid_reg], ro2["beta"][valid_reg],
                               vis_tgt[valid_reg])["total"]
            loss_reg = (r1 + r2) * 0.5

        # ---- 时序 EDL 回归 ----
        loss_ts = zero.clone()
        if valid_reg.any() and valid_contra.any():
            ts_mask = valid_reg & valid_contra
            if ts_mask.any():
                ts = outputs["ts_vis_reg_out"]
                loss_ts = self.edl_reg(
                    ts["gamma"][ts_mask], ts["nu"][ts_mask],
                    ts["alpha"][ts_mask], ts["beta"][ts_mask],
                    vis_tgt[ts_mask])["total"]

        # ---- 对比损失（只在有时序数据时计算）----
        loss_contra = zero.clone()
        if valid_contra.sum() >= 2:
            loss_contra = self.contra(
                outputs["img_feat"][valid_contra],
                outputs["ts_feat"][valid_contra],
            )

        # ---- 单步预测 ----
        loss_pred = zero.clone()
        if future_vis is not None:
            valid_fut = future_vis >= 0
            if valid_fut.any():
                tf = outputs["ts_future_reg_out"]
                loss_pred = self.edl_reg(
                    tf["gamma"][valid_fut], tf["nu"][valid_fut],
                    tf["alpha"][valid_fut], tf["beta"][valid_fut],
                    future_vis[valid_fut])["total"]

        # ---- 方向一: 时序→视觉知识蒸馏（只在有时序数据时计算）----
        loss_kd = zero.clone()
        kd_dict = {"total": loss_kd, "lambda": 0.0}
        if valid_contra.sum() >= 2:
            from experiments.ablation_common import _stage3_ts_cls_teacher_logits
            ts_teacher = _stage3_ts_cls_teacher_logits(outputs, device)
            if ts_teacher is None:
                ts_teacher = outputs["fused_cls_out"]["cls_logits"].detach()
            kd_dict = self.tvkd(
                outputs["img_cls_out"]["cls_logits"],
                ts_teacher,
                epoch=epoch,
            )
            loss_kd = kd_dict["total"]

        # ---- 方向二: 门控正则（只在有时序数据时计算）----
        loss_gate_reg = zero.clone()
        if valid_contra.any():
            ugg_dict = self.ugg_reg(
                outputs["gate_alpha"],
                outputs["gate_u_img"],
                batch["available_steps"],
            )
            loss_gate_reg = ugg_dict["total"]

        # ---- 方向三: TCAM 损失 ----
        tcam_dict   = self.tcam_loss(outputs, batch)
        loss_tcam   = tcam_dict["total"]

        # ---- 时间先验软标签 (无时序数据时) ----
        loss_time_prior = zero.clone()
        if time_prior_prob is not None and avail_steps is not None:
            prior_mask = avail_steps <= 0
            if prior_mask.any():
                prob = outputs["fused_cls_out"]["prob"][prior_mask]
                prior = time_prior_prob[prior_mask].to(prob.device)
                loss_time_prior = F.kl_div(
                    (prob + 1e-8).log(), prior, reduction="batchmean"
                )

        total = (
            config.LOSS_CLS_WEIGHT    * loss_cls      +
            config.LOSS_REG_WEIGHT    * loss_reg      +
            config.LOSS_TS_WEIGHT     * loss_ts       +
            config.LOSS_CONTRA_WEIGHT * loss_contra   +
            config.LOSS_PRED_WEIGHT   * loss_pred     +
            config.KD_WEIGHT_MAX      * loss_kd       +   # 方向一
            loss_gate_reg                              +   # 方向二
            loss_tcam                                      # 方向三
            + config.TIME_PRIOR_WEIGHT * loss_time_prior
        )

        return {
            "total":      total,
            "cls":        loss_cls,
            "reg":        loss_reg,
            "ts":         loss_ts,
            "contra":     loss_contra,
            "pred":       loss_pred,
            "kd":         loss_kd,           # 方向一
            "kd_lambda":  kd_dict["lambda"],
            "gate_reg":   loss_gate_reg,     # 方向二
            "tcam":       loss_tcam,         # 方向三
            "time_prior": loss_time_prior,
        }


# ================================================================
# NOAA 预训练损失 (不变)
# ================================================================
class NOAAPretrainLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.edl_reg = EDLRegressionLoss(coeff=config.EDL_REG_COEFF)
        self.edl_cls = EDLClassificationLoss(
            num_classes=config.NUM_VIS_CLASSES,
            annealing_start=config.EDL_ANNEAL_START,
            annealing_step=config.EDL_ANNEAL_STEP,
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, outputs: dict, batch: dict) -> dict:
        target_vis = batch["target_vis"]
        future_vis = batch["future_vis"]
        target_cls = batch.get("target_cls", None)
        ts  = outputs["ts_vis_reg_out"]
        tf  = outputs["ts_future_reg_out"]
        loss_reg  = self.edl_reg(
            ts["gamma"], ts["nu"], ts["alpha"], ts["beta"], target_vis)["total"]
        loss_pred = self.edl_reg(
            tf["gamma"], tf["nu"], tf["alpha"], tf["beta"], future_vis)["total"]

        loss_cls = torch.tensor(0.0, device=target_vis.device)
        if target_cls is not None and self.edl_cls is not None:
            valid = target_cls >= 0
            if valid.any():
                loss_cls = self.edl_cls(
                    ts.get("alpha", torch.ones(
                        target_cls.shape[0], config.NUM_VIS_CLASSES,
                        device=target_vis.device)),
                    target_cls[valid] if valid.all() else target_cls.clamp(0),
                    class_weights=self.class_weights,
                )["total"]

        total = loss_reg + 0.3 * loss_pred + 0.5 * loss_cls
        return {"total": total, "reg": loss_reg, "pred": loss_pred, "cls": loss_cls}
