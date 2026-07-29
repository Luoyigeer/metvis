"""
证据深度学习 (Evidential Deep Learning, EDL) 核心模块

理论基础:
  - 分类: Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty"
          NeurIPS 2018. 用 Dirichlet 分布对 K 类预测建模.
  - 回归: Amini et al., "Deep Evidential Regression"
          NeurIPS 2020. 用 Normal-Inverse-Gamma (NIG) 分布对回归建模.

核心思想:
  网络不直接输出预测值，而是输出"证据"(evidence)，
  即主观逻辑意义上的置信度累积量，从中推导出:
    - 预测值 (期望)
    - 认知不确定性 (Epistemic): 模型对自身认知的不确定, 可被更多数据减少
    - 偶然不确定性 (Aleatoric): 数据本身固有噪声, 不可被数据减少

在能见度估计中的意义:
  - 高认知不确定性 → 模型没见过类似场景 (例如极端罕见天气)
  - 高偶然不确定性 → 该场景本身能见度模糊 (例如薄雾过渡区)
  - 两者均可用于可视化和下游决策 (如自动驾驶置信度阈值)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ================================================================
# ① 证据分类头  (Dirichlet)
# ================================================================
class EvidentialClsHead(nn.Module):
    """
    输出 Dirichlet 分布参数的分类头

    网络输出: logits -> softplus -> evidence e ∈ ℝ⁺^K
    Dirichlet 参数: α = e + 1  (确保 α > 1)

    推导:
        预测类别概率 = α_k / S,  S = Σ α_k
        总不确定性   = K / S          (vacuity, 越大越不确定)
        认知不确定性 = K / S          (与总不确定性相同, 分类任务)
        分布散度     = Var[p_k] = α_k(S-α_k) / (S²(S+1))
    """
    def __init__(self, in_dim: int, num_classes: int = config.NUM_VIS_CLASSES):
        super().__init__()
        self.K = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        返回 dict:
            evidence:    [B, K]  非负证据 (softplus激活)
            alpha:       [B, K]  Dirichlet参数 α = evidence + 1
            S:           [B, 1]  Dirichlet强度 S = Σ α_k
            prob:        [B, K]  期望类别概率 α_k / S
            uncertainty: [B, 1]  总不确定性 K / S ∈ (0, 1]
            cls_logits:  [B, K]  兼容旧接口的 log概率
        """
        logits = self.net(x)                              # [B, K]
        evidence = F.softplus(logits)                     # [B, K] ≥ 0
        alpha    = evidence + 1.0                         # [B, K] ≥ 1
        S        = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [B, 1]
        prob     = alpha / S                              # [B, K]
        prob     = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        uncertainty = (self.K / S).clamp(max=1.0)         # [B, 1] ∈ (0,1]
        uncertainty = torch.nan_to_num(uncertainty, nan=1.0, posinf=1.0, neginf=1.0)

        return {
            "evidence":    evidence,
            "alpha":       alpha,
            "S":           S,
            "prob":        prob,
            "uncertainty": uncertainty,
            "raw_logits":  logits,
            "cls_logits":  torch.log(prob + 1e-8),        # 兼容CE损失接口
        }


# ================================================================
# ② 证据回归头  (Normal-Inverse-Gamma, NIG)
# ================================================================
class EvidentialRegHead(nn.Module):
    """
    输出 Normal-Inverse-Gamma 分布参数的回归头

    NIG 参数: (γ, ν, α, β)
        γ  : 预测均值 (即回归输出)
        ν  : 虚拟观测数 > 0  (控制均值的不确定性)
        α  : 逆Gamma形状参数 > 1
        β  : 逆Gamma尺度参数 > 0

    不确定性分解:
        认知不确定性 (Epistemic)  = β / (ν · (α - 1))
        偶然不确定性 (Aleatoric)  = β / (α - 1)
        总方差                    = 偶然 + 认知  (近似)

    输出激活:
        γ   → Sigmoid (能见度归一化到[0,1])
        ν   → Softplus + 1e-6
        α   → Softplus + 1 + 1e-6  (确保 α > 1)
        β   → Softplus + 1e-6
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_gamma = nn.Linear(hidden_dim, 1)   # 预测均值
        self.head_nu    = nn.Linear(hidden_dim, 1)   # 虚拟观测数
        self.head_alpha = nn.Linear(hidden_dim, 1)   # 形状参数
        self.head_beta  = nn.Linear(hidden_dim, 1)   # 尺度参数

        # 特殊初始化：让 gamma 初始输出 0.5（sigmoid(0)=0.5）
        nn.init.zeros_(self.head_gamma.weight)
        nn.init.constant_(self.head_gamma.bias, 0.0)

        # nu: 初始输出 ~1 (softplus(0)=~0.69, +1e-6)
        nn.init.zeros_(self.head_nu.weight)
        nn.init.constant_(self.head_nu.bias, 0.0)

        # alpha: 初始输出 ~2 (softplus(0)+1+1e-6 ≈ 1.69+1=2.69)
        nn.init.zeros_(self.head_alpha.weight)
        nn.init.constant_(self.head_alpha.bias, 0.0)

        # beta: 初始输出 ~1
        nn.init.zeros_(self.head_beta.weight)
        nn.init.constant_(self.head_beta.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict:
        """
        返回 dict:
            gamma:       [B, 1]  预测均值 (归一化能见度)
            nu:          [B, 1]  ν > 0
            alpha:       [B, 1]  α > 1
            beta:        [B, 1]  β > 0
            vis_pred:    [B, 1]  同 gamma (兼容旧接口)
            epistemic:   [B, 1]  认知不确定性
            aleatoric:   [B, 1]  偶然不确定性
            total_unc:   [B, 1]  总不确定性
        """
        h = self.shared(x)

        gamma = torch.sigmoid(self.head_gamma(h))             # [B,1] ∈ (0,1)
        nu    = F.softplus(self.head_nu(h))    + 1e-6         # [B,1] > 0
        alpha = F.softplus(self.head_alpha(h)) + 1.0 + 1e-6  # [B,1] > 1
        beta  = F.softplus(self.head_beta(h))  + 1e-6         # [B,1] > 0

        # 不确定性分解
        denom = (nu * (alpha - 1.0)).clamp_min(1e-6)
        epistemic  = beta / denom                 # [B,1]
        aleatoric  = beta / (alpha - 1.0).clamp_min(1e-6)  # [B,1]
        total_unc  = epistemic + aleatoric         # [B,1]

        return {
            "gamma":      gamma,
            "nu":         nu,
            "alpha":      alpha,
            "beta":       beta,
            "vis_pred":   gamma,         # 兼容旧接口
            "epistemic":  epistemic,
            "aleatoric":  aleatoric,
            "total_unc":  total_unc,
        }


# ================================================================
# ③ EDL 分类损失  (Dirichlet NLL + KL 正则)
# ================================================================
class EDLClassificationLoss(nn.Module):
    """
    证据分类损失

    L = Σ_i [ NLL_i + λ_t · KL(Dir(α̃_i) || Dir(1)) ]

    NLL项: 期望的交叉熵 E_Dir[−log p_y] = ψ(S) − ψ(α_y)
           其中 ψ = digamma 函数

    KL正则项:
        α̃ = y(1 − α) + α  (将标签类证据归零后计算KL, 防止错误类积累证据)
        KL 惩罚非标签类的证据累积

    λ_t: 退火系数, 训练中线性增大 (防止过早惩罚)
    """
    def __init__(self, num_classes: int = config.NUM_VIS_CLASSES,
                 annealing_start: float = 0.01,
                 annealing_step:  float = 0.01):
        super().__init__()
        self.K = num_classes
        self.annealing_start = annealing_start
        self.annealing_step  = annealing_step

    def forward(self, alpha: torch.Tensor, labels: torch.Tensor,
                epoch: int = 0, max_epochs: int = 50,
                class_weights=None) -> dict:
        """
        alpha:         [B, K]  Dirichlet 参数 ≥ 1
        labels:        [B]     类别标签 0..K-1
        class_weights: [K] FloatTensor 或 None，用于处理类别不平衡
                       每个样本的损失乘以对应类别的权重
        """
        B = alpha.shape[0]
        y_onehot = F.one_hot(labels, self.K).float()    # [B, K]

        S = alpha.sum(dim=1, keepdim=True)               # [B, 1]

        # NLL: E_Dir[-log p_y] = digamma(S) - digamma(alpha_y)
        # clamp alpha 避免 alpha<1 时 digamma 趋向 -inf
        alpha_safe = alpha.clamp(min=1.0 + 1e-6)
        S_safe     = alpha_safe.sum(dim=1, keepdim=True)
        loss_nll = (y_onehot * (torch.digamma(S_safe)
                                - torch.digamma(alpha_safe))).sum(dim=1)

        # KL 正则
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        loss_kl     = self._kl_divergence(alpha_tilde)

        # 退火系数
        lam = min(1.0, self.annealing_start + epoch * self.annealing_step)

        per_sample = loss_nll + lam * loss_kl           # [B]

        # 类别加权：每个样本乘以其标签对应的权重
        if class_weights is not None:
            w = class_weights.to(alpha.device)          # [K]
            sample_w = w[labels]                        # [B]
            loss = (per_sample * sample_w).mean()
        else:
            loss = per_sample.mean()

        return {
            "total":   loss,
            "nll":     loss_nll.mean(),
            "kl":      loss_kl.mean(),
            "lambda":  lam,
        }

    @staticmethod
    def _kl_divergence(alpha: torch.Tensor) -> torch.Tensor:
        """
        KL(Dir(α) || Dir(1))  逐样本计算（数值稳定版）
        α 全部 clamp ≥ 1+1e-6，避免 lgamma/digamma 在小值处爆炸
        KL 结果 clamp ≥ 0（理论上KL≥0，数值误差可能导致微小负值）
        """
        K      = alpha.shape[1]
        # 确保 alpha > 1，lgamma 和 digamma 在此范围单调且有界
        a      = alpha.clamp(min=1.0 + 1e-6)
        S      = a.sum(dim=1)                             # [B]
        kl = (
            torch.lgamma(S)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            - torch.lgamma(a).sum(dim=1)
            + ((a - 1.0) * (torch.digamma(a)
               - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
        )
        return kl.clamp(min=0.0)   # KL 理论上 ≥ 0


# ================================================================
# ④ EDL 回归损失  (NIG Likelihood + 证据正则)
# ================================================================
class EDLRegressionLoss(nn.Module):
    """
    深度证据回归损失 (Deep Evidential Regression, DER)

    L = L_NIG + λ · L_reg

    L_NIG: −log p(y | γ, ν, α, β) under NIG
         = (α + 1/2) log(2β(1 + Ω))
           − α log β
           − 1/2 log(ν/π)
           − log Γ(α + 1/2) + log Γ(α)
           (Ω = ν(γ - y)²/ (2β))

    L_reg: |y - γ| · (2ν + α)   (错误预测时惩罚高证据)

    参考: Amini et al. NeurIPS 2020, Eq.(8) & Eq.(9)
    """
    def __init__(self, coeff: float = 1e-3):
        super().__init__()
        self.coeff = coeff

    def forward(self, gamma: torch.Tensor, nu: torch.Tensor,
                alpha: torch.Tensor, beta: torch.Tensor,
                targets: torch.Tensor) -> dict:
        """
        gamma, nu, alpha, beta: [B, 1] NIG参数
        targets: [B] 或 [B,1] 归一化能见度真值
        """
        # AMP 下统一用 float32 做数值计算，避免溢出/NaN
        gamma = gamma.float()
        nu = nu.float()
        alpha = alpha.float()
        beta = beta.float()
        targets = targets.float()
        y = targets.view(-1, 1)
        
        # 防止数值溢出
        nu    = nu.clamp(min=1e-6)
        alpha = alpha.clamp(min=1.0 + 1e-6)
        beta  = beta.clamp(min=1e-6)

        omega = 2.0 * beta * (1.0 + nu)                  # [B, 1]
        diff2 = (y - gamma) ** 2                          # [B, 1]

        # ---- NIG log-likelihood (数值稳定版) ----
        # term3 的参数 diff2*nu + omega 必须 > 0，用 clamp 保护
        eps       = torch.tensor(1e-8, device=gamma.device)
        log_arg   = (diff2 * nu + omega).clamp(min=1e-8)

        term1 = 0.5 * torch.log(torch.tensor(math.pi, device=gamma.device) / nu.clamp(min=eps))
        term2 = alpha * torch.log(beta.clamp(min=eps))
        term3 = (alpha + 0.5) * torch.log(log_arg)
        term4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        loss_nig = (term3 - term2 + term4 - term1).squeeze(-1)  # [B]

        # 证据正则
        loss_reg = (torch.abs(y - gamma) * (2.0 * nu + alpha)).squeeze(-1)  # [B]

        # 对单个样本的极端loss值clamp，防止单点爆炸拉崩整个batch
        loss_per_sample = loss_nig + self.coeff * loss_reg
        loss_per_sample = loss_per_sample.clamp(min=-100.0, max=100.0)
        loss = loss_per_sample.mean()
        return {
            "total":    loss,
            "nig":      loss_nig.mean(),
            "reg":      loss_reg.mean(),
        }


# ================================================================
# ⑤ 不确定性提取工具函数
# ================================================================
def extract_cls_uncertainty(edl_cls_out: dict) -> dict:
    """
    从 EvidentialClsHead 输出提取不确定性指标
    返回:
        pred_class:    [B] 预测类别
        confidence:    [B] 预测置信度 (max prob)
        vacuity:       [B] 总不确定性 K/S ∈ (0,1]  越大越不确定
        dissonance:    [B] 证据分歧度 (不同类别证据相互冲突程度)
    """
    alpha       = edl_cls_out["alpha"]          # [B, K]
    S           = edl_cls_out["S"].squeeze(-1)  # [B]
    prob        = edl_cls_out["prob"]           # [B, K]
    uncertainty = edl_cls_out["uncertainty"].squeeze(-1)  # [B]

    pred_class = prob.argmax(dim=1)             # [B]
    confidence = prob.max(dim=1).values         # [B]
    vacuity    = uncertainty                    # [B]

    # 分歧度: 加权平均的归一化证据冲突
    # dissonance_k = Σ_{j≠k} b_j · Jac(b_k, b_j),  b_k = (α_k-1)/S
    beliefs = (alpha - 1.0) / S.unsqueeze(1)   # [B, K] 信念质量
    B, K = beliefs.shape
    diss = torch.zeros(B, device=alpha.device)
    for k in range(K):
        bk = beliefs[:, k]                     # [B]
        for j in range(K):
            if j == k:
                continue
            bj = beliefs[:, j]                 # [B]
            # Jaccard 相似度 (近似冲突度)
            bal = 1.0 - torch.abs(bk - bj) / (bk + bj + 1e-8)
            diss += bk * bal * bj
    # 归一化到 [0,1]
    diss = diss / (beliefs.sum(dim=1) + 1e-8)

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "vacuity":    vacuity,
        "dissonance": diss,
    }


def extract_reg_uncertainty(edl_reg_out: dict) -> dict:
    """
    从 EvidentialRegHead 输出提取不确定性指标 (反归一化到米)
    返回:
        vis_m:       [B] 预测能见度 (米)
        epistemic_m: [B] 认知不确定性 (米, 越大说明数据越稀缺)
        aleatoric_m: [B] 偶然不确定性 (米, 数据本身噪声)
        total_unc_m: [B] 总不确定性 (米)
        ci_low_m:    [B] 95% 置信下界 (米)  γ - 1.96·σ
        ci_high_m:   [B] 95% 置信上界 (米)  γ + 1.96·σ
    """
    MAX_VIS = 50000.0

    gamma = edl_reg_out["gamma"].squeeze(-1)  # [B] 归一化值
    epistemic = edl_reg_out["epistemic"].squeeze(-1)  # [B] 归一化值
    aleatoric = edl_reg_out["aleatoric"].squeeze(-1)  # [B] 归一化值
    total_unc = edl_reg_out["total_unc"].squeeze(-1)  # [B] 归一化值

    # 反归一化到米
    vis_m = gamma * MAX_VIS
    epistemic_m = epistemic * MAX_VIS
    aleatoric_m = aleatoric * MAX_VIS
    total_unc_m = total_unc * MAX_VIS

    # 近似 95% 置信区间 (σ ≈ sqrt(total_unc))
    sigma_m = torch.sqrt(total_unc_m.clamp(min=0) + 1e-8)
    ci_low_m = (vis_m - 1.96 * sigma_m).clamp(min=0)
    ci_high_m = (vis_m + 1.96 * sigma_m).clamp(max=MAX_VIS)

    return {
        "vis_m": vis_m,
        "epistemic_m": epistemic_m,
        "aleatoric_m": aleatoric_m,
        "total_unc_m": total_unc_m,
        "ci_low_m": ci_low_m,
        "ci_high_m": ci_high_m,
    }
