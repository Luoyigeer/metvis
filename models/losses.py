# models/losses.py
"""
自定义损失函数模块
- Focal Loss: 解决类别极度不平衡问题
- 加权回归损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-class Classification
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 类别权重，可以是 [K] tensor 或 float
        gamma: 聚焦参数，γ>=0，越大越关注难样本
        label_smoothing: 标签平滑系数
        reduction: 'mean', 'sum', 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, K] logits
            targets: [B] class indices
        """
        K = inputs.size(1)
        
        # 1. Label Smoothing
        if self.label_smoothing > 0:
            smooth_pos = 1.0 - self.label_smoothing
            smooth_neg = self.label_smoothing / (K - 1)
            # 创建平滑后的标签分布
            smoothed_targets = torch.full_like(inputs, smooth_neg)
            smoothed_targets.scatter_(1, targets.unsqueeze(1), smooth_pos)
            # 使用 KL 散度或直接使用平滑标签计算 CE
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            # 标准 cross entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # 2. 计算 p_t (模型对正确类的预测概率)
        probs = F.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 3. Focal Weight: (1 - p_t)^γ
        focal_weight = (1 - pt) ** self.gamma
        
        # 4. 应用类别权重 α_t
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # 5. Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    组合损失：分类 Focal Loss + 回归 SmoothL1Loss
    
    支持动态权重调整
    """
    def __init__(self, 
                 cls_weight=1.0, 
                 reg_weight=1.0,
                 focal_gamma=2.0,
                 label_smoothing=0.05):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # 回归损失
        self.reg_loss = nn.SmoothL1Loss()
        
        # Focal Loss (alpha 在训练前设置)
        self.cls_loss = None
        
    def set_class_weights(self, alpha):
        """设置类别权重"""
        self.cls_loss = FocalLoss(
            alpha=alpha,
            gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing
        )
    
    def forward(self, cls_logits, cls_targets, pred_vis, gt_vis):
        """
        Args:
            cls_logits: [B, K] 分类 logits
            cls_targets: [B] 分类标签
            pred_vis: [B] 预测能见度（归一化）
            gt_vis: [B] 真值能见度（归一化或米制）
        """
        total_loss = torch.tensor(0.0, device=cls_logits.device)
        
        # 分类损失
        valid_cls = cls_targets >= 0
        if valid_cls.any() and self.cls_loss is not None:
            cls_loss_val = self.cls_loss(
                cls_logits[valid_cls], 
                cls_targets[valid_cls]
            )
            total_loss = total_loss + self.cls_weight * cls_loss_val
        
        # 回归损失
        valid_reg = gt_vis >= 0
        if valid_reg.any():
            reg_loss_val = self.reg_loss(
                pred_vis[valid_reg], 
                gt_vis[valid_reg]
            )
            total_loss = total_loss + self.reg_weight * reg_loss_val
        
        return total_loss