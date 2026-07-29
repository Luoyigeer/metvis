"""TimesNet 纯时序基线（不使用图像）"""
# 直接复用 temporal_branch.TimesNetBranch
from models.temporal_branch import TimesNetBranch
TimesNetBaseline = TimesNetBranch
