"""Task-specific losses for segmentation and detection training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation import LOSSES


@LOSSES.register("seg_ce_loss")
class SegCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for semantic segmentation.

    Args:
        ignore_index: Label index to ignore.
    """

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] prediction logits.
            targets: [B, H, W] integer class labels.
        """
        if logits.shape[2:] != targets.shape[1:]:
            logits = F.interpolate(
                logits, size=targets.shape[1:],
                mode="bilinear", align_corners=False
            )
        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)


@LOSSES.register("focal_loss")
class FocalLoss(nn.Module):
    """Focal loss for detection classification (Lin et al., 2017).

    Args:
        alpha: Balancing factor.
        gamma: Focusing parameter.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        return loss.mean()
