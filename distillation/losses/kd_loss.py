"""Logit-level knowledge distillation losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation import LOSSES


@LOSSES.register("kl_div_loss")
class KLDivLoss(nn.Module):
    """Standard KL-divergence distillation loss (Hinton et al., 2015).

    Args:
        temperature: Softmax temperature for softening logits.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature
        s_log_prob = F.log_softmax(student_logits / T, dim=1)
        t_prob = F.softmax(teacher_logits / T, dim=1)
        loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T)
        return loss


@LOSSES.register("seg_kd_loss")
class SegKDLoss(nn.Module):
    """Pixel-wise KD loss for segmentation logits.

    Applies KL divergence at every spatial location.

    Args:
        temperature: Softmax temperature.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, C, H, W] student segmentation logits.
            teacher_logits: [B, C, H, W] teacher segmentation logits.
        """
        # Spatially align if needed
        if student_logits.shape[2:] != teacher_logits.shape[2:]:
            student_logits = F.interpolate(
                student_logits, size=teacher_logits.shape[2:],
                mode="bilinear", align_corners=False
            )

        T = self.temperature
        b, c, h, w = student_logits.shape

        # Reshape to [B*H*W, C]
        s = student_logits.permute(0, 2, 3, 1).reshape(-1, c)
        t = teacher_logits.permute(0, 2, 3, 1).reshape(-1, c)

        s_log_prob = F.log_softmax(s / T, dim=1)
        t_prob = F.softmax(t / T, dim=1)

        loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T)
        return loss


@LOSSES.register("det_kd_loss")
class DetKDLoss(nn.Module):
    """Multi-level logit distillation for detection heads.

    Applies KD loss at each FPN level and averages.

    Args:
        temperature: Softmax temperature.
        cls_weight: Weight for classification KD.
        reg_weight: Weight for regression KD (L1 loss).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        student_cls_list: list,
        teacher_cls_list: list,
        student_reg_list: list = None,
        teacher_reg_list: list = None,
    ) -> torch.Tensor:
        T = self.temperature
        cls_loss = torch.tensor(0.0, device=student_cls_list[0].device)
        reg_loss = torch.tensor(0.0, device=student_cls_list[0].device)
        n_levels = len(student_cls_list)

        for s_cls, t_cls in zip(student_cls_list, teacher_cls_list):
            if s_cls.shape[2:] != t_cls.shape[2:]:
                s_cls = F.interpolate(
                    s_cls, size=t_cls.shape[2:],
                    mode="bilinear", align_corners=False
                )
            b, c, h, w = s_cls.shape
            s_flat = s_cls.permute(0, 2, 3, 1).reshape(-1, c)
            t_flat = t_cls.permute(0, 2, 3, 1).reshape(-1, c)

            s_log_prob = F.log_softmax(s_flat / T, dim=1)
            t_prob = F.softmax(t_flat / T, dim=1)
            cls_loss += F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T)

        if student_reg_list and teacher_reg_list:
            for s_reg, t_reg in zip(student_reg_list, teacher_reg_list):
                if s_reg.shape[2:] != t_reg.shape[2:]:
                    s_reg = F.interpolate(
                        s_reg, size=t_reg.shape[2:],
                        mode="bilinear", align_corners=False
                    )
                reg_loss += F.smooth_l1_loss(s_reg, t_reg.detach())

        total = (self.cls_weight * cls_loss + self.reg_weight * reg_loss) / n_levels
        return total
