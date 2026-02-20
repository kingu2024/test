"""Feature-level distillation losses and alignment modules."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation import LOSSES, ALIGNERS


# ---------------------------------------------------------------------------
# Feature aligners: bridge dimension gaps between teacher and student features
# ---------------------------------------------------------------------------


@ALIGNERS.register("conv1x1")
class Conv1x1Aligner(nn.Module):
    """1x1 convolution to align student feature channels to teacher channels."""

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.align(x)


@ALIGNERS.register("conv3x3")
class Conv3x3Aligner(nn.Module):
    """3x3 convolution aligner with more capacity."""

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.align(x)


@ALIGNERS.register("mlp")
class MLPAligner(nn.Module):
    """Two-layer 1x1 MLP aligner with a hidden bottleneck."""

    def __init__(self, student_channels: int, teacher_channels: int,
                 hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(teacher_channels * hidden_ratio), 16)
        self.align = nn.Sequential(
            nn.Conv2d(student_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, teacher_channels, 1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.align(x)


class FeatureAlignmentModule(nn.Module):
    """Manages multi-layer feature alignment between teacher and student.

    Creates one aligner per (student_layer, teacher_layer) pair.

    Args:
        student_channels: List of channel dims for student features.
        teacher_channels: List of channel dims for teacher features.
        aligner_type: Registered aligner type name.
        align_indices: Which index pairs to align. None = align all positions.
    """

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        aligner_type: str = "conv1x1",
        align_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if align_indices is None:
            align_indices = list(range(min(len(student_channels),
                                           len(teacher_channels))))
        self.align_indices = align_indices

        self.aligners = nn.ModuleList()
        for idx in align_indices:
            self.aligners.append(
                ALIGNERS.build(
                    aligner_type,
                    student_channels=student_channels[idx],
                    teacher_channels=teacher_channels[idx],
                )
            )

    def forward(
        self, student_feats: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Align student features to match teacher dimensions.

        Args:
            student_feats: List of student feature tensors.

        Returns:
            List of aligned student feature tensors (only at align_indices).
        """
        aligned = []
        for aligner, idx in zip(self.aligners, self.align_indices):
            aligned.append(aligner(student_feats[idx]))
        return aligned


# ---------------------------------------------------------------------------
# Feature distillation loss functions
# ---------------------------------------------------------------------------


@LOSSES.register("mse_feature_loss")
class MSEFeatureLoss(nn.Module):
    """L2 feature distillation loss with optional spatial normalization."""

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        # Spatially align if sizes differ
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(
                student_feat, size=teacher_feat.shape[2:],
                mode="bilinear", align_corners=False
            )

        if self.normalize:
            student_feat = F.normalize(student_feat, dim=1)
            teacher_feat = F.normalize(teacher_feat, dim=1)

        return F.mse_loss(student_feat, teacher_feat)


@LOSSES.register("cosine_feature_loss")
class CosineFeatureLoss(nn.Module):
    """Cosine similarity based feature distillation loss."""

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(
                student_feat, size=teacher_feat.shape[2:],
                mode="bilinear", align_corners=False
            )

        # Flatten spatial dims: [B, C, H, W] -> [B*H*W, C]
        b, c, h, w = teacher_feat.shape
        s = student_feat.reshape(b * h * w, c)
        t = teacher_feat.reshape(b * h * w, c)

        return (1 - F.cosine_similarity(s, t, dim=1)).mean()


@LOSSES.register("attention_feature_loss")
class AttentionFeatureLoss(nn.Module):
    """Attention transfer loss (Zagoruyko & Komodakis, 2017).

    Aligns spatial attention maps (channel-wise L2 norm) between
    teacher and student.
    """

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def _attention_map(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(feat.pow(self.p).mean(dim=1, keepdim=True), dim=(2, 3))

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        s_att = self._attention_map(student_feat)
        t_att = self._attention_map(teacher_feat)

        if s_att.shape[2:] != t_att.shape[2:]:
            s_att = F.interpolate(
                s_att, size=t_att.shape[2:],
                mode="bilinear", align_corners=False
            )

        return F.mse_loss(s_att, t_att)


@LOSSES.register("channel_wise_feature_loss")
class ChannelWiseFeatureLoss(nn.Module):
    """Channel-wise distillation: aligns per-channel statistics."""

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(
                student_feat, size=teacher_feat.shape[2:],
                mode="bilinear", align_corners=False
            )

        # Per-channel mean and variance
        s_mean = student_feat.mean(dim=(2, 3))
        t_mean = teacher_feat.mean(dim=(2, 3))
        s_var = student_feat.var(dim=(2, 3))
        t_var = teacher_feat.var(dim=(2, 3))

        mean_loss = F.mse_loss(s_mean, t_mean)
        var_loss = F.mse_loss(s_var, t_var)
        return mean_loss + var_loss
