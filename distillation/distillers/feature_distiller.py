"""Feature-level distillation between teacher and student backbones."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from distillation import DISTILLERS, LOSSES
from distillation.losses.feature_loss import FeatureAlignmentModule


@DISTILLERS.register("feature_distiller")
class FeatureDistiller(nn.Module):
    """Distills multi-scale backbone features from teacher to student.

    Handles:
      - Channel dimension alignment via learnable adapters.
      - Spatial size interpolation.
      - Multiple feature loss types per layer.

    Args:
        student_channels: Channel dims for student feature levels.
        teacher_channels: Channel dims for teacher feature levels.
        feature_keys: Which feature map keys to distill (e.g., ["s3", "s4", "s5"]).
        aligner_type: How to align channels ("conv1x1", "conv3x3", "mlp").
        loss_type: Feature loss type ("mse_feature_loss", "cosine_feature_loss", etc.).
        loss_weights: Per-level loss weight. Scalar or list matching feature_keys.
    """

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        feature_keys: Optional[List[str]] = None,
        aligner_type: str = "conv1x1",
        loss_type: str = "mse_feature_loss",
        loss_weights: Optional[list] = None,
    ):
        super().__init__()
        self.feature_keys = feature_keys or ["s2", "s3", "s4", "s5"]
        n_levels = len(self.feature_keys)

        # Build alignment module
        self.aligner = FeatureAlignmentModule(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            aligner_type=aligner_type,
            align_indices=list(range(n_levels)),
        )

        # Build loss function
        self.loss_fn = LOSSES.build(loss_type)

        # Per-level weights
        if loss_weights is None:
            self.loss_weights = [1.0] * n_levels
        else:
            assert len(loss_weights) == n_levels
            self.loss_weights = loss_weights

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute feature distillation losses.

        Args:
            student_features: Dict from student backbone.
            teacher_features: Dict from teacher backbone.

        Returns:
            Dict with per-level losses and total loss:
                - "feat_loss_s3", "feat_loss_s4", ... per-level losses
                - "feat_loss_total": weighted sum
        """
        s_feats = [student_features[k] for k in self.feature_keys]
        t_feats = [teacher_features[k] for k in self.feature_keys]

        # Align student features to teacher dimensions
        aligned = self.aligner(s_feats)

        losses = {}
        total = torch.tensor(0.0, device=aligned[0].device)

        for i, (key, s_aligned, t_feat, w) in enumerate(
            zip(self.feature_keys, aligned, t_feats, self.loss_weights)
        ):
            loss = self.loss_fn(s_aligned, t_feat.detach())
            losses[f"feat_loss_{key}"] = loss
            total = total + w * loss

        losses["feat_loss_total"] = total
        return losses
