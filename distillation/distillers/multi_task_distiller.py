"""Multi-task distiller: orchestrates feature + logit distillation with head swapping."""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from distillation import ALIGNERS, DISTILLERS, LOSSES
from distillation.distillers.feature_distiller import FeatureDistiller
from distillation.losses.feature_loss import FeatureAlignmentModule
from distillation.models.multi_task_model import MultiTaskModel


@DISTILLERS.register("multi_task_distiller")
class MultiTaskDistiller(nn.Module):
    """Orchestrates knowledge distillation between teacher and student models.

    Supports:
      1. Feature-level distillation at configurable backbone layers.
      2. Logit-level distillation per task head.
      3. Cross-head prediction: student backbone + teacher head (and vice versa).
         Automatically builds channel adapters when backbones differ.
      4. Flexible loss weighting.

    Args:
        teacher: Teacher MultiTaskModel (frozen during distillation).
        student: Student MultiTaskModel (trained).
        feature_distill_cfg: Config for FeatureDistiller. None = skip feature KD.
        logit_distill_cfg: Per-head logit distillation config.
            Dict of {head_name: {"loss_type": ..., "weight": ...}}.
        cross_head_cfg: Config for cross-head prediction distillation.
            Dict of {head_name: {"weight": ..., "aligner_type": ...}}.
        task_loss_cfg: Per-head task loss config (for supervised training).
            Dict of {head_name: {"loss_type": ..., "weight": ...}}.
        freeze_teacher: Whether to freeze teacher weights (default True).
    """

    def __init__(
        self,
        teacher: MultiTaskModel,
        student: MultiTaskModel,
        feature_distill_cfg: Optional[dict] = None,
        logit_distill_cfg: Optional[Dict[str, dict]] = None,
        cross_head_cfg: Optional[Dict[str, dict]] = None,
        task_loss_cfg: Optional[Dict[str, dict]] = None,
        freeze_teacher: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        # --- Feature distillation ---
        self.feature_distiller: Optional[FeatureDistiller] = None
        self.feat_weight = 1.0
        if feature_distill_cfg:
            cfg = dict(feature_distill_cfg)
            self.feat_weight = cfg.pop("weight", 1.0)

            keys = cfg.get("feature_keys", ["s2", "s3", "s4", "s5"])
            key_to_idx = {k: i for i, k in enumerate(student.backbone.feature_keys)}
            s_channels = [student.backbone.out_channels[key_to_idx[k]] for k in keys]
            t_channels = [teacher.backbone.out_channels[key_to_idx[k]] for k in keys]

            self.feature_distiller = FeatureDistiller(
                student_channels=s_channels,
                teacher_channels=t_channels,
                **cfg,
            )

        # --- Logit distillation per head ---
        self.logit_losses: nn.ModuleDict = nn.ModuleDict()
        self.logit_weights: Dict[str, float] = {}
        if logit_distill_cfg:
            for head_name, cfg in logit_distill_cfg.items():
                cfg = dict(cfg)
                loss_type = cfg.pop("loss_type", "seg_kd_loss")
                self.logit_weights[head_name] = cfg.pop("weight", 1.0)
                self.logit_losses[head_name] = LOSSES.build(loss_type, **cfg)

        # --- Cross-head prediction ---
        self.cross_head_weights: Dict[str, float] = {}
        self.cross_head_losses: nn.ModuleDict = nn.ModuleDict()
        self._cross_head_adapters: nn.ModuleDict = nn.ModuleDict()

        if cross_head_cfg:
            s_ch = student.backbone.out_channels
            t_ch = teacher.backbone.out_channels
            need_adapter = s_ch != t_ch

            for head_name, cfg in cross_head_cfg.items():
                cfg = dict(cfg)
                loss_type = cfg.pop("loss_type", "mse_feature_loss")
                self.cross_head_weights[head_name] = cfg.pop("weight", 1.0)
                aligner_type = cfg.pop("aligner_type", "conv1x1")
                self.cross_head_losses[head_name] = LOSSES.build(loss_type, **cfg)

                # Build per-level channel adapter: student_ch -> teacher_ch
                if need_adapter:
                    self._cross_head_adapters[head_name] = FeatureAlignmentModule(
                        student_channels=s_ch,
                        teacher_channels=t_ch,
                        aligner_type=aligner_type,
                    )

        # --- Task losses (supervised) ---
        self.task_losses: nn.ModuleDict = nn.ModuleDict()
        self.task_weights: Dict[str, float] = {}
        if task_loss_cfg:
            for head_name, cfg in task_loss_cfg.items():
                cfg = dict(cfg)
                loss_type = cfg.pop("loss_type")
                self.task_weights[head_name] = cfg.pop("weight", 1.0)
                self.task_losses[head_name] = LOSSES.build(loss_type, **cfg)

    def _adapt_features_for_cross_head(
        self,
        head_name: str,
        student_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Adapt student backbone features to teacher channel dimensions.

        If student and teacher share the same backbone channels, returns
        features as-is. Otherwise, applies a learned channel adapter.
        """
        if head_name not in self._cross_head_adapters:
            return student_features

        adapter = self._cross_head_adapters[head_name]
        keys = list(student_features.keys())
        feat_list = [student_features[k] for k in keys]
        adapted = adapter(feat_list)

        # Re-pack into dict
        return {k: v for k, v in zip(keys, adapted)}

    @torch.no_grad()
    def _teacher_forward(
        self, x: torch.Tensor, active_heads: Optional[Set[str]] = None
    ) -> Dict[str, dict]:
        """Run teacher in no-grad mode."""
        was_training = self.teacher.training
        self.teacher.eval()
        out = self.teacher(x, active_heads=active_heads)
        if was_training:
            self.teacher.train()
        return out

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        active_heads: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run distillation forward pass.

        Args:
            images: Input images [B, C, H, W].
            targets: Dict of task targets, e.g.:
                {"seg": seg_labels, "det_cls": det_cls_targets, ...}.
            active_heads: Which heads to distill. None = all shared heads.

        Returns:
            Dict of all loss components and total loss.
        """
        target_size = images.shape[2:]
        targets = targets or {}

        # Determine which heads to run
        if active_heads is None:
            student_heads = set(self.student.head_names)
            teacher_heads = set(self.teacher.head_names)
            active_heads = student_heads & teacher_heads

        # Forward passes
        teacher_out = self._teacher_forward(images, active_heads)
        student_out = self.student(images, active_heads, target_size=target_size)

        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=images.device)

        # --- 1. Feature distillation ---
        if self.feature_distiller is not None:
            feat_losses = self.feature_distiller(
                student_out["features"], teacher_out["features"]
            )
            for k, v in feat_losses.items():
                losses[k] = v
            total_loss = total_loss + self.feat_weight * feat_losses["feat_loss_total"]

        # --- 2. Logit distillation per head ---
        for head_name in active_heads:
            if head_name not in self.logit_losses:
                continue

            s_head_out = student_out[head_name]
            t_head_out = teacher_out[head_name]

            loss_fn = self.logit_losses[head_name]
            w = self.logit_weights[head_name]

            # Segmentation head
            if "seg_logits" in s_head_out:
                kd_loss = loss_fn(s_head_out["seg_logits"], t_head_out["seg_logits"])
                losses[f"logit_kd_{head_name}"] = kd_loss
                total_loss = total_loss + w * kd_loss

            # Detection head
            if "det_cls_logits" in s_head_out:
                kd_loss = loss_fn(
                    student_cls_list=s_head_out["det_cls_logits"],
                    teacher_cls_list=t_head_out["det_cls_logits"],
                    student_reg_list=s_head_out.get("det_reg_preds"),
                    teacher_reg_list=t_head_out.get("det_reg_preds"),
                )
                losses[f"logit_kd_{head_name}"] = kd_loss
                total_loss = total_loss + w * kd_loss

        # --- 3. Cross-head prediction ---
        # Feed student features (adapted to teacher channels) into the teacher
        # head, then compare the resulting intermediate features with the
        # student head's intermediate features.
        for head_name in active_heads:
            if head_name not in self.cross_head_losses:
                continue

            # Adapt student features to teacher channel dimensions
            adapted_features = self._adapt_features_for_cross_head(
                head_name, student_out["features"]
            )

            # Run teacher head on (adapted) student features
            teacher_head = self.teacher.get_head(head_name)
            with torch.no_grad():
                t_on_s_out = teacher_head(adapted_features)

            # Compare with student head output
            s_head_out = student_out[head_name]
            loss_fn = self.cross_head_losses[head_name]
            w = self.cross_head_weights[head_name]

            if "seg_feat" in s_head_out and "seg_feat" in t_on_s_out:
                cross_loss = loss_fn(
                    s_head_out["seg_feat"], t_on_s_out["seg_feat"]
                )
                losses[f"cross_head_{head_name}"] = cross_loss
                total_loss = total_loss + w * cross_loss
            elif "det_cls_feats" in s_head_out and "det_cls_feats" in t_on_s_out:
                cross_loss = torch.tensor(0.0, device=images.device)
                for s_f, t_f in zip(
                    s_head_out["det_cls_feats"], t_on_s_out["det_cls_feats"]
                ):
                    cross_loss = cross_loss + loss_fn(s_f, t_f)
                cross_loss = cross_loss / len(s_head_out["det_cls_feats"])
                losses[f"cross_head_{head_name}"] = cross_loss
                total_loss = total_loss + w * cross_loss

        # --- 4. Task losses (supervised) ---
        for head_name, loss_fn in self.task_losses.items():
            if head_name not in active_heads:
                continue
            w = self.task_weights[head_name]
            s_head_out = student_out[head_name]

            if "seg_logits" in s_head_out and "seg" in targets:
                task_loss = loss_fn(s_head_out["seg_logits"], targets["seg"])
                losses[f"task_{head_name}"] = task_loss
                total_loss = total_loss + w * task_loss

        losses["total_loss"] = total_loss
        return losses

    def get_student_predictions(
        self,
        images: torch.Tensor,
        active_heads: Optional[Set[str]] = None,
        target_size: Optional[tuple] = None,
    ) -> Dict[str, dict]:
        """Run student inference only (no distillation)."""
        return self.student(images, active_heads, target_size=target_size)
