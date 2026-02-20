"""Training script for multi-task model distillation."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import yaml

# Ensure all components are registered via imports
import distillation.backbones  # noqa: F401
import distillation.heads  # noqa: F401
import distillation.losses  # noqa: F401
from distillation.models.multi_task_model import MultiTaskModel
from distillation.distillers.multi_task_distiller import MultiTaskDistiller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy dataset for demonstration / testing
# ---------------------------------------------------------------------------

class DummyMultiTaskDataset(Dataset):
    """Generates random images with segmentation and detection targets.

    Replace this with your actual dataset implementation.
    """

    def __init__(self, num_samples=100, image_size=(512, 512),
                 num_seg_classes=19):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_seg_classes = num_seg_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, *self.image_size)
        seg_target = torch.randint(0, self.num_seg_classes, self.image_size)
        return {"image": image, "seg": seg_target}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> MultiTaskModel:
    """Build a MultiTaskModel from a config dict."""
    return MultiTaskModel(
        backbone_name=cfg["backbone"]["name"],
        backbone_cfg={
            k: v for k, v in cfg["backbone"].items() if k != "name"
        },
        heads_cfg=cfg.get("heads", {}),
    )


def build_distiller(cfg: dict) -> MultiTaskDistiller:
    """Build the full distillation pipeline from config."""
    teacher = build_model(cfg["teacher"])
    student = build_model(cfg["student"])

    # Load teacher checkpoint if provided
    ckpt_path = cfg["teacher"].get("checkpoint")
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        teacher.load_state_dict(state, strict=False)
        logger.info(f"Loaded teacher checkpoint from {ckpt_path}")

    distiller = MultiTaskDistiller(
        teacher=teacher,
        student=student,
        feature_distill_cfg=cfg.get("feature_distill"),
        logit_distill_cfg=cfg.get("logit_distill"),
        cross_head_cfg=cfg.get("cross_head"),
        task_loss_cfg=cfg.get("task_loss"),
        freeze_teacher=True,
    )
    return distiller


# ---------------------------------------------------------------------------
# LR scheduler builder
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_type = cfg["training"].get("lr_scheduler", "cosine")
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"].get("warmup_epochs", 0)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch

    if sched_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    elif sched_type == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30 * steps_per_epoch, gamma=0.1
        )
    else:
        return None

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    return main_scheduler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build distiller
    distiller = build_distiller(cfg)
    distiller.to(device)

    # Build dataset & dataloader
    data_cfg = cfg.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))
    num_workers = data_cfg.get("num_workers", 4)

    dataset = DummyMultiTaskDataset(
        num_samples=200, image_size=image_size,
        num_seg_classes=cfg["student"]["heads"]["seg"]["num_classes"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer (only student + aligner parameters)
    trainable_params = [
        p for p in distiller.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=cfg["training"]["lr"],
        momentum=0.9,
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = build_scheduler(optimizer, cfg, len(dataloader))

    # Active heads
    active_heads = cfg["training"].get("active_heads")
    if active_heads:
        active_heads = set(active_heads)

    epochs = cfg["training"]["epochs"]
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Active heads: {active_heads or 'all'}")
    logger.info(f"Student backbone: {cfg['student']['backbone']['name']}")
    logger.info(f"Teacher backbone: {cfg['teacher']['backbone']['name']}")

    for epoch in range(epochs):
        distiller.student.train()
        epoch_losses = {}

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            targets = {}
            if "seg" in batch:
                targets["seg"] = batch["seg"].to(device)

            optimizer.zero_grad()
            losses = distiller(images, targets=targets, active_heads=active_heads)
            total_loss = losses["total_loss"]
            total_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Accumulate losses for logging
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()

        # Log epoch summary
        n_steps = len(dataloader)
        loss_strs = [f"{k}: {v / n_steps:.4f}" for k, v in epoch_losses.items()]
        logger.info(f"Epoch [{epoch+1}/{epochs}] {' | '.join(loss_strs)}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            ckpt = {
                "epoch": epoch + 1,
                "student_state_dict": distiller.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if distiller.feature_distiller is not None:
                ckpt["aligner_state_dict"] = (
                    distiller.feature_distiller.aligner.state_dict()
                )
            save_path = Path("checkpoints") / f"student_epoch_{epoch+1}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Head swap demo
# ---------------------------------------------------------------------------

def demo_head_swap(cfg: dict):
    """Demonstrate head swapping and cross-prediction capabilities."""
    device = torch.device("cpu")

    teacher = build_model(cfg["teacher"]).to(device)
    student = build_model(cfg["student"]).to(device)

    logger.info("=== Head Swap Demo ===")
    logger.info(f"Teacher backbone: {cfg['teacher']['backbone']['name']} "
                f"channels={teacher.backbone.out_channels}")
    logger.info(f"Student backbone: {cfg['student']['backbone']['name']} "
                f"channels={student.backbone.out_channels}")
    logger.info(f"Teacher heads: {teacher.head_names}")
    logger.info(f"Student heads: {student.head_names}")

    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # 1. Normal forward - each model uses its own heads
    t_out = teacher(dummy_input, active_heads={"seg"})
    s_out = student(dummy_input, active_heads={"seg"})
    logger.info(
        f"[Normal] Teacher seg: {t_out['seg']['seg_logits'].shape}, "
        f"Student seg: {s_out['seg']['seg_logits'].shape}"
    )

    # 2. Selective head execution
    s_det = student(dummy_input, active_heads={"det"})
    logger.info(f"[Selective] Student det levels: {len(s_det['det']['det_cls_logits'])}")

    s_all = student(dummy_input)
    logger.info(f"[All heads] Student output keys: {list(s_all.keys())}")

    # 3. Cross-prediction via MultiTaskDistiller
    # The distiller handles channel adaptation automatically
    logger.info("\n=== Cross-Head Distillation Demo ===")
    distiller = MultiTaskDistiller(
        teacher=teacher,
        student=student,
        feature_distill_cfg={
            "feature_keys": ["s3", "s4", "s5"],
            "aligner_type": "conv1x1",
            "loss_type": "mse_feature_loss",
            "weight": 1.0,
        },
        logit_distill_cfg={
            "seg": {"loss_type": "seg_kd_loss", "weight": 2.0, "temperature": 4.0},
        },
        cross_head_cfg={
            "seg": {"loss_type": "mse_feature_loss", "weight": 0.5},
        },
        task_loss_cfg={
            "seg": {"loss_type": "seg_ce_loss", "weight": 1.0},
        },
    ).to(device)

    seg_target = torch.randint(0, 19, (1, 256, 256))
    losses = distiller(dummy_input, targets={"seg": seg_target}, active_heads={"seg"})
    logger.info("Distillation losses:")
    for k, v in losses.items():
        logger.info(f"  {k}: {v.item():.4f}")

    # 4. Same-backbone head swap (direct)
    logger.info("\n=== Same-Backbone Head Swap ===")
    teacher2 = build_model(cfg["teacher"]).to(device)
    teacher2.attach_head_from(teacher, "seg", alias="seg_v2")
    logger.info(f"Teacher2 heads after attach: {teacher2.head_names}")
    out = teacher2(dummy_input, active_heads={"seg_v2"})
    logger.info(f"seg_v2 output: {out['seg_v2']['seg_logits'].shape}")
    teacher2.remove_head("seg_v2")

    logger.info("\nDemo complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Distillation Training")
    parser.add_argument(
        "--config", type=str,
        default="distillation/configs/default_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "demo_swap"],
        help="Run mode: 'train' for distillation, 'demo_swap' for head-swap demo.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "demo_swap":
        demo_head_swap(cfg)


if __name__ == "__main__":
    main()
