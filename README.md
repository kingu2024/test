# jinyan add 20260221
# Multi-Task Model Distillation Framework

灵活的多任务模型蒸馏框架，支持语义分割和目标检测模型的知识蒸馏，可自由切换预测头并进行局部特征层对齐。

## 核心特性

- **多任务支持**: 同时蒸馏语义分割 (FPN / ASPP head) 和目标检测 (FCOS / SSD head)
- **自由切换头**: 支持 teacher/student 之间的 head 交换，实现交叉预测
- **局部特征对齐**: 可选择性地对齐指定层的特征图（支持 conv1x1 / conv3x3 / MLP 三种 aligner）
- **多级蒸馏**: 同时支持特征级蒸馏和 logit 级蒸馏
- **组件注册机制**: 所有 backbone、head、loss 均通过 Registry 注册，方便扩展

## 项目结构

```
distillation/
├── backbones/          # Backbone 网络 (ResNet18/34/50/101, MobileNetV2)
├── heads/              # 任务头 (FPN Seg, ASPP Seg, FCOS Det, SSD Det)
├── models/             # 多任务模型封装 (支持 head 切换)
├── distillers/         # 蒸馏器 (特征蒸馏 + 多任务蒸馏)
├── losses/             # 损失函数 (特征损失, KD 损失, 任务损失)
├── utils/              # 工具 (Registry)
├── configs/            # YAML 配置文件
└── train.py            # 训练入口
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 训练 (使用默认配置)

```bash
python -m distillation.train --config distillation/configs/default_config.yaml --mode train
```

### Head 交换演示

```bash
python -m distillation.train --config distillation/configs/default_config.yaml --mode demo_swap
```

## 架构设计

### 1. MultiTaskModel

将 backbone 与多个任务头组合，支持：

```python
from distillation.models import MultiTaskModel

model = MultiTaskModel(
    backbone_name="resnet50",
    heads_cfg={
        "seg": {"type": "fpn_seg_head", "num_classes": 19},
        "det": {"type": "anchor_free_det_head", "num_classes": 80,
                "input_keys": ["s3", "s4", "s5"]},
    },
)

# 选择性运行某些 head
out = model(images, active_heads={"seg"})

# 从另一个模型借用 head（交叉预测）
student.attach_head_from(teacher, "seg", alias="teacher_seg")
cross_out = student(images, active_heads={"teacher_seg"})
```

### 2. FeatureDistiller - 局部特征对齐

灵活选择要对齐的特征层和对齐方式：

```python
from distillation.distillers import FeatureDistiller

distiller = FeatureDistiller(
    student_channels=[128, 256, 512],
    teacher_channels=[512, 1024, 2048],
    feature_keys=["s3", "s4", "s5"],   # 选择对齐层
    aligner_type="conv1x1",            # 对齐方式
    loss_type="mse_feature_loss",      # 损失类型
    loss_weights=[1.0, 1.0, 2.0],      # 每层权重
)
```

### 3. MultiTaskDistiller - 完整蒸馏流程

整合特征蒸馏、logit 蒸馏、交叉预测和任务损失：

```python
from distillation.distillers import MultiTaskDistiller

distiller = MultiTaskDistiller(
    teacher=teacher_model,
    student=student_model,
    feature_distill_cfg={
        "feature_keys": ["s3", "s4", "s5"],
        "aligner_type": "conv1x1",
        "loss_type": "mse_feature_loss",
        "weight": 1.0,
    },
    logit_distill_cfg={
        "seg": {"loss_type": "seg_kd_loss", "weight": 2.0, "temperature": 4.0},
        "det": {"loss_type": "det_kd_loss", "weight": 2.0},
    },
    cross_head_cfg={
        "seg": {"loss_type": "mse_feature_loss", "weight": 0.5},
    },
    task_loss_cfg={
        "seg": {"loss_type": "seg_ce_loss", "weight": 1.0},
    },
)

# 训练一步
losses = distiller(images, targets={"seg": seg_labels})
losses["total_loss"].backward()
```

## 可用组件

| 类型 | 名称 | 说明 |
|------|------|------|
| Backbone | `resnet18/34/50/101` | ResNet 系列 |
| Backbone | `mobilenetv2` | MobileNetV2 (轻量 student) |
| Seg Head | `fpn_seg_head` | FPN 多尺度分割头 |
| Seg Head | `aspp_seg_head` | ASPP 空洞卷积分割头 |
| Det Head | `anchor_free_det_head` | FCOS 风格无锚框检测头 |
| Det Head | `ssd_det_head` | SSD 风格多框检测头 |
| Aligner | `conv1x1` / `conv3x3` / `mlp` | 特征通道对齐模块 |
| Loss | `mse_feature_loss` | L2 特征蒸馏损失 |
| Loss | `cosine_feature_loss` | 余弦相似度特征损失 |
| Loss | `attention_feature_loss` | 注意力迁移损失 |
| Loss | `channel_wise_feature_loss` | 通道统计量对齐损失 |
| Loss | `seg_kd_loss` | 像素级 KD 损失 |
| Loss | `det_kd_loss` | 多层级检测 KD 损失 |

## 扩展新组件

通过 Registry 注册即可：

```python
from distillation import BACKBONES

@BACKBONES.register("my_backbone")
class MyBackbone(nn.Module):
    ...
```
