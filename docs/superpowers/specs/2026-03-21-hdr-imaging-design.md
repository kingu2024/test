# HDR 高动态范围成像算法库 — 设计规格

## 概述

为现有计算机视觉算法库新增 HDR（High Dynamic Range）成像模块，覆盖完整 HDR 流程：多曝光图像对齐、相机响应函数标定、HDR 辐射图重建、色调映射（10 种算法）、多曝光融合、单张图像 HDR 增强。

采用混合实现策略：核心算法手动 NumPy 实现（含详细中文数学注释），同时提供 OpenCV 内置版本作为参考对比。

**关于子包结构的说明：** 现有模块（panorama_stitching、video_stabilization）使用扁平文件布局。HDR 模块由于算法数量多（仅色调映射就有 10 种），采用子包结构以保持每个文件聚焦。这是经过评估后的有意选择，不影响外部 API 的使用方式。

## 模块结构

```
hdr_imaging/
├── __init__.py                    # 包导出，统一公开 API
├── alignment/                     # 图像对齐
│   ├── __init__.py
│   ├── mtb_alignment.py           # MTB 中值阈值位图对齐 (Ward 2003)
│   └── feature_alignment.py       # 特征点对齐（复用 panorama_stitching 模块）
├── calibration/                   # 相机响应函数标定
│   ├── __init__.py
│   ├── debevec.py                 # Debevec & Malik (1997) 线性求解
│   └── robertson.py               # Robertson et al. (2003) 迭代 MLE
├── merge/                         # HDR 辐射图合并
│   ├── __init__.py
│   └── hdr_merge.py               # 加权合并多曝光图像为 HDR 辐射图
├── tone_mapping/                  # 色调映射（按类型分 3 个文件）
│   ├── __init__.py
│   ├── global_operators.py        # Reinhard 全局、Drago、自适应对数映射
│   ├── local_operators.py         # Reinhard 局部、Durand 双边滤波、Fattal 梯度域
│   └── perceptual_operators.py    # ACES、Filmic (Uncharted 2)、Mantiuk、直方图调整
├── exposure_fusion/               # 多曝光融合（不经过 HDR 辐射图）
│   ├── __init__.py
│   └── mertens.py                 # Mertens 曝光融合算法
├── single_image/                  # 单张图像 HDR
│   ├── __init__.py
│   └── single_image_hdr.py        # CLAHE + 局部色调映射 + 自适应增强
└── hdr_pipeline.py                # 主控流水线，串联各子模块
```

顶层新增文件：
- `demo_hdr.py` — HDR 演示脚本
- `README_HDR.md` — HDR 模块完整中文说明文档
- `hdr_imaging/docs/hdr_math_principles.md` — 数学原理文档（与 calibration_algorithms/docs/ 结构一致）

### __init__.py 导出规范

根包 `hdr_imaging/__init__.py` 遵循现有模块模式：
```python
"""
HDR 高动态范围成像算法库 / HDR Imaging Algorithm Library
"""
from .hdr_pipeline import HDRPipeline
from .alignment import MTBAlignment, FeatureAlignment
from .calibration import DebevecCalibration, RobertsonCalibration
from .merge import HDRMerge
from .tone_mapping import (ReinhardGlobal, ReinhardLocal, DragoToneMap,
                           DurandToneMap, FattalToneMap, AdaptiveLog,
                           ACESToneMap, FilmicToneMap, MantiukToneMap,
                           HistogramToneMap)
from .exposure_fusion import MertensFusion
from .single_image import SingleImageHDR

__all__ = ['HDRPipeline', 'MTBAlignment', 'FeatureAlignment',
           'DebevecCalibration', 'RobertsonCalibration', 'HDRMerge',
           'ReinhardGlobal', 'ReinhardLocal', 'DragoToneMap',
           'DurandToneMap', 'FattalToneMap', 'AdaptiveLog',
           'ACESToneMap', 'FilmicToneMap', 'MantiukToneMap',
           'HistogramToneMap', 'MertensFusion', 'SingleImageHDR']
__version__ = '1.0.0'
__author__ = 'Computer Vision Algorithm Library'
```

各子包 `__init__.py` 导出该子包的核心类，例如 `alignment/__init__.py`：
```python
from .mtb_alignment import MTBAlignment
from .feature_alignment import FeatureAlignment
```

## 算法清单

### 1. 图像对齐 (alignment/)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `MTBAlignment` | Ward MTB (2003) | 图像→中值阈值位图→金字塔逐层位移搜索，对曝光变化不敏感 |
| `FeatureAlignment` | SIFT/ORB + RANSAC | 复用 panorama_stitching 模块，提取特征→匹配→估计单应性→变换对齐 |

MTB 为默认对齐方法（速度快），特征点对齐为高精度备选。

### 2. 相机响应函数标定 (calibration/)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `DebevecCalibration` | Debevec & Malik (1997) | 构建过约束线性系统 SVD 求解，恢复 CRF: g(Z) = ln(E) + ln(Δt) |
| `RobertsonCalibration` | Robertson (2003) | EM 迭代：E-step 估计辐照度，M-step 更新响应函数，最大似然收敛 |

### 3. HDR 辐射图合并 (merge/)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `HDRMerge` | 加权辐照度合并 | E(x,y) = Σ w(Z)·(g(Z) - ln(Δt)) / Σ w(Z)，权重函数抑制过曝/欠曝像素 |

### 4. 色调映射 (tone_mapping/)

#### 4.1 全局算子 (global_operators.py)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `ReinhardGlobal` | Reinhard (2002) 全局 | 对数域亮度缩放 + 白点映射 |
| `DragoToneMap` | Drago (2003) | 自适应对数基底映射 |
| `AdaptiveLog` | 自适应对数映射 | 场景自适应的对数压缩 |

#### 4.2 局部算子 (local_operators.py)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `ReinhardLocal` | Reinhard (2002) 局部 | 多尺度高斯中心-环绕自适应 |
| `DurandToneMap` | Durand & Dorsey (2002) | 双边滤波分离基础/细节层 |
| `FattalToneMap` | Fattal (2002) | 梯度域衰减 + 泊松重建（使用 scipy.sparse.linalg.spsolve，大图像自动降采样） |

#### 4.3 感知驱动算子 (perceptual_operators.py)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `ACESToneMap` | ACES 电影曲线 | 影视工业标准 S 曲线 |
| `FilmicToneMap` | Uncharted 2 Filmic | 游戏行业常用分段曲线 |
| `MantiukToneMap` | Mantiuk (2006) | 基于人眼对比度感知模型 |
| `HistogramToneMap` | 直方图均衡化映射 | 累积直方图重分布亮度 |

### 5. 多曝光融合 (exposure_fusion/)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `MertensFusion` | Mertens (2007) | 计算对比度/饱和度/曝光度权重→拉普拉斯金字塔多分辨率融合 |

不经过 HDR 辐射图，直接从多曝光 LDR 图像融合。

### 6. 单张图像 HDR (single_image/)

| 类 | 算法 | 核心原理 |
|---|---|---|
| `SingleImageHDR` | CLAHE + 局部增强 | 亮度通道 CLAHE 自适应均衡 + 多尺度细节增强 + 色彩保持映射 |

## 数据流

### 完整 HDR 流水线

```
输入多曝光图像 + 曝光时间
        │
        ▼
   ┌─────────────┐
   │  Alignment   │  MTB（默认）或 特征点对齐
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Calibration  │  Debevec / Robertson → 相机响应曲线 g(Z)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   Merge      │  加权合并 → HDR 辐射图 (float32)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Tone Mapping │  10 种算法可选 → LDR 输出图像
   └─────────────┘
```

### 替代路径

- **多曝光融合路径**：输入图像 → Alignment → MertensFusion → LDR 输出（跳过标定/合并/色调映射）
- **单张 HDR 路径**：单张 LDR 输入 → SingleImageHDR → 增强输出

## 接口设计

### 主控类 HDRPipeline

```python
class HDRPipeline:
    def __init__(self, align_method='mtb', calibration_method='debevec',
                 tone_mapping_method='reinhard_global', use_opencv=False):
        """
        初始化 HDR 流水线
        Args:
            align_method: 'mtb' 或 'feature'
            calibration_method: 'debevec' 或 'robertson'
            tone_mapping_method: 算法名称字符串
            use_opencv: True 时使用 OpenCV 内置实现
        """

    def process(self, images, exposure_times):
        """
        完整 HDR 流程：对齐→标定→合并→色调映射
        Args:
            images: list of numpy arrays (BGR, uint8)
            exposure_times: list of float (秒)
        Returns:
            HDRResult (NamedTuple):
                ldr_result: 色调映射后的 LDR 图像 (uint8)
                hdr_radiance_map: HDR 辐射图 (float32)
                response_curve: 相机响应曲线 (numpy array)
        """

    def exposure_fusion(self, images):
        """多曝光融合（不经过 HDR 辐射图）"""

    def single_image_hdr(self, image):
        """单张图像 HDR 增强"""
```

### 返回值类型

```python
from collections import namedtuple

HDRResult = namedtuple('HDRResult', ['ldr_result', 'hdr_radiance_map', 'response_curve'])
```

### 各算法类统一接口模式

```python
class XxxAlgorithm:
    def __init__(self, **params):
        """参数初始化，含默认值"""

    def process(self, *args):
        """手动 NumPy 实现（默认），含详细中文注释"""

    def process_opencv(self, *args):
        """OpenCV 内置实现作为参考对比（如有对应方法）。
        无 OpenCV 对应实现的算法（ACES、Filmic、Fattal、AdaptiveLog、HistogramToneMap）
        调用 process_opencv() 时抛出 NotImplementedError 并提示使用 process()。"""
```

### 色调映射方法名称映射

`HDRPipeline` 的 `tone_mapping_method` 参数接受以下字符串值：

| 字符串值 | 对应类 |
|---|---|
| `'reinhard_global'` | `ReinhardGlobal` |
| `'reinhard_local'` | `ReinhardLocal` |
| `'drago'` | `DragoToneMap` |
| `'durand'` | `DurandToneMap` |
| `'fattal'` | `FattalToneMap` |
| `'adaptive_log'` | `AdaptiveLog` |
| `'aces'` | `ACESToneMap` |
| `'filmic'` | `FilmicToneMap` |
| `'mantiuk'` | `MantiukToneMap` |
| `'histogram'` | `HistogramToneMap` |

## 实现策略

### 混合实现

- `process()` — 手动 NumPy 实现，从数学公式出发，每一步附中文注释说明数学运算含义
- `process_opencv()` — 调用 OpenCV 对应 API（`cv2.createMergeDebevec()`、`cv2.createTonemapReinhard()` 等），作为结果对比参考
- 不是所有算法在 OpenCV 中都有对应实现（ACES、Filmic、Fattal、AdaptiveLog、HistogramToneMap），这些仅提供手动实现，`process_opencv()` 抛出 `NotImplementedError`

### 跨模块复用

- `feature_alignment.py` 通过 import 复用 `panorama_stitching.feature_extraction.FeatureExtractor` 和 `panorama_stitching.homography.HomographyEstimator`
- 其余所有模块无跨包依赖

## Demo 脚本 (demo_hdr.py)

### 运行模式

```bash
# 模式1：合成数据演示（默认）
python demo_hdr.py --test

# 模式2：用户自定义多曝光输入
python demo_hdr.py --images img1.jpg img2.jpg img3.jpg --exposures 0.033 0.25 1.0

# 模式3：单张图像 HDR
python demo_hdr.py --single input.jpg

# 模式4：色调映射算法对比（生成所有算法的并排对比网格图）
python demo_hdr.py --compare
```

### 合成数据生成

- 程序生成虚拟 HDR 场景（高亮区域如灯光 + 暗部区域如阴影），动态范围跨越 4-5 个数量级
- 模拟不同曝光时间（1/30s、1/4s、1s、4s）生成多张 LDR 图像
- 添加适量高斯噪声模拟真实传感器
- 可选对图像施加微小随机位移测试对齐模块

### 输出内容

保存到 `output_hdr/` 目录：
- 各曝光输入图像对比
- 对齐前后对比（如有位移）
- 恢复的相机响应曲线（Debevec vs Robertson）
- HDR 辐射图伪彩色可视化
- 10 种色调映射算法结果并排对比
- Mertens 曝光融合结果
- 单张图像 HDR 增强前后对比
- 手动实现 vs OpenCV 实现结果对比

使用 Matplotlib `Agg` 后端，与现有 demo 一致。

## 文档 (README_HDR.md)

中文文档，与现有 `README_algorithms.md` 和 `calibration_math_principles.md` 风格一致，包含：

1. **项目简介** — HDR 成像概述
2. **目录结构** — 完整文件树
3. **算法原理详解** — 每个子模块一个章节：
   - 图像对齐（MTB + 特征点）
   - 相机响应函数标定（Debevec + Robertson）
   - HDR 辐射图合并
   - 色调映射（全局/局部/感知驱动，10 种算法各有公式推导）
   - 多曝光融合（Mertens）
   - 单张图像 HDR
4. **算法对比总结表** — 各色调映射算法适用场景、优缺点
5. **快速开始** — 安装、运行、自定义输入
6. **参数调优指南** — 关键参数说明与推荐值
7. **参考文献** — 原始论文引用

每个算法小节包含：数学公式、关键步骤说明、适用场景。

## 依赖

无需新增依赖。现有 `requirements.txt` 已覆盖：
- `opencv-python>=4.8.0`
- `opencv-contrib-python>=4.8.0`
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `matplotlib>=3.7.0`

## 错误处理

| 失败场景 | 异常类型 | 行为 |
|---|---|---|
| 输入图像少于 2 张（HDR 流程） | `ValueError` | 提示至少需要 2 张不同曝光图像 |
| 曝光时间数量与图像数量不匹配 | `ValueError` | 提示数量必须一致 |
| 图像尺寸不一致 | 对齐步骤自动处理 | 以第一张图像尺寸为基准裁剪/缩放 |
| `exposure_fusion()` 传入单张图像 | `ValueError` | 提示至少需要 2 张图像 |
| `tone_mapping_method` 字符串无效 | `ValueError` | 列出所有可用方法名称 |
| `process_opencv()` 无 OpenCV 对应实现 | `NotImplementedError` | 提示该算法仅支持手动实现，请使用 `process()` |
| Debevec SVD 求解失败（曝光差异不足） | `RuntimeError` | 提示输入图像曝光差异不足，建议增加曝光范围 |
| HDR 合并时某像素位置所有曝光均饱和 | 退化处理 | 使用最接近中间曝光的像素值，记录 warning 日志 |
| Fattal 泊松求解超出内存（大图像） | 自动降采样 | 图像尺寸超过 2048x2048 时自动 0.5x 降采样求解后上采样 |

通用规则：
- 使用 `logging` 模块输出各步骤进度，与现有模块一致
- 所有异常消息使用英文（与现有模块一致），日志中文注释

## .gitignore 更新

在现有 `.gitignore` 中添加 `output_hdr/` 目录。

## 文档关联

在现有 `README_algorithms.md` 末尾添加指向 `README_HDR.md` 的链接，保持文档入口统一。
