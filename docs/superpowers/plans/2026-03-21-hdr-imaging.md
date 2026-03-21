# HDR Imaging Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a comprehensive HDR imaging module with 10 tone mapping algorithms, multi-exposure fusion, CRF calibration, image alignment, and single-image HDR enhancement.

**Architecture:** Sub-package structure under `hdr_imaging/` with 6 sub-packages (alignment, calibration, merge, tone_mapping, exposure_fusion, single_image) plus a main pipeline controller. Each algorithm class provides `process()` (hand-written NumPy) and `process_opencv()` (OpenCV reference) methods. A `HDRPipeline` class orchestrates the full flow.

**Tech Stack:** Python 3, NumPy, OpenCV (opencv-python + opencv-contrib-python), SciPy, Matplotlib

**Spec:** `docs/superpowers/specs/2026-03-21-hdr-imaging-design.md`

---

## File Structure

### New Files to Create

| File | Responsibility |
|------|---------------|
| `hdr_imaging/__init__.py` | Root package exports, `__all__`, version |
| `hdr_imaging/alignment/__init__.py` | Sub-package exports for alignment |
| `hdr_imaging/alignment/mtb_alignment.py` | MTB (Median Threshold Bitmap) alignment - Ward 2003 |
| `hdr_imaging/alignment/feature_alignment.py` | Feature-based alignment reusing panorama_stitching |
| `hdr_imaging/calibration/__init__.py` | Sub-package exports for calibration |
| `hdr_imaging/calibration/debevec.py` | Debevec & Malik 1997 CRF recovery |
| `hdr_imaging/calibration/robertson.py` | Robertson 2003 iterative MLE CRF recovery |
| `hdr_imaging/merge/__init__.py` | Sub-package exports for merge |
| `hdr_imaging/merge/hdr_merge.py` | Weighted irradiance merge to HDR radiance map |
| `hdr_imaging/tone_mapping/__init__.py` | Sub-package exports for all 10 tone mappers |
| `hdr_imaging/tone_mapping/global_operators.py` | ReinhardGlobal, DragoToneMap, AdaptiveLog |
| `hdr_imaging/tone_mapping/local_operators.py` | ReinhardLocal, DurandToneMap, FattalToneMap |
| `hdr_imaging/tone_mapping/perceptual_operators.py` | ACESToneMap, FilmicToneMap, MantiukToneMap, HistogramToneMap |
| `hdr_imaging/exposure_fusion/__init__.py` | Sub-package exports for exposure fusion |
| `hdr_imaging/exposure_fusion/mertens.py` | Mertens 2007 exposure fusion |
| `hdr_imaging/single_image/__init__.py` | Sub-package exports for single image HDR |
| `hdr_imaging/single_image/single_image_hdr.py` | CLAHE + multi-scale detail enhancement |
| `hdr_imaging/hdr_pipeline.py` | HDRPipeline main controller, HDRResult NamedTuple |
| `hdr_imaging/docs/hdr_math_principles.md` | Mathematical principles document |
| `demo_hdr.py` | Demo script with synthetic data + user input + compare mode |
| `README_HDR.md` | Complete Chinese documentation with math formulas |

### Files to Modify

| File | Change |
|------|--------|
| `.gitignore` | Add `output_hdr/` line |
| `README_algorithms.md` | Add link to `README_HDR.md` at the end |

---

## Task 1: Project Scaffolding and .gitignore

**Files:**
- Modify: `.gitignore`
- Create: `hdr_imaging/__init__.py` (stub)
- Create: All sub-package `__init__.py` files (stubs)

- [ ] **Step 1: Update .gitignore**

Add `output_hdr/` to `.gitignore` after the existing `output_stabilization/` line:

```
__pycache__/
*.pyc
*.pyo
output_panorama/
output_stabilization/
output_hdr/
*.mp4
```

- [ ] **Step 2: Create all sub-package directories and stub __init__.py files**

Create the following empty `__init__.py` files to establish the package structure:

```
hdr_imaging/__init__.py
hdr_imaging/alignment/__init__.py
hdr_imaging/calibration/__init__.py
hdr_imaging/merge/__init__.py
hdr_imaging/tone_mapping/__init__.py
hdr_imaging/exposure_fusion/__init__.py
hdr_imaging/single_image/__init__.py
hdr_imaging/docs/   (created in Task 14 when hdr_math_principles.md is written)
```

Each sub-package `__init__.py` should start as an empty file — imports will be added as modules are implemented.

The root `hdr_imaging/__init__.py` should be a stub with just the docstring and version:

```python
"""
HDR 高动态范围成像算法库 / HDR Imaging Algorithm Library

模块结构:
- alignment: MTB / 特征点图像对齐
- calibration: Debevec / Robertson 相机响应函数标定
- merge: 多曝光 HDR 辐射图合并
- tone_mapping: 10种色调映射算法（全局/局部/感知驱动）
- exposure_fusion: Mertens 多曝光融合
- single_image: 单张图像 HDR 增强
- hdr_pipeline: HDR 处理主控流水线
"""

__version__ = '1.0.0'
__author__ = 'HDR Imaging Algorithm Implementation'
```

- [ ] **Step 3: Verify package structure**

Run: `python -c "import hdr_imaging; print(hdr_imaging.__version__)"`
Expected: `1.0.0`

- [ ] **Step 4: Commit**

```bash
git add .gitignore hdr_imaging/
git commit -m "feat(hdr): scaffold hdr_imaging package structure"
```

---

## Task 2: MTB Alignment (alignment/mtb_alignment.py)

**Files:**
- Create: `hdr_imaging/alignment/mtb_alignment.py`
- Modify: `hdr_imaging/alignment/__init__.py`

- [ ] **Step 1: Implement MTBAlignment class**

Create `hdr_imaging/alignment/mtb_alignment.py` with `MTBAlignment` class:

```python
"""
MTB 中值阈值位图对齐模块
Median Threshold Bitmap Alignment (Ward 2003)

【算法原理】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MTB (Median Threshold Bitmap) 利用图像的中值将像素二值化，
生成的位图对曝光变化不敏感，因此可用于不同曝光图像间的对齐。

1) 中值阈值位图:
   对灰度图计算中值 median，生成:
   MTB(x,y) = 1 if I(x,y) > median, else 0

2) 排除位图 (Exclusion Bitmap):
   靠近中值的像素不稳定，需排除:
   EB(x,y) = 0 if |I(x,y) - median| < threshold, else 1

3) 金字塔粗到细搜索:
   - 构建图像金字塔（下采样）
   - 从最粗层开始，搜索 [-1,0,1] 的 XY 位移
   - 用 XOR 计算 MTB 差异（排除 EB 为 0 的区域）
   - 逐层细化位移量（上层位移 ×2 + 当前层修正）

优点: 计算简单高效，对曝光差异鲁棒
局限: 只能估计平移，不能处理旋转或缩放
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
```

The class should implement:

- `__init__(self, max_level: int = 5, threshold: int = 4)` — `max_level` controls pyramid depth, `threshold` controls exclusion bitmap tolerance
- `_compute_mtb(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]` — returns (mtb, exclusion_bitmap) for a grayscale image. MTB: pixel > median → 1, else 0. EB: |pixel - median| >= threshold → 1, else 0.
- `_compute_shift(self, mtb1, eb1, mtb2, eb2) -> Tuple[int, int]` — search 9 offsets [-1,0,1]×[-1,0,1], compute XOR between shifted MTBs masked by AND of EBs, return offset with minimum XOR bit count
- `process(self, images: List[np.ndarray], reference_index: int = 0) -> List[np.ndarray]` — align all images to reference using pyramid coarse-to-fine. For each pyramid level from coarsest to finest: downsample, compute MTB/EB, find best shift, accumulate shift (previous_shift * 2 + current_shift). Apply final shift with `np.roll` or `cv2.warpAffine`. Return aligned images list.
- `process_opencv(self, images: List[np.ndarray], reference_index: int = 0) -> List[np.ndarray]` — use `cv2.createAlignMTB()` and `alignMTB.process()`.

Validation: raise `ValueError("At least 2 images required for alignment")` if `len(images) < 2`.

Follow the existing code style:
- Chinese docstrings with math formulas in the module header
- Type hints on all methods
- `logger.info()` for progress messages

- [ ] **Step 2: Update alignment/__init__.py**

```python
from .mtb_alignment import MTBAlignment
```

- [ ] **Step 3: Verify MTB alignment works**

Run: `python -c "
import numpy as np
from hdr_imaging.alignment import MTBAlignment
aligner = MTBAlignment()
# Create two shifted images
img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
img2 = np.roll(img1, 3, axis=1)  # shift right by 3
aligned = aligner.process([img1, img2])
print(f'Input shapes: {img1.shape}, {img2.shape}')
print(f'Aligned count: {len(aligned)}')
print('MTB alignment OK')
"`

Expected: prints shapes and "MTB alignment OK" without errors.

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/alignment/
git commit -m "feat(hdr): implement MTB alignment (Ward 2003)"
```

---

## Task 3: Feature-Based Alignment (alignment/feature_alignment.py)

**Files:**
- Create: `hdr_imaging/alignment/feature_alignment.py`
- Modify: `hdr_imaging/alignment/__init__.py`

- [ ] **Step 1: Implement FeatureAlignment class**

Create `hdr_imaging/alignment/feature_alignment.py` with `FeatureAlignment` class:

Module docstring should explain: reuses `panorama_stitching.FeatureExtractor` and `HomographyEstimator` for SIFT/ORB feature matching + RANSAC homography estimation to handle translation, rotation, and scale changes between exposures.

```python
import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# 复用全景拼接模块的特征提取和单应性估计
from panorama_stitching.feature_extraction import FeatureExtractor, FeatureMatcher
from panorama_stitching.homography import HomographyEstimator
```

The class should implement:

- `__init__(self, feature_method: str = 'SIFT', max_features: int = 1000)` — create `FeatureExtractor` and `HomographyEstimator` instances
- `process(self, images: List[np.ndarray], reference_index: int = 0) -> List[np.ndarray]` — for each non-reference image: extract features from both, match features, estimate homography via RANSAC, warp with `cv2.warpPerspective` to align to reference. Return aligned images list. Reference image passes through unchanged.
- `process_opencv(self, images, reference_index=0)` — same as process() since both use OpenCV under the hood. Just call `self.process()`.

Validation: same `ValueError` for < 2 images.

- [ ] **Step 2: Update alignment/__init__.py**

```python
from .mtb_alignment import MTBAlignment
from .feature_alignment import FeatureAlignment
```

- [ ] **Step 3: Verify feature alignment imports work**

Run: `python -c "from hdr_imaging.alignment import MTBAlignment, FeatureAlignment; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/alignment/
git commit -m "feat(hdr): implement feature-based alignment (reusing panorama_stitching)"
```

---

## Task 4: Debevec CRF Calibration (calibration/debevec.py)

**Files:**
- Create: `hdr_imaging/calibration/debevec.py`
- Modify: `hdr_imaging/calibration/__init__.py`

- [ ] **Step 1: Implement DebevecCalibration class**

Create `hdr_imaging/calibration/debevec.py` with detailed module docstring explaining:

**Debevec & Malik (1997) 数学原理:**
- 成像模型: Z = f(E·Δt)，其中 Z 是像素值, E 是辐照度, Δt 是曝光时间
- 定义 g(Z) = ln(f⁻¹(Z)) = ln(E) + ln(Δt)
- 对 N 张图像中的 P 个采样像素，构建过约束线性系统:
  - g(Z_ij) - ln(E_i) = ln(Δt_j)  对每个像素 i、图像 j
  - 加入平滑约束: λ · g''(z) = 0
- 写成矩阵形式 Ax = b，SVD 求解最小二乘
- 权重函数 w(z): 三角帽形，z=128 时最大，z=0 和 z=255 时为 0:
  w(z) = z - Z_min      if z <= (Z_min + Z_max) / 2
  w(z) = Z_max - z      if z > (Z_min + Z_max) / 2

The class should implement:

- `__init__(self, samples: int = 50, lambda_smooth: float = 10.0)` — `samples`: number of pixel locations to sample, `lambda_smooth`: smoothness weight
- `_weight(self, z: int) -> float` — triangle hat weighting function
- `_sample_pixels(self, images: List[np.ndarray]) -> np.ndarray` — randomly select `self.samples` pixel locations, return Z values matrix (samples × num_images)
- `process(self, images: List[np.ndarray], exposure_times: List[float]) -> np.ndarray` — Hand-written implementation:
  1. Convert images to grayscale (or process each channel separately)
  2. Sample pixel locations
  3. Build the overdetermined linear system A·x = b with smoothness constraint
  4. Solve via `np.linalg.lstsq` or SVD (`np.linalg.svd`)
  5. Extract g(z) curve (256 values) and ln(E) values
  6. Return response curve as numpy array shape (256,) or (256, 3) for color
  7. On SVD failure, raise `RuntimeError("CRF recovery failed: insufficient exposure variation")`
- `process_opencv(self, images, exposure_times)` — use `cv2.createCalibrateDebevec(samples, lambda_smooth)` and `.process()`.

Process each BGR channel independently to get a (256, 3) response curve.

- [ ] **Step 2: Update calibration/__init__.py**

```python
from .debevec import DebevecCalibration
```

- [ ] **Step 3: Verify Debevec calibration**

Run: `python -c "
import numpy as np
from hdr_imaging.calibration import DebevecCalibration
cal = DebevecCalibration(samples=20)
# Create synthetic multi-exposure images
np.random.seed(42)
scene = np.random.rand(64, 64, 3) * 1000  # HDR scene
exposures = [0.033, 0.25, 1.0, 4.0]
images = []
for t in exposures:
    ldr = np.clip(scene * t, 0, 255).astype(np.uint8)
    images.append(ldr)
curve = cal.process(images, exposures)
print(f'Response curve shape: {curve.shape}')
print(f'Curve range: [{curve.min():.2f}, {curve.max():.2f}]')
print('Debevec OK')
"`

Expected: Response curve shape (256, 3) or (256,), prints "Debevec OK".

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/calibration/
git commit -m "feat(hdr): implement Debevec CRF calibration (1997)"
```

---

## Task 5: Robertson CRF Calibration (calibration/robertson.py)

**Files:**
- Create: `hdr_imaging/calibration/robertson.py`
- Modify: `hdr_imaging/calibration/__init__.py`

- [ ] **Step 1: Implement RobertsonCalibration class**

Module docstring should explain Robertson (2003) EM iteration:

**Robertson 数学原理:**
- E-step: 估计辐照度 E_i = Σ_j w(Z_ij) · g⁻¹(Z_ij) / Δt_j / Σ_j w(Z_ij)
- M-step: 更新响应函数 g(z) = Σ_{ij where Z_ij=z} E_i · Δt_j / count
- 迭代直到收敛 (response curve change < epsilon 或达到 max_iter)
- 初始 g(z) = z / 255（线性假设）

The class should implement:

- `__init__(self, max_iter: int = 30, epsilon: float = 1e-3)` — convergence parameters
- `process(self, images, exposure_times)` — EM iteration, returns response curve (256, 3)
- `process_opencv(self, images, exposure_times)` — use `cv2.createCalibrateRobertson(max_iter, epsilon)` and `.process()`

- [ ] **Step 2: Update calibration/__init__.py**

```python
from .debevec import DebevecCalibration
from .robertson import RobertsonCalibration
```

- [ ] **Step 3: Verify Robertson calibration**

Run: `python -c "
import numpy as np
from hdr_imaging.calibration import RobertsonCalibration
cal = RobertsonCalibration()
scene = np.random.rand(64, 64, 3) * 1000
exposures = [0.033, 0.25, 1.0, 4.0]
images = [np.clip(scene * t, 0, 255).astype(np.uint8) for t in exposures]
curve = cal.process(images, exposures)
print(f'Robertson curve shape: {curve.shape}')
print('Robertson OK')
"`

Expected: prints shape and "Robertson OK".

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/calibration/
git commit -m "feat(hdr): implement Robertson CRF calibration (2003)"
```

---

## Task 6: HDR Merge (merge/hdr_merge.py)

**Files:**
- Create: `hdr_imaging/merge/hdr_merge.py`
- Modify: `hdr_imaging/merge/__init__.py`

- [ ] **Step 1: Implement HDRMerge class**

Module docstring should explain the weighted irradiance merge:

**合并公式:**
- ln(E_i) = Σ_j w(Z_ij) · (g(Z_ij) - ln(Δt_j)) / Σ_j w(Z_ij)
- w(z): triangle hat weight function (same as Debevec)
- 当某像素所有曝光均饱和时: 退化到最接近中间曝光的值, log warning

The class should implement:

- `__init__(self)` — minimal, just store logger
- `_weight(self, z)` — same triangle hat as Debevec (can be a module-level utility)
- `process(self, images, exposure_times, response_curve)` — hand-written weighted merge returning float32 HDR radiance map. Handle the all-saturated pixel edge case with degraded fallback.
- `process_opencv(self, images, exposure_times, response_curve)` — use `cv2.createMergeDebevec()` and `.process()`

- [ ] **Step 2: Update merge/__init__.py**

```python
from .hdr_merge import HDRMerge
```

- [ ] **Step 3: Verify HDR merge**

Run: `python -c "
import numpy as np
from hdr_imaging.calibration import DebevecCalibration
from hdr_imaging.merge import HDRMerge
scene = np.random.rand(64, 64, 3) * 1000
exposures = [0.033, 0.25, 1.0, 4.0]
images = [np.clip(scene * t, 0, 255).astype(np.uint8) for t in exposures]
cal = DebevecCalibration(samples=20)
curve = cal.process(images, exposures)
merger = HDRMerge()
hdr_map = merger.process(images, exposures, curve)
print(f'HDR map shape: {hdr_map.shape}, dtype: {hdr_map.dtype}')
print(f'HDR range: [{hdr_map.min():.4f}, {hdr_map.max():.4f}]')
print('HDR merge OK')
"`

Expected: float32 HDR map with shape matching input, prints "HDR merge OK".

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/merge/
git commit -m "feat(hdr): implement weighted HDR radiance map merge"
```

---

## Task 7: Global Tone Mapping Operators (tone_mapping/global_operators.py)

**Files:**
- Create: `hdr_imaging/tone_mapping/global_operators.py`
- Modify: `hdr_imaging/tone_mapping/__init__.py`

- [ ] **Step 1: Implement ReinhardGlobal class**

**Reinhard (2002) 全局算子数学原理:**
- 将 HDR 辐照度映射到显示范围
- 计算对数平均亮度: L̄ = exp(Σ ln(δ + L(x,y)) / N)
- 缩放亮度: L_scaled(x,y) = (a / L̄) · L(x,y)，a 为 key value (默认 0.18)
- 色调映射: L_d(x,y) = L_scaled · (1 + L_scaled / L_white²) / (1 + L_scaled)
  L_white 是映射到纯白的最小亮度值
- 最后从亮度恢复彩色: C_out = C_in · (L_d / L_in)

Parameters: `__init__(self, key_value: float = 0.18, white_point: float = None)`. If `white_point` is None, use max luminance.

Methods:
- `process(self, hdr_image: np.ndarray) -> np.ndarray` — hand-written, return uint8 LDR
- `process_opencv(self, hdr_image) -> np.ndarray` — use `cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)`

- [ ] **Step 2: Implement DragoToneMap class**

**Drago (2003) 数学原理:**
- 自适应对数映射: L_d = L_max_d · log(1 + L_scaled) / log(1 + L_max) / log(2 + 8·(L/L_max)^(log(b)/log(0.5)))
- b 为偏差参数 (默认 0.85)

Parameters: `__init__(self, gamma: float = 2.2, saturation: float = 1.0, bias: float = 0.85)`

Methods:
- `process(self, hdr_image)` — hand-written
- `process_opencv(self, hdr_image)` — use `cv2.createTonemapDrago(gamma, saturation, bias)`

- [ ] **Step 3: Implement AdaptiveLog class**

**自适应对数映射原理:**
- L_d(x,y) = L_d_max · log(1 + L(x,y)/L_w_avg) / log(1 + L_max/L_w_avg)
- L_w_avg: 场景平均亮度, L_max: 场景最大亮度, L_d_max: 显示最大亮度

Parameters: `__init__(self, gamma: float = 2.2)`

Methods:
- `process(self, hdr_image)` — hand-written
- `process_opencv(self, hdr_image)` — raise `NotImplementedError("AdaptiveLog has no OpenCV equivalent. Use process() instead.")`

- [ ] **Step 4: Update tone_mapping/__init__.py**

```python
from .global_operators import ReinhardGlobal, DragoToneMap, AdaptiveLog
```

- [ ] **Step 5: Verify all global operators**

Run: `python -c "
import numpy as np
from hdr_imaging.tone_mapping import ReinhardGlobal, DragoToneMap, AdaptiveLog
# Create a synthetic HDR image
hdr = np.random.rand(64, 64, 3).astype(np.float32) * 1000
for Cls in [ReinhardGlobal, DragoToneMap, AdaptiveLog]:
    op = Cls()
    result = op.process(hdr)
    print(f'{Cls.__name__}: shape={result.shape}, dtype={result.dtype}, range=[{result.min()}, {result.max()}]')
print('Global operators OK')
"`

Expected: Each produces uint8 images, prints "Global operators OK".

- [ ] **Step 6: Commit**

```bash
git add hdr_imaging/tone_mapping/
git commit -m "feat(hdr): implement global tone mapping (Reinhard, Drago, AdaptiveLog)"
```

---

## Task 8: Local Tone Mapping Operators (tone_mapping/local_operators.py)

**Files:**
- Create: `hdr_imaging/tone_mapping/local_operators.py`
- Modify: `hdr_imaging/tone_mapping/__init__.py`

- [ ] **Step 1: Implement ReinhardLocal class**

**Reinhard (2002) 局部算子数学原理:**
- 使用多尺度高斯滤波计算局部适应亮度
- V1(x,y,s) = G(x,y,s) * L(x,y) — 高斯卷积
- 搜索最大尺度 s 使得: |V1(x,y,s) - V2(x,y,s)| / (2^φ · a/s² + V1(x,y,s)) < ε
- 色调映射: L_d(x,y) = L(x,y) / (1 + V1(x,y,s_max))

Parameters: `__init__(self, key_value: float = 0.18, phi: float = 8.0, num_scales: int = 8, epsilon: float = 0.05)`

Methods:
- `process(self, hdr_image)` — multi-scale Gaussian surround, return uint8
- `process_opencv(self, hdr_image)` — `cv2.createTonemapReinhard()` with `light_adapt=1.0` for local behavior

- [ ] **Step 2: Implement DurandToneMap class**

**Durand & Dorsey (2002) 数学原理:**
- 转换到对数域: L_log = log(L + ε)
- 双边滤波分离: base = bilateralFilter(L_log), detail = L_log - base
- 压缩基础层: base_compressed = (base - max(base)) · (target_contrast / (max(base) - min(base)))
- 重构: L_out = exp(base_compressed + detail)

Parameters: `__init__(self, sigma_spatial: float = 2.0, sigma_range: float = 2.0, target_contrast: float = 5.0)`

Methods:
- `process(self, hdr_image)` — bilateral filter decomposition, return uint8
- `process_opencv(self, hdr_image)` — `cv2.createTonemapDurand(gamma, contrast, saturation, sigma_space, sigma_color)`

- [ ] **Step 3: Implement FattalToneMap class**

**Fattal (2002) 梯度域色调映射数学原理:**
- 计算 HDR 亮度对数梯度: ∇H = ∇(log L)
- 根据梯度幅值衰减: φ(x,y) = (α / |∇H|) · (|∇H| / α)^β，β < 1
- 衰减后的梯度场: G = φ · ∇H
- 泊松方程重建: ∇²I = div(G)
- 使用 scipy.sparse.linalg.spsolve 求解
- 大图像 (>2048×2048) 自动 0.5x 降采样

Parameters: `__init__(self, alpha: float = 0.1, beta: float = 0.8, saturation: float = 1.0, max_size: int = 2048)`

Methods:
- `process(self, hdr_image)` — gradient domain attenuation + Poisson solve, return uint8
- `process_opencv(self, hdr_image)` — raise `NotImplementedError`

Important: import `from scipy.sparse import linalg as splinalg, csc_matrix` for the Poisson solver.

- [ ] **Step 4: Update tone_mapping/__init__.py**

Add imports:
```python
from .local_operators import ReinhardLocal, DurandToneMap, FattalToneMap
```

- [ ] **Step 5: Verify all local operators**

Run: `python -c "
import numpy as np
from hdr_imaging.tone_mapping import ReinhardLocal, DurandToneMap, FattalToneMap
hdr = np.random.rand(64, 64, 3).astype(np.float32) * 1000 + 1.0
for Cls in [ReinhardLocal, DurandToneMap, FattalToneMap]:
    op = Cls()
    result = op.process(hdr)
    print(f'{Cls.__name__}: shape={result.shape}, dtype={result.dtype}')
print('Local operators OK')
"`

Expected: prints shapes and "Local operators OK".

- [ ] **Step 6: Commit**

```bash
git add hdr_imaging/tone_mapping/
git commit -m "feat(hdr): implement local tone mapping (Reinhard local, Durand, Fattal)"
```

---

## Task 9: Perceptual Tone Mapping Operators (tone_mapping/perceptual_operators.py)

**Files:**
- Create: `hdr_imaging/tone_mapping/perceptual_operators.py`
- Modify: `hdr_imaging/tone_mapping/__init__.py`

- [ ] **Step 1: Implement ACESToneMap class**

**ACES 电影曲线:**
- 标准 ACES 拟合曲线 (Narkowicz 2015 近似):
  f(x) = (x · (a·x + b)) / (x · (c·x + d) + e)
  参数: a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
- 先乘以曝光系数，再应用曲线，最后 gamma 校正

Parameters: `__init__(self, exposure: float = 1.0, gamma: float = 2.2)`
Methods: `process()` — hand-written; `process_opencv()` — raise `NotImplementedError`

- [ ] **Step 2: Implement FilmicToneMap class**

**Uncharted 2 Filmic 曲线:**
- Hable (2010) 分段函数:
  f(x) = ((x·(A·x+C·B)+D·E) / (x·(A·x+B)+D·F)) - E/F
  标准参数: A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30
- 归一化: result = f(exposure * color) / f(white_point)

Parameters: `__init__(self, exposure: float = 2.0, white_point: float = 11.2)`
Methods: `process()` — hand-written; `process_opencv()` — raise `NotImplementedError`

- [ ] **Step 3: Implement MantiukToneMap class**

**Mantiuk (2006) 感知对比度映射:**
- 将 HDR 亮度转换到感知对比度空间 (transducer model)
- 使用人眼 CSF (Contrast Sensitivity Function) 加权
- 优化映射使感知对比度失真最小
- 简化实现: 使用对数亮度域的感知对比度均衡化

Parameters: `__init__(self, gamma: float = 2.2, scale: float = 0.85, saturation: float = 1.0)`
Methods: `process()` — hand-written simplified version; `process_opencv()` — `cv2.createTonemapMantiuk(gamma, scale, saturation)`

- [ ] **Step 4: Implement HistogramToneMap class**

**直方图均衡化色调映射:**
- 计算 HDR 亮度的对数直方图
- 构建累积直方图 (CDF)
- 用 CDF 重映射亮度到 [0, 1]
- 从亮度恢复彩色

Parameters: `__init__(self, clip_limit: float = 0.0, num_bins: int = 256)`
Methods: `process()` — hand-written; `process_opencv()` — raise `NotImplementedError`

- [ ] **Step 5: Update tone_mapping/__init__.py**

Add imports for all perceptual operators. Final `tone_mapping/__init__.py`:
```python
from .global_operators import ReinhardGlobal, DragoToneMap, AdaptiveLog
from .local_operators import ReinhardLocal, DurandToneMap, FattalToneMap
from .perceptual_operators import ACESToneMap, FilmicToneMap, MantiukToneMap, HistogramToneMap
```

- [ ] **Step 6: Verify all perceptual operators**

Run: `python -c "
import numpy as np
from hdr_imaging.tone_mapping import ACESToneMap, FilmicToneMap, MantiukToneMap, HistogramToneMap
hdr = np.random.rand(64, 64, 3).astype(np.float32) * 1000 + 1.0
for Cls in [ACESToneMap, FilmicToneMap, MantiukToneMap, HistogramToneMap]:
    op = Cls()
    result = op.process(hdr)
    print(f'{Cls.__name__}: shape={result.shape}, dtype={result.dtype}')
print('Perceptual operators OK')
"`

Expected: prints shapes and "Perceptual operators OK".

- [ ] **Step 7: Commit**

```bash
git add hdr_imaging/tone_mapping/
git commit -m "feat(hdr): implement perceptual tone mapping (ACES, Filmic, Mantiuk, Histogram)"
```

---

## Task 10: Mertens Exposure Fusion (exposure_fusion/mertens.py)

**Files:**
- Create: `hdr_imaging/exposure_fusion/mertens.py`
- Modify: `hdr_imaging/exposure_fusion/__init__.py`

- [ ] **Step 1: Implement MertensFusion class**

**Mertens (2007) 曝光融合数学原理:**
- 对每张图像计算三个质量度量:
  - 对比度 C: 拉普拉斯滤波绝对值
  - 饱和度 S: RGB 通道标准差
  - 曝光度 E: exp(-0.5 · ((pixel - 0.5) / σ)²)，σ 默认 0.2
- 组合权重: W_k(x,y) = C_k^wc · S_k^ws · E_k^we + ε
- 归一化权重: Ŵ_k = W_k / Σ_k W_k
- 拉普拉斯金字塔融合:
  - 对每张图像构建拉普拉斯金字塔 L_k
  - 对权重构建高斯金字塔 G(Ŵ_k)
  - 融合: L_fused[l] = Σ_k G(Ŵ_k)[l] · L_k[l]
  - 从融合金字塔重建最终图像

Parameters: `__init__(self, contrast_weight: float = 1.0, saturation_weight: float = 1.0, exposure_weight: float = 1.0, sigma: float = 0.2, pyramid_levels: int = None)`

Methods:
- `_compute_weights(self, images)` — compute C, S, E for each image
- `_build_laplacian_pyramid(self, image, levels)` — Gaussian → Laplacian decomposition
- `_build_gaussian_pyramid(self, image, levels)` — Gaussian downsampling pyramid
- `_reconstruct_from_pyramid(self, pyramid)` — bottom-up reconstruction
- `process(self, images)` — full Mertens pipeline, return uint8
- `process_opencv(self, images)` — use `cv2.createMergeMertens(contrast_weight, saturation_weight, exposure_weight)` and `.process()`

Validation: raise `ValueError` if < 2 images.

- [ ] **Step 2: Update exposure_fusion/__init__.py**

```python
from .mertens import MertensFusion
```

- [ ] **Step 3: Verify Mertens fusion**

Run: `python -c "
import numpy as np
from hdr_imaging.exposure_fusion import MertensFusion
images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
fusion = MertensFusion()
result = fusion.process(images)
print(f'Mertens result: shape={result.shape}, dtype={result.dtype}')
print('Mertens OK')
"`

Expected: uint8 result, prints "Mertens OK".

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/exposure_fusion/
git commit -m "feat(hdr): implement Mertens exposure fusion (2007)"
```

---

## Task 11: Single Image HDR (single_image/single_image_hdr.py)

**Files:**
- Create: `hdr_imaging/single_image/single_image_hdr.py`
- Modify: `hdr_imaging/single_image/__init__.py`

- [ ] **Step 1: Implement SingleImageHDR class**

**单张图像 HDR 增强原理:**
- 转换到 LAB 色彩空间，分离亮度通道 L
- 对 L 通道应用 CLAHE (Contrast Limited Adaptive Histogram Equalization):
  - 将图像分为 tile_size × tile_size 块
  - 每块独立做直方图均衡，但限制对比度增强幅度 (clip_limit)
  - 双线性插值消除块边界
- 多尺度细节增强:
  - 构建高斯金字塔提取不同尺度细节
  - 加权增强细节层
- 色彩保持: 通过 LAB 空间处理，保持 a, b 通道不变
- 合并回 BGR

Parameters: `__init__(self, clip_limit: float = 3.0, tile_size: int = 8, detail_boost: float = 1.5)`

Methods:
- `process(self, image)` — hand-written CLAHE + multi-scale detail, return uint8
- `process_opencv(self, image)` — use `cv2.createCLAHE(clipLimit, tileGridSize)` for the CLAHE step

- [ ] **Step 2: Update single_image/__init__.py**

```python
from .single_image_hdr import SingleImageHDR
```

- [ ] **Step 3: Verify single image HDR**

Run: `python -c "
import numpy as np
from hdr_imaging.single_image import SingleImageHDR
img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
enhancer = SingleImageHDR()
result = enhancer.process(img)
print(f'SingleImageHDR: shape={result.shape}, dtype={result.dtype}')
print('SingleImageHDR OK')
"`

Expected: uint8 result, prints "SingleImageHDR OK".

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/single_image/
git commit -m "feat(hdr): implement single image HDR (CLAHE + multi-scale detail)"
```

---

## Task 12: HDR Pipeline Controller (hdr_pipeline.py)

**Files:**
- Create: `hdr_imaging/hdr_pipeline.py`
- Modify: `hdr_imaging/__init__.py` (add full exports)

- [ ] **Step 1: Implement HDRPipeline class and HDRResult**

Create `hdr_imaging/hdr_pipeline.py`:

```python
"""
HDR 处理主控流水线
HDR Processing Pipeline Controller

串联各子模块完成完整 HDR 流程:
输入多曝光图像 → 对齐 → CRF标定 → HDR合并 → 色调映射 → LDR输出

也支持替代路径:
- 多曝光融合 (Mertens): 输入图像 → 对齐 → 融合 → LDR输出
- 单张图像HDR: 输入单张LDR → CLAHE增强 → 输出
"""

from collections import namedtuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

HDRResult = namedtuple('HDRResult', ['ldr_result', 'hdr_radiance_map', 'response_curve'])
```

The class should implement:

- `__init__(self, align_method='mtb', calibration_method='debevec', tone_mapping_method='reinhard_global', use_opencv=False)`:
  - Validate `align_method` in `{'mtb', 'feature'}`
  - Validate `calibration_method` in `{'debevec', 'robertson'}`
  - Validate `tone_mapping_method` against the mapping table (all 10 strings)
  - Raise `ValueError` listing valid options on invalid input
  - Instantiate the appropriate alignment, calibration, merge, and tone mapping objects

- `process(self, images, exposure_times) -> HDRResult`:
  1. Validate: `len(images) >= 2`, `len(images) == len(exposure_times)`
  2. Align images
  3. Recover CRF
  4. Merge to HDR radiance map
  5. Tone map to LDR
  6. Call `process()` or `process_opencv()` based on `self.use_opencv`
  7. Return `HDRResult(ldr_result, hdr_radiance_map, response_curve)`

- `exposure_fusion(self, images) -> np.ndarray`:
  1. Validate: `len(images) >= 2`
  2. Align (if align_method set)
  3. MertensFusion.process() or process_opencv()
  4. Return uint8 result

- `single_image_hdr(self, image) -> np.ndarray`:
  1. SingleImageHDR.process() or process_opencv()
  2. Return uint8 result

Tone mapping method string mapping (from spec):
```python
TONE_MAPPING_METHODS = {
    'reinhard_global': ReinhardGlobal,
    'reinhard_local': ReinhardLocal,
    'drago': DragoToneMap,
    'durand': DurandToneMap,
    'fattal': FattalToneMap,
    'adaptive_log': AdaptiveLog,
    'aces': ACESToneMap,
    'filmic': FilmicToneMap,
    'mantiuk': MantiukToneMap,
    'histogram': HistogramToneMap,
}
```

- [ ] **Step 2: Update root hdr_imaging/__init__.py with full exports**

Replace the stub with the full version from the spec (lines 49-71 of the design doc), importing `HDRResult` as well:

```python
from .hdr_pipeline import HDRPipeline, HDRResult
```

Add `HDRResult` to `__all__` (total 20 entries: 19 classes + HDRResult).

- [ ] **Step 3: Verify full pipeline**

Run: `python -c "
import numpy as np
from hdr_imaging import HDRPipeline
pipeline = HDRPipeline(tone_mapping_method='reinhard_global')
scene = np.random.rand(64, 64, 3) * 1000
exposures = [0.033, 0.25, 1.0, 4.0]
images = [np.clip(scene * t, 0, 255).astype(np.uint8) for t in exposures]
result = pipeline.process(images, exposures)
print(f'LDR: {result.ldr_result.shape}, HDR: {result.hdr_radiance_map.shape}')
print(f'Curve: {result.response_curve.shape}')

# Test exposure fusion
fused = pipeline.exposure_fusion(images)
print(f'Fusion: {fused.shape}')

# Test single image
single = pipeline.single_image_hdr(images[0])
print(f'Single: {single.shape}')
print('Pipeline OK')
"`

Expected: all shapes printed, "Pipeline OK".

- [ ] **Step 4: Commit**

```bash
git add hdr_imaging/hdr_pipeline.py hdr_imaging/__init__.py
git commit -m "feat(hdr): implement HDRPipeline controller with full flow"
```

---

## Task 13: Demo Script (demo_hdr.py)

**Files:**
- Create: `demo_hdr.py`

- [ ] **Step 1: Implement demo_hdr.py**

Follow the pattern from `demo_panorama.py`:
- Module docstring with usage examples (Chinese + English)
- `import matplotlib; matplotlib.use('Agg')` before other matplotlib imports
- `logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')`
- argparse with: `--test`, `--images` (nargs='+'), `--exposures` (nargs='+', type=float), `--single`, `--compare`
- Output to `output_hdr/` directory

Functions to implement:

`generate_synthetic_hdr_scene(width=256, height=256)`:
- Create a virtual HDR scene with bright areas (sun/lights, values up to 10000) and dark areas (shadows, values near 1-10)
- Use geometric shapes (circles for lights, gradients for shadows) to create a scene with 4-5 orders of magnitude dynamic range
- Return float32 HDR scene

`simulate_exposures(hdr_scene, exposure_times, noise_sigma=3.0, add_shift=False)`:
- For each exposure time: LDR = clip(hdr_scene × Δt, 0, 255).astype(uint8)
- Add Gaussian noise
- If `add_shift`: apply random small translations (1-5 pixels) to simulate hand-held
- Return list of LDR images

`run_test_mode(output_dir)`:
- Generate synthetic scene
- Run full HDR pipeline with Debevec calibration
- Save exposure input comparison
- Save response curves (Debevec vs Robertson)
- Save HDR radiance map (pseudocolor via matplotlib)
- Save tone mapping results for default method
- Run Mertens exposure fusion, save result
- Run single image HDR, save before/after

`run_compare_mode(output_dir)`:
- Generate synthetic scene + exposures
- Run pipeline calibration + merge once
- Apply all 10 tone mapping algorithms to the same HDR map
- Create a 2×5 or 3×4 grid figure comparing all results
- Save the comparison grid image
- Also compare hand-written vs OpenCV implementations (for algorithms that have both)

`run_custom_mode(image_paths, exposure_times, output_dir)`:
- Load user images
- Run full pipeline
- Save results

`run_single_mode(image_path, output_dir)`:
- Load single image
- Run SingleImageHDR
- Save before/after comparison

`main()`:
- Parse args
- Create output_dir
- Route to appropriate mode
- Print summary of saved files

All matplotlib figures saved with `plt.savefig()`, `plt.close()`. Use `Agg` backend.

- [ ] **Step 2: Verify --test mode runs**

Run: `python demo_hdr.py --test`
Expected: creates `output_hdr/` with saved images, no errors.

- [ ] **Step 3: Verify --compare mode runs**

Run: `python demo_hdr.py --compare`
Expected: creates comparison grid image in `output_hdr/`.

- [ ] **Step 4: Commit**

```bash
git add demo_hdr.py
git commit -m "feat(hdr): add demo script with synthetic data and comparison modes"
```

---

## Task 14: Math Principles Document (hdr_imaging/docs/hdr_math_principles.md)

**Files:**
- Create: `hdr_imaging/docs/hdr_math_principles.md`

- [ ] **Step 1: Write the math principles document**

Chinese document following the style of `calibration_algorithms/docs/calibration_math_principles.md`. Structure:

```markdown
# HDR 高动态范围成像 — 数学原理

## 1. HDR 成像基础
- 动态范围定义
- 人眼 vs 相机传感器动态范围
- HDR 成像流程概述

## 2. 相机响应函数 (CRF)
### 2.1 成像模型
- Z = f(E · Δt)
- g(Z) = ln(f⁻¹(Z)) = ln(E) + ln(Δt)

### 2.2 Debevec & Malik 方法
- 完整公式推导
- 线性系统构建
- SVD 求解

### 2.3 Robertson 方法
- MLE 框架
- EM 迭代公式

## 3. HDR 辐射图合并
- 加权合并公式
- 权重函数设计

## 4. 色调映射
### 4.1 全局算子
- Reinhard 全局 (完整公式)
- Drago (完整公式)
- 自适应对数

### 4.2 局部算子
- Reinhard 局部 (多尺度)
- Durand 双边滤波分解
- Fattal 梯度域 + 泊松方程

### 4.3 感知驱动
- ACES 曲线
- Filmic 曲线
- Mantiuk 感知对比度
- 直方图映射

## 5. 多曝光融合 (Mertens)
- 质量度量
- 拉普拉斯金字塔融合

## 6. 图像对齐
### 6.1 MTB
- 中值阈值位图
- 金字塔搜索

### 6.2 特征点对齐
- SIFT/ORB + RANSAC

## 参考文献
```

Each section should include the full mathematical formulas using plain text notation (matching the style of existing docs in the project).

- [ ] **Step 2: Commit**

```bash
git add hdr_imaging/docs/
git commit -m "docs(hdr): add mathematical principles document"
```

---

## Task 15: README_HDR.md Documentation

**Files:**
- Create: `README_HDR.md`
- Modify: `README_algorithms.md`

- [ ] **Step 1: Write README_HDR.md**

Full Chinese documentation following the structure from the spec (Section "文档 (README_HDR.md)"):

1. 项目简介
2. 目录结构 (完整文件树)
3. 算法原理详解 — 每个子模块一个章节:
   - 图像对齐 (MTB + 特征点) — 简明数学公式 + 流程
   - 相机响应函数标定 (Debevec + Robertson) — 公式 + 对比
   - HDR 辐射图合并 — 加权合并公式
   - 色调映射 — 分全局/局部/感知三类，每种算法有公式 + 适用场景
   - 多曝光融合 (Mertens) — 权重计算 + 金字塔融合
   - 单张图像 HDR — CLAHE + 细节增强
4. 算法对比总结表 — 10 种色调映射算法的类型、适用场景、优缺点
5. 快速开始:
   - 安装依赖: `pip install -r requirements.txt`
   - 合成数据演示: `python demo_hdr.py --test`
   - 色调映射对比: `python demo_hdr.py --compare`
   - 自定义输入示例
   - 代码调用示例 (HDRPipeline)
6. 参数调优指南 — 关键参数说明与推荐值
7. 参考文献

- [ ] **Step 2: Add link in README_algorithms.md**

At the end of `README_algorithms.md` (before the Sources/references section), add:

```markdown
---

## HDR 高动态范围成像

HDR 成像算法的完整文档请参见 [README_HDR.md](README_HDR.md)。
```

- [ ] **Step 3: Commit**

```bash
git add README_HDR.md README_algorithms.md
git commit -m "docs(hdr): add comprehensive HDR documentation and cross-reference"
```

---

## Task 16: Final Integration Verification

**Files:** None (verification only)

- [ ] **Step 1: Verify full import**

Run: `python -c "
import hdr_imaging
from hdr_imaging import (HDRPipeline, HDRResult,
    MTBAlignment, FeatureAlignment,
    DebevecCalibration, RobertsonCalibration,
    HDRMerge,
    ReinhardGlobal, ReinhardLocal, DragoToneMap, DurandToneMap,
    FattalToneMap, AdaptiveLog, ACESToneMap, FilmicToneMap,
    MantiukToneMap, HistogramToneMap,
    MertensFusion, SingleImageHDR)
print(f'All {len(hdr_imaging.__all__)} exports available')
print(f'Version: {hdr_imaging.__version__}')
"`

Expected: "All 20 exports available" (19 classes + HDRResult), version 1.0.0

- [ ] **Step 2: Run demo --test end-to-end**

Run: `python demo_hdr.py --test`
Expected: completes without error, output_hdr/ contains saved images.

- [ ] **Step 3: Run demo --compare end-to-end**

Run: `python demo_hdr.py --compare`
Expected: completes without error, generates tone mapping comparison grid.

- [ ] **Step 4: Verify error handling**

Run: `python -c "
from hdr_imaging import HDRPipeline
import numpy as np

# Test invalid tone mapping method
try:
    HDRPipeline(tone_mapping_method='invalid')
except ValueError as e:
    print(f'Caught: {e}')

# Test insufficient images
try:
    p = HDRPipeline()
    p.process([np.zeros((10,10,3), dtype=np.uint8)], [1.0])
except ValueError as e:
    print(f'Caught: {e}')

# Test mismatched lengths
try:
    p = HDRPipeline()
    imgs = [np.zeros((10,10,3), dtype=np.uint8)] * 3
    p.process(imgs, [1.0, 2.0])
except ValueError as e:
    print(f'Caught: {e}')

print('Error handling OK')
"`

Expected: all three errors caught, prints "Error handling OK".

- [ ] **Step 5: Final commit (if any fixes needed)**

Only if fixes were required during verification:
```bash
git add -A
git commit -m "fix(hdr): address issues found during integration verification"
```
