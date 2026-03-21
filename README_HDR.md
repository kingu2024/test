# HDR 高动态范围成像算法库

> 基于 2024-2025 最新方法，融合经典算法与现代优化技术，附完整中文注释与数学公式推导

---

## 目录

- [项目简介](#项目简介)
- [目录结构](#目录结构)
- [算法原理详解](#算法原理详解)
  - [1. 图像对齐 (Alignment)](#1-图像对齐-alignment)
  - [2. 相机响应函数标定 (Camera Response Function)](#2-相机响应函数标定-camera-response-function)
  - [3. HDR 辐射图合并](#3-hdr-辐射图合并)
  - [4. 色调映射 (Tone Mapping)](#4-色调映射-tone-mapping)
  - [5. 多曝光融合 (Exposure Fusion)](#5-多曝光融合-exposure-fusion)
  - [6. 单张图像 HDR 增强](#6-单张图像-hdr-增强)
- [快速开始](#快速开始)
- [参数调优指南](#参数调优指南)
- [参考文献](#参考文献)

---

## 项目简介

### HDR 成像概述

高动态范围（HDR, High Dynamic Range）成像技术旨在捕获和显示现实世界中远超普通相机传感器所能记录的亮度范围。自然场景的动态范围可达 5 个数量级（10⁵:1），而普通显示器仅能呈现约 2–3 个数量级（10²–10³:1）。HDR 技术通过多曝光合并、色调映射等手段弥合这一差距。

### 本模块功能

本模块提供完整的 HDR 成像算法实现，涵盖以下六大核心功能：

| 功能 | 描述 |
|------|------|
| **多曝光对齐** | MTB 中值阈值位图对齐 / 特征点对齐，消除手持拍摄带来的帧间位移 |
| **CRF 标定** | Debevec 线性系统法 / Robertson EM 迭代法，恢复相机响应函数 |
| **HDR 辐射图重建** | 加权合并多曝光图像，构建真实物理辐照度图 |
| **10 种色调映射** | 全局 / 局部 / 感知驱动三类共 10 种算子，将 HDR 映射到可显示 LDR |
| **多曝光融合** | Mertens 算法，无需 CRF 标定直接融合出高质量 LDR 结果 |
| **单张 HDR 增强** | CLAHE + 多尺度细节增强，从单张 LDR 模拟 HDR 视觉效果 |

### 混合实现策略

每个子模块均提供两种实现路径：

- **手动 NumPy 实现** (`process()`)：完整按照论文公式实现，附详细中文注释和数学推导，适合学习与研究
- **OpenCV 参考实现** (`process_opencv()`)：调用 OpenCV 内置算法，可作为性能基准或生产环境选择

---

## 目录结构

```
hdr_imaging/                     # HDR 成像算法包
├── __init__.py                  # 模块导出（HDRPipeline 及各子类）
├── hdr_pipeline.py              # HDR 处理主控流水线
│
├── alignment/                   # 图像对齐子模块
│   ├── __init__.py
│   ├── mtb_alignment.py         # MTB 中值阈值位图对齐（Ward 2003）
│   └── feature_alignment.py     # 特征点对齐（复用 panorama_stitching）
│
├── calibration/                 # 相机响应函数标定子模块
│   ├── __init__.py
│   ├── debevec.py               # Debevec & Malik 1997 线性系统法
│   └── robertson.py             # Robertson et al. 2003 EM 迭代法
│
├── merge/                       # HDR 辐射图合并子模块
│   ├── __init__.py
│   └── hdr_merge.py             # 加权合并多曝光图像
│
├── tone_mapping/                # 色调映射子模块（10 种算子）
│   ├── __init__.py
│   ├── global_operators.py      # 全局算子：Reinhard全局 / Drago / 自适应对数
│   ├── local_operators.py       # 局部算子：Reinhard局部 / Durand / Fattal
│   └── perceptual_operators.py  # 感知算子：ACES / Filmic / Mantiuk / 直方图
│
├── exposure_fusion/             # 多曝光融合子模块
│   ├── __init__.py
│   └── mertens.py               # Mertens 2007 拉普拉斯金字塔融合
│
└── single_image/                # 单张图像 HDR 增强子模块
    ├── __init__.py
    └── single_image_hdr.py      # CLAHE + 多尺度细节增强

demo_hdr.py                      # HDR 演示脚本（--test / --compare / --single）
requirements.txt                 # 依赖包列表
```

---

## 算法原理详解

### 完整 HDR 流程

```
多曝光输入图像 [Δt₁, Δt₂, ..., Δtₙ]
    ↓
[Step 1] 图像对齐
         MTB 金字塔：二值位图 XOR 差异最小化
         特征点法：SIFT/ORB 特征匹配 + 单应性变换
    ↓
[Step 2] 相机响应函数标定 (CRF)
         Debevec：构建线性系统 A·x = b，SVD 求解 g(z)
         Robertson：EM 迭代交替估计 E 和 g(z)
    ↓
[Step 3] HDR 辐射图合并
         ln(E_i) = Σ w(Z_ij)·[g(Z_ij) - ln(Δtⱼ)] / Σ w(Z_ij)
         E = exp(ln(E))，单位：辐照度 (float32)
    ↓
[Step 4] 色调映射 → LDR 输出
         全局 / 局部 / 感知驱动，共 10 种算子
    ↓
LDR 图像输出 (uint8, BGR)

替代路径 A（无 CRF 标定）：
    多曝光图像 → 对齐 → Mertens 融合 → LDR 输出

替代路径 B（单张图像）：
    单张 LDR → CLAHE → 多尺度细节增强 → 输出
```

---

### 1. 图像对齐 (Alignment)

多曝光拍摄中，帧间存在几像素至数十像素的位移误差，对齐是 HDR 重建的前提。

#### 1.1 MTB 中值阈值位图对齐

**参考文献**：Ward, "Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Hand-Held Exposures." JGT 2003.

**核心思想**：将图像转化为二值位图（以中值亮度为阈值），二值化结果对曝光变化天然不敏感，可直接比较不同曝光图像的结构信息。

**核心公式**

```
中值阈值位图 (MTB):
    MTB(x,y) = 1   if I(x,y) > median(I)
             = 0   otherwise

排除位图 (EB):
    EB(x,y)  = 1   if |I(x,y) - median(I)| >= threshold
             = 0   otherwise   （接近中值的不稳定像素被排除）

位移差异度量:
    diff = (MTB₁_shifted XOR MTB₂) AND (EB₁_shifted AND EB₂)
    error = countNonZero(diff)   （异或后1的个数，即不匹配像素数）
```

**金字塔粗到细搜索流程**

```
构建 max_level 层图像金字塔（每层 1/2 下采样）
    ↓
从最粗层（第 max_level 层）开始搜索
每层在 [-1, 0, 1] × [-1, 0, 1] 共 9 个候选位移中选最优
    ↓
逐层细化，位移累积：shift_total = shift_coarser × 2 + shift_current
    ↓
在原始分辨率上用 warpAffine 施加最终平移
```

**关键参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_level` | 5 | 金字塔层数，最大可校正位移约 2^max_level 像素 |
| `threshold` | 4 | 排除位图容差，值域 [0, 255] |

#### 1.2 基于特征点的对齐

复用 `panorama_stitching` 模块中的特征提取与单应性估计（见 `feature_alignment.py`），适用于旋转较大或包含视角变化的多曝光序列。

---

### 2. 相机响应函数标定 (Camera Response Function)

相机响应函数 (CRF) 描述了传感器从场景辐照度到像素值的非线性映射关系。标定 CRF 是精确重建 HDR 辐射图的关键步骤。

**基础成像模型**

```
Z = f(E · Δt)

Z   ∈ [0, 255]  — 像素值
E               — 场景辐照度（irradiance），待恢复的物理量
Δt              — 曝光时间（shutter speed）
f(·)            — 相机响应函数（单调递增）

定义对数反函数：
    g(Z) = ln(f⁻¹(Z)) = ln(E) + ln(Δt)
```

#### 2.1 Debevec & Malik 方法 (1997)

**参考文献**：Debevec & Malik, "Recovering High Dynamic Range Radiance Maps from Photographs." SIGGRAPH 1997.

**线性系统构建**

设有 N 张图像，采样 P 个像素位置，未知数向量 x 长度为 256 + P：

```
x = [g(0), g(1), ..., g(255),  ln(E₁), ln(E₂), ..., ln(E_P)]

方程组：
  数据拟合方程（P × N 条）：
      w(Z_ij) · [g(Z_ij) - ln(E_i) - ln(Δtⱼ)] = 0

  平滑约束（254 条，z = 1..254）：
      λ · w(z) · [g(z-1) - 2·g(z) + g(z+1)] = 0

  固定约束（1 条，消除尺度不确定性）：
      g(128) = 0

矩阵形式：A · x = b   （用 np.linalg.lstsq 最小二乘求解）
```

**三角帽形权重函数**

```
w(z) = z - 0          if z <= 127.5   （即 z ≤ 127）
     = 255 - z        if z > 127.5    （即 z ≥ 128）

特殊值：w(0) = 0，w(128) = 127，w(255) = 0
```

权重设计使得过曝（z≈255）和欠曝（z≈0）像素贡献降低，减少量化误差和噪声影响。

**关键参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `samples` | 50 | 随机采样像素点数，推荐 20–200 |
| `lambda_smooth` | 10.0 | 平滑约束权重 λ，越大曲线越平滑 |

#### 2.2 Robertson 迭代方法 (2003)

**参考文献**：Robertson et al., "Estimation-Theoretic Approach to Dynamic Range Enhancement Using Multiple Exposures." JEI 2003.

**EM 迭代框架**

```
初始化：g(z) = (z + 1) / 256   （线性假设）

E-step — 估计辐照度（固定 g，更新 E）：

    E_i = Σ_j [w(Z_ij) · g(Z_ij) / Δtⱼ]
          ─────────────────────────────────
          Σ_j [w(Z_ij) · g(Z_ij)² / Δtⱼ²]

M-step — 更新响应函数（固定 E，更新 g）：

    g(z) = Σ_{(i,j): Z_ij = z} (E_i · Δtⱼ) / count_z

归一化：g = g / g[128]   （消除尺度不确定性）

收敛条件：max_z |g_new(z) - g_old(z)| < epsilon
```

**高斯帽形权重函数**

```
w(z) = exp(-4 · ((z - 127.5) / 127.5)²)

特殊值：w(0) ≈ 0.018，w(127) ≈ 1.0，w(255) ≈ 0.018
```

相比 Debevec 的三角帽，高斯帽在边界处权重不为零，保留了更多极端像素的信息。

**关键参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_iter` | 30 | 最大 EM 迭代次数 |
| `epsilon` | 1e-3 | 收敛阈值，越小精度越高 |

---

### 3. HDR 辐射图合并

#### 加权合并公式

```
ln(E_i) = Σ_j w(Z_ij) · (g(Z_ij) - ln(Δtⱼ))
          ──────────────────────────────────────
                    Σ_j w(Z_ij)

E_i = exp(ln(E_i))   （线性辐照度，单位 float32）
```

其中：
- `E_i`：像素位置 i 的辐照度估计
- `Z_ij`：像素 i 在图像 j 中的像素值
- `g(Z)`：由 Debevec 或 Robertson 标定的 CRF 对数曲线
- `Δtⱼ`：图像 j 的曝光时间

#### 权重函数

```
w(z) = z + 1       if z <= 127   （极暗区仍有小权重贡献）
w(z) = 256 - z     if z > 127    （过饱和区权重趋近零）

特殊值：w(0) = 1，w(127) = 128，w(128) = 128，w(255) = 1
```

#### 全饱和像素退化处理

当某像素在所有曝光下均饱和（分母 Σw = 0）时，使用曝光时间最接近几何均值的图像直接估计：

```
t_geom = exp(mean(ln(Δtⱼ)))
j* = argmin |Δtⱼ - t_geom|
ln(E_i) = g(Z_{i,j*}) - ln(Δt_{j*})
```

---

### 4. 色调映射 (Tone Mapping)

色调映射将 HDR 辐射图（动态范围可达 10⁵:1）压缩到显示器可呈现的 LDR 范围 [0, 1]，同时尽可能保留视觉感知的细节与对比度。

#### 亮度提取（通用）

所有算子均使用 ITU-R BT.709 线性亮度公式：

```
L = 0.0722·B + 0.7152·G + 0.2126·R

彩色恢复：C_out = C_in · (L_mapped / (L_original + ε))
```

---

#### 4.1 全局算子

全局算子对所有像素统一应用相同的映射函数，计算速度快，适合整体亮度较均衡的场景。

##### Reinhard 全局 (2002)

模拟人眼感知的经典算子，目前仍是最广泛使用的基准方法之一。

```
1. 对数平均亮度（感知亮度中心）：
       L̄ = exp((1/N) · Σ ln(δ + L(x,y)))   （δ = 1e-6）

2. 亮度缩放到感知区间：
       L_scaled = (a / L̄) · L              （a = 0.18，"中等灰度"关键值）

3. 色调映射（含白点限制）：
       L_d = L_scaled · (1 + L_scaled / L_white²) / (1 + L_scaled)

4. 彩色恢复 + Gamma 校正 (γ = 2.2) → uint8
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `key_value` | 0.18 | 场景关键值 a，控制整体曝光感知 |
| `white_point` | None | 白点亮度；None 时使用场景最大亮度 |

##### Drago 自适应对数 (2003)

通过偏差参数 b 自适应调整对数基底，均衡保留亮部和暗部细节。

```
L_d = (L_d_max / log₁₀(1 + L_max))
      · log(1 + L)
      / log(2 + 8·(L/L_max)^(log(b)/log(0.5)))

b   — 偏差参数，默认 0.85
L_d_max = 100 cd/m²（显示器最大亮度）
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bias` | 0.85 | 偏差参数，范围 (0, 1)，越大对数基底越大 |
| `gamma` | 2.2 | Gamma 校正值 |

##### 自适应对数 (AdaptiveLog)

以场景对数平均亮度 L_avg 作为参考自适应调整映射范围，使中等亮度区域获得最大对比度保留。无对应 OpenCV 实现。

```
L_d = L_d_max · log(1 + L/L_avg) / log(1 + L_max/L_avg)

L_avg — 场景对数平均亮度（感知亮度中心）
L_max — 场景最大亮度
```

---

#### 4.2 局部算子

局部算子根据每个像素的空间邻域动态调整映射参数，能更好保留局部对比度，但计算代价更高。

##### Reinhard 局部 (2002)

使用多尺度高斯滤波估计局部适应亮度，自适应选择每个像素最佳感知尺度。

```
1. 对数平均亮度归一化：L_scaled = (a / L̄) · L

2. 多尺度高斯滤波（尺度 σ = 1.6^s，s = 1..num_scales）：
       V1(x,y,s) = L_scaled * G(x,y,σ_s)

3. 活度量（决定最佳感知尺度）：
       act(x,y,s) = |V1(s) - V1(s+1)| / (2^φ · a/s² + V1(s))
   选最大 s 使得 act < ε

4. 局部色调映射：
       L_d(x,y) = L_scaled / (1 + V1_best(x,y))
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `key_value` | 0.18 | 场景关键值 |
| `phi` | 8.0 | 活度量指数参数 |
| `num_scales` | 8 | 高斯滤波尺度数 |
| `epsilon` | 0.05 | 活度量阈值 |

##### Durand 双边滤波 (2002)

将图像分解为"基础层"（低频亮度）和"细节层"（高频局部细节），仅压缩基础层的动态范围。

```
1. 对数变换：L_log = log₁₀(L + ε)

2. 双边滤波分离：
       base   = bilateralFilter(L_log, σ_spatial, σ_range)
       detail = L_log - base

3. 压缩基础层到 target_contrast 范围：
       base_compressed = (base - max(base)) · target_contrast / (max(base) - min(base))

4. 重构并还原：
       L_out = 10^(base_compressed + detail)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sigma_spatial` | 2.0 | 双边滤波空间域标准差 |
| `sigma_range` | 2.0 | 双边滤波值域标准差 |
| `target_contrast` | 5.0 | 基础层目标动态范围（对数域） |

##### Fattal 梯度域 (2002)

在梯度域衰减大梯度、保留小梯度，通过泊松方程重建压缩后的对数亮度图。

```
1. 对数域：H = log(L + ε)
2. 计算梯度：∇H = (∂H/∂x, ∂H/∂y)
3. 梯度幅值：|∇H| = √((∂H/∂x)² + (∂H/∂y)²)

4. 衰减函数 φ（衰减大梯度）：
       φ(x,y) = (α / |∇H|) · (|∇H| / α)^β
              = α^(1-β) · |∇H|^(β-1)
   α 控制衰减中心，β ∈ (0,1) 控制衰减强度（越小越强）

5. 衰减后梯度场：
       Gx = φ · ∂H/∂x，  Gy = φ · ∂H/∂y

6. 求散度：div(G) = ∂Gx/∂x + ∂Gy/∂y

7. 泊松方程重建（稀疏线性系统）：
       ∇²I = div(G)   →   scipy.sparse.linalg.spsolve
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | 0.1 | 梯度衰减中心值 |
| `beta` | 0.8 | 衰减指数，越小压缩越强 |
| `max_size` | 2048 | 超过此尺寸自动 0.5x 降采样求解 |

---

#### 4.3 感知驱动算子

感知算子以人类视觉系统（HVS）模型为驱动，注重视觉真实感和艺术风格，广泛应用于电影工业和游戏渲染管线。

##### ACES 电影色调映射 (Narkowicz 2015)

ACES（Academy Color Encoding System）是电影工业标准色彩管线中的标准色调映射曲线。

```
f(x) = (x·(2.51x + 0.03)) / (x·(2.43x + 0.59) + 0.14)

操作流程：
    1. 乘以曝光系数：hdr = hdr_image × exposure
    2. 对每个 BGR 通道直接应用 ACES 有理函数曲线
    3. Gamma 校正 (γ = 2.2) → uint8
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exposure` | 1.0 | 曝光系数，控制整体亮度 |
| `gamma` | 2.2 | Gamma 校正值 |

##### Filmic 色调映射 (Hable 2010)

John Hable 为游戏《神秘海域 2》提出的分段胶片响应曲线，成为游戏行业广泛使用的标准。

```
f(x) = ((x·(Ax + CB) + DE) / (x·(Ax + B) + DF)) - E/F

标准参数：
    A = 0.15（肩部强度）  B = 0.50（线性强度）
    C = 0.10（线性角度）  D = 0.20（趾部强度）
    E = 0.02（趾部数值）  F = 0.30（趾部分母）

归一化：result = f(exposure × color) / f(white_point)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exposure` | 2.0 | 曝光系数（Hable 推荐值） |
| `white_point` | 11.2 | 白点亮度，用于归一化 |

##### Mantiuk 感知对比度映射 (2006)

在对数亮度域中进行局部对比度均衡化，模拟人眼对对比度的非线性感知（Weber-Fechner 定律）。

```
1. 对数亮度：log_L = log(L + ε)
2. 局部均值：local_mean = GaussianBlur(log_L, σ=2.0)
3. 局部对比度：contrast = log_L - local_mean
4. 对比度压缩：log_L_new = local_mean + scale × contrast
5. 还原：L_new = exp(log_L_new)，归一化到 [0, 1]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scale` | 0.85 | 对比度压缩系数，越小越平坦 |
| `saturation` | 1.0 | 色彩饱和度系数 |

##### 直方图均衡化色调映射 (HistogramToneMap)

通过 HDR 亮度对数域的累积分布函数（CDF）将亮度重映射到 [0, 1]，自适应充分利用显示动态范围。

```
1. 对数变换：log_L = log(L + ε)
2. 计算 log_L 直方图（num_bins 个区间）
3. 若 clip_limit > 0，截断直方图峰值（抑制过度对比度增强，类 CLAHE 概念）
4. 计算归一化 CDF：
       cdf_normalized = (cdf - cdf_min) / (cdf_max - cdf_min)
5. 用 CDF 将每个像素的 log_L 映射到 [0, 1]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_limit` | 0.0 | 直方图截断阈值；0 表示不截断 |
| `num_bins` | 256 | 直方图区间数 |

---

#### 4.4 算法对比总结表

| 算法 | 类型 | 适用场景 | 优点 | 缺点 |
|------|------|---------|------|------|
| `reinhard_global` | 全局 | 通用基准，光线均匀场景 | 实现简单，结果自然 | 局部对比度损失 |
| `drago` | 全局 | 室外高反差场景 | 亮暗均衡，细节保留好 | 偏好参数依赖场景 |
| `adaptive_log` | 全局 | 动态范围适中场景 | 自适应，无需调参 | 极高动态范围下效果一般 |
| `reinhard_local` | 局部 | 高细节复杂场景 | 局部对比度优秀 | 计算较慢，可能有光晕 |
| `durand` | 局部 | 纹理丰富自然场景 | 细节/基础层分离精准 | 双边滤波参数敏感 |
| `fattal` | 局部 | 艺术风格细节增强 | 梯度衰减独特效果 | 最慢，大图像需降采样 |
| `aces` | 感知 | 电影级色调，自然场景 | 工业标准，肩部柔和 | 对低动态范围图像偏亮 |
| `filmic` | 感知 | 游戏渲染，写实风格 | 视觉对比度好，胶片感 | 暗部细节略有损失 |
| `mantiuk` | 感知 | 高对比度局部均衡 | Weber-Fechner 感知建模 | 可能引入伪影 |
| `histogram` | 感知 | 自适应对比度增强 | 自适应，充分利用动态范围 | 可能过度增强噪声 |

---

### 5. 多曝光融合 (Exposure Fusion)

**参考文献**：Mertens et al., "Exposure Fusion." PG 2007.

Mertens 算法直接从多张 LDR 图像融合出高质量结果，无需 HDR 辐射图重建或 CRF 标定，是一种轻量级的 HDR 替代方案。

#### Mertens 算法原理

**质量度量计算**（对每张图像分别计算）

```
对比度 C(x,y) = |Laplacian(gray)|   （拉普拉斯滤波绝对值）

饱和度 S(x,y) = std(R, G, B)        （RGB 三通道标准差）

曝光度 E(x,y) = Π_c exp(-0.5 · ((I_c - 0.5) / σ)²)
               （各通道与理想曝光值 0.5 的偏差，σ = 0.2）

组合权重（带 ε 防止全零）：
    W_k = C_k^wc · S_k^ws · E_k^we + ε
```

**归一化**

```
Ŵ_k = W_k / Σ_k W_k   （跨图像归一化，确保各像素权重之和为 1）
```

#### 拉普拉斯金字塔融合

```
构建：Lₖ = Gₖ - EXPAND(Gₖ₊₁)   （每张图像的拉普拉斯金字塔）
构建：G(Ŵ_k)                    （每张权重图的高斯金字塔）

逐层融合：
    L_fused[l] = Σ_k G(Ŵ_k)[l] · L_k[l]

重建：Ĝₖ₋₁ = Lₖ₋₁ + EXPAND(Ĝₖ)  （从融合金字塔自底向上重建）
```

**关键参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `contrast_weight` | 1.0 | 对比度权重指数 wc |
| `saturation_weight` | 1.0 | 饱和度权重指数 ws |
| `exposure_weight` | 1.0 | 曝光度权重指数 we |
| `sigma` | 0.2 | 曝光度高斯函数标准差 |
| `pyramid_levels` | None | 金字塔层数；None 时自动计算 floor(log2(min(H,W)))-1 |

---

### 6. 单张图像 HDR 增强

当无法获取多曝光序列时，可对单张 LDR 图像进行 HDR 风格增强。

#### CLAHE 原理

CLAHE（Contrast Limited Adaptive Histogram Equalization，对比度受限自适应直方图均衡化）在 LAB 色彩空间的 L 通道上工作，保持 a、b 色度通道不变以避免色彩失真。

```
处理流程：
    BGR → LAB → 提取 L 通道
        ↓
    将 L 分割为 tile_size × tile_size 个块
    每块独立做直方图均衡化：
        clip_threshold = clip_limit × (tile_pixels / 256)
        超出阈值的频数截断并均匀重分配
        计算 CDF，归一化为 [0, 255] 映射表
        ↓
    双线性插值消除块边界接缝
        ↓
    合并增强后的 L 通道 → LAB → BGR
```

#### 多尺度细节增强

在三个高斯尺度上提取高频细节并加权叠加：

```
detail_k = L - GaussianBlur(L, σ = σ_k)   （σ ∈ {1, 2, 4}）

L_enhanced = L + detail_boost × (Σ detail_k) / 3
```

**关键参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_limit` | 3.0 | CLAHE 对比度限制阈值，推荐 1.0–10.0 |
| `tile_size` | 8 | CLAHE 分块大小，推荐 4–16 |
| `detail_boost` | 1.5 | 多尺度细节增强强度，推荐 0.5–3.0 |

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 使用合成测试数据（自动生成4张曝光图像，无需真实输入）
python demo_hdr.py --test

# 色调映射算法对比（生成全部10种算法的2×5对比网格图）
python demo_hdr.py --compare

# 使用真实多曝光图像（需提供图像路径和曝光时间）
python demo_hdr.py --images img1.jpg img2.jpg img3.jpg --exposures 0.033 0.25 1.0

# 单张图像 HDR 增强
python demo_hdr.py --single input.jpg

# 指定输出目录
python demo_hdr.py --test --output_dir my_results/
```

### 代码调用示例

```python
from hdr_imaging import HDRPipeline

# 完整 HDR 流程（Debevec 标定 + Reinhard 全局色调映射）
pipeline = HDRPipeline(
    align_method='mtb',               # 对齐：'mtb' 或 'feature'
    calibration_method='debevec',     # CRF 标定：'debevec' 或 'robertson'
    tone_mapping_method='reinhard_global',  # 色调映射：10 种可选
    use_opencv=False                  # True 使用 OpenCV 实现
)
result = pipeline.process(images, exposure_times)
# result.ldr_result      — uint8 BGR LDR 图像
# result.hdr_radiance_map — float32 HDR 辐射图
# result.response_curve  — (256, 3) CRF 对数曲线

# 多曝光融合（Mertens，无需 CRF 标定）
fused = pipeline.exposure_fusion(images)  # 返回 uint8 BGR

# 单张图像 HDR 增强（CLAHE + 多尺度细节增强）
enhanced = pipeline.single_image_hdr(image)  # 返回 uint8 BGR
```

**直接使用子模块**

```python
from hdr_imaging import (MTBAlignment, DebevecCalibration,
                          HDRMerge, ReinhardGlobal, MertensFusion,
                          SingleImageHDR)

# 对齐
aligner = MTBAlignment(max_level=5, threshold=4)
aligned = aligner.process(images)

# CRF 标定
calibrator = DebevecCalibration(samples=50, lambda_smooth=10.0)
response_curve = calibrator.process(aligned, exposure_times)

# HDR 合并
merger = HDRMerge()
hdr_map = merger.process(aligned, exposure_times, response_curve)

# 色调映射
tone_mapper = ReinhardGlobal(key_value=0.18)
ldr = tone_mapper.process(hdr_map)
```

---

## 参数调优指南

### 图像对齐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `align_method` | `mtb` | 手持拍摄首选；场景旋转大时用 `feature` |
| `max_level` | 5 | 金字塔层数，可校正最大位移约 32 像素；抖动大时增大 |
| `threshold` | 4 | 排除位图容差，噪声大时可适当增大至 8–16 |

### CRF 标定

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `calibration_method` | `debevec` | 精度更高；曝光数少（2–3 张）时用 `robertson` |
| `samples` (Debevec) | 50–100 | 越多越精准，但速度越慢 |
| `lambda_smooth` (Debevec) | 10–50 | 越大曲线越平滑；噪声大时增大 |
| `max_iter` (Robertson) | 30 | 通常 10–20 次内收敛 |
| `epsilon` (Robertson) | 1e-3 | 越小精度越高；1e-4 以下收益有限 |

### 色调映射

| 算法 | 关键参数 | 推荐值 | 说明 |
|------|---------|--------|------|
| `reinhard_global` | `key_value` | 0.18 | 降低可使图像整体变暗；升高变亮 |
| `drago` | `bias` | 0.75–0.95 | 越大亮部越亮；接近 0.5 时效果趋向对数映射 |
| `reinhard_local` | `num_scales` | 6–10 | 越多局部感知越精细，速度越慢 |
| `durand` | `target_contrast` | 3.0–8.0 | 越小压缩越强；5.0 适合大多数场景 |
| `fattal` | `beta` | 0.7–0.9 | 越小梯度衰减越强，动态范围压缩越大 |
| `aces` | `exposure` | 0.5–2.0 | 依场景亮度调整；过曝场景降低 |
| `mantiuk` | `scale` | 0.7–0.95 | 越小对比度越平坦，细节越均匀 |

### 多曝光融合（Mertens）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `contrast_weight` | 1.0 | 增大可强调边缘和纹理区域 |
| `saturation_weight` | 1.0 | 增大可偏好色彩饱和的曝光 |
| `exposure_weight` | 1.0 | 增大可偏好接近中灰的曝光 |

### 单张图像 HDR 增强

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `clip_limit` | 2.0–5.0 | 越大对比度增强越强；噪声大图像用 1.0–2.0 |
| `tile_size` | 8 | 越小局部自适应越细，但可能引入块效应 |
| `detail_boost` | 1.0–2.5 | 越大细节越锐利；3.0 以上可能产生噪声放大 |

---

## 参考文献

### 图像对齐
- Ward, G. "Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Hand-Held Exposures." *Journal of Graphics Tools*, 8(2):17-30, 2003. — MTB 算法原始论文

### 相机响应函数标定
- Debevec, P.E. & Malik, J. "Recovering High Dynamic Range Radiance Maps from Photographs." *ACM SIGGRAPH 1997*, pp. 369–378. — CRF 线性系统求解方法
- Robertson, M.A., Borman, S. & Stevenson, R.L. "Estimation-Theoretic Approach to Dynamic Range Enhancement Using Multiple Exposures." *Journal of Electronic Imaging*, 12(2):219–228, 2003. — CRF EM 迭代方法

### 色调映射
- Reinhard, E. et al. "Photographic Tone Reproduction for Digital Images." *ACM SIGGRAPH 2002*, pp. 267–276. — Reinhard 全局/局部算子
- Drago, F. et al. "Adaptive Logarithmic Mapping for Displaying High Contrast Scenes." *Eurographics 2003*. — Drago 自适应对数映射
- Durand, F. & Dorsey, J. "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images." *ACM SIGGRAPH 2002*. — 双边滤波分解
- Fattal, R., Lischinski, D. & Werman, M. "Gradient Domain High Dynamic Range Compression." *ACM SIGGRAPH 2002*. — 梯度域泊松重建
- Narkowicz, K. "ACES Filmic Tone Mapping Curve." 2015. — ACES 有理函数近似
- Hable, J. "Filmic Tonemapping Operators." *GDC 2010*. — Uncharted 2 Filmic 曲线
- Mantiuk, R. et al. "Perceptual Framework for Contrast Processing of High Dynamic Range Images." *ACM Transactions on Applied Perception*, 3(3):286–308, 2006.

### 多曝光融合
- Mertens, T., Kautz, J. & Van Reeth, F. "Exposure Fusion." *Pacific Graphics 2007*. — 无 HDR 辐射图的直接融合方法

### 综述与最新进展
- Banterle, F. et al. *Advanced High Dynamic Range Imaging*, 2nd Edition. CRC Press, 2017.
- [HDR Image Survey (IEEE 2024)](https://ieeexplore.ieee.org/document/10371582) — 深度学习 HDR 成像综述
