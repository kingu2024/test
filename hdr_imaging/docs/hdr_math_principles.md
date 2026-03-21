# HDR 高动态范围成像 — 数学原理

本文档详细介绍高动态范围（HDR）成像的数学原理，涵盖图像对齐、相机响应函数标定、HDR 辐射图合并、色调映射、多曝光融合及单张图像 HDR 增强等核心算法。

---

## 目录

1. [HDR 成像基础](#1-hdr-成像基础)
2. [图像对齐 (Alignment)](#2-图像对齐-alignment)
3. [相机响应函数 (Camera Response Function)](#3-相机响应函数-camera-response-function)
4. [HDR 辐射图合并](#4-hdr-辐射图合并)
5. [色调映射 (Tone Mapping)](#5-色调映射-tone-mapping)
6. [多曝光融合 (Mertens 2007)](#6-多曝光融合-mertens-2007)
7. [单张图像 HDR](#7-单张图像-hdr)
8. [参考文献](#8-参考文献)

---

## 1. HDR 成像基础

### 1.1 动态范围定义

**动态范围**（Dynamic Range, DR）定义为场景中最大亮度与最小可感知亮度之比：

$$
DR = \frac{L_{max}}{L_{min}}
$$

通常以对数形式（单位：EV，Exposure Value）或 $\log_{10}$ 形式表示：

$$
DR_{dB} = 20 \cdot \log_{10}\!\left(\frac{L_{max}}{L_{min}}\right)
$$

$$
DR_{EV} = \log_2\!\left(\frac{L_{max}}{L_{min}}\right)
$$

### 1.2 人眼与传感器的动态范围对比

| 系统 | 动态范围（数量级） | EV 范围 |
|------|-------------------|---------|
| 人眼（瞬时适应） | $\sim 10^4$ | $\sim 14\,\text{EV}$ |
| 人眼（全范围适应） | $\sim 10^{14}$ | $\sim 46\,\text{EV}$ |
| 相机传感器（标准） | $\sim 10^3$ | $\sim 10\,\text{EV}$ |
| HDR 显示器 | $\sim 10^4$ | $\sim 14\,\text{EV}$ |

由于标准相机传感器的动态范围远低于自然场景，单张曝光图像往往在高光区域过曝或阴影区域欠曝。HDR 成像通过合并多张不同曝光的图像来克服这一限制。

### 1.3 HDR 成像流程概述

完整 HDR 成像流程分为以下步骤：

1. **多曝光拍摄**：以不同曝光时间 $\Delta t_1, \Delta t_2, \ldots, \Delta t_N$ 拍摄同一场景，相邻曝光通常相差 1–2 EV。
2. **图像对齐**：消除手持拍摄或运动场景引起的帧间位移。
3. **相机响应函数（CRF）标定**：恢复从辐射量 $E$ 到像素值 $Z$ 的非线性映射关系。
4. **HDR 辐射图合并**：利用 CRF 和曝光参数，从多张 LDR 图像重建线性辐射图。
5. **色调映射**：将高动态范围辐射图压缩至显示设备可显示的低动态范围。

---

## 2. 图像对齐 (Alignment)

### 2.1 MTB 中值阈值位图对齐 (Ward 2003)

MTB（Median Threshold Bitmap）算法是一种计算效率极高的图像对齐方法，无需计算梯度，仅依赖二值位图的异或（XOR）运算。

#### 2.1.1 中值阈值位图构造

对于灰度图像 $I$，设其像素灰度中值为 $\text{median}(I)$，则**中值阈值位图**定义为：

$$
\text{MTB}(x,y) = \begin{cases} 1 & \text{if } I(x,y) > \text{median}(I) \\ 0 & \text{otherwise} \end{cases}
$$

为了减少中值附近噪声像素的干扰，定义**排除位图**（Exclusion Bitmap），标记灰度值接近中值的像素：

$$
\text{EB}(x,y) = \begin{cases} 0 & \text{if } |I(x,y) - \text{median}(I)| < \tau \\ 1 & \text{otherwise} \end{cases}
$$

其中阈值 $\tau$ 通常取 4（灰度值范围 0–255）。$\text{EB}=0$ 的像素在位移计算中被排除。

#### 2.1.2 位移误差度量

两幅图像 MTB 之间、在位移 $\boldsymbol{\delta} = (\delta_x, \delta_y)$ 下的**差异度量**为：

$$
D(\boldsymbol{\delta}) = \sum_{x,y} \left[\text{MTB}_1(x,y) \oplus \text{MTB}_2(x+\delta_x, y+\delta_y)\right] \cdot \text{EB}_1(x,y) \cdot \text{EB}_2(x+\delta_x, y+\delta_y)
$$

其中 $\oplus$ 为逐像素异或运算。$D$ 越小表示两帧对齐越好。

#### 2.1.3 多尺度金字塔搜索

为提高效率并避免陷入局部最优，MTB 采用**高斯图像金字塔**进行由粗到精的搜索：

设共有 $L$ 层金字塔（第 0 层为原图，第 $L-1$ 层为最粗尺度）。

**搜索过程**（从最粗尺度到最细）：

1. 在当前尺度下，遍历 $3 \times 3$ 候选位移集合 $\{-1, 0, 1\}^2$（共 9 种），找到使 $D$ 最小的位移 $\boldsymbol{\delta}_l$。
2. 将当前层位移传播到下一层（更精细尺度），并换算到精细层坐标系：

$$
\boldsymbol{\delta}_{total} = \boldsymbol{\delta}_{prev} \times 2 + \boldsymbol{\delta}_{current}
$$

3. 重复直至第 0 层（原始分辨率），得到最终亚像素级精度的对齐位移。

#### 2.1.4 图像平移

得到位移 $\boldsymbol{\delta} = (\delta_x, \delta_y)$ 后，将目标图像做平移变换：

$$
I_{aligned}(x, y) = I_{src}(x + \delta_x,\; y + \delta_y)
$$

### 2.2 基于特征点的对齐

对于存在旋转或透视变形的情形，需要使用特征点匹配和单应性矩阵估计进行更一般的对齐。

#### 2.2.1 特征提取与匹配

常用特征：**SIFT**（尺度不变特征变换）或 **ORB**（快速旋转不变特征）。

特征匹配使用 **FLANN**（快速近似最近邻搜索）或 **暴力匹配（BF）**。对于 SIFT，采用 Lowe's ratio test 过滤误匹配：

$$
\frac{d_1}{d_2} < r_{thresh} \quad (r_{thresh} \approx 0.75)
$$

其中 $d_1, d_2$ 分别为最近邻和次近邻的特征描述子距离。

#### 2.2.2 RANSAC 鲁棒单应性估计

给定一组匹配点对 $\{(\mathbf{p}_i, \mathbf{p}'_i)\}$，RANSAC（随机采样一致性）迭代估计 3×3 **单应性矩阵** $\mathbf{H}$，满足：

$$
s \begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}
$$

每次迭代随机采样 4 组点对（DLT 算法最少需要 4 对点），计算 $\mathbf{H}$ 后统计内点数（重投影误差 $< \varepsilon$）：

$$
d_i = \left\| \mathbf{p}'_i - \frac{\mathbf{H}\,\mathbf{p}_i}{\mathbf{H}_{3\cdot}\,\mathbf{p}_i} \right\|_2 < \varepsilon
$$

迭代次数 $N$ 根据期望置信度 $p$ 和内点率 $\hat{w}$ 确定：

$$
N = \frac{\log(1 - p)}{\log(1 - \hat{w}^4)}
$$

#### 2.2.3 透视变换对齐

利用估计的 $\mathbf{H}$ 对图像进行**透视变换**（warpPerspective）：

$$
I_{aligned}(x', y') = I_{src}\!\left(\frac{h_{11}x' + h_{12}y' + h_{13}}{h_{31}x' + h_{32}y' + h_{33}},\; \frac{h_{21}x' + h_{22}y' + h_{23}}{h_{31}x' + h_{32}y' + h_{33}}\right)
$$

---

## 3. 相机响应函数 (Camera Response Function)

### 3.1 成像模型

相机传感器对场景辐射量的响应是非线性的。设：

- $E_i$：像素 $i$ 处的场景辐照度（Irradiance），单位 $\text{W/m}^2$
- $\Delta t_j$：第 $j$ 张图像的曝光时间（快门速度）
- $Z_{ij} \in [0, 255]$：像素 $i$ 在第 $j$ 张图像中的观测值

传感器模型为：

$$
Z_{ij} = f(E_i \cdot \Delta t_j)
$$

其中 $f(\cdot)$ 是单调非线性函数（相机响应函数，CRF）。由于 $f$ 通常未知，HDR 成像需要先标定 $f$。

取对数形式，定义**对数响应函数** $g = \ln(f^{-1})$：

$$
g(Z_{ij}) = \ln(f^{-1}(Z_{ij})) = \ln(E_i) + \ln(\Delta t_j)
$$

目标：从多曝光图像集合 $\{Z_{ij}\}$ 中，在无需先验知识的情况下**同时**恢复 $g(\cdot)$ 和各像素辐照度 $\{E_i\}$。

### 3.2 Debevec & Malik 方法 (1997)

Debevec 方法将 CRF 恢复转化为一个**过约束线性最小二乘**问题，通过 SVD 求解。

#### 3.2.1 问题建模

设有 $P$ 个采样像素、$N$ 张曝光图像，未知数为：

$$
\mathbf{x} = \underbrace{[g(0),\, g(1),\, \ldots,\, g(255)]}_{\text{256个CRF值}},\; \underbrace{[\ln E_1,\, \ln E_2,\, \ldots,\, \ln E_P]}_{\text{P个辐照度}}
$$

共 $256 + P$ 个未知数。

#### 3.2.2 目标函数

最小化目标函数：

$$
\mathcal{O} = \underbrace{\sum_{i=1}^{P}\sum_{j=1}^{N} \left\{w(Z_{ij})\left[g(Z_{ij}) - \ln E_i - \ln \Delta t_j\right]\right\}^2}_{\text{数据拟合项}} + \underbrace{\lambda \sum_{z=1}^{254} \left[w(z)\,g''(z)\right]^2}_{\text{平滑正则项}}
$$

其中：
- **数据拟合项**：要求 $g(Z_{ij}) = \ln E_i + \ln \Delta t_j$（理想情况下成立）
- **平滑正则项**：通过对 $g$ 的二阶差分施加惩罚，使 $g$ 曲线光滑。$g''(z) \approx g(z-1) - 2g(z) + g(z+1)$
- $\lambda$：平滑权重（典型值 $\lambda = 10$–$50$），平衡拟合精度与平滑度

#### 3.2.3 权重函数

**三角帽形**权重函数，抑制过曝和欠曝的像素：

$$
w(z) = \begin{cases}
z - Z_{min} & \text{if } z \leq \frac{Z_{min} + Z_{max}}{2} \\
Z_{max} - z & \text{if } z > \frac{Z_{min} + Z_{max}}{2}
\end{cases}
$$

对于 8 位图像（$Z_{min}=0$，$Z_{max}=255$）：

$$
w(z) = \begin{cases} z & z \leq 127 \\ 255 - z & z > 127 \end{cases}
$$

#### 3.2.4 线性系统构建与求解

将目标函数中的各项展开，整理为矩阵形式 $\mathbf{A}\mathbf{x} = \mathbf{b}$：

- 每个数据点 $(i, j)$ 贡献一行方程：

$$
w(Z_{ij}) \cdot g(Z_{ij}) - w(Z_{ij}) \cdot \ln E_i = w(Z_{ij}) \cdot \ln \Delta t_j
$$

- 平滑约束对每个内部灰度值 $z = 1, \ldots, 254$ 贡献一行：

$$
\lambda\, w(z) \cdot [g(z-1) - 2g(z) + g(z+1)] = 0
$$

- 固定约束消除尺度模糊（系统欠秩）：

$$
g(128) = 0
$$

矩阵 $\mathbf{A}$ 的行数为 $P \cdot N + 254 + 1$，使用**奇异值分解（SVD）**或最小二乘法求解最优解 $\mathbf{x}^*$。

### 3.3 Robertson 方法 (2003)

Robertson 方法采用**最大似然估计（MLE）**框架，通过**期望最大化（EM）**迭代交替优化 $g(\cdot)$ 和 $\{E_i\}$。

#### 3.3.1 概率模型

假设像素观测值 $Z_{ij}$ 由辐照度 $E_i$ 经过响应函数 $g^{-1}$ 映射后加独立高斯噪声得到：

$$
Z_{ij} \sim \mathcal{N}\!\left(g^{-1}(E_i \cdot \Delta t_j),\; \sigma^2 / w(Z_{ij})\right)
$$

对数似然函数为：

$$
\mathcal{L}(E, g) = -\sum_{i,j} w(Z_{ij}) \left[g(Z_{ij}) - \ln E_i - \ln \Delta t_j\right]^2
$$

#### 3.3.2 EM 迭代

**E 步（更新辐照度 $E_i$）**：固定当前 $g$，对每个像素 $i$ 求最大似然辐照度：

$$
E_i = \frac{\displaystyle\sum_{j=1}^{N} w(Z_{ij}) \cdot g(Z_{ij}) / \Delta t_j}{\displaystyle\sum_{j=1}^{N} w(Z_{ij}) \cdot \left[g(Z_{ij})\right]^2 / \Delta t_j^2}
$$

**M 步（更新响应函数 $g$）**：固定当前 $\{E_i\}$，对每个灰度值 $z$ 统计所有满足 $Z_{ij} = z$ 的像素，计算：

$$
g(z) = \frac{\displaystyle\sum_{\{(i,j)\,:\,Z_{ij}=z\}} E_i \cdot \Delta t_j}{\displaystyle\text{count}(\{(i,j) : Z_{ij} = z\})}
$$

#### 3.3.3 收敛判据

迭代至相邻两次 $g$ 变化低于阈值：

$$
\frac{\|g^{(k+1)} - g^{(k)}\|_2}{\|g^{(k)}\|_2} < \varepsilon \quad (\text{典型值}\;\varepsilon = 10^{-4})
$$

---

## 4. HDR 辐射图合并

### 4.1 加权合并公式

利用已标定的 CRF $g(\cdot)$ 和各曝光时间 $\Delta t_j$，从多张 LDR 图像中恢复每个像素的对数辐照度：

$$
\ln \hat{E}_i = \frac{\displaystyle\sum_{j=1}^{N} w(Z_{ij}) \cdot \left[g(Z_{ij}) - \ln \Delta t_j\right]}{\displaystyle\sum_{j=1}^{N} w(Z_{ij})}
$$

即对所有曝光进行加权平均，权重为 $w(Z_{ij})$，偏差由曝光时间 $\ln \Delta t_j$ 补偿。

### 4.2 权重函数设计

权重函数 $w(z)$ 的设计原则：对处于中间曝光区域的像素赋予较高权重，对饱和（过曝）或近黑（欠曝）像素赋予较低权重。

**标准三角帽权重**（适用于 8 位图像）：

$$
w(z) = \begin{cases}
z - 0 & 0 \leq z \leq 127 \\
255 - z & 128 \leq z \leq 255
\end{cases}
$$

**均匀截断权重**（简化版）：

$$
w(z) = \begin{cases}
0 & z < Z_{lo} \text{ 或 } z > Z_{hi} \\
1 & \text{otherwise}
\end{cases}
$$

其中 $Z_{lo} \approx 5$，$Z_{hi} \approx 250$，排除极端过曝和欠曝的像素。

### 4.3 退化情况处理

若某像素在所有曝光下权重之和为零（即 $\sum_j w(Z_{ij}) = 0$，该像素在所有帧中均饱和或均欠曝），则回退到使用中间曝光帧 $j^*$ 的值：

$$
j^* = \arg\min_j |\ln \Delta t_j - \overline{\ln \Delta t}|
$$

$$
\ln \hat{E}_i = g(Z_{i,j^*}) - \ln \Delta t_{j^*}
$$

### 4.4 线性辐射图

最终辐射图（线性空间）由对数辐照度取指数得到：

$$
\hat{E}_i = \exp\!\left(\ln \hat{E}_i\right)
$$

对于彩色图像，分别对 R、G、B 三通道独立执行上述流程，或使用亮度通道估计 CRF 后统一应用于三通道。

---

## 5. 色调映射 (Tone Mapping)

色调映射算子（Tone Mapping Operator, TMO）将 HDR 辐射图 $L \in [L_{min}, L_{max}]$ 压缩至显示设备可显示的范围 $L_d \in [0, 1]$（或 $[0, 255]$）。

### 5.1 全局算子

全局算子对所有像素使用相同的映射函数，计算效率高，但可能损失局部对比度。

#### Reinhard 全局算子 (2002)

**步骤 1**：计算场景的**对数平均亮度**（感知上近似于"场景键值"）：

$$
\bar{L} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} \ln(\delta + L_i)\right)
$$

其中 $\delta$ 为极小正数（避免 $\ln 0$），$N$ 为像素总数。

**步骤 2**：使用键值参数 $a$（通常取 0.18）**缩放场景亮度**：

$$
L_s = \frac{a}{\bar{L}} \cdot L
$$

**步骤 3**：**Reinhard 映射**，将 $L_s$ 压缩至 $[0, 1)$：

$$
L_d = \frac{L_s \cdot \left(1 + \dfrac{L_s}{L_w^2}\right)}{1 + L_s}
$$

其中 $L_w$ 为场景中最亮点的亮度（白点，White Point），控制高光压缩的程度。

当 $L_w \to \infty$ 时退化为简单的 Reinhard 映射：$L_d = L_s / (1 + L_s)$。

#### Drago 自适应对数算子 (2003)

利用自适应对数基底，使低亮度区域得到较大压缩比，高亮度区域得到较小压缩比：

$$
L_d = L_{d,max} \cdot \frac{\log_{10}(1 + L)}{\log_{10}(1 + L_{max})} \cdot \frac{1}{\log\!\left(2 + 8 \cdot \left(\dfrac{L}{L_{max}}\right)^{\!\frac{\log b}{\log 0.5}}\right)}
$$

其中：
- $L_{d,max}$：目标最大亮度（典型值 100 cd/m²）
- $b \in [0.1, 0.9]$：偏移参数，控制对数底数的自适应程度（典型值 $b = 0.85$）
- $L_{max}$：场景最大亮度

#### 自适应对数映射

一种简化的对数映射，基于场景平均亮度自适应调节：

$$
L_d = L_{d,max} \cdot \frac{\log(1 + L / L_{avg})}{\log(1 + L_{max} / L_{avg})}
$$

其中 $L_{avg}$ 为场景平均亮度，起归一化作用。

### 5.2 局部算子

局部算子根据像素的局部邻域动态调整映射，能更好地保留局部对比度和细节。

#### Reinhard 局部算子 (2002)

**多尺度高斯卷积**：

$$
V_1(x, y, s) = G_s * L(x, y)
$$

$$
V_2(x, y, s) = G_{1.6s} * L(x, y)
$$

其中 $G_s$ 是标准差为 $s$ 的高斯核，$*$ 为卷积运算。

**局部活动度量**（检测是否处于纹理区域）：

$$
\hat{V}(x, y, s) = \frac{V_1(x,y,s) - V_2(x,y,s)}{\dfrac{2^\phi \cdot a}{s^2} + V_1(x,y,s)}
$$

选取满足 $|\hat{V}| < \varepsilon$ 的最大尺度 $s^* = s_{best}$（即找到最大的区域，使该区域内部对比度尚未超过阈值）。

**局部色调映射**：

$$
L_d(x,y) = \frac{L_s(x,y)}{1 + V_1(x, y, s_{best})}
$$

#### Durand 双边滤波算子 (2002)

**核心思想**：在对数域中将图像分解为**基础层**（Base Layer，大尺度光照变化）和**细节层**（Detail Layer，纹理细节），分别处理。

**对数域分解**：

$$
\log L(x,y) = B(x,y) + D(x,y)
$$

其中基础层由双边滤波提取（边缘保持平滑）：

$$
B(x,y) = \frac{\displaystyle\sum_{p \in \Omega} \log L(p) \cdot f\!\left(\|p-q\|\right) \cdot g\!\left(|\log L(p) - \log L(q)|\right)}{\displaystyle\sum_{p \in \Omega} f\!\left(\|p-q\|\right) \cdot g\!\left(|\log L(p) - \log L(q)|\right)}
$$

$f$ 为空间高斯核，$g$ 为强度高斯核（双边滤波的两个核函数）。细节层 $D = \log L - B$。

**压缩与重建**：

$$
\log L_d = \frac{B - B_{max}}{C_{factor}} + D
$$

其中 $C_{factor} = (B_{max} - B_{min}) / C_{target}$，$C_{target}$ 为目标对比度范围（如 5）。

#### Fattal 梯度域算子 (2002)

**核心思想**：直接操作辐射图的**梯度场**，通过衰减大梯度来压缩对比度，再泊松重建。

**梯度衰减函数**：

$$
\phi(x,y) = \frac{\alpha}{|\nabla H(x,y)|} \cdot \left(\frac{|\nabla H(x,y)|}{\alpha}\right)^{\!\beta}
$$

其中 $H = \log L$ 为对数辐射图，$\alpha$ 控制梯度尺度（通常取梯度平均值的 0.1），$\beta \in [0.8, 0.9]$ 控制压缩强度（$\beta < 1$ 时大梯度被衰减）。

**目标梯度场**：

$$
\mathbf{G} = \phi(x,y) \cdot \nabla H(x,y)
$$

**泊松重建**：求解泊松方程，从修改后的梯度场重建压缩后的图像 $I$：

$$
\nabla^2 I = \mathrm{div}\!\left(\mathbf{G}\right) = \frac{\partial G_x}{\partial x} + \frac{\partial G_y}{\partial y}
$$

在频域中使用 DCT/FFT 高效求解：

$$
\hat{I}(u,v) = \frac{\widehat{\mathrm{div}(\mathbf{G})}(u,v)}{-4\pi^2(u^2 + v^2)}
$$

### 5.3 感知驱动算子

基于人类视觉系统（HVS）或经验拟合的算子，主要用于电影/游戏渲染。

#### ACES 色调映射 (Narkowicz 2015)

Academy Color Encoding System 拟合曲线，形式简洁且视觉效果好：

$$
f(x) = \frac{x\,(2.51x + 0.03)}{x\,(2.43x + 0.59) + 0.14}
$$

分母保证输出不超过 1（对 $x \in [0, 1]$ 的输入），曲线接近 S 形，兼顾阴影和高光。

#### Filmic 色调映射 (Hable 2010)

John Hable（Uncharted 2 渲染工程师）提出的分段有理函数曲线：

$$
f(x) = \frac{x(Ax + CB) + DE}{x(Ax + B) + DF} - \frac{E}{F}
$$

标准参数集（W = 11.2 为白点）：

| 参数 | 值 |
|------|-----|
| $A$（肩部强度） | 0.15 |
| $B$（线性强度） | 0.50 |
| $C$（线性角度） | 0.10 |
| $D$（趾部强度） | 0.20 |
| $E$（趾部数值） | 0.02 |
| $F$（趾部角度） | 0.30 |

最终输出经过白点归一化：$L_d = f(L_s) / f(W)$，保证最大输出为 1。

#### Mantiuk 感知对比度算子 (2006)

基于**对比敏感函数（CSF）**的感知驱动色调映射：

**对比度计算**：在对数亮度图的多个频带上计算对比度响应，使用 CSF（通常是 Mannos-Sakrison 等）对各频带加权：

$$
CSF(f) = 2.6 \cdot (0.0192 + 0.114 f) \cdot e^{-(0.114 f)^{1.1}}
$$

其中 $f$ 为空间频率（单位：周期/度视角）。

**压缩流程**：将各频带对比度压缩到显示设备的可感知范围，再在对数域中重建，确保感知效果一致。

#### 直方图映射

利用 HDR 图像的**累积分布函数（CDF）**进行自适应映射：

$$
L_d = \mathrm{CDF}\!\left(\log L\right)
$$

即将对数亮度的 CDF 作为映射函数，使输出亮度均匀分布（类似直方图均衡化）。实践中常使用截断 CDF（Clipped CDF），避免极端值过分拉伸对比度。

---

## 6. 多曝光融合 (Mertens 2007)

多曝光融合（MEF）不经过 HDR 辐射图合并步骤，而是直接在像素值域对多张 LDR 图像进行加权融合，输出仍为 LDR。

### 6.1 质量度量

对每张曝光图像 $I_k$，计算三种质量度量：

**对比度（Contrast）**：

使用**拉普拉斯算子**衡量局部对比度：

$$
C_k(x,y) = \left|L * I_k(x,y)\right|
$$

其中 $L = \begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$ 为离散拉普拉斯核，$*$ 为卷积。

**饱和度（Saturation）**：

衡量颜色的鲜艳程度（偏灰白的颜色饱和度低）：

$$
S_k(x,y) = \sigma\!\left(R_k, G_k, B_k\right) = \sqrt{\frac{(R-\mu)^2 + (G-\mu)^2 + (B-\mu)^2}{3}}
$$

其中 $\mu = (R+G+B)/3$。

**曝光度（Exposedness）**：

衡量像素的曝光是否合适，使用以 0.5 为中心的高斯函数：

$$
E_k(x,y) = \prod_{c \in \{R,G,B\}} \exp\!\left(-\frac{(I_k^c(x,y) - 0.5)^2}{2\sigma_E^2}\right)
$$

其中 $\sigma_E = 0.2$。

### 6.2 组合权重

将三种质量度量相乘并通过指数加权：

$$
\tilde{W}_k(x,y) = C_k^{w_C}(x,y) \cdot S_k^{w_S}(x,y) \cdot E_k^{w_E}(x,y)
$$

归一化（使所有曝光在同一位置的权重之和为 1）：

$$
W_k(x,y) = \frac{\tilde{W}_k(x,y)}{\displaystyle\sum_{k'=1}^{N} \tilde{W}_{k'}(x,y) + \varepsilon}
$$

典型权重指数：$w_C = w_S = w_E = 1$（等权重）。

### 6.3 拉普拉斯金字塔融合

直接用权重图加权融合会产生边缘瑕疵（ghosting）。Mertens 方法使用多尺度金字塔实现无缝融合。

**构建拉普拉斯金字塔**：

对每张曝光图像 $I_k$ 构建 $L$ 层拉普拉斯金字塔 $\{LP_k^l\}_{l=0}^{L-1}$：

$$
LP_k^l = GP_k^l - \text{Upsample}(GP_k^{l+1})
$$

其中 $GP_k^l$ 为对应的高斯金字塔层（逐层降采样），最顶层 $LP_k^{L-1} = GP_k^{L-1}$。

**构建权重金字塔**：

对每个权重图 $W_k$ 构建高斯金字塔 $\{GW_k^l\}$（仅做平滑下采样）。

**各层加权融合**：

$$
LP_{fused}^l(x,y) = \sum_{k=1}^{N} GW_k^l(x,y) \cdot LP_k^l(x,y)
$$

### 6.4 重建

由融合后的拉普拉斯金字塔 $\{LP_{fused}^l\}$ 重建最终融合图像：

$$
I_{fused} = \sum_{l=0}^{L-1} \text{Upsample}^l\!\left(LP_{fused}^l\right)
$$

即从最顶层逐层上采样后相加。

---

## 7. 单张图像 HDR

当仅有单张 LDR 图像时，可通过图像增强技术近似扩展感知动态范围。

### 7.1 CLAHE 自适应直方图均衡

**限制对比度自适应直方图均衡**（Contrast Limited Adaptive Histogram Equalization, CLAHE）在局部块内进行直方图均衡，并限制最大对比度放大倍数。

对每个局部块内的像素，均衡化映射函数 $T(z)$ 由截断 CDF 构成：

$$
T(z) = \frac{Z_{max} - Z_{min}}{M} \sum_{k=Z_{min}}^{z} \min\!\left(h(k),\, C_{limit}\right)
$$

其中：
- $h(k)$：局部块内灰度直方图
- $M$：局部块的像素总数
- $C_{limit}$：对比度限制阈值（截断值，超出部分均匀重分配至全部灰度级）

### 7.2 多尺度细节增强

在对数亮度域进行多尺度分解，分别增强不同尺度的细节：

**对数域分解**：

$$
\log L = B_\sigma + D
$$

其中 $B_\sigma$ 为高斯平滑的基础层，$D = \log L - B_\sigma$ 为细节层。

**细节增强**：

$$
\log L_{enhanced} = B_\sigma + \alpha \cdot D
$$

其中 $\alpha > 1$ 为细节增益系数。转换回线性空间：$L_{enhanced} = \exp(\log L_{enhanced})$。

### 7.3 LAB 色彩空间保持

直接在 RGB 空间增强亮度会引起色彩偏移。使用 CIE LAB 色彩空间，仅对亮度通道 $L^*$ 进行增强，保持色度不变：

$$
[L^*, a^*, b^*] = \text{RGB2LAB}(R, G, B)
$$

$$
L^*_{enhanced} = \text{TMO}(L^*)
$$

$$
[R_{out}, G_{out}, B_{out}] = \text{LAB2RGB}(L^*_{enhanced},\; a^*,\; b^*)
$$

其中 $\text{TMO}(\cdot)$ 可为任意色调映射算子（如 CLAHE 或 Reinhard 全局算子）。

---

## 8. 参考文献

1. **Debevec, P. E., & Malik, J. (1997)**. Recovering high dynamic range radiance maps from photographs. *ACM SIGGRAPH*, 369–378.

2. **Robertson, M. A., Borman, S., & Stevenson, R. L. (2003)**. Estimation-theoretic approach to dynamic range enhancement using multiple exposures. *Journal of Electronic Imaging*, 12(2), 219–228.

3. **Ward, G. (2003)**. Fast, robust image registration for compositing high dynamic range photographs from hand-held exposures. *Journal of Graphics Tools*, 8(2), 17–30.

4. **Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. (2002)**. Photographic tone reproduction for digital images. *ACM SIGGRAPH*, 267–276.

5. **Drago, F., Myszkowski, K., Annen, T., & Chiba, N. (2003)**. Adaptive logarithmic mapping for displaying high contrast scenes. *Computer Graphics Forum*, 22(3), 419–426.

6. **Durand, F., & Dorsey, J. (2002)**. Fast bilateral filtering for the display of high-dynamic-range images. *ACM SIGGRAPH*, 257–266.

7. **Fattal, R., Lischinski, D., & Werman, M. (2002)**. Gradient domain high dynamic range compression. *ACM SIGGRAPH*, 249–256.

8. **Mantiuk, R., Daly, S., Myszkowski, K., & Seidel, H.-P. (2006)**. Predicting visible differences in high dynamic range images — model and its calibration. *Proc. SPIE Human Vision and Electronic Imaging*.

9. **Mertens, T., Kautz, J., & Van Reeth, F. (2007)**. Exposure fusion. *15th Pacific Conference on Computer Graphics and Applications*, 382–390.

10. **Narkowicz, K. (2015)**. ACES filmic tone mapping curve. *Blog post*, https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/.

11. **Hable, J. (2010)**. Filmic tonemapping operators. *Uncharted 2: HDR Lighting, GDC 2010*.
