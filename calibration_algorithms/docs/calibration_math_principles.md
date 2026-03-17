# 标定算法数学原理

本文档详细介绍三种标定算法的数学原理：相机标定（Zhang 方法）、LiDAR-相机外参标定（PnP 方法）、手眼标定（AX=XB 问题）。

---

## 目录

1. [针孔相机模型](#1-针孔相机模型)
2. [相机标定 —— Zhang 方法](#2-相机标定--zhang-方法)
3. [LiDAR-相机外参标定 —— PnP 方法](#3-lidar-相机外参标定--pnp-方法)
4. [手眼标定 —— AX=XB 问题](#4-手眼标定--axxb-问题)
5. [参考文献](#5-参考文献)

---

## 1. 针孔相机模型

### 1.1 基本投影方程

针孔相机模型描述了三维世界中的点如何映射到二维图像平面。其核心是**透视投影**。

设世界坐标系中的一个三维点为 $\mathbf{P}_w = [X_w, Y_w, Z_w]^T$，对应的图像像素坐标为 $\mathbf{m} = [u, v]^T$，则投影关系为：

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

其中：
- $s$ 是尺度因子（等于点在相机坐标系下的深度 $Z_c$）
- $\mathbf{K}$ 是 **内参矩阵**（Intrinsic Matrix）
- $[\mathbf{R} | \mathbf{t}]$ 是 **外参矩阵**（Extrinsic Matrix），$\mathbf{R} \in SO(3)$ 为旋转矩阵，$\mathbf{t} \in \mathbb{R}^3$ 为平移向量

### 1.2 内参矩阵

$$
\mathbf{K} = \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

各参数含义：
- $f_x = f / d_x$，$f_y = f / d_y$：焦距（以像素为单位），其中 $f$ 为物理焦距，$d_x, d_y$ 为像素的物理尺寸
- $(c_x, c_y)$：**主点**（Principal Point），即光轴与图像平面的交点坐标
- $\gamma$：**倾斜因子**（Skew Factor），通常为 0（表示像素坐标轴正交）

### 1.3 畸变模型

实际镜头会引入畸变，使直线在图像中变弯。OpenCV 使用以下畸变模型：

设归一化相机坐标为：

$$
x = X_c / Z_c, \quad y = Y_c / Z_c
$$

径向距离的平方：

$$
r^2 = x^2 + y^2
$$

**径向畸变**（Radial Distortion）：

$$
\begin{aligned}
x_{radial} &= x \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
y_{radial} &= y \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
\end{aligned}
$$

- $k_1, k_2, k_3$：径向畸变系数
- $k_1 > 0$：桶形畸变（Barrel Distortion）
- $k_1 < 0$：枕形畸变（Pincushion Distortion）

**切向畸变**（Tangential Distortion）：

$$
\begin{aligned}
x_{tangential} &= 2p_1 xy + p_2(r^2 + 2x^2) \\
y_{tangential} &= p_1(r^2 + 2y^2) + 2p_2 xy
\end{aligned}
$$

- $p_1, p_2$：切向畸变系数（由镜头与传感器不平行引起）

**完整畸变模型**：

$$
\begin{aligned}
x' &= x \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 xy + p_2(r^2 + 2x^2) \\
y' &= y \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2 xy
\end{aligned}
$$

最终像素坐标：

$$
u = f_x \cdot x' + c_x, \quad v = f_y \cdot y' + c_y
$$

### 1.4 外参变换

世界坐标系到相机坐标系的变换：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{R} \begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} + \mathbf{t}
$$

或者用齐次坐标表示：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix} = \mathbf{T} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

其中 $\mathbf{T} \in SE(3)$ 是 4×4 齐次变换矩阵，属于特殊欧氏群。

---

## 2. 相机标定 —— Zhang 方法

Zhang 标定法（Zhang, 2000）使用平面棋盘格标定板，通过计算单应性矩阵来求解相机内参。

### 2.1 单应性矩阵

假设标定板放在世界坐标系的 $Z_w = 0$ 平面上，则投影方程简化为：

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{r}_3 & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ 0 \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix}
$$

其中 $\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3$ 是旋转矩阵 $\mathbf{R}$ 的三列。

定义**单应性矩阵**：

$$
\mathbf{H} = \lambda \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix} = \begin{bmatrix} \mathbf{h}_1 & \mathbf{h}_2 & \mathbf{h}_3 \end{bmatrix}
$$

其中 $\lambda$ 是任意非零尺度因子。$\mathbf{H}$ 是 3×3 矩阵，有 8 个自由度（除去尺度）。

### 2.2 单应性矩阵的求解

每个 3D-2D 点对提供 2 个方程，$\mathbf{H}$ 有 8 个未知数，因此至少需要 4 个点对。

对于每对对应点 $(\mathbf{M}_i, \mathbf{m}_i)$：

$$
\begin{bmatrix} \mathbf{M}_i^T & \mathbf{0}^T & -u_i \mathbf{M}_i^T \\ \mathbf{0}^T & \mathbf{M}_i^T & -v_i \mathbf{M}_i^T \end{bmatrix} \begin{bmatrix} \mathbf{h}_1 \\ \mathbf{h}_2 \\ \mathbf{h}_3 \end{bmatrix} = \mathbf{0}
$$

将所有点的方程堆叠成 $\mathbf{A}\mathbf{h} = \mathbf{0}$，使用 **SVD 分解**求解，取 $\mathbf{V}$ 的最后一列即为 $\mathbf{h}$ 的最小二乘解。

### 2.3 内参约束方程

由 $\mathbf{H}$ 和旋转矩阵的正交性约束：

$$
\mathbf{r}_1^T \mathbf{r}_2 = 0 \quad (\text{正交约束})
$$
$$
\|\mathbf{r}_1\| = \|\mathbf{r}_2\| \quad (\text{等模约束})
$$

将上式代入 $\mathbf{H} = \lambda \mathbf{K} [\mathbf{r}_1, \mathbf{r}_2, \mathbf{t}]$，得到：

$$
\mathbf{h}_1^T \mathbf{K}^{-T} \mathbf{K}^{-1} \mathbf{h}_2 = 0
$$
$$
\mathbf{h}_1^T \mathbf{K}^{-T} \mathbf{K}^{-1} \mathbf{h}_1 = \mathbf{h}_2^T \mathbf{K}^{-T} \mathbf{K}^{-1} \mathbf{h}_2
$$

定义 $\mathbf{B} = \mathbf{K}^{-T} \mathbf{K}^{-1}$（对称正定矩阵），它有 6 个独立元素：

$$
\mathbf{B} = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}
$$

每张图像提供 2 个约束方程，$\mathbf{B}$ 有 5 个自由度（6 个元素减 1 个尺度），因此：
- **至少需要 3 张图像**（$\gamma = 0$ 时）
- 使用更多图像可以提高精度

### 2.4 求解内参

将 $\mathbf{B}$ 的元素向量化为 $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^T$，构建线性方程组 $\mathbf{V}\mathbf{b} = \mathbf{0}$，使用 SVD 求解。

由 $\mathbf{B} = \mathbf{K}^{-T}\mathbf{K}^{-1}$ 反解内参：

$$
\begin{aligned}
c_y &= (B_{12} B_{13} - B_{11} B_{23}) / (B_{11} B_{22} - B_{12}^2) \\
\lambda &= B_{33} - [B_{13}^2 + c_y(B_{12} B_{13} - B_{11} B_{23})] / B_{11} \\
f_x &= \sqrt{\lambda / B_{11}} \\
f_y &= \sqrt{\lambda B_{11} / (B_{11} B_{22} - B_{12}^2)} \\
\gamma &= -B_{12} f_x^2 f_y / \lambda \\
c_x &= \gamma c_y / f_y - B_{13} f_x^2 / \lambda
\end{aligned}
$$

### 2.5 求解外参

得到 $\mathbf{K}$ 后，对每个视角求解外参：

$$
\begin{aligned}
\mathbf{r}_1 &= \lambda \mathbf{K}^{-1} \mathbf{h}_1 \\
\mathbf{r}_2 &= \lambda \mathbf{K}^{-1} \mathbf{h}_2 \\
\mathbf{r}_3 &= \mathbf{r}_1 \times \mathbf{r}_2 \\
\mathbf{t} &= \lambda \mathbf{K}^{-1} \mathbf{h}_3
\end{aligned}
$$

其中 $\lambda = 1 / \|\mathbf{K}^{-1} \mathbf{h}_1\|$。

由于噪声的存在，上述求得的 $[\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3]$ 不一定是正交矩阵，需要通过 SVD 投影到最近的旋转矩阵：

$$
\mathbf{R}_{approx} = \mathbf{U} \mathbf{V}^T \quad (\text{其中} \ \mathbf{R}_{approx} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)
$$

### 2.6 非线性优化（最大似然估计）

上述线性解仅是初始估计，需要通过非线性优化得到最终解。

**目标函数**——最小化总重投影误差：

$$
\min_{\mathbf{K}, k_1, k_2, \{\mathbf{R}_i, \mathbf{t}_i\}} \sum_{i=1}^{n} \sum_{j=1}^{m} \| \mathbf{m}_{ij} - \hat{\mathbf{m}}(\mathbf{K}, k_1, k_2, \mathbf{R}_i, \mathbf{t}_i, \mathbf{M}_j) \|^2
$$

其中：
- $n$ 是图像数量，$m$ 是每张图像的角点数量
- $\mathbf{m}_{ij}$ 是第 $i$ 张图像中第 $j$ 个角点的检测坐标
- $\hat{\mathbf{m}}(\cdot)$ 是投影函数（包括畸变模型）
- 使用 **Levenberg-Marquardt** 算法求解

### 2.7 重投影误差

标定质量的核心评估指标：

$$
\text{RMS Error} = \sqrt{\frac{1}{nm} \sum_{i=1}^{n} \sum_{j=1}^{m} \| \mathbf{m}_{ij} - \hat{\mathbf{m}}_{ij} \|^2}
$$

一般标准：
- $< 0.5$ 像素：优秀
- $0.5 \sim 1.0$ 像素：良好
- $> 1.0$ 像素：需要重新标定

---

## 3. LiDAR-相机外参标定 —— PnP 方法

### 3.1 问题定义

LiDAR-相机外参标定的目标是求解从 LiDAR 坐标系到相机坐标系的刚体变换 $\mathbf{T}_{L \to C}$：

$$
\mathbf{P}_C = \mathbf{R} \mathbf{P}_L + \mathbf{t}
$$

其中：
- $\mathbf{P}_L \in \mathbb{R}^3$：LiDAR 坐标系下的 3D 点
- $\mathbf{P}_C \in \mathbb{R}^3$：相机坐标系下的 3D 点
- $\mathbf{R} \in SO(3)$：旋转矩阵（3 个自由度）
- $\mathbf{t} \in \mathbb{R}^3$：平移向量（3 个自由度）

已知条件：
- 相机内参 $\mathbf{K}$（通过相机标定获得）
- 一组 LiDAR 3D 点 $\{\mathbf{P}_{L,i}\}$ 与对应的图像 2D 点 $\{\mathbf{m}_i\}$

### 3.2 PnP 问题（Perspective-n-Point）

PnP 问题：已知 $n$ 个 3D 点及其在图像上的 2D 投影，以及相机内参，求解相机位姿（即外参）。

投影关系：

$$
s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \mathbf{K} (\mathbf{R} \mathbf{P}_{L,i} + \mathbf{t})
$$

### 3.3 DLT（Direct Linear Transform）方法

将投影方程展开，消去尺度因子 $s$：

对每个点对 $(\mathbf{P}_i, \mathbf{m}_i)$，令 $\mathbf{Q}_i = \mathbf{R}\mathbf{P}_i + \mathbf{t} = [X_i, Y_i, Z_i]^T$，则：

$$
u_i = \frac{f_x X_i + c_x Z_i}{Z_i}, \quad v_i = \frac{f_y Y_i + c_y Z_i}{Z_i}
$$

交叉相乘消去 $Z_i$，得到每个点 2 个线性方程。$n$ 个点共 $2n$ 个方程，待估参数为 $\mathbf{R}$（3 个自由度）和 $\mathbf{t}$（3 个自由度）共 6 个未知数。

**至少需要 3 个非共线点**（P3P），但通常使用更多点以提高精度。

### 3.4 EPnP（Efficient PnP）

EPnP（Lepetit et al., 2009）通过 4 个**控制点**来表达所有 3D 点：

$$
\mathbf{P}_i = \sum_{j=1}^{4} \alpha_{ij} \mathbf{C}_j, \quad \sum_{j=1}^{4} \alpha_{ij} = 1
$$

将问题转化为求解控制点在相机坐标系中的位置，极大降低了计算复杂度。

### 3.5 迭代法（SOLVEPNP_ITERATIVE）

OpenCV 默认使用的方法：

1. 使用 DLT 获取初始解
2. 将初始解参数化为旋转向量 $\mathbf{r}$（Rodrigues 表示）和平移向量 $\mathbf{t}$
3. 使用 **Levenberg-Marquardt** 最小化重投影误差：

$$
\min_{\mathbf{r}, \mathbf{t}} \sum_{i=1}^{n} \| \mathbf{m}_i - \pi(\mathbf{K}, \mathbf{R}(\mathbf{r}), \mathbf{t}, \mathbf{P}_i) \|^2
$$

其中 $\pi(\cdot)$ 是完整的投影函数（含畸变），$\mathbf{R}(\mathbf{r})$ 通过 Rodrigues 公式将旋转向量转为旋转矩阵：

$$
\mathbf{R} = \mathbf{I} + \sin\theta \cdot [\mathbf{n}]_\times + (1 - \cos\theta) \cdot [\mathbf{n}]_\times^2
$$

其中 $\theta = \|\mathbf{r}\|$，$\mathbf{n} = \mathbf{r}/\theta$，$[\mathbf{n}]_\times$ 是反对称矩阵。

### 3.6 RANSAC 鲁棒估计

当对应点中存在离群点（错误匹配）时，使用 RANSAC 提高鲁棒性。

**RANSAC 算法流程：**

1. 随机选择最小点集（4 个点）
2. 用这些点求解 PnP 得到候选解 $(\hat{\mathbf{R}}, \hat{\mathbf{t}})$
3. 对所有点计算重投影误差：$e_i = \| \mathbf{m}_i - \pi(\mathbf{K}, \hat{\mathbf{R}}, \hat{\mathbf{t}}, \mathbf{P}_i) \|$
4. 若 $e_i < \tau$（阈值），则标记为内点
5. 若内点数量 $> d$（最低要求），用所有内点重新求解 PnP
6. 重复步骤 1-5，保留内点最多的解

**迭代次数**的估计：

$$
k = \frac{\log(1 - p)}{\log(1 - w^n)}
$$

其中：
- $p$：置信度（通常 0.99）
- $w$：内点比例
- $n$：最小点集大小（4）

例如，$w = 0.8$，$p = 0.99$ 时，$k = \lceil \frac{\log(0.01)}{\log(1 - 0.8^4)} \rceil = 7$ 次。

### 3.7 误差评估

**旋转误差**：

$$
\theta_{err} = \arccos\left(\frac{\text{tr}(\mathbf{R}_{true} \mathbf{R}_{est}^T) - 1}{2}\right)
$$

**平移误差**：

$$
e_t = \|\mathbf{t}_{true} - \mathbf{t}_{est}\|
$$

**相对平移误差**：

$$
e_t^{rel} = \frac{\|\mathbf{t}_{true} - \mathbf{t}_{est}\|}{\|\mathbf{t}_{true}\|} \times 100\%
$$

---

## 4. 手眼标定 —— AX=XB 问题

### 4.1 问题建模

#### Eye-in-Hand 配置

相机固定在机器人末端执行器上。运动链关系：

$$
\mathbf{T}_{target}^{base} = \mathbf{T}_{gripper}^{base} \cdot \mathbf{T}_{cam}^{gripper} \cdot \mathbf{T}_{target}^{cam}
$$

其中：
- $\mathbf{T}_{gripper}^{base}$：末端在基座坐标系下的位姿（机器人正运动学，已知）
- $\mathbf{T}_{cam}^{gripper}$：相机在末端坐标系下的位姿（$\mathbf{X}$，待求解）
- $\mathbf{T}_{target}^{cam}$：标定板在相机坐标系下的位姿（标定板检测，已知）
- $\mathbf{T}_{target}^{base}$：标定板在基座坐标系下的位姿（固定不变）

对于两个不同位姿 $i$ 和 $j$，由于标定板位置不变：

$$
\mathbf{T}_{gripper,i}^{base} \cdot \mathbf{X} \cdot \mathbf{T}_{target}^{cam,i} = \mathbf{T}_{gripper,j}^{base} \cdot \mathbf{X} \cdot \mathbf{T}_{target}^{cam,j}
$$

整理得：

$$
\underbrace{(\mathbf{T}_{gripper,j}^{base})^{-1} \mathbf{T}_{gripper,i}^{base}}_{\mathbf{A}} \cdot \mathbf{X} = \mathbf{X} \cdot \underbrace{\mathbf{T}_{target}^{cam,j} (\mathbf{T}_{target}^{cam,i})^{-1}}_{\mathbf{B}}
$$

即经典的 **AX = XB** 方程。

#### Eye-to-Hand 配置

相机固定在外部（如工作台），方程变为 $\mathbf{AX} = \mathbf{ZB}$，其中 $\mathbf{Z}$ 是相机在基座坐标系下的位姿。

### 4.2 AX=XB 的可解性

将 $\mathbf{A}$, $\mathbf{B}$, $\mathbf{X}$ 分解为旋转和平移部分：

$$
\mathbf{A} = \begin{bmatrix} \mathbf{R}_A & \mathbf{t}_A \\ \mathbf{0}^T & 1 \end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix} \mathbf{R}_B & \mathbf{t}_B \\ \mathbf{0}^T & 1 \end{bmatrix}, \quad
\mathbf{X} = \begin{bmatrix} \mathbf{R}_X & \mathbf{t}_X \\ \mathbf{0}^T & 1 \end{bmatrix}
$$

展开 $\mathbf{AX} = \mathbf{XB}$：

$$
\begin{cases}
\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B & \quad \text{(旋转方程)} \\
\mathbf{R}_A \mathbf{t}_X + \mathbf{t}_A = \mathbf{R}_X \mathbf{t}_B + \mathbf{t}_X & \quad \text{(平移方程)}
\end{cases}
$$

简化平移方程：

$$
(\mathbf{R}_A - \mathbf{I}) \mathbf{t}_X = \mathbf{R}_X \mathbf{t}_B - \mathbf{t}_A
$$

**求解条件**：
- 至少需要 **2 组**不同的运动 ($\mathbf{A}_i, \mathbf{B}_i$)
- 两次运动的旋转轴**不能平行**（否则方程退化）
- 实际中需要更多位姿（$\geq 3$）以提高精度

### 4.3 Tsai-Lenz 方法（1989）

Tsai-Lenz 方法将问题分为两步：先解旋转，再解平移。

#### 第一步：求解旋转 $\mathbf{R}_X$

利用旋转向量（Rodrigues 表示）的性质。设旋转矩阵 $\mathbf{R}$ 对应的旋转向量为 $\mathbf{r}$（方向为旋转轴，模长为旋转角度），定义修正 Rodrigues 向量：

$$
\mathbf{r}' = 2 \sin(\theta/2) \cdot \hat{\mathbf{n}}
$$

其中 $\hat{\mathbf{n}}$ 为单位旋转轴，$\theta$ 为旋转角度。

对于旋转方程 $\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B$，可以证明：

$$
\mathbf{r}'_A \times \mathbf{r}'_X = \mathbf{r}'_B \times \mathbf{r}'_X + (\mathbf{r}'_A + \mathbf{r}'_B) \times \mathbf{r}'_X
$$

简化为线性方程（Tsai 的推导）：

$$
(\mathbf{r}'_A - \mathbf{r}'_B) \times \mathbf{r}'_X = \mathbf{r}'_A + \mathbf{r}'_B
$$

写成矩阵形式：

$$
[\mathbf{r}'_A - \mathbf{r}'_B]_\times \mathbf{r}'_X = \mathbf{r}'_A + \mathbf{r}'_B
$$

其中 $[\cdot]_\times$ 表示反对称矩阵（叉积矩阵）。

将多组运动的方程堆叠，用最小二乘法求解 $\mathbf{r}'_X$，再恢复旋转矩阵 $\mathbf{R}_X$。

#### 第二步：求解平移 $\mathbf{t}_X$

利用平移方程：

$$
(\mathbf{R}_{A_i} - \mathbf{I}) \mathbf{t}_X = \mathbf{R}_X \mathbf{t}_{B_i} - \mathbf{t}_{A_i}
$$

这是关于 $\mathbf{t}_X$ 的线性方程组，将多组运动堆叠后用最小二乘法求解。

### 4.4 Park 方法（1994）

Park 方法基于**李群/李代数**理论，在 $SO(3)$ 上直接操作。

将旋转方程 $\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B$ 两边取对数映射到李代数 $\mathfrak{so}(3)$：

$$
\log(\mathbf{R}_A) = \mathbf{R}_X \log(\mathbf{R}_B) \mathbf{R}_X^T
$$

设 $\boldsymbol{\alpha} = \log(\mathbf{R}_A)^\vee$，$\boldsymbol{\beta} = \log(\mathbf{R}_B)^\vee$（从反对称矩阵提取向量），则：

$$
\boldsymbol{\alpha} = \mathbf{R}_X \boldsymbol{\beta}
$$

这意味着 $\mathbf{R}_X$ 将 $\mathbf{R}_B$ 的旋转轴映射到 $\mathbf{R}_A$ 的旋转轴。

对于多组数据，构建矩阵 $\mathbf{M}$：

$$
\mathbf{M} = \sum_{i=1}^{n} \boldsymbol{\beta}_i \boldsymbol{\alpha}_i^T
$$

通过 SVD 分解 $\mathbf{M} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$ 求解：

$$
\mathbf{R}_X = \mathbf{V} \mathbf{U}^T
$$

（需确保 $\det(\mathbf{R}_X) = +1$）

平移部分同 Tsai 方法。

### 4.5 Daniilidis 方法（1999）—— 对偶四元数

Daniilidis 方法使用**对偶四元数**（Dual Quaternion）同时求解旋转和平移。

#### 对偶四元数简介

对偶四元数 $\hat{\mathbf{q}} = \mathbf{q} + \varepsilon \mathbf{q}'$，其中：
- $\mathbf{q}$：表示旋转的四元数
- $\mathbf{q}'$：与平移相关的对偶部分
- $\varepsilon$：对偶单位，$\varepsilon^2 = 0$

刚体运动的对偶四元数表示：

$$
\hat{\mathbf{q}} = \mathbf{q} + \frac{\varepsilon}{2} \mathbf{t} \cdot \mathbf{q}
$$

#### 求解方法

$\mathbf{AX} = \mathbf{XB}$ 用对偶四元数表示为 $\hat{\mathbf{a}} \hat{\mathbf{x}} = \hat{\mathbf{x}} \hat{\mathbf{b}}$。

展开得到关于 $\hat{\mathbf{x}}$ 的线性约束。将多组运动的约束堆叠成矩阵：

$$
\mathbf{T} \begin{bmatrix} \mathbf{q} \\ \mathbf{q}' \end{bmatrix} = \mathbf{0}
$$

其中 $\mathbf{T}$ 是 $6n \times 8$ 矩阵。通过 SVD 分解 $\mathbf{T}$，解空间是 $\mathbf{T}$ 零空间的线性组合。

利用对偶四元数的约束条件：
1. $\mathbf{q}^T \mathbf{q} = 1$（单位四元数）
2. $\mathbf{q}^T \mathbf{q}' = 0$（正交约束）

从零空间中求解满足约束的 $\hat{\mathbf{x}}$。

**优点**：同时求解旋转和平移，避免了 Tsai 方法中旋转误差累积到平移的问题。

### 4.6 各方法对比

| 方法 | 年份 | 特点 | 旋转/平移耦合 |
|------|------|------|---------------|
| **Tsai** | 1989 | 分离求解，简单高效 | 分离（误差累积） |
| **Park** | 1994 | 李群理论，几何直观 | 分离 |
| **Horaud** | 1995 | 四元数表示 | 分离 |
| **Andreff** | 2001 | Kronecker 积，线性化 | 耦合 |
| **Daniilidis** | 1999 | 对偶四元数，同时求解 | 耦合 |

一般建议：
- 数据质量好时，各方法差异不大
- 数据噪声大时，Park 和 Daniilidis 通常更稳定
- 建议同时使用多种方法并对比结果

### 4.7 退化条件

以下情况会导致手眼标定退化（结果不可靠）：

1. **旋转轴平行**：所有运动的旋转轴方向相同，无法唯一确定 $\mathbf{R}_X$
2. **纯平移运动**：没有旋转信息，$\mathbf{R}_X$ 无法确定
3. **旋转角度太小**：小角度旋转对噪声敏感
4. **位姿数量不足**：至少需要 2 个（理论最少），实际建议 15 个以上

**最佳实践**：
- 旋转角度变化 $\geq 30°$
- 覆盖三个轴向的旋转
- 位姿分布在半球形空间内
- 采集 15-25 个位姿

---

## 5. 参考文献

1. **Zhang, Z.** (2000). "A Flexible New Technique for Camera Calibration." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334.

2. **Tsai, R.Y. and Lenz, R.K.** (1989). "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration." *IEEE Transactions on Robotics and Automation*, 5(3), 345-358.

3. **Park, F.C. and Martin, B.J.** (1994). "Robot Sensor Calibration: Solving AX=XB on the Euclidean Group." *IEEE Transactions on Robotics and Automation*, 10(5), 717-721.

4. **Horaud, R. and Dornaika, F.** (1995). "Hand-Eye Calibration." *International Journal of Robotics Research*, 14(3), 195-210.

5. **Daniilidis, K.** (1999). "Hand-Eye Calibration Using Dual Quaternions." *International Journal of Robotics Research*, 18(3), 286-298.

6. **Andreff, N., Horaud, R., and Espiau, B.** (2001). "Robot Hand-Eye Calibration Using Structure-from-Motion." *International Journal of Robotics Research*, 20(3), 228-248.

7. **Lepetit, V., Moreno-Noguer, F., and Fua, P.** (2009). "EPnP: An Accurate O(n) Solution to the PnP Problem." *International Journal of Computer Vision*, 81(2), 155-166.

8. **Hartley, R. and Zisserman, A.** (2003). *Multiple View Geometry in Computer Vision.* Cambridge University Press.

9. **OpenCV 官方文档**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

10. **KITTI 数据集**: https://www.cvlibs.net/datasets/kitti/ (LiDAR-相机标定参考数据)
