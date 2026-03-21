"""
运动轨迹平滑模块 - 视频防抖核心算法
Trajectory Smoothing Module for Video Stabilization

【视频防抖核心思想】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
相机真实运动 = 期望的平滑运动 + 手抖/震动（噪声）

防抖目标: 只保留期望的平滑运动，去除高频噪声抖动

设:
  p[k] = 第k帧的运动参数（平移/旋转）累积轨迹
  p̂[k] = 平滑后的目标轨迹
  c[k] = 补偿量 = p[k] - p̂[k]（需要施加的反向补偿）

将 c[k] 对应的变换矩阵应用到第k帧，即可去除抖动。

【本模块实现的平滑算法】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 移动平均平滑 (Moving Average):
   p̂[k] = (1/2R+1) · Σ_{j=k-R}^{k+R} p[j]
   简单高效，适合离线处理

2. 高斯加权平均 (Gaussian-Weighted Moving Average):
   p̂[k] = Σ_j g(j-k) · p[j] / Σ_j g(j-k)
   g(x) = exp(-x²/(2σ²))
   对近邻帧权重更高，平滑更自然

3. 卡尔曼滤波 (Kalman Filter) - 在线/实时版:
   适合实时防抖（只需历史帧，不需要未来帧）

   状态模型: x[k] = F·x[k-1] + w[k]  (w: 过程噪声)
   观测模型: z[k] = H·x[k] + v[k]    (v: 观测噪声)

   预测步骤:
     x̂⁻[k] = F·x̂[k-1]              (先验状态估计)
     P⁻[k]  = F·P[k-1]·Fᵀ + Q       (先验误差协方差)

   更新步骤:
     K[k] = P⁻[k]·Hᵀ·(H·P⁻[k]·Hᵀ + R)⁻¹  (卡尔曼增益)
     x̂[k] = x̂⁻[k] + K[k]·(z[k] - H·x̂⁻[k]) (后验状态估计)
     P[k]  = (I - K[k]·H)·P⁻[k]               (后验误差协方差)

   状态变量 x = [位置; 速度]ᵀ (2×1 或 6×1 含旋转)
   F = [1 1; 0 1] (匀速运动模型)
   H = [1 0] (只观测位置)
   Q: 过程噪声（抖动方差，需调参）
   R: 观测噪声（测量误差，需调参）

4. L1 优化轨迹平滑 (L1-Optimal Camera Path):
   参考: Liu et al. "Subspace Video Stabilization" (2011)
   最小化: Σ_k ||p̂[k] - p̂[k-1]||₁ + λ||p̂[k] - p[k]||²
   L1 范数使平滑路径对突变鲁棒
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MovingAverageSmoother:
    """
    移动平均轨迹平滑器（离线，全视频处理）

    【数学原理】
    简单移动平均（窗口大小2R+1）:
    p̂[k] = (1/(2R+1)) · Σ_{j=k-R}^{k+R} p[j]

    边界处理:
    - 对视频开头/结尾: 使用填充（clamp）处理
    - p[j<0] = p[0], p[j≥N] = p[N-1]

    频率域分析:
    移动平均等价于与矩形窗卷积 → 低通滤波器
    频率响应: H(f) = sin(πf(2R+1)) / ((2R+1)·sin(πf))
    截止频率约为 1/(2R+1)
    """

    def __init__(self, radius: int = 15):
        """
        Args:
            radius: 平均窗口半径 R（总窗口大小 = 2R+1 帧）
                   radius=15 → 31帧窗口（约1秒@30fps）
                   radius更大 → 更平滑但更多裁剪
        """
        self.radius = radius
        logger.info(f"移动平均平滑器: 窗口半径={radius} (窗口大小={2*radius+1}帧)")

    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """
        平滑轨迹序列

        Args:
            trajectory: 累积运动轨迹 (N, D)
                       N=帧数, D=运动参数维度

        Returns:
            平滑后的轨迹 (N, D)
        """
        N, D = trajectory.shape
        smoothed = np.zeros_like(trajectory)

        for k in range(N):
            # 计算窗口范围（边界clamp处理）
            j_start = max(0, k - self.radius)
            j_end = min(N - 1, k + self.radius)

            # 简单平均
            smoothed[k] = trajectory[j_start:j_end+1].mean(axis=0)

        return smoothed


class GaussianSmoother:
    """
    高斯加权移动平均平滑器

    比简单移动平均更平滑，权重随距离按高斯分布递减:
    g(j-k) = exp(-(j-k)²/(2σ²))

    等价于与高斯核卷积 → 频率域: H(f) = exp(-2π²f²σ²)
    真正的低通滤波器，无旁瓣，无振铃效应
    """

    def __init__(self, sigma: float = 10.0, radius: Optional[int] = None):
        """
        Args:
            sigma: 高斯标准差（控制平滑程度）
                  sigma大 → 更平滑（更强低通）
                  sigma小 → 保留更多原始运动（响应更快）
            radius: 窗口半径（None时取3σ）
        """
        self.sigma = sigma
        self.radius = radius if radius is not None else int(3 * sigma)
        logger.info(f"高斯平滑器: σ={sigma}, 窗口半径={self.radius}")

    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """
        高斯加权平滑

        Args:
            trajectory: 累积轨迹 (N, D)

        Returns:
            平滑轨迹 (N, D)
        """
        N, D = trajectory.shape
        smoothed = np.zeros_like(trajectory)

        # 预计算高斯权重
        offsets = np.arange(-self.radius, self.radius + 1)
        weights = np.exp(-offsets**2 / (2 * self.sigma**2))  # 未归一化

        for k in range(N):
            total_weight = 0.0
            weighted_sum = np.zeros(D)

            for idx, j_offset in enumerate(offsets):
                j = k + j_offset
                j = max(0, min(N - 1, j))  # Clamp 处理边界

                w = weights[idx]
                weighted_sum += w * trajectory[j]
                total_weight += w

            smoothed[k] = weighted_sum / total_weight

        return smoothed


class KalmanSmoother:
    """
    卡尔曼滤波器 - 实时在线视频防抖

    【状态空间模型】
    状态向量: x[k] = [pos, vel]ᵀ  (位置和速度)
    观测向量: z[k] = pos[k]        (只观测位置)

    状态转移（匀速模型，dt=1帧）:
    x[k] = F·x[k-1] + w[k]
    F = [1  1]    pos[k] = pos[k-1] + vel[k-1]
        [0  1]    vel[k] = vel[k-1]

    观测模型:
    z[k] = H·x[k] + v[k]
    H = [1  0]

    噪声协方差:
    Q = E[w·wᵀ] = diag(q_pos, q_vel)  (过程噪声)
    R = E[v·vᵀ] = r_obs               (观测噪声)

    物理含义:
    Q/R 比值越大 → 越相信观测（抖动补偿越强，但可能丢失有意运动）
    Q/R 比值越小 → 越相信模型（平滑更强，但响应较慢）

    【卡尔曼滤波迭代公式】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    时间更新（预测）:
    x̂⁻[k]  = F · x̂[k-1]                先验状态
    P⁻[k]   = F · P[k-1] · Fᵀ + Q        先验误差协方差

    量测更新（校正）:
    K[k] = P⁻[k] · Hᵀ · (H·P⁻[k]·Hᵀ + R)⁻¹  卡尔曼增益
    x̂[k] = x̂⁻[k] + K[k] · (z[k] - H·x̂⁻[k])   后验状态
    P[k] = (I - K[k]·H) · P⁻[k]                  后验误差协方差

    直觉解释:
    K[k] 在 [0,1] 之间，K→1 时更信任测量，K→0 时更信任预测
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1.0
    ):
        """
        Args:
            process_noise: 过程噪声方差 q
                          越大: 相机运动变化越剧烈（更灵活）
                          越小: 相机路径更平滑（更平稳）
            measurement_noise: 观测噪声方差 r
                              越大: 对测量值更不信任（平滑更强）
                              越小: 对测量值更信任（响应更快）
        """
        self.q = process_noise
        self.r = measurement_noise

        # 状态转移矩阵 F (匀速运动模型)
        # pos[k] = pos[k-1] + vel[k-1]
        # vel[k] = vel[k-1]
        self.F = np.array([[1., 1.],
                           [0., 1.]])

        # 观测矩阵 H (只观测位置)
        self.H = np.array([[1., 0.]])

        # 过程噪声协方差 Q
        self.Q = np.array([[self.q, 0.],
                           [0.,     self.q]])

        # 观测噪声协方差 R (标量)
        self.R = np.array([[self.r]])

        logger.info(f"卡尔曼滤波器: 过程噪声q={process_noise}, 观测噪声r={measurement_noise}")

    def smooth_1d(self, measurements: np.ndarray) -> np.ndarray:
        """
        对1D时间序列应用卡尔曼滤波（前向-后向双向）

        双向卡尔曼 (Rauch-Tung-Striebel Smoother):
        1. 前向卡尔曼: 得到因果估计 x̂[k|k]
        2. 后向平滑: 利用未来观测改善估计 x̂[k|N]
        公式: x̂[k|N] = x̂[k|k] + J[k]·(x̂[k+1|N] - x̂[k+1|k])
              J[k] = P[k|k]·Fᵀ·(P[k+1|k])⁻¹  (平滑增益)

        实际应用: 离线处理时使用双向提升平滑质量

        Args:
            measurements: 1D 观测序列 (N,)

        Returns:
            平滑后序列 (N,)
        """
        N = len(measurements)

        # ── 前向卡尔曼滤波 ────────────────────────────────
        # 初始化状态和协方差
        x = np.array([[measurements[0]], [0.]])  # [位置; 速度]
        P = np.eye(2) * 100.  # 初始不确定性大

        # 存储前向估计（用于后向平滑）
        xs_forward = np.zeros((N, 2))    # x̂[k|k]
        Ps_forward = np.zeros((N, 2, 2)) # P[k|k]
        xs_predict = np.zeros((N, 2))    # x̂[k|k-1]
        Ps_predict = np.zeros((N, 2, 2)) # P[k|k-1]

        for k in range(N):
            z = np.array([[measurements[k]]])

            # ── 预测步骤 ──────────────────────────────
            # 先验状态: x̂⁻[k] = F · x̂[k-1]
            x_pred = self.F @ x
            # 先验协方差: P⁻[k] = F·P·Fᵀ + Q
            P_pred = self.F @ P @ self.F.T + self.Q

            xs_predict[k] = x_pred.flatten()
            Ps_predict[k] = P_pred

            # ── 更新步骤 ──────────────────────────────
            # 新息: y[k] = z[k] - H·x̂⁻[k]
            innovation = z - self.H @ x_pred

            # 新息协方差: S = H·P⁻·Hᵀ + R
            S = self.H @ P_pred @ self.H.T + self.R

            # 卡尔曼增益: K = P⁻·Hᵀ·S⁻¹
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            # 后验状态: x̂[k] = x̂⁻[k] + K·y[k]
            x = x_pred + K @ innovation

            # 后验协方差: P[k] = (I-K·H)·P⁻[k]
            P = (np.eye(2) - K @ self.H) @ P_pred

            xs_forward[k] = x.flatten()
            Ps_forward[k] = P

        # ── 后向 RTS 平滑 ─────────────────────────────────
        # Rauch-Tung-Striebel 后向平滑
        xs_smooth = xs_forward.copy()
        Ps_smooth = Ps_forward.copy()

        for k in range(N - 2, -1, -1):
            # 平滑增益: J[k] = P[k|k]·Fᵀ·(P[k+1|k])⁻¹
            J = Ps_forward[k] @ self.F.T @ np.linalg.inv(Ps_predict[k + 1])

            # 平滑状态: x̂[k|N] = x̂[k|k] + J·(x̂[k+1|N] - x̂[k+1|k])
            xs_smooth[k] = (
                xs_forward[k]
                + J @ (xs_smooth[k + 1] - xs_predict[k + 1])
            )

            # 平滑协方差
            Ps_smooth[k] = (
                Ps_forward[k]
                + J @ (Ps_smooth[k + 1] - Ps_predict[k + 1]) @ J.T
            )

        # 返回位置分量
        return xs_smooth[:, 0]

    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """
        对多维轨迹的每个维度独立应用卡尔曼平滑

        Args:
            trajectory: (N, D) 轨迹矩阵

        Returns:
            (N, D) 平滑后的轨迹
        """
        N, D = trajectory.shape
        smoothed = np.zeros_like(trajectory)

        # 对每个运动参数维度（dx, dy, da, ds）独立平滑
        for d in range(D):
            smoothed[:, d] = self.smooth_1d(trajectory[:, d])

        return smoothed


class L1TrajectorySmoother:
    """
    L1 最优轨迹平滑器
    参考: Liu et al. "Subspace Video Stabilization" SIGGRAPH 2011

    【L1 优化原理】
    最小化目标函数:
    E(p̂) = λ·Σₖ||p̂[k] - p[k]||² + Σₖ||p̂[k] - p̂[k-1]||₁

    - 第一项: 数据忠诚项（不偏离原始轨迹太远）
    - 第二项: L1 平滑项（允许轨迹有"稀疏跳变"）

    L1 平滑的优势:
    - 对突变（如快速摇摄）保持边缘锐利
    - 不像 L2 那样过度平滑意图运动
    - 手抖（高频小抖动）被 L1 惩罚并消除

    数值求解: ADMM（交替方向乘子法）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    引入辅助变量 z[k] = p̂[k] - p̂[k-1]:
    最小化: λ·||p̂-p||² + ||z||₁
    约束:   D·p̂ = z  (D为差分算子)

    增广拉格朗日:
    L(p̂, z, u) = λ·||p̂-p||² + ||z||₁ + (ρ/2)||D·p̂-z+u||²

    ADMM 交替更新:
    p̂ := argmin_p L  → 最小二乘（有闭合解）
    z  := argmin_z L  → 软阈值收缩 prox_{1/ρ·||·||₁}
    u  := u + D·p̂ - z (对偶更新)

    软阈值算子: S_λ(x) = sign(x)·max(|x|-λ, 0)
    """

    def __init__(self, lambda_: float = 1.0, rho: float = 1.0, max_iter: int = 100):
        """
        Args:
            lambda_: 数据忠诚权重（越大越保留原始运动）
            rho: ADMM 惩罚参数（影响收敛速度）
            max_iter: ADMM 最大迭代次数
        """
        self.lambda_ = lambda_
        self.rho = rho
        self.max_iter = max_iter

    def smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """
        L1 最优轨迹平滑（对每个维度独立处理）

        Args:
            trajectory: (N, D) 原始轨迹

        Returns:
            (N, D) L1 优化后的轨迹
        """
        N, D = trajectory.shape
        smoothed = np.zeros_like(trajectory)

        for d in range(D):
            smoothed[:, d] = self._smooth_1d_admm(trajectory[:, d])

        return smoothed

    def _smooth_1d_admm(self, p: np.ndarray) -> np.ndarray:
        """
        1D L1 平滑（ADMM求解）

        最小化: (λ/2)||p̂-p||² + ||D·p̂||₁
        其中 D 为一阶差分矩阵

        Args:
            p: 1D 观测轨迹 (N,)

        Returns:
            平滑轨迹 (N,)
        """
        N = len(p)
        lam = self.lambda_
        rho = self.rho

        # 构建一阶差分矩阵 D (N-1 × N)
        # D[k, k] = -1, D[k, k+1] = 1
        D_mat = np.zeros((N - 1, N))
        for k in range(N - 1):
            D_mat[k, k] = -1.
            D_mat[k, k + 1] = 1.

        # 预计算: (λI + ρ·DᵀD)⁻¹ （只需计算一次）
        A = lam * np.eye(N) + rho * D_mat.T @ D_mat
        A_inv = np.linalg.inv(A)

        # 初始化 ADMM 变量
        p_hat = p.copy()  # 主变量
        z = np.zeros(N - 1)  # 辅助变量 z = D·p̂
        u = np.zeros(N - 1)  # 对偶变量（缩放后的拉格朗日乘子）

        for _ in range(self.max_iter):
            # ── p̂ 更新（最小二乘，有闭合解）────────────────
            rhs = lam * p + rho * D_mat.T @ (z - u)
            p_hat = A_inv @ rhs

            # ── z 更新（软阈值收缩，proximal operator of ||·||₁）
            # z := S_{1/ρ}(D·p̂ + u)
            Dp = D_mat @ p_hat
            z_old = z.copy()
            threshold = 1.0 / rho
            z = np.sign(Dp + u) * np.maximum(np.abs(Dp + u) - threshold, 0.)

            # ── 对偶变量更新 ─────────────────────────────────
            u = u + Dp - z

            # ── 检查收敛 ─────────────────────────────────────
            # 原始残差: r = D·p̂ - z
            # 对偶残差: s = ρ·Dᵀ·(z - z_old)
            primal_res = np.linalg.norm(Dp - z)
            dual_res = rho * np.linalg.norm(D_mat.T @ (z - z_old))

            if primal_res < 1e-4 and dual_res < 1e-4:
                break

        return p_hat
