"""
光流估计模块 - 视频防抖算法
Optical Flow Estimation Module for Video Stabilization

【光流基本原理】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
光流 (Optical Flow) 描述了视频中像素的运动场。
对于相机抖动引起的全局运动，我们希望估计帧间的整体变换。

光流约束方程 (Brightness Constancy Assumption):
    I(x, y, t) = I(x+u, y+v, t+1)
    其中 (u, v) 是像素的运动向量

泰勒展开（线性化）得到光流方程:
    Ix·u + Iy·v + It = 0
    其中 Ix = ∂I/∂x, Iy = ∂I/∂y, It = ∂I/∂t

这是一个约束方程（孔径问题），每个像素给出1个方程2个未知数，
需要额外约束才能唯一求解。

【Lucas-Kanade 稀疏光流】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
假设 W×W 窗口内运动相同（局部平滑约束），
对窗口内所有像素列方程组:

    A · d = b
    [Ix₁ Iy₁] [u]   [-It₁]
    [Ix₂ Iy₂] [v] = [-It₂]
    [  ⋮   ⋮ ]       [  ⋮ ]

最小二乘解 (Normal Equation):
    d = (AᵀA)⁻¹ · Aᵀ · b

其中:
    AᵀA = [Σ Ix² , Σ IxIy]  (2×2 结构张量 / Harris 矩阵)
          [Σ IxIy, Σ Iy²  ]

    Aᵀb = [Σ IxIt]
          [Σ IyIt]

AᵀA 行列式大则光流可靠（角点特性）

多尺度 LK 光流（金字塔）:
在图像金字塔各层从粗到细迭代，处理大位移:
  粗层: 大运动变小，便于估计
  细层: 精化估计，提高精度

【Farneback 稠密光流】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
将图像局部用二次多项式近似:
    f(x) ≈ xᵀAx + bᵀx + c

通过比较相邻帧的多项式展开系数计算稠密光流：
    A₁(x) = A₂(x+d)  → 求解 d

全局运动估计（用于视频防抖）:
    将稠密光流转化为仿射变换参数
    [u(x,y)]   [a₁₁ a₁₂] [x]   [t_x]
    [v(x,y)] = [a₂₁ a₂₂]·[y] + [t_y]
    最小二乘拟合全局仿射参数
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class LucasKanadeFlow:
    """
    Lucas-Kanade 稀疏光流估计器（用于特征点追踪）

    适用于视频防抖中的全局运动估计：
    追踪角点特征的运动 → 估计全局变换（平移/旋转/缩放）
    """

    def __init__(
        self,
        win_size: int = 21,
        max_level: int = 3,
        max_iter: int = 30,
        epsilon: float = 0.01,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: float = 10.0
    ):
        """
        Args:
            win_size: LK 算法窗口大小 W（W×W 邻域）
                     越大能处理更大位移，但计算越慢
            max_level: 图像金字塔层数（越多能处理越大的全局运动）
            max_iter: 每层最大迭代次数
            epsilon: 迭代收敛阈值 ||d_k - d_{k-1}|| < epsilon
            max_corners: Shi-Tomasi 检测的最大角点数
            quality_level: 角点质量阈值（相对于最强角点的比例）
            min_distance: 角点间最小距离（像素）
        """
        # LK 光流参数
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iter,   # 最大迭代次数
                epsilon     # 收敛精度
            )
        )

        # Shi-Tomasi 角点检测参数（选好特征点再追踪）
        self.corner_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,  # 角点响应强度阈值
            minDistance=min_distance,    # 角点间最小距离（NMS）
            blockSize=7                  # 计算梯度的窗口大小
        )

        self.prev_gray = None
        self.prev_points = None

    def detect_features(self, gray: np.ndarray) -> np.ndarray:
        """
        检测 Shi-Tomasi 角点（用于追踪）

        【Shi-Tomasi vs Harris 角点】
        Harris 评分: R = det(M) - k·trace(M)²
        Shi-Tomasi 评分: R = min(λ₁, λ₂)  (最小特征值)

        Shi-Tomasi 更保守，选出的特征点更适合追踪
        M = [Σ Gσ·Ix²  Σ Gσ·IxIy]  (加权结构张量)
            [Σ Gσ·IxIy Σ Gσ·Iy²  ]

        Args:
            gray: 灰度图像

        Returns:
            特征点坐标 (N, 1, 2)
        """
        points = cv2.goodFeaturesToTrack(gray, **self.corner_params)
        return points

    def track(
        self,
        curr_gray: np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
        prev_points: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        追踪特征点并估计帧间仿射变换

        【追踪流程】
        1. 若无前帧特征点，检测新特征点
        2. LK 光流追踪: 从前帧特征点位置追踪到当前帧
        3. 过滤追踪失败点（状态=0）和误差大的点
        4. 用有效追踪点对估计全局仿射变换

        Args:
            curr_gray: 当前帧灰度图
            prev_gray: 前一帧灰度图（None使用内部存储）
            prev_points: 前一帧特征点（None使用内部存储）

        Returns:
            transform: 2×3 仿射变换矩阵
            p0_valid: 有效的前帧特征点
            p1_valid: 对应的当前帧特征点
        """
        # 使用传入参数或内部存储
        p0_gray = prev_gray if prev_gray is not None else self.prev_gray
        p0_pts = prev_points if prev_points is not None else self.prev_points

        # ── 首帧或特征点不足: 重新检测 ──────────────────────
        if p0_gray is None or p0_pts is None or len(p0_pts) < 10:
            self.prev_gray = curr_gray.copy()
            self.prev_points = self.detect_features(curr_gray)
            if self.prev_points is None:
                return None, None, None
            logger.debug(f"重新检测特征点: {len(self.prev_points)} 个")
            return None, None, None

        # ── LK 金字塔光流追踪 ────────────────────────────────
        # p1: 当前帧预测位置
        # status: 1=追踪成功, 0=追踪失败
        # err: 追踪误差（光度残差）
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            p0_gray, curr_gray,
            p0_pts, None,
            **self.lk_params
        )

        if p1 is None:
            return None, None, None

        # ── 过滤无效追踪点 ───────────────────────────────────
        # status==1: 追踪成功的点
        good_old = p0_pts[status == 1]  # 前帧有效点
        good_new = p1[status == 1]      # 当前帧对应点

        n_valid = len(good_old)
        logger.debug(f"光流追踪: {len(p0_pts)} → {n_valid} 有效点")

        if n_valid < 6:
            # 有效点不足，重新检测
            self.prev_gray = curr_gray.copy()
            self.prev_points = self.detect_features(curr_gray)
            return None, None, None

        # ── 估计全局仿射变换（RANSAC鲁棒估计）─────────────────
        # 全局变换模型（6自由度仿射）:
        # [x']   [a  b  tx] [x]
        # [y'] = [c  d  ty]·[y]
        #                   [1]
        # RANSAC 过滤运动场中的局部运动（非刚体运动的像素，如行人）
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            good_old, good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,    # 内点误差阈值（像素）
            confidence=0.99
        )

        # 更新前帧信息
        self.prev_gray = curr_gray.copy()

        # 定期重新检测特征点（当前帧成为下一帧的"前帧"）
        if n_valid < 50:
            # 特征点减少时重新检测
            self.prev_points = self.detect_features(curr_gray)
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)

        return transform, good_old, good_new


class FarnebackFlow:
    """
    Farneback 稠密光流估计器

    计算每个像素的运动向量，然后全局拟合仿射变换
    比稀疏LK更稳健，但计算量更大

    【Farneback 算法原理】
    1. 多尺度金字塔: 从粗到细估计运动
    2. 多项式近似: 局部图像用二次多项式拟合
       f₁(x) ≈ xᵀA₁x + b₁ᵀx + c₁
       f₂(x) ≈ xᵀA₂x + b₂ᵀx + c₂
    3. 若 f₁(x) = f₂(x-d)，则通过比较多项式系数求 d:
       A₁ = A₂, b₁ = b₂ + 2A₁d
       → d = A₁⁻¹(b₂-b₁)/2

    全局运动估计（仿射拟合）:
    对所有像素 (x_i, y_i) 和其光流 (u_i, v_i):
    最小化 Σᵢ ||[a·xᵢ+b·yᵢ+tx, c·xᵢ+d·yᵢ+ty] - [uᵢ, vᵢ]||²
    用最小二乘求解 [a, b, tx, c, d, ty]
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        win_size: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2
    ):
        """
        Args:
            pyr_scale: 金字塔每层缩放比例 (0.5 = 每层减半)
            levels: 金字塔层数
            win_size: 平均窗口大小（影响速度/鲁棒性）
            iterations: 每层迭代次数
            poly_n: 多项式展开邻域大小 (5或7)
            poly_sigma: 多项式展开高斯标准差
        """
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.win_size = win_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def estimate_transform(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        估计两帧间的全局仿射变换

        【步骤】
        1. 计算稠密光流场 (u, v) - 每像素的运动向量
        2. 创建像素坐标网格 (x, y)
        3. 用最小二乘将光流拟合为全局仿射变换
           去除因场景动态物体导致的局部非全局运动（鲁棒估计）

        Args:
            prev_gray, curr_gray: 相邻两帧灰度图

        Returns:
            2×3 仿射变换矩阵，失败返回 None
        """
        # ── 计算 Farneback 稠密光流 ──────────────────────────
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,               # 初始流场（None=从零开始）
            self.pyr_scale,     # 金字塔缩放比例
            self.levels,        # 金字塔层数
            self.win_size,      # 平均窗口大小
            self.iterations,    # 每层迭代次数
            self.poly_n,        # 多项式邻域大小
            self.poly_sigma,    # 多项式高斯标准差
            0                   # 标志（0=无特殊选项）
        )
        # flow: (H, W, 2)  flow[y,x] = (u, v) 像素运动向量

        h, w = prev_gray.shape

        # ── 创建像素坐标网格 ─────────────────────────────────
        # xs: (H, W) 每个像素的x坐标
        # ys: (H, W) 每个像素的y坐标
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))

        # 展平为向量
        xs_flat = xs.flatten().astype(np.float32)
        ys_flat = ys.flatten().astype(np.float32)
        us_flat = flow[:, :, 0].flatten()  # x方向运动
        vs_flat = flow[:, :, 1].flatten()  # y方向运动

        # ── 最小二乘拟合全局仿射变换（鲁棒版本）────────────
        # 过滤光流幅度的极端值（可能是遮挡或错误匹配）
        flow_magnitude = np.sqrt(us_flat**2 + vs_flat**2)
        median_mag = np.median(flow_magnitude)
        valid_mask = flow_magnitude < (median_mag * 5 + 1)  # 过滤离群值

        if valid_mask.sum() < 100:
            logger.warning("有效光流点不足，跳过当前帧")
            return None

        xs_v = xs_flat[valid_mask]
        ys_v = ys_flat[valid_mask]
        us_v = us_flat[valid_mask]
        vs_v = vs_flat[valid_mask]

        # 目标点 = 源点 + 光流
        src_pts = np.column_stack([xs_v, ys_v]).astype(np.float32)
        dst_pts = np.column_stack([xs_v + us_v, ys_v + vs_v]).astype(np.float32)

        # 使用 RANSAC 鲁棒仿射估计（去除动态物体影响）
        # estimateAffinePartial2D: 只估计4DoF (平移+旋转+缩放)
        transform, _ = cv2.estimateAffinePartial2D(
            src_pts[::10],  # 采样1/10点加速
            dst_pts[::10],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )

        return transform


class MotionEstimator:
    """
    帧间运动估计器 - 整合多种光流方法

    自动选择最优方法并提取平移、旋转、缩放参数
    """

    def __init__(self, method: str = 'lk', **kwargs):
        """
        Args:
            method: 光流方法 'lk'（稀疏LK）或 'farneback'（稠密）
        """
        self.method = method
        if method == 'lk':
            self.flow_estimator = LucasKanadeFlow(**kwargs)
        else:
            self.flow_estimator = FarnebackFlow(**kwargs)

        self.prev_gray = None

    def estimate(
        self, frame: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        估计当前帧相对于前帧的运动参数

        【仿射矩阵参数提取】
        仿射矩阵 M = [a  b  tx]
                     [c  d  ty]

        等价于:
        M = s · [cos(θ) -sin(θ)] + [tx]
                [sin(θ)  cos(θ)]   [ty]

        其中:
        - 旋转角度: θ = arctan2(b, a)  (近似纯旋转时)
        - 缩放因子: s = √(a² + b²)
        - 平移: (tx, ty) = (M[0,2], M[1,2])

        Returns:
            (dx, dy, da, ds): 平移x, 平移y, 旋转角(弧度), 缩放因子
        """
        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame.copy()

        # 首帧无运动
        if self.prev_gray is None:
            self.prev_gray = curr_gray.copy()
            return 0.0, 0.0, 0.0, 1.0

        # 估计变换矩阵
        if self.method == 'lk':
            transform, _, _ = self.flow_estimator.track(curr_gray, self.prev_gray)
        else:
            transform = self.flow_estimator.estimate_transform(
                self.prev_gray, curr_gray
            )

        self.prev_gray = curr_gray.copy()

        if transform is None:
            return 0.0, 0.0, 0.0, 1.0

        # ── 从仿射矩阵提取运动参数 ────────────────────────────
        # 平移参数
        dx = transform[0, 2]  # tx
        dy = transform[1, 2]  # ty

        # 旋转角度: θ = arctan2(M[1,0], M[0,0])
        # 注: M = s·R + t，其中 R=[cos θ, -sin θ; sin θ, cos θ]
        da = np.arctan2(transform[1, 0], transform[0, 0])

        # 缩放因子: s = √(M[0,0]² + M[1,0]²)
        ds = np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)

        return dx, dy, da, ds
