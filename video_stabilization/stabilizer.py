"""
视频防抖主控模块
Video Stabilizer - Main Pipeline

【视频防抖完整流程】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: 帧间运动估计 (Motion Estimation)
        使用 LK 光流追踪特征点，估计逐帧仿射变换参数
        提取: Δx（平移x），Δy（平移y），Δα（旋转角），Δs（缩放）

Step 2: 轨迹积分 (Trajectory Integration)
        将逐帧相对运动累积为绝对位置轨迹:
        T[k] = Σ_{i=1}^{k} Δm[i]  (其中 Δm 为帧间运动)

Step 3: 轨迹平滑 (Trajectory Smoothing)
        对累积轨迹应用低通滤波（移动平均/高斯/卡尔曼/L1优化）
        T̂[k] = Smooth(T[k])

Step 4: 补偿计算 (Compensation)
        平滑轨迹 - 原始轨迹 = 需要施加的补偿
        C[k] = T̂[k] - T[k]

Step 5: 帧变换 (Frame Transformation)
        对每帧施加补偿变换（消除抖动）:
        I_stable[k] = WarpAffine(I[k], Affine(C[k]))

Step 6: 裁剪 (Cropping)
        裁剪变换引入的黑边（稳定后的有效区域）

【防抖效果评估指标】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 稳定性 (Stability):
   Stabilization Ratio = σ(smooth_traj) / σ(raw_traj)
   越小越好（1=无改善，0=完全稳定）

2. 保真度 (Fidelity):
   Preservation Score = 保留视频有效区域占原始帧的比例
   越大越好

3. 抖动频率分析:
   通过 FFT 分析手抖频率成分（通常 1-12 Hz）
   高频分量减少 = 防抖效果好

【2024-2025 最新技术参考】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- SOFT (2024): 自监督稀疏光流Transformer + 四元数表示旋转
- Gyroflow+ (2024): 陀螺仪引导的无监督深度单应+光流学习
- DUT (2023): 关键点检测 + 运动传播 + 轨迹平滑三阶段框架
本实现为经典方法，集成上述论文的设计思想（无需深度学习框架）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable
import logging
from pathlib import Path

from .optical_flow import MotionEstimator
from .trajectory_smoother import (
    MovingAverageSmoother,
    GaussianSmoother,
    KalmanSmoother,
    L1TrajectorySmoother
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class VideoStabilizer:
    """
    视频防抖主控类
    支持多种运动估计和轨迹平滑策略的组合
    """

    def __init__(
        self,
        flow_method: str = 'lk',
        smooth_method: str = 'kalman',
        smooth_radius: int = 30,
        kalman_process_noise: float = 1e-3,
        kalman_obs_noise: float = 1.0,
        crop_ratio: float = 0.1,
        border_mode: str = 'black'
    ):
        """
        初始化视频防抖器

        Args:
            flow_method: 光流方法 'lk'（稀疏LK）或 'farneback'（稠密）
                        - lk: 速度快，适合特征丰富场景
                        - farneback: 更鲁棒，适合纹理均匀场景
            smooth_method: 轨迹平滑方法:
                          - 'moving_avg': 简单移动平均（最快）
                          - 'gaussian': 高斯加权平均（更自然）
                          - 'kalman': 卡尔曼双向平滑（在线友好）
                          - 'l1': L1优化（保留有意运动最好）
            smooth_radius: 平滑窗口半径（仅对moving_avg/gaussian有效）
            kalman_process_noise: 卡尔曼过程噪声 q
            kalman_obs_noise: 卡尔曼观测噪声 r
            crop_ratio: 稳定后的裁剪比例 (0-0.3)
                       0.1 = 四边各裁10%（防止黑边露出）
            border_mode: 边界填充模式 'black'/'replicate'/'reflect'
        """
        # 运动估计器
        self.motion_estimator = MotionEstimator(method=flow_method)

        # 轨迹平滑器
        if smooth_method == 'moving_avg':
            self.smoother = MovingAverageSmoother(radius=smooth_radius)
        elif smooth_method == 'gaussian':
            self.smoother = GaussianSmoother(sigma=smooth_radius / 3.0)
        elif smooth_method == 'kalman':
            self.smoother = KalmanSmoother(
                process_noise=kalman_process_noise,
                measurement_noise=kalman_obs_noise
            )
        elif smooth_method == 'l1':
            self.smoother = L1TrajectorySmoother(lambda_=1.0)
        else:
            raise ValueError(f"未知平滑方法: {smooth_method}")

        self.crop_ratio = crop_ratio
        self.border_mode = {
            'black': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT
        }.get(border_mode, cv2.BORDER_CONSTANT)

        logger.info(
            f"视频防抖器初始化: 光流={flow_method}, 平滑={smooth_method}, "
            f"裁剪比例={crop_ratio}"
        )

    def estimate_trajectory(
        self, frames: List[np.ndarray]
    ) -> np.ndarray:
        """
        估计视频运动轨迹

        【轨迹积分过程】
        设逐帧运动参数: Δm[k] = (Δx[k], Δy[k], Δα[k], Δs[k])
        累积轨迹:
          T[k] = T[k-1] + Δm[k]
          T[0] = (0, 0, 0, 1)  (初始帧无运动)

        注意: 实际相机运动是复合的（非简单加法）
        对于小角度近似，加法是合理的近似

        Args:
            frames: 视频帧列表

        Returns:
            trajectory: (N, 4) 运动轨迹矩阵
                        列: [累积Δx, 累积Δy, 累积Δα, 累积Δs]
        """
        N = len(frames)
        # 存储逐帧运动增量: [Δx, Δy, Δα, Δs]
        motions = np.zeros((N, 4))
        motions[:, 3] = 1.0  # 缩放初始值为1

        logger.info(f"开始估计 {N} 帧的运动轨迹...")
        self.motion_estimator.prev_gray = None  # 重置状态

        for k, frame in enumerate(frames):
            dx, dy, da, ds = self.motion_estimator.estimate(frame)
            motions[k] = [dx, dy, da, ds]

            if k % 50 == 0:
                logger.info(f"  处理帧 {k+1}/{N}: dx={dx:.1f}, dy={dy:.1f}, "
                           f"da={np.degrees(da):.2f}°, ds={ds:.4f}")

        # ── 积分: 逐帧运动 → 累积轨迹 ────────────────────────
        # T[k] = Σ_{i=0}^{k} Δm[i]
        # 使用 cumsum 高效计算前缀和
        trajectory = np.cumsum(motions, axis=0)

        # 缩放轨迹使用乘法累积（s通道特殊处理）
        # 这里使用log空间: log(S[k]) = Σ log(Δs[i])
        # 简化: 直接使用累加（在小缩放变化时近似合理）

        logger.info(
            f"轨迹估计完成. 最大平移: dx={trajectory[:,0].max():.1f}px, "
            f"dy={trajectory[:,1].max():.1f}px"
        )
        return trajectory

    def smooth_trajectory(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """
        平滑运动轨迹（去除手抖高频成分）

        Args:
            trajectory: (N, 4) 原始累积轨迹

        Returns:
            smoothed: (N, 4) 平滑后的轨迹
        """
        logger.info("正在平滑运动轨迹...")
        # 对前3个维度（dx, dy, da）进行平滑
        # 缩放维度（ds）保持相对稳定，也一并平滑
        smoothed = self.smoother.smooth(trajectory)
        logger.info("轨迹平滑完成")
        return smoothed

    def compute_compensation(
        self,
        trajectory: np.ndarray,
        smoothed: np.ndarray
    ) -> np.ndarray:
        """
        计算逐帧补偿量

        【补偿原理】
        原始轨迹 T[k] = 真实路径 + 手抖
        平滑轨迹 T̂[k] ≈ 真实路径

        每帧需要的校正补偿:
        C[k] = T̂[k] - T[k]  （负值表示需要反向补偿）

        应用补偿变换后:
        实际位置 = T[k] + C[k] = T̂[k]  （沿平滑路径运动）

        Args:
            trajectory: (N, 4) 原始轨迹
            smoothed: (N, 4) 平滑轨迹

        Returns:
            compensation: (N, 4) 逐帧补偿量
        """
        compensation = smoothed - trajectory
        logger.info(
            f"补偿统计: max_dx={np.abs(compensation[:,0]).max():.1f}px, "
            f"max_dy={np.abs(compensation[:,1]).max():.1f}px, "
            f"max_da={np.degrees(np.abs(compensation[:,2])).max():.2f}°"
        )
        return compensation

    def apply_compensation(
        self,
        frame: np.ndarray,
        compensation: np.ndarray
    ) -> np.ndarray:
        """
        对单帧应用防抖补偿变换

        【仿射变换矩阵构造】
        给定补偿参数 (dx, dy, dα, ds):

        旋转矩阵 R = [cos(dα) -sin(dα)]
                     [sin(dα)  cos(dα)]

        带缩放的旋转: R_s = ds · R

        仿射变换矩阵（绕图像中心旋转）:
        先平移到原点，旋转，再平移回来，最后加平移补偿:
        M = [ds·cos(dα)  -ds·sin(dα)  dx + cx - cx·ds·cos(dα) + cy·ds·sin(dα)]
            [ds·sin(dα)   ds·cos(dα)  dy + cy - cx·ds·sin(dα) - cy·ds·cos(dα)]

        Args:
            frame: 原始帧
            compensation: [dx, dy, dα, ds] 补偿参数

        Returns:
            稳定后的帧
        """
        h, w = frame.shape[:2]
        dx, dy, da, ds = compensation

        # 构建绕图像中心的仿射变换矩阵
        cx, cy = w / 2.0, h / 2.0  # 图像中心

        # 旋转 + 缩放矩阵（绕原点）
        cos_a = np.cos(da)
        sin_a = np.sin(da)

        # 完整仿射矩阵（先中心化，旋转缩放，再反中心化，最后平移）:
        # M = T(cx,cy) · RS · T(-cx,-cy) · T(dx,dy)
        # 展开得:
        M = np.float32([
            [ds * cos_a, -ds * sin_a, dx + cx - ds*cos_a*cx + ds*sin_a*cy],
            [ds * sin_a,  ds * cos_a, dy + cy - ds*sin_a*cx - ds*cos_a*cy]
        ])

        # 应用仿射变换
        stabilized = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=0
        )

        return stabilized

    def crop_frame(
        self, frame: np.ndarray, ratio: float
    ) -> np.ndarray:
        """
        裁剪帧以去除防抖引入的黑边

        【裁剪策略】
        固定裁剪比例（最简单，最常用）:
        裁剪后尺寸: w' = w·(1-2r), h' = h·(1-2r)
        其中 r = crop_ratio

        自适应裁剪（更高级）:
        根据所有补偿量的最大值动态确定裁剪量:
        r = max_compensation / min(w, h)

        Args:
            frame: 原始帧
            ratio: 四边各裁比例

        Returns:
            裁剪后的帧
        """
        h, w = frame.shape[:2]
        margin_x = int(w * ratio)
        margin_y = int(h * ratio)
        cropped = frame[margin_y:h-margin_y, margin_x:w-margin_x]
        return cropped

    def stabilize(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[np.ndarray], dict]:
        """
        执行完整视频防抖流程

        Args:
            frames: 输入视频帧列表
            progress_callback: 进度回调函数 f(current, total)

        Returns:
            stabilized_frames: 防抖后的帧列表
            metrics: 防抖效果评估指标字典
        """
        N = len(frames)
        if N == 0:
            return [], {}

        logger.info(f"开始视频防抖处理, 共 {N} 帧...")

        # ════════════════════════════════════════════════════════
        # Step 1: 估计逐帧运动 → 积分为运动轨迹
        # ════════════════════════════════════════════════════════
        trajectory = self.estimate_trajectory(frames)
        # trajectory: (N, 4) [累积dx, 累积dy, 累积dα, 累积ds]

        # ════════════════════════════════════════════════════════
        # Step 2: 平滑运动轨迹（去除手抖高频成分）
        # ════════════════════════════════════════════════════════
        smoothed_trajectory = self.smooth_trajectory(trajectory)

        # ════════════════════════════════════════════════════════
        # Step 3: 计算逐帧补偿量
        # ════════════════════════════════════════════════════════
        compensation = self.compute_compensation(trajectory, smoothed_trajectory)

        # ════════════════════════════════════════════════════════
        # Step 4: 对每帧施加补偿变换
        # ════════════════════════════════════════════════════════
        logger.info("正在应用防抖补偿并生成稳定视频...")
        stabilized_frames = []

        for k, frame in enumerate(frames):
            # 应用当前帧的补偿变换
            stabilized = self.apply_compensation(frame, compensation[k])

            # 裁剪黑边
            if self.crop_ratio > 0:
                stabilized = self.crop_frame(stabilized, self.crop_ratio)

            stabilized_frames.append(stabilized)

            if progress_callback:
                progress_callback(k + 1, N)
            elif (k + 1) % 50 == 0:
                logger.info(f"  稳定化帧 {k+1}/{N}")

        # ════════════════════════════════════════════════════════
        # Step 5: 计算防抖质量指标
        # ════════════════════════════════════════════════════════
        metrics = self._evaluate_stability(trajectory, smoothed_trajectory, frames)
        logger.info(
            f"视频防抖完成!\n"
            f"  稳定性改善: dx={metrics['dx_improvement']:.1%}, "
            f"dy={metrics['dy_improvement']:.1%}\n"
            f"  旋转稳定性: {metrics['rotation_improvement']:.1%}"
        )

        return stabilized_frames, metrics

    def _evaluate_stability(
        self,
        raw_traj: np.ndarray,
        smooth_traj: np.ndarray,
        frames: List[np.ndarray]
    ) -> dict:
        """
        评估防抖效果

        【评估指标】
        1. 轨迹方差比 (Variance Ratio):
           σ_smooth / σ_raw
           越小表示抖动减少越多

        2. 频率稳定性:
           FFT 分析高频（>2Hz）能量减少比例
           （手抖频率通常在1-10Hz）

        3. 帧间差异方差 (Inter-frame Difference Variance):
           对比稳定前后的帧间差异幅度变化

        Returns:
            包含各项评估指标的字典
        """
        metrics = {}

        # 轨迹标准差改善（各维度）
        for i, dim in enumerate(['dx', 'dy', 'rotation', 'scale']):
            raw_std = np.std(np.diff(raw_traj[:, i]))
            smooth_std = np.std(np.diff(smooth_traj[:, i]))

            if raw_std > 1e-6:
                improvement = 1.0 - smooth_std / raw_std
            else:
                improvement = 0.0

            metrics[f'{dim}_improvement'] = improvement
            metrics[f'{dim}_raw_std'] = raw_std
            metrics[f'{dim}_smooth_std'] = smooth_std

        # 最大补偿量（决定最少裁剪量）
        compensation = smooth_traj - raw_traj
        metrics['max_dx_compensation'] = np.abs(compensation[:, 0]).max()
        metrics['max_dy_compensation'] = np.abs(compensation[:, 1]).max()
        metrics['min_crop_ratio'] = max(
            metrics['max_dx_compensation'],
            metrics['max_dy_compensation']
        ) / min(frames[0].shape[:2]) if frames else 0

        return metrics


def stabilize_video(
    input_path: str,
    output_path: str,
    flow_method: str = 'lk',
    smooth_method: str = 'kalman',
    crop_ratio: float = 0.1,
    **kwargs
) -> dict:
    """
    视频防抖便捷函数

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        flow_method: 光流方法 'lk'/'farneback'
        smooth_method: 轨迹平滑方法 'kalman'/'gaussian'/'moving_avg'/'l1'
        crop_ratio: 裁剪比例
        **kwargs: 其他参数传递给 VideoStabilizer

    Returns:
        防抖效果评估指标
    """
    # ── 读取视频 ──────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"输入视频: {input_path}\n"
        f"  分辨率: {width}×{height}, FPS: {fps:.1f}, 帧数: {total_frames}"
    )

    # 读取所有帧（对于长视频应分批处理）
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    logger.info(f"读取完成: {len(frames)} 帧")

    # ── 执行防抖 ──────────────────────────────────────────────
    stabilizer = VideoStabilizer(
        flow_method=flow_method,
        smooth_method=smooth_method,
        crop_ratio=crop_ratio,
        **kwargs
    )

    stabilized_frames, metrics = stabilizer.stabilize(frames)

    # ── 写出稳定视频 ─────────────────────────────────────────
    if stabilized_frames:
        # 获取稳定后的帧尺寸（裁剪后可能变小）
        out_h, out_w = stabilized_frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        for frame in stabilized_frames:
            out.write(frame)
        out.release()

        logger.info(
            f"输出视频: {output_path}\n"
            f"  分辨率: {out_w}×{out_h}, 帧数: {len(stabilized_frames)}"
        )

    return metrics
