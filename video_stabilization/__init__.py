"""
视频防抖算法包
Video Stabilization Algorithm Package

模块结构:
- optical_flow: Lucas-Kanade / Farneback 光流估计与运动参数提取
- trajectory_smoother: 移动平均 / 高斯 / 卡尔曼 / L1 轨迹平滑
- stabilizer: 视频防抖主控流程

【算法选择指南】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
光流方法:
  lk (Lucas-Kanade): 快速，适合大多数场景
  farneback: 稠密，对纹理少的场景更鲁棒

平滑方法:
  kalman:     实时防抖首选，双向RTS平滑质量最高
  gaussian:   简单高效，适合轻微抖动
  moving_avg: 最简单，但在边界处效果较差
  l1:         保留有意摇摄运动最好，计算较慢
"""

from .stabilizer import VideoStabilizer, stabilize_video
from .optical_flow import MotionEstimator, LucasKanadeFlow, FarnebackFlow
from .trajectory_smoother import (
    MovingAverageSmoother,
    GaussianSmoother,
    KalmanSmoother,
    L1TrajectorySmoother
)

__all__ = [
    'VideoStabilizer',
    'stabilize_video',
    'MotionEstimator',
    'LucasKanadeFlow',
    'FarnebackFlow',
    'MovingAverageSmoother',
    'GaussianSmoother',
    'KalmanSmoother',
    'L1TrajectorySmoother',
]

__version__ = '1.0.0'
