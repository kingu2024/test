"""
视频防抖演示脚本
Video Stabilization Demo Script

使用方法:
    # 使用合成抖动视频
    python demo_stabilization.py --test

    # 使用真实视频
    python demo_stabilization.py --input shaky_video.mp4

    # 对比不同平滑方法
    python demo_stabilization.py --test --compare

    # 指定平滑方法和裁剪比例
    python demo_stabilization.py --test --smooth kalman --crop 0.1
"""

import cv2
import numpy as np
import argparse
import os
import sys
import logging
import time
from typing import List

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def generate_shaky_video(
    n_frames: int = 120,
    width: int = 640,
    height: int = 480,
    shake_amplitude: float = 20.0,
    shake_freq_range: tuple = (1.0, 8.0)
) -> List[np.ndarray]:
    """
    生成合成抖动视频（用于防抖算法测试）

    【抖动模型】
    真实手持相机抖动可以建模为多个正弦分量的叠加:
    x_shake(t) = Σᵢ Aᵢ · sin(2π·fᵢ·t + φᵢ)

    典型手抖频率:
    - 低频 (1-3 Hz): 手臂整体晃动
    - 中频 (3-6 Hz): 腕部颤抖
    - 高频 (6-12 Hz): 肌肉微颤

    Args:
        n_frames: 视频帧数
        width, height: 视频分辨率
        shake_amplitude: 最大抖动幅度（像素）
        shake_freq_range: 抖动频率范围 (min_Hz, max_Hz)

    Returns:
        frames: 带抖动的视频帧列表
    """
    logger.info(f"生成合成抖动视频: {width}×{height}, {n_frames}帧, 抖动幅度={shake_amplitude}px")

    # ── 创建场景内容 ──────────────────────────────────────────
    # 用一个大场景裁剪，模拟相机视图
    scene_w = width + int(shake_amplitude * 4)
    scene_h = height + int(shake_amplitude * 4)
    margin = int(shake_amplitude * 2)

    np.random.seed(123)
    scene = np.ones((scene_h, scene_w, 3), dtype=np.uint8) * 180

    # 添加网格线（便于观察抖动效果）
    for x in range(0, scene_w, 50):
        cv2.line(scene, (x, 0), (x, scene_h), (150, 150, 150), 1)
    for y in range(0, scene_h, 50):
        cv2.line(scene, (0, y), (scene_w, y), (150, 150, 150), 1)

    # 添加显著特征（用于光流追踪）
    for _ in range(80):
        cx = np.random.randint(30, scene_w - 30)
        cy = np.random.randint(30, scene_h - 30)
        size = np.random.randint(10, 50)
        color = tuple(np.random.randint(30, 200, 3).tolist())
        shape = np.random.randint(3)
        if shape == 0:
            cv2.rectangle(scene, (cx, cy), (cx+size, cy+size), color, -1)
        elif shape == 1:
            cv2.circle(scene, (cx+size//2, cy+size//2), size//2, color, -1)
        else:
            pts = np.array([[cx, cy+size], [cx+size//2, cy], [cx+size, cy+size]])
            cv2.fillPoly(scene, [pts], color)

    # 添加文本参考点
    for i in range(5):
        x = i * (scene_w // 5) + 20
        for j in range(3):
            y = j * (scene_h // 3) + 40
            cv2.putText(scene, f'({i},{j})', (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # ── 生成抖动轨迹 ─────────────────────────────────────────
    # 使用多频率正弦叠加模拟真实手抖
    fps = 30.0
    t = np.linspace(0, n_frames / fps, n_frames)

    # X方向抖动: 多频率叠加
    n_components = 4  # 频率分量数
    shake_x = np.zeros(n_frames)
    shake_y = np.zeros(n_frames)

    for _ in range(n_components):
        freq = np.random.uniform(*shake_freq_range)    # 随机频率
        phase = np.random.uniform(0, 2 * np.pi)       # 随机初相
        amp = shake_amplitude / n_components           # 各分量幅度
        shake_x += amp * np.sin(2 * np.pi * freq * t + phase)
        shake_y += amp * np.sin(2 * np.pi * freq * t + phase + np.pi/3)

    # 添加随机漫步成分（低频漂移）
    drift_x = np.cumsum(np.random.randn(n_frames) * 0.5)
    drift_y = np.cumsum(np.random.randn(n_frames) * 0.5)

    # 限制漂移幅度
    drift_max = shake_amplitude * 0.5
    drift_x = np.clip(drift_x, -drift_max, drift_max)
    drift_y = np.clip(drift_y, -drift_max, drift_max)

    # 添加随机旋转抖动（弧度）
    shake_rot = np.random.randn(n_frames) * 0.005  # 约0.3°标准差

    total_x = shake_x + drift_x
    total_y = shake_y + drift_y

    logger.info(
        f"抖动统计: dx_std={total_x.std():.1f}px, dy_std={total_y.std():.1f}px, "
        f"rot_std={np.degrees(shake_rot.std()):.2f}°"
    )

    # ── 生成视频帧 ────────────────────────────────────────────
    frames = []

    for k in range(n_frames):
        # 当前帧相机偏移量
        ox = int(total_x[k]) + margin
        oy = int(total_y[k]) + margin
        angle = total_x[k] * 0.02  # 微小旋转（与平移相关）

        # 确保裁剪范围有效
        ox = max(0, min(ox, scene_w - width))
        oy = max(0, min(oy, scene_h - height))

        # 裁剪当前视图（模拟相机平移）
        frame = scene[oy:oy+height, ox:ox+width].copy()

        # 模拟旋转抖动（绕图像中心）
        if abs(shake_rot[k]) > 0.001:
            M_rot = cv2.getRotationMatrix2D(
                (width / 2, height / 2),
                np.degrees(shake_rot[k]),
                1.0
            )
            frame = cv2.warpAffine(
                frame, M_rot, (width, height),
                borderMode=cv2.BORDER_REPLICATE
            )

        # 添加轻微模糊（模拟运动模糊）
        if abs(total_x[k]) + abs(total_y[k]) > shake_amplitude * 0.5:
            frame = cv2.GaussianBlur(frame, (3, 3), 0.5)

        # 帧序号水印
        cv2.putText(frame, f'Frame {k+1}/{n_frames}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        frames.append(frame)

    return frames


def compare_smoothing_methods(
    frames: List[np.ndarray],
    output_dir: str = 'output_stabilization'
) -> dict:
    """
    对比不同轨迹平滑方法的防抖效果

    测试的方法组合:
    1. Moving Average (窗口=30帧)
    2. Gaussian Smoothing (σ=10)
    3. Kalman Filter (双向RTS平滑)
    4. L1 Optimization

    Args:
        frames: 原始抖动帧
        output_dir: 输出目录

    Returns:
        各方法的评估指标字典
    """
    from video_stabilization import VideoStabilizer

    os.makedirs(output_dir, exist_ok=True)

    methods = {
        'moving_avg': {'smooth_method': 'moving_avg', 'smooth_radius': 30},
        'gaussian':   {'smooth_method': 'gaussian',   'smooth_radius': 30},
        'kalman':     {'smooth_method': 'kalman',
                      'kalman_process_noise': 1e-3, 'kalman_obs_noise': 1.0},
        'l1':         {'smooth_method': 'l1'},
    }

    all_metrics = {}

    logger.info("\n=== 对比不同轨迹平滑方法 ===")

    for method_name, params in methods.items():
        logger.info(f"\n--- 方法: {method_name} ---")

        t0 = time.time()
        stabilizer = VideoStabilizer(
            flow_method='lk',
            crop_ratio=0.08,
            **params
        )

        stabilized, metrics = stabilizer.stabilize(frames[:])
        elapsed = time.time() - t0

        metrics['elapsed_time'] = elapsed
        all_metrics[method_name] = metrics

        # 保存稳定结果
        if stabilized:
            h, w = stabilized[0].shape[:2]
            out_path = os.path.join(output_dir, f'stabilized_{method_name}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
            for frame in stabilized:
                out.write(frame)
            out.release()

        logger.info(
            f"  dx改善: {metrics.get('dx_improvement', 0):.1%}, "
            f"dy改善: {metrics.get('dy_improvement', 0):.1%}, "
            f"耗时: {elapsed:.2f}s"
        )

    # 打印对比表格
    logger.info("\n" + "="*65)
    logger.info(f"{'方法':<15} {'dx改善':>10} {'dy改善':>10} {'旋转改善':>12} {'耗时':>8}")
    logger.info("-"*65)
    for method_name, m in all_metrics.items():
        logger.info(
            f"{method_name:<15} "
            f"{m.get('dx_improvement',0):>9.1%} "
            f"{m.get('dy_improvement',0):>9.1%} "
            f"{m.get('rotation_improvement',0):>11.1%} "
            f"{m.get('elapsed_time',0):>7.2f}s"
        )
    logger.info("="*65)

    return all_metrics


def analyze_trajectory(
    frames: List[np.ndarray],
    output_dir: str = 'output_stabilization'
) -> None:
    """
    分析并可视化运动轨迹（原始 vs 平滑）

    生成轨迹对比图（如果有 matplotlib）

    Args:
        frames: 视频帧
        output_dir: 输出目录
    """
    from video_stabilization import VideoStabilizer

    stabilizer = VideoStabilizer(smooth_method='kalman')

    # 估计轨迹（不实际稳定）
    trajectory = stabilizer.estimate_trajectory(frames)
    smoothed = stabilizer.smooth_trajectory(trajectory)

    # 保存轨迹数据（用于分析）
    np.save(os.path.join(output_dir, 'raw_trajectory.npy'), trajectory)
    np.save(os.path.join(output_dir, 'smooth_trajectory.npy'), smoothed)

    logger.info(f"轨迹数据已保存到 {output_dir}/")

    # 尝试用matplotlib绘图
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('视频运动轨迹分析 (原始 vs 卡尔曼平滑)', fontsize=14)

        labels = ['X方向平移 (px)', 'Y方向平移 (px)', '旋转角度 (°)', '缩放因子']
        dims = [0, 1, 2, 3]
        scales = [1, 1, 180/np.pi, 1]  # 旋转转为度数

        for ax, label, d, scale in zip(axes.flat, labels, dims, scales):
            ax.plot(trajectory[:, d] * scale, 'r-', alpha=0.7, linewidth=1, label='原始轨迹')
            ax.plot(smoothed[:, d] * scale, 'b-', linewidth=2, label='平滑轨迹')
            ax.set_title(label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('帧序号')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'trajectory_analysis.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"轨迹分析图已保存: {plot_path}")

    except ImportError:
        logger.info("未安装 matplotlib，跳过轨迹可视化（可 pip install matplotlib）")


def run_stabilization_demo(
    frames: List[np.ndarray],
    output_dir: str = 'output_stabilization',
    smooth: str = 'kalman',
    crop: float = 0.1
):
    """
    运行单个视频防抖演示

    Args:
        frames: 输入帧列表
        output_dir: 输出目录
        smooth: 平滑方法
        crop: 裁剪比例
    """
    os.makedirs(output_dir, exist_ok=True)

    from video_stabilization import VideoStabilizer

    logger.info(f"\n{'='*60}")
    logger.info(f"视频防抖演示: 平滑方法={smooth}, 裁剪比例={crop}")
    logger.info(f"{'='*60}")

    stabilizer = VideoStabilizer(
        flow_method='lk',
        smooth_method=smooth,
        crop_ratio=crop
    )

    t0 = time.time()
    stabilized, metrics = stabilizer.stabilize(frames)
    elapsed = time.time() - t0

    logger.info(f"\n防抖完成! 耗时: {elapsed:.2f}s")
    logger.info(f"防抖效果:")
    logger.info(f"  水平稳定性提升: {metrics.get('dx_improvement', 0):.1%}")
    logger.info(f"  垂直稳定性提升: {metrics.get('dy_improvement', 0):.1%}")
    logger.info(f"  旋转稳定性提升: {metrics.get('rotation_improvement', 0):.1%}")

    # 保存原始视频
    h_orig, w_orig = frames[0].shape[:2]
    orig_path = os.path.join(output_dir, 'original.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_orig = cv2.VideoWriter(orig_path, fourcc, 30.0, (w_orig, h_orig))
    for frame in frames:
        out_orig.write(frame)
    out_orig.release()

    # 保存稳定视频
    if stabilized:
        h_stab, w_stab = stabilized[0].shape[:2]
        stab_path = os.path.join(output_dir, f'stabilized_{smooth}.mp4')
        out_stab = cv2.VideoWriter(stab_path, fourcc, 30.0, (w_stab, h_stab))
        for frame in stabilized:
            out_stab.write(frame)
        out_stab.release()
        logger.info(f"结果保存: {stab_path}")

        # 创建并排对比帧（原始 vs 稳定）
        comparison_frames = []
        for i in range(min(len(frames), len(stabilized))):
            orig_resized = cv2.resize(frames[i], (w_stab, h_stab))
            stab_frame = stabilized[i]

            # 添加标签
            cv2.putText(orig_resized, 'ORIGINAL (Shaky)',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(stab_frame, f'STABILIZED ({smooth.upper()})',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            combined = np.hstack([orig_resized, stab_frame])
            comparison_frames.append(combined)

        comp_path = os.path.join(output_dir, f'comparison_{smooth}.mp4')
        out_comp = cv2.VideoWriter(
            comp_path, fourcc, 30.0,
            (w_stab * 2, h_stab)
        )
        for frame in comparison_frames:
            out_comp.write(frame)
        out_comp.release()
        logger.info(f"对比视频: {comp_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='视频防抖演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo_stabilization.py --test
  python demo_stabilization.py --test --smooth gaussian --crop 0.08
  python demo_stabilization.py --test --compare
  python demo_stabilization.py --input shaky.mp4 --smooth kalman
        """
    )
    parser.add_argument('--test', action='store_true', help='使用合成抖动视频')
    parser.add_argument('--input', help='输入视频路径')
    parser.add_argument('--smooth', default='kalman',
                       choices=['moving_avg', 'gaussian', 'kalman', 'l1'],
                       help='轨迹平滑方法')
    parser.add_argument('--flow', default='lk',
                       choices=['lk', 'farneback'], help='光流估计方法')
    parser.add_argument('--crop', type=float, default=0.1, help='裁剪比例 (0-0.3)')
    parser.add_argument('--n_frames', type=int, default=120, help='测试视频帧数')
    parser.add_argument('--compare', action='store_true', help='对比所有平滑方法')
    parser.add_argument('--analyze', action='store_true', help='分析并可视化运动轨迹')
    parser.add_argument('--output_dir', default='output_stabilization', help='输出目录')

    args = parser.parse_args()

    # 准备输入帧
    if args.test:
        frames = generate_shaky_video(
            n_frames=args.n_frames,
            shake_amplitude=15.0
        )
    elif args.input:
        cap = cv2.VideoCapture(args.input)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        logger.info(f"读取视频: {args.input}, 共 {len(frames)} 帧")
    else:
        logger.info("未指定输入，使用合成测试视频")
        frames = generate_shaky_video(n_frames=args.n_frames)

    if not frames:
        logger.error("无有效帧，退出")
        sys.exit(1)

    # 轨迹分析
    if args.analyze:
        analyze_trajectory(frames, args.output_dir)

    # 对比所有方法
    if args.compare:
        compare_smoothing_methods(frames, args.output_dir)
    else:
        # 运行单一方法演示
        run_stabilization_demo(
            frames,
            output_dir=args.output_dir,
            smooth=args.smooth,
            crop=args.crop
        )

    logger.info(f"\n演示完成! 结果保存在 '{args.output_dir}' 目录")


if __name__ == '__main__':
    main()
