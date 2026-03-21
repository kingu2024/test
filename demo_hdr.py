"""
HDR 高动态范围成像演示脚本
HDR Imaging Demo Script

使用方法:
    # 使用合成测试数据（无需真实图像文件）
    python demo_hdr.py --test

    # 使用真实多曝光图像
    python demo_hdr.py --images img1.jpg img2.jpg img3.jpg --exposures 0.033 0.25 1.0

    # 单张图像 HDR 增强
    python demo_hdr.py --single input.jpg

    # 色调映射算法对比
    python demo_hdr.py --compare
"""

import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import argparse
import os
import sys
import logging
import time
import matplotlib.pyplot as plt
from typing import List

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_hdr_scene(width: int = 256, height: int = 256) -> np.ndarray:
    """
    生成合成 HDR 场景（float32，4-5 个数量级动态范围）

    【场景组成】
    - 亮圆形区域（模拟太阳/灯光），值范围 5000–10000
    - 中等亮度区域（模拟室内/阴影区），值范围 100–500
    - 暗区域（模拟阴暗角落），值范围 1–10
    - 渐变过渡以平滑区域边界

    Args:
        width:  图像宽度（像素）
        height: 图像高度（像素）

    Returns:
        hdr_scene: float32 HDR 场景，形状 (height, width, 3)，值域约 1–10000
    """
    np.random.seed(42)

    # 初始化背景为暗区域（值 5–15）
    scene = np.ones((height, width, 3), dtype=np.float32) * 8.0

    # ── 添加渐变中等亮度背景 ─────────────────────────────────
    # 水平渐变：左侧较暗，右侧较亮
    x_grad = np.linspace(20.0, 300.0, width, dtype=np.float32)
    y_grad = np.linspace(10.0, 150.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_grad, y_grad)
    scene[:, :, 0] += xx * 0.3 + yy * 0.2
    scene[:, :, 1] += xx * 0.35 + yy * 0.25
    scene[:, :, 2] += xx * 0.25 + yy * 0.15

    # ── 中等亮度矩形区域（模拟室内物体）────────────────────────
    mid_regions = [
        (int(0.1 * width), int(0.1 * height), int(0.4 * width), int(0.5 * height), 200.0, 350.0, 280.0),
        (int(0.5 * width), int(0.5 * height), int(0.9 * width), int(0.9 * height), 300.0, 200.0, 150.0),
        (int(0.2 * width), int(0.6 * height), int(0.5 * width), int(0.9 * height), 150.0, 250.0, 400.0),
    ]
    for x1, y1, x2, y2, b, g, r in mid_regions:
        scene[y1:y2, x1:x2, 0] = b
        scene[y1:y2, x1:x2, 1] = g
        scene[y1:y2, x1:x2, 2] = r

    # ── 亮圆形光源（太阳/灯泡），值 5000–10000 ──────────────────
    light_sources = [
        (int(0.75 * width), int(0.2 * height), int(0.06 * min(width, height)), 9000.0, 8500.0, 7000.0),
        (int(0.3 * width), int(0.75 * height), int(0.05 * min(width, height)), 7000.0, 7500.0, 9000.0),
        (int(0.85 * width), int(0.8 * height), int(0.04 * min(width, height)), 6000.0, 8000.0, 6500.0),
    ]
    yy_grid, xx_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    for cx, cy, radius, b, g, r in light_sources:
        dist = np.sqrt((xx_grid - cx) ** 2 + (yy_grid - cy) ** 2)
        # 核心高亮区
        mask_core = dist < radius
        scene[mask_core, 0] = b
        scene[mask_core, 1] = g
        scene[mask_core, 2] = r
        # 光晕渐变区（从光源值渐减到周围场景值）
        mask_halo = (dist >= radius) & (dist < radius * 3)
        alpha = 1.0 - (dist[mask_halo] - radius) / (radius * 2)
        surrounding_b = scene[mask_halo, 0].copy()
        surrounding_g = scene[mask_halo, 1].copy()
        surrounding_r = scene[mask_halo, 2].copy()
        scene[mask_halo, 0] = alpha * b + (1 - alpha) * surrounding_b
        scene[mask_halo, 1] = alpha * g + (1 - alpha) * surrounding_g
        scene[mask_halo, 2] = alpha * r + (1 - alpha) * surrounding_r

    # ── 极暗角落区域（值 1–5）────────────────────────────────
    dark_corners = [
        (0, 0, int(0.15 * width), int(0.15 * height)),
        (int(0.85 * width), 0, width, int(0.15 * height)),
    ]
    for x1, y1, x2, y2 in dark_corners:
        scene[y1:y2, x1:x2] = np.random.uniform(1.0, 5.0, (y2 - y1, x2 - x1, 3)).astype(np.float32)

    # 确保值域下限 > 0（避免对数运算问题）
    scene = np.clip(scene, 0.5, None)

    logger.info(
        f"合成 HDR 场景生成完毕: shape={scene.shape}, "
        f"值域 [{scene.min():.1f}, {scene.max():.1f}], "
        f"动态范围 ~{scene.max()/scene.min():.0f}:1"
    )
    return scene


def simulate_exposures(
    hdr_scene: np.ndarray,
    exposure_times: List[float],
    noise_sigma: float = 3.0,
    add_shift: bool = False
) -> List[np.ndarray]:
    """
    从 HDR 场景模拟多曝光 LDR 图像

    【模型】
        LDR = clip(HDR × Δt, 0, 255).astype(uint8)
    再叠加高斯噪声，模拟真实相机成像过程。

    Args:
        hdr_scene:      float32 HDR 辐照图，形状 (H, W, 3)
        exposure_times: 各张 LDR 图像的曝光时间列表（秒）
        noise_sigma:    高斯噪声标准差（默认 3.0）
        add_shift:      是否添加随机平移（1–5 像素），模拟手持抖动

    Returns:
        images: uint8 BGR LDR 图像列表，长度 = len(exposure_times)
    """
    images = []
    for i, dt in enumerate(exposure_times):
        # 线性曝光：辐照度 × 曝光时间
        ldr_float = hdr_scene * dt
        ldr_float = np.clip(ldr_float, 0, 255)

        # 添加高斯噪声
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, ldr_float.shape)
            ldr_float = np.clip(ldr_float + noise, 0, 255)

        ldr = ldr_float.astype(np.uint8)

        # 可选：随机平移（模拟手持抖动）
        if add_shift:
            shift_x = np.random.randint(1, 6)
            shift_y = np.random.randint(1, 6)
            ldr = np.roll(ldr, shift_x, axis=1)
            ldr = np.roll(ldr, shift_y, axis=0)

        images.append(ldr)
        logger.info(
            f"  曝光 {i+1}/{len(exposure_times)}: Δt={dt:.4f}s, "
            f"像素值范围 [{ldr.min()}, {ldr.max()}]"
        )

    return images


def run_test_mode(output_dir: str):
    """
    测试模式：使用合成数据演示完整 HDR 流程

    步骤：
    1. 生成合成 HDR 场景 + 4 张曝光图像
    2. 保存曝光输入对比图
    3. 运行 Debevec 流水线，保存色调映射结果
    4. 运行 Robertson，保存响应曲线对比（Debevec vs Robertson）
    5. 保存 HDR 辐射图伪彩色图
    6. 运行 Mertens 曝光融合，保存结果
    7. 运行单张图像 HDR 增强，保存前后对比图
    8. 打印输出文件汇总

    Args:
        output_dir: 结果输出目录
    """
    from hdr_imaging import (HDRPipeline, DebevecCalibration, RobertsonCalibration,
                              HDRMerge, MertensFusion, SingleImageHDR)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info("HDR 测试模式：使用合成数据")
    logger.info(f"{'='*60}")

    # ── 1. 生成合成场景及曝光图像 ────────────────────────────
    logger.info("\n[1/7] 生成合成 HDR 场景...")
    hdr_scene = generate_synthetic_hdr_scene(256, 256)

    exposure_times = [1/30.0, 1/4.0, 1.0, 4.0]
    logger.info(f"[1/7] 模拟 {len(exposure_times)} 张曝光图像: {exposure_times}")
    exposures = simulate_exposures(hdr_scene, exposure_times, noise_sigma=3.0)

    # ── 2. 保存曝光输入对比图 ────────────────────────────────
    logger.info("\n[2/7] 保存曝光输入对比图...")
    fig, axes = plt.subplots(1, len(exposures), figsize=(16, 4))
    fig.suptitle('Multi-Exposure Input Images', fontsize=14, fontweight='bold')
    for i, (img, dt) in enumerate(zip(exposures, exposure_times)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(f'Δt = {dt:.4f}s')
        axes[i].axis('off')
    plt.tight_layout()
    exposures_path = os.path.join(output_dir, 'exposures_input.png')
    plt.savefig(exposures_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  保存: {exposures_path}")

    # ── 3. 运行 Debevec 流水线，保存色调映射结果 ─────────────
    logger.info("\n[3/7] 运行 Debevec HDR 流水线 (Reinhard 全局色调映射)...")
    t0 = time.time()
    pipeline_debevec = HDRPipeline(
        align_method='mtb',
        calibration_method='debevec',
        tone_mapping_method='reinhard_global'
    )
    result_debevec = pipeline_debevec.process(exposures, exposure_times)
    elapsed = time.time() - t0
    logger.info(f"  Debevec 流水线完成，耗时 {elapsed:.2f}s")

    tonemapped_path = os.path.join(output_dir, 'tonemapped_debevec_reinhard.png')
    cv2.imwrite(tonemapped_path, result_debevec.ldr_result)
    logger.info(f"  色调映射结果: {tonemapped_path}")

    # ── 4. 运行 Robertson，保存响应曲线对比 ──────────────────
    logger.info("\n[4/7] 运行 Robertson 标定，对比响应曲线...")
    from hdr_imaging.alignment import MTBAlignment

    aligner = MTBAlignment()
    aligned = aligner.process(exposures)

    debevec_cal = DebevecCalibration(samples=50, lambda_smooth=10.0)
    robertson_cal = RobertsonCalibration()

    debevec_curve = debevec_cal.process(aligned, exposure_times)
    robertson_curve = robertson_cal.process(aligned, exposure_times)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Camera Response Function: Debevec vs Robertson', fontsize=13, fontweight='bold')
    channel_names = ['Blue', 'Green', 'Red']
    channel_colors = ['blue', 'green', 'red']
    z_vals = np.arange(256)

    for c in range(3):
        ax = axes[c]
        ax.plot(z_vals, debevec_curve[:, c], color=channel_colors[c],
                linewidth=2, label='Debevec', linestyle='-')
        ax.plot(z_vals, robertson_curve[:, c], color=channel_colors[c],
                linewidth=2, label='Robertson', linestyle='--', alpha=0.8)
        ax.set_title(f'{channel_names[c]} Channel')
        ax.set_xlabel('Pixel Value Z')
        ax.set_ylabel('g(Z) = ln(E)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(output_dir, 'response_curve_comparison.png')
    plt.savefig(curve_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  响应曲线对比: {curve_path}")

    # ── 5. 保存 HDR 辐射图伪彩色图 ───────────────────────────
    logger.info("\n[5/7] 保存 HDR 辐射图伪彩色图...")
    hdr_map = result_debevec.hdr_radiance_map
    # 取亮度通道用于可视化
    hdr_lum = hdr_map[:, :, 0] * 0.0722 + hdr_map[:, :, 1] * 0.7152 + hdr_map[:, :, 2] * 0.2126
    hdr_log = np.log1p(hdr_lum)  # 对数压缩显示

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(hdr_log, cmap='jet')
    ax.set_title('HDR Radiance Map (log scale, jet colormap)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='log(1 + Luminance)')
    plt.tight_layout()
    radiance_path = os.path.join(output_dir, 'hdr_radiance_map.png')
    plt.savefig(radiance_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  辐射图伪彩色: {radiance_path}")

    # ── 6. 运行 Mertens 曝光融合 ─────────────────────────────
    logger.info("\n[6/7] 运行 Mertens 曝光融合...")
    t0 = time.time()
    pipeline_mertens = HDRPipeline(align_method='mtb', calibration_method='debevec',
                                    tone_mapping_method='reinhard_global')
    fusion_result = pipeline_mertens.exposure_fusion(exposures)
    elapsed = time.time() - t0
    logger.info(f"  Mertens 融合完成，耗时 {elapsed:.2f}s")

    fusion_path = os.path.join(output_dir, 'mertens_fusion.png')
    cv2.imwrite(fusion_path, fusion_result)
    logger.info(f"  Mertens 融合结果: {fusion_path}")

    # ── 7. 单张图像 HDR 增强（使用最暗曝光图像）────────────────
    logger.info("\n[7/7] 运行单张图像 HDR 增强（最暗曝光帧）...")
    darkest_img = exposures[0]  # 最短曝光 = 最暗图像

    t0 = time.time()
    pipeline_single = HDRPipeline(align_method='mtb', calibration_method='debevec',
                                   tone_mapping_method='reinhard_global')
    enhanced = pipeline_single.single_image_hdr(darkest_img)
    elapsed = time.time() - t0
    logger.info(f"  单张 HDR 增强完成，耗时 {elapsed:.2f}s")

    # 保存前后对比图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Single Image HDR Enhancement', fontsize=13, fontweight='bold')
    axes[0].imshow(cv2.cvtColor(darkest_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original (Δt={exposure_times[0]:.4f}s)')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Enhanced (Single-Image HDR)')
    axes[1].axis('off')
    plt.tight_layout()
    single_path = os.path.join(output_dir, 'single_image_hdr.png')
    plt.savefig(single_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  单张 HDR 前后对比: {single_path}")

    # ── 打印汇总 ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("测试模式完成！输出文件汇总:")
    saved_files = [
        exposures_path,
        tonemapped_path,
        curve_path,
        radiance_path,
        fusion_path,
        single_path,
    ]
    for f in saved_files:
        logger.info(f"  {f}")
    logger.info(f"{'='*60}")


def run_compare_mode(output_dir: str):
    """
    色调映射算法对比模式

    对同一 HDR 场景应用全部 10 种色调映射算法，
    生成 2×5 对比网格图。

    Args:
        output_dir: 结果输出目录
    """
    from hdr_imaging.hdr_pipeline import TONE_MAPPING_METHODS
    from hdr_imaging.alignment import MTBAlignment
    from hdr_imaging.calibration import DebevecCalibration
    from hdr_imaging.merge import HDRMerge

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info("色调映射算法对比模式")
    logger.info(f"{'='*60}")

    # ── 生成场景和曝光图像 ───────────────────────────────────
    logger.info("\n生成合成 HDR 场景及曝光图像...")
    hdr_scene = generate_synthetic_hdr_scene(256, 256)
    exposure_times = [1/30.0, 1/4.0, 1.0, 4.0]
    exposures = simulate_exposures(hdr_scene, exposure_times)

    # ── 对齐 + 标定 + 合并为 HDR 辐射图 ─────────────────────
    logger.info("\n对齐图像...")
    aligner = MTBAlignment()
    aligned = aligner.process(exposures)

    logger.info("Debevec CRF 标定...")
    calibrator = DebevecCalibration(samples=50, lambda_smooth=10.0)
    response_curve = calibrator.process(aligned, exposure_times)

    logger.info("合并为 HDR 辐射图...")
    merger = HDRMerge()
    hdr_map = merger.process(aligned, exposure_times, response_curve)
    logger.info(f"  HDR 辐射图: shape={hdr_map.shape}, 值域 [{hdr_map.min():.3f}, {hdr_map.max():.3f}]")

    # ── 对每种色调映射算法生成结果 ───────────────────────────
    logger.info("\n应用全部 10 种色调映射算法...")
    algo_names = list(TONE_MAPPING_METHODS.keys())
    results = {}

    for name in algo_names:
        t0 = time.time()
        try:
            tone_mapper = TONE_MAPPING_METHODS[name]()
            ldr = tone_mapper.process(hdr_map)
            elapsed = time.time() - t0
            if ldr.dtype != np.uint8:
                ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
            results[name] = ldr
            logger.info(f"  {name:20s}: OK ({elapsed:.3f}s)")
        except Exception as e:
            logger.warning(f"  {name:20s}: 失败 ({e})")
            results[name] = np.zeros((256, 256, 3), dtype=np.uint8)

    # ── 构建 2×5 对比网格图 ──────────────────────────────────
    logger.info("\n生成 2×5 对比网格图...")
    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9))
    fig.suptitle('Tone Mapping Algorithm Comparison (10 Methods)', fontsize=15, fontweight='bold')

    for idx, name in enumerate(algo_names[:10]):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ldr = results.get(name, np.zeros((256, 256, 3), dtype=np.uint8))
        ldr_rgb = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
        ax.imshow(ldr_rgb)
        ax.set_title(name.replace('_', '\n'), fontsize=9, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    compare_path = os.path.join(output_dir, 'tone_mapping_comparison.png')
    plt.savefig(compare_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  对比网格图: {compare_path}")

    # ── 可选：保存单独的每种算法结果 ─────────────────────────
    logger.info("\n保存各算法单独结果...")
    for name, ldr in results.items():
        path = os.path.join(output_dir, f'tonemapped_{name}.png')
        cv2.imwrite(path, ldr)
    logger.info(f"  {len(results)} 张算法结果已保存到 {output_dir}/")

    logger.info(f"\n{'='*60}")
    logger.info(f"对比模式完成！主要输出: {compare_path}")
    logger.info(f"{'='*60}")


def run_custom_mode(image_paths: List[str], exposure_times: List[float], output_dir: str):
    """
    自定义图像模式：加载用户提供的多曝光图像并运行完整 HDR 流水线

    Args:
        image_paths:    输入图像文件路径列表
        exposure_times: 各图像对应曝光时间（秒）
        output_dir:     结果输出目录
    """
    from hdr_imaging import HDRPipeline

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info("自定义图像模式")
    logger.info(f"{'='*60}")

    # ── 加载图像 ─────────────────────────────────────────────
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            logger.error(f"无法读取图像: {path}")
            sys.exit(1)
        images.append(img)
        logger.info(f"  读取: {path} ({img.shape[1]}×{img.shape[0]})")

    # ── 验证输入 ─────────────────────────────────────────────
    if len(images) < 2:
        logger.error("至少需要 2 张图像用于 HDR 处理")
        sys.exit(1)

    if len(images) != len(exposure_times):
        logger.error(f"图像数量 ({len(images)}) 与曝光时间数量 ({len(exposure_times)}) 不匹配")
        sys.exit(1)

    # 检查所有图像尺寸一致
    ref_shape = images[0].shape
    for i, img in enumerate(images[1:], 1):
        if img.shape != ref_shape:
            logger.warning(f"图像 {i+1} 尺寸 {img.shape} 与参考尺寸 {ref_shape} 不一致，将调整")
            images[i] = cv2.resize(img, (ref_shape[1], ref_shape[0]))

    # ── 运行 HDR 流水线 ──────────────────────────────────────
    logger.info("\n运行 HDR 流水线 (Debevec + Reinhard 全局)...")
    t0 = time.time()
    pipeline = HDRPipeline(
        align_method='mtb',
        calibration_method='debevec',
        tone_mapping_method='reinhard_global'
    )
    result = pipeline.process(images, exposure_times)
    elapsed = time.time() - t0
    logger.info(f"  HDR 流水线完成，耗时 {elapsed:.2f}s")

    # ── 保存色调映射结果 ─────────────────────────────────────
    out_path = os.path.join(output_dir, 'hdr_result.png')
    cv2.imwrite(out_path, result.ldr_result)
    logger.info(f"  色调映射结果: {out_path}")

    # ── 保存曝光输入 + 结果对比图 ────────────────────────────
    thumb_w, thumb_h = 320, 240
    thumbs = [cv2.resize(img, (thumb_w, thumb_h)) for img in images]
    thumbs_strip = np.hstack(thumbs)
    result_resized = cv2.resize(result.ldr_result, (thumb_w * len(images), thumb_h))
    comparison = np.vstack([thumbs_strip, result_resized])
    compare_path = os.path.join(output_dir, 'comparison.jpg')
    cv2.imwrite(compare_path, comparison)
    logger.info(f"  对比图: {compare_path}")

    # ── 运行 Mertens 融合（对比）────────────────────────────
    logger.info("\n运行 Mertens 曝光融合...")
    fusion_result = pipeline.exposure_fusion(images)
    fusion_path = os.path.join(output_dir, 'mertens_fusion.png')
    cv2.imwrite(fusion_path, fusion_result)
    logger.info(f"  Mertens 融合结果: {fusion_path}")

    logger.info(f"\n自定义模式完成！结果保存在 {output_dir}/")


def run_single_mode(image_path: str, output_dir: str):
    """
    单张图像 HDR 增强模式

    加载一张 LDR 图像，运行 SingleImageHDR（CLAHE + 多尺度增强），
    保存原图与增强图的并排对比。

    Args:
        image_path: 输入图像文件路径
        output_dir: 结果输出目录
    """
    from hdr_imaging import HDRPipeline

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info("单张图像 HDR 增强模式")
    logger.info(f"{'='*60}")

    # ── 加载图像 ─────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"无法读取图像: {image_path}")
        sys.exit(1)
    logger.info(f"  读取: {image_path} ({img.shape[1]}×{img.shape[0]})")

    # ── 运行单张图像 HDR ──────────────────────────────────────
    t0 = time.time()
    pipeline = HDRPipeline(align_method='mtb', calibration_method='debevec',
                            tone_mapping_method='reinhard_global')
    enhanced = pipeline.single_image_hdr(img)
    elapsed = time.time() - t0
    logger.info(f"  单张 HDR 增强完成，耗时 {elapsed:.2f}s")

    # ── 保存结果图像 ─────────────────────────────────────────
    result_path = os.path.join(output_dir, 'single_hdr_result.png')
    cv2.imwrite(result_path, enhanced)
    logger.info(f"  增强结果: {result_path}")

    # ── 保存前后对比图 ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Single Image HDR Enhancement', fontsize=13, fontweight='bold')

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original LDR Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Enhanced (CLAHE + Multi-scale)')
    axes[1].axis('off')

    plt.tight_layout()
    compare_path = os.path.join(output_dir, 'single_hdr_comparison.png')
    plt.savefig(compare_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  前后对比图: {compare_path}")

    logger.info(f"\n单张图像模式完成！结果保存在 {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='HDR 高动态范围成像演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo_hdr.py --test
  python demo_hdr.py --compare
  python demo_hdr.py --images img1.jpg img2.jpg img3.jpg --exposures 0.033 0.25 1.0
  python demo_hdr.py --single input.jpg
        """
    )
    parser.add_argument('--test', action='store_true',
                        help='使用合成测试数据（无需真实图像）')
    parser.add_argument('--images', nargs='+',
                        help='多曝光输入图像路径列表')
    parser.add_argument('--exposures', nargs='+', type=float,
                        help='各图像对应曝光时间（秒），与 --images 配合使用')
    parser.add_argument('--single', type=str,
                        help='单张图像路径（进行 HDR 增强）')
    parser.add_argument('--compare', action='store_true',
                        help='色调映射算法对比模式（所有 10 种算法）')
    parser.add_argument('--output_dir', default='output_hdr',
                        help='输出目录（默认: output_hdr）')

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.test:
        run_test_mode(output_dir)
    elif args.compare:
        run_compare_mode(output_dir)
    elif args.images:
        if not args.exposures:
            logger.error("使用 --images 时必须同时指定 --exposures 曝光时间")
            sys.exit(1)
        run_custom_mode(args.images, args.exposures, output_dir)
    elif args.single:
        run_single_mode(args.single, output_dir)
    else:
        parser.print_help()
        logger.info("\n未指定模式。请使用 --test 进行快速测试，或 --compare 进行算法对比。")

    if args.test or args.compare or args.images or args.single:
        logger.info(f"\n演示完成！结果保存在 '{output_dir}' 目录")


if __name__ == '__main__':
    main()
