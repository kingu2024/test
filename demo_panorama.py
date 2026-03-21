"""
全景图像拼接演示脚本
Panorama Stitching Demo Script

使用方法:
    # 使用合成测试图像（无需真实图像文件）
    python demo_panorama.py --test

    # 使用真实图像
    python demo_panorama.py --images img1.jpg img2.jpg img3.jpg

    # 指定投影和融合方法
    python demo_panorama.py --test --projection cylindrical --blend multiband
"""

import cv2
import numpy as np
import argparse
import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def generate_test_images(n_images: int = 3) -> list:
    """
    生成合成测试图像集（模拟相机水平旋转拍摄的场景）

    【测试图像生成原理】
    创建一个宽幅"世界"场景，然后用透视变换裁剪出模拟相机
    从不同角度拍摄的图像（相邻图像有~40%重叠）

    Args:
        n_images: 要生成的图像数量

    Returns:
        图像列表（BGR格式）
    """
    logger.info(f"生成 {n_images} 张合成测试图像（带透视畸变）")

    # 创建宽幅场景背景
    world_w, world_h = 1600, 600

    # 用随机几何图形填充场景（模拟建筑/风景纹理）
    np.random.seed(42)
    world = np.ones((world_h, world_w, 3), dtype=np.uint8) * 200  # 浅灰背景

    # 添加随机矩形（模拟建筑窗户）
    for _ in range(50):
        x1 = np.random.randint(0, world_w - 60)
        y1 = np.random.randint(0, world_h - 60)
        x2 = x1 + np.random.randint(30, 80)
        y2 = y1 + np.random.randint(30, 80)
        color = tuple(np.random.randint(50, 220, 3).tolist())
        cv2.rectangle(world, (x1, y1), (x2, y2), color, -1)

    # 添加随机圆形（模拟树木/灯柱）
    for _ in range(30):
        cx = np.random.randint(20, world_w - 20)
        cy = np.random.randint(20, world_h - 20)
        r = np.random.randint(10, 40)
        color = tuple(np.random.randint(20, 180, 3).tolist())
        cv2.circle(world, (cx, cy), r, color, -1)

    # 添加文字标记（测试特征匹配）
    for i in range(8):
        x = i * 200
        cv2.putText(world, f'P{i}', (x + 10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # 水平线和垂直线（测试几何特征保持）
    for y in range(0, world_h, 100):
        cv2.line(world, (0, y), (world_w, y), (100, 100, 100), 1)

    # ── 从宽幅场景中截取各角度视图 ────────────────────────────
    # 目标图像尺寸
    img_w, img_h = 640, 480

    # 计算每张图的水平起始位置（重叠40%）
    # 有效移动范围 = world_w - img_w
    effective_width = world_w - img_w
    overlap_ratio = 0.4
    step = int(img_w * (1 - overlap_ratio))  # 步长 = 图像宽*(1-重叠率)

    images = []
    for i in range(n_images):
        x_start = min(i * step, world_w - img_w)

        # 裁剪基础图像
        crop = world[:, x_start:x_start + img_w].copy()
        crop = cv2.resize(crop, (img_w, img_h))

        # 添加轻微透视畸变（模拟真实相机倾斜）
        # 透视变换矩阵（模拟相机轻微偏转）
        angle_offset = (i - n_images // 2) * 0.02  # 微小角度差
        tilt = np.float32([
            [1, np.tan(angle_offset) * 0.1, 0],
            [0, 1, 0]
        ])

        # 添加随机细微噪声（模拟手持相机）
        noise = np.random.normal(0, 3, crop.shape).astype(np.int16)
        crop_noisy = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 轻微亮度变化（模拟曝光差异）
        brightness_factor = 0.9 + 0.2 * i / max(n_images - 1, 1)  # 0.9~1.1
        crop_bright = np.clip(crop_noisy * brightness_factor, 0, 255).astype(np.uint8)

        images.append(crop_bright)
        logger.info(f"  图像 {i+1}/{n_images}: 场景位置 x=[{x_start}, {x_start+img_w}], "
                   f"亮度系数={brightness_factor:.2f}")

    return images


def visualize_stitching_process(images: list, output_dir: str = 'output_panorama'):
    """
    可视化拼接中间步骤（特征点、匹配、结果）

    Args:
        images: 输入图像列表
        output_dir: 可视化结果输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 延迟导入避免循环依赖
    from panorama_stitching import FeatureExtractor, FeatureMatcher

    extractor = FeatureExtractor('SIFT', max_features=500)
    matcher = FeatureMatcher('SIFT')

    logger.info("\n=== 步骤1: 特征提取可视化 ===")
    all_kps = []
    all_descs = []

    for i, img in enumerate(images):
        kp, desc = extractor.detect_and_compute(img)
        all_kps.append(kp)
        all_descs.append(desc)

        # 绘制关键点（圆圈大小=尺度，线段方向=主方向）
        vis_kp = extractor.visualize_keypoints(img, kp)
        kp_path = os.path.join(output_dir, f'keypoints_{i+1}.jpg')
        cv2.imwrite(kp_path, vis_kp)
        logger.info(f"  图像{i+1}: {len(kp)} 个特征点 → {kp_path}")

    logger.info("\n=== 步骤2: 特征匹配可视化 ===")
    for i in range(len(images) - 1):
        matches = matcher.match(all_descs[i], all_descs[i+1])
        vis_match = matcher.visualize_matches(
            images[i], all_kps[i],
            images[i+1], all_kps[i+1],
            matches, max_display=30
        )
        match_path = os.path.join(output_dir, f'matches_{i+1}_{i+2}.jpg')
        cv2.imwrite(match_path, vis_match)
        logger.info(f"  图像对({i+1},{i+2}): {len(matches)} 个匹配 → {match_path}")


def run_panorama_demo(
    images: list,
    output_dir: str = 'output_panorama',
    feature: str = 'SIFT',
    projection: str = 'cylindrical',
    blend: str = 'multiband'
):
    """
    运行全景拼接演示

    测试不同参数组合并比较结果
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info("开始全景图像拼接演示")
    logger.info(f"参数: 特征={feature}, 投影={projection}, 融合={blend}")
    logger.info(f"{'='*60}")

    from panorama_stitching import PanoramaStitcher

    # 初始化拼接器
    stitcher = PanoramaStitcher(
        feature_method=feature,
        projection=projection,
        blend_method=blend,
        max_features=2000,
        ratio_threshold=0.75,
        ransac_threshold=4.0
    )

    # 执行拼接
    t0 = time.time()
    panorama = stitcher.stitch(images)
    elapsed = time.time() - t0

    if panorama is None:
        logger.error("拼接失败！")
        return

    # 保存结果
    result_path = os.path.join(output_dir, f'panorama_{feature}_{projection}_{blend}.jpg')
    cv2.imwrite(result_path, panorama)

    logger.info(f"\n拼接成功!")
    logger.info(f"  输出尺寸: {panorama.shape[1]}×{panorama.shape[0]} px")
    logger.info(f"  处理时间: {elapsed:.2f}s")
    logger.info(f"  保存路径: {result_path}")

    # 保存对比图（输入+输出）
    input_strip = np.hstack([
        cv2.resize(img, (320, 240)) for img in images
    ])
    comparison = np.vstack([
        input_strip,
        cv2.resize(panorama, (input_strip.shape[1], 240))
    ])
    compare_path = os.path.join(output_dir, 'comparison.jpg')
    cv2.imwrite(compare_path, comparison)
    logger.info(f"  对比图: {compare_path}")

    return panorama


def main():
    parser = argparse.ArgumentParser(
        description='全景图像拼接演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo_panorama.py --test
  python demo_panorama.py --test --projection spherical --blend feather
  python demo_panorama.py --images left.jpg center.jpg right.jpg
        """
    )
    parser.add_argument('--test', action='store_true', help='使用合成测试图像')
    parser.add_argument('--images', nargs='+', help='输入图像路径列表')
    parser.add_argument('--feature', default='SIFT',
                       choices=['SIFT', 'ORB', 'AKAZE'], help='特征检测方法')
    parser.add_argument('--projection', default='cylindrical',
                       choices=['planar', 'cylindrical', 'spherical'], help='投影类型')
    parser.add_argument('--blend', default='multiband',
                       choices=['simple', 'feather', 'multiband'], help='融合方法')
    parser.add_argument('--n_images', type=int, default=3, help='测试图像数量（--test模式）')
    parser.add_argument('--output_dir', default='output_panorama', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='可视化中间步骤')

    args = parser.parse_args()

    # 准备输入图像
    if args.test:
        images = generate_test_images(args.n_images)
        logger.info(f"生成了 {len(images)} 张测试图像")
    elif args.images:
        images = []
        for path in args.images:
            img = cv2.imread(path)
            if img is None:
                logger.error(f"无法读取: {path}")
                sys.exit(1)
            images.append(img)
            logger.info(f"读取图像: {path} ({img.shape[1]}×{img.shape[0]})")
    else:
        logger.info("未指定输入，使用默认合成测试图像（--test）")
        images = generate_test_images(3)

    # 可视化特征提取和匹配（可选）
    if args.visualize:
        visualize_stitching_process(images, args.output_dir)

    # 运行拼接
    run_panorama_demo(
        images,
        output_dir=args.output_dir,
        feature=args.feature,
        projection=args.projection,
        blend=args.blend
    )

    logger.info(f"\n演示完成! 结果保存在 '{args.output_dir}' 目录")


if __name__ == '__main__':
    main()
