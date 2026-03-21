"""
ISP 流程演示脚本
=================

本脚本演示如何使用 ISP Pipeline 处理模拟 RAW 图像，
展示各模块的效果和完整流程。

用法:
    python -m isp_pipeline.demo               # 生成测试图并演示
    python -m isp_pipeline.demo --input xx.npy # 处理指定 RAW 文件
    python -m isp_pipeline.demo --compare       # 对比不同算法效果
"""

import numpy as np
import argparse
import os
from pathlib import Path

from .pipeline import ISPPipeline, ISPConfig


# ─────────────────────────────────────────────
# 测试用合成 RAW 图像生成函数
# ─────────────────────────────────────────────

def generate_synthetic_raw(
    height: int = 512,
    width: int = 512,
    bayer_pattern: str = 'RGGB',
    bit_depth: int = 12,
    noise_level: float = 0.01,
    add_vignetting: bool = True,
    add_bad_pixels: bool = True,
) -> np.ndarray:
    """
    生成合成 Bayer RAW 测试图像

    模拟真实相机 RAW 图像的以下特性：
    1. Bayer 色彩滤波阵列（每像素只有一种颜色）
    2. 传感器暗电流/黑电平偏置
    3. 镜头渐晕（四角变暗）
    4. 高斯噪声（模拟热噪声和读出噪声）
    5. 少量坏点

    参数:
        height, width: 图像尺寸（必须为偶数）
        bayer_pattern: Bayer 排列模式
        bit_depth: 传感器位深（12 位 = 0~4095）
        noise_level: 噪声强度（相对于最大值）
        add_vignetting: 是否添加镜头渐晕
        add_bad_pixels: 是否添加坏点

    返回:
        模拟 RAW Bayer 图像，形状 (H, W)，dtype=uint16
    """
    max_val = (1 << bit_depth) - 1  # 12 位 → 4095
    black_level = 64                  # 模拟黑电平偏置

    # ── 步骤1: 创建合成场景（线性 RGB）──────────────────
    # 生成包含多种颜色区块和渐变的测试场景
    y_coords = np.linspace(0, 1, height)
    x_coords = np.linspace(0, 1, width)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # 创建 RGB 场景（线性光，模拟真实物体反射率）
    scene_r = np.zeros((height, width), dtype=np.float32)
    scene_g = np.zeros((height, width), dtype=np.float32)
    scene_b = np.zeros((height, width), dtype=np.float32)

    # 背景：蓝色渐变天空
    scene_r += 0.1 + 0.2 * yy
    scene_g += 0.2 + 0.3 * yy
    scene_b += 0.5 + 0.4 * (1 - yy)

    # 左上角：红色色块
    mask_red = (xx < 0.25) & (yy < 0.25)
    scene_r[mask_red] = 0.8
    scene_g[mask_red] = 0.1
    scene_b[mask_red] = 0.1

    # 右上角：绿色色块
    mask_green = (xx > 0.75) & (yy < 0.25)
    scene_r[mask_green] = 0.1
    scene_g[mask_green] = 0.8
    scene_b[mask_green] = 0.1

    # 左下角：蓝色色块
    mask_blue = (xx < 0.25) & (yy > 0.75)
    scene_r[mask_blue] = 0.1
    scene_g[mask_blue] = 0.1
    scene_b[mask_blue] = 0.8

    # 右下角：黄色色块
    mask_yellow = (xx > 0.75) & (yy > 0.75)
    scene_r[mask_yellow] = 0.9
    scene_g[mask_yellow] = 0.8
    scene_b[mask_yellow] = 0.05

    # 中央：灰色阶梯（用于测试灰度响应）
    for i, gray_val in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        col_start = int(0.3 * width + i * width * 0.08)
        col_end = col_start + int(width * 0.08)
        row_start = int(0.4 * height)
        row_end = int(0.6 * height)
        scene_r[row_start:row_end, col_start:col_end] = gray_val
        scene_g[row_start:row_end, col_start:col_end] = gray_val
        scene_b[row_start:row_end, col_start:col_end] = gray_val

    # 圆形白色高光（模拟镜面反射）
    cx, cy = width * 0.5, height * 0.35
    dist_to_center = np.sqrt((xx * width - cx)**2 + (yy * height - cy)**2)
    highlight_mask = dist_to_center < width * 0.06
    scene_r[highlight_mask] = 1.0
    scene_g[highlight_mask] = 1.0
    scene_b[highlight_mask] = 1.0

    # 裁剪到 [0, 1]
    scene_r = np.clip(scene_r, 0, 1)
    scene_g = np.clip(scene_g, 0, 1)
    scene_b = np.clip(scene_b, 0, 1)

    # ── 步骤2: 模拟镜头渐晕 ──────────────────────────────
    if add_vignetting:
        # cos⁴(θ) 渐晕模型
        cx_n = (xx - 0.5) * 2  # 归一化坐标 [-1, 1]
        cy_n = (yy - 0.5) * 2
        r = np.sqrt(cx_n**2 + cy_n**2)
        vignette = np.maximum(1 - 0.7 * r**2, 0.2)  # 最暗处保留 20% 亮度
        scene_r *= vignette
        scene_g *= vignette
        scene_b *= vignette

    # ── 步骤3: 模拟色温偏移（模拟钨丝灯照明，偏黄）──────
    # 真实相机在钨丝灯下拍摄时，RAW 图会偏黄橙色
    scene_r *= 1.4   # R 通道增益（暖光偏红）
    scene_g *= 1.1   # G 通道轻微增益
    scene_b *= 0.7   # B 通道衰减（暖光偏少蓝）

    scene_r = np.clip(scene_r, 0, 1)
    scene_g = np.clip(scene_g, 0, 1)
    scene_b = np.clip(scene_b, 0, 1)

    # ── 步骤4: 生成 Bayer 图（CFA 采样）─────────────────
    # 每个像素只保留对应颜色通道的值
    bayer_patterns_map = {
        'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
        'BGGR': {'R': (1, 1), 'Gr': (1, 0), 'Gb': (0, 1), 'B': (0, 0)},
        'GRBG': {'R': (0, 1), 'Gr': (0, 0), 'Gb': (1, 1), 'B': (1, 0)},
        'GBRG': {'R': (1, 0), 'Gr': (1, 1), 'Gb': (0, 0), 'B': (0, 1)},
    }
    pattern = bayer_patterns_map[bayer_pattern]

    bayer = np.zeros((height, width), dtype=np.float32)
    bayer[pattern['R'][0]::2,  pattern['R'][1]::2]  = scene_r[pattern['R'][0]::2,  pattern['R'][1]::2]
    bayer[pattern['Gr'][0]::2, pattern['Gr'][1]::2] = scene_g[pattern['Gr'][0]::2, pattern['Gr'][1]::2]
    bayer[pattern['Gb'][0]::2, pattern['Gb'][1]::2] = scene_g[pattern['Gb'][0]::2, pattern['Gb'][1]::2]
    bayer[pattern['B'][0]::2,  pattern['B'][1]::2]  = scene_b[pattern['B'][0]::2,  pattern['B'][1]::2]

    # ── 步骤5: 添加传感器噪声 ────────────────────────────
    # 散粒噪声（信号相关，高亮处噪声更大）
    shot_noise = np.random.poisson(bayer * 200) / 200.0 - bayer
    # 读出噪声（固定高斯噪声）
    read_noise = np.random.normal(0, noise_level, bayer.shape)

    bayer = bayer + 0.5 * shot_noise + read_noise
    bayer = np.clip(bayer, 0, 1)

    # ── 步骤6: 转换为整型并添加黑电平偏置 ────────────────
    # 模拟相机 ADC 输出：线性值 → 带黑电平偏置的整型
    bayer_int = (bayer * (max_val - black_level) + black_level).astype(np.uint16)

    # ── 步骤7: 添加坏点 ──────────────────────────────────
    if add_bad_pixels:
        # 随机添加 0.1% 的热像素（过亮）
        num_hot = int(height * width * 0.001)
        hot_rows = np.random.randint(0, height, num_hot)
        hot_cols = np.random.randint(0, width, num_hot)
        bayer_int[hot_rows, hot_cols] = max_val  # 饱和值

        # 随机添加 0.05% 的死像素（固定为0）
        num_dead = int(height * width * 0.0005)
        dead_rows = np.random.randint(0, height, num_dead)
        dead_cols = np.random.randint(0, width, num_dead)
        bayer_int[dead_rows, dead_cols] = 0

    return bayer_int


def save_image(image: np.ndarray, path: str):
    """
    保存图像到文件

    支持格式：PNG（推荐，无损）、JPG、BMP 等
    需要 Pillow 库支持。

    参数:
        image: RGB 图像，uint8 (H, W, 3) 或灰度 (H, W)
        path: 输出文件路径
    """
    try:
        from PIL import Image
        img_pil = Image.fromarray(image)
        img_pil.save(path)
        print(f"  图像已保存: {path}")
    except ImportError:
        # 如果没有 Pillow，用 numpy 保存为 npy 格式
        npy_path = path.replace('.png', '.npy').replace('.jpg', '.npy')
        np.save(npy_path, image)
        print(f"  Pillow 未安装，图像保存为 NumPy 格式: {npy_path}")


def visualize_intermediate_steps(
    pipeline: ISPPipeline,
    raw: np.ndarray,
    output_dir: str = './isp_output',
):
    """
    可视化 ISP 各中间步骤的输出

    参数:
        pipeline: ISP 流程实例
        raw: 输入 RAW 图像
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n正在提取各步骤中间结果，输出到: {output_dir}")

    intermediate = pipeline.get_intermediate_results(raw)

    for step_name, img in intermediate.items():
        if img.ndim == 2:
            # Bayer 图：可视化为灰度图
            img_vis = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
            # 将单通道转为三通道以便统一保存
            img_vis = np.stack([img_vis] * 3, axis=-1)
        else:
            img_vis = img

        save_image(img_vis, os.path.join(output_dir, f"{step_name}.png"))

    print(f"共保存 {len(intermediate)} 个中间步骤图像")


def demo_basic(output_dir: str = './isp_output'):
    """
    基础演示：使用默认配置处理合成 RAW 图像
    """
    print("=" * 60)
    print("ISP 全流程演示 - 基础模式")
    print("=" * 60)

    # 生成合成 RAW 测试图像
    print("\n[1/3] 生成合成 RAW 测试图像...")
    raw = generate_synthetic_raw(
        height=512,
        width=512,
        bayer_pattern='RGGB',
        bit_depth=12,
        noise_level=0.015,
        add_vignetting=True,
        add_bad_pixels=True,
    )
    print(f"  RAW 图像: shape={raw.shape}, dtype={raw.dtype}, "
          f"range=[{raw.min()}, {raw.max()}]")

    # 初始化 ISP 流程（使用默认配置）
    print("\n[2/3] 初始化 ISP 流程（默认配置）...")
    config = ISPConfig(
        bayer_pattern='RGGB',
        bit_depth=12,
        black_level=64.0,
        white_level=4095.0,
        demosaic_method='malvar',      # Malvar 去马赛克（速度快、质量好）
        awb_method='gray_world',        # 灰色世界白平衡
        tone_mapping_method='aces',     # ACES 色调映射
        nr_method='bilateral',          # 双边滤波去噪
        sharp_method='usm',             # 反锐化掩蔽
        sharp_strength=0.5,
    )
    pipeline = ISPPipeline(config)

    # 执行 ISP 处理
    print("\n[3/3] 执行 ISP 全流程处理...")
    output = pipeline.process(raw, verbose=True)
    print(f"  输出图像: shape={output.shape}, dtype={output.dtype}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    save_image(output, os.path.join(output_dir, 'isp_output_default.png'))

    # 可视化中间步骤
    visualize_intermediate_steps(pipeline, raw, output_dir)

    return output


def demo_compare_methods(output_dir: str = './isp_output'):
    """
    对比演示：比较不同算法配置的效果
    """
    print("\n" + "=" * 60)
    print("ISP 算法对比演示")
    print("=" * 60)

    raw = generate_synthetic_raw(height=256, width=256, noise_level=0.02)

    # 定义多组对比配置
    configs = {
        'bilinear_reinhard': ISPConfig(
            demosaic_method='bilinear',
            tone_mapping_method='reinhard',
            nr_method='gaussian',
            sharp_strength=0.3,
        ),
        'malvar_aces': ISPConfig(
            demosaic_method='malvar',
            tone_mapping_method='aces',
            nr_method='bilateral',
            sharp_strength=0.5,
        ),
        'ahd_filmic': ISPConfig(
            demosaic_method='ahd',
            tone_mapping_method='filmic',
            nr_method='guided',
            sharp_strength=0.6,
        ),
        'malvar_drago_nlm': ISPConfig(
            demosaic_method='malvar',
            tone_mapping_method='drago',
            nr_method='nlm',
            sharp_method='adaptive',
            sharp_strength=0.4,
        ),
    }

    os.makedirs(output_dir, exist_ok=True)

    print("\n对比不同配置效果:")
    for config_name, config in configs.items():
        pipeline = ISPPipeline(config)
        output = pipeline.process(raw, verbose=False)
        save_path = os.path.join(output_dir, f'compare_{config_name}.png')
        save_image(output, save_path)
        print(f"  [{config_name}] → {save_path}")

    print(f"\n对比结果已保存到: {output_dir}")


def demo_noise_levels(output_dir: str = './isp_output'):
    """
    演示不同噪声水平下的 ISP 处理效果
    """
    print("\n" + "=" * 60)
    print("噪声水平测试演示")
    print("=" * 60)

    noise_levels = [0.005, 0.02, 0.05, 0.1]
    os.makedirs(output_dir, exist_ok=True)

    # 强去噪配置（适合高噪声场景）
    config_strong_nr = ISPConfig(
        nr_method='bilateral',
        nr_sigma_spatial=2.0,
        nr_sigma_color=0.15,
        sharp_strength=0.3,  # 去噪强时适当降低锐化
    )

    pipeline = ISPPipeline(config_strong_nr)

    for noise in noise_levels:
        raw = generate_synthetic_raw(height=256, width=256, noise_level=noise)
        output = pipeline.process(raw)
        save_image(output, os.path.join(output_dir, f'noise_{noise:.3f}.png'))
        print(f"  噪声水平 {noise:.3f} → 处理完成")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='ISP 全流程演示')
    parser.add_argument('--input', type=str, default=None,
                        help='输入 RAW 文件路径（.npy 格式），None 使用合成测试图')
    parser.add_argument('--output-dir', type=str, default='./isp_output',
                        help='输出目录（默认 ./isp_output）')
    parser.add_argument('--compare', action='store_true',
                        help='运行算法对比演示')
    parser.add_argument('--noise-test', action='store_true',
                        help='运行噪声水平测试')
    parser.add_argument('--bayer', type=str, default='RGGB',
                        choices=['RGGB', 'BGGR', 'GRBG', 'GBRG'],
                        help='Bayer 排列模式')
    parser.add_argument('--demosaic', type=str, default='malvar',
                        choices=['bilinear', 'malvar', 'ahd'],
                        help='去马赛克算法')
    parser.add_argument('--tonemapping', type=str, default='aces',
                        choices=['reinhard', 'reinhard_ext', 'filmic', 'aces', 'drago', 'gamma_only'],
                        help='色调映射算法')

    args = parser.parse_args()

    if args.input is not None:
        # 处理指定的 RAW 文件
        print(f"加载 RAW 文件: {args.input}")
        raw = np.load(args.input)
        config = ISPConfig(
            bayer_pattern=args.bayer,
            demosaic_method=args.demosaic,
            tone_mapping_method=args.tonemapping,
        )
        pipeline = ISPPipeline(config)
        output = pipeline.process(raw, verbose=True)
        os.makedirs(args.output_dir, exist_ok=True)
        save_image(output, os.path.join(args.output_dir, 'output.png'))

    elif args.compare:
        demo_compare_methods(args.output_dir)

    elif args.noise_test:
        demo_noise_levels(args.output_dir)

    else:
        demo_basic(args.output_dir)


if __name__ == '__main__':
    main()
