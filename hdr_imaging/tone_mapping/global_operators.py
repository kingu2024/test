"""
全局色调映射算子模块

全局算子（Global Tone Mapping Operators）对图像中所有像素统一应用相同的映射函数，
不考虑像素的空间邻域关系。其核心思想是通过全局亮度统计信息（如对数平均亮度、最大亮度）
来构建一个单调的压缩曲线，将 HDR 宽动态范围（High Dynamic Range）映射到 LDR
可显示范围 [0, 1]。

本模块实现三种经典全局算子：
  1. ReinhardGlobal  — Reinhard et al. (2002) 基于感知的全局色调映射
  2. DragoToneMap    — Drago et al. (2003) 自适应对数基底映射
  3. AdaptiveLog     — 自适应对数映射（无 OpenCV 对应实现）

输入：float32 HDR 辐照图（BGR 通道顺序，OpenCV 约定）
输出：uint8 LDR 图像（BGR，范围 [0, 255]）
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_luminance(bgr_image: np.ndarray) -> np.ndarray:
    """从 BGR 图像中提取亮度通道。

    使用 ITU-R BT.709 线性亮度公式：
        L = 0.0722·B + 0.7152·G + 0.2126·R

    其中 OpenCV BGR 通道顺序：
        B = img[:, :, 0]
        G = img[:, :, 1]
        R = img[:, :, 2]

    参数
    ----
    bgr_image : np.ndarray
        输入 BGR float32 图像，形状 (H, W, 3)。

    返回
    ----
    np.ndarray
        亮度图，形状 (H, W)，float32。
    """
    B = bgr_image[:, :, 0]
    G = bgr_image[:, :, 1]
    R = bgr_image[:, :, 2]
    return 0.0722 * B + 0.7152 * G + 0.2126 * R


def _recover_color(
    hdr_image: np.ndarray,
    L_original: np.ndarray,
    L_mapped: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """根据映射后亮度恢复彩色图像。

    对每个通道按亮度比例恢复色彩：
        C_out = C_in · (L_mapped / (L_original + eps))

    参数
    ----
    hdr_image : np.ndarray
        原始 HDR BGR 图像，形状 (H, W, 3)。
    L_original : np.ndarray
        映射前亮度，形状 (H, W)。
    L_mapped : np.ndarray
        映射后亮度，形状 (H, W)。
    eps : float
        防止除零的小量，默认 1e-6。

    返回
    ----
    np.ndarray
        恢复彩色后的图像，形状 (H, W, 3)，float32。
    """
    ratio = L_mapped / (L_original + eps)
    output = hdr_image * ratio[:, :, np.newaxis]
    return output.astype(np.float32)


def _apply_gamma_and_convert(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """对图像应用 Gamma 校正并转换为 uint8。

    步骤：
        1. 裁剪到 [0, 1]
        2. output = image^(1/gamma)
        3. 乘以 255 并转为 uint8

    参数
    ----
    image : np.ndarray
        float32 图像，值域理论上在 [0, 1]。
    gamma : float
        Gamma 值，默认 2.2。

    返回
    ----
    np.ndarray
        uint8 图像，形状与输入相同，值域 [0, 255]。
    """
    clipped = np.clip(image, 0.0, 1.0)
    gamma_corrected = np.power(clipped, 1.0 / gamma)
    return (gamma_corrected * 255.0).astype(np.uint8)


class ReinhardGlobal:
    """Reinhard (2002) 全局色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Reinhard 等人在 2002 年提出一种模拟人眼感知的全局色调映射方法。

    1. 计算对数平均亮度（感知亮度中心）：
           L̄ = exp((1/N) · Σ ln(δ + L(x,y)))
       δ = 1e-6 防止 ln(0) 奇异性

    2. 将场景亮度缩放到 key value a 所在的感知区间：
           L_scaled = (a / L̄) · L
       key value a 默认 0.18，对应"中等灰度"

    3. 应用色调映射公式（含白点限制）：
           L_d = L_scaled · (1 + L_scaled / L_white²) / (1 + L_scaled)
       L_white 是映射到纯白的最小亮度值；默认取场景最大亮度

    4. 从亮度恢复彩色：
           C_out = (C_in / (L_in + ε)) · L_d

    5. Gamma 校正（γ = 2.2）并转换为 uint8
    """

    def __init__(self, key_value: float = 0.18, white_point: Optional[float] = None):
        """初始化 Reinhard 全局算子。

        参数
        ----
        key_value : float
            场景关键值 a，控制整体曝光感知，默认 0.18（中等灰度）。
        white_point : float or None
            映射到纯白的最小亮度值 L_white；若为 None，则使用场景最大亮度。
        """
        self.key_value = key_value
        self.white_point = white_point
        logger.debug(
            "ReinhardGlobal 初始化：key_value=%.4f, white_point=%s",
            key_value,
            white_point,
        )

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 Reinhard 全局色调映射。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像，形状 (H, W, 3)。

        返回
        ----
        np.ndarray
            uint8 BGR 图像，形状 (H, W, 3)，值域 [0, 255]。
        """
        hdr = hdr_image.astype(np.float32)

        # 1. 提取亮度
        L = _extract_luminance(hdr)

        # 2. 计算对数平均亮度
        delta = 1e-6
        log_avg_lum = np.exp(np.mean(np.log(delta + L)))
        logger.debug("对数平均亮度 L̄ = %.6f", log_avg_lum)

        # 3. 缩放亮度
        L_scaled = (self.key_value / (log_avg_lum + delta)) * L

        # 4. 确定白点
        if self.white_point is None:
            L_white = L_scaled.max()
        else:
            L_white = float(self.white_point)

        if L_white < delta:
            L_white = 1.0
        logger.debug("白点 L_white = %.6f", L_white)

        # 5. 色调映射公式
        L_white2 = L_white * L_white
        L_d = L_scaled * (1.0 + L_scaled / L_white2) / (1.0 + L_scaled)

        # 6. 恢复彩色
        output = _recover_color(hdr, L, L_d)

        # 7. Gamma 校正并转 uint8
        return _apply_gamma_and_convert(output, gamma=2.2)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """使用 OpenCV 内置 Reinhard 色调映射算子。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像。

        返回
        ----
        np.ndarray
            uint8 BGR 图像。
        """
        tonemap = cv2.createTonemapReinhard(
            gamma=2.2,
            intensity=0.0,
            light_adapt=0.0,
            color_adapt=0.0,
        )
        ldr = tonemap.process(hdr_image)
        return (np.clip(ldr, 0.0, 1.0) * 255).astype(np.uint8)


class DragoToneMap:
    """Drago (2003) 自适应对数基底色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Drago 等人在 2003 年提出基于自适应对数基底的色调映射方法，通过偏差参数 b
    自适应调整对数基底，使亮部和暗部细节得到均衡保留。

    映射公式：
        L_d = (L_d_max / log10(1 + L_max))
              · log(1 + L)
              / log(2 + 8·(L/L_max)^(log(b)/log(0.5)))

    其中：
        b       — 偏差参数，控制对数基底自适应程度，默认 0.85
        L_d_max — 显示器最大亮度，通常取 100 cd/m²
        L_max   — 场景最大亮度

    恢复彩色后进行 Gamma 校正并转换为 uint8。
    """

    def __init__(
        self,
        gamma: float = 2.2,
        saturation: float = 1.0,
        bias: float = 0.85,
    ):
        """初始化 Drago 色调映射算子。

        参数
        ----
        gamma : float
            Gamma 校正值，默认 2.2。
        saturation : float
            饱和度系数（仅用于 OpenCV 接口），默认 1.0。
        bias : float
            偏差参数 b，控制自适应对数基底，范围 (0, 1)，默认 0.85。
        """
        self.gamma = gamma
        self.saturation = saturation
        self.bias = bias
        logger.debug(
            "DragoToneMap 初始化：gamma=%.2f, saturation=%.2f, bias=%.4f",
            gamma,
            saturation,
            bias,
        )

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 Drago 自适应对数色调映射。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像，形状 (H, W, 3)。

        返回
        ----
        np.ndarray
            uint8 BGR 图像，形状 (H, W, 3)，值域 [0, 255]。
        """
        hdr = hdr_image.astype(np.float32)

        # 1. 提取亮度
        L = _extract_luminance(hdr)

        L_max = float(L.max())
        if L_max < 1e-6:
            L_max = 1.0
        logger.debug("场景最大亮度 L_max = %.6f", L_max)

        # 2. Drago 映射公式
        L_d_max = 100.0  # 显示器最大亮度 (cd/m²)

        # log(b) / log(0.5)
        log_b_over_log05 = np.log(self.bias) / np.log(0.5)

        # 自适应对数基底分母：log(2 + 8·(L/L_max)^(log(b)/log(0.5)))
        exponent = np.power(np.clip(L / L_max, 0.0, 1.0), log_b_over_log05)
        denom = np.log(2.0 + 8.0 * exponent)
        # 防止分母为零
        denom = np.where(denom < 1e-6, 1e-6, denom)

        # 分子：log(1 + L)
        numer = np.log(1.0 + L)

        # 全局缩放系数
        scale = L_d_max / np.log10(1.0 + L_max)

        L_d = scale * numer / denom

        # 归一化到 [0, 1]
        L_d_max_val = L_d.max()
        if L_d_max_val > 1e-6:
            L_d = L_d / L_d_max_val

        # 3. 恢复彩色
        output = _recover_color(hdr, L, L_d)

        # 4. Gamma 校正并转 uint8
        return _apply_gamma_and_convert(output, gamma=self.gamma)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """使用 OpenCV 内置 Drago 色调映射算子。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像。

        返回
        ----
        np.ndarray
            uint8 BGR 图像。
        """
        tonemap = cv2.createTonemapDrago(self.gamma, self.saturation, self.bias)
        ldr = tonemap.process(hdr_image)
        return (np.clip(ldr, 0.0, 1.0) * 255).astype(np.uint8)


class AdaptiveLog:
    """自适应对数色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    基于场景对数平均亮度的自适应对数映射。以对数平均亮度 L_avg 作为参考，
    自适应缩放对数映射的范围，使得中等亮度区域获得最大对比度保留。

    映射公式：
        L_d = L_d_max · log(1 + L/L_avg) / log(1 + L_max/L_avg)

    其中：
        L_avg   — 场景对数平均亮度（感知亮度中心）
        L_max   — 场景最大亮度
        L_d_max — 显示最大亮度，归一化后取 1.0

    该算子无对应的 OpenCV 内置实现，调用 process_opencv() 将抛出 NotImplementedError。
    """

    def __init__(self, gamma: float = 2.2):
        """初始化自适应对数色调映射算子。

        参数
        ----
        gamma : float
            Gamma 校正值，默认 2.2。
        """
        self.gamma = gamma
        logger.debug("AdaptiveLog 初始化：gamma=%.2f", gamma)

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行自适应对数色调映射。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像，形状 (H, W, 3)。

        返回
        ----
        np.ndarray
            uint8 BGR 图像，形状 (H, W, 3)，值域 [0, 255]。
        """
        hdr = hdr_image.astype(np.float32)

        # 1. 提取亮度
        L = _extract_luminance(hdr)

        # 2. 计算对数平均亮度 L_avg 和最大亮度 L_max
        delta = 1e-6
        L_avg = np.exp(np.mean(np.log(delta + L)))
        L_max = float(L.max())

        if L_avg < delta:
            L_avg = 1.0
        if L_max < delta:
            L_max = 1.0
        logger.debug("L_avg=%.6f, L_max=%.6f", L_avg, L_max)

        # 3. 自适应对数映射公式
        L_d_max = 1.0  # 归一化显示最大亮度

        numer = np.log(1.0 + L / L_avg)
        denom = np.log(1.0 + L_max / L_avg)

        if denom < delta:
            denom = delta

        L_d = L_d_max * numer / denom

        # 4. 恢复彩色
        output = _recover_color(hdr, L, L_d)

        # 5. Gamma 校正并转 uint8
        return _apply_gamma_and_convert(output, gamma=self.gamma)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """AdaptiveLog 无对应的 OpenCV 内置实现。

        抛出
        ----
        NotImplementedError
            始终抛出，提示用户改用 process() 方法。
        """
        raise NotImplementedError(
            "AdaptiveLog has no OpenCV equivalent. Use process() instead."
        )
