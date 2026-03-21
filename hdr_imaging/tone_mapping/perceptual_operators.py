"""
感知色调映射算子模块

感知算子（Perceptual Tone Mapping Operators）以人类视觉感知模型为驱动，
通过模拟人眼对亮度、对比度和色彩的响应特性，将 HDR 宽动态范围图像映射到
LDR 可显示范围 [0, 1]。与全局算子和局部算子不同，感知算子更注重视觉上的
真实感与艺术风格，广泛应用于电影工业和游戏行业的色调管线（Color Pipeline）。

本模块实现四种感知驱动算子：
  1. ACESToneMap      — ACES 电影色调映射曲线（Narkowicz 2015 近似）
  2. FilmicToneMap    — Uncharted 2 Filmic 曲线（Hable 2010）
  3. MantiukToneMap   — Mantiuk (2006) 感知对比度映射
  4. HistogramToneMap — 直方图均衡化色调映射

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
    saturation: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """根据映射后亮度恢复彩色图像，可附加饱和度调整。

    对每个通道按亮度比例恢复色彩：
        C_out = C_in · (L_mapped / (L_original + eps))

    若 saturation != 1.0，先对原始色彩进行饱和度缩放：
        C_sat = L_original + saturation · (C_in - L_original)

    参数
    ----
    hdr_image : np.ndarray
        原始 HDR BGR 图像，形状 (H, W, 3)。
    L_original : np.ndarray
        映射前亮度，形状 (H, W)。
    L_mapped : np.ndarray
        映射后亮度，形状 (H, W)。
    saturation : float
        饱和度系数，默认 1.0（不调整）。
    eps : float
        防止除零的小量，默认 1e-6。

    返回
    ----
    np.ndarray
        恢复彩色后的图像，形状 (H, W, 3)，float32。
    """
    L_orig_3 = L_original[:, :, np.newaxis]
    if saturation != 1.0:
        hdr_sat = L_orig_3 + saturation * (hdr_image - L_orig_3)
    else:
        hdr_sat = hdr_image

    ratio = L_mapped / (L_original + eps)
    output = hdr_sat * ratio[:, :, np.newaxis]
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


class ACESToneMap:
    """ACES 电影色调映射曲线（Narkowicz 2015 近似）。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ACES（Academy Color Encoding System）是电影工业标准色彩管线中的
    标准色调映射曲线。Narkowicz (2015) 提出一种高效的有理函数近似：

        f(x) = (x·(a·x + b)) / (x·(c·x + d) + e)

    标准参数：a=2.51, b=0.03, c=2.43, d=0.59, e=0.14

    该曲线具备以下特性：
        - 暗部细节保留优秀（类似对数响应）
        - 中间调对比度适中
        - 亮部柔和压缩（肩部曲线）

    操作流程：
        1. 乘以曝光系数 exposure
        2. 对每个 BGR 通道直接应用 ACES 曲线（非亮度域）
        3. Gamma 校正（γ = 2.2）并转换为 uint8
    """

    def __init__(self, exposure: float = 1.0, gamma: float = 2.2):
        """初始化 ACES 色调映射算子。

        参数
        ----
        exposure : float
            曝光系数，乘以输入 HDR 数据，默认 1.0。
        gamma : float
            Gamma 校正值，默认 2.2。
        """
        self.exposure = exposure
        self.gamma = gamma
        logger.debug("ACESToneMap 初始化：exposure=%.4f, gamma=%.2f", exposure, gamma)

    def _aces_curve(self, x: np.ndarray) -> np.ndarray:
        """应用 ACES Narkowicz 2015 有理函数近似曲线。

        公式：
            f(x) = (x·(2.51·x + 0.03)) / (x·(2.43·x + 0.59) + 0.14)

        参数
        ----
        x : np.ndarray
            输入值（任意形状，float32）。

        返回
        ----
        np.ndarray
            映射后的值，与输入形状相同。
        """
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        numer = x * (a * x + b)
        denom = x * (c * x + d) + e
        return numer / denom

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 ACES 电影色调映射。

        步骤：
            1. 按曝光系数缩放：hdr = hdr_image × exposure
            2. 对每个通道直接应用 ACES 有理函数曲线
            3. Gamma 校正（γ = 2.2），截断到 [0, 1]，转换为 uint8

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像，形状 (H, W, 3)。

        返回
        ----
        np.ndarray
            uint8 BGR 图像，形状 (H, W, 3)，值域 [0, 255]。
        """
        hdr = hdr_image.astype(np.float32) * self.exposure

        # 对每个通道直接应用 ACES 曲线（非亮度域，直接 RGB 映射）
        mapped = self._aces_curve(hdr)

        # Gamma 校正并转 uint8
        return _apply_gamma_and_convert(mapped, gamma=self.gamma)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """ACES 无对应的 OpenCV 内置实现。

        抛出
        ----
        NotImplementedError
            始终抛出，提示用户改用 process() 方法。
        """
        raise NotImplementedError("ACES has no OpenCV equivalent. Use process() instead.")


class FilmicToneMap:
    """Uncharted 2 Filmic 曲线（Hable 2010）色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    John Hable 在 2010 年 GDC 演讲中为游戏《神秘海域 2》提出一种
    模拟胶片响应曲线的分段色调映射函数：

        f(x) = ((x·(A·x + C·B) + D·E) / (x·(A·x + B) + D·F)) - E/F

    标准参数（游戏行业常用值）：
        A = 0.15（肩部强度 Shoulder Strength）
        B = 0.50（线性强度 Linear Strength）
        C = 0.10（线性角度 Linear Angle）
        D = 0.20（趾部强度 Toe Strength）
        E = 0.02（趾部数值 Toe Numerator）
        F = 0.30（趾部分母 Toe Denominator）

    归一化：
        result = f(exposure × color) / f(white_point)

    最终结果裁剪到 [0, 1] 后做 Gamma 校正。
    """

    def __init__(self, exposure: float = 2.0, white_point: float = 11.2):
        """初始化 Filmic 色调映射算子。

        参数
        ----
        exposure : float
            曝光系数，默认 2.0（Hable 推荐值）。
        white_point : float
            白点亮度，用于归一化，默认 11.2（Hable 推荐值）。
        """
        self.exposure = exposure
        self.white_point = white_point
        logger.debug(
            "FilmicToneMap 初始化：exposure=%.4f, white_point=%.4f",
            exposure,
            white_point,
        )

    def _filmic_curve(self, x: np.ndarray) -> np.ndarray:
        """应用 Uncharted 2 Filmic 分段曲线。

        公式：
            f(x) = ((x·(A·x + C·B) + D·E) / (x·(A·x + B) + D·F)) - E/F

        参数
        ----
        x : np.ndarray
            输入值（任意形状，float32）。

        返回
        ----
        np.ndarray
            映射后的值，与输入形状相同。
        """
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        numer = x * (A * x + C * B) + D * E
        denom = x * (A * x + B) + D * F
        return numer / denom - E / F

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 Filmic 色调映射。

        步骤：
            1. 对每个通道应用 f(exposure × color)
            2. 除以 f(white_point) 归一化
            3. Gamma 校正（γ = 2.2），截断到 [0, 1]，转换为 uint8

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

        # 应用曲线后归一化
        numerator = self._filmic_curve(self.exposure * hdr)
        denominator = self._filmic_curve(
            np.full(1, self.white_point, dtype=np.float32)
        )[0]

        if abs(denominator) < 1e-6:
            denominator = 1e-6

        mapped = numerator / denominator

        # Gamma 校正并转 uint8
        return _apply_gamma_and_convert(mapped, gamma=2.2)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """Filmic 无对应的 OpenCV 内置实现。

        抛出
        ----
        NotImplementedError
            始终抛出，提示用户改用 process() 方法。
        """
        raise NotImplementedError("Filmic has no OpenCV equivalent. Use process() instead.")


class MantiukToneMap:
    """Mantiuk (2006) 感知对比度映射色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Mantiuk 等人在 2006 年提出基于感知对比度的色调映射方法，
    在对数亮度域中进行局部对比度均衡化，模拟人眼对对比度的
    非线性感知（Weber–Fechner 定律）。

    简化实现步骤：
        1. 提取亮度 L，计算对数亮度 log_L = log(L + ε)
        2. 通过高斯模糊估计局部均值 local_mean
        3. 局部对比度 = log_L - local_mean
        4. 对局部对比度乘以压缩系数 scale（减小对比度差异）
        5. 重建对数亮度：log_L_new = local_mean + scale × contrast
        6. 映射回线性域：L_new = exp(log_L_new)，归一化到 [0, 1]
        7. 恢复色彩（可调饱和度），Gamma 校正，转 uint8

    该方法能在保留局部细节的同时均衡整体对比度分布。
    """

    def __init__(self, gamma: float = 2.2, scale: float = 0.85, saturation: float = 1.0):
        """初始化 Mantiuk 感知对比度色调映射算子。

        参数
        ----
        gamma : float
            Gamma 校正值，默认 2.2。
        scale : float
            对比度压缩系数，值越小对比度越平坦，默认 0.85。
        saturation : float
            色彩饱和度系数，默认 1.0（不调整）。
        """
        self.gamma = gamma
        self.scale = scale
        self.saturation = saturation
        logger.debug(
            "MantiukToneMap 初始化：gamma=%.2f, scale=%.4f, saturation=%.4f",
            gamma,
            scale,
            saturation,
        )

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 Mantiuk 感知对比度色调映射。

        步骤：
            1. 提取亮度 L
            2. 计算对数亮度 log_L，通过高斯模糊获取 local_mean
            3. 局部对比度 = log_L - local_mean
            4. 对比度乘以 scale 压缩
            5. 重建 log_L_new = local_mean + scale × contrast
            6. 映射回线性域并归一化
            7. 恢复色彩，Gamma 校正，转 uint8

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

        eps = 1e-6

        # 2. 对数亮度与局部均值（高斯核估计邻域均值）
        log_L = np.log(L + eps)
        local_mean = cv2.GaussianBlur(log_L, (0, 0), sigmaX=2.0)

        # 3. 局部对比度
        local_contrast = log_L - local_mean

        # 4-5. 用 scale 压缩对比度，重建对数亮度
        log_L_new = local_mean + self.scale * local_contrast

        # 6. 映射回线性域，归一化到 [0, 1]
        L_new = np.exp(log_L_new)
        L_max = L_new.max()
        if L_max > eps:
            L_new = L_new / L_max

        # 7. 恢复色彩（含饱和度调整），Gamma 校正，转 uint8
        output = _recover_color(hdr, L, L_new, saturation=self.saturation)
        return _apply_gamma_and_convert(output, gamma=self.gamma)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """使用 OpenCV 内置 Mantiuk 色调映射算子。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像。

        返回
        ----
        np.ndarray
            uint8 BGR 图像。
        """
        tonemap = cv2.createTonemapMantiuk(self.gamma, self.scale, self.saturation)
        ldr = tonemap.process(hdr_image)
        return (np.clip(ldr, 0.0, 1.0) * 255).astype(np.uint8)


class HistogramToneMap:
    """直方图均衡化色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    基于直方图均衡化（Histogram Equalization）的 HDR 色调映射方法。
    通过 HDR 亮度的对数域直方图累积分布函数（CDF）将亮度重映射到
    [0, 1]，使输出亮度分布更加均匀，充分利用显示动态范围。

    操作步骤：
        1. 提取亮度 L
        2. 对数变换：log_L = log(L + ε)，将宽动态范围压缩到对数域
        3. 计算 log_L 的直方图（num_bins 个区间）
        4. 若 clip_limit > 0，截断直方图峰值（类 CLAHE 概念），
           抑制过度对比度增强
        5. 计算归一化 CDF（累积分布函数）
        6. 用 CDF 将每个像素的对数亮度映射到 [0, 1]
        7. 恢复色彩，Gamma 校正，转 uint8

    直方图均衡化能使输出亮度直方图趋向均匀分布，自适应地增强
    HDR 场景中各亮度范围的细节与对比度。
    """

    def __init__(self, clip_limit: float = 0.0, num_bins: int = 256):
        """初始化直方图均衡化色调映射算子。

        参数
        ----
        clip_limit : float
            直方图截断阈值（相对于均匀分布频率的倍数）；
            0.0 表示不截断，默认 0.0。
        num_bins : int
            直方图区间数，默认 256。
        """
        self.clip_limit = clip_limit
        self.num_bins = num_bins
        logger.debug(
            "HistogramToneMap 初始化：clip_limit=%.4f, num_bins=%d",
            clip_limit,
            num_bins,
        )

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行直方图均衡化色调映射。

        步骤：
            1. 提取亮度 L
            2. 对数变换 log_L = log(L + ε)
            3. 计算 log_L 直方图（num_bins 个区间）
            4. 若 clip_limit > 0 则截断直方图
            5. 计算归一化 CDF，映射亮度到 [0, 1]
            6. 恢复色彩，Gamma 校正，转 uint8

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

        eps = 1e-6

        # 2. 对数变换
        log_L = np.log(L + eps)

        log_min = float(log_L.min())
        log_max = float(log_L.max())
        log_range = log_max - log_min
        if log_range < eps:
            log_range = eps

        # 3. 计算对数亮度直方图
        hist, bin_edges = np.histogram(log_L, bins=self.num_bins)
        hist = hist.astype(np.float32)

        # 4. 若 clip_limit > 0，截断直方图峰值（抑制过度对比度增强）
        if self.clip_limit > 0.0:
            n_pixels = L.size
            clip_value = self.clip_limit * (n_pixels / self.num_bins)
            excess = np.sum(np.maximum(hist - clip_value, 0.0))
            hist = np.minimum(hist, clip_value)
            # 将截断的频数均匀分配回各区间
            hist += excess / self.num_bins

        # 5. 计算归一化 CDF
        cdf = np.cumsum(hist)
        cdf_min = cdf[0]
        cdf_max = cdf[-1]
        if (cdf_max - cdf_min) < eps:
            cdf_normalized = np.zeros_like(cdf)
        else:
            cdf_normalized = (cdf - cdf_min) / (cdf_max - cdf_min)

        # 6. 将每个像素的 log_L 映射到 [0, 1]：查找对应 bin，取 CDF 值
        # 计算每个像素所在的 bin 索引
        bin_indices = np.floor(
            (log_L - log_min) / log_range * (self.num_bins - 1)
        ).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        L_mapped = cdf_normalized[bin_indices]

        # 7. 恢复色彩，Gamma 校正，转 uint8
        output = _recover_color(hdr, L, L_mapped)
        return _apply_gamma_and_convert(output, gamma=2.2)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """HistogramToneMap 无对应的 OpenCV 内置实现。

        抛出
        ----
        NotImplementedError
            始终抛出，提示用户改用 process() 方法。
        """
        raise NotImplementedError(
            "HistogramToneMap has no OpenCV equivalent. Use process() instead."
        )
