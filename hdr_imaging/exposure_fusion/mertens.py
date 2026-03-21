"""
Mertens 曝光融合模块 - Mertens et al. (2007) 多曝光图像融合

【算法原理】Mertens (2007) 数学原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

不经过 HDR 辐射图，直接从多曝光 LDR 图像融合出高质量结果。

1) 质量度量计算 (对每张图像):
   - 对比度 C(x,y): 灰度图的拉普拉斯滤波绝对值 |Laplacian(gray)|
   - 饱和度 S(x,y): RGB 三通道的标准差 std(R,G,B)
   - 曝光度 E(x,y): 各通道与 0.5 的偏差的高斯函数乘积
     E = Π_c exp(-0.5 · ((I_c - 0.5) / σ)²)，σ 默认 0.2

2) 组合权重:
     W_k(x,y) = C_k^wc · S_k^ws · E_k^we + ε (ε=1e-6 防止全零)

3) 归一化:
     Ŵ_k = W_k / Σ_k W_k

4) 拉普拉斯金字塔融合:
   - 对每张图像构建拉普拉斯金字塔 L_k
   - 对归一化权重构建高斯金字塔 G(Ŵ_k)
   - 逐层融合: L_fused[l] = Σ_k G(Ŵ_k)[l] · L_k[l]
   - 从融合金字塔重建最终图像
"""

import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class MertensFusion:
    """
    Mertens 多曝光融合算法 (Mertens et al., 2007)

    直接从多张不同曝光的 LDR 图像融合出高质量结果，
    无需 HDR 辐射图重建或色调映射步骤。
    """

    def __init__(
        self,
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        exposure_weight: float = 1.0,
        sigma: float = 0.2,
        pyramid_levels: int = None,
    ):
        """
        初始化 Mertens 融合器。

        Args:
            contrast_weight:   对比度权重指数 wc，默认 1.0
            saturation_weight: 饱和度权重指数 ws，默认 1.0
            exposure_weight:   曝光度权重指数 we，默认 1.0
            sigma:             曝光度高斯函数的标准差 σ，默认 0.2
            pyramid_levels:    金字塔层数；若为 None，则根据图像尺寸自动计算
                               floor(log2(min(h,w))) - 1，上限为 8
        """
        self.contrast_weight = contrast_weight
        self.saturation_weight = saturation_weight
        self.exposure_weight = exposure_weight
        self.sigma = sigma
        self.pyramid_levels = pyramid_levels

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _auto_levels(self, h: int, w: int) -> int:
        """根据图像尺寸自动计算金字塔层数。"""
        levels = int(np.floor(np.log2(min(h, w)))) - 1
        return min(max(levels, 1), 8)

    def _compute_weights(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        计算每张图像的归一化融合权重图。

        Args:
            images: float32 图像列表，每张形状 (H, W, 3)，值域 [0, 1]

        Returns:
            归一化权重图列表，每张形状 (H, W)
        """
        laplacian_kernel = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=np.float32
        )

        raw_weights = []
        for img in images:
            # --- 对比度：灰度拉普拉斯绝对值 ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = np.abs(
                cv2.filter2D(gray, -1, laplacian_kernel)
            ).astype(np.float32)

            # --- 饱和度：RGB 三通道标准差 ---
            saturation = np.std(img, axis=2).astype(np.float32)

            # --- 曝光度：各通道偏离 0.5 的高斯乘积 ---
            gauss = np.exp(
                -0.5 * ((img - 0.5) / self.sigma) ** 2
            )  # (H, W, 3)
            well_exposed = np.prod(gauss, axis=2).astype(np.float32)

            # --- 组合权重（带 ε 防止全零） ---
            w = (
                (contrast ** self.contrast_weight)
                * (saturation ** self.saturation_weight)
                * (well_exposed ** self.exposure_weight)
                + 1e-6
            )
            raw_weights.append(w)

        # --- 跨图像归一化 ---
        weight_sum = np.sum(raw_weights, axis=0)  # (H, W)
        normalized = [w / weight_sum for w in raw_weights]
        return normalized

    def _build_gaussian_pyramid(
        self, image: np.ndarray, levels: int
    ) -> List[np.ndarray]:
        """
        构建高斯金字塔。

        Args:
            image:  输入图像，float32，形状 (H, W) 或 (H, W, C)
            levels: 金字塔层数

        Returns:
            高斯金字塔层列表（从大到小），长度为 levels
        """
        pyramid = [image]
        current = image
        for _ in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _build_laplacian_pyramid(
        self, image: np.ndarray, levels: int
    ) -> List[np.ndarray]:
        """
        构建拉普拉斯金字塔。

        每一层 = 该层图像 - upsample(downsample(该层图像))。
        最后一层保留最小尺寸的高斯层。

        Args:
            image:  输入图像，float32，形状 (H, W, C)
            levels: 金字塔层数

        Returns:
            拉普拉斯金字塔层列表（从大到小），长度为 levels
        """
        gaussian_pyr = self._build_gaussian_pyramid(image, levels)
        laplacian_pyr = []
        for i in range(levels - 1):
            g_current = gaussian_pyr[i]
            g_next = gaussian_pyr[i + 1]
            # 上采样到当前层尺寸
            h, w = g_current.shape[:2]
            g_up = cv2.pyrUp(g_next, dstsize=(w, h))
            lap = g_current - g_up
            laplacian_pyr.append(lap)
        # 最后一层：最小尺寸的高斯层原样保留
        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr

    def _reconstruct_from_pyramid(
        self, pyramid: List[np.ndarray]
    ) -> np.ndarray:
        """
        从拉普拉斯金字塔重建图像（自底向上）。

        Args:
            pyramid: 拉普拉斯金字塔层列表（从大到小）

        Returns:
            重建图像，float32
        """
        # 从最小层开始，逐层向上叠加
        result = pyramid[-1]
        for lap in reversed(pyramid[:-1]):
            h, w = lap.shape[:2]
            result = cv2.pyrUp(result, dstsize=(w, h)) + lap
        return result

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def process(self, images: List[np.ndarray]) -> np.ndarray:
        """
        执行 Mertens 曝光融合（自定义拉普拉斯金字塔实现）。

        Args:
            images: uint8 BGR 图像列表，至少 2 张，尺寸须一致

        Returns:
            融合后的 uint8 BGR 图像

        Raises:
            ValueError: 输入图像数量不足 2 张
        """
        if len(images) < 2:
            raise ValueError(
                f"至少需要 2 张图像进行融合，当前仅提供 {len(images)} 张"
            )

        # 转换为 float32 [0, 1]
        float_images = [img.astype(np.float32) / 255.0 for img in images]

        h, w = float_images[0].shape[:2]
        levels = (
            self.pyramid_levels
            if self.pyramid_levels is not None
            else self._auto_levels(h, w)
        )
        logger.debug(
            "Mertens fusion: %d images, pyramid levels=%d, size=(%d,%d)",
            len(images), levels, h, w,
        )

        # 计算归一化权重图
        norm_weights = self._compute_weights(float_images)

        # 构建每张图像的拉普拉斯金字塔
        lap_pyramids = [
            self._build_laplacian_pyramid(img, levels)
            for img in float_images
        ]

        # 构建每张权重图的高斯金字塔
        gauss_weight_pyramids = [
            self._build_gaussian_pyramid(w_map, levels)
            for w_map in norm_weights
        ]

        # 逐层加权融合
        fused_pyramid = []
        for level_idx in range(levels):
            fused_level = None
            for k in range(len(float_images)):
                lap_level = lap_pyramids[k][level_idx]       # (H_l, W_l, 3)
                w_level = gauss_weight_pyramids[k][level_idx]  # (H_l, W_l)
                # 广播权重到三通道
                weighted = lap_level * w_level[:, :, np.newaxis]
                if fused_level is None:
                    fused_level = weighted
                else:
                    fused_level = fused_level + weighted
            fused_pyramid.append(fused_level)

        # 从融合金字塔重建图像
        result = self._reconstruct_from_pyramid(fused_pyramid)

        # 裁剪并转换为 uint8
        result = np.clip(result, 0.0, 1.0)
        return (result * 255.0).astype(np.uint8)

    def process_opencv(self, images: List[np.ndarray]) -> np.ndarray:
        """
        执行 Mertens 曝光融合（OpenCV 官方实现）。

        使用 cv2.createMergeMertens 作为参考实现。

        Args:
            images: uint8 BGR 图像列表，至少 2 张

        Returns:
            融合后的 uint8 BGR 图像
        """
        merge_mertens = cv2.createMergeMertens(
            self.contrast_weight,
            self.saturation_weight,
            self.exposure_weight,
        )
        result = merge_mertens.process(images)
        result = np.clip(result, 0.0, 1.0)
        return (result * 255.0).astype(np.uint8)
