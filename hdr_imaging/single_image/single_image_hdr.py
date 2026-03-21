"""
单张图像 HDR 增强模块

单张图像 HDR 增强原理:

从单张 LDR 图像估计 HDR 效果，无需多曝光输入。

1) 色彩空间转换:
   BGR → LAB，分离亮度通道 L 和色度通道 a, b

2) CLAHE (Contrast Limited Adaptive Histogram Equalization):
   - 将 L 通道划分为 tile_size × tile_size 个块
   - 每块独立做直方图均衡化
   - clip_limit 限制对比度增强幅度，防止过度增强噪声
   - 块边界使用双线性插值消除接缝

3) 多尺度细节增强:
   - 构建高斯金字塔提取不同尺度的细节信息
   - detail_level_k = image_k - GaussianBlur(image_k, sigma_k)
   - 加权增强: enhanced = base + detail_boost × Σ detail_k
   - detail_boost 控制细节增强程度

4) 色彩保持:
   - 在 LAB 空间处理保持 a, b 通道不变
   - 仅增强 L 通道，保证色彩不失真

5) 转回 BGR 输出
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SingleImageHDR:
    """
    单张图像 HDR 增强器

    通过 CLAHE 自适应直方图均衡化和多尺度细节增强，
    从单张 LDR 图像模拟 HDR 视觉效果。

    支持两种实现方式：
      - process()        — 手写 NumPy CLAHE 实现 + 多尺度细节增强
      - process_opencv() — 调用 cv2.createCLAHE() + 多尺度细节增强

    Attributes:
        clip_limit (float): CLAHE 对比度限制阈值
        tile_size (int): CLAHE 分块大小（tile_size × tile_size）
        detail_boost (float): 多尺度细节增强强度
    """

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_size: int = 8,
        detail_boost: float = 1.5,
    ):
        """
        初始化单张图像 HDR 增强器

        Args:
            clip_limit: CLAHE 对比度限制阈值，防止过度增强噪声。
                        推荐范围：1.0–10.0，默认 3.0。
            tile_size: CLAHE 分块大小，将图像划分为 tile_size × tile_size 个块。
                       推荐范围：4–16，默认 8。
            detail_boost: 多尺度细节增强强度，控制高频细节叠加幅度。
                          推荐范围：0.5–3.0，默认 1.5。
        """
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.detail_boost = detail_boost
        logger.info(
            f"SingleImageHDR 初始化: clip_limit={clip_limit}, "
            f"tile_size={tile_size}, detail_boost={detail_boost}"
        )

    def _apply_clahe_manual(self, L: np.ndarray) -> np.ndarray:
        """
        手写 CLAHE 实现（近似实现）

        算法步骤：
        1. 将图像分割为 tile_size × tile_size 大小的块（边缘块按需裁剪）
        2. 对每块计算直方图（256 bins）
        3. 按 clip_limit 限制直方图并重新分配裁剪计数
        4. 计算每块的 CDF，归一化为均衡化映射表
        5. 双线性插值合并相邻块的映射结果，消除块边界接缝

        Args:
            L: 亮度通道，uint8，形状 (H, W)，值域 [0, 255]

        Returns:
            enhanced_L: CLAHE 增强后的亮度通道，uint8，形状 (H, W)
        """
        H, W = L.shape
        num_bins = 256

        tile_h = self.tile_size
        tile_w = self.tile_size

        # 计算块数（向上取整，保证覆盖全图）
        n_tiles_y = (H + tile_h - 1) // tile_h
        n_tiles_x = (W + tile_w - 1) // tile_w

        # 为每个块构建均衡化映射表，形状 (n_tiles_y, n_tiles_x, 256)
        lut = np.zeros((n_tiles_y, n_tiles_x, num_bins), dtype=np.float32)

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # 当前块的像素范围
                y0 = ty * tile_h
                y1 = min(y0 + tile_h, H)
                x0 = tx * tile_w
                x1 = min(x0 + tile_w, W)

                tile = L[y0:y1, x0:x1]
                tile_pixels = tile.size

                # 计算直方图
                hist, _ = np.histogram(tile.ravel(), bins=num_bins, range=(0, 256))
                hist = hist.astype(np.float64)

                # 计算裁剪阈值：clip_limit × (像素数 / bin数)
                clip_threshold = self.clip_limit * (tile_pixels / num_bins)

                # 裁剪超出阈值的部分并重新分配
                excess = hist - clip_threshold
                excess = np.maximum(excess, 0.0)
                total_excess = excess.sum()

                hist = np.minimum(hist, clip_threshold)

                # 将裁剪掉的计数均匀分配到所有 bin
                hist += total_excess / num_bins

                # 计算 CDF 并归一化为 [0, 255] 的映射
                cdf = hist.cumsum()
                cdf_min = cdf[cdf > 0][0] if (cdf > 0).any() else 0.0
                if tile_pixels > cdf_min:
                    cdf_normalized = (cdf - cdf_min) / (tile_pixels - cdf_min) * 255.0
                else:
                    cdf_normalized = np.arange(num_bins, dtype=np.float64)

                lut[ty, tx] = cdf_normalized.astype(np.float32)

        # 双线性插值：每个像素根据其相对于块中心的位置，
        # 在相邻四块的映射表之间进行双线性插值
        result = np.zeros_like(L, dtype=np.float32)

        for y in range(H):
            for x in range(W):
                pixel_val = L[y, x]

                # 计算像素所属块坐标
                ty_f = (y + 0.5) / tile_h - 0.5
                tx_f = (x + 0.5) / tile_w - 0.5

                # 四邻块索引（钳制到有效范围）
                ty0 = int(np.floor(ty_f))
                ty1 = ty0 + 1
                tx0 = int(np.floor(tx_f))
                tx1 = tx0 + 1

                # 插值权重
                wy1 = ty_f - ty0
                wy0 = 1.0 - wy1
                wx1 = tx_f - tx0
                wx0 = 1.0 - wx1

                # 钳制索引到有效范围
                ty0c = max(0, min(ty0, n_tiles_y - 1))
                ty1c = max(0, min(ty1, n_tiles_y - 1))
                tx0c = max(0, min(tx0, n_tiles_x - 1))
                tx1c = max(0, min(tx1, n_tiles_x - 1))

                # 双线性插值映射值
                v00 = lut[ty0c, tx0c, pixel_val]
                v01 = lut[ty0c, tx1c, pixel_val]
                v10 = lut[ty1c, tx0c, pixel_val]
                v11 = lut[ty1c, tx1c, pixel_val]

                val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11)
                result[y, x] = val

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_clahe_manual_vectorized(self, L: np.ndarray) -> np.ndarray:
        """
        向量化手写 CLAHE 实现（加速版本）

        与 _apply_clahe_manual 算法相同，但使用向量化操作替代像素级循环，
        显著提升计算速度。

        Args:
            L: 亮度通道，uint8，形状 (H, W)，值域 [0, 255]

        Returns:
            enhanced_L: CLAHE 增强后的亮度通道，uint8，形状 (H, W)
        """
        H, W = L.shape
        num_bins = 256

        tile_h = self.tile_size
        tile_w = self.tile_size

        n_tiles_y = (H + tile_h - 1) // tile_h
        n_tiles_x = (W + tile_w - 1) // tile_w

        # 构建所有块的均衡化映射表
        lut = np.zeros((n_tiles_y, n_tiles_x, num_bins), dtype=np.float32)

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y0 = ty * tile_h
                y1 = min(y0 + tile_h, H)
                x0 = tx * tile_w
                x1 = min(x0 + tile_w, W)

                tile = L[y0:y1, x0:x1]
                tile_pixels = tile.size

                hist, _ = np.histogram(tile.ravel(), bins=num_bins, range=(0, 256))
                hist = hist.astype(np.float64)

                clip_threshold = self.clip_limit * (tile_pixels / num_bins)
                excess = np.maximum(hist - clip_threshold, 0.0)
                total_excess = excess.sum()

                hist = np.minimum(hist, clip_threshold)
                hist += total_excess / num_bins

                cdf = hist.cumsum()
                cdf_min = cdf[cdf > 0][0] if (cdf > 0).any() else 0.0
                if tile_pixels > cdf_min:
                    cdf_normalized = (cdf - cdf_min) / (tile_pixels - cdf_min) * 255.0
                else:
                    cdf_normalized = np.arange(num_bins, dtype=np.float64)

                lut[ty, tx] = cdf_normalized.astype(np.float32)

        # 向量化双线性插值
        # 构建每个像素对应的块坐标和权重
        y_coords = np.arange(H, dtype=np.float32)
        x_coords = np.arange(W, dtype=np.float32)

        ty_f = (y_coords + 0.5) / tile_h - 0.5  # (H,)
        tx_f = (x_coords + 0.5) / tile_w - 0.5  # (W,)

        ty0 = np.floor(ty_f).astype(np.int32)  # (H,)
        tx0 = np.floor(tx_f).astype(np.int32)  # (W,)

        wy1 = (ty_f - ty0).astype(np.float32)  # (H,)
        wx1 = (tx_f - tx0).astype(np.float32)  # (W,)
        wy0 = 1.0 - wy1                          # (H,)
        wx0 = 1.0 - wx1                          # (W,)

        ty0c = np.clip(ty0, 0, n_tiles_y - 1)   # (H,)
        ty1c = np.clip(ty0 + 1, 0, n_tiles_y - 1)
        tx0c = np.clip(tx0, 0, n_tiles_x - 1)   # (W,)
        tx1c = np.clip(tx0 + 1, 0, n_tiles_x - 1)

        # 对每个像素查表：lut[ty, tx, pixel_val]
        # L 的形状 (H, W)，需要映射到 lut
        pv = L.astype(np.int32)  # (H, W)，像素值作为 bin 索引

        # 展开查找四个邻居的映射值
        v00 = lut[ty0c[:, None], tx0c[None, :], pv]  # (H, W)
        v01 = lut[ty0c[:, None], tx1c[None, :], pv]
        v10 = lut[ty1c[:, None], tx0c[None, :], pv]
        v11 = lut[ty1c[:, None], tx1c[None, :], pv]

        # 双线性插值
        result = (
            wy0[:, None] * (wx0[None, :] * v00 + wx1[None, :] * v01)
            + wy1[:, None] * (wx0[None, :] * v10 + wx1[None, :] * v11)
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _multi_scale_detail_enhance(self, L: np.ndarray) -> np.ndarray:
        """
        多尺度细节增强

        在三个尺度（sigma = 1, 2, 4）上提取高频细节，
        加权叠加到原图，增强图像中的纹理和边缘细节。

        算法：
            detail_k = L - GaussianBlur(L, sigma=sigma_k)
            L_enhanced = L + detail_boost × (Σ detail_k) / 3

        Args:
            L: 输入亮度通道，uint8 或 float32，形状 (H, W)

        Returns:
            enhanced: 细节增强后的亮度，float32，形状 (H, W)
        """
        L_float = L.astype(np.float32)
        sigmas = [1, 2, 4]
        detail_sum = np.zeros_like(L_float)

        for sigma in sigmas:
            # 高斯模糊核大小：6*sigma + 1（奇数）
            ksize = 6 * sigma + 1
            blurred = cv2.GaussianBlur(L_float, (ksize, ksize), sigma)
            detail = L_float - blurred
            detail_sum += detail

        # 加权增强
        enhanced = L_float + self.detail_boost * detail_sum / len(sigmas)
        return enhanced

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        手写 CLAHE + 多尺度细节增强的单张图像 HDR 增强

        处理流程：
        1. BGR → LAB 色彩空间转换
        2. 提取 L 通道（uint8, [0, 255]）
        3. 手写 CLAHE 对 L 通道做自适应直方图均衡化
        4. 多尺度细节增强（sigma = 1, 2, 4）
        5. 将增强后的 L 放回 LAB，LAB → BGR

        Args:
            image: 输入 BGR 图像，uint8，形状 (H, W, 3)

        Returns:
            result: HDR 增强后的 BGR 图像，uint8，形状 (H, W, 3)
        """
        logger.debug(f"process() 开始: 输入形状={image.shape}, dtype={image.dtype}")

        # 步骤 1：BGR → LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 步骤 2：分离 L 通道
        L, a, b = cv2.split(lab)
        # L 为 uint8，范围 [0, 255]（OpenCV LAB 编码）

        logger.debug(f"L 通道: shape={L.shape}, min={L.min()}, max={L.max()}")

        # 步骤 3：手写 CLAHE（向量化版本）
        L_clahe = self._apply_clahe_manual_vectorized(L)
        logger.debug(
            f"CLAHE 后 L: min={L_clahe.min()}, max={L_clahe.max()}"
        )

        # 步骤 4：多尺度细节增强
        L_enhanced_float = self._multi_scale_detail_enhance(L_clahe)

        # 步骤 5：裁剪并转回 uint8
        L_enhanced = np.clip(L_enhanced_float, 0, 255).astype(np.uint8)
        logger.debug(
            f"细节增强后 L: min={L_enhanced.min()}, max={L_enhanced.max()}"
        )

        # 步骤 6：合并通道，LAB → BGR
        lab_enhanced = cv2.merge([L_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        logger.debug(f"process() 完成: 输出形状={result.shape}, dtype={result.dtype}")
        return result

    def process_opencv(self, image: np.ndarray) -> np.ndarray:
        """
        使用 OpenCV CLAHE + 多尺度细节增强的单张图像 HDR 增强

        处理流程：
        1. BGR → LAB 色彩空间转换
        2. 提取 L 通道（uint8）
        3. 调用 cv2.createCLAHE 对 L 通道做自适应直方图均衡化
        4. 多尺度细节增强（sigma = 1, 2, 4），与 process() 相同
        5. 将增强后的 L 放回 LAB，LAB → BGR

        Args:
            image: 输入 BGR 图像，uint8，形状 (H, W, 3)

        Returns:
            result: HDR 增强后的 BGR 图像，uint8，形状 (H, W, 3)
        """
        logger.debug(
            f"process_opencv() 开始: 输入形状={image.shape}, dtype={image.dtype}"
        )

        # 步骤 1：BGR → LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 步骤 2：分离 L 通道
        L, a, b = cv2.split(lab)

        # 步骤 3：OpenCV CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size),
        )
        L_clahe = clahe.apply(L)
        logger.debug(
            f"OpenCV CLAHE 后 L: min={L_clahe.min()}, max={L_clahe.max()}"
        )

        # 步骤 4：多尺度细节增强（与 process() 相同）
        L_enhanced_float = self._multi_scale_detail_enhance(L_clahe)

        # 步骤 5：裁剪并转回 uint8
        L_enhanced = np.clip(L_enhanced_float, 0, 255).astype(np.uint8)
        logger.debug(
            f"细节增强后 L: min={L_enhanced.min()}, max={L_enhanced.max()}"
        )

        # 步骤 6：合并通道，LAB → BGR
        lab_enhanced = cv2.merge([L_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        logger.debug(
            f"process_opencv() 完成: 输出形状={result.shape}, dtype={result.dtype}"
        )
        return result
