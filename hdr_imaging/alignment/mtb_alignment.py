"""
MTB 图像对齐模块 - Ward 2003 中值阈值位图算法
MTB Alignment Module based on Ward (2003) Median Threshold Bitmap

【算法原理】
MTB (Median Threshold Bitmap) 算法将图像转换为二值位图，以中值亮度为阈值，
对曝光变化具有天然的不敏感性，适用于多曝光 HDR 图像序列的对齐。

【核心数学公式】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) 中值阈值位图 (Median Threshold Bitmap):
   MTB(x,y) = 1  if I(x,y) > median(I)
            = 0  otherwise

   以中值亮度为阈值，亮于中值的像素标记为1，暗于中值的标记为0。
   由于中值亮度会随曝光等比例变化，该二值化对曝光变化不敏感。

2) 排除位图 (Exclusion Bitmap):
   EB(x,y) = 0  if |I(x,y) - median(I)| < threshold   （接近中值，不稳定）
           = 1  otherwise                               （远离中值，稳定）

   排除接近中值的"不稳定"像素，防止噪声干扰对齐计算。
   threshold 通常取 4（在0-255灰度范围内）。

3) 金字塔粗到细搜索 (Coarse-to-Fine Pyramid Search):
   - 构建图像金字塔：每层分辨率为上层的 1/2
   - 从最粗层（第 max_level 层）开始搜索
   - 每层搜索 [-1, 0, 1] × [-1, 0, 1] 共9个候选位移
   - 用 XOR 计算两幅 MTB 的差异，排除 EB=0 的像素：
       diff = (MTB1_shifted XOR MTB2) AND (EB1_shifted AND EB2)
       error = sum(diff)   （异或结果中1的个数，即不匹配像素数）
   - 选取 error 最小的位移作为当前层最优位移
   - 逐层细化，位移累积方式：
       shift_total = shift_coarser × 2 + shift_current
   - 最终位移在原始分辨率下即为各层位移之和（经2倍上采样展开）

【参考文献】
Greg Ward. "Fast, robust image registration for compositing high
dynamic range photographs from hand-held exposures." Journal of
Graphics Tools, 8(2):17-30, 2003.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class MTBAlignment:
    """
    MTB 图像对齐器 (Ward 2003)

    使用中值阈值位图 + 图像金字塔进行多曝光图像对齐。
    对曝光变化不敏感，计算速度快，适合手持拍摄的 HDR 序列对齐。
    """

    def __init__(self, max_level: int = 5, threshold: int = 4):
        """
        初始化 MTB 对齐器

        Args:
            max_level: 图像金字塔最大层数，控制搜索范围（层数越多，可校正的最大位移越大）
                       最大可校正位移约为 2^max_level 像素
            threshold: 排除位图容差，绝对亮度差小于此值的像素被排除
                       默认值 4 适合 0-255 灰度范围
        """
        self.max_level = max_level
        self.threshold = threshold
        logger.info(f"MTB 对齐器初始化: max_level={max_level}, threshold={threshold}")

    def _compute_mtb(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算中值阈值位图和排除位图

        【数学公式】
        设 m = median(I)
        MTB(x,y) = 1  if I(x,y) > m, else 0
        EB(x,y)  = 1  if |I(x,y) - m| >= threshold, else 0

        Args:
            gray: 单通道灰度图像 (H × W, uint8)

        Returns:
            mtb: 中值阈值位图 (H × W, uint8, 值为0或255)
            exclusion_bitmap: 排除位图 (H × W, uint8, 值为0或255)
        """
        median = int(np.median(gray))

        # 中值阈值位图：亮于中值 → 255，否则 → 0
        mtb = np.where(gray > median, np.uint8(255), np.uint8(0)).astype(np.uint8)

        # 排除位图：绝对亮度差 >= threshold → 255（稳定像素），否则 → 0（排除）
        diff = np.abs(gray.astype(np.int32) - median)
        exclusion_bitmap = np.where(diff >= self.threshold, np.uint8(255), np.uint8(0)).astype(np.uint8)

        return mtb, exclusion_bitmap

    def _compute_shift(
        self,
        mtb1: np.ndarray,
        eb1: np.ndarray,
        mtb2: np.ndarray,
        eb2: np.ndarray
    ) -> Tuple[int, int]:
        """
        在 [-1, 0, 1] × [-1, 0, 1] 范围内搜索最优平移量

        【搜索策略】
        对每个候选位移 (dx, dy)：
          1. 将 MTB1 平移 (dx, dy) 得到 MTB1_shifted
          2. 将 EB1  平移 (dx, dy) 得到 EB1_shifted
          3. 计算差异：diff = (MTB1_shifted XOR MTB2) AND (EB1_shifted AND EB2)
          4. 统计差异像素数：error = countNonZero(diff)
        选取 error 最小的 (dx, dy) 作为最优位移。

        Args:
            mtb1: 参考图像的 MTB
            eb1:  参考图像的排除位图
            mtb2: 待对齐图像的 MTB
            eb2:  待对齐图像的排除位图

        Returns:
            (dx, dy): 最优位移，表示将 mtb1 平移到 mtb2 的偏移
        """
        h, w = mtb1.shape
        best_error = float('inf')
        best_dx, best_dy = 0, 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # 构建平移变换矩阵，将 MTB1/EB1 平移 (dx, dy)
                M = np.float32([[1, 0, dx], [0, 1, dy]])

                # 使用仿射变换平移（边界用0填充）
                mtb1_shifted = cv2.warpAffine(
                    mtb1, M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                eb1_shifted = cv2.warpAffine(
                    eb1, M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # XOR 计算 MTB 差异，AND 排除不稳定像素
                diff = cv2.bitwise_and(
                    cv2.bitwise_xor(mtb1_shifted, mtb2),
                    cv2.bitwise_and(eb1_shifted, eb2)
                )

                error = cv2.countNonZero(diff)

                if error < best_error:
                    best_error = error
                    best_dx, best_dy = dx, dy

        return best_dx, best_dy

    def process(
        self,
        images: List[np.ndarray],
        reference_index: int = 0
    ) -> List[np.ndarray]:
        """
        使用 MTB 金字塔对齐所有图像到参考图像

        【金字塔粗到细流程】
        1. 构建 max_level 层图像金字塔（逐层下采样 2×）
        2. 从最粗层开始，在 [-1,0,1]×[-1,0,1] 搜索最优位移
        3. 逐层累积位移：shift_total = shift_coarser × 2 + shift_current
        4. 最终在原始分辨率上用 warpAffine 进行平移校正

        Args:
            images: 多曝光图像列表，每张图像为 BGR 格式 (H × W × 3, uint8)
            reference_index: 参考图像索引，默认为0（第一张）

        Returns:
            aligned_images: 对齐后的图像列表，参考图像原样返回

        Raises:
            ValueError: 若图像数量少于2张
        """
        if len(images) < 2:
            raise ValueError("At least 2 images required for alignment")

        logger.info(f"开始 MTB 对齐: {len(images)} 张图像，参考帧索引={reference_index}")

        ref_img = images[reference_index]
        h, w = ref_img.shape[:2]

        # 将参考图像转换为灰度
        if len(ref_img.shape) == 3:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_img.copy()

        aligned_images = []

        for i, img in enumerate(images):
            if i == reference_index:
                # 参考图像直接保留
                aligned_images.append(img.copy())
                logger.info(f"  图像 {i}: 参考图像，跳过对齐")
                continue

            # 将待对齐图像转换为灰度
            if len(img.shape) == 3:
                src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                src_gray = img.copy()

            # 构建图像金字塔（参考图与待对齐图）
            ref_pyramid = [ref_gray]
            src_pyramid = [src_gray]
            for level in range(1, self.max_level + 1):
                ref_pyramid.append(cv2.pyrDown(ref_pyramid[-1]))
                src_pyramid.append(cv2.pyrDown(src_pyramid[-1]))

            # 从最粗层开始累积位移
            total_dx, total_dy = 0, 0

            for level in range(self.max_level, -1, -1):
                ref_level = ref_pyramid[level]
                src_level = src_pyramid[level]

                # 计算当前层的 MTB 和排除位图
                mtb_ref, eb_ref = self._compute_mtb(ref_level)
                mtb_src, eb_src = self._compute_mtb(src_level)

                # 搜索当前层的最优位移（在累积位移基础上微调）
                # 先将当前累积位移施加到参考 MTB/EB，再搜索
                lvl_h, lvl_w = ref_level.shape
                M_accum = np.float32([[1, 0, total_dx], [0, 1, total_dy]])
                mtb_ref_shifted = cv2.warpAffine(
                    mtb_ref, M_accum, (lvl_w, lvl_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                eb_ref_shifted = cv2.warpAffine(
                    eb_ref, M_accum, (lvl_w, lvl_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # 搜索当前层的微调位移
                dx, dy = self._compute_shift(mtb_ref_shifted, eb_ref_shifted, mtb_src, eb_src)

                # 累积位移：上层位移 ×2（因降采样），加上当前层微调
                total_dx = total_dx + dx
                total_dy = total_dy + dy

                if level > 0:
                    # 向下一层（更细层）前乘以2
                    total_dx *= 2
                    total_dy *= 2

            logger.info(f"  图像 {i}: 最终位移 dx={total_dx}, dy={total_dy}")

            # 在原始分辨率上应用平移变换
            M_final = np.float32([[1, 0, total_dx], [0, 1, total_dy]])
            if len(img.shape) == 3:
                aligned = cv2.warpAffine(
                    img, M_final, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
            else:
                aligned = cv2.warpAffine(
                    img, M_final, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
            aligned_images.append(aligned)

        logger.info("MTB 对齐完成")
        return aligned_images

    def process_opencv(
        self,
        images: List[np.ndarray],
        reference_index: int = 0
    ) -> List[np.ndarray]:
        """
        使用 OpenCV 内置 AlignMTB 进行图像对齐

        调用 cv2.createAlignMTB() 创建对齐器，再调用 alignMTB.process() 执行对齐。
        OpenCV 的实现与本模块手动实现等效，可作为对比验证使用。

        Args:
            images: 多曝光图像列表 (BGR, uint8)
            reference_index: 参考图像索引

        Returns:
            aligned_images: 对齐后的图像列表

        Raises:
            ValueError: 若图像数量少于2张
        """
        if len(images) < 2:
            raise ValueError("At least 2 images required for alignment")

        logger.info(f"使用 OpenCV AlignMTB 对齐: {len(images)} 张图像，参考帧索引={reference_index}")

        align_mtb = cv2.createAlignMTB()
        aligned_images = list(images)  # 复制列表
        align_mtb.process(images, aligned_images)

        logger.info("OpenCV MTB 对齐完成")
        return aligned_images
