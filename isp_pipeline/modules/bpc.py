"""
坏点校正模块 (Bad Pixel Correction, BPC)
=========================================

参考论文/资料:
- "Inside Image Processing Pipelines" Ignitarium Technical Blog
  https://ignitarium.com/inside-image-processing-pipelines/
- Infinite-ISP Pipeline: https://github.com/xx-isp/infinite-isp

原理说明:
    相机传感器上由于制造缺陷、宇宙射线辐射或老化，会产生"坏点"（Bad Pixels）：
    - 热像素 (Hot Pixels): 始终过亮的像素，输出值远大于周围正常像素
    - 死像素 (Dead Pixels): 始终输出固定低值（通常为0）
    - 粘滞像素 (Stuck Pixels): 对光照变化不响应的像素

    坏点检测算法：
        通过比较像素与其邻域像素的差值来检测坏点。
        若一个像素与邻域中值的差值超过阈值，则判定为坏点。

    坏点校正算法：
        用该像素的同色通道邻域中值来替换坏点像素值。
        在 Bayer 域中操作时，只使用相同颜色通道的邻居（间隔2步）。

处理流程:
    1. 对 Bayer 图每个通道进行坏点检测
    2. 将检测到的坏点用邻域中值替换
"""

import numpy as np
from scipy import ndimage
from typing import Optional


class BadPixelCorrection:
    """
    坏点校正类

    支持两种模式:
    1. 动态检测模式：自动从图像中检测并修复坏点（基于阈值）
    2. 坏点地图模式：使用预先标定的坏点坐标图进行修复

    参数:
        threshold: 坏点检测阈值（相对于邻域中值的差异比例，范围 0~1）
        correction_method: 校正方法，'median'（中值）或 'mean'（均值）
        bayer_pattern: Bayer 排列模式
    """

    BAYER_PATTERNS = {
        'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
        'BGGR': {'R': (1, 1), 'Gr': (1, 0), 'Gb': (0, 1), 'B': (0, 0)},
        'GRBG': {'R': (0, 1), 'Gr': (0, 0), 'Gb': (1, 1), 'B': (1, 0)},
        'GBRG': {'R': (1, 0), 'Gr': (1, 1), 'Gb': (0, 0), 'B': (0, 1)},
    }

    def __init__(
        self,
        threshold: float = 0.2,
        correction_method: str = 'median',
        bayer_pattern: str = 'RGGB',
        bad_pixel_map: Optional[np.ndarray] = None,
    ):
        """
        初始化坏点校正模块

        参数:
            threshold: 坏点检测阈值。像素与邻域中值差异超过此比例则被判为坏点
                       建议范围 0.1 ~ 0.3。过小会误判正常边缘，过大会漏判坏点
            correction_method: 校正方法
                - 'median': 使用邻域中值替换（推荐，对边缘更鲁棒）
                - 'mean':   使用邻域均值替换
            bayer_pattern: Bayer 排列模式
            bad_pixel_map: 预标定坏点地图，形状与 RAW 图相同，坏点位置为 True
                           如果提供，则跳过自动检测阶段
        """
        self.threshold = threshold
        self.correction_method = correction_method
        self.bayer_pattern = bayer_pattern.upper()
        self.bad_pixel_map = bad_pixel_map

        assert correction_method in ('median', 'mean'), \
            f"不支持的校正方法: {correction_method}"

    def _get_channel_neighbors_median(self, channel: np.ndarray) -> np.ndarray:
        """
        计算单个颜色通道每个像素的邻域中值

        在 Bayer 图中，同色通道相邻像素间距为 2。
        提取后对应的单通道子图（通道尺寸为 H/2 × W/2），
        使用 3×3 滑动窗口中值滤波。

        参数:
            channel: 形状 (H/2, W/2) 的单通道图像（已从 Bayer 图提取）

        返回:
            邻域中值图，形状与 channel 相同
        """
        # footprint=True 表示使用 3x3 方形邻域（不包括中心像素）
        # 通过先做中值滤波再替换，避免坏点自身参与计算
        return ndimage.median_filter(channel, size=3, mode='mirror')

    def _get_channel_neighbors_mean(self, channel: np.ndarray) -> np.ndarray:
        """
        计算单个颜色通道每个像素的邻域均值

        参数:
            channel: 形状 (H/2, W/2) 的单通道图像

        返回:
            邻域均值图
        """
        return ndimage.uniform_filter(channel, size=3, mode='mirror')

    def _detect_bad_pixels(self, channel: np.ndarray) -> np.ndarray:
        """
        检测单个通道中的坏点

        坏点判定标准：
            |pixel - median(neighbors)| > threshold * median(neighbors)

        即：若像素值与邻域中值的相对差异超过阈值，则判为坏点。
        这种相对阈值方式对不同亮度区域都有较好的适应性。

        参数:
            channel: 单颜色通道图像

        返回:
            布尔型坏点掩膜，True 表示该位置为坏点
        """
        # 计算邻域中值作为参考值
        neighbor_median = self._get_channel_neighbors_median(channel)

        # 计算相对差值：|pixel - neighbor_median| / (neighbor_median + epsilon)
        # epsilon 避免除零（黑色区域中值为0时）
        epsilon = 1e-6
        relative_diff = np.abs(channel - neighbor_median) / (neighbor_median + epsilon)

        # 超过阈值的像素判为坏点
        bad_mask = relative_diff > self.threshold

        return bad_mask

    def process(self, raw: np.ndarray) -> np.ndarray:
        """
        对 RAW Bayer 图像执行坏点检测与校正

        处理流程:
            1. 对 Bayer 图的每个颜色通道（R, Gr, Gb, B）分别提取子图
            2. 在子图上检测坏点（或使用预设坏点地图）
            3. 用邻域中值/均值替换坏点像素
            4. 将校正后的子图写回 Bayer 图对应位置

        参数:
            raw: 黑电平校正后的 Bayer 图，形状 (H, W)，值域 [0, 1]

        返回:
            坏点校正后的 Bayer 图，形状 (H, W)，值域 [0, 1]
        """
        corrected = raw.copy()
        pattern = self.BAYER_PATTERNS[self.bayer_pattern]

        for channel_name, (row_off, col_off) in pattern.items():
            # 提取当前颜色通道的子图（步长为2，对应 Bayer 中同色像素间距）
            channel = corrected[row_off::2, col_off::2]

            if self.bad_pixel_map is not None:
                # 使用预标定坏点地图
                bad_mask = self.bad_pixel_map[row_off::2, col_off::2]
            else:
                # 动态检测坏点
                bad_mask = self._detect_bad_pixels(channel)

            if np.any(bad_mask):
                # 计算邻域替换值
                if self.correction_method == 'median':
                    replacement = self._get_channel_neighbors_median(channel)
                else:
                    replacement = self._get_channel_neighbors_mean(channel)

                # 将坏点像素替换为邻域值
                channel_corrected = channel.copy()
                channel_corrected[bad_mask] = replacement[bad_mask]

                # 写回 Bayer 图
                corrected[row_off::2, col_off::2] = channel_corrected

        return corrected
