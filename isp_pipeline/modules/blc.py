"""
黑电平校正模块 (Black Level Correction, BLC)
=============================================

参考论文:
- "A Comprehensive Survey on Image Signal Processing" arXiv:2502.05995
- Infinite-ISP Pipeline: https://github.com/xx-isp/infinite-isp

原理说明:
    相机传感器在完全黑暗环境下，像素值不为零，存在固定偏置（暗电流噪声）。
    这个偏置值被称为"黑电平"。在进行后续 ISP 处理之前，必须先将其减去，
    使得真正的黑色像素值趋近于 0，保证后续算法处理的正确性。

    Bayer 图案中四个通道（R, Gr, Gb, B）各有独立的黑电平值，
    因为不同颜色滤镜下的传感器响应可能不同。

处理步骤:
    1. 按 Bayer 模式分离四个颜色通道
    2. 从各通道中减去对应的黑电平值
    3. 裁剪到 [0, white_level] 范围（归一化）
    4. 可选：将像素值归一化到 [0, 1] 浮点范围
"""

import numpy as np
from typing import Optional, Tuple


class BlackLevelCorrection:
    """
    黑电平校正类

    校正步骤：
        output = clip((raw_pixel - black_level) / (white_level - black_level), 0, 1)

    参数说明:
        black_level: 黑电平值，可以是标量（四通道共用）或长度为4的元组
                     顺序为 (R, Gr, Gb, B) 对应 RGGB Bayer 排列
        white_level: 传感器饱和值（白电平），通常为 2^bit_depth - 1
                     例如 12 位传感器为 4095
        bayer_pattern: Bayer 排列模式，支持 'RGGB', 'BGGR', 'GRBG', 'GBRG'
    """

    # Bayer 排列模式到四通道位置的映射
    # 每个元组表示 (R行偏移, R列偏移, Gr行偏移, Gr列偏移,
    #               Gb行偏移, Gb列偏移, B行偏移, B列偏移)
    BAYER_PATTERNS = {
        'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
        'BGGR': {'R': (1, 1), 'Gr': (1, 0), 'Gb': (0, 1), 'B': (0, 0)},
        'GRBG': {'R': (0, 1), 'Gr': (0, 0), 'Gb': (1, 1), 'B': (1, 0)},
        'GBRG': {'R': (1, 0), 'Gr': (1, 1), 'Gb': (0, 0), 'B': (0, 1)},
    }

    def __init__(
        self,
        black_level: float | Tuple[float, float, float, float] = 64,
        white_level: float = 4095,
        bayer_pattern: str = 'RGGB',
        normalize: bool = True,
    ):
        """
        初始化黑电平校正模块

        参数:
            black_level: 黑电平值。
                - 标量: 所有通道使用相同黑电平
                - 长度4的元组/列表: 分别对应 (R, Gr, Gb, B) 通道的黑电平
            white_level: 白电平（传感器最大值），12位传感器通常为 4095
            bayer_pattern: Bayer 排列，默认 'RGGB'
            normalize: 是否将输出归一化到 [0, 1]，默认 True
        """
        # 处理黑电平参数：标量扩展为四通道独立值
        if np.isscalar(black_level):
            self.black_levels = {
                'R': float(black_level),
                'Gr': float(black_level),
                'Gb': float(black_level),
                'B': float(black_level),
            }
        else:
            bl = list(black_level)
            assert len(bl) == 4, "黑电平元组长度必须为 4，顺序为 (R, Gr, Gb, B)"
            self.black_levels = {
                'R': float(bl[0]),
                'Gr': float(bl[1]),
                'Gb': float(bl[2]),
                'B': float(bl[3]),
            }

        self.white_level = float(white_level)
        self.bayer_pattern = bayer_pattern.upper()
        self.normalize = normalize

        assert self.bayer_pattern in self.BAYER_PATTERNS, \
            f"不支持的 Bayer 模式: {bayer_pattern}，支持: {list(self.BAYER_PATTERNS.keys())}"

    def process(self, raw: np.ndarray) -> np.ndarray:
        """
        对 RAW Bayer 图像执行黑电平校正

        算法步骤:
            1. 创建与输入相同大小的黑电平掩膜
            2. 根据 Bayer 排列，给对应位置的像素赋予各通道黑电平值
            3. 减去黑电平掩膜
            4. 除以有效动态范围 (white_level - min_black_level) 进行归一化

        参数:
            raw: 输入 RAW Bayer 图像，形状为 (H, W)，数值类型为整型或浮点型

        返回:
            校正后的图像，如果 normalize=True 则为 [0,1] 范围的 float32
        """
        # 转为浮点数以便精确计算
        corrected = raw.astype(np.float32)

        # 获取当前 Bayer 模式的通道位置信息
        pattern = self.BAYER_PATTERNS[self.bayer_pattern]

        # 构建黑电平掩膜：对不同位置的像素施加对应通道的黑电平
        bl_mask = np.zeros_like(corrected)
        for channel, (row_off, col_off) in pattern.items():
            # 对每个通道（R/Gr/Gb/B）在 2x2 像素块内的对应位置减去黑电平
            bl_mask[row_off::2, col_off::2] = self.black_levels[channel]

        # 减去黑电平
        corrected = corrected - bl_mask

        if self.normalize:
            # 归一化：除以有效动态范围（白电平 - 最小黑电平）
            # 确保图像值域在 [0, 1]
            effective_range = self.white_level - min(self.black_levels.values())
            corrected = corrected / effective_range

        # 截断到有效范围，消除负值（因暗电流导致的超黑区域）
        corrected = np.clip(corrected, 0.0, 1.0 if self.normalize else self.white_level)

        return corrected

    @staticmethod
    def estimate_black_level(dark_frame: np.ndarray, bayer_pattern: str = 'RGGB') -> dict:
        """
        从暗帧（遮光帧）自动估计各通道黑电平

        方法：使用遮光镜头下拍摄的"暗帧"图像，对四个通道分别取均值，
        得到各通道的黑电平估计值。

        参数:
            dark_frame: 遮光拍摄的 RAW 图像，形状 (H, W)
            bayer_pattern: Bayer 排列模式

        返回:
            字典格式的各通道黑电平 {'R': ..., 'Gr': ..., 'Gb': ..., 'B': ...}
        """
        pattern_map = BlackLevelCorrection.BAYER_PATTERNS[bayer_pattern.upper()]
        black_levels = {}

        for channel, (row_off, col_off) in pattern_map.items():
            # 提取该通道的所有像素并取均值
            channel_pixels = dark_frame[row_off::2, col_off::2]
            black_levels[channel] = float(np.mean(channel_pixels))

        return black_levels
