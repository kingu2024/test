"""
Gamma 校正模块 (Gamma Correction)
===================================

参考论文/资料:
- IEC 61966-2-1:1999 sRGB 标准（定义 sRGB Gamma 转换曲线）
- "A Comprehensive Survey on Image Signal Processing" arXiv:2502.05995
- Poynton, C.A. (2012) "Digital Video and HDTV Algorithms and Interfaces"
- OpenCV Gamma 处理参考实现

原理说明:
    人眼对亮度的感知是非线性的（近似对数曲线），
    对暗部变化更敏感，对亮部变化相对不敏感。

    相机传感器输出的是线性光强度值（线性 RGB）。
    如果直接将线性值存储为 8 位图像（0-255），
    会将大量编码空间浪费在人眼不敏感的高亮区域，
    而人眼敏感的暗部则因编码位数不足产生明显的条带伪影（Banding）。

    Gamma 编码（Display Encoding）：
        将线性光值压缩，使得暗部获得更多编码空间
        encoded = linear^(1/gamma)

    Gamma 解码（Display Decoding）：
        将 gamma 编码值恢复为线性光值
        linear = encoded^gamma

    标准 Gamma 值：
        - sRGB: 近似 2.2（实际为分段函数，详见 sRGB 标准）
        - Rec.709（HDTV）: 2.2
        - BT.2020（HDR）: 2.2
        - Adobe RGB: 2.2
        - Apple Display P3: 2.2

sRGB 标准 Gamma 函数（IEC 61966-2-1）:
    编码（线性 → sRGB）：
        if linear <= 0.0031308:
            srgb = 12.92 × linear
        else:
            srgb = 1.055 × linear^(1/2.4) − 0.055

    这个分段函数在低亮度区域用线性段，避免无穷大导数，
    整体近似 gamma=2.2 的幂函数。
"""

import numpy as np
from typing import Optional


class GammaCorrection:
    """
    Gamma 校正类

    支持以下模式：
    1. 'srgb':     标准 sRGB Gamma（IEC 61966-2-1 分段函数，近似 2.2）
    2. 'power':    简单幂函数 Gamma（output = input^(1/gamma)）
    3. 'rec709':   Rec.709 (BT.709) 分段 Gamma（HDTV 标准）
    4. 'lut':      使用查找表（LUT），通过预计算加速

    参数:
        mode: Gamma 模式
        gamma: 幂函数模式下的 Gamma 值（默认 2.2）
        lut_size: LUT 模式下查找表的大小（精度 vs 内存的折中）
        direction: 'encode'（线性→显示）或 'decode'（显示→线性）
    """

    def __init__(
        self,
        mode: str = 'srgb',
        gamma: float = 2.2,
        lut_size: int = 4096,
        direction: str = 'encode',
    ):
        """
        初始化 Gamma 校正模块

        参数:
            mode: Gamma 标准模式
                - 'srgb':   sRGB 分段 Gamma（最常用）
                - 'power':  简单幂函数，使用 gamma 参数指定值
                - 'rec709': BT.709 Gamma
                - 'lut':    预计算查找表（速度最快）
            gamma: 幂函数模式的 Gamma 值，通常为 2.2 或 2.4
            lut_size: LUT 查找表大小，影响精度
            direction: 转换方向
                - 'encode': 线性光 → Gamma 编码（ISP 后处理阶段）
                - 'decode': Gamma 编码 → 线性光（ISP 预处理阶段）
        """
        assert mode in ('srgb', 'power', 'rec709', 'lut'), \
            f"不支持的 Gamma 模式: {mode}"
        assert direction in ('encode', 'decode'), \
            f"方向必须为 'encode' 或 'decode'"

        self.mode = mode
        self.gamma = gamma
        self.lut_size = lut_size
        self.direction = direction

        # 预构建 LUT（如果使用 lut 模式）
        if mode == 'lut':
            self._lut = self._build_lut(lut_size)

    def _srgb_encode(self, linear: np.ndarray) -> np.ndarray:
        """
        sRGB Gamma 编码：线性光 → sRGB 显示值

        IEC 61966-2-1 标准分段公式：
            if linear <= 0.0031308:
                srgb = 12.92 × linear           （线性段，避免 0 处导数无穷大）
            else:
                srgb = 1.055 × linear^(1/2.4) − 0.055  （幂函数段）

        参数:
            linear: 线性光强度值，值域 [0, 1]

        返回:
            sRGB 编码值，值域 [0, 1]
        """
        linear = np.clip(linear, 0.0, 1.0)
        # 分段处理
        low_mask = linear <= 0.0031308
        srgb = np.where(
            low_mask,
            12.92 * linear,                            # 线性段
            1.055 * np.power(np.maximum(linear, 0.0031308), 1.0 / 2.4) - 0.055  # 幂函数段
        )
        return srgb.astype(np.float32)

    def _srgb_decode(self, srgb: np.ndarray) -> np.ndarray:
        """
        sRGB Gamma 解码：sRGB 显示值 → 线性光

        逆变换公式：
            if srgb <= 0.04045:
                linear = srgb / 12.92
            else:
                linear = ((srgb + 0.055) / 1.055) ^ 2.4

        参数:
            srgb: sRGB 编码值，值域 [0, 1]

        返回:
            线性光强度值，值域 [0, 1]
        """
        srgb = np.clip(srgb, 0.0, 1.0)
        low_mask = srgb <= 0.04045
        linear = np.where(
            low_mask,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        return linear.astype(np.float32)

    def _rec709_encode(self, linear: np.ndarray) -> np.ndarray:
        """
        BT.709 Gamma 编码（HDTV 标准）

        分段公式：
            if linear < 0.018:
                encoded = 4.5 × linear
            else:
                encoded = 1.099 × linear^0.45 - 0.099
        """
        linear = np.clip(linear, 0.0, 1.0)
        low_mask = linear < 0.018
        encoded = np.where(
            low_mask,
            4.5 * linear,
            1.099 * np.power(np.maximum(linear, 0.018), 0.45) - 0.099
        )
        return encoded.astype(np.float32)

    def _rec709_decode(self, encoded: np.ndarray) -> np.ndarray:
        """BT.709 Gamma 解码"""
        encoded = np.clip(encoded, 0.0, 1.0)
        low_mask = encoded < 0.081
        linear = np.where(
            low_mask,
            encoded / 4.5,
            np.power((encoded + 0.099) / 1.099, 1.0 / 0.45)
        )
        return linear.astype(np.float32)

    def _power_encode(self, linear: np.ndarray) -> np.ndarray:
        """
        简单幂函数 Gamma 编码

        公式：encoded = linear^(1/gamma)

        优点：计算简单，参数直观
        缺点：在 0 处导数无穷大（数值稳定性差），不符合显示标准
        """
        return np.power(np.clip(linear, 0.0, 1.0), 1.0 / self.gamma).astype(np.float32)

    def _power_decode(self, encoded: np.ndarray) -> np.ndarray:
        """简单幂函数 Gamma 解码：linear = encoded^gamma"""
        return np.power(np.clip(encoded, 0.0, 1.0), self.gamma).astype(np.float32)

    def _build_lut(self, size: int) -> np.ndarray:
        """
        预构建 Gamma 查找表（LUT）

        将 [0, 1] 范围等分为 size 个采样点，
        预先计算 Gamma 变换结果，推理时用索引查表代替逐像素计算。

        参数:
            size: LUT 大小（影响精度）

        返回:
            形状 (size,) 的 LUT 数组
        """
        x = np.linspace(0.0, 1.0, size, dtype=np.float32)

        # 根据当前模式建立对应 LUT
        if self.direction == 'encode':
            if self.mode == 'srgb':
                lut = self._srgb_encode(x)
            elif self.mode == 'rec709':
                lut = self._rec709_encode(x)
            else:
                lut = self._power_encode(x)
        else:
            if self.mode == 'srgb':
                lut = self._srgb_decode(x)
            elif self.mode == 'rec709':
                lut = self._rec709_decode(x)
            else:
                lut = self._power_decode(x)

        return lut

    def _apply_lut(self, img: np.ndarray) -> np.ndarray:
        """
        使用查找表应用 Gamma 变换（速度最快）

        将浮点输入 [0,1] 映射到 LUT 索引，查表获取结果。
        使用线性插值提高精度。

        参数:
            img: 输入图像，值域 [0, 1]

        返回:
            Gamma 变换后的图像
        """
        # 将 [0,1] 映射到 LUT 索引范围 [0, size-1]
        idx_float = np.clip(img, 0.0, 1.0) * (self.lut_size - 1)
        idx_low = np.floor(idx_float).astype(np.int32)
        idx_high = np.minimum(idx_low + 1, self.lut_size - 1)
        frac = (idx_float - idx_low).astype(np.float32)

        # 线性插值
        result = (1 - frac) * self._lut[idx_low] + frac * self._lut[idx_high]
        return result.astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        对图像执行 Gamma 校正

        参数:
            image: 输入图像（线性或 gamma 编码），值域 [0, 1]
                   形状可以是 (H, W) 或 (H, W, C)

        返回:
            Gamma 处理后的图像，值域 [0, 1]
        """
        img = image.astype(np.float32)

        if self.mode == 'lut':
            return self._apply_lut(img)

        if self.direction == 'encode':
            if self.mode == 'srgb':
                return self._srgb_encode(img)
            elif self.mode == 'rec709':
                return self._rec709_encode(img)
            else:
                return self._power_encode(img)
        else:  # decode
            if self.mode == 'srgb':
                return self._srgb_decode(img)
            elif self.mode == 'rec709':
                return self._rec709_decode(img)
            else:
                return self._power_decode(img)
