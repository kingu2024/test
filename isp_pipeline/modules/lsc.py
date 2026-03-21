"""
镜头阴影校正模块 (Lens Shading Correction, LSC)
================================================

参考资料:
- "Rockchip on-chip ISP pipeline – AWB, LSC, CCM"
  https://ridiqulous.com/rockchip-isp-pipeline/
- "Inside Image Processing Pipelines" Ignitarium
  https://ignitarium.com/inside-image-processing-pipelines/
- Infinite-ISP: https://github.com/xx-isp/infinite-isp

原理说明:
    镜头阴影（Lens Shading / Lens Vignetting）是由于以下原因导致的图像亮度不均匀：
    1. 光学渐晕（Optical Vignetting）：斜入射光线被光圈遮挡，边缘亮度降低
    2. 自然渐晕（Natural Vignetting）：符合 cos⁴(θ) 规律，θ 为入射角
    3. 机械渐晕（Mechanical Vignetting）：镜筒等机械结构遮挡光线

    典型表现：图像中心亮，四角暗（"隧道效应"）。
    不同颜色通道的渐晕程度可能不同，因此四个通道（R, Gr, Gb, B）需要独立校正。

校正方法：
    方法1 - 增益图校正（Gain Map）：
        对每个位置 (i,j)，预先标定增益值 G(i,j)，校正时乘以增益
        output(i,j) = input(i,j) × G(i,j)

    方法2 - 多项式模型校正（Polynomial Model）：
        用径向多项式拟合渐晕曲线：
        G(r) = 1 + k₂r² + k₄r⁴ + k₆r⁶
        其中 r 是到图像中心的归一化距离

    方法3 - 高斯模型校正（Gaussian Model）：
        用二维高斯函数拟合渐晕：
        G(x,y) = A × exp(-(x²/(2σx²) + y²/(2σy²)))
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Optional, Tuple


class LensShadingCorrection:
    """
    镜头阴影校正类

    支持三种校正模式：
    1. gain_map: 使用预标定的增益图直接校正（精度最高）
    2. polynomial: 使用径向多项式模型（参数少，适合标定数据不足时）
    3. gaussian: 使用二维高斯模型估计渐晕（无需标定，自动估计）

    参数:
        mode: 校正模式 ('gain_map', 'polynomial', 'gaussian')
        gain_maps: 预标定增益图字典，键为通道名 'R'/'Gr'/'Gb'/'B'
        poly_coeffs: 多项式系数 [k2, k4, k6]，适用于 polynomial 模式
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
        mode: str = 'gaussian',
        gain_maps: Optional[dict] = None,
        poly_coeffs: Optional[dict] = None,
        bayer_pattern: str = 'RGGB',
    ):
        """
        初始化镜头阴影校正模块

        参数:
            mode: 校正模式
                - 'gain_map':   使用精确增益图（需要相机标定数据）
                - 'polynomial': 使用多项式模型（需要 poly_coeffs 参数）
                - 'gaussian':   自动用高斯模型估计并校正（无标定数据时使用）
            gain_maps: dict，键为 'R'/'Gr'/'Gb'/'B'，值为对应的增益图 ndarray
                       增益图尺寸通常比原图小（稀疏采样），处理时会插值到原图尺寸
            poly_coeffs: dict，每个通道的多项式系数
                         格式: {'R': [k2, k4, k6], 'Gr': [...], 'Gb': [...], 'B': [...]}
            bayer_pattern: Bayer 排列模式
        """
        assert mode in ('gain_map', 'polynomial', 'gaussian'), \
            f"不支持的模式: {mode}"

        self.mode = mode
        self.gain_maps = gain_maps
        self.poly_coeffs = poly_coeffs
        self.bayer_pattern = bayer_pattern.upper()

        if mode == 'gain_map' and gain_maps is None:
            raise ValueError("gain_map 模式需要提供 gain_maps 参数")
        if mode == 'polynomial' and poly_coeffs is None:
            raise ValueError("polynomial 模式需要提供 poly_coeffs 参数")

    def _build_radial_gain_map(
        self,
        h: int,
        w: int,
        coeffs: list,
        center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        根据径向多项式系数构建增益图

        多项式模型：
            G(r) = 1 + k₂ × r² + k₄ × r⁴ + k₆ × r⁶
            其中 r = sqrt((x-cx)² + (y-cy)²) / r_max 为归一化半径

        参数:
            h, w: 图像通道高宽（原图的一半）
            coeffs: 多项式系数 [k2, k4, k6]
            center: 光学中心 (cy, cx)，默认为图像中心

        返回:
            增益图，形状 (h, w)
        """
        if center is None:
            cy, cx = h / 2.0, w / 2.0
        else:
            cy, cx = center

        # 构建坐标网格
        y_coords = np.arange(h, dtype=np.float32)
        x_coords = np.arange(w, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # 计算每个像素到光学中心的归一化距离
        r_max = np.sqrt(cx**2 + cy**2)  # 以角落到中心为最大半径
        r2 = ((xx - cx)**2 + (yy - cy)**2) / (r_max**2)
        r4 = r2 ** 2
        r6 = r2 ** 3

        k2, k4, k6 = coeffs[0], coeffs[1], coeffs[2]
        gain = 1.0 + k2 * r2 + k4 * r4 + k6 * r6

        # 增益值 >= 1（亮度只能被放大来补偿衰减）
        gain = np.maximum(gain, 1.0)

        return gain.astype(np.float32)

    def _estimate_gaussian_vignetting(self, channel: np.ndarray) -> np.ndarray:
        """
        自动估计并补偿高斯渐晕

        方法：
            1. 对通道图像进行强烈平滑，得到低频渐晕估计
            2. 用实际亮度 / 低频分量 得到增益图（等效去除空间不均匀性）

        这是一种无参数的自适应方法，适合没有相机标定数据时使用。
        但注意：对于大面积均匀区域外的复杂场景，效果有限。

        参数:
            channel: 单颜色通道图像，形状 (H/2, W/2)

        返回:
            增益图，形状与输入相同
        """
        from scipy.ndimage import gaussian_filter

        h, w = channel.shape

        # 用大窗口高斯平滑模拟镜头渐晕的低频分量
        # sigma 取图像对角线长度的 1/4，确保覆盖全图低频
        sigma = max(h, w) / 4.0
        vignette_estimate = gaussian_filter(channel, sigma=sigma, mode='reflect')

        # 计算增益图：理想均匀亮度 / 实际渐晕分布
        # 取中心区域均值作为参考"理想亮度"
        cy, cx = h // 2, w // 2
        ref_region = vignette_estimate[
            max(0, cy - h // 8): cy + h // 8,
            max(0, cx - w // 8): cx + w // 8,
        ]
        ref_value = np.mean(ref_region)

        # 增益 = 参考亮度 / 各位置估计亮度
        epsilon = 1e-6
        gain = ref_value / (vignette_estimate + epsilon)

        return gain.astype(np.float32)

    def _interpolate_gain_map(
        self,
        sparse_gain: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        将稀疏增益图双三次插值到目标尺寸

        相机厂商通常提供较小分辨率的增益图（如 32×24 个控制点），
        需要插值到与图像相同的尺寸。

        参数:
            sparse_gain: 稀疏增益图
            target_h, target_w: 目标尺寸

        返回:
            插值后的增益图，形状 (target_h, target_w)
        """
        src_h, src_w = sparse_gain.shape
        src_y = np.linspace(0, target_h - 1, src_h)
        src_x = np.linspace(0, target_w - 1, src_w)
        dst_y = np.arange(target_h)
        dst_x = np.arange(target_w)

        # 双三次样条插值（kx=3, ky=3）
        interp = RectBivariateSpline(src_y, src_x, sparse_gain, kx=3, ky=3)
        gain_full = interp(dst_y, dst_x)

        return gain_full.astype(np.float32)

    def process(self, raw: np.ndarray) -> np.ndarray:
        """
        对 RAW Bayer 图像执行镜头阴影校正

        处理流程:
            1. 对每个颜色通道（R, Gr, Gb, B）分别提取子图
            2. 根据选定模式计算增益图
            3. 将子图乘以增益图（亮度补偿）
            4. 写回 Bayer 图

        参数:
            raw: 坏点校正后的 Bayer 图，形状 (H, W)，值域 [0, 1]

        返回:
            镜头阴影校正后的 Bayer 图，形状 (H, W)，值域 [0, 1]
        """
        corrected = raw.copy()
        pattern = self.BAYER_PATTERNS[self.bayer_pattern]
        h, w = raw.shape

        for channel_name, (row_off, col_off) in pattern.items():
            channel = corrected[row_off::2, col_off::2]
            ch, cw = channel.shape

            if self.mode == 'gain_map':
                # 使用预标定增益图
                sparse_gain = self.gain_maps[channel_name]
                if sparse_gain.shape != (ch, cw):
                    # 需要插值到目标尺寸
                    gain = self._interpolate_gain_map(sparse_gain, ch, cw)
                else:
                    gain = sparse_gain.astype(np.float32)

            elif self.mode == 'polynomial':
                # 使用径向多项式模型
                coeffs = self.poly_coeffs.get(channel_name, [0.0, 0.0, 0.0])
                gain = self._build_radial_gain_map(ch, cw, coeffs)

            else:  # gaussian 模式
                # 自动估计高斯渐晕
                gain = self._estimate_gaussian_vignetting(channel)

            # 应用增益校正，并裁剪到 [0, 1]
            corrected[row_off::2, col_off::2] = np.clip(channel * gain, 0.0, 1.0)

        return corrected
