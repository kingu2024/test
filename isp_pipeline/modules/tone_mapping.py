"""
色调映射模块 (Tone Mapping)
============================

参考论文/资料:
- Reinhard et al. (2002) "Photographic Tone Reproduction for Digital Images"
  SIGGRAPH 2002. （经典摄影风格色调映射）
- Filmic Tone Mapping: John Hable (2010) "Filmic Tonemapping Operators"
  Uncharted 2 GDC Presentation. （电影级色调映射）
- ACES: Academy Color Encoding System (ACES) Tone Mapping
  （好莱坞/影视工业标准）
- Drago et al. (2003) "Adaptive Logarithmic Mapping for Displaying High Contrast Scenes"
  Eurographics 2003.
- "ISP Meets Deep Learning" ACM Computing Surveys 2024. https://dl.acm.org/doi/full/10.1145/3708516

背景说明:
    现实世界的亮度动态范围极大（高动态范围，HDR）：
    - 阴天室内：约 100:1
    - 晴天室外：约 100,000:1
    - 从阴影到直射阳光：约 10,000,000:1

    普通显示器的动态范围有限（约 100:1 ~ 1000:1）。
    色调映射（Tone Mapping）的目标是：将 HDR 内容压缩映射到显示器可表示的范围，
    同时尽可能保留局部对比度和视觉感知效果。

    不同色调映射曲线的特点：
    1. 线性截断：最简单，但高光区域直接截断，损失细节
    2. 全局操作符（Reinhard）：基于对数模型，整体亮度压缩，保留高光细节
    3. S 型曲线（Filmic）：类似胶片响应曲线，暗部对比高，亮部柔和渐变
    4. ACES：模拟胶片的色彩和动态响应，获得电影感
    5. 对数映射（Drago）：自适应对数压缩，对比度最大化
"""

import numpy as np
from typing import Optional


class ToneMapping:
    """
    色调映射类

    参数:
        method: 色调映射算法
            - 'reinhard':   Reinhard et al. 2002 摄影风格（经典）
            - 'reinhard_ext': 扩展 Reinhard（考虑白点）
            - 'filmic':     Filmic S 型曲线（John Hable 2010）
            - 'aces':       ACES 电影标准色调映射
            - 'drago':      Drago 自适应对数映射
            - 'gamma_only': 仅做 Gamma 压缩（最简单）
        key_value: Reinhard 中的曝光"键值"，控制整体亮度（默认 0.18）
        white_point: Reinhard_ext 的白点值（超过此值的区域映射为纯白）
        exposure: 曝光补偿系数（乘以输入），> 1 增亮，< 1 减暗
        saturation: 色调映射后的饱和度增强系数（1.0=不变）
    """

    def __init__(
        self,
        method: str = 'aces',
        key_value: float = 0.18,
        white_point: float = 1.0,
        exposure: float = 1.0,
        saturation: float = 1.0,
    ):
        valid_methods = ('reinhard', 'reinhard_ext', 'filmic', 'aces', 'drago', 'gamma_only')
        assert method in valid_methods, f"不支持的方法: {method}, 支持: {valid_methods}"

        self.method = method
        self.key_value = key_value
        self.white_point = white_point
        self.exposure = exposure
        self.saturation = saturation

    def _luminance(self, rgb: np.ndarray) -> np.ndarray:
        """
        计算图像的感知亮度（luminance）

        使用 BT.709 权重：Y = 0.2126R + 0.7152G + 0.0722B
        这些权重反映了人眼对不同颜色的敏感度（绿色最敏感）

        参数:
            rgb: 形状 (H, W, 3) 的 RGB 图像

        返回:
            亮度图，形状 (H, W)
        """
        return (0.2126 * rgb[..., 0] +
                0.7152 * rgb[..., 1] +
                0.0722 * rgb[..., 2])

    def _reinhard_operator(self, rgb: np.ndarray) -> np.ndarray:
        """
        Reinhard 全局色调映射算子（2002）

        参考: Reinhard et al. "Photographic Tone Reproduction for Digital Images" SIGGRAPH 2002.

        步骤：
            1. 计算场景对数平均亮度 Lw（反映场景整体感知亮度）：
               log_avg_L = exp(mean(log(L + δ)))
            2. 将场景亮度缩放到"键值"（key_value，通常为 0.18 的中灰）：
               L_scaled = L × (key_value / log_avg_L)
            3. 压缩到 [0,1]：
               L_mapped = L_scaled / (1 + L_scaled)  ← Reinhard 算子核心

        直觉：分母 (1 + L_scaled) 对亮部的压缩更强，暗部接近线性。
        高光不会硬截断，而是柔和地趋向最大值。

        参数:
            rgb: 线性 HDR RGB 图像，形状 (H, W, 3)

        返回:
            色调映射后的 LDR RGB 图像，值域 [0, 1]
        """
        # 计算场景亮度
        L = self._luminance(rgb)

        # 对数平均亮度（排除黑色像素，加 δ 防止 log(0)）
        delta = 1e-6
        log_avg_L = np.exp(np.mean(np.log(L + delta)))

        # 缩放亮度到键值
        L_scaled = (self.key_value / (log_avg_L + delta)) * L

        # Reinhard 压缩算子
        L_mapped = L_scaled / (1.0 + L_scaled)

        # 保持色调：缩放 RGB 使其亮度变为 L_mapped
        # 避免除零
        L_safe = np.maximum(L, delta)[..., np.newaxis]
        rgb_mapped = rgb * (L_mapped[..., np.newaxis] / L_safe)

        return np.clip(rgb_mapped, 0.0, 1.0).astype(np.float32)

    def _reinhard_ext_operator(self, rgb: np.ndarray) -> np.ndarray:
        """
        扩展 Reinhard 算子（考虑白点）

        改进公式（Reinhard 2002 论文第 4 节）：
            L_mapped = L_scaled × (1 + L_scaled / L_white²) / (1 + L_scaled)

        加入白点 L_white：当 L_scaled = L_white 时，输出接近 1（纯白）。
        使得高亮区域的压缩可控，防止整体过暗。

        参数:
            rgb: 线性 HDR RGB 图像，形状 (H, W, 3)

        返回:
            色调映射后的图像
        """
        delta = 1e-6
        L = self._luminance(rgb)
        log_avg_L = np.exp(np.mean(np.log(L + delta)))
        L_scaled = (self.key_value / (log_avg_L + delta)) * L

        L_white_sq = self.white_point ** 2
        L_mapped = L_scaled * (1.0 + L_scaled / L_white_sq) / (1.0 + L_scaled)

        L_safe = np.maximum(L, delta)[..., np.newaxis]
        rgb_mapped = rgb * (L_mapped[..., np.newaxis] / L_safe)

        return np.clip(rgb_mapped, 0.0, 1.0).astype(np.float32)

    def _filmic_operator(self, rgb: np.ndarray) -> np.ndarray:
        """
        Filmic 色调映射（John Hable / Uncharted 2 S 型曲线）

        参考: John Hable (2010) "Filmic Tonemapping Operators" GDC Presentation.
        https://www.gdcvault.com/play/1012351/Uncharted-2-HDR

        公式（Hable 的 Uncharted 2 近似）：
            f(x) = ((x(Ax+CB)+DE) / (x(Ax+B)+DF)) - E/F

        其中参数 A-F 为经过艺术调整的常数：
            A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30

        S 型曲线特点：
        - 暗部（低输入）：高对比度，还原阴影细节
        - 中间调：接近线性
        - 高光（高输入）：柔和压缩，模拟胶片高光宽容度

        参数:
            rgb: 线性 HDR RGB 图像，形状 (H, W, 3)

        返回:
            Filmic 风格色调映射后的 RGB 图像，值域 [0, 1]
        """
        # Uncharted 2 色调映射参数（John Hable 原版）
        A = 0.15  # 肩部强度（Shoulder Strength）
        B = 0.50  # 线性强度（Linear Strength）
        C = 0.10  # 线性角度（Linear Angle）
        D = 0.20  # 脚趾强度（Toe Strength）
        E = 0.02  # 脚趾数值（Toe Numerator）
        F = 0.30  # 脚趾分母（Toe Denominator）
        W = 11.2  # 线性白点预缩放值

        def uncharted2_tonemap(x):
            """Uncharted 2 S 型曲线核心函数"""
            return ((x * (A * x + C * B) + D * E) /
                    (x * (A * x + B) + D * F)) - E / F

        # 应用曝光补偿
        img = rgb * self.exposure * 2.0  # Hable 建议曝光偏移 2 stops

        # 映射并归一化（除以白点映射值，确保纯白归一）
        curr = uncharted2_tonemap(img)
        white_scale = 1.0 / uncharted2_tonemap(np.array([W], dtype=np.float32))

        mapped = curr * white_scale

        return np.clip(mapped, 0.0, 1.0).astype(np.float32)

    def _aces_operator(self, rgb: np.ndarray) -> np.ndarray:
        """
        ACES 色调映射（Academy Color Encoding System）

        参考: ACES Reference Implementation, Academy of Motion Picture Arts and Sciences.
        实际应用中被 Naughty Dog, Epic Games (Unreal Engine) 等采用。

        ACES 包含两个步骤：
        1. RRT (Reference Rendering Transform)：从输入场景线性光到 ACES 中间格式
        2. ODT (Output Device Transform)：从 ACES 中间格式到输出设备

        此处使用 Knarkowicz (2016) 的近似公式，速度快且效果接近完整 ACES：
            ACES(x) = (x(ax + b)) / (x(cx + d) + e)
            其中 a=2.51, b=0.03, c=2.43, d=0.59, e=0.14

        特点：
        - 色彩鲜艳，高光保留好
        - 对比度自然，阴影细节丰富
        - 广泛应用于游戏和影视渲染

        参数:
            rgb: 线性 HDR RGB 图像，形状 (H, W, 3)

        返回:
            ACES 色调映射后的图像，值域 [0, 1]
        """
        # ACES 输入变换矩阵（sRGB/Rec.709 → ACES AP0）
        # 简化版：直接在 sRGB 空间使用 Knarkowicz 近似
        img = rgb * self.exposure

        # Knarkowicz ACES 近似（逐像素，无需色彩空间转换）
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        mapped = (img * (a * img + b)) / (img * (c * img + d) + e)

        return np.clip(mapped, 0.0, 1.0).astype(np.float32)

    def _drago_operator(self, rgb: np.ndarray) -> np.ndarray:
        """
        Drago 自适应对数色调映射（2003）

        参考: Drago et al. "Adaptive Logarithmic Mapping for Displaying High Contrast Scenes"
              Eurographics 2003.

        算法思想：
            受 Weber-Fechner 感知定律启发（人眼对亮度感知约为对数响应）：
            L_out = (Ld_max/0.94) × log(L_w+1) / log(L_max+1) ×
                    log(2 + 8 × (L_w/L_max)^log_b(0.5/b))

            其中参数 b ∈ [0.6, 0.9] 控制对比度压缩强度。

        简化版实现：
            使用对数映射 + 自适应归一化

        参数:
            rgb: 线性 HDR RGB 图像，形状 (H, W, 3)

        返回:
            Drago 色调映射后的图像
        """
        b = 0.85   # 偏置参数，控制压缩强度（0.6~0.9，越大压缩越强）
        L_max = np.max(self._luminance(rgb)) + 1e-6

        L = self._luminance(rgb)
        L_normalized = L / L_max  # 归一化亮度

        # 自适应对数压缩
        log_b = np.log(b) / np.log(0.5)  # 改变对数底为 b
        L_log = np.log(L + 1) / np.log(L_max + 1)
        adjust = np.log(2 + 8 * np.power(np.maximum(L_normalized, 1e-8), log_b))
        L_mapped = L_log * adjust / np.log(10)  # 转为 log₁₀

        # 归一化到 [0, 1]
        L_mapped = (L_mapped - L_mapped.min()) / (L_mapped.max() - L_mapped.min() + 1e-8)

        # 保持色调
        delta = 1e-6
        L_safe = np.maximum(L, delta)[..., np.newaxis]
        rgb_mapped = rgb * (L_mapped[..., np.newaxis] / L_safe)

        return np.clip(rgb_mapped, 0.0, 1.0).astype(np.float32)

    def _adjust_saturation(self, rgb: np.ndarray, saturation: float) -> np.ndarray:
        """
        调整图像饱和度

        方法：在 HSL 空间中调整饱和度，通过 RGB ↔ 灰度 插值实现：
            output = lerp(gray, rgb, saturation)
            其中 gray 为图像亮度灰度版本

        参数:
            rgb: RGB 图像，(H, W, 3)
            saturation: 饱和度系数（1.0=不变，0=灰度，>1 饱和度增强）

        返回:
            调整饱和度后的图像
        """
        L = self._luminance(rgb)[..., np.newaxis]  # (H, W, 1)
        # 线性插值
        result = L + saturation * (rgb - L)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def process(self, rgb: np.ndarray) -> np.ndarray:
        """
        对线性 RGB 图像执行色调映射

        注意：色调映射应在 Gamma 校正之前（在线性光空间）执行！

        参数:
            rgb: 线性 HDR RGB 图像，形状 (H, W, 3)，值域 [0, ∞)
                 值域可以超过 1（HDR 内容），色调映射后映射到 [0, 1]

        返回:
            LDR RGB 图像，形状 (H, W, 3)，值域 [0, 1]
        """
        if self.method == 'reinhard':
            result = self._reinhard_operator(rgb)
        elif self.method == 'reinhard_ext':
            result = self._reinhard_ext_operator(rgb)
        elif self.method == 'filmic':
            result = self._filmic_operator(rgb)
        elif self.method == 'aces':
            result = self._aces_operator(rgb)
        elif self.method == 'drago':
            result = self._drago_operator(rgb)
        else:  # gamma_only
            result = np.clip(rgb * self.exposure, 0.0, 1.0).astype(np.float32)

        # 可选的饱和度调整（后处理）
        if self.saturation != 1.0:
            result = self._adjust_saturation(result, self.saturation)

        return result
