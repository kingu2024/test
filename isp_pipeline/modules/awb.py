"""
自动白平衡模块 (Auto White Balance, AWB)
=========================================

参考论文/资料:
- "ISP Meets Deep Learning: A Survey on Deep Learning Methods for ISP"
  ACM Computing Surveys 2024. https://dl.acm.org/doi/full/10.1145/3708516
- "Color constancy" — Computational color science
- Infinite-ISP AWB 实现: https://github.com/xx-isp/infinite-isp

原理说明:
    白平衡的目标是消除场景光源颜色对图像的影响，使得在不同光照下
    同一物体的颜色保持一致（颜色恒常性，Color Constancy）。

    白平衡本质上是对 R、G、B 三个通道分别乘以增益系数 (Wr, Wg, Wb)，
    使得白色物体在最终图像中呈现中性灰色。

    本模块实现以下三种经典算法：

    1. 灰色世界算法 (Gray World Algorithm) [von Kries, 1902]
       假设：场景中所有颜色的平均值为中性灰
       增益：G_r = mean(G_channel) / mean(R_channel)
              G_g = 1.0（以 G 通道为参考）
              G_b = mean(G_channel) / mean(B_channel)

    2. 白色块算法 (White Patch Algorithm / Retinex)
       假设：场景中存在纯白物体，其反射了最多光线
       增益：G_r = max(G_channel) / max(R_channel)
              G_g = 1.0
              G_b = max(G_channel) / max(B_channel)

    3. 完美反射算法 (Perfect Reflector，改进的白色块)
       取亮度最高的前 p% 像素的平均值，比纯最大值更鲁棒

    4. 灰色边缘算法 (Gray Edge Algorithm) [van de Weijer et al., JOSA 2007]
       假设：图像梯度的均值接近中性灰（考虑了边缘）
       比灰色世界算法更适合场景颜色分布不均匀的情况
"""

import numpy as np
from typing import Optional, Tuple


class AutoWhiteBalance:
    """
    自动白平衡类

    在 Bayer 域对四个通道（R, Gr, Gb, B）施加增益，
    或在 RGB 图像上对三个通道施加增益。

    参数:
        method: AWB 算法
            - 'gray_world':       灰色世界算法（最常用）
            - 'white_patch':      白色块算法
            - 'perfect_reflector': 完美反射算法（更鲁棒的白色块）
            - 'gray_edge':        灰色边缘算法
            - 'manual':           手动指定增益
        percentile: 完美反射算法中取最亮区域的百分位，默认 95（前5%的亮像素）
        manual_gains: 手动增益，字典格式 {'R': float, 'G': float, 'B': float}
        bayer_pattern: Bayer 排列模式（在 Bayer 域操作时使用）
        clip_highlights: 是否裁剪高光溢出，防止白平衡后出现过曝
    """

    BAYER_PATTERNS = {
        'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
        'BGGR': {'R': (1, 1), 'Gr': (1, 0), 'Gb': (0, 1), 'B': (0, 0)},
        'GRBG': {'R': (0, 1), 'Gr': (0, 0), 'Gb': (1, 1), 'B': (1, 0)},
        'GBRG': {'R': (1, 0), 'Gr': (1, 1), 'Gb': (0, 0), 'B': (0, 1)},
    }

    def __init__(
        self,
        method: str = 'gray_world',
        percentile: float = 95.0,
        manual_gains: Optional[dict] = None,
        bayer_pattern: str = 'RGGB',
        clip_highlights: bool = True,
    ):
        """
        初始化自动白平衡模块

        参数:
            method: 白平衡算法名称
            percentile: 完美反射算法参数，取最亮 (100-percentile)% 的像素
            manual_gains: 手动增益字典，method='manual' 时必须提供
            bayer_pattern: Bayer 排列（Bayer 域操作时使用）
            clip_highlights: 是否裁剪 >1.0 的溢出高光
        """
        valid_methods = ('gray_world', 'white_patch', 'perfect_reflector', 'gray_edge', 'manual')
        assert method in valid_methods, f"不支持的方法: {method}, 支持: {valid_methods}"

        self.method = method
        self.percentile = percentile
        self.manual_gains = manual_gains
        self.bayer_pattern = bayer_pattern.upper()
        self.clip_highlights = clip_highlights

        if method == 'manual' and manual_gains is None:
            raise ValueError("method='manual' 时必须提供 manual_gains 参数")

    def _gray_world_gains(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
        """
        灰色世界算法计算白平衡增益

        理论依据：假设自然场景中各种颜色均匀分布，
        所有像素平均后应为中性灰（R_mean = G_mean = B_mean）。
        通过调整增益使三通道均值相等来消除色偏。

        公式：
            Wr = mean(G) / mean(R)
            Wg = 1.0
            Wb = mean(G) / mean(B)

        参数:
            r, g, b: 三个通道的像素数组

        返回:
            (gain_r, gain_g, gain_b) 增益元组
        """
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        epsilon = 1e-6
        gain_r = mean_g / (mean_r + epsilon)
        gain_g = 1.0
        gain_b = mean_g / (mean_b + epsilon)

        return gain_r, gain_g, gain_b

    def _white_patch_gains(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
        """
        白色块算法（最大值法）计算白平衡增益

        理论依据：场景中亮度最高的点对应纯白色，应将其调整为中性白。
        选取各通道最大值，通过增益让所有通道最大值相等。

        公式：
            Wr = max(G) / max(R)
            Wg = 1.0
            Wb = max(G) / max(B)

        局限性：对图像中的高光噪声、镜面反射敏感，鲁棒性较差。
        """
        max_r = np.max(r)
        max_g = np.max(g)
        max_b = np.max(b)

        epsilon = 1e-6
        gain_r = max_g / (max_r + epsilon)
        gain_g = 1.0
        gain_b = max_g / (max_b + epsilon)

        return gain_r, gain_g, gain_b

    def _perfect_reflector_gains(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
        """
        完美反射算法计算白平衡增益

        改进的白色块算法：不取最大值，而是取前 p% 亮像素的均值，
        对高光噪声和孤立亮点更具鲁棒性。

        步骤：
            1. 计算每个像素的亮度（简化为 R+G+B 总和）
            2. 找出亮度在 percentile 百分位以上的像素集合
            3. 取该集合中各通道的均值作为"白色参考"
            4. 调整增益使白色参考变为中性白

        参数:
            r, g, b: 三个通道的像素数组
        """
        # 计算每像素亮度
        brightness = r.ravel() + g.ravel() + b.ravel()

        # 找出亮度超过 percentile 百分位的像素索引
        threshold = np.percentile(brightness, self.percentile)
        bright_mask = brightness >= threshold

        # 提取高亮区域各通道像素
        bright_r = r.ravel()[bright_mask]
        bright_g = g.ravel()[bright_mask]
        bright_b = b.ravel()[bright_mask]

        epsilon = 1e-6
        ref_r = np.mean(bright_r)
        ref_g = np.mean(bright_g)
        ref_b = np.mean(bright_b)

        gain_r = ref_g / (ref_r + epsilon)
        gain_g = 1.0
        gain_b = ref_g / (ref_b + epsilon)

        return gain_r, gain_g, gain_b

    def _gray_edge_gains(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
        """
        灰色边缘算法计算白平衡增益

        参考: van de Weijer et al., "Edge-based Color Constancy", JOSA 2007

        原理：改进版灰色世界，利用图像梯度（边缘信息）而非像素均值。
        假设：图像梯度的均值为零（颜色变化平均下来为中性）。

        步骤：
            1. 计算三通道的 Sobel 梯度幅值
            2. 以梯度幅值均值作为"照明色"估计
            3. 计算增益消除色偏

        优势：对场景颜色分布不均匀（如蓝天、绿地等大块纯色区域）更鲁棒。
        """
        from scipy.ndimage import sobel

        # 确保输入是 2D 数组（2D 图的单通道）
        def gradient_magnitude(channel):
            """计算 Sobel 梯度幅值"""
            # 需要 2D 输入
            if channel.ndim == 1:
                n = int(np.sqrt(len(channel)))
                channel = channel.reshape(n, n) if n * n == len(channel) else channel.reshape(-1, 1)
            gx = sobel(channel, axis=1, mode='reflect')
            gy = sobel(channel, axis=0, mode='reflect')
            return np.sqrt(gx**2 + gy**2)

        # 如果是扁平数组，先重塑（AWB 输入是子图）
        grad_r = np.mean(np.abs(gradient_magnitude(r)))
        grad_g = np.mean(np.abs(gradient_magnitude(g)))
        grad_b = np.mean(np.abs(gradient_magnitude(b)))

        epsilon = 1e-6
        gain_r = grad_g / (grad_r + epsilon)
        gain_g = 1.0
        gain_b = grad_g / (grad_b + epsilon)

        return gain_r, gain_g, gain_b

    def process_bayer(self, raw: np.ndarray) -> np.ndarray:
        """
        在 Bayer 域执行白平衡增益校正

        在 Bayer 域操作的优点：
        - 避免去马赛克前的颜色插值误差
        - 直接对原始传感器数据进行校正，精度更高

        处理流程:
            1. 提取 Bayer 图中 R、G（Gr+Gb 平均）、B 三通道子图
            2. 用选定算法估计增益
            3. 对各颜色通道独立施加增益
            4. 裁剪到 [0, 1] 范围

        参数:
            raw: 镜头阴影校正后的 Bayer 图，形状 (H, W)，值域 [0, 1]

        返回:
            白平衡校正后的 Bayer 图，形状 (H, W)，值域 [0, 1]
        """
        result = raw.copy()
        pattern = self.BAYER_PATTERNS[self.bayer_pattern]

        # 提取四个通道
        r  = raw[pattern['R'][0]::2,  pattern['R'][1]::2]
        gr = raw[pattern['Gr'][0]::2, pattern['Gr'][1]::2]
        gb = raw[pattern['Gb'][0]::2, pattern['Gb'][1]::2]
        b  = raw[pattern['B'][0]::2,  pattern['B'][1]::2]

        # G 通道使用 Gr 和 Gb 的平均值来估计增益
        g = (gr + gb) / 2.0

        # 根据算法计算增益
        gains = self._compute_gains(r, g, b)
        gain_r, gain_g, gain_b = gains

        # 应用增益到 Bayer 图各位置
        result[pattern['R'][0]::2,  pattern['R'][1]::2]  = r  * gain_r
        result[pattern['Gr'][0]::2, pattern['Gr'][1]::2] = gr * gain_g
        result[pattern['Gb'][0]::2, pattern['Gb'][1]::2] = gb * gain_g
        result[pattern['B'][0]::2,  pattern['B'][1]::2]  = b  * gain_b

        if self.clip_highlights:
            result = np.clip(result, 0.0, 1.0)

        return result

    def process_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """
        在 RGB 域执行白平衡增益校正（去马赛克后使用）

        参数:
            rgb: RGB 图像，形状 (H, W, 3)，值域 [0, 1]
                 通道顺序: [..., 0]=R, [..., 1]=G, [..., 2]=B

        返回:
            白平衡校正后的 RGB 图像
        """
        result = rgb.copy().astype(np.float32)
        r, g, b = result[..., 0], result[..., 1], result[..., 2]

        gain_r, gain_g, gain_b = self._compute_gains(r, g, b)

        result[..., 0] = r * gain_r
        result[..., 1] = g * gain_g
        result[..., 2] = b * gain_b

        if self.clip_highlights:
            result = np.clip(result, 0.0, 1.0)

        return result

    def _compute_gains(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        根据选定算法计算三通道白平衡增益

        返回:
            (gain_r, gain_g, gain_b) 增益值元组
        """
        if self.method == 'gray_world':
            return self._gray_world_gains(r, g, b)
        elif self.method == 'white_patch':
            return self._white_patch_gains(r, g, b)
        elif self.method == 'perfect_reflector':
            return self._perfect_reflector_gains(r, g, b)
        elif self.method == 'gray_edge':
            return self._gray_edge_gains(r, g, b)
        else:  # manual
            gains = self.manual_gains
            return gains.get('R', 1.0), gains.get('G', 1.0), gains.get('B', 1.0)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        通用接口：根据输入维度自动选择 Bayer 或 RGB 处理模式

        参数:
            image: Bayer 图 (H, W) 或 RGB 图 (H, W, 3)

        返回:
            白平衡校正后的图像
        """
        if image.ndim == 2:
            return self.process_bayer(image)
        elif image.ndim == 3 and image.shape[2] == 3:
            return self.process_rgb(image)
        else:
            raise ValueError(f"不支持的输入形状: {image.shape}，期望 (H,W) 或 (H,W,3)")
