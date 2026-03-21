"""
锐化增强模块 (Sharpening & Edge Enhancement)
=============================================

参考论文/资料:
- Unsharp Masking: 传统暗房技术，数字化后广泛用于 ISP 和图像编辑
- Laplacian Sharpening: 基于 Laplacian 算子的经典图像锐化
- "Inside Image Processing Pipelines" Ignitarium
  https://ignitarium.com/inside-image-processing-pipelines/
- Infinite-ISP 锐化模块: https://github.com/xx-isp/infinite-isp

背景说明:
    经过去噪、插值等处理后，图像可能出现一定程度的模糊。
    锐化的目的是增强边缘和细节，提升图像视觉清晰度。

    过度锐化会引入"光晕伪影"（Halo Artifacts）：
    在边缘两侧出现明显的亮/暗条纹，影响真实感。
    因此锐化通常在 ISP 的最后阶段进行，并需要合理控制强度。

    在 ISP 中，锐化通常在 YUV 色彩空间的亮度（Y）通道进行，
    避免对色度通道过度处理导致色彩失真。

本模块实现：
1. 反锐化掩蔽 (Unsharp Masking, USM)：最常用的图像锐化方法
2. Laplacian 锐化：基于二阶导数的锐化
3. 自适应锐化 (Adaptive Sharpening)：在平坦区域减弱锐化，在边缘增强锐化
"""

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, uniform_filter
from typing import Optional


class Sharpening:
    """
    锐化增强类

    参数:
        method: 锐化算法
            - 'usm':       反锐化掩蔽（Unsharp Masking，最常用）
            - 'laplacian': Laplacian 锐化（基于二阶导数）
            - 'adaptive':  自适应锐化（保护平坦区域，增强边缘）
        strength: 锐化强度，值域 [0, 2]，1.0 为标准强度
        sigma: 高斯模糊标准差（USM 中控制锐化细节尺度）
        threshold: 自适应锐化的边缘检测阈值（低于此值的区域不锐化）
        apply_to_luma: 是否仅对亮度通道锐化（True=更自然，False=全通道）
    """

    def __init__(
        self,
        method: str = 'usm',
        strength: float = 0.5,
        sigma: float = 1.0,
        threshold: float = 0.05,
        apply_to_luma: bool = True,
    ):
        assert method in ('usm', 'laplacian', 'adaptive'), \
            f"不支持的方法: {method}"
        assert 0.0 <= strength <= 5.0, "strength 建议在 0~2 之间"

        self.method = method
        self.strength = strength
        self.sigma = sigma
        self.threshold = threshold
        self.apply_to_luma = apply_to_luma

    def _unsharp_masking(self, image: np.ndarray) -> np.ndarray:
        """
        反锐化掩蔽 (Unsharp Masking, USM)

        历史背景：
            USM 源于模拟暗房技术：将原始图像与其模糊版本相减得到"细节掩蔽"，
            然后将细节加回原图，从而增强高频细节（边缘和纹理）。

        算法步骤：
            1. 对原图做高斯模糊，得到低频版本（blur）
            2. 计算高频细节（mask）= 原图 - 模糊图
            3. 锐化结果 = 原图 + strength × mask
                        = 原图 + strength × (原图 - 模糊图)
                        = (1 + strength) × 原图 - strength × 模糊图

        参数控制：
            - sigma: 控制被增强的细节尺度（越大锐化半径越大）
            - strength: 锐化强度（越大越锐利，但也越容易出光晕）

        参数:
            image: 单通道或多通道图像，值域 [0, 1]

        返回:
            USM 锐化后的图像
        """
        if image.ndim == 3:
            # 对每个通道分别处理
            blurred = np.stack([
                gaussian_filter(image[..., c], sigma=self.sigma, mode='mirror')
                for c in range(image.shape[2])
            ], axis=-1)
        else:
            blurred = gaussian_filter(image, sigma=self.sigma, mode='mirror')

        # 高频细节（锐化掩模）
        mask = image - blurred

        # 锐化结果
        sharpened = image + self.strength * mask

        return np.clip(sharpened, 0.0, 1.0).astype(np.float32)

    def _laplacian_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Laplacian 锐化

        原理：
            Laplacian 算子 ∇²f 计算图像的二阶导数，
            在边缘处（亮度变化剧烈区域）响应大，在均匀区域响应为零。

            锐化公式：
                output = input - k × ∇²input

            其中 k > 0 时，边缘被增强（加强中心，抑制扩散）。

        Laplacian 卷积核（4-邻域）：
            [0, -1,  0]
            [-1, 4, -1]
            [0, -1,  0]

        Laplacian 卷积核（8-邻域）：
            [-1, -1, -1]
            [-1,  8, -1]
            [-1, -1, -1]

        参数:
            image: 单通道或多通道图像，值域 [0, 1]

        返回:
            Laplacian 锐化后的图像
        """
        def sharpen_channel(ch):
            """对单通道施加 Laplacian 锐化"""
            # scipy.ndimage.laplace 使用 4-邻域 Laplacian 核
            lap = laplace(ch.astype(np.float64), mode='mirror').astype(np.float32)
            # 减去 Laplacian（正的 Laplacian 对应凹区域，减去可以尖锐化峰值）
            return ch - self.strength * lap

        if image.ndim == 3:
            result = np.stack([
                sharpen_channel(image[..., c])
                for c in range(image.shape[2])
            ], axis=-1)
        else:
            result = sharpen_channel(image)

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _adaptive_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        自适应锐化

        核心思想：
            USM 对整幅图像施加相同的锐化强度，会在以下区域产生问题：
            - 平坦区域（天空、皮肤）：噪声被放大，出现颗粒感
            - 过强边缘：出现"光晕"（亮暗条纹伪影）

            自适应锐化根据局部边缘强度调整锐化强度：
            - 平坦区域（梯度小）：弱锐化或不锐化
            - 边缘/纹理区域（梯度大）：强锐化

        算法步骤：
            1. 计算局部梯度幅值（边缘强度图）
            2. 构建自适应锐化权重图：权重 ∝ 边缘强度（非线性映射）
            3. 应用 USM 得到全局锐化结果
            4. 按自适应权重混合：原图 + 权重 × USM细节

        参数:
            image: 单通道或多通道图像，值域 [0, 1]

        返回:
            自适应锐化后的图像
        """
        # 计算亮度（用于边缘检测）
        if image.ndim == 3:
            luma = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        else:
            luma = image

        # 计算梯度幅值（使用简单差分）
        gx = np.abs(np.gradient(luma, axis=1))
        gy = np.abs(np.gradient(luma, axis=0))
        gradient_mag = (gx + gy)  # L1 近似，比 sqrt(gx²+gy²) 更快

        # 平滑梯度图（去除单像素噪声导致的过检测）
        gradient_mag_smooth = uniform_filter(gradient_mag, size=3, mode='mirror')

        # 构建自适应权重：阈值之上的梯度区域才进行锐化
        # 使用 sigmoid 型软门控：平滑过渡，避免锐利的边界
        weight = gradient_mag_smooth / (gradient_mag_smooth + self.threshold)

        if image.ndim == 3:
            weight = weight[..., np.newaxis]

        # 计算 USM 细节
        if image.ndim == 3:
            blurred = np.stack([
                gaussian_filter(image[..., c], sigma=self.sigma, mode='mirror')
                for c in range(image.shape[2])
            ], axis=-1)
        else:
            blurred = gaussian_filter(image, sigma=self.sigma, mode='mirror')

        detail = image - blurred

        # 自适应应用：只在有边缘的区域增强细节
        result = image + self.strength * weight * detail

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _rgb_to_yuv(self, rgb: np.ndarray) -> np.ndarray:
        """
        RGB 转换到 YUV 色彩空间（BT.601 标准）

        YUV 分离亮度（Y）和色差（U, V）：
            Y =  0.299R + 0.587G + 0.114B  （亮度，人眼最敏感）
            U = -0.147R - 0.289G + 0.436B  （蓝色差）
            V =  0.615R - 0.515G - 0.100B  （红色差）

        在 YUV 空间进行锐化的优点：
            只对 Y 通道锐化，不影响色彩，视觉效果更自然。
            UV 通道通常低分辨率存储（如 YUV 4:2:0），对其锐化意义不大。
        """
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        Y =  0.299 * R + 0.587 * G + 0.114 * B
        U = -0.147 * R - 0.289 * G + 0.436 * B
        V =  0.615 * R - 0.515 * G - 0.100 * B
        return np.stack([Y, U, V], axis=-1).astype(np.float32)

    def _yuv_to_rgb(self, yuv: np.ndarray) -> np.ndarray:
        """YUV 转回 RGB（BT.601 逆变换）"""
        Y, U, V = yuv[..., 0], yuv[..., 1], yuv[..., 2]
        R = Y + 1.140 * V
        G = Y - 0.394 * U - 0.581 * V
        B = Y + 2.032 * U
        return np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0).astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        对图像执行锐化增强

        如果 apply_to_luma=True（推荐），先转换到 YUV 空间，
        只对 Y（亮度）通道锐化，保持色彩不变，视觉效果更自然。

        参数:
            image: 输入图像，(H, W) 单通道 或 (H, W, 3) RGB，值域 [0, 1]

        返回:
            锐化后的图像，形状和值域与输入相同
        """
        is_color = (image.ndim == 3 and image.shape[2] == 3)

        if is_color and self.apply_to_luma:
            # 转到 YUV 空间，只处理 Y 通道
            yuv = self._rgb_to_yuv(image)
            y_channel = yuv[..., 0]

            # 对亮度通道锐化
            if self.method == 'usm':
                y_sharp = self._unsharp_masking(y_channel)
            elif self.method == 'laplacian':
                y_sharp = self._laplacian_sharpening(y_channel)
            else:  # adaptive
                y_sharp = self._adaptive_sharpening(y_channel)

            yuv_sharp = yuv.copy()
            yuv_sharp[..., 0] = y_sharp

            return self._yuv_to_rgb(yuv_sharp)

        else:
            # 直接对原图处理（单通道或多通道）
            if self.method == 'usm':
                return self._unsharp_masking(image)
            elif self.method == 'laplacian':
                return self._laplacian_sharpening(image)
            else:  # adaptive
                return self._adaptive_sharpening(image)
