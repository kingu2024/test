"""
噪声消除模块 (Noise Reduction, NR / Denoising)
================================================

参考论文/资料:
- Bilateral Filter: Tomasi & Manduchi (1998) "Bilateral Filtering for Gray and Color Images"
  ICCV 1998. （双边滤波器原始论文）
- BM3D: Dabov et al. (2007) "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering"
  IEEE TIP 16(8):2080-2095. （工业界最高质量传统去噪算法）
- NLM: Buades et al. (2005) "A Non-Local Algorithm for Image Denoising" CVPR 2005.
  （非局部均值去噪）
- "Physics-Guided ISO-Dependent Sensor Noise Modeling" CVPR 2023.
- "A Comprehensive Survey on Image Signal Processing" arXiv:2502.05995

噪声类型说明:
    相机图像中主要有以下几种噪声：
    1. 散粒噪声 (Shot Noise / Photon Noise)：
       - 遵循泊松分布，与信号幅度有关（Signal-dependent）
       - 光子到达探测器时的量子涨落导致
       - 高 ISO 时尤为明显

    2. 读出噪声 (Read Noise)：
       - 在模数转换过程中引入的电路噪声
       - 遵循高斯分布，与信号无关（Signal-independent）

    3. 热噪声 (Thermal Noise / Dark Current)：
       - 传感器温度导致的暗电流，与曝光时间有关

    4. 量化噪声 (Quantization Noise)：
       - 模数转换时有限位深引入的误差

本模块实现以下去噪算法：

1. 高斯滤波 (Gaussian Filter)：简单，速度快，但会模糊边缘
2. 双边滤波 (Bilateral Filter)：保边去噪，经典算法
3. 非局部均值 (Non-Local Means, NLM)：高质量，较慢
4. 中值滤波 (Median Filter)：对脉冲噪声（椒盐噪声）效果好
5. 导向滤波 (Guided Filter)：快速保边滤波，适合 ISP 中使用
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from typing import Optional, Tuple


class NoiseReduction:
    """
    噪声消除类

    参数:
        method: 去噪算法
            - 'gaussian':   高斯低通滤波（速度最快，质量最差）
            - 'bilateral':  双边滤波（经典保边去噪）
            - 'nlm':        非局部均值（高质量，慢）
            - 'median':     中值滤波（适合椒盐/脉冲噪声）
            - 'guided':     导向滤波（快速保边，适合 ISP）
        sigma_spatial: 空间域高斯标准差（像素），控制平滑半径
        sigma_color:   色彩域高斯标准差，控制双边/NLM 的边缘保持强度
                       值越小越保边，值越大越平滑
        nlm_patch_size: NLM 算法的比较块大小（奇数）
        nlm_search_size: NLM 算法的搜索窗口大小（奇数）
    """

    def __init__(
        self,
        method: str = 'bilateral',
        sigma_spatial: float = 1.5,
        sigma_color: float = 0.1,
        nlm_patch_size: int = 5,
        nlm_search_size: int = 15,
    ):
        assert method in ('gaussian', 'bilateral', 'nlm', 'median', 'guided'), \
            f"不支持的方法: {method}"

        self.method = method
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.nlm_patch_size = nlm_patch_size
        self.nlm_search_size = nlm_search_size

    def _gaussian_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        高斯低通滤波去噪

        原理：对每个像素用以空间距离为权重的高斯核做加权平均。
        优点：计算极快（可分离核）
        缺点：不区分噪声和边缘，对所有高频信号都做平滑，导致图像模糊

        参数:
            image: 输入图像，(H, W) 或 (H, W, 3)

        返回:
            高斯滤波后的图像
        """
        if image.ndim == 3:
            # 对每个通道独立处理
            result = np.stack([
                gaussian_filter(image[..., c], sigma=self.sigma_spatial, mode='mirror')
                for c in range(image.shape[2])
            ], axis=-1)
        else:
            result = gaussian_filter(image, sigma=self.sigma_spatial, mode='mirror')

        return result.astype(np.float32)

    def _bilateral_filter_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        单通道双边滤波

        原理（Tomasi & Manduchi 1998）：
            output(p) = Σ_{q∈N(p)} w_s(p,q) × w_c(I(p), I(q)) × I(q)
                        ─────────────────────────────────────────────────
                        Σ_{q∈N(p)} w_s(p,q) × w_c(I(p), I(q))

            其中：
            - w_s(p,q) = exp(-||p-q||² / (2σ_s²))  空间高斯权重（距离越远权重越小）
            - w_c(I_p, I_q) = exp(-|I_p - I_q|² / (2σ_c²))  色彩高斯权重（颜色差异越大权重越小）

            关键思想：在边缘处，相邻像素颜色差异大，色彩权重 w_c 趋近于0，
            因此边缘两侧不会相互平滑，从而保留了边缘。

        参数:
            channel: 单通道图像，形状 (H, W)

        返回:
            双边滤波后的单通道图像
        """
        H, W = channel.shape

        # 计算空间核半径（通常取 3σ）
        radius = max(1, int(3 * self.sigma_spatial))
        kernel_size = 2 * radius + 1

        # 预计算空间高斯权重（一维，可分离）
        dist = np.arange(-radius, radius + 1, dtype=np.float32)
        spatial_weight_1d = np.exp(-dist**2 / (2 * self.sigma_spatial**2))

        # 构建空间权重核（2D）
        spatial_weight_2d = np.outer(spatial_weight_1d, spatial_weight_1d)

        # 填充图像边界（镜像填充）
        padded = np.pad(channel, radius, mode='reflect')

        result = np.zeros_like(channel)

        # 逐像素处理（效率低但清晰直观；实际应用中应用 C++ 扩展或 OpenCV）
        for i in range(H):
            for j in range(W):
                # 提取邻域窗口
                window = padded[i:i + kernel_size, j:j + kernel_size]

                # 计算色彩权重：基于中心像素与窗口内各像素的色差
                center_val = channel[i, j]
                color_weight = np.exp(-(window - center_val)**2 / (2 * self.sigma_color**2))

                # 联合权重 = 空间权重 × 色彩权重
                weight = spatial_weight_2d * color_weight
                weight_sum = weight.sum()

                if weight_sum > 0:
                    result[i, j] = (weight * window).sum() / weight_sum
                else:
                    result[i, j] = center_val

        return result.astype(np.float32)

    def _bilateral_filter_fast(self, channel: np.ndarray) -> np.ndarray:
        """
        快速双边滤波近似（基于直方图/分箱）

        将色彩空间分成若干个 bin，对每个 bin 单独做高斯平滑，
        然后用各 bin 的权重加权合并，近似精确双边滤波。

        这比精确实现快数十倍，质量损失很小。

        参数:
            channel: 单通道图像，值域 [0, 1]

        返回:
            快速双边滤波后的图像
        """
        num_bins = 16  # 将值域 [0,1] 分为 16 个 bin
        H, W = channel.shape
        result_num = np.zeros((H, W), dtype=np.float32)
        result_den = np.zeros((H, W), dtype=np.float32)

        bin_width = 1.0 / num_bins

        for k in range(num_bins):
            bin_center = (k + 0.5) * bin_width

            # 计算每个像素与当前 bin 中心的色彩权重
            color_diff = channel - bin_center
            color_w = np.exp(-color_diff**2 / (2 * self.sigma_color**2))

            # 对 channel × color_w 和 color_w 分别做高斯平滑
            num_smoothed = gaussian_filter(channel * color_w, sigma=self.sigma_spatial, mode='mirror')
            den_smoothed = gaussian_filter(color_w, sigma=self.sigma_spatial, mode='mirror')

            result_num += num_smoothed
            result_den += den_smoothed

        epsilon = 1e-8
        result = result_num / (result_den + epsilon)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _nlm_denoise_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        非局部均值去噪（Non-Local Means，NLM）

        参考: Buades, Coll & Morel (2005) "A Non-Local Algorithm for Image Denoising"

        算法核心思想：
            图像中存在大量结构相似的区域（自相似性）。
            例如均匀纹理、重复图案等。
            利用这种自相似性：用图像中所有结构相似块的加权平均来估计目标块。

        公式：
            output(p) = Σ_{q∈Ω} w(p,q) × I(q)

            权重：w(p,q) ∝ exp(-||P(p) - P(q)||² / (2h²))
            其中 P(p) 和 P(q) 是以 p 和 q 为中心的小块（patch），
            h 为平滑参数（类似 sigma_color）。

        参数:
            channel: 单通道图像，值域 [0, 1]

        返回:
            NLM 去噪后的图像
        """
        H, W = channel.shape
        half_p = self.nlm_patch_size // 2   # 块的半径
        half_s = self.nlm_search_size // 2  # 搜索窗口半径
        h2 = (self.sigma_color * self.nlm_patch_size)**2  # 平滑参数

        # 边界填充
        pad = half_p + half_s
        padded = np.pad(channel, pad, mode='reflect')

        result = np.zeros_like(channel)
        weights_sum = np.zeros_like(channel)

        # 在搜索窗口内遍历所有位移 (di, dj)
        for di in range(-half_s, half_s + 1):
            for dj in range(-half_s, half_s + 1):
                # 计算两个位移块之间的平方差（通过卷积/滑窗均值）
                shifted = padded[
                    pad + di: pad + di + H,
                    pad + dj: pad + dj + W,
                ]

                # 块内像素的平方差均值 ||P(p) - P(q)||² / n
                diff_sq = (channel - shifted)**2
                # 用均值滤波计算块均值（等效于块内平均）
                patch_dist = uniform_filter(diff_sq,
                                            size=self.nlm_patch_size,
                                            mode='reflect')

                # 计算相似性权重
                w = np.exp(-np.maximum(patch_dist, 0.0) / (h2 + 1e-8))

                # 加权累积
                result += w * shifted
                weights_sum += w

        # 归一化
        epsilon = 1e-8
        result = result / (weights_sum + epsilon)

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _guided_filter(self, image: np.ndarray, guide: Optional[np.ndarray] = None) -> np.ndarray:
        """
        导向滤波 (Guided Filter)

        参考: He, Sun & Tang (2013) "Guided Image Filtering"
              IEEE TPAMI 35(6):1397-1409.

        算法核心：
            利用"引导图像"（guide）的结构信息，对输入图像进行保边平滑。
            引导图像通常就是输入图像本身（self-guided），此时等效于保边平滑。

        数学原理：
            假设输出 q 与引导图 I 在局部窗口 k 内存在线性关系：
                q_i = a_k × I_i + b_k，∀i ∈ ω_k

            最小化目标（最小二乘）：
                E(a_k, b_k) = Σ_{i∈ω_k} [(a_k I_i + b_k - p_i)² + ε a_k²]

            最优解（正则化线性回归）：
                a_k = (Cov(I, p) / (Var(I) + ε))
                b_k = mean(p) - a_k × mean(I)

            最终输出：
                q_i = mean(a)_i × I_i + mean(b)_i

        参数:
            image: 输入图像（待平滑），(H, W) 或 (H, W, 3)
            guide: 引导图像（与输入相同尺寸），None 时使用输入本身

        返回:
            导向滤波后的图像
        """
        if guide is None:
            guide = image

        # 转为浮点，处理单通道情况
        I = guide.astype(np.float32)
        p = image.astype(np.float32)
        is_color = (p.ndim == 3)

        if not is_color:
            I = I[..., np.newaxis]
            p = p[..., np.newaxis]

        H, W, C = p.shape
        # 引导图如果是彩色，简化为亮度
        if I.ndim == 3:
            I_gray = 0.299 * I[..., 0] + 0.587 * I[..., 1] + 0.114 * I[..., 2]
        else:
            I_gray = I

        # 导向滤波参数
        radius = max(1, int(self.sigma_spatial))
        eps = self.sigma_color ** 2  # 正则化系数（值越大越平滑）
        size = 2 * radius + 1

        # 计算局部统计量（使用均值滤波近似滑动窗口均值）
        mean_I = uniform_filter(I_gray, size=size, mode='mirror')
        mean_I2 = uniform_filter(I_gray * I_gray, size=size, mode='mirror')
        var_I = mean_I2 - mean_I * mean_I  # 局部方差

        result_channels = []
        for c in range(C):
            pc = p[..., c]
            mean_p = uniform_filter(pc, size=size, mode='mirror')
            mean_Ip = uniform_filter(I_gray * pc, size=size, mode='mirror')
            cov_Ip = mean_Ip - mean_I * mean_p  # 局部协方差

            # 线性系数
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            # 平均系数（每个像素属于多个窗口，取均值）
            mean_a = uniform_filter(a, size=size, mode='mirror')
            mean_b = uniform_filter(b, size=size, mode='mirror')

            # 输出
            q = mean_a * I_gray + mean_b
            result_channels.append(np.clip(q, 0.0, 1.0))

        result = np.stack(result_channels, axis=-1)
        if not is_color:
            result = result[..., 0]

        return result.astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        对 RGB 图像执行噪声消除

        参数:
            image: 输入图像，(H, W) 或 (H, W, 3)，值域 [0, 1]

        返回:
            去噪后的图像，形状与输入相同
        """
        if self.method == 'gaussian':
            return self._gaussian_denoise(image)

        elif self.method == 'bilateral':
            # 使用快速近似双边滤波（逐通道处理）
            if image.ndim == 3:
                return np.stack([
                    self._bilateral_filter_fast(image[..., c])
                    for c in range(image.shape[2])
                ], axis=-1)
            else:
                return self._bilateral_filter_fast(image)

        elif self.method == 'nlm':
            if image.ndim == 3:
                return np.stack([
                    self._nlm_denoise_channel(image[..., c])
                    for c in range(image.shape[2])
                ], axis=-1)
            else:
                return self._nlm_denoise_channel(image)

        elif self.method == 'median':
            if image.ndim == 3:
                size = max(3, int(2 * self.sigma_spatial) * 2 + 1)
                return np.stack([
                    median_filter(image[..., c], size=size, mode='mirror')
                    for c in range(image.shape[2])
                ], axis=-1).astype(np.float32)
            else:
                size = max(3, int(2 * self.sigma_spatial) * 2 + 1)
                return median_filter(image, size=size, mode='mirror').astype(np.float32)

        else:  # guided
            return self._guided_filter(image)
