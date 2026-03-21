"""
局部色调映射算子模块

局部算子（Local Tone Mapping Operators）根据每个像素的空间邻域动态调整映射函数，
能够比全局算子更好地保留局部对比度。其核心思想是：在亮部区域和暗部区域分别使用
不同的映射参数，使得整体动态范围被压缩的同时，局部细节与对比度得到最大限度的保留。

本模块实现三种经典局部算子：
  1. ReinhardLocal  — Reinhard et al. (2002) 多尺度高斯局部适应色调映射
  2. DurandToneMap  — Durand & Dorsey (2002) 双边滤波基础层压缩
  3. FattalToneMap  — Fattal et al. (2002) 梯度域泊松重建色调映射

输入：float32 HDR 辐照图（BGR 通道顺序，OpenCV 约定）
输出：uint8 LDR 图像（BGR，范围 [0, 255]）
"""

import cv2
import numpy as np
import logging
from typing import Optional
from scipy import sparse
from scipy.sparse import linalg as splinalg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公共辅助函数（与 global_operators.py 相同模式）
# ---------------------------------------------------------------------------

def _extract_luminance(bgr_image: np.ndarray) -> np.ndarray:
    """从 BGR 图像中提取亮度通道。

    使用 ITU-R BT.709 线性亮度公式：
        L = 0.0722·B + 0.7152·G + 0.2126·R

    参数
    ----
    bgr_image : np.ndarray
        输入 BGR float32 图像，形状 (H, W, 3)。

    返回
    ----
    np.ndarray
        亮度图，形状 (H, W)，float32。
    """
    B = bgr_image[:, :, 0]
    G = bgr_image[:, :, 1]
    R = bgr_image[:, :, 2]
    return 0.0722 * B + 0.7152 * G + 0.2126 * R


def _recover_color(
    hdr_image: np.ndarray,
    L_original: np.ndarray,
    L_mapped: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """根据映射后亮度恢复彩色图像。

    对每个通道按亮度比例恢复色彩：
        C_out = C_in · (L_mapped / (L_original + eps))

    参数
    ----
    hdr_image : np.ndarray
        原始 HDR BGR 图像，形状 (H, W, 3)。
    L_original : np.ndarray
        映射前亮度，形状 (H, W)。
    L_mapped : np.ndarray
        映射后亮度，形状 (H, W)。
    eps : float
        防止除零的小量，默认 1e-6。

    返回
    ----
    np.ndarray
        恢复彩色后的图像，形状 (H, W, 3)，float32。
    """
    ratio = L_mapped / (L_original + eps)
    output = hdr_image * ratio[:, :, np.newaxis]
    return output.astype(np.float32)


def _apply_gamma_and_convert(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """对图像应用 Gamma 校正并转换为 uint8。

    步骤：
        1. 裁剪到 [0, 1]
        2. output = image^(1/gamma)
        3. 乘以 255 并转为 uint8

    参数
    ----
    image : np.ndarray
        float32 图像，值域理论上在 [0, 1]。
    gamma : float
        Gamma 值，默认 2.2。

    返回
    ----
    np.ndarray
        uint8 图像，形状与输入相同，值域 [0, 255]。
    """
    clipped = np.clip(image, 0.0, 1.0)
    gamma_corrected = np.power(clipped, 1.0 / gamma)
    return (gamma_corrected * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. ReinhardLocal
# ---------------------------------------------------------------------------

class ReinhardLocal:
    """Reinhard (2002) 局部色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Reinhard 局部算子使用多尺度高斯滤波计算每个像素的局部适应亮度，
    相比全局算子能更好地保留局部对比度。

    核心数学原理：

    1. 对数平均亮度归一化：
           L̄ = exp((1/N) · Σ ln(δ + L(x,y)))
           L_scaled = (a / L̄) · L

    2. 多尺度高斯滤波：
           V1(x,y,s) = L_scaled * G(x,y,s)  — 尺度 s 的高斯卷积
           V2(x,y,s) = V1(x,y,s+1)           — 下一个更大尺度

    3. 活度量（activity measure）决定最佳尺度：
           act(x,y,s) = |V1 - V2| / (2^φ · a/s² + V1)
       搜索最大尺度 s，使得 act(x,y,s) < ε

    4. 局部色调映射：
           L_d(x,y) = L_scaled / (1 + V1_best(x,y))

    5. 恢复彩色并进行 Gamma 校正后输出 uint8
    """

    def __init__(
        self,
        key_value: float = 0.18,
        phi: float = 8.0,
        num_scales: int = 8,
        epsilon: float = 0.05,
    ):
        """初始化 Reinhard 局部算子。

        参数
        ----
        key_value : float
            场景关键值 a，控制整体曝光感知，默认 0.18。
        phi : float
            活度量中的指数参数，控制尺度选择灵敏度，默认 8.0。
        num_scales : int
            多尺度高斯滤波的尺度数量，默认 8。
        epsilon : float
            活度量阈值，低于此值时选定当前尺度为最佳尺度，默认 0.05。
        """
        self.key_value = key_value
        self.phi = phi
        self.num_scales = num_scales
        self.epsilon = epsilon
        logger.debug(
            "ReinhardLocal 初始化：key_value=%.4f, phi=%.2f, num_scales=%d, epsilon=%.4f",
            key_value, phi, num_scales, epsilon,
        )

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 Reinhard 局部色调映射。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像，形状 (H, W, 3)。

        返回
        ----
        np.ndarray
            uint8 BGR 图像，形状 (H, W, 3)，值域 [0, 255]。
        """
        hdr = hdr_image.astype(np.float32)

        # 1. 提取亮度
        L = _extract_luminance(hdr)

        # 2. 计算对数平均亮度并缩放
        delta = 1e-6
        log_avg_lum = np.exp(np.mean(np.log(delta + L)))
        if log_avg_lum < delta:
            log_avg_lum = 1.0
        L_scaled = (self.key_value / log_avg_lum) * L
        logger.debug("对数平均亮度 L̄=%.6f", log_avg_lum)

        # 3. 构建各尺度的高斯滤波结果 V1[s]
        # V1[s] 为尺度 s 的高斯平滑，V2[s] = V1[s+1]
        # sigma_s = 1.6^s （s 从 1 开始）
        V1_list = []
        for s in range(1, self.num_scales + 1):
            sigma = 1.6 ** s
            # 核大小取 sigma 的 6 倍，保证至少 3
            ksize = max(3, int(6 * sigma + 1))
            if ksize % 2 == 0:
                ksize += 1
            v1 = cv2.GaussianBlur(L_scaled, (ksize, ksize), sigma)
            V1_list.append(v1)

        # 4. 逐像素搜索最佳尺度
        # 初始化：最佳尺度使用最粗尺度的 V1
        V1_best = V1_list[-1].copy()

        # 从细到粗遍历（s=1 最细，num_scales 最粗）
        # 对每个 s，计算活度量；若活度 < epsilon 则记录该尺度
        # 使用向量化操作替代逐像素循环
        # 按照论文：从最细尺度开始，当活度量 < epsilon 时停止
        selected = np.zeros(L_scaled.shape, dtype=bool)  # 已选定的像素掩码

        for s_idx in range(self.num_scales - 1):
            V1_s = V1_list[s_idx]
            V2_s = V1_list[s_idx + 1]
            s = s_idx + 1  # 实际尺度编号（1-based）

            # 活度量分母：2^φ · a/s² + V1
            denom = (2.0 ** self.phi) * (self.key_value / (s * s)) + V1_s
            denom = np.where(np.abs(denom) < delta, delta, denom)

            activity = np.abs(V1_s - V2_s) / np.abs(denom)

            # 当活度量 < epsilon 且该像素尚未选定时，记录当前 V1_s
            newly_selected = (~selected) & (activity < self.epsilon)
            V1_best = np.where(newly_selected, V1_s, V1_best)
            selected = selected | newly_selected

        # 5. 局部色调映射
        L_d = L_scaled / (1.0 + V1_best)

        # 6. 归一化到 [0, 1]
        L_d_max = L_d.max()
        if L_d_max > delta:
            L_d = L_d / L_d_max

        # 7. 恢复彩色
        output = _recover_color(hdr, L, L_d)

        # 8. Gamma 校正并转 uint8
        return _apply_gamma_and_convert(output, gamma=2.2)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """使用 OpenCV 内置 Reinhard 色调映射算子（局部模式）。

        light_adapt=1.0 时启用局部适应行为。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像。

        返回
        ----
        np.ndarray
            uint8 BGR 图像。
        """
        tonemap = cv2.createTonemapReinhard(
            gamma=2.2,
            intensity=0.0,
            light_adapt=1.0,
            color_adapt=0.0,
        )
        ldr = tonemap.process(hdr_image)
        return (np.clip(ldr, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. DurandToneMap
# ---------------------------------------------------------------------------

class DurandToneMap:
    """Durand & Dorsey (2002) 双边滤波局部色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Durand 算法将图像分解为"基础层"（低频、大范围亮度变化）和
    "细节层"（局部高频细节），仅压缩基础层的动态范围，保留细节层不变。

    数学原理：

    1. 对数变换：
           L_log = log10(L + ε)

    2. 双边滤波分离：
           base   = bilateralFilter(L_log)
           detail = L_log - base

    3. 压缩基础层（线性压缩到 target_contrast 范围）：
           base_compressed = (base - max(base)) · target_contrast / (max(base) - min(base))

    4. 重构对数亮度并还原：
           L_out = 10^(base_compressed + detail)

    5. 归一化到 [0, 1]，恢复彩色，Gamma 校正，输出 uint8
    """

    def __init__(
        self,
        sigma_spatial: float = 2.0,
        sigma_range: float = 2.0,
        target_contrast: float = 5.0,
    ):
        """初始化 Durand 色调映射算子。

        参数
        ----
        sigma_spatial : float
            双边滤波空间域标准差，默认 2.0。
        sigma_range : float
            双边滤波值域标准差，默认 2.0。
        target_contrast : float
            基础层目标动态范围（对数域），默认 5.0。
        """
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.target_contrast = target_contrast
        logger.debug(
            "DurandToneMap 初始化：sigma_spatial=%.2f, sigma_range=%.2f, target_contrast=%.2f",
            sigma_spatial, sigma_range, target_contrast,
        )

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 Durand 双边滤波色调映射。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像，形状 (H, W, 3)。

        返回
        ----
        np.ndarray
            uint8 BGR 图像，形状 (H, W, 3)，值域 [0, 255]。
        """
        hdr = hdr_image.astype(np.float32)

        # 1. 提取亮度
        L = _extract_luminance(hdr)

        # 2. 对数变换（log10）
        eps = 1e-6
        log_L = np.log10(L + eps)

        # 3. 双边滤波得到基础层
        # OpenCV bilateralFilter 需要 8bit 或 float32，此处直接用 float32
        # d=-1 时根据 sigma_spatial 自动确定邻域大小
        base = cv2.bilateralFilter(
            log_L,
            d=-1,
            sigmaColor=float(self.sigma_range),
            sigmaSpace=float(self.sigma_spatial),
        )

        # 4. 细节层
        detail = log_L - base

        # 5. 压缩基础层到 target_contrast 范围
        base_max = float(base.max())
        base_min = float(base.min())
        base_range = base_max - base_min

        if base_range < eps:
            base_compressed = base - base_max
        else:
            base_compressed = (base - base_max) * (self.target_contrast / base_range)

        # 6. 重构：L_out = 10^(base_compressed + detail)
        L_out = np.power(10.0, base_compressed + detail)

        # 7. 归一化到 [0, 1]
        L_out_max = float(L_out.max())
        if L_out_max > eps:
            L_out = L_out / L_out_max

        # 8. 恢复彩色
        output = _recover_color(hdr, L, L_out)

        # 9. Gamma 校正并转 uint8
        return _apply_gamma_and_convert(output, gamma=2.2)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """使用 OpenCV 内置 Durand 色调映射算子。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像。

        返回
        ----
        np.ndarray
            uint8 BGR 图像。
        """
        tonemap = cv2.createTonemapDurand(
            gamma=2.2,
            contrast=self.target_contrast,
            saturation=self.sigma_range,
            sigma_space=self.sigma_spatial,
            sigma_color=self.sigma_range,
        )
        ldr = tonemap.process(hdr_image)
        return (np.clip(ldr, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. FattalToneMap
# ---------------------------------------------------------------------------

class FattalToneMap:
    """Fattal et al. (2002) 梯度域色调映射算子。

    【算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Fattal 算法在梯度域中对 HDR 亮度的对数进行处理：
    通过衰减大梯度（亮度跳变）、保留小梯度（细节），
    再用泊松方程重建压缩后的对数亮度图，实现局部动态范围压缩。

    数学原理：

    1. 对数域梯度计算：
           H = log(L + ε)
           ∇H = (∂H/∂x, ∂H/∂y)

    2. 梯度幅值衰减函数（Φ）：
           |∇H| = sqrt((∂H/∂x)² + (∂H/∂y)²)
           φ(x,y) = (α / |∇H|) · (|∇H| / α)^β，β ∈ (0,1)
       α 控制衰减中心，β 控制衰减强度（β 越小压缩越强）

    3. 衰减后梯度场：
           Gx = φ · ∂H/∂x
           Gy = φ · ∂H/∂y

    4. 散度场（泊松方程右端项）：
           div(G) = ∂Gx/∂x + ∂Gy/∂y

    5. 泊松方程重建（求解线性方程组）：
           ∇²I = div(G)
       使用 scipy.sparse.linalg.spsolve 求解稀疏线性系统

    6. 大图像 (max(H,W) > max_size) 自动 0.5x 降采样求解后上采样

    7. 归一化到 [0,1]，恢复彩色，Gamma 校正，输出 uint8
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.8,
        saturation: float = 1.0,
        max_size: int = 2048,
    ):
        """初始化 Fattal 梯度域色调映射算子。

        参数
        ----
        alpha : float
            梯度衰减中心值，控制衰减函数的参考幅值，默认 0.1。
        beta : float
            衰减指数，β < 1 时产生压缩效果，默认 0.8。
        saturation : float
            色彩饱和度系数（用于颜色恢复），默认 1.0。
        max_size : int
            稀疏求解的最大图像尺寸，超过时自动降采样，默认 2048。
        """
        self.alpha = alpha
        self.beta = beta
        self.saturation = saturation
        self.max_size = max_size
        logger.debug(
            "FattalToneMap 初始化：alpha=%.4f, beta=%.4f, saturation=%.2f, max_size=%d",
            alpha, beta, saturation, max_size,
        )

    def _solve_poisson(self, div_G: np.ndarray) -> np.ndarray:
        """使用稀疏线性方程组求解泊松方程 ∇²I = div_G。

        构造 5 点差分拉普拉斯矩阵并使用 scipy.sparse.linalg.spsolve 求解。
        为确保方程组唯一解，固定左上角像素为零（狄利克雷边界条件）。

        参数
        ----
        div_G : np.ndarray
            散度场，形状 (H, W)，float64。

        返回
        ----
        np.ndarray
            重建的对数亮度图，形状 (H, W)，float64。
        """
        H, W = div_G.shape
        N = H * W

        # 将 2D 坐标展平为 1D 索引：idx = row * W + col
        # 构造 5 点拉普拉斯矩阵（内部节点）
        # L·I = div_G  →  (D - A)·I = div_G
        # D: 对角线（每个节点的邻居数，内部=4，边=3，角=2）
        # A: 邻接矩阵

        row_idx = []
        col_idx = []
        data = []

        def flat(r, c):
            return r * W + c

        for r in range(H):
            for c in range(W):
                i = flat(r, c)
                neighbors = 0
                if r > 0:
                    row_idx.append(i)
                    col_idx.append(flat(r - 1, c))
                    data.append(-1.0)
                    neighbors += 1
                if r < H - 1:
                    row_idx.append(i)
                    col_idx.append(flat(r + 1, c))
                    data.append(-1.0)
                    neighbors += 1
                if c > 0:
                    row_idx.append(i)
                    col_idx.append(flat(r, c - 1))
                    data.append(-1.0)
                    neighbors += 1
                if c < W - 1:
                    row_idx.append(i)
                    col_idx.append(flat(r, c + 1))
                    data.append(-1.0)
                    neighbors += 1
                # 对角项
                row_idx.append(i)
                col_idx.append(i)
                data.append(float(neighbors))

        L_mat = sparse.csr_matrix(
            (data, (row_idx, col_idx)), shape=(N, N), dtype=np.float64
        )

        # 右端项
        b = div_G.flatten().astype(np.float64)

        # 固定左上角像素（索引 0）为零以唯一化解
        # 通过修改第 0 行：将 L_mat[0,:] 设为 [1, 0, 0, ...]，b[0] = 0
        L_mat = L_mat.tolil()
        L_mat[0, :] = 0.0
        L_mat[0, 0] = 1.0
        b[0] = 0.0
        L_mat = L_mat.tocsr()

        # 求解
        solution = splinalg.spsolve(L_mat, b)
        return solution.reshape(H, W)

    def _build_poisson_matrix(self, H: int, W: int):
        """使用向量化方法构造稀疏拉普拉斯矩阵（5 点差分）。

        相比逐像素循环，向量化构造速度更快，适合中等尺寸图像。

        参数
        ----
        H : int
            图像高度。
        W : int
            图像宽度。

        返回
        ----
        scipy.sparse.csr_matrix
            形状 (H*W, H*W) 的稀疏拉普拉斯矩阵。
        """
        N = H * W

        # 主对角线：每个节点的度数
        diag_main = np.full(N, 4.0)

        # 边界节点度数修正
        # 上边（row=0）缺少上邻居
        diag_main[:W] -= 1.0
        # 下边（row=H-1）缺少下邻居
        diag_main[(H - 1) * W:] -= 1.0
        # 左边（col=0）缺少左邻居
        diag_main[::W] -= 1.0
        # 右边（col=W-1）缺少右邻居
        diag_main[W - 1::W] -= 1.0

        # 水平邻居（±1 偏移）
        diag_h = -np.ones(N - 1)
        # 行末尾到行首的连接需要清零（不是真正的邻居）
        diag_h[W - 1::W] = 0.0

        # 垂直邻居（±W 偏移）
        diag_v = -np.ones(N - W)

        L_mat = (
            sparse.diags(diag_main, 0)
            + sparse.diags(diag_h, 1)
            + sparse.diags(diag_h, -1)
            + sparse.diags(diag_v, W)
            + sparse.diags(diag_v, -W)
        )
        return L_mat.tocsr()

    def _solve_poisson_fast(self, div_G: np.ndarray) -> np.ndarray:
        """向量化稀疏泊松求解器（_solve_poisson 的快速版本）。

        参数
        ----
        div_G : np.ndarray
            散度场，形状 (H, W)。

        返回
        ----
        np.ndarray
            重建的对数亮度图，形状 (H, W)，float64。
        """
        H, W = div_G.shape
        N = H * W

        L_mat = self._build_poisson_matrix(H, W)
        b = div_G.flatten().astype(np.float64)

        # 固定左上角像素为零以唯一化解
        L_mat = L_mat.tolil()
        L_mat[0, :] = 0.0
        L_mat[0, 0] = 1.0
        b[0] = 0.0
        L_mat = L_mat.tocsr()

        solution = splinalg.spsolve(L_mat, b)
        return solution.reshape(H, W)

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """对 HDR 图像执行 Fattal 梯度域色调映射。

        参数
        ----
        hdr_image : np.ndarray
            输入 HDR BGR float32 图像，形状 (H, W, 3)。

        返回
        ----
        np.ndarray
            uint8 BGR 图像，形状 (H, W, 3)，值域 [0, 255]。
        """
        hdr = hdr_image.astype(np.float32)

        # 1. 提取亮度
        L_orig = _extract_luminance(hdr)
        H_orig, W_orig = L_orig.shape

        # 2. 大图像自动降采样（0.5x）
        downsampled = False
        if max(H_orig, W_orig) > self.max_size:
            logger.debug(
                "图像尺寸 (%d, %d) 超过 max_size=%d，自动降采样 0.5x",
                H_orig, W_orig, self.max_size,
            )
            L = cv2.resize(
                L_orig,
                (W_orig // 2, H_orig // 2),
                interpolation=cv2.INTER_AREA,
            )
            downsampled = True
        else:
            L = L_orig.copy()

        H, W = L.shape
        eps = 1e-6

        # 3. 对数域：H_log = log(L + ε)
        H_log = np.log(L.astype(np.float64) + eps)

        # 4. 计算梯度（前向差分）
        # dx: 形状 (H, W-1)，dy: 形状 (H-1, W)
        dx = H_log[:, 1:] - H_log[:, :-1]   # ∂H/∂x
        dy = H_log[1:, :] - H_log[:-1, :]   # ∂H/∂y

        # 5. 梯度幅值（需对齐到相同尺寸）
        # 在 dx 和 dy 共同覆盖的区域 (H-1, W-1) 上计算幅值
        # 取两者的裁剪区域：dx[:H-1, :] 和 dy[:, :W-1]
        dx_crop = dx[:H - 1, :]   # (H-1, W-1)
        dy_crop = dy[:, :W - 1]   # (H-1, W-1)
        grad_mag = np.sqrt(dx_crop ** 2 + dy_crop ** 2)  # (H-1, W-1)

        # 6. 衰减函数 φ：
        #    φ = (α / |∇H|) · (|∇H| / α)^β
        #      = α^(1-β) · |∇H|^(β-1)
        alpha = self.alpha
        beta = self.beta
        phi = (alpha / (grad_mag + eps)) * np.power((grad_mag + eps) / alpha, beta)

        # 7. 衰减后梯度场（对齐到 (H-1, W-1) 区域）
        Gx = phi * dx_crop   # (H-1, W-1)
        Gy = phi * dy_crop   # (H-1, W-1)

        # 8. 计算散度 div(G) = ∂Gx/∂x + ∂Gy/∂y
        # 使用后向差分（对应于前向梯度的伴随算子）
        # div_G 尺寸与 H_log 相同 (H, W)

        # ∂Gx/∂x（水平方向后向差分）
        # Gx 定义在 (H-1, W-1) 的节点 (i, j) 对应位置 (i, j+1) - (i, j)
        # 后向散度：div_x[i,j] = Gx[i,j] - Gx[i,j-1]
        div_x = np.zeros((H, W), dtype=np.float64)
        # 内部列（1 到 W-2）：Gx[:H-1, j-1] 对应 j-1 列
        div_x[:H - 1, 1:W - 1] = Gx[:, 1:] - Gx[:, :-1]
        # 左边界（j=0）：-Gx[:, 0]
        div_x[:H - 1, 0] = -Gx[:, 0]
        # 右边界（j=W-1）：+Gx[:, W-2]
        div_x[:H - 1, W - 1] = Gx[:, -1]

        # ∂Gy/∂y（垂直方向后向差分）
        div_y = np.zeros((H, W), dtype=np.float64)
        div_y[1:H - 1, :W - 1] = Gy[1:, :] - Gy[:-1, :]
        div_y[0, :W - 1] = -Gy[0, :]
        div_y[H - 1, :W - 1] = Gy[-1, :]

        div_G = div_x + div_y

        # 9. 泊松方程求解
        logger.debug("求解泊松方程，图像尺寸 (%d, %d)", H, W)
        I = self._solve_poisson_fast(div_G)

        # 10. 若已降采样，则上采样回原始尺寸
        if downsampled:
            I = cv2.resize(
                I.astype(np.float32),
                (W_orig, H_orig),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float64)

        # 11. 归一化到 [0, 1]
        I_min = I.min()
        I_max = I.max()
        if I_max - I_min > eps:
            I_norm = (I - I_min) / (I_max - I_min)
        else:
            I_norm = np.zeros_like(I)

        I_norm = I_norm.astype(np.float32)

        # 12. 恢复彩色
        output = _recover_color(hdr, L_orig, I_norm)

        # 13. Gamma 校正并转 uint8
        return _apply_gamma_and_convert(output, gamma=2.2)

    def process_opencv(self, hdr_image: np.ndarray) -> np.ndarray:
        """Fattal 色调映射无对应的 OpenCV 实现。

        抛出
        ----
        NotImplementedError
            始终抛出，提示用户改用 process() 方法。
        """
        raise NotImplementedError(
            "Fattal tone mapping has no OpenCV equivalent. Use process() instead."
        )
