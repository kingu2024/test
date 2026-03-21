"""
去马赛克/色彩插值模块 (Demosaicing / CFA Interpolation)
=======================================================

参考论文/资料:
- Bilinear Interpolation: 经典方法，详见 Bayer, B.E. (1976) 原始专利
- AHD: Hirakawa, K. & Parks, T.W. (2005)
  "Adaptive Homogeneity-Directed Demosaicing Algorithm"
  IEEE Transactions on Image Processing, 14(3), 360–369.
- "Demosaicing" Wikipedia: https://en.wikipedia.org/wiki/Demosaicing
- A Comprehensive Survey on ISP, arXiv:2502.05995

背景说明:
    数字相机传感器每个像素只能感知一种颜色（R、G 或 B），
    按 Bayer 模式排列（RGGB 最常见）。要得到完整的彩色图像，
    必须通过插值重建每个像素缺失的两种颜色分量，这就是"去马赛克"。

    Bayer 模式 RGGB（2×2 重复块）:
        R  Gr
        Gb  B

    G 通道像素密度是 R/B 的两倍，因为人眼对绿色最敏感。

本模块实现两种算法：

1. 双线性插值 (Bilinear Interpolation)
   - 最简单的方法，计算量小
   - 在边缘区域会产生"锯齿"和"彩色伪影"（色差）
   - 适合实时处理或质量要求不高的场景

2. 自适应同质性定向算法 (AHD - Adaptive Homogeneity-Directed)
   - 参考 dcraw 的高质量去马赛克实现
   - 核心思想：在每个像素位置，比较水平方向和垂直方向的"同质性"，
     选择同质性更高的方向进行插值，从而减少边缘处的伪影
   - 质量显著优于双线性，但计算量更大
"""

import numpy as np
from scipy.ndimage import convolve
from typing import Optional


class Demosaicing:
    """
    去马赛克/色彩插值类

    参数:
        method: 去马赛克算法
            - 'bilinear': 双线性插值（快速，质量一般）
            - 'ahd':      自适应同质性定向（慢速，高质量）
            - 'malvar':   Malvar-He-Cutler 算法（中等速度，较好质量）
        bayer_pattern: Bayer 排列模式
        output_bits: 输出位深（None 表示保持浮点 [0,1]）
    """

    BAYER_PATTERNS = {
        'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
        'BGGR': {'R': (1, 1), 'Gr': (1, 0), 'Gb': (0, 1), 'B': (0, 0)},
        'GRBG': {'R': (0, 1), 'Gr': (0, 0), 'Gb': (1, 1), 'B': (1, 0)},
        'GBRG': {'R': (1, 0), 'Gr': (1, 1), 'Gb': (0, 0), 'B': (0, 1)},
    }

    def __init__(
        self,
        method: str = 'malvar',
        bayer_pattern: str = 'RGGB',
    ):
        assert method in ('bilinear', 'ahd', 'malvar'), \
            f"不支持的方法: {method}, 支持: bilinear, ahd, malvar"
        self.method = method
        self.bayer_pattern = bayer_pattern.upper()

    def _normalize_bayer_to_rggb(self, raw: np.ndarray) -> np.ndarray:
        """
        将非 RGGB 排列的 Bayer 图转换为 RGGB 排列

        内部统一用 RGGB 处理，处理完后再转回原始排列。
        这样只需实现 RGGB 版本的算法，通过裁切/补零实现其他排列。

        参数:
            raw: 原始 Bayer 图，形状 (H, W)

        返回:
            RGGB 排列的 Bayer 图（可能有1像素行列偏移）
        """
        pattern = self.BAYER_PATTERNS[self.bayer_pattern]
        r_row, r_col = pattern['R']
        # 如果 R 通道在 (0,0) 则已经是 RGGB，无需变换
        if r_row == 0 and r_col == 0:
            return raw
        # 通过切片将 R 通道移到 (0,0) 位置
        return raw[r_row:, r_col:]

    def _bilinear_demosaic_rggb(self, raw: np.ndarray) -> np.ndarray:
        """
        双线性插值去马赛克（RGGB 排列）

        算法原理：
            - G 通道：R/B 位置的 G 值 = 上下左右4个 G 邻居的均值
            - R 通道：Gr 位置的 R 值 = 左右2个 R 邻居均值；
                      Gb 位置的 R 值 = 对角4个 R 邻居均值；
                      B 位置的 R 值 = 上下左右4个 R 邻居均值
            - B 通道：类似 R 通道，对称处理

        使用卷积核实现，比逐像素循环快很多。

        参数:
            raw: RGGB 排列的 Bayer 图，形状 (H, W)

        返回:
            RGB 图像，形状 (H, W, 3)，值域 [0, 1]
        """
        H, W = raw.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)

        # ---------- 步骤1: 提取已知像素 ----------
        # R 通道：位置 (偶行, 偶列)
        r_known = np.zeros((H, W), dtype=np.float32)
        r_known[0::2, 0::2] = raw[0::2, 0::2]

        # G 通道：位置 (偶行, 奇列) 和 (奇行, 偶列)
        g_known = np.zeros((H, W), dtype=np.float32)
        g_known[0::2, 1::2] = raw[0::2, 1::2]  # Gr
        g_known[1::2, 0::2] = raw[1::2, 0::2]  # Gb

        # B 通道：位置 (奇行, 奇列)
        b_known = np.zeros((H, W), dtype=np.float32)
        b_known[1::2, 1::2] = raw[1::2, 1::2]

        # ---------- 步骤2: 双线性插值卷积核 ----------
        # G 通道插值核：对 R/B 位置用4邻域均值插值
        # 注：G 在自己位置已知，只需对缺失位置插值
        g_interp_kernel = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float32) / 4.0

        # R 通道插值核：
        # - 对"十字"位置（上下/左右有R邻居）用2点均值
        # - 对"对角"位置（四角有R邻居）用4点均值
        r_cross_kernel = np.array([
            [0, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0],
        ], dtype=np.float32)

        r_diag_kernel = np.array([
            [0.25, 0, 0.25],
            [0, 0, 0],
            [0.25, 0, 0.25],
        ], dtype=np.float32)

        # ---------- 步骤3: 卷积插值 ----------
        # G 通道完整插值
        g_interp = convolve(g_known, g_interp_kernel, mode='mirror')
        g_full = g_known + g_interp  # 已知位置保留原值，未知位置用插值

        # R 通道：先用十字核，再用对角核
        r_cross = convolve(r_known, r_cross_kernel, mode='mirror')
        r_diag = convolve(r_known, r_diag_kernel, mode='mirror')

        # 创建位置掩膜
        # R 已知位置
        r_mask = np.zeros((H, W), dtype=bool)
        r_mask[0::2, 0::2] = True
        # Gr 位置（需要水平/垂直插值）
        gr_mask = np.zeros((H, W), dtype=bool)
        gr_mask[0::2, 1::2] = True
        # Gb 位置
        gb_mask = np.zeros((H, W), dtype=bool)
        gb_mask[1::2, 0::2] = True
        # B 位置（需要对角插值）
        b_mask = np.zeros((H, W), dtype=bool)
        b_mask[1::2, 1::2] = True

        # 在 R 已知位置使用原值，在 Gr/Gb 位置用十字核，在 B 位置用对角核
        r_full = np.where(r_mask, r_known, np.where(gr_mask | gb_mask, r_cross, r_diag))

        # B 通道：对称处理
        b_cross = convolve(b_known, r_cross_kernel, mode='mirror')
        b_diag = convolve(b_known, r_diag_kernel, mode='mirror')
        b_full = np.where(b_mask, b_known, np.where(gr_mask | gb_mask, b_diag, b_cross))

        rgb[..., 0] = np.clip(r_full, 0.0, 1.0)
        rgb[..., 1] = np.clip(g_full, 0.0, 1.0)
        rgb[..., 2] = np.clip(b_full, 0.0, 1.0)

        return rgb

    def _malvar_demosaic_rggb(self, raw: np.ndarray) -> np.ndarray:
        """
        Malvar-He-Cutler (MHC) 高质量线性去马赛克（RGGB 排列）

        参考: Malvar, H.S., He, L.W., Cutler, R. (2004)
              "High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images"
              ICASSP 2004.

        核心思想：在双线性插值基础上，加入"色差校正项"：
            G_at_R = bilinear_G + (1/4) * Laplacian(R)
        通过利用颜色通道间的相关性（颜色差值变化缓慢），
        显著减少高频伪彩色（false color）伪影。

        参数:
            raw: RGGB 排列的 Bayer 图，形状 (H, W)

        返回:
            RGB 图像，形状 (H, W, 3)，值域 [0, 1]
        """
        H, W = raw.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        raw_f = raw.astype(np.float32)

        # -------- Malvar 卷积核（5×5）--------
        # 参考论文 Table 1，系数乘以 1/8 归一化

        # 在 R 位置插值 G
        K_Gr_at_R = np.array([
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  2,  0,  0],
            [-1,  2,  4,  2, -1],
            [ 0,  0,  2,  0,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=np.float32) / 8.0

        # 在 B 位置插值 G
        K_Gb_at_B = K_Gr_at_R  # 对称，使用相同核

        # 在 Gr 位置（偶行奇列）插值 R（水平方向有 R 邻居）
        K_R_at_Gr = np.array([
            [ 0,  0,  0.5,  0,  0],
            [ 0, -1,  0,  -1,  0],
            [-1,  4,  5,   4, -1],
            [ 0, -1,  0,  -1,  0],
            [ 0,  0,  0.5, 0,  0],
        ], dtype=np.float32) / 8.0

        # 在 Gb 位置（奇行偶列）插值 R（垂直方向有 R 邻居）
        K_R_at_Gb = np.array([
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  4, -1,  0],
            [ 0.5, 0,  5,  0, 0.5],
            [ 0, -1,  4, -1,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=np.float32) / 8.0

        # 在 B 位置插值 R（对角方向）
        K_R_at_B = np.array([
            [ 0,  0, -1.5,  0,  0],
            [ 0,  2,  0,   2,  0],
            [-1.5, 0,  6,   0, -1.5],
            [ 0,  2,  0,   2,  0],
            [ 0,  0, -1.5,  0,  0],
        ], dtype=np.float32) / 8.0

        # 在 R 位置插值 B（对角方向）
        K_B_at_R = K_R_at_B  # 对称

        # 在 Gr 位置插值 B
        K_B_at_Gr = K_R_at_Gb

        # 在 Gb 位置插值 B
        K_B_at_Gb = K_R_at_Gr

        # -------- 构建位置掩膜 --------
        r_mask  = np.zeros((H, W), dtype=np.float32)
        gr_mask = np.zeros((H, W), dtype=np.float32)
        gb_mask = np.zeros((H, W), dtype=np.float32)
        b_mask  = np.zeros((H, W), dtype=np.float32)

        r_mask[0::2, 0::2]  = 1.0
        gr_mask[0::2, 1::2] = 1.0
        gb_mask[1::2, 0::2] = 1.0
        b_mask[1::2, 1::2]  = 1.0

        # -------- 卷积计算各位置插值值 --------
        # 注：convolve 用 boundary='symm'（镜像边界）处理边缘
        G_interp = (
            convolve(raw_f, K_Gr_at_R, mode='mirror') * r_mask +   # R 位置的 G
            convolve(raw_f, K_Gb_at_B, mode='mirror') * b_mask      # B 位置的 G
        )
        G_full = raw_f * (gr_mask + gb_mask) + G_interp  # G 位置保留原值

        R_interp = (
            convolve(raw_f, K_R_at_Gr, mode='mirror') * gr_mask +   # Gr 位置的 R
            convolve(raw_f, K_R_at_Gb, mode='mirror') * gb_mask +   # Gb 位置的 R
            convolve(raw_f, K_R_at_B,  mode='mirror') * b_mask      # B 位置的 R
        )
        R_full = raw_f * r_mask + R_interp  # R 位置保留原值

        B_interp = (
            convolve(raw_f, K_B_at_Gr, mode='mirror') * gr_mask +   # Gr 位置的 B
            convolve(raw_f, K_B_at_Gb, mode='mirror') * gb_mask +   # Gb 位置的 B
            convolve(raw_f, K_B_at_R,  mode='mirror') * r_mask      # R 位置的 B
        )
        B_full = raw_f * b_mask + B_interp  # B 位置保留原值

        rgb[..., 0] = np.clip(R_full, 0.0, 1.0)
        rgb[..., 1] = np.clip(G_full, 0.0, 1.0)
        rgb[..., 2] = np.clip(B_full, 0.0, 1.0)

        return rgb

    def _ahd_demosaic_rggb(self, raw: np.ndarray) -> np.ndarray:
        """
        自适应同质性定向去马赛克（AHD，RGGB 排列）

        参考: Hirakawa & Parks (2005) "Adaptive Homogeneity-Directed Demosaicing"
              IEEE TIP 14(3):360-369

        算法步骤:
            1. 先用双线性/Malvar 得到初始 RGB 估计
            2. 将 RGB 转换到 CIE Lab 色彩空间（感知均匀空间）
            3. 分别计算水平方向和垂直方向插值结果的"同质性指标"
               同质性 = 局部颜色差异的倒数（差异越小越同质）
            4. 在每个像素位置，选择同质性更高的方向的插值结果
               → 边缘处会自动选择沿边缘方向（差异小），避免跨边缘插值伪影

        注：完整 AHD 实现复杂，此处使用简化的方向自适应版本。

        参数:
            raw: RGGB 排列的 Bayer 图，形状 (H, W)

        返回:
            RGB 图像，形状 (H, W, 3)，值域 [0, 1]
        """
        H, W = raw.shape
        raw_f = raw.astype(np.float32)

        # ---------- 步骤1: 两个方向的 G 通道插值 ----------
        # 水平方向 G 插值核（只用左右邻居）
        K_G_horiz = np.array([[0, 0.5, 0, 0.5, 0]], dtype=np.float32)
        # 垂直方向 G 插值核（只用上下邻居）
        K_G_vert = np.array([[0], [0.5], [0], [0.5], [0]], dtype=np.float32)

        g_horiz = convolve(raw_f, K_G_horiz, mode='mirror')
        g_vert  = convolve(raw_f, K_G_vert,  mode='mirror')

        # ---------- 步骤2: 利用色差在各方向插值 R/B ----------
        # AHD 原理：颜色差值（如 R-G）在边缘处比单通道 R 更平滑，
        # 利用色差插值可以减少彩色伪影

        def interp_rb_from_diff(channel_known, g_interp, known_mask, horiz=True):
            """通过色差插值重建 R 或 B 通道"""
            # 计算已知位置的色差
            diff = channel_known - g_interp * known_mask

            if horiz:
                kernel = np.array([[0.5, 0, 0.5]], dtype=np.float32)
            else:
                kernel = np.array([[0.5], [0], [0.5]], dtype=np.float32)

            diff_interp = convolve(diff, kernel, mode='mirror')

            # 未知位置用色差+插值G重建
            result = np.where(known_mask > 0, channel_known,
                              diff_interp + g_interp * (1 - known_mask))
            return result

        # R/B 已知掩膜
        r_mask = np.zeros((H, W), dtype=np.float32)
        r_mask[0::2, 0::2] = 1.0
        b_mask = np.zeros((H, W), dtype=np.float32)
        b_mask[1::2, 1::2] = 1.0
        g_mask = np.zeros((H, W), dtype=np.float32)
        g_mask[0::2, 1::2] = 1.0
        g_mask[1::2, 0::2] = 1.0

        r_known = raw_f * r_mask
        b_known = raw_f * b_mask
        g_known = raw_f * g_mask

        # 水平方向完整 G（已知G + 水平插值）
        g_h = np.where(g_mask > 0, g_known,
                       np.where(r_mask > 0, g_horiz, g_horiz))
        # 垂直方向完整 G
        g_v = np.where(g_mask > 0, g_known,
                       np.where(r_mask > 0, g_vert, g_vert))

        # 两方向下的 R/B 完整重建（简化：对角位置用双方向均值）
        r_h = interp_rb_from_diff(r_known, g_h, r_mask, horiz=True)
        r_v = interp_rb_from_diff(r_known, g_v, r_mask, horiz=False)
        b_h = interp_rb_from_diff(b_known, g_h, b_mask, horiz=True)
        b_v = interp_rb_from_diff(b_known, g_v, b_mask, horiz=False)

        # ---------- 步骤3: 计算同质性指标 ----------
        # 在 CIE Lab 空间中计算局部颜色差异作为同质性衡量标准
        # 简化实现：用 |ΔR| + |ΔG| + |ΔB| 的局部方差估计同质性

        def homogeneity(r, g, b):
            """计算局部颜色差异（越小越同质）"""
            from scipy.ndimage import uniform_filter
            # 颜色梯度
            dr = np.abs(np.gradient(r)[0]) + np.abs(np.gradient(r)[1])
            dg = np.abs(np.gradient(g)[0]) + np.abs(np.gradient(g)[1])
            db = np.abs(np.gradient(b)[0]) + np.abs(np.gradient(b)[1])
            diff = dr + dg + db
            # 局部均值平滑
            return uniform_filter(diff, size=3, mode='mirror')

        homo_h = homogeneity(r_h, g_h, b_h)
        homo_v = homogeneity(r_v, g_v, b_v)

        # ---------- 步骤4: 自适应选择方向 ----------
        # 同质性低（色差大）的地方用另一方向，同质性高的地方用当前方向
        # 选择同质性指标较小（颜色更均匀）的方向
        select_h = homo_h <= homo_v  # True → 用水平方向

        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[..., 0] = np.where(select_h, r_h, r_v)
        rgb[..., 1] = np.where(select_h, g_h, g_v)
        rgb[..., 2] = np.where(select_h, b_h, b_v)

        return np.clip(rgb, 0.0, 1.0)

    def process(self, raw: np.ndarray) -> np.ndarray:
        """
        对 Bayer 图执行去马赛克处理

        参数:
            raw: 白平衡校正后的 Bayer 图，形状 (H, W)，值域 [0, 1]

        返回:
            RGB 彩色图像，形状 (H, W, 3)，值域 [0, 1]
            通道顺序：[..., 0]=R, [..., 1]=G, [..., 2]=B
        """
        # 将非 RGGB 排列转换为 RGGB 进行处理
        pattern = self.BAYER_PATTERNS[self.bayer_pattern]
        r_row, r_col = pattern['R']
        raw_rggb = raw[r_row:, r_col:]  # 偏移使 R 通道在 (0,0)

        H_orig, W_orig = raw.shape
        H_proc, W_proc = raw_rggb.shape

        # 确保尺寸为偶数（去马赛克算法要求）
        H_even = (H_proc // 2) * 2
        W_even = (W_proc // 2) * 2
        raw_rggb = raw_rggb[:H_even, :W_even]

        # 根据方法调用对应的去马赛克算法
        if self.method == 'bilinear':
            rgb = self._bilinear_demosaic_rggb(raw_rggb)
        elif self.method == 'malvar':
            rgb = self._malvar_demosaic_rggb(raw_rggb)
        else:  # ahd
            rgb = self._ahd_demosaic_rggb(raw_rggb)

        # 如果有偏移，补零恢复原始尺寸
        if r_row > 0 or r_col > 0:
            full_rgb = np.zeros((H_orig, W_orig, 3), dtype=np.float32)
            full_rgb[r_row:r_row + H_even, r_col:r_col + W_even] = rgb
            return full_rgb

        return rgb
