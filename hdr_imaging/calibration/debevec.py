"""
Debevec CRF 标定模块 - Debevec & Malik (1997) 相机响应函数恢复

【算法原理】Debevec & Malik (1997) 数学原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【成像模型】
相机成像过程：
    Z = f(E·Δt)
其中：
    Z   ∈ [0, 255]   — 像素值（量化后的数字灰度值）
    E               — 场景辐照度（irradiance），待恢复的真实物理量
    Δt              — 曝光时间（shutter speed）
    f(·)            — 相机响应函数（Camera Response Function, CRF）

由于 f(·) 单调递增，定义其对数反函数：
    g(Z) = ln(f⁻¹(Z)) = ln(E) + ln(Δt)

即：对同一场景像素位置 i、第 j 张曝光图像，有：
    g(Z_ij) = ln(E_i) + ln(Δt_j)

【线性系统构建】
设有 N 张图像、采样 P 个像素位置，未知数为：
    x = [g(0), g(1), ..., g(255),  ln(E_1), ln(E_2), ..., ln(E_P)]
       共 256 + P 个未知数

方程组：

1) 数据拟合方程（P × N 条）：
       w(Z_ij) · [g(Z_ij) - ln(E_i) - ln(Δt_j)] = 0
   对每个像素位置 i (1..P) 和图像 j (1..N)。

2) 平滑约束（254 条，z = 1..254）：
       λ · w(z) · [g(z-1) - 2·g(z) + g(z+1)] = 0
   使 g(·) 曲线光滑，λ 为平滑权重。

3) 固定约束（1 条）：
       g(128) = 0
   消除 g(·) 的不确定缩放因子（scale ambiguity）。

写成矩阵形式：
    A · x = b
用 SVD 或最小二乘 (np.linalg.lstsq) 求解，提取 x 的前 256 个分量，
即得到相机响应函数曲线 g(z)（z = 0..255）。

【权重函数 w(z)】— 三角帽形（Triangle Hat）
    w(z) = z - Z_min       if z <= (Z_min + Z_max) / 2
    w(z) = Z_max - z       if z >  (Z_min + Z_max) / 2
其中 Z_min = 0, Z_max = 255，中点 = 127.5。

特殊值验证：
    w(0)   = 0 - 0   = 0
    w(128) = 255 - 128 = 127   （注：128 > 127.5，故用第二公式）
    w(255) = 255 - 255 = 0

【参考文献】
Paul E. Debevec and Jitendra Malik. "Recovering High Dynamic Range Radiance
Maps from Photographs." Proceedings of SIGGRAPH 1997, pp. 369–378.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class DebevecCalibration:
    """
    Debevec CRF 标定器 (Debevec & Malik 1997)

    通过多张不同曝光时间的 LDR 图像，恢复相机响应函数 g(z) = ln(f⁻¹(z))。
    所得曲线可供 HDR 合并模块用于重建真实辐照度图。

    支持两种实现方式：
      - process()        — 手写 NumPy 线性系统（完整实现 Debevec 论文算法）
      - process_opencv() — 调用 cv2.createCalibrateDebevec() OpenCV 实现

    Attributes:
        samples (int): 随机采样像素点数量
        lambda_smooth (float): 平滑约束权重 λ
    """

    def __init__(self, samples: int = 50, lambda_smooth: float = 10.0):
        """
        初始化 Debevec CRF 标定器

        Args:
            samples: 随机采样像素点数量，控制方程组规模。
                     推荐范围：20–200。越多精度越高但速度越慢。
            lambda_smooth: 平滑约束权重 λ，控制 g(·) 曲线平滑程度。
                           越大曲线越平滑，通常取 10–100。
        """
        self.samples = samples
        self.lambda_smooth = lambda_smooth
        logger.info(
            f"Debevec 标定器初始化: samples={samples}, lambda_smooth={lambda_smooth}"
        )

    def _weight(self, z: int) -> float:
        """
        三角帽形权重函数

        【数学定义】
            w(z) = z - Z_min       if z <= (Z_min + Z_max) / 2
            w(z) = Z_max - z       if z >  (Z_min + Z_max) / 2
        其中 Z_min=0, Z_max=255，中点 = 127.5。

        边界值：
            w(0)   = 0
            w(128) = 127   （128 > 127.5，使用第二分支）
            w(255) = 0

        过饱和（z=0或z=255）和极暗像素的权重接近0，
        减少噪声和量化误差对 CRF 估计的影响。

        Args:
            z: 像素值，整数，范围 [0, 255]

        Returns:
            对应权重值（float）
        """
        z_min = 0
        z_max = 255
        mid = (z_min + z_max) / 2.0  # 127.5
        if z <= mid:
            return float(z - z_min)
        else:
            return float(z_max - z)

    def _sample_pixels(self, images: List[np.ndarray]) -> np.ndarray:
        """
        随机采样像素位置，返回所有图像在这些位置的像素值矩阵

        从图像中随机选取 self.samples 个像素位置（行列坐标），
        收集每张图像在这些位置的像素值，构造采样矩阵。

        注意：此方法处理单通道图像（二维数组），颜色通道在外部循环分离。

        Args:
            images: 单通道图像列表，每张形状 (H, W)，uint8

        Returns:
            Z: 像素值矩阵，形状 (samples, num_images)，dtype=int
               Z[i, j] 为第 i 个采样点在第 j 张图像中的像素值
        """
        h, w = images[0].shape[:2]
        num_images = len(images)

        # 随机选取 self.samples 个 (row, col) 坐标
        rows = np.random.randint(0, h, size=self.samples)
        cols = np.random.randint(0, w, size=self.samples)

        # 构建采样矩阵 Z[i, j]
        Z = np.zeros((self.samples, num_images), dtype=np.int32)
        for j, img in enumerate(images):
            # 若传入三维数组，取第一通道（调用方负责传入单通道）
            if img.ndim == 3:
                channel = img[:, :, 0]
            else:
                channel = img
            Z[:, j] = channel[rows, cols]

        return Z

    def _recover_crf_single_channel(
        self,
        Z: np.ndarray,
        ln_dt: np.ndarray,
    ) -> np.ndarray:
        """
        对单通道像素采样矩阵恢复 CRF 曲线 g(z)

        【线性系统构建详述】
        未知数向量 x，长度 = 256 + P：
            x[0..255]       — g(0), g(1), ..., g(255)
            x[256..256+P-1] — ln(E_1), ..., ln(E_P)

        方程行数：
            数据方程：P × N 行
            平滑方程：254 行
            约束方程：1 行

        数据方程（行 k = i*N + j）：
            A[k, Z[i,j]]         += w_ij
            A[k, 256 + i]        -= w_ij
            b[k]                  = w_ij * ln_dt[j]

        平滑方程（行 P*N + (z-1)，z = 1..254）：
            A[row, z-1]  = +lambda * w(z)
            A[row, z]    = -2*lambda * w(z)
            A[row, z+1]  = +lambda * w(z)
            b[row]       = 0

        约束方程（最后一行）：
            A[last, 128] = 1
            b[last]      = 0

        Args:
            Z: 采样像素矩阵，形状 (P, N)，dtype=int，值域 [0, 255]
            ln_dt: 曝光时间的对数数组，形状 (N,)，ln_dt[j] = ln(Δt_j)

        Returns:
            g_curve: CRF 对数曲线，形状 (256,)，float64
                     g_curve[z] = ln(f⁻¹(z))

        Raises:
            RuntimeError: 当 lstsq 求解失败（矩阵秩不足）时
        """
        P, N = Z.shape
        n_unknowns = 256 + P

        # 预计算所有 z 的权重（向量化）
        w_table = np.array([self._weight(z) for z in range(256)], dtype=np.float64)

        # 行数：P*N 数据方程 + 254 平滑方程 + 1 约束方程
        n_data = P * N
        n_smooth = 254
        n_eq = n_data + n_smooth + 1

        A = np.zeros((n_eq, n_unknowns), dtype=np.float64)
        b = np.zeros(n_eq, dtype=np.float64)

        # ── 数据拟合方程 ──────────────────────────────────────────────
        # 方程：w(Z_ij) * g(Z_ij)  -  w(Z_ij) * ln(E_i)  =  w(Z_ij) * ln(Δt_j)
        row = 0
        for i in range(P):
            for j in range(N):
                z_val = int(Z[i, j])
                w = w_table[z_val]
                A[row, z_val] = w           # 对应 g(Z_ij)
                A[row, 256 + i] = -w        # 对应 -ln(E_i)
                b[row] = w * ln_dt[j]       # 右端 w * ln(Δt_j)
                row += 1

        # ── 平滑约束 ──────────────────────────────────────────────────
        # 方程：λ * w(z) * [g(z-1) - 2*g(z) + g(z+1)] = 0
        for z in range(1, 255):  # z = 1..254
            w = self.lambda_smooth * w_table[z]
            A[row, z - 1] = w
            A[row, z]     = -2.0 * w
            A[row, z + 1] = w
            # b[row] = 0 （已初始化为0）
            row += 1

        # ── 固定约束 g(128) = 0 ───────────────────────────────────────
        A[row, 128] = 1.0
        b[row] = 0.0
        row += 1

        assert row == n_eq, f"方程行数不匹配: {row} != {n_eq}"

        # ── 最小二乘求解 ──────────────────────────────────────────────
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"CRF recovery failed: insufficient exposure variation"
            ) from e

        # 检查秩：若秩严重不足，说明曝光变化不足
        if rank < 10:
            raise RuntimeError(
                "CRF recovery failed: insufficient exposure variation"
            )

        # 提取 g(z) 曲线（前 256 个分量）
        g_curve = x[:256]
        return g_curve

    def process(
        self,
        images: List[np.ndarray],
        exposure_times: List[float],
    ) -> np.ndarray:
        """
        手写 NumPy 实现 Debevec CRF 标定

        对 B、G、R 三个颜色通道分别独立地：
          1. 随机采样 self.samples 个像素位置，构建 Z 矩阵
          2. 构建过约束线性系统 A·x = b
          3. 用 np.linalg.lstsq 求解
          4. 提取 g(z) 曲线（256 点）

        最终将三通道曲线堆叠为 (256, 3) 矩阵返回。

        【通道顺序】
        输入图像为 OpenCV 惯例 BGR 格式，输出 response curve 的通道顺序
        对应 B=0, G=1, R=2（与 cv2.createCalibrateDebevec 返回一致）。

        Args:
            images: 多曝光 LDR 图像列表，BGR 格式，uint8，形状 (H, W, 3)
                    列表长度 N >= 2
            exposure_times: 各图像对应曝光时间（秒），长度与 images 相同

        Returns:
            response_curve: CRF 对数曲线，形状 (256, 3)，dtype=float64
                            response_curve[z, c] = g_c(z) = ln(f_c⁻¹(z))

        Raises:
            RuntimeError: 若 SVD/最小二乘求解失败（曝光变化不足）
        """
        N = len(images)
        ln_dt = np.log(np.array(exposure_times, dtype=np.float64))

        logger.info(
            f"Debevec 标定开始: {N} 张图像, {self.samples} 采样点, "
            f"lambda={self.lambda_smooth}"
        )

        channel_curves = []

        for c, ch_name in enumerate(["B", "G", "R"]):
            logger.debug(f"  处理通道 {ch_name} (index={c})")

            # 提取单通道图像列表
            ch_images = [img[:, :, c] for img in images]

            # 采样像素值矩阵 Z，形状 (P, N)
            Z = self._sample_pixels(ch_images)

            # 恢复该通道的 g(z) 曲线
            g_curve = self._recover_crf_single_channel(Z, ln_dt)
            channel_curves.append(g_curve)

            logger.debug(
                f"  通道 {ch_name}: g 范围 [{g_curve.min():.3f}, {g_curve.max():.3f}]"
            )

        # 堆叠三通道：形状 (256, 3)
        response_curve = np.stack(channel_curves, axis=1)  # (256, 3)

        logger.info(
            f"Debevec 标定完成: response_curve shape={response_curve.shape}, "
            f"范围 [{response_curve.min():.3f}, {response_curve.max():.3f}]"
        )

        return response_curve

    def process_opencv(
        self,
        images: List[np.ndarray],
        exposure_times: List[float],
    ) -> np.ndarray:
        """
        使用 OpenCV 内置 CalibrateDebevec 进行 CRF 标定

        调用 cv2.createCalibrateDebevec(samples, lambda_smooth) 创建标定器，
        再调用 .process(images_list, exposure_times_array) 执行标定。

        OpenCV 期望的输入格式：
          - images_list:         Python list，每元素为 uint8 BGR ndarray
          - exposure_times_array: np.float32 一维数组

        Args:
            images: 多曝光 LDR 图像列表，BGR 格式，uint8
            exposure_times: 各图像曝光时间（秒）

        Returns:
            response_curve: CRF 对数曲线，形状 (256, 3)，dtype=float32
                            与 cv2.createCalibrateDebevec 的输出格式一致

        Notes:
            cv2.createCalibrateDebevec 返回的曲线值域、归一化方式可能与手写实现
            略有差异，但两者在数学上等价（相差一个常数偏移）。
        """
        logger.info(
            f"使用 OpenCV CalibrateDebevec: {len(images)} 张图像, "
            f"samples={self.samples}, lambda={self.lambda_smooth}"
        )

        calibrate = cv2.createCalibrateDebevec(self.samples, self.lambda_smooth)
        exposure_array = np.array(exposure_times, dtype=np.float32)

        response_curve = calibrate.process(images, exposure_array)

        logger.info(
            f"OpenCV Debevec 标定完成: shape={response_curve.shape}, "
            f"范围 [{response_curve.min():.3f}, {response_curve.max():.3f}]"
        )

        return response_curve
