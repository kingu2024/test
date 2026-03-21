"""
Robertson CRF 标定模块 - Robertson et al. (2003) 相机响应函数恢复

【算法原理】Robertson et al. (2003) 数学原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【成像模型】
与 Debevec 方法相同的基础模型：
    Z_ij = f(E_i · Δt_j)
其中：
    Z_ij ∈ [0, 255]  — 第 i 个像素位置在第 j 张图像中的像素值
    E_i              — 第 i 个像素位置的场景辐照度（待恢复）
    Δt_j             — 第 j 张图像的曝光时间
    f(·)             — 相机响应函数（Camera Response Function, CRF）

定义响应函数：
    g(z) = f⁻¹(z)  （即 g 是像素值到辐照度的映射，线性域而非对数域）

因此：
    E_i · Δt_j ≈ g(Z_ij)

【EM 迭代框架】
Robertson 方法通过期望最大化 (EM) 迭代交替估计辐照度 E 和响应函数 g：

初始化：
    g(z) = z / 255    （线性假设，即 response[z] = z/255 for z in 0..255）

────────────────────────────────────────────
E-step — 估计辐照度（固定 g，更新 E）
────────────────────────────────────────────
对每个像素位置 i，基于所有曝光图像加权估计辐照度：

    E_i = Σ_j [w(Z_ij) · g(Z_ij) / Δt_j]
          ─────────────────────────────────────
          Σ_j [w(Z_ij) · g(Z_ij)² / Δt_j²]

推导：设误差为 ε_ij = g(Z_ij) - E_i · Δt_j，最小化加权平方误差：
    min Σ_j w(Z_ij) · ε_ij²
对 E_i 求导并置零，得到上述公式。

────────────────────────────────────────────
M-step — 更新响应函数（固定 E，更新 g）
────────────────────────────────────────────
对每个像素值 z，用所有在该值处的像素位置更新响应值：

    g(z) = Σ_{(i,j): Z_ij = z} (E_i · Δt_j)
           ────────────────────────────────────
           count_z

其中 count_z = |{(i,j): Z_ij = z}| 为像素值等于 z 的 (i,j) 对的数量。
若 count_z = 0（某像素值未出现），则保留上一轮的 g(z)。

────────────────────────────────────────────
归一化
────────────────────────────────────────────
    g = g / g[128]

使中间灰度值（z=128）对应的响应为参考单位 1.0，
消除响应函数的尺度不确定性（scale ambiguity）。

────────────────────────────────────────────
迭代终止条件
────────────────────────────────────────────
    max_k |g_new(k) - g_old(k)| < epsilon    （逐点最大绝对差）
    或已达到 max_iter 次迭代上限

【权重函数 w(z)】— 高斯帽形（Gaussian Hat）
    w(z) = exp(-4 · ((z - 127.5) / 127.5)²)

特殊值验证：
    w(0)     ≈ exp(-4) ≈ 0.018   （边界，权重极低）
    w(127)   ≈ exp(-4·(0.5/127.5)²) ≈ 1.0   （近中心，权重最高）
    w(128)   同上，对称
    w(255)   ≈ exp(-4) ≈ 0.018   （边界，权重极低）

相比 Debevec 的三角帽，高斯帽在边界权重不为零，保留更多极端值信息，
但对过曝/欠曝区域的惩罚也更柔和。

【输出格式】
最终将 g（线性域）转换为对数域以与 Debevec 输出格式兼容：
    response(z) = ln(g(z))
并平移使 response[128] = 0（与 Debevec 的 g(128) = 0 约束一致）。

【参考文献】
Mark A. Robertson, Sean Borman, and Robert L. Stevenson.
"Estimation-Theoretic Approach to Dynamic Range Enhancement Using
Multiple Exposures." Journal of Electronic Imaging, 12(2):219–228, 2003.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class RobertsonCalibration:
    """
    Robertson CRF 标定器 (Robertson et al. 2003)

    通过多张不同曝光时间的 LDR 图像，以 EM 迭代方式恢复相机响应函数 g(z)。
    所得曲线（对数域）可供 HDR 合并模块用于重建真实辐照度图。

    与 Debevec 方法的主要区别：
      - Robertson 直接在线性域迭代（无需构建大型线性系统）
      - 使用高斯帽形权重（而非三角帽形）
      - 适合内存受限场景，迭代收敛快（通常 10–30 次）

    支持两种实现方式：
      - process()        — 手写 NumPy EM 迭代（完整实现 Robertson 2003 算法）
      - process_opencv() — 调用 cv2.createCalibrateRobertson() OpenCV 实现

    Attributes:
        max_iter (int): 最大 EM 迭代次数
        epsilon (float): 收敛阈值，g 的逐点最大变化量
    """

    def __init__(self, max_iter: int = 30, epsilon: float = 1e-3):
        """
        初始化 Robertson CRF 标定器

        Args:
            max_iter: 最大 EM 迭代次数。收敛后提前退出；
                      若 30 次内未收敛，返回当前结果并发出警告。
            epsilon:  收敛阈值。当 max|g_new - g_old| < epsilon 时停止迭代。
                      典型值 1e-3，越小精度越高但迭代次数越多。
        """
        self.max_iter = max_iter
        self.epsilon = epsilon
        logger.info(
            f"Robertson 标定器初始化: max_iter={max_iter}, epsilon={epsilon}"
        )

    def _weight(self, z: int) -> float:
        """
        高斯帽形权重函数

        【数学定义】
            w(z) = exp(-4 · ((z - 127.5) / 127.5)²)

        中心（z=127.5）处权重最大（=1.0），边界（z=0 或 z=255）处权重约 0.018。
        相比三角帽，高斯帽在整个值域均大于零，避免边界处权重硬截断。

        Args:
            z: 像素值，整数，范围 [0, 255]

        Returns:
            对应的高斯权重值（float），值域 (0, 1]
        """
        return float(np.exp(-4.0 * ((z - 127.5) / 127.5) ** 2))

    def process(
        self,
        images: List[np.ndarray],
        exposure_times: List[float],
    ) -> np.ndarray:
        """
        手写 NumPy EM 迭代实现 Robertson CRF 标定

        对 B、G、R 三个颜色通道分别独立地执行 EM 迭代：
          a. 初始化 g(z) = (z + 1) / 256（避免零值，保证 M-step 数值稳定）
          b. 预计算权重表 w[0..255]（高斯帽形）
          c. EM 迭代循环（至多 max_iter 次）：
             - E-step: 对每个像素位置，基于当前 g 计算加权辐照度估计 E_i
             - M-step: 对每个像素值 z，用所有匹配像素位置的 (E_i · Δt_j) 更新 g(z)
             - 归一化: g = g / g[128]，使中间参考值为 1.0
             - 收敛检测: 若 max|g_new - g_old| < epsilon，提前退出
          d. 线性域转对数域: response = ln(g)
          e. 平移使 response[128] = 0（与 Debevec 输出格式兼容）

        【数值稳定性措施】
          - E-step 分母加 eps 防止除零（分子为零时 E_i = 0）
          - E 值截断至合理上界 1/min(Δt)，防止辐照度估计爆炸式增长引起迭代发散
          - g 值保持非负，M-step 后截断至 [eps, +∞) 防止负值传播
          - M-step 统计 count_z，仅在 count_z > 0 时更新 g(z)（稀疏像素值保留旧值）
          - 迭代期间用均值归一化（而非 g[128] 归一化），防止 z=128 bin 稀疏时
            小分母造成尺度震荡导致指数发散；收敛后再统一做 g[128]=1.0 归一化
          - 收敛检测在均值归一化域上进行，避免尺度漂移导致误判
          - log 运算前将 g 截断至正数域 max(g, eps)

        【通道顺序】
        输入图像为 OpenCV 惯例 BGR 格式，输出 response curve 的通道顺序
        对应 B=0, G=1, R=2（与 cv2.createCalibrateRobertson 返回一致）。

        Args:
            images: 多曝光 LDR 图像列表，BGR 格式，uint8，形状 (H, W, 3)
                    列表长度 N >= 2
            exposure_times: 各图像对应曝光时间（秒），长度与 images 相同

        Returns:
            response_curve: CRF 对数曲线，形状 (256, 3)，dtype=float64
                            response_curve[z, c] = ln(g_c(z))，且 response_curve[128, c] = 0

        Raises:
            ValueError: 若 images 或 exposure_times 为空，或长度不匹配
        """
        N = len(images)
        if N == 0:
            raise ValueError("images 列表不能为空")
        if len(exposure_times) != N:
            raise ValueError(
                f"images 数量({N})与 exposure_times 数量({len(exposure_times)})不匹配"
            )

        dt = np.array(exposure_times, dtype=np.float64)  # (N,)

        logger.info(
            f"Robertson 标定开始: {N} 张图像, "
            f"图像尺寸={images[0].shape[:2]}, "
            f"max_iter={self.max_iter}, epsilon={self.epsilon}"
        )

        # 预计算权重表 w[z] for z in 0..255
        w_table = np.array([self._weight(z) for z in range(256)], dtype=np.float64)

        # 将所有图像展平为像素矩阵，shape (N, H*W, 3)
        # Z_all[j, i, c] = 第 j 张图像第 i 个像素位置通道 c 的值
        h, w_img = images[0].shape[:2]
        num_pixels = h * w_img

        Z_all = np.stack(
            [img.reshape(num_pixels, 3).astype(np.int32) for img in images],
            axis=0
        )  # (N, num_pixels, 3)

        channel_curves = []

        for c, ch_name in enumerate(["B", "G", "R"]):
            logger.debug(f"  处理通道 {ch_name} (index={c})")

            # Z[j, i] = 第 j 张图像第 i 个像素的该通道值，shape (N, num_pixels)
            Z = Z_all[:, :, c]  # (N, num_pixels), int32

            # ── 初始化 g(z) = (z + 1) / 256 ──────────────────────────
            # 使用 (z+1)/256 而非 z/255，确保 g(0) > 0，避免 M-step 中零值问题
            g = np.arange(1, 257, dtype=np.float64) / 256.0  # g[z] for z=0..255

            eps = 1e-10  # 防止除零的小量

            # dt_col 在整个通道循环中保持不变，提前计算
            dt_col = dt[:, np.newaxis]  # (N, 1)

            # 预计算辐照度的合理上界：当最短曝光时间下像素值为 255 时，
            # g(255) ≈ 1.0（初始化），故 E_max ≈ 1 / min(dt)
            E_max = 1.0 / (np.min(dt) + eps)

            # ── 初始均值归一化 ────────────────────────────────────────
            # 迭代期间用均值归一化（而非 g[128] 归一化），以防止稀疏 bin
            # （如 z=128 只有少量像素）导致尺度剧烈震荡引发指数发散。
            # 收敛后统一做 g[128]=1.0 的最终归一化。
            mean_g = np.mean(g)
            g = g / mean_g if mean_g > eps else g

            converged = False
            delta = float('inf')
            for iteration in range(self.max_iter):
                # 保存当前均值归一化 g 作为收敛比较基准
                g_old = g.copy()

                # ── E-step: 估计每个像素的辐照度 ──────────────────────
                # 对每个像素 i：
                #   E_i = Σ_j [w(Z_ij) · g(Z_ij) / Δt_j]
                #         ────────────────────────────────────
                #         Σ_j [w(Z_ij) · g(Z_ij)² / Δt_j²]
                #
                # Z: (N, P), g: (256,), w_table: (256,)
                # g_z[j, i] = g[Z[j, i]]
                g_z = g[Z]          # (N, num_pixels)
                w_z = w_table[Z]    # (N, num_pixels)

                numerator = np.sum(w_z * g_z / dt_col, axis=0)                # (num_pixels,)
                denominator = np.sum(w_z * g_z ** 2 / dt_col ** 2, axis=0)    # (num_pixels,)

                # 防止除零：分母为零时 E_i 置零；同时将 E 截断至合理范围防止发散
                E = np.where(denominator > eps, numerator / denominator, 0.0)
                E = np.clip(E, 0.0, E_max)
                # E: (num_pixels,), 每个像素的辐照度估计

                # ── M-step: 更新响应函数 g(z) ─────────────────────────
                # 对每个像素值 z：
                #   g_new(z) = Σ_{(i,j): Z_ij=z} (E_i · Δt_j) / count_z
                #
                # E: (num_pixels,), dt: (N,)
                # E_dt[j, i] = E[i] * Δt_j
                E_dt = E[np.newaxis, :] * dt_col    # (N, num_pixels)

                g_new = np.zeros(256, dtype=np.float64)
                count = np.zeros(256, dtype=np.float64)

                # 向量化累加：对所有 (j, i) 对，按 Z[j, i] 累加
                # Z.ravel(): (N * num_pixels,), E_dt.ravel(): (N * num_pixels,)
                z_flat = Z.ravel()        # (N * num_pixels,)
                e_dt_flat = E_dt.ravel()  # (N * num_pixels,)

                np.add.at(g_new, z_flat, e_dt_flat)
                np.add.at(count, z_flat, 1.0)

                # 仅在 count > 0 处更新 g；其余保留旧值（防止未出现像素值导致 g=0）
                mask = count > 0
                g_new[mask] = g_new[mask] / count[mask]
                g_new[~mask] = g_old[~mask]

                # 确保 g 值非负（负值仅由数值误差引起）
                g_new = np.maximum(g_new, eps)

                # ── 均值归一化（稳定迭代用）────────────────────────────
                # 在迭代期间用均值归一化而非 g[128] 归一化，保持尺度稳定
                mean_val = np.mean(g_new)
                if mean_val > eps:
                    g_new = g_new / mean_val

                # ── 收敛检测（在均值归一化域上比较）─────────────────────
                # 对两个均值归一化的 g 向量计算逐点最大绝对差
                delta = np.max(np.abs(g_new - g_old))
                g = g_new

                logger.debug(
                    f"  通道 {ch_name} 迭代 {iteration+1}/{self.max_iter}: "
                    f"delta={delta:.6f}"
                )

                if delta < self.epsilon:
                    logger.debug(
                        f"  通道 {ch_name} 在第 {iteration+1} 次迭代收敛 "
                        f"(delta={delta:.2e} < epsilon={self.epsilon})"
                    )
                    converged = True
                    break

            if not converged:
                logger.warning(
                    f"  通道 {ch_name} 在 {self.max_iter} 次迭代后未收敛，"
                    f"最终 delta={delta:.2e}"
                )

            # ── 最终归一化: g = g / g[128] ────────────────────────────
            # 收敛后将尺度调整为 Robertson 论文约定：g[128] = 1.0
            ref_val = g[128]
            if abs(ref_val) > eps:
                g = g / ref_val
            else:
                # 回退：用中位数归一化（g[128] 极少情况下为零）
                median_val = np.median(g[g > eps]) if np.any(g > eps) else 1.0
                g = g / median_val
                logger.warning(
                    f"  通道 {ch_name}: 最终归一化时 g[128] 接近零，使用中位数归一化"
                )

            # ── 转换为对数域 ──────────────────────────────────────────
            # response(z) = ln(g(z))，截断 g 至正数域防止 log 域错误
            g_clipped = np.maximum(g, eps)
            response = np.log(g_clipped)

            # ── 平移使 response[128] = 0 ─────────────────────────────
            # 与 Debevec 输出中 g(128) = 0 的约束一致
            response = response - response[128]

            channel_curves.append(response)

            logger.debug(
                f"  通道 {ch_name}: response 范围 "
                f"[{response.min():.3f}, {response.max():.3f}]"
            )

        # 堆叠三通道：形状 (256, 3)
        response_curve = np.stack(channel_curves, axis=1)  # (256, 3)

        logger.info(
            f"Robertson 标定完成: response_curve shape={response_curve.shape}, "
            f"范围 [{response_curve.min():.3f}, {response_curve.max():.3f}]"
        )

        return response_curve

    def process_opencv(
        self,
        images: List[np.ndarray],
        exposure_times: List[float],
    ) -> np.ndarray:
        """
        使用 OpenCV 内置 CalibrateRobertson 进行 CRF 标定

        调用 cv2.createCalibrateRobertson(max_iter, epsilon) 创建标定器，
        再调用 .process(images_list, exposure_times_array) 执行标定。

        OpenCV 期望的输入格式：
          - images_list:          Python list，每元素为 uint8 BGR ndarray
          - exposure_times_array: np.float32 一维数组

        Args:
            images: 多曝光 LDR 图像列表，BGR 格式，uint8
            exposure_times: 各图像曝光时间（秒）

        Returns:
            response_curve: CRF 对数曲线，形状 (256, 3)，dtype=float32
                            与 cv2.createCalibrateRobertson 的输出格式一致

        Notes:
            cv2.createCalibrateRobertson 返回的曲线值域、归一化方式可能与手写实现
            略有差异，但两者在数学上等价（相差一个常数偏移）。
        """
        logger.info(
            f"使用 OpenCV CalibrateRobertson: {len(images)} 张图像, "
            f"max_iter={self.max_iter}, epsilon={self.epsilon}"
        )

        calibrate = cv2.createCalibrateRobertson(self.max_iter, self.epsilon)
        exposure_array = np.array(exposure_times, dtype=np.float32)

        response_curve = calibrate.process(images, exposure_array)

        logger.info(
            f"OpenCV Robertson 标定完成: shape={response_curve.shape}, "
            f"范围 [{response_curve.min():.3f}, {response_curve.max():.3f}]"
        )

        return response_curve
