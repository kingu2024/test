"""
HDR 辐射图合并模块 - 多曝光图像加权合并为 HDR 辐照度图

【HDR 辐射图合并 数学原理】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【加权辐照度合并公式】

    ln(E_i) = Σ_j w(Z_ij) · (g(Z_ij) - ln(Δt_j)) / Σ_j w(Z_ij)

其中：
    E_i     — 像素位置 i 的辐照度估计（真实物理量）
    Z_ij    — 像素位置 i 在图像 j 中的像素值（0~255 整数）
    g(Z)    — 相机响应函数，由 Debevec 或 Robertson 标定得到
              g(Z) = ln(f⁻¹(Z))，即像素值到对数辐照度的映射
    Δt_j    — 图像 j 的曝光时间（秒）
    w(Z)    — 权重函数，抑制过曝（Z≈255）和欠曝（Z≈0）像素

【权重函数 w(z)：三角帽形（Triangle Hat）】

    w(z) = z + 1       if z <= 127
    w(z) = 256 - z     if z > 127

特殊值验证：
    w(0)   = 0 + 1  = 1    （极暗区权重较小）
    w(127) = 127+1  = 128  （中间值最大权重）
    w(128) = 256-128= 128
    w(255) = 256-255= 1    （过饱和区权重较小）

注意：此定义与 Debevec 标定器中的 w(0)=0 略有不同，
      合并阶段中 Z=0 赋权重 1，保证极暗像素仍有贡献，
      避免全黑像素导致分母为零。

【退化处理：全饱和像素】

当某像素位置所有曝光均饱和（Σw = 0），即所有图像中该像素
均为极端值（0 或 255），无法由加权公式可靠估计辐照度。

退化策略：
    - 计算所有曝光时间的几何均值 t_geom = exp(mean(ln(Δt_j)))
    - 找到曝光时间最接近 t_geom 的图像索引 j*
    - 直接使用该图像的像素值 Z_{i,j*} 通过响应曲线估计辐照度：
          ln(E_i) = g(Z_{i,j*}) - ln(Δt_{j*})
    - 记录 warning 日志

【参考文献】
Paul E. Debevec and Jitendra Malik. "Recovering High Dynamic Range Radiance
Maps from Photographs." Proceedings of SIGGRAPH 1997, pp. 369–378.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class HDRMerge:
    """
    HDR 辐射图合并器

    将多张不同曝光时间的 LDR 图像，结合相机响应函数，
    合并为一张 float32 HDR 辐照度图。

    支持两种实现方式：
      - process()        — 手写 NumPy 加权合并（完整实现公式）
      - process_opencv() — 调用 cv2.createMergeDebevec() OpenCV 实现

    Attributes:
        logger: 模块日志记录器
    """

    def __init__(self):
        """
        初始化 HDR 合并器

        仅存储日志器引用，无其他配置参数。
        合并参数由 process() 调用时传入。
        """
        self.logger = logger
        logger.info("HDRMerge 初始化完成")

    def _weight(self, z: np.ndarray) -> np.ndarray:
        """
        向量化三角帽形权重函数

        【数学定义】
            w(z) = z + 1       if z <= 127
            w(z) = 256 - z     if z > 127

        Args:
            z: 像素值数组，dtype 可为 uint8 或 int，值域 [0, 255]

        Returns:
            weights: 与 z 同形状的权重数组，dtype=float32
                     值域 [1, 128]，在 z=127/128 处取最大值 128
        """
        z = z.astype(np.float32)
        weights = np.where(z <= 127, z + 1.0, 256.0 - z)
        return weights.astype(np.float32)

    def process(
        self,
        images: List[np.ndarray],
        exposure_times: List[float],
        response_curve: np.ndarray,
    ) -> np.ndarray:
        """
        手写 NumPy 加权合并，生成 HDR 辐照度图

        【算法步骤】

        对每个像素位置 i 和颜色通道 c：
          1. 从响应曲线查表：g_val = response_curve[Z_ij, c]
          2. 计算权重：w_val = w(Z_ij)
          3. 累加分子：numerator   += w_val * (g_val - ln(Δt_j))
          4. 累加分母：denominator += w_val
          5. ln(E) = numerator / denominator（分母 > 0 时）
          6. E = exp(ln(E))，转换为辐照度

        退化处理（全饱和像素，denominator == 0）：
          - 使用曝光时间最接近几何均值的图像像素直接估计

        Args:
            images: 多曝光 LDR 图像列表，BGR uint8，形状 (H, W, 3)
                    列表长度 N >= 2
            exposure_times: 各图像曝光时间（秒），长度与 images 相同
                            必须严格正值
            response_curve: CRF 对数曲线，形状 (256, 3)，dtype=float64 或 float32
                            response_curve[z, c] = g_c(z) = ln(f_c⁻¹(z))
                            由 DebevecCalibration 或 RobertsonCalibration 生成

        Returns:
            hdr_map: HDR 辐照度图，形状 (H, W, 3)，dtype=float32
                     每个通道值为辐照度估计 E_i（线性物理量）

        Notes:
            - 返回值为线性辐照度（非对数），可直接传入色调映射模块
            - response_curve 通道顺序应与 images 通道顺序一致（BGR）
        """
        N = len(images)
        H, W, C = images[0].shape
        ln_dt = np.log(np.array(exposure_times, dtype=np.float64))

        logger.info(
            f"HDR 加权合并开始: {N} 张图像, 尺寸=({H},{W},{C}), "
            f"曝光=[{', '.join(f'{t:.4f}' for t in exposure_times)}]"
        )

        # 将响应曲线转为 float64 保证精度
        g = response_curve.astype(np.float64)  # (256, C)

        # 累加器：分子和分母，形状 (H, W, C)
        numerator = np.zeros((H, W, C), dtype=np.float64)
        denominator = np.zeros((H, W, C), dtype=np.float64)

        for j, (img, ln_t) in enumerate(zip(images, ln_dt)):
            # img: (H, W, C) uint8
            Z = img.astype(np.int32)  # (H, W, C)，值域 [0, 255]

            # 查响应曲线：对每个通道查 g[Z[:,:,c], c]
            # g_val[h, w, c] = g[Z[h,w,c], c]
            g_val = np.zeros((H, W, C), dtype=np.float64)
            for c in range(C):
                g_val[:, :, c] = g[Z[:, :, c], c]

            # 计算权重 w(Z)，形状 (H, W, C)
            w_val = self._weight(img).astype(np.float64)

            # 累加：numerator += w * (g(Z) - ln(Δt))
            #        denominator += w
            numerator += w_val * (g_val - ln_t)
            denominator += w_val

        # 计算 ln(E)：仅在分母 > 0 的像素处有效
        ln_E = np.zeros((H, W, C), dtype=np.float64)
        valid_mask = denominator > 0  # (H, W, C)
        ln_E[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        # 退化处理：找到分母为 0 的像素（全饱和）
        saturated_mask = ~valid_mask  # (H, W, C)
        n_saturated = int(saturated_mask.sum())

        if n_saturated > 0:
            logger.warning(
                f"检测到 {n_saturated} 个全饱和像素（所有曝光均饱和），"
                f"将使用最接近几何均值曝光时间的图像像素值代替"
            )

            # 几何均值曝光时间：t_geom = exp(mean(ln(Δt_j)))
            ln_dt_arr = np.array(ln_dt)
            t_geom = np.exp(np.mean(ln_dt_arr))

            # 找最接近几何均值的图像索引
            exp_times_arr = np.array(exposure_times)
            j_star = int(np.argmin(np.abs(exp_times_arr - t_geom)))

            logger.warning(
                f"几何均值曝光时间 t_geom={t_geom:.4f}s，"
                f"选用第 {j_star} 张图像（t={exposure_times[j_star]:.4f}s）"
            )

            # 对全饱和像素，用 j* 图像的响应曲线估计辐照度
            fallback_img = images[j_star].astype(np.int32)  # (H, W, C)
            ln_dt_star = ln_dt[j_star]

            for c in range(C):
                # 取该通道全饱和的像素掩码 (H, W)
                ch_mask = saturated_mask[:, :, c]
                if ch_mask.any():
                    z_fallback = fallback_img[:, :, c][ch_mask]
                    g_fallback = g[z_fallback, c]
                    ln_E[:, :, c][ch_mask] = g_fallback - ln_dt_star

        # 转换：E = exp(ln(E))
        E = np.exp(ln_E).astype(np.float32)

        logger.info(
            f"HDR 合并完成: shape={E.shape}, dtype={E.dtype}, "
            f"范围=[{E.min():.4f}, {E.max():.4f}]"
        )

        return E

    def process_opencv(
        self,
        images: List[np.ndarray],
        exposure_times: List[float],
        response_curve: np.ndarray,
    ) -> np.ndarray:
        """
        使用 OpenCV 内置 MergeDebevec 合并 HDR 图像

        调用 cv2.createMergeDebevec() 创建合并器，
        再调用 .process(images, exposure_times, response_curve) 执行合并。

        【注意】OpenCV 的 MergeDebevec 期望响应曲线为形状 (256, 1, 3) 的 float32 数组，
        格式与 cv2.createCalibrateDebevec 的输出一致。
        若传入形状为 (256, 3) 的数组，需在此处进行格式转换。

        Args:
            images: 多曝光 LDR 图像列表，BGR uint8，形状 (H, W, 3)
            exposure_times: 各图像曝光时间（秒）
            response_curve: CRF 对数曲线
                            支持形状 (256, 3) 或 (256, 1, 3)

        Returns:
            hdr_map: HDR 辐照度图，形状 (H, W, 3)，dtype=float32

        Notes:
            OpenCV 的合并结果与手写实现在数值上一致，
            但内部实现细节（如权重函数边界）可能略有差异。
        """
        logger.info(
            f"使用 OpenCV MergeDebevec: {len(images)} 张图像, "
            f"response_curve shape={response_curve.shape}"
        )

        # OpenCV 期望响应曲线形状为 (256, 1, 3)，dtype=float32
        curve = response_curve.astype(np.float32)
        if curve.ndim == 2:
            # (256, 3) → (256, 1, 3)
            curve = curve[:, np.newaxis, :]

        merge_debevec = cv2.createMergeDebevec()
        exposure_array = np.array(exposure_times, dtype=np.float32)

        hdr_map = merge_debevec.process(images, exposure_array, curve)

        logger.info(
            f"OpenCV MergeDebevec 完成: shape={hdr_map.shape}, dtype={hdr_map.dtype}, "
            f"范围=[{hdr_map.min():.4f}, {hdr_map.max():.4f}]"
        )

        return hdr_map
