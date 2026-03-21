"""
颜色校正矩阵模块 (Color Correction Matrix, CCM)
================================================

参考论文/资料:
- "Color Correction Matrix" Imatest Documentation
  https://www.imatest.com/docs/colormatrix/
- "CCMNet: Leveraging Calibrated Color Correction Matrices for Cross-Camera Color Constancy"
  ICCV 2025. https://arxiv.org/html/2504.07959v1
- "Color correction pipeline optimization for digital cameras"
  ResearchGate. https://www.researchgate.net/publication/258796047
- OpenCV Color Correction Model:
  https://docs.opencv.org/4.x/d1/dc1/tutorial_ccm_color_correction_model.html

原理说明:
    相机传感器的光谱响应曲线与人眼或标准色彩空间（sRGB、Adobe RGB等）不同，
    导致相机拍摄的颜色与真实颜色存在偏差（颜色失真）。

    颜色校正矩阵（CCM）是一个 3×3 线性变换矩阵，
    将相机原生 RGB 空间的颜色映射到标准颜色空间（如 sRGB）。

    变换公式：
        [R_out]   [m11 m12 m13] [R_in]
        [G_out] = [m21 m22 m23] × [G_in]
        [B_out]   [m31 m32 m33] [B_in]

    每行系数之和通常约为1（保持灰色中性不变）：
        m11 + m12 + m13 ≈ 1

    标准 sRGB CCM 示例（从相机 RGB 到 sRGB，D65 光源下）：
        通常包含较大的对角元素（主对角线主导），
        非对角元素为负值（交叉校正通道间的溢出）。

颜色校正矩阵的标定：
    使用 ColorChecker 标准色卡拍摄，对已知颜色的色块进行最小二乘拟合，
    求解最优 CCM 矩阵使得相机颜色尽可能接近标准颜色。
"""

import numpy as np
from typing import Optional


class ColorCorrectionMatrix:
    """
    颜色校正矩阵类

    参数:
        matrix: 3×3 颜色校正矩阵（行优先）
                None 时使用默认的 sRGB 近似矩阵
        clip_output: 是否将输出裁剪到 [0, 1]
        apply_in_linear: 是否在线性光（gamma=1）空间应用 CCM
                         CCM 必须在线性光空间应用，应用前需确保图像已线性化
    """

    # 默认 CCM：适用于大多数相机的通用近似 sRGB 矩阵
    # 来源：参考 Adobe Camera Raw 标准 D65 下的典型校正矩阵
    DEFAULT_CCM = np.array([
        [ 1.6410, -0.5183, -0.0832],
        [-0.1170,  1.3139, -0.1969],
        [ 0.0195, -0.2120,  1.0790],
    ], dtype=np.float32)

    # 标准 sRGB 到 XYZ（D65）矩阵，用于色域转换
    SRGB_TO_XYZ_D65 = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    def __init__(
        self,
        matrix: Optional[np.ndarray] = None,
        clip_output: bool = True,
    ):
        """
        初始化颜色校正矩阵模块

        参数:
            matrix: 3×3 CCM 矩阵，None 时使用内置默认矩阵
            clip_output: 是否将输出裁剪到 [0, 1]，处理高饱和色时可能超出范围
        """
        if matrix is None:
            self.matrix = self.DEFAULT_CCM.copy()
        else:
            matrix = np.array(matrix, dtype=np.float32)
            assert matrix.shape == (3, 3), \
                f"CCM 矩阵必须为 3×3，当前形状: {matrix.shape}"
            self.matrix = matrix

        self.clip_output = clip_output

    def process(self, rgb: np.ndarray) -> np.ndarray:
        """
        对线性 RGB 图像应用颜色校正矩阵

        矩阵乘法：对每个像素 [R, G, B] 向量，左乘 CCM 矩阵
            [R', G', B'] = CCM × [R, G, B]^T

        注意：CCM 必须在线性光空间（gamma 校正之前）应用！
             如果输入图像已经过 gamma 编码，需先逆 gamma 再应用 CCM。

        参数:
            rgb: 线性 RGB 图像，形状 (H, W, 3)，值域 [0, 1]

        返回:
            颜色校正后的 RGB 图像，形状 (H, W, 3)
        """
        H, W = rgb.shape[:2]
        img_flat = rgb.reshape(-1, 3).astype(np.float32)

        # 矩阵乘法：每行（每像素）的 [R, G, B] 与 CCM^T 相乘
        # CCM 是 (3,3)，img_flat 是 (N,3)
        # 结果 = img_flat @ CCM^T = (N,3) @ (3,3) → (N,3)
        corrected_flat = img_flat @ self.matrix.T

        corrected = corrected_flat.reshape(H, W, 3)

        if self.clip_output:
            corrected = np.clip(corrected, 0.0, 1.0)

        return corrected.astype(np.float32)

    @classmethod
    def calibrate_from_colorchecker(
        cls,
        measured_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        regularization: float = 0.0,
    ) -> 'ColorCorrectionMatrix':
        """
        从 ColorChecker 测量数据标定 CCM 矩阵

        使用最小二乘法求解最优 CCM：
            min ||Reference - Measured × CCM^T||²_F

        参数:
            measured_rgb: 相机实测颜色，形状 (N, 3)，N 为色卡色块数
                          通常使用 24 色 ColorChecker（N=24）
            reference_rgb: 色块的标准参考颜色（线性 sRGB），形状 (N, 3)
            regularization: Tikhonov 正则化系数，防止过拟合

        返回:
            标定好的 ColorCorrectionMatrix 实例
        """
        # 确保输入为浮点数
        M = np.array(measured_rgb, dtype=np.float64)   # (N, 3) 相机测量值
        R = np.array(reference_rgb, dtype=np.float64)  # (N, 3) 标准参考值

        N = M.shape[0]
        assert M.shape == R.shape, "实测值和参考值的形状必须相同"

        if regularization > 0:
            # 带正则化的最小二乘：(M^T M + λI) X = M^T R
            # X = CCM^T
            A = M.T @ M + regularization * np.eye(3)
            b = M.T @ R
            CCM_T = np.linalg.solve(A, b)
        else:
            # 无正则化最小二乘
            CCM_T, _, _, _ = np.linalg.lstsq(M, R, rcond=None)

        ccm_matrix = CCM_T.T.astype(np.float32)

        return cls(matrix=ccm_matrix)

    def get_matrix_info(self) -> dict:
        """
        获取 CCM 矩阵的统计信息，用于验证矩阵合理性

        返回:
            包含矩阵信息的字典：行和、对角元素、行列式等
        """
        info = {
            'matrix': self.matrix.tolist(),
            'row_sums': self.matrix.sum(axis=1).tolist(),  # 每行应近似为1
            'diagonal': np.diag(self.matrix).tolist(),      # 对角元素应为最大值
            'determinant': float(np.linalg.det(self.matrix)),  # 行列式应>0
            'condition_number': float(np.linalg.cond(self.matrix)),  # 条件数越小越稳定
        }
        return info
