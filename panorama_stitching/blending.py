"""
图像融合模块 - 图像全景拼接
Image Blending Module

【图像融合的必要性】
直接拼接两幅图像会在重叠区域产生明显的接缝（seam）。
原因包括：
1. 曝光差异（不同时刻光线变化）
2. 相机响应函数非线性
3. 视角变化导致的视差（parallax）
4. 几何配准误差（sub-pixel level）

【融合算法对比】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 简单平均融合 (Naive Averaging):
   I_blend(x) = α·I₁(x) + (1-α)·I₂(x)
   问题：重叠区出现"鬼影"（ghost）

2. 最大值融合 (Maximum):
   I_blend(x) = max(I₁(x), I₂(x))
   适合: 低照度场景，但忽略了强度连续性

3. 线性加权融合 (Linear Blending / Feathering):
   权重根据到图像边界的距离线性变化:
   w(x) = dist(x, boundary)
   I_blend = Σᵢ wᵢ(x)·Iᵢ(x) / Σᵢ wᵢ(x)
   优点: 消除硬边界
   缺点: 视差大时仍有鬼影

4. 多分辨率融合 (Laplacian Pyramid Blending):
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   【最优融合方法，本模块重点实现】

   理论基础:
   将图像分解为不同频率的层次：
   - 低频：整体亮度和颜色，需要平滑过渡（宽过渡带）
   - 高频：细节纹理，需要锐利边界（窄过渡带）

   构建高斯金字塔 (Gaussian Pyramid):
   G₀ = I (原图)
   Gₖ = REDUCE(Gₖ₋₁) = (Gₖ₋₁ * h) ↓2
   其中 h 为 5×5 高斯核: h = [1 4 6 4 1]ᵀ[1 4 6 4 1]/256

   构建拉普拉斯金字塔 (Laplacian Pyramid):
   Lₖ = Gₖ - EXPAND(Gₖ₊₁)
   其中 EXPAND(G) = G ↑2 * 4h
   Lₙ = Gₙ (最顶层保留低频)

   融合拉普拉斯金字塔:
   L̃ₖ(x) = wₖ(x)·L¹ₖ(x) + (1-wₖ(x))·L²ₖ(x)
   wₖ: 在第k层对掩码进行高斯平滑（宽度随层级增大而增大）

   重建融合图像:
   Ĝ₀ = L̃₀ + EXPAND(L̃₁ + EXPAND(L̃₂ + ... + EXPAND(L̃ₙ)...))

5. 最优缝合线 (Optimal Seam Finding) + 图切割 (Graph Cut):
   将接缝选择建模为最小割问题:
   最小化: E = Σ_{p∈seam} C(p)
   C(p) = |I₁(p) - I₂(p)| + |I₁(q) - I₂(q)|  (邻域色差)
   使用 Dijkstra 或 Graph Cut 求解最优缝合线
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ImageBlender:
    """
    多方法图像融合器
    主要方法: 多分辨率拉普拉斯金字塔融合 + 最优缝合线
    """

    def __init__(self, method: str = 'multiband', n_levels: int = 6):
        """
        Args:
            method: 融合方法 'multiband' / 'feather' / 'simple'
            n_levels: 拉普拉斯金字塔层数（多分辨率融合）
        """
        self.method = method.lower()
        self.n_levels = n_levels
        logger.info(f"图像融合器: 方法={method}, 金字塔层数={n_levels}")

    def blend(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> np.ndarray:
        """
        融合两幅图像

        Args:
            img1, img2: 待融合图像（相同尺寸）
            mask1, mask2: 各图像的有效区域掩码 (0 or 255)

        Returns:
            融合后的图像
        """
        if self.method == 'multiband':
            return self._multiband_blend(img1, img2, mask1, mask2)
        elif self.method == 'feather':
            return self._feather_blend(img1, img2, mask1, mask2)
        else:
            return self._simple_blend(img1, img2, mask1, mask2)

    def _simple_blend(
        self,
        img1: np.ndarray, img2: np.ndarray,
        mask1: np.ndarray, mask2: np.ndarray
    ) -> np.ndarray:
        """
        简单拷贝融合（优先使用img1覆盖img2）
        策略: 仅在 img1 无效时使用 img2

        I_blend = I₁  if mask₁ > 0
                  I₂  if mask₁ = 0 and mask₂ > 0
                  0   otherwise
        """
        result = np.zeros_like(img1)
        m1 = mask1 > 0
        m2 = mask2 > 0

        # img2 填充
        result[m2] = img2[m2]
        # img1 覆盖
        result[m1] = img1[m1]

        return result

    def _feather_blend(
        self,
        img1: np.ndarray, img2: np.ndarray,
        mask1: np.ndarray, mask2: np.ndarray
    ) -> np.ndarray:
        """
        羽化（线性加权）融合
        权重 = 到图像边界的欧氏距离变换

        【距离变换原理】
        DT(p) = min_{q∈boundary} ||p - q||₂
        使用 cv2.distanceTransform 高效计算（基于 Meijster 算法）

        归一化权重:
        w₁(p) = DT₁(p) / (DT₁(p) + DT₂(p))
        w₂(p) = 1 - w₁(p)

        融合:
        I_blend(p) = w₁(p)·I₁(p) + w₂(p)·I₂(p)
        """
        # 计算距离变换权重（到边界的距离越大权重越高）
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)

        # 归一化权重（避免除零）
        total = dist1 + dist2
        total = np.where(total < 1e-6, 1.0, total)
        w1 = (dist1 / total)[:, :, np.newaxis]  # (H, W, 1)

        # 线性加权融合
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        result = w1 * img1_f + (1 - w1) * img2_f

        # 处理两个掩码都为0的区域
        both_invalid = (mask1 == 0) & (mask2 == 0)
        result[both_invalid] = 0

        return np.clip(result, 0, 255).astype(np.uint8)

    def _multiband_blend(
        self,
        img1: np.ndarray, img2: np.ndarray,
        mask1: np.ndarray, mask2: np.ndarray
    ) -> np.ndarray:
        """
        多分辨率拉普拉斯金字塔融合
        参考: Burt & Adelson (1983) "A multiresolution spline with application to image mosaics"

        【算法步骤】
        1. 生成融合权重掩码（结合距离变换）
        2. 对掩码进行高斯平滑 → 高斯金字塔
        3. 分别对 img1, img2 构建拉普拉斯金字塔
        4. 在每一层按权重融合拉普拉斯系数
        5. 从最顶层逐层重建图像

        【数学细节】
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        高斯核（5×5）:
        h = [1,4,6,4,1]ᵀ·[1,4,6,4,1] / 256

        下采样 REDUCE:
        Gₖ[i,j] = Σₘₙ h[m,n] · Gₖ₋₁[2i+m, 2j+n]

        上采样 EXPAND:
        G̃ₖ₋₁ = 4 · Σₘₙ h[m,n] · Gₖ[⌊(i-m)/2⌋, ⌊(j-n)/2⌋]

        拉普拉斯层:
        Lₖ = Gₖ - EXPAND(Gₖ₊₁)

        融合（第k层）:
        L̃ₖ = αₖ · L¹ₖ + (1-αₖ) · L²ₖ
        其中 αₖ 是第k层的权重金字塔（高斯平滑后）

        重建:
        Ĝₖ₋₁ = Lₖ₋₁ + EXPAND(Ĝₖ)
        """
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)

        # ── 步骤1: 计算融合权重掩码 ───────────────────────────
        # 结合距离变换，在重叠区域平滑过渡
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
        total = dist1 + dist2
        total = np.where(total < 1e-6, 1.0, total)
        # 融合权重 α: img1 的权重，取值 [0,1]
        alpha = (dist1 / total).astype(np.float32)

        # ── 步骤2: 构建高斯金字塔（对权重掩码）────────────────
        # GP_alpha[k]: 第k层权重（经过k次高斯平滑+下采样）
        # 低频层用更宽的过渡带，高频层用更窄的过渡带
        GP_alpha = self._build_gaussian_pyramid(alpha)

        # ── 步骤3: 构建拉普拉斯金字塔（对两幅图像）──────────
        LP1 = self._build_laplacian_pyramid(img1_f)
        LP2 = self._build_laplacian_pyramid(img2_f)

        # ── 步骤4: 按权重融合每一层的拉普拉斯系数 ────────────
        LP_blend = []
        for k, (l1, l2, ga) in enumerate(zip(LP1, LP2, GP_alpha)):
            # ga 形状: (H,W)，扩展到 (H,W,3) 适配彩色图像
            ga_3ch = ga[:, :, np.newaxis] if len(l1.shape) == 3 else ga
            blended_layer = ga_3ch * l1 + (1 - ga_3ch) * l2
            LP_blend.append(blended_layer)

        # ── 步骤5: 从最顶层逐层重建 ─────────────────────────
        result = self._reconstruct_from_laplacian(LP_blend)

        # 裁剪到有效范围
        result = np.clip(result, 0, 255).astype(np.uint8)

        # 处理两个掩码都为0的区域
        both_invalid = (mask1 == 0) & (mask2 == 0)
        result[both_invalid] = 0

        return result

    def _build_gaussian_pyramid(
        self, image: np.ndarray
    ) -> List[np.ndarray]:
        """
        构建高斯金字塔

        每层对前一层进行高斯平滑后下采样2倍：
        Gₖ = REDUCE(Gₖ₋₁) = downsample(GaussianBlur(Gₖ₋₁))
        """
        pyramid = [image.copy()]
        current = image.copy()

        for _ in range(self.n_levels - 1):
            # cv2.pyrDown 等价于 5×5 高斯平滑 + 2倍下采样
            current = cv2.pyrDown(current)
            pyramid.append(current)

        return pyramid  # [G₀, G₁, ..., G_{n-1}]

    def _build_laplacian_pyramid(
        self, image: np.ndarray
    ) -> List[np.ndarray]:
        """
        构建拉普拉斯金字塔

        Lₖ = Gₖ - EXPAND(Gₖ₊₁)
        EXPAND: 2倍上采样 + 5×5高斯平滑（×4补偿能量）

        最顶层 Lₙ = Gₙ（保留低频基础）
        """
        GP = self._build_gaussian_pyramid(image)
        LP = []

        for k in range(len(GP) - 1):
            # 上采样下一层（恢复尺寸）
            Gk_expanded = cv2.pyrUp(GP[k + 1], dstsize=(GP[k].shape[1], GP[k].shape[0]))
            # 拉普拉斯层 = 当前层 - 展开的下一层（带通滤波）
            Lk = GP[k].astype(np.float32) - Gk_expanded.astype(np.float32)
            LP.append(Lk)

        # 最顶层直接保留（低频残差）
        LP.append(GP[-1].astype(np.float32))

        return LP  # [L₀, L₁, ..., L_{n-1}, G_n]

    def _reconstruct_from_laplacian(
        self, LP: List[np.ndarray]
    ) -> np.ndarray:
        """
        从拉普拉斯金字塔重建图像

        从最顶层开始，逐层上采样并加上对应拉普拉斯层:
        Ĝₖ₋₁ = Lₖ₋₁ + EXPAND(Ĝₖ)
        """
        # 从最顶层（最小分辨率）开始重建
        reconstructed = LP[-1].copy()

        for k in range(len(LP) - 2, -1, -1):
            # 上采样当前重建结果到下一层的尺寸
            target_h, target_w = LP[k].shape[:2]
            reconstructed = cv2.pyrUp(
                reconstructed,
                dstsize=(target_w, target_h)
            )
            # 加上对应的拉普拉斯层（还原细节）
            reconstructed = reconstructed + LP[k]

        return reconstructed


class SeamFinder:
    """
    最优缝合线查找器（用于多分辨率融合前的区域划分）

    【图切割 (Graph Cut) 寻找最优缝合线】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    将重叠区域建模为图:
    - 节点 v: 每个像素
    - 边权 e(p,q): 两图像在相邻像素处的颜色差
      C(p,q) = |I₁(p) - I₂(p)| + |I₁(q) - I₂(q)|

    最优缝合线 = 最小割 (Min-Cut)
    → 使得缝合线两侧像素颜色差最小（视觉上最自然）

    使用 Dijkstra 算法实现简化版本（沿垂直方向最优路径）:
    dp[i,j] = min_k(dp[i-1,k]) + cost[i,j]
    """

    def find_seam(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        overlap_mask: np.ndarray
    ) -> np.ndarray:
        """
        在重叠区域内找到最优缝合线

        Args:
            img1, img2: 两幅图像
            overlap_mask: 重叠区域掩码

        Returns:
            seam_mask: 缝合线掩码（img1使用seam左侧，img2使用右侧）
        """
        # 计算重叠区域的颜色差异代价图
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        cost_map = diff.mean(axis=2)  # (H, W) 颜色差异

        # 限制到重叠区域
        cost_map[overlap_mask == 0] = np.inf

        h, w = cost_map.shape

        # 动态规划求最小代价路径（竖向缝合线）
        dp = cost_map.copy()
        back_ptr = np.zeros_like(dp, dtype=np.int32)

        for i in range(1, h):
            for j in range(w):
                # 从上一行的三个邻居中选最小代价
                neighbors = []
                if j > 0:
                    neighbors.append((dp[i-1, j-1], j-1))
                neighbors.append((dp[i-1, j], j))
                if j < w-1:
                    neighbors.append((dp[i-1, j+1], j+1))

                if neighbors:
                    min_val, min_j = min(neighbors)
                    dp[i, j] += min_val if not np.isinf(min_val) else np.inf
                    back_ptr[i, j] = min_j

        # 回溯找到最优路径
        seam_col = np.zeros(h, dtype=np.int32)
        # 从最后一行找最小值位置
        last_row_finite = dp[-1].copy()
        last_row_finite[np.isinf(last_row_finite)] = np.inf
        seam_col[-1] = np.argmin(last_row_finite)

        for i in range(h-2, -1, -1):
            seam_col[i] = back_ptr[i+1, seam_col[i+1]]

        # 生成缝合线掩码（左侧使用img1，右侧使用img2）
        seam_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            seam_mask[i, :seam_col[i]] = 255

        return seam_mask
