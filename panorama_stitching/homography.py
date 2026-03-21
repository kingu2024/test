"""
单应性矩阵估计模块 - 图像全景拼接
Homography Estimation Module

【单应性矩阵原理】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
单应性矩阵 H 描述了两个平面之间的投影变换关系。
当两幅图像拍摄同一平面或相机绕光心旋转时，点的对应关系满足:

    [x']     [h₁₁ h₁₂ h₁₃] [x]
    [y'] = H·[h₂₁ h₂₂ h₂₃]·[y]
    [w']     [h₃₁ h₃₂ h₃₃] [w]

齐次坐标形式: p' = H · p
实际坐标:     x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
              y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)

H 有 9 个参数，但因为齐次性（可缩放），实际自由度 = 8

【直接线性变换(DLT)求解单应性矩阵】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每对匹配点 (x,y) ↔ (x',y') 给出两个方程:

    [x  y  1  0  0  0  -x'x  -x'y  -x'] · h = 0
    [0  0  0  x  y  1  -y'x  -y'y  -y'] · h = 0

其中 h = [h₁₁, h₁₂, h₁₃, h₂₁, h₂₂, h₂₃, h₃₁, h₃₂, h₃₃]ᵀ

n 对点构成线性方程组: A · h = 0  (A 为 2n×9 矩阵)
通过 SVD 分解求解: A = UΣVᵀ
解为 V 的最后一列（对应最小奇异值）

最少需要 4 对非共线点求解 8 个自由度

【RANSAC 鲁棒估计原理】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
由于特征匹配存在误匹配（外点），需要 RANSAC 鲁棒估计:

RANSAC 迭代:
1. 随机采样最小点集 (4对点)
2. 用 DLT 估计候选 H
3. 计算所有点的重投影误差:
   e_i = ||p'_i - H·p_i||  (Sampson误差更精确)
4. 统计内点数 (误差 < threshold 的点)
5. 若内点数 > 最佳记录，更新最优解

所需迭代次数 (置信度 p 下):
   N = log(1-p) / log(1-(1-ε)ˢ)
   p: 期望置信度 (0.99)
   ε: 外点比例估计
   s: 最小样本数 (4)

最终用所有内点重新拟合 H（最小二乘）

【归一化DLT】
为提高数值稳定性，先对坐标进行各向同性缩放:
   T: 将点集平移到原点，平均距离缩放至 √2
   归一化: p̃ = T · p,  p̃' = T' · p'
   求解: Ã · h̃ = 0
   还原: H = T'⁻¹ · H̃ · T
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class HomographyEstimator:
    """
    单应性矩阵估计器
    集成 RANSAC 鲁棒估计和内点精化
    """

    def __init__(
        self,
        method: str = 'RANSAC',
        ransac_reproj_threshold: float = 4.0,
        confidence: float = 0.995,
        max_iters: int = 10000
    ):
        """
        初始化单应性估计器

        Args:
            method: 估计方法 'RANSAC' 或 'LMEDS'
            ransac_reproj_threshold: RANSAC 内点判定阈值（像素）
                                    越小越严格，通常取 3-5
            confidence: RANSAC 期望置信度 (0-1)
            max_iters: RANSAC 最大迭代次数
        """
        self.method = cv2.RANSAC if method == 'RANSAC' else cv2.LMEDS
        self.threshold = ransac_reproj_threshold
        self.confidence = confidence
        self.max_iters = max_iters

    def estimate(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        min_matches: int = 10
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从匹配点对估计单应性矩阵

        【算法流程】
        1. 从匹配中提取对应点坐标
        2. 坐标归一化（提高DLT数值稳定性）
        3. RANSAC 鲁棒估计 H
        4. 验证 H 的有效性

        Args:
            kp1, kp2: 两幅图像的关键点
            matches: 匹配列表
            min_matches: 最少有效匹配数（少于此数则估计失败）

        Returns:
            H: 3×3 单应性矩阵，估计失败返回 None
            mask: 内点掩码（1=内点，0=外点）
        """
        if len(matches) < min_matches:
            logger.warning(f"匹配点数不足 ({len(matches)} < {min_matches})，无法估计单应性矩阵")
            return None, None

        # ── 步骤1: 提取匹配点坐标 ────────────────────────────────
        # src_pts: 源图像点坐标 (N, 1, 2)
        # dst_pts: 目标图像点坐标 (N, 1, 2)
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # ── 步骤2: RANSAC 鲁棒单应性估计 ─────────────────────────
        # cv2.findHomography 内部流程:
        #   a) RANSAC 迭代随机采样4对点
        #   b) 用 DLT 估计候选单应性矩阵
        #   c) 计算 Sampson 误差（比点对点误差更精确）
        #      Sampson误差: e = (x'Hx)² / [(Hx)₁² + (Hx)₂² + (H'x')₁² + (H'x')₂²]
        #   d) 统计误差 < threshold 的内点
        #   e) 用所有内点重新估计 H（最小二乘）
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=self.method,
            ransacReprojThreshold=self.threshold,
            confidence=self.confidence,
            maxIters=self.max_iters
        )

        if H is None:
            logger.warning("单应性矩阵估计失败")
            return None, None

        # ── 步骤3: 验证单应性矩阵的有效性 ───────────────────────
        # 检查行列式避免退化变换
        det = np.linalg.det(H)
        if abs(det) < 1e-6 or abs(det) > 1e6:
            logger.warning(f"单应性矩阵退化 (det={det:.2e})，可能图像重叠区域太小")
            return None, None

        # 统计内点比例
        n_inliers = int(mask.sum())
        inlier_ratio = n_inliers / len(matches)
        logger.info(
            f"单应性估计成功: 内点 {n_inliers}/{len(matches)} "
            f"(比率: {inlier_ratio:.2%}), det={det:.4f}"
        )

        return H, mask

    def decompose_homography(
        self, H: np.ndarray, K: np.ndarray
    ) -> dict:
        """
        从单应性矩阵分解旋转和平移（针对平面场景）

        【原理】
        对于平面场景（或纯旋转），单应性与相机参数的关系为:
        H = K · [r₁ r₂ t] · K⁻¹
        其中 [r₁ r₂] 是旋转矩阵的前两列，t 是平移向量

        分解步骤:
        Ĥ = K⁻¹ · H · K  (归一化单应性)
        对 Ĥ 进行 SVD: Ĥ = U·Σ·Vᵀ
        旋转矩阵: R = U·Vᵀ (保证行列式为+1)

        Args:
            H: 单应性矩阵
            K: 相机内参矩阵

        Returns:
            包含旋转矩阵、平移向量的字典（可能有4个解）
        """
        # 使用 OpenCV 的分解函数（存在4个候选解，需要通过可见性约束筛选）
        num_solutions, rotations, translations, normals = \
            cv2.decomposeHomographyMat(H, K)

        return {
            'num_solutions': num_solutions,
            'rotations': rotations,
            'translations': translations,
            'normals': normals
        }

    @staticmethod
    def compute_reprojection_error(
        H: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> np.ndarray:
        """
        计算重投影误差

        【重投影误差公式】
        对每对匹配点 (p_i, p'_i):
        e_i = ||p'_i - proj(H, p_i)||₂

        其中 proj(H, p_i) = (H · p̃_i) / (H · p̃_i)_w  (透视除法)

        Args:
            H: 单应性矩阵
            src_pts: 源点 (N, 2)
            dst_pts: 目标点 (N, 2)

        Returns:
            每个点的重投影误差数组 (N,)
        """
        n = len(src_pts)
        # 转换为齐次坐标: (x, y) → (x, y, 1)
        src_h = np.hstack([src_pts, np.ones((n, 1))])  # N×3

        # 应用单应性变换
        dst_pred_h = (H @ src_h.T).T  # N×3

        # 透视除法（齐次 → 笛卡尔坐标）
        w = dst_pred_h[:, 2:3]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)  # 避免除零
        dst_pred = dst_pred_h[:, :2] / w

        # 计算欧氏距离误差
        errors = np.sqrt(np.sum((dst_pts - dst_pred) ** 2, axis=1))
        return errors


class GlobalAlignmentOptimizer:
    """
    全局对齐优化器 - 多图像全景拼接的图像图优化

    【原理：捆集调整 (Bundle Adjustment)】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    当拼接多张（>2）图像时，逐对估计单应性会累积误差。
    捆集调整同时优化所有相机参数以最小化全局重投影误差:

    最小化: E = Σ_{ij} Σ_k ||p'_{ijk} - proj(H_{ij}, p_{ijk})||²

    其中:
    - H_{ij}: 图像i到图像j的变换
    - p_{ijk}: 第k个匹配点对
    - proj: 透视投影函数

    用 Levenberg-Marquardt 算法迭代求解:
    (JᵀJ + λI)·δ = Jᵀ·e
    其中 J 为雅可比矩阵，e 为残差向量，λ 为阻尼参数

    参考: OpenCV Stitching模块的 BundleAdjuster 实现
    """

    def __init__(self, n_images: int):
        """
        Args:
            n_images: 待拼接图像数量
        """
        self.n_images = n_images
        # 存储图像间匹配关系 (图邻接表)
        self.matches_graph = {}  # {(i,j): (H, mask, n_inliers)}

    def add_match(
        self, i: int, j: int,
        H: np.ndarray, mask: np.ndarray, n_inliers: int
    ):
        """添加图像对的匹配结果"""
        self.matches_graph[(i, j)] = (H, mask, n_inliers)
        # 添加逆变换
        try:
            H_inv = np.linalg.inv(H)
            self.matches_graph[(j, i)] = (H_inv, mask, n_inliers)
        except np.linalg.LinAlgError:
            pass

    def find_spanning_tree(self) -> List[Tuple[int, int]]:
        """
        用最大生成树（按内点数排序）确定图像拼接顺序

        选择内点最多的匹配边构建生成树，确保每步拼接质量最优
        时间复杂度: O(E·log(E)) 排序 + O(E·α(V)) 并查集
        """
        # 按内点数降序排序所有匹配边
        edges = [
            (n_inliers, i, j, H)
            for (i, j), (H, mask, n_inliers) in self.matches_graph.items()
            if i < j
        ]
        edges.sort(reverse=True)

        # Kruskal 最大生成树（并查集）
        parent = list(range(self.n_images))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        spanning_tree = []
        for n_inliers, i, j, H in edges:
            if find(i) != find(j):
                union(i, j)
                spanning_tree.append((i, j))

        return spanning_tree
