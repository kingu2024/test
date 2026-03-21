"""
特征提取模块 - 图像全景拼接算法核心组件
Feature Extraction Module for Panorama Stitching

【算法原理】
特征提取是全景拼接的第一步，目标是从图像中提取出具有尺度、旋转、光照不变性的特征点。

本模块实现了以下特征检测器：
1. SIFT (Scale-Invariant Feature Transform) - 尺度不变特征变换
2. ORB  (Oriented FAST and Rotated BRIEF)  - 定向FAST旋转BRIEF特征
3. SuperPoint (基于深度学习的特征检测，可选)

【SIFT 数学原理】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) 构建高斯尺度空间 (DoG金字塔)
   L(x, y, σ) = G(x, y, σ) * I(x, y)
   其中 G(x, y, σ) = (1/(2πσ²)) · exp(-(x²+y²)/(2σ²))

   差分高斯 (DoG):
   D(x, y, σ) = L(x, y, kσ) - L(x, y, σ)
   ≈ (k-1)σ² · ∇²G * I  (近似LoG)

2) 极值点检测
   在 DoG 空间的 3D 邻域（26个邻居）中寻找局部极值

3) 关键点精确定位 (泰勒展开)
   D(x) ≈ D + (∂D/∂x)ᵀ·x + (1/2)·xᵀ·(∂²D/∂x²)·x
   求导令 ∂D/∂x = 0，得到精确位置:
   x̂ = -(∂²D/∂x²)⁻¹ · (∂D/∂x)

4) 方向分配 (梯度直方图)
   m(x,y) = √[(L(x+1,y)-L(x-1,y))² + (L(x,y+1)-L(x,y-1))²]
   θ(x,y) = arctan[(L(x,y+1)-L(x,y-1)) / (L(x+1,y)-L(x-1,y))]

5) 128维描述子计算
   在16×16邻域内划分4×4子区域，每个子区域8方向梯度直方图
   描述子维度: 4 × 4 × 8 = 128

【ORB 数学原理】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) FAST角点检测 + Harris评分排序
   V(p, t) = {p' ∈ circle | I(p') > I(p)+t 或 I(p') < I(p)-t}
   Harris得分: R = det(M) - k·trace(M)²
   M = [Σ(Ix²) Σ(IxIy); Σ(IxIy) Σ(Iy²)]

2) 矩计算确定主方向 (保证旋转不变性)
   m_{p,q} = Σ_{x,y} x^p · y^q · I(x,y)
   质心: C = (m_{1,0}/m_{0,0}, m_{0,1}/m_{0,0})
   方向: θ = arctan(m_{0,1}/m_{1,0})

3) rBRIEF 描述子 (旋转后的BRIEF)
   τ(p; x, y) = { 1, if I(p_x) < I(p_y)
                 { 0, otherwise
   旋转后用 θ 对采样对进行旋转：
   S_θ = R_θ · S  (R_θ 为旋转矩阵)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    多算法特征提取器
    支持 SIFT / ORB / AKAZE 三种特征描述算法
    """

    def __init__(self, method: str = 'SIFT', max_features: int = 2000):
        """
        初始化特征提取器

        Args:
            method: 特征检测方法，可选 'SIFT', 'ORB', 'AKAZE'
            max_features: 最大特征点数量
        """
        self.method = method.upper()
        self.max_features = max_features
        self.detector = self._create_detector()
        logger.info(f"特征提取器初始化完成: 方法={method}, 最大特征数={max_features}")

    def _create_detector(self):
        """
        根据方法名称创建对应的特征检测器

        【各算法性能对比】
        SIFT:  精度高，计算较慢，128维描述子，尺度/旋转不变
        ORB:   速度快，精度略低，256位二进制描述子，旋转不变
        AKAZE: 非线性扩散滤波，抗噪声，适合纹理丰富场景
        """
        if self.method == 'SIFT':
            # SIFT: 尺度不变特征变换
            # nfeatures: 保留最佳特征数
            # nOctaveLayers: 每个octave的层数
            # contrastThreshold: 低对比度过滤阈值
            # edgeThreshold: 边缘响应过滤阈值
            # sigma: 第0层的高斯模糊参数
            return cv2.SIFT_create(
                nfeatures=self.max_features,
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
        elif self.method == 'ORB':
            # ORB: 定向FAST旋转BRIEF
            # scaleFactor: 图像金字塔缩放比例
            # nlevels: 金字塔层数
            # edgeThreshold: 不检测特征的边缘宽度
            # WTA_K: 产生每个oriented BRIEF描述符元素的点数
            return cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31
            )
        elif self.method == 'AKAZE':
            # AKAZE: 加速的KAZE特征 (非线性扩散空间)
            # descriptor_type: DESCRIPTOR_MLDB (多尺度局部差异二进制)
            # descriptor_size: 描述子大小(0=全尺寸)
            # descriptor_channels: 描述子通道数
            return cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=0.001
            )
        else:
            raise ValueError(f"不支持的特征检测方法: {self.method}。请选择 SIFT/ORB/AKAZE")

    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        检测关键点并计算描述子

        Args:
            image: 输入图像 (BGR或灰度)

        Returns:
            keypoints: 关键点列表，每个KeyPoint包含:
                       - pt: (x, y) 坐标
                       - size: 特征点尺度
                       - angle: 主方向角度 [0, 360)
                       - response: 响应强度
                       - octave: 所在金字塔层级
            descriptors: 描述子矩阵 (N × D)
                        SIFT: N × 128 float32
                        ORB:  N × 32  uint8 (位描述子)
        """
        # 转换为灰度图像（特征检测在灰度空间进行）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 自适应直方图均衡化 (CLAHE) - 增强对比度，改善特征检测效果
        # 原理: 限制对比度的局部直方图均衡化，避免过度放大噪声
        # clipLimit: 对比度限制阈值
        # tileGridSize: 局部区域大小
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # 检测关键点并计算描述子
        keypoints, descriptors = self.detector.detectAndCompute(gray_enhanced, None)

        if descriptors is None or len(keypoints) == 0:
            logger.warning("未检测到特征点！图像可能纹理不足或质量较低。")
            return [], None

        logger.info(f"检测到 {len(keypoints)} 个特征点")
        return keypoints, descriptors

    def visualize_keypoints(
        self, image: np.ndarray, keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """
        可视化关键点
        - 圆圈大小表示特征尺度
        - 线段方向表示主方向

        Args:
            image: 原始图像
            keypoints: 关键点列表

        Returns:
            带关键点可视化的图像
        """
        vis = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return vis


class FeatureMatcher:
    """
    特征点匹配器

    【匹配算法原理】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1) 暴力匹配 (BFMatcher):
       对每个描述子，遍历所有候选描述子，找最近邻
       - 浮点描述子(SIFT): L2距离 d = √Σ(a_i - b_i)²
       - 二进制描述子(ORB): 汉明距离 d = popcount(a XOR b)

    2) FLANN快速近似最近邻 (用于SIFT加速):
       使用KD-Tree或层次聚类索引进行近似搜索
       时间复杂度: O(N·log(N)) vs 暴力匹配的 O(N²)

    3) Lowe's比值测试 (去除歧义匹配):
       对每个点找2个最近邻，仅保留满足以下条件的匹配:
       d₁/d₂ < ratio_threshold  (通常 0.75)
       含义: 最近邻距离 << 次近邻距离，表示匹配唯一性高
    """

    def __init__(self, method: str = 'SIFT', ratio_threshold: float = 0.75):
        """
        Args:
            method: 特征类型，影响距离度量选择
            ratio_threshold: Lowe's比值测试阈值 (越小越严格)
        """
        self.method = method.upper()
        self.ratio_threshold = ratio_threshold
        self.matcher = self._create_matcher()

    def _create_matcher(self):
        """创建匹配器"""
        if self.method in ['SIFT', 'AKAZE']:
            # SIFT/AKAZE 使用 FLANN + L2 距离
            # FLANN参数: 使用KD-Tree索引，trees=5
            FLANN_INDEX_KDTREE = 1
            index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
            search_params = {'checks': 50}  # 检查次数越多越精确但越慢
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # ORB 使用暴力匹配 + 汉明距离
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        执行特征匹配并应用Lowe's比值测试

        【Lowe's比值测试数学推导】
        设 m₁ = 最近邻, m₂ = 次近邻
        比值: r = dist(q, m₁) / dist(q, m₂)
        若 r < threshold (0.75): 接受匹配 (最近邻比次近邻明显更近)
        若 r ≥ threshold:        拒绝匹配 (匹配存在歧义)

        原始论文建议: threshold = 0.8 (95%正确率)
        实际使用通常取 0.7-0.75 以获得更高精度

        Args:
            desc1, desc2: 两幅图像的描述子矩阵

        Returns:
            过滤后的匹配列表
        """
        if desc1 is None or desc2 is None:
            return []

        # KNN匹配，k=2 找两个最近邻用于比值测试
        try:
            # FLANN需要float32
            if self.method in ['SIFT', 'AKAZE']:
                desc1 = desc1.astype(np.float32)
                desc2 = desc2.astype(np.float32)
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            logger.error(f"匹配失败: {e}")
            return []

        # 应用Lowe's比值测试筛选优质匹配
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # 核心判断: 最佳匹配距离 / 次佳匹配距离 < 阈值
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        logger.info(
            f"原始匹配: {len(raw_matches)} → 比值测试后: {len(good_matches)} "
            f"(保留率: {len(good_matches)/max(len(raw_matches),1)*100:.1f}%)"
        )
        return good_matches

    def visualize_matches(
        self,
        img1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_display: int = 50
    ) -> np.ndarray:
        """
        可视化匹配对 (绘制连线)
        """
        # 随机颜色显示匹配连线，便于视觉检查
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2,
            matches[:max_display], None,
            matchColor=(0, 255, 0),      # 绿色连线
            singlePointColor=(255, 0, 0), # 蓝色孤立点
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return vis
