"""
基于特征的图像对齐模块 - HDR 成像对齐算法
Feature-Based Image Alignment for HDR Imaging

【算法原理】
本模块复用全景拼接模块（panorama_stitching）中已实现的特征提取与单应性估计组件，
对不同曝光度的图像进行精确对齐，处理平移、旋转和尺度变化。

对齐流程：
1. 使用 FeatureExtractor（SIFT 或 ORB）从参考图和待对齐图中提取关键点与描述子
2. 使用 FeatureMatcher（Lowe's比值测试）筛选高质量匹配对
3. 使用 HomographyEstimator（RANSAC）鲁棒估计 3×3 单应性矩阵 H
4. 使用 cv2.warpPerspective 将待对齐图像变换到参考图坐标系

相比 MTB（中值阈值位图）对齐，基于特征的对齐可处理更大的几何形变，
适用于手持拍摄等存在旋转、缩放的场景。
"""

import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# 复用全景拼接模块的特征提取和单应性估计
from panorama_stitching.feature_extraction import FeatureExtractor, FeatureMatcher
from panorama_stitching.homography import HomographyEstimator


class FeatureAlignment:
    """
    基于特征匹配的 HDR 图像对齐器

    使用 SIFT/ORB 特征点检测与 RANSAC 单应性估计，将多曝光图像对齐到
    同一参考帧，从而消除拍摄时的相机抖动与几何形变。
    """

    def __init__(self, feature_method: str = 'SIFT', max_features: int = 1000):
        """
        初始化对齐器

        Args:
            feature_method: 特征检测算法，可选 'SIFT' 或 'ORB'
            max_features:   最大特征点数量
        """
        self.feature_method = feature_method
        self.extractor = FeatureExtractor(method=feature_method, max_features=max_features)
        self.estimator = HomographyEstimator(method='RANSAC')
        logger.info(
            f"FeatureAlignment 初始化完成: 方法={feature_method}, 最大特征数={max_features}"
        )

    def process(
        self,
        images: List[np.ndarray],
        reference_index: int = 0
    ) -> List[np.ndarray]:
        """
        将所有图像对齐到参考图像

        对每张非参考图像：
          1. 从参考图和该图中提取特征点与描述子
          2. 使用 FeatureMatcher 进行特征匹配（含 Lowe's 比值测试）
          3. 使用 HomographyEstimator 以 RANSAC 估计单应性矩阵
          4. 使用 cv2.warpPerspective 将图像变换到参考图坐标系

        Args:
            images:          输入图像列表（BGR 或灰度），至少 2 张
            reference_index: 参考图像在列表中的索引，默认为 0

        Returns:
            与输入顺序一致的对齐后图像列表；参考图像原样返回

        Raises:
            ValueError: 当输入图像数量少于 2 时
        """
        if len(images) < 2:
            raise ValueError("At least 2 images required for alignment")

        ref_image = images[reference_index]
        h, w = ref_image.shape[:2]

        # 只提取一次参考图的特征
        kp_ref, desc_ref = self.extractor.detect_and_compute(ref_image)

        matcher = FeatureMatcher(method=self.feature_method)

        aligned = []
        for i, img in enumerate(images):
            if i == reference_index:
                aligned.append(img.copy())
                continue

            # 提取当前图像的特征
            kp_cur, desc_cur = self.extractor.detect_and_compute(img)

            # 特征匹配：当前图 (query) → 参考图 (train)
            matches = matcher.match(desc_cur, desc_ref)
            logger.info(f"图像 {i}: 找到 {len(matches)} 个有效匹配")

            # 估计单应性矩阵 H（将当前图坐标映射到参考图坐标）
            H, mask = self.estimator.estimate(kp_cur, kp_ref, matches)

            if H is None:
                logger.warning(
                    f"图像 {i} 单应性估计失败，返回原始图像（未对齐）"
                )
                aligned.append(img.copy())
                continue

            # 透视变换对齐到参考图坐标系
            warped = cv2.warpPerspective(img, H, (w, h))
            aligned.append(warped)
            logger.info(f"图像 {i} 对齐完成")

        return aligned

    def process_opencv(
        self,
        images: List[np.ndarray],
        reference_index: int = 0
    ) -> List[np.ndarray]:
        """
        与 process() 等价的接口（两者均使用 OpenCV 内部实现）

        Args:
            images:          输入图像列表，至少 2 张
            reference_index: 参考图像索引，默认为 0

        Returns:
            对齐后图像列表
        """
        return self.process(images, reference_index)
