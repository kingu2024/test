"""
全景图像拼接算法包
Panorama Image Stitching Algorithm Package

模块结构:
- feature_extraction: SIFT/ORB/AKAZE 特征提取与匹配
- homography: RANSAC 单应性矩阵估计
- warping: 柱面/球面/平面投影变换
- blending: 多分辨率拉普拉斯金字塔融合
- stitcher: 全景拼接主控流程
"""

from .stitcher import PanoramaStitcher, stitch_images
from .feature_extraction import FeatureExtractor, FeatureMatcher
from .homography import HomographyEstimator
from .warping import ImageWarper
from .blending import ImageBlender, SeamFinder

__all__ = [
    'PanoramaStitcher',
    'stitch_images',
    'FeatureExtractor',
    'FeatureMatcher',
    'HomographyEstimator',
    'ImageWarper',
    'ImageBlender',
    'SeamFinder',
]

__version__ = '1.0.0'
__author__ = 'Panorama Stitching Algorithm Implementation'
