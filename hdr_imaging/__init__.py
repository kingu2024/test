"""
HDR 高动态范围成像算法库 / HDR Imaging Algorithm Library

模块结构:
- alignment: MTB / 特征点图像对齐
- calibration: Debevec / Robertson 相机响应函数标定
- merge: 多曝光 HDR 辐射图合并
- tone_mapping: 10种色调映射算法（全局/局部/感知驱动）
- exposure_fusion: Mertens 多曝光融合
- single_image: 单张图像 HDR 增强
- hdr_pipeline: HDR 处理主控流水线
"""

from .hdr_pipeline import HDRPipeline, HDRResult
from .alignment import MTBAlignment, FeatureAlignment
from .calibration import DebevecCalibration, RobertsonCalibration
from .merge import HDRMerge
from .tone_mapping import (ReinhardGlobal, ReinhardLocal, DragoToneMap,
                           DurandToneMap, FattalToneMap, AdaptiveLog,
                           ACESToneMap, FilmicToneMap, MantiukToneMap,
                           HistogramToneMap)
from .exposure_fusion import MertensFusion
from .single_image import SingleImageHDR

__all__ = ['HDRPipeline', 'HDRResult',
           'MTBAlignment', 'FeatureAlignment',
           'DebevecCalibration', 'RobertsonCalibration',
           'HDRMerge',
           'ReinhardGlobal', 'ReinhardLocal', 'DragoToneMap',
           'DurandToneMap', 'FattalToneMap', 'AdaptiveLog',
           'ACESToneMap', 'FilmicToneMap', 'MantiukToneMap',
           'HistogramToneMap',
           'MertensFusion', 'SingleImageHDR']
__version__ = '1.0.0'
__author__ = 'HDR Imaging Algorithm Implementation'
