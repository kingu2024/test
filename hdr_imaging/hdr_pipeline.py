"""
HDR 处理主控流水线
HDR Processing Pipeline Controller

串联各子模块完成完整 HDR 流程:
输入多曝光图像 → 对齐 → CRF标定 → HDR合并 → 色调映射 → LDR输出

也支持替代路径:
- 多曝光融合 (Mertens): 输入图像 → 对齐 → 融合 → LDR输出
- 单张图像HDR: 输入单张LDR → CLAHE增强 → 输出
"""

from collections import namedtuple
import numpy as np
import logging

from .alignment import MTBAlignment, FeatureAlignment
from .calibration import DebevecCalibration, RobertsonCalibration
from .merge import HDRMerge
from .tone_mapping import (ReinhardGlobal, ReinhardLocal, DragoToneMap,
                           DurandToneMap, FattalToneMap, AdaptiveLog,
                           ACESToneMap, FilmicToneMap, MantiukToneMap,
                           HistogramToneMap)
from .exposure_fusion import MertensFusion
from .single_image import SingleImageHDR

logger = logging.getLogger(__name__)

HDRResult = namedtuple('HDRResult', ['ldr_result', 'hdr_radiance_map', 'response_curve'])

TONE_MAPPING_METHODS = {
    'reinhard_global': ReinhardGlobal,
    'reinhard_local': ReinhardLocal,
    'drago': DragoToneMap,
    'durand': DurandToneMap,
    'fattal': FattalToneMap,
    'adaptive_log': AdaptiveLog,
    'aces': ACESToneMap,
    'filmic': FilmicToneMap,
    'mantiuk': MantiukToneMap,
    'histogram': HistogramToneMap,
}


class HDRPipeline:
    """HDR 处理主控流水线 / HDR Processing Pipeline Controller.

    串联对齐、标定、合并、色调映射等子模块完成完整 HDR 处理流程。
    """

    def __init__(self, align_method='mtb', calibration_method='debevec',
                 tone_mapping_method='reinhard_global', use_opencv=False):
        """初始化 HDR 流水线 / Initialize HDR pipeline.

        Args:
            align_method: 对齐方法，'mtb' 或 'feature'。
            calibration_method: CRF 标定方法，'debevec' 或 'robertson'。
            tone_mapping_method: 色调映射方法，见 TONE_MAPPING_METHODS 键。
            use_opencv: 若为 True 则优先使用 OpenCV 实现。
        """
        valid_align = {'mtb', 'feature'}
        if align_method not in valid_align:
            raise ValueError(
                f"Invalid align_method '{align_method}'. Valid options: {sorted(valid_align)}"
            )

        valid_calibration = {'debevec', 'robertson'}
        if calibration_method not in valid_calibration:
            raise ValueError(
                f"Invalid calibration_method '{calibration_method}'. "
                f"Valid options: {sorted(valid_calibration)}"
            )

        if tone_mapping_method not in TONE_MAPPING_METHODS:
            raise ValueError(
                f"Invalid tone_mapping_method '{tone_mapping_method}'. "
                f"Valid options: {sorted(TONE_MAPPING_METHODS.keys())}"
            )

        # Alignment
        if align_method == 'mtb':
            self.aligner = MTBAlignment()
        else:
            self.aligner = FeatureAlignment()

        # Calibration
        if calibration_method == 'debevec':
            self.calibrator = DebevecCalibration()
        else:
            self.calibrator = RobertsonCalibration()

        # Merge
        self.merger = HDRMerge()

        # Tone mapping
        self.tone_mapper = TONE_MAPPING_METHODS[tone_mapping_method]()

        # Alternative paths
        self.fusion = MertensFusion()
        self.single_hdr = SingleImageHDR()

        self.use_opencv = use_opencv

    def process(self, images, exposure_times):
        """执行完整 HDR 流程 / Execute full HDR pipeline.

        Args:
            images: 多曝光输入图像列表 (uint8, BGR)。
            exposure_times: 各图像对应曝光时间列表。

        Returns:
            HDRResult(ldr_result, hdr_radiance_map, response_curve)
        """
        if len(images) < 2:
            raise ValueError("At least 2 images required for HDR processing")
        if len(images) != len(exposure_times):
            raise ValueError("Number of images must match number of exposure times")

        # Step 1: Align images
        logger.info("Step 1: Aligning %d images...", len(images))
        if self.use_opencv:
            aligned = self.aligner.process_opencv(images)
        else:
            aligned = self.aligner.process(images)
        logger.info("Alignment complete.")

        # Step 2: Calibrate camera response function
        logger.info("Step 2: Calibrating CRF...")
        if self.use_opencv:
            response_curve = self.calibrator.process_opencv(aligned, exposure_times)
        else:
            response_curve = self.calibrator.process(aligned, exposure_times)
        logger.info("CRF calibration complete.")

        # Step 3: Merge exposures into HDR radiance map
        logger.info("Step 3: Merging exposures into HDR radiance map...")
        if self.use_opencv:
            hdr_radiance_map = self.merger.process_opencv(aligned, exposure_times, response_curve)
        else:
            hdr_radiance_map = self.merger.process(aligned, exposure_times, response_curve)
        logger.info("HDR merge complete.")

        # Step 4: Tone map to LDR
        logger.info("Step 4: Applying tone mapping...")
        if self.use_opencv:
            ldr_result = self.tone_mapper.process_opencv(hdr_radiance_map)
        else:
            ldr_result = self.tone_mapper.process(hdr_radiance_map)
        logger.info("Tone mapping complete.")

        return HDRResult(ldr_result, hdr_radiance_map, response_curve)

    def exposure_fusion(self, images):
        """多曝光融合 (Mertens) / Multi-exposure fusion via Mertens.

        Args:
            images: 多曝光输入图像列表 (uint8, BGR)。

        Returns:
            融合后的 LDR 图像 (uint8)。
        """
        if len(images) < 2:
            raise ValueError("At least 2 images required for exposure fusion")

        logger.info("Exposure fusion: aligning %d images...", len(images))
        if self.use_opencv:
            aligned = self.aligner.process_opencv(images)
        else:
            aligned = self.aligner.process(images)

        logger.info("Exposure fusion: applying Mertens fusion...")
        if self.use_opencv:
            result = self.fusion.process_opencv(aligned)
        else:
            result = self.fusion.process(aligned)

        if result.dtype != np.uint8:
            result = np.clip(result * 255, 0, 255).astype(np.uint8)

        logger.info("Exposure fusion complete.")
        return result

    def single_image_hdr(self, image):
        """单张图像 HDR 增强 / Single-image HDR enhancement via CLAHE.

        Args:
            image: 输入 LDR 图像 (uint8, BGR)。

        Returns:
            增强后的图像 (uint8)。
        """
        logger.info("Single-image HDR enhancement...")
        if self.use_opencv:
            result = self.single_hdr.process_opencv(image)
        else:
            result = self.single_hdr.process(image)

        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        logger.info("Single-image HDR complete.")
        return result
