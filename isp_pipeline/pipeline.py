"""
ISP 主流程编排器 (ISP Pipeline Orchestrator)
=============================================

参考论文/资料:
- "ISP Meets Deep Learning: A Survey on Deep Learning Methods for ISP"
  ACM Computing Surveys 2024. https://dl.acm.org/doi/full/10.1145/3708516
- "A Comprehensive Survey on Image Signal Processing" arXiv:2502.05995
- Infinite-ISP: https://github.com/xx-isp/infinite-isp
- openISP: https://github.com/cruxopen/openISP

完整 ISP 流程概述：
    ┌─────────────────────────────────────────────────────────────────┐
    │                     RAW Bayer 传感器输出                          │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────┐
    │  阶段1: Bayer 域预处理                                             │
    │  BLC → BPC → LSC → AWB（Bayer域增益）                             │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────┐
    │  阶段2: 去马赛克（Bayer → RGB）                                    │
    │  Demosaicing（双线性/Malvar/AHD）                                  │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────┐
    │  阶段3: RGB 域线性处理                                             │
    │  CCM（颜色校正矩阵）                                               │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────┐
    │  阶段4: 色调处理（在线性光空间）                                    │
    │  ToneMapping → Gamma 编码                                        │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────┐
    │  阶段5: 空间域增强（在 Gamma 编码空间）                             │
    │  NoiseReduction → Sharpening                                    │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────┐
    │                  sRGB 输出图像（可保存/显示）                        │
    └─────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .modules.blc import BlackLevelCorrection
from .modules.bpc import BadPixelCorrection
from .modules.lsc import LensShadingCorrection
from .modules.awb import AutoWhiteBalance
from .modules.demosaic import Demosaicing
from .modules.ccm import ColorCorrectionMatrix
from .modules.gamma import GammaCorrection
from .modules.noise_reduction import NoiseReduction
from .modules.tone_mapping import ToneMapping
from .modules.sharpening import Sharpening


@dataclass
class ISPConfig:
    """
    ISP 流程配置参数数据类

    包含所有模块的参数设置，方便统一管理和序列化。
    """

    # ---- 传感器基本参数 ----
    # Bayer 排列模式：RGGB（最常见）, BGGR, GRBG, GBRG
    bayer_pattern: str = 'RGGB'
    # 传感器位深（决定最大像素值）：12 位传感器 = 4095
    bit_depth: int = 12

    # ---- BLC: 黑电平校正 ----
    # 黑电平值（可以是标量或4元组 (R, Gr, Gb, B)）
    black_level: float = 64.0
    # 白电平（传感器饱和值）
    white_level: float = 4095.0

    # ---- BPC: 坏点校正 ----
    # 坏点检测相对阈值（0.1~0.3）
    bpc_threshold: float = 0.2
    # 坏点校正方法：'median' 或 'mean'
    bpc_method: str = 'median'

    # ---- LSC: 镜头阴影校正 ----
    # 校正模式：'gain_map', 'polynomial', 'gaussian'
    lsc_mode: str = 'gaussian'

    # ---- AWB: 自动白平衡 ----
    # 算法：'gray_world', 'white_patch', 'perfect_reflector', 'gray_edge', 'manual'
    awb_method: str = 'gray_world'

    # ---- Demosaicing: 去马赛克 ----
    # 算法：'bilinear'（快速）, 'malvar'（中等）, 'ahd'（高质量）
    demosaic_method: str = 'malvar'

    # ---- CCM: 颜色校正矩阵 ----
    # None 使用默认 sRGB 矩阵，或提供 3×3 自定义矩阵
    ccm_matrix: Optional[np.ndarray] = None

    # ---- Tone Mapping: 色调映射 ----
    # 算法：'reinhard', 'filmic', 'aces', 'drago', 'gamma_only'
    tone_mapping_method: str = 'aces'
    # 曝光补偿（1.0=正常曝光，>1.0=过曝，<1.0=欠曝）
    exposure: float = 1.0

    # ---- Gamma 校正 ----
    # 模式：'srgb'（标准）, 'power', 'rec709'
    gamma_mode: str = 'srgb'

    # ---- Noise Reduction: 噪声消除 ----
    # 算法：'gaussian', 'bilateral', 'nlm', 'median', 'guided'
    nr_method: str = 'bilateral'
    # 空间平滑半径
    nr_sigma_spatial: float = 1.0
    # 颜色域保边强度（越小越保边）
    nr_sigma_color: float = 0.1

    # ---- Sharpening: 锐化增强 ----
    # 算法：'usm', 'laplacian', 'adaptive'
    sharp_method: str = 'usm'
    # 锐化强度（0~2，推荐 0.3~0.8）
    sharp_strength: float = 0.5
    # 锐化高斯核 sigma
    sharp_sigma: float = 1.0

    # ---- 流程控制开关 ----
    # 可以禁用某些模块（调试或特殊用途）
    enable_bpc: bool = True
    enable_lsc: bool = True
    enable_awb: bool = True
    enable_ccm: bool = True
    enable_tone_mapping: bool = True
    enable_nr: bool = True
    enable_sharpening: bool = True


class ISPPipeline:
    """
    ISP 全流程主类

    将各个 ISP 模块串联起来，提供统一的处理接口。

    用法示例:
        # 使用默认配置
        pipeline = ISPPipeline()
        rgb_output = pipeline.process(raw_bayer_image)

        # 自定义配置
        config = ISPConfig(
            bayer_pattern='RGGB',
            black_level=64,
            demosaic_method='ahd',
            tone_mapping_method='aces',
        )
        pipeline = ISPPipeline(config)
        rgb_output = pipeline.process(raw_bayer_image)
    """

    def __init__(self, config: Optional[ISPConfig] = None):
        """
        初始化 ISP 流程

        参数:
            config: ISP 配置对象，None 时使用所有默认值
        """
        if config is None:
            config = ISPConfig()

        self.config = config
        self._build_pipeline()

    def _build_pipeline(self):
        """
        根据配置构建各个处理模块

        按照 ISP 处理顺序初始化所有模块对象。
        """
        cfg = self.config

        # 1. 黑电平校正（BLC）- 始终启用，是 ISP 的基础步骤
        self.blc = BlackLevelCorrection(
            black_level=cfg.black_level,
            white_level=cfg.white_level,
            bayer_pattern=cfg.bayer_pattern,
            normalize=True,  # 归一化到 [0, 1]
        )

        # 2. 坏点校正（BPC）- 可选
        self.bpc = BadPixelCorrection(
            threshold=cfg.bpc_threshold,
            correction_method=cfg.bpc_method,
            bayer_pattern=cfg.bayer_pattern,
        ) if cfg.enable_bpc else None

        # 3. 镜头阴影校正（LSC）- 可选
        self.lsc = LensShadingCorrection(
            mode=cfg.lsc_mode,
            bayer_pattern=cfg.bayer_pattern,
        ) if cfg.enable_lsc else None

        # 4. 自动白平衡（AWB）- 可选，在 Bayer 域施加增益
        self.awb = AutoWhiteBalance(
            method=cfg.awb_method,
            bayer_pattern=cfg.bayer_pattern,
        ) if cfg.enable_awb else None

        # 5. 去马赛克（Demosaicing）- 始终启用，Bayer → RGB
        self.demosaic = Demosaicing(
            method=cfg.demosaic_method,
            bayer_pattern=cfg.bayer_pattern,
        )

        # 6. 颜色校正矩阵（CCM）- 可选
        self.ccm = ColorCorrectionMatrix(
            matrix=cfg.ccm_matrix,
        ) if cfg.enable_ccm else None

        # 7. 色调映射（Tone Mapping）- 可选，在线性空间
        self.tone_mapper = ToneMapping(
            method=cfg.tone_mapping_method,
            exposure=cfg.exposure,
        ) if cfg.enable_tone_mapping else None

        # 8. Gamma 编码（始终启用，最终输出需要 gamma 编码）
        self.gamma = GammaCorrection(
            mode=cfg.gamma_mode,
            direction='encode',
        )

        # 9. 噪声消除（NR）- 可选
        self.nr = NoiseReduction(
            method=cfg.nr_method,
            sigma_spatial=cfg.nr_sigma_spatial,
            sigma_color=cfg.nr_sigma_color,
        ) if cfg.enable_nr else None

        # 10. 锐化增强（Sharpening）- 可选
        self.sharpen = Sharpening(
            method=cfg.sharp_method,
            strength=cfg.sharp_strength,
            sigma=cfg.sharp_sigma,
            apply_to_luma=True,
        ) if cfg.enable_sharpening else None

    def process(
        self,
        raw: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        执行完整 ISP 流程，将 RAW Bayer 图像转换为 sRGB 输出

        处理顺序:
            raw → BLC → BPC → LSC → AWB → Demosaic → CCM → ToneMap → Gamma → NR → Sharpen → output

        参数:
            raw: 输入 RAW Bayer 图像，形状 (H, W)
                 数值类型可以是整型（如 uint12, uint16）或浮点型
                 值域取决于传感器位深（如 12 位则为 0~4095）
            verbose: 是否打印每个模块的处理时间（调试用）

        返回:
            输出 sRGB 图像，形状 (H, W, 3)，uint8 类型（值域 0~255）
        """
        timings: Dict[str, float] = {}

        def timed_step(name: str, func, img):
            """计时执行一个处理步骤"""
            t0 = time.time()
            result = func(img)
            elapsed = time.time() - t0
            timings[name] = elapsed
            if verbose:
                print(f"  [{name:20s}] {elapsed*1000:.1f} ms | "
                      f"shape={result.shape}, "
                      f"range=[{result.min():.3f}, {result.max():.3f}]")
            return result

        if verbose:
            print(f"\n{'='*60}")
            print(f"ISP Pipeline Processing")
            print(f"  Input: shape={raw.shape}, dtype={raw.dtype}")
            print(f"  Config: bayer={self.config.bayer_pattern}, "
                  f"demosaic={self.config.demosaic_method}, "
                  f"tm={self.config.tone_mapping_method}")
            print(f"{'='*60}")

        # ── 阶段1: Bayer 域预处理 ──────────────────────────────
        # 步骤1: 黑电平校正 → 输出 float32 [0,1] 的 Bayer 图
        img = timed_step('BLC', self.blc.process, raw)

        # 步骤2: 坏点校正（可选）
        if self.bpc is not None:
            img = timed_step('BPC', self.bpc.process, img)

        # 步骤3: 镜头阴影校正（可选）
        if self.lsc is not None:
            img = timed_step('LSC', self.lsc.process, img)

        # 步骤4: 自动白平衡在 Bayer 域施加增益（可选）
        if self.awb is not None:
            img = timed_step('AWB', self.awb.process_bayer, img)

        # ── 阶段2: 去马赛克（Bayer → RGB）─────────────────────
        # 步骤5: 去马赛克 → 输出 (H, W, 3) RGB float32 [0,1]
        img = timed_step('Demosaicing', self.demosaic.process, img)

        # ── 阶段3: RGB 域线性处理 ──────────────────────────────
        # 步骤6: 颜色校正矩阵（可选，必须在线性光空间）
        if self.ccm is not None:
            img = timed_step('CCM', self.ccm.process, img)

        # ── 阶段4: 色调处理（在线性光空间）────────────────────
        # 步骤7: 色调映射（可选，压缩 HDR → LDR）
        if self.tone_mapper is not None:
            img = timed_step('ToneMapping', self.tone_mapper.process, img)

        # 步骤8: Gamma 编码（线性 → 显示域，必须最后执行）
        img = timed_step('Gamma', self.gamma.process, img)

        # ── 阶段5: 空间域增强（在 Gamma 编码空间）──────────────
        # 步骤9: 噪声消除（可选，在 gamma 编码空间效果更符合视觉感知）
        if self.nr is not None:
            img = timed_step('NoiseReduction', self.nr.process, img)

        # 步骤10: 锐化增强（可选，最后执行以强化最终效果）
        if self.sharpen is not None:
            img = timed_step('Sharpening', self.sharpen.process, img)

        # ── 最终输出转换 ──────────────────────────────────────
        # 将 [0, 1] float32 转换为 [0, 255] uint8（标准 sRGB 图像格式）
        output = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        if verbose:
            total = sum(timings.values())
            print(f"{'─'*60}")
            print(f"  Total: {total*1000:.1f} ms | Output: shape={output.shape}")
            print(f"{'='*60}\n")

        return output

    def process_to_float(
        self,
        raw: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        执行 ISP 流程，返回 float32 格式输出（值域 [0, 1]）

        适用于需要后续数值处理（如深度学习训练数据）的场景。

        参数:
            raw: 输入 RAW Bayer 图像
            verbose: 是否打印调试信息

        返回:
            float32 sRGB 图像，形状 (H, W, 3)，值域 [0, 1]
        """
        uint8_output = self.process(raw, verbose=verbose)
        return uint8_output.astype(np.float32) / 255.0

    def get_intermediate_results(
        self,
        raw: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        获取 ISP 各阶段的中间结果（用于调试和可视化）

        参数:
            raw: 输入 RAW Bayer 图像

        返回:
            字典，键为步骤名称，值为该步骤输出的图像
        """
        results = {'input_raw': raw.copy()}

        # BLC
        img = self.blc.process(raw)
        results['after_blc'] = img.copy()

        # BPC
        if self.bpc is not None:
            img = self.bpc.process(img)
            results['after_bpc'] = img.copy()

        # LSC
        if self.lsc is not None:
            img = self.lsc.process(img)
            results['after_lsc'] = img.copy()

        # AWB (Bayer)
        if self.awb is not None:
            img = self.awb.process_bayer(img)
            results['after_awb_bayer'] = img.copy()

        # Demosaicing
        img = self.demosaic.process(img)
        results['after_demosaic'] = np.clip(img * 255, 0, 255).astype(np.uint8)

        # CCM
        if self.ccm is not None:
            img = self.ccm.process(img)
            results['after_ccm'] = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Tone Mapping
        if self.tone_mapper is not None:
            img = self.tone_mapper.process(img)
            results['after_tone_mapping'] = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Gamma
        img = self.gamma.process(img)
        results['after_gamma'] = np.clip(img * 255, 0, 255).astype(np.uint8)

        # NR
        if self.nr is not None:
            img = self.nr.process(img)
            results['after_nr'] = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Sharpening
        if self.sharpen is not None:
            img = self.sharpen.process(img)
            results['after_sharpen'] = np.clip(img * 255, 0, 255).astype(np.uint8)

        return results

    def update_config(self, **kwargs):
        """
        动态更新配置参数并重建流程

        用法：
            pipeline.update_config(
                tone_mapping_method='filmic',
                exposure=1.2,
                nr_method='guided',
            )
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"未知配置参数: {key}")

        # 重建流程以应用新配置
        self._build_pipeline()
