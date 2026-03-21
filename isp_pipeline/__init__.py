"""
图像信号处理器 (ISP) 全流程算法包
=====================================

参考论文:
- "ISP Meets Deep Learning: A Survey on Deep Learning Methods for Image Signal Processing"
  ACM Computing Surveys, 2024. https://dl.acm.org/doi/full/10.1145/3708516
- "A Comprehensive Survey on Image Signal Processing"
  arXiv:2502.05995, 2025. https://arxiv.org/pdf/2502.05995
- "AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection"
  NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/
- "RealCamNet: An End-to-End Real-World Camera Imaging Pipeline"
  ACM MM 2024. https://kepengxu.github.io/projects/realcamnet/realcamnet.pdf

ISP 全流程处理顺序 (Bayer 域 → RGB 域 → YUV 域):
1. BLC  - 黑电平校正 (Black Level Correction)
2. BPC  - 坏点校正 (Bad Pixel Correction)
3. LSC  - 镜头阴影校正 (Lens Shading Correction)
4. AWB  - 自动白平衡增益 (Auto White Balance Gain)
5. DM   - 去马赛克/色彩插值 (Demosaicing)
6. CCM  - 颜色校正矩阵 (Color Correction Matrix)
7. GC   - Gamma 校正 (Gamma Correction)
8. NR   - 噪声消除 (Noise Reduction)
9. TM   - 色调映射 (Tone Mapping)
10. SE  - 锐化增强 (Sharpening Enhancement)
"""

from .pipeline import ISPPipeline

__version__ = "1.0.0"
__all__ = ["ISPPipeline"]
