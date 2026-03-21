"""
图像变形与投影模块 - 图像全景拼接
Image Warping & Projection Module

【投影变换类型】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 平面投影 (Planar):    直接应用单应性变换，适合小视角
2. 柱面投影 (Cylindrical): 将图像映射到柱面，适合水平拼接
3. 球面投影 (Spherical):  将图像映射到球面，适合360°全景

【柱面投影数学推导】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
给定相机内参 K = [f  0  cx; 0  f  cy; 0  0  1]
（假设无畸变，像素为正方形，fx=fy=f）

像素坐标 → 归一化相机坐标:
    x_n = (u - cx) / f
    y_n = (v - cy) / f

归一化相机坐标 → 单位球面坐标:
    P = (x_n, y_n, 1) / ||（x_n, y_n, 1)||

球面坐标 → 柱面坐标 (θ, h):
    θ = arctan(x_n / 1) = arctan(x_n)  [水平角]
    h = y_n / √(x_n² + 1)               [垂直高度]

柱面图像坐标:
    u_cyl = f · θ + cx
    v_cyl = f · h + cy

逆变换（柱面 → 平面，用于反向映射避免空洞）:
    θ = (u_cyl - cx) / f
    h = (v_cyl - cy) / f
    x_n = tan(θ)
    y_n = h · √(x_n² + 1) = h · sec(θ)
    u = f · x_n + cx
    v = f · y_n + cy

【球面投影数学推导】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
等距柱面投影 (Equirectangular):
    球面坐标 (φ, λ):
        φ = arcsin(y_n / ||P||)   [纬度, -π/2 到 π/2]
        λ = arctan2(x_n, z_n)     [经度, -π 到 π]

    图像坐标:
        u = f · λ + cx
        v = f · φ + cy
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageWarper:
    """
    图像变形器 - 支持平面、柱面、球面三种投影模式
    """

    def __init__(self, projection: str = 'cylindrical'):
        """
        Args:
            projection: 投影类型 'planar' / 'cylindrical' / 'spherical'
        """
        self.projection = projection.lower()
        logger.info(f"图像变形器初始化: 投影类型 = {projection}")

    def warp_cylindrical(
        self,
        image: np.ndarray,
        focal_length: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将图像变形到柱面坐标系

        柱面投影优点:
        - 水平方向图像宽度与实际角度成正比（等角度间距）
        - 避免平面拼接中大角度产生的"V形"畸变
        - 拼接后只需水平平移对齐（大幅简化对齐算法）

        Args:
            image: 输入图像 (H, W, 3)
            focal_length: 相机焦距（像素为单位）
                         f = (image_width / 2) / tan(fov/2)

        Returns:
            warped: 柱面投影后的图像
            mask: 有效像素掩码（0=无效/黑色边界）
        """
        h, w = image.shape[:2]
        f = focal_length

        # ── 创建目标像素坐标网格 ─────────────────────────────
        # u_out, v_out: 柱面图像中每个像素的坐标
        u_out, v_out = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )

        # ── 柱面 → 平面 逆映射 ───────────────────────────────
        # 使用逆映射（从目标采样源）避免空洞问题
        cx, cy = w / 2.0, h / 2.0

        # 1. 柱面坐标 → 水平角θ 和 归一化高度h
        theta = (u_out - cx) / f   # 水平角 (弧度)
        h_norm = (v_out - cy) / f  # 归一化垂直坐标

        # 2. 角度坐标 → 归一化相机坐标
        # x_n = tan(θ): 单位深度z=1时的x坐标
        # y_n = h_norm · sec(θ) = h_norm / cos(θ)
        x_n = np.tan(theta)           # tan(θ)
        sec_theta = 1.0 / np.cos(theta)  # sec(θ) = 1/cos(θ)
        y_n = h_norm * sec_theta      # y_n

        # 3. 归一化相机坐标 → 像素坐标（反投影到原图）
        u_src = f * x_n + cx
        v_src = f * y_n + cy

        # ── 双线性插值重采样 ──────────────────────────────────
        # 创建反向映射 (map_x, map_y): 目标像素 → 源像素坐标
        map_x = u_src.astype(np.float32)
        map_y = v_src.astype(np.float32)

        # cv2.remap: 使用双线性插值 (INTER_LINEAR) 进行重采样
        # 双线性插值公式:
        #   I(x,y) = (1-α)(1-β)·I(x₀,y₀) + α(1-β)·I(x₁,y₀)
        #          + (1-α)β·I(x₀,y₁) + αβ·I(x₁,y₁)
        #   α = x - x₀, β = y - y₀
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # ── 生成有效像素掩码 ──────────────────────────────────
        # 超出源图像边界的像素标记为无效（黑色边界问题）
        valid_u = (u_src >= 0) & (u_src < w)
        valid_v = (v_src >= 0) & (v_src < h)
        mask = (valid_u & valid_v).astype(np.uint8) * 255

        return warped, mask

    def warp_spherical(
        self,
        image: np.ndarray,
        focal_length: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将图像变形到球面坐标系（等距柱面投影）

        适用于垂直视角也较大的场景（如鱼眼镜头、手机广角）

        【投影数学】
        球面 → 平面的反向映射:
        给定球面图像坐标 (u, v):
            λ = (u - cx) / f   [经度]
            φ = (v - cy) / f   [纬度]
        球面坐标 → 3D单位向量:
            P = (cos(φ)·sin(λ), sin(φ), cos(φ)·cos(λ))
        3D向量 → 归一化平面坐标 (z=1):
            x_n = P_x / P_z = cos(φ)·sin(λ) / (cos(φ)·cos(λ)) = tan(λ)
            y_n = P_y / P_z = sin(φ) / (cos(φ)·cos(λ)) = tan(φ)·sec(λ)
        归一化坐标 → 像素坐标:
            u_src = f · x_n + cx
            v_src = f · y_n + cy
        """
        h, w = image.shape[:2]
        f = focal_length
        cx, cy = w / 2.0, h / 2.0

        u_out, v_out = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )

        # 球面 → 平面 逆映射
        lam = (u_out - cx) / f  # 经度 λ
        phi = (v_out - cy) / f  # 纬度 φ

        # 角度坐标 → 归一化相机坐标
        x_n = np.tan(lam)
        y_n = np.tan(phi) / np.cos(lam)  # tan(φ)·sec(λ)

        # 归一化坐标 → 像素坐标
        u_src = f * x_n + cx
        v_src = f * y_n + cy

        map_x = u_src.astype(np.float32)
        map_y = v_src.astype(np.float32)

        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        valid_u = (u_src >= 0) & (u_src < w)
        valid_v = (v_src >= 0) & (v_src < h)
        mask = (valid_u & valid_v).astype(np.uint8) * 255

        return warped, mask

    def warp_perspective(
        self,
        image: np.ndarray,
        H: np.ndarray,
        output_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用透视变换（基于单应性矩阵）

        【透视变换公式】
        目标图像坐标 (u', v') ← 源图像坐标 (u, v):
        [u']   [h₁₁ h₁₂ h₁₃] [u]
        [v'] = [h₂₁ h₂₂ h₂₃]·[v]
        [w']   [h₃₁ h₃₂ h₃₃] [1]
        实际坐标: u' = u'/w', v' = v'/w'

        反向映射（避免空洞）:
        对每个目标像素 (u', v')，使用 H⁻¹ 找到源像素:
        [u]   H⁻¹   [u']
        [v] = ───── [v']

        Args:
            image: 源图像
            H: 3×3 单应性矩阵（从源图到目标图）
            output_size: 输出图像尺寸 (width, height)

        Returns:
            warped: 透视变换后的图像
            mask: 有效像素掩码
        """
        out_w, out_h = output_size

        # cv2.warpPerspective 内部使用反向映射 + 双线性插值
        warped = cv2.warpPerspective(
            image, H, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # 生成掩码：对全白图像做同样变换
        mask_src = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        mask = cv2.warpPerspective(
            mask_src, H, (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return warped, mask

    def compute_canvas_size(
        self,
        images: list,
        homographies: list
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        根据所有图像和变换矩阵计算拼接画布的大小和偏移量

        【思路】
        将每张图像的四个角点用对应的单应性矩阵变换到参考坐标系中，
        取所有变换后角点的包围盒（bbox）作为画布范围。

        对于参考图像（H=I），四角不变。
        对其他图像:
        corner_transformed = H · corner_homogeneous

        Args:
            images: 图像列表
            homographies: 每张图像到参考系的变换矩阵列表

        Returns:
            offset: 平移向量（用于将坐标移到画布左上角）
            canvas_size: (width, height) 画布尺寸
        """
        all_corners = []

        for img, H in zip(images, homographies):
            h, w = img.shape[:2]
            # 图像四角点（齐次坐标）
            corners = np.float32([
                [0,   0,   1],
                [w-1, 0,   1],
                [w-1, h-1, 1],
                [0,   h-1, 1]
            ]).T  # 3×4

            # 变换四角点到参考坐标系
            corners_transformed = H @ corners  # 3×4
            # 透视除法
            corners_2d = corners_transformed[:2] / corners_transformed[2:3]  # 2×4
            all_corners.append(corners_2d.T)  # 4×2

        all_corners = np.vstack(all_corners)  # (4*N)×2

        # 计算包围盒
        x_min, y_min = all_corners.min(axis=0)
        x_max, y_max = all_corners.max(axis=0)

        # 画布尺寸（向上取整）
        canvas_w = int(np.ceil(x_max - x_min))
        canvas_h = int(np.ceil(y_max - y_min))

        # 偏移量（确保最小坐标为0）
        offset = np.float32([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        logger.info(f"画布尺寸: {canvas_w}×{canvas_h}, 偏移: ({-x_min:.0f}, {-y_min:.0f})")
        return offset, (canvas_w, canvas_h)
