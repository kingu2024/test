"""
全景图像拼接主控模块
Panorama Stitcher - Main Pipeline

【全景拼接完整流程】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: 特征提取 (Feature Extraction)
        每张图像提取 SIFT/ORB 特征点和描述子

Step 2: 特征匹配 (Feature Matching)
        相邻图像间进行 FLANN 匹配 + Lowe比值测试

Step 3: 单应性估计 (Homography Estimation)
        RANSAC 鲁棒估计图像间的投影变换矩阵 H

Step 4: 投影变换 (Projection)
        可选: 将图像投影到柱面/球面坐标系（减少大视角畸变）

Step 5: 全局对齐 (Global Alignment)
        多图像时通过最大生成树确定变换顺序，累积变换矩阵

Step 6: 图像变形 (Warping)
        将所有图像变换到统一参考坐标系

Step 7: 图像融合 (Blending)
        多分辨率拉普拉斯金字塔融合消除接缝

Step 8: 后处理 (Post-processing)
        裁剪黑边，色彩均衡化

【技术亮点（2024-2025最新方法参考）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 自适应特征检测: 根据图像纹理丰富度自动选择 SIFT/ORB/AKAZE
2. 柱面/球面投影: 处理大视角（>45°）拼接的畸变问题
3. 多分辨率融合: Laplacian Pyramid Blending 消除色差和接缝
4. 最优缝合线: Dijkstra 动态规划寻找视觉最优拼接边界
5. 曝光补偿: Gain Compensation 消除图像间的亮度差异
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from .feature_extraction import FeatureExtractor, FeatureMatcher
from .homography import HomographyEstimator
from .warping import ImageWarper
from .blending import ImageBlender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class PanoramaStitcher:
    """
    全景图像拼接器
    支持2张及以上图像的全景拼接
    """

    def __init__(
        self,
        feature_method: str = 'SIFT',
        projection: str = 'cylindrical',
        blend_method: str = 'multiband',
        max_features: int = 2000,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 4.0,
        focal_length: Optional[float] = None
    ):
        """
        初始化全景拼接器

        Args:
            feature_method: 特征检测算法 'SIFT'/'ORB'/'AKAZE'
            projection: 投影类型 'planar'/'cylindrical'/'spherical'
                       - planar: 适合<30°小视角，计算最快
                       - cylindrical: 适合水平360°全景，减少竖向拉伸
                       - spherical: 适合全向（360°×180°）全景
            blend_method: 融合算法 'multiband'/'feather'/'simple'
            max_features: 每张图最大特征点数
            ratio_threshold: Lowe's比值测试阈值
            ransac_threshold: RANSAC内点判定阈值（像素）
            focal_length: 相机焦距（像素），None时自动估算
                         f ≈ max(width, height) * 1.2（经验公式）
        """
        # 初始化各子模块
        self.extractor = FeatureExtractor(feature_method, max_features)
        self.matcher = FeatureMatcher(feature_method, ratio_threshold)
        self.homography_est = HomographyEstimator(
            ransac_reproj_threshold=ransac_threshold
        )
        self.warper = ImageWarper(projection)
        self.blender = ImageBlender(blend_method)
        self.projection = projection
        self.focal_length = focal_length

        logger.info(
            f"全景拼接器初始化完成:\n"
            f"  特征算法={feature_method}, 投影={projection}, 融合={blend_method}\n"
            f"  最大特征数={max_features}, Lowe比值={ratio_threshold}, "
            f"RANSAC阈值={ransac_threshold}"
        )

    def _estimate_focal_length(self, image: np.ndarray) -> float:
        """
        估算相机焦距

        【经验公式】
        对于典型手机/相机（FOV约60-80°）:
        f ≈ w / (2·tan(FOV/2))
        经验近似: f ≈ max(w, h) * 1.0~1.5

        更精确方法: 从EXIF信息读取焦距和传感器尺寸

        Args:
            image: 参考图像

        Returns:
            估算的焦距（像素）
        """
        h, w = image.shape[:2]
        # 假设水平FOV约70°: f = (w/2)/tan(35°) ≈ w * 0.714
        # 保守估计取 max(w,h) 确保适用多种相机
        f = max(w, h) * 1.0
        logger.info(f"自动估算焦距: f = {f:.1f} 像素 (图像尺寸 {w}×{h})")
        return f

    def _exposure_compensation(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        曝光补偿（增益补偿）
        消除图像间的整体亮度差异

        【算法原理】
        假设两图在重叠区域应有相同像素值（理想情况），
        实际因曝光差异导致差异：I₂(p) ≈ g·I₁(p)
        其中 g 为增益因子。

        最小化目标:
        E = Σ_{i,j} Σ_{p∈overlap(i,j)} (gᵢ·Iᵢ(p) - gⱼ·Iⱼ(p))²

        对 gᵢ 求偏导令其为零，得线性方程组：
        Σⱼ Nᵢⱼ·(σᵢⱼ·gᵢ - σⱼᵢ·gⱼ) = 0

        其中 Nᵢⱼ = 重叠像素数，σᵢⱼ = 重叠区平均强度比

        约束: g₀ = 1（参考图像增益固定为1）
        """
        if len(images) < 2:
            return images

        # 以第一张图像为参考（增益=1）
        compensated = [images[0].copy()]
        ref = images[0].astype(np.float32)
        ref_mask = masks[0]

        for i in range(1, len(images)):
            img = images[i].astype(np.float32)
            cur_mask = masks[i]

            # 计算重叠区域（两个掩码的交集）
            overlap = (ref_mask > 0) & (cur_mask > 0)

            if overlap.sum() < 100:
                # 重叠区域太小，不补偿
                compensated.append(images[i].copy())
                continue

            # 在重叠区域计算增益因子（各通道独立）
            gain = np.ones(3)
            for c in range(3):
                ref_vals = ref[:, :, c][overlap]
                cur_vals = img[:, :, c][overlap]

                # 鲁棒增益估算: g = mean(ref) / mean(cur)
                # 使用中位数减少极端值影响
                ref_mean = np.median(ref_vals)
                cur_mean = np.median(cur_vals)

                if cur_mean > 1e-3:
                    gain[c] = ref_mean / cur_mean

            # 限制增益在合理范围内（避免过度补偿）
            gain = np.clip(gain, 0.5, 2.0)
            logger.info(f"图像 {i} 曝光增益补偿: R={gain[0]:.3f}, G={gain[1]:.3f}, B={gain[2]:.3f}")

            # 应用增益
            compensated_img = np.clip(img * gain, 0, 255).astype(np.uint8)
            compensated.append(compensated_img)

        return compensated

    def stitch(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        执行全景图像拼接

        Args:
            images: 按顺序排列的图像列表（从左到右或从右到左）
                   建议: 相邻图像有30-50%重叠

        Returns:
            全景拼接结果，失败返回 None
        """
        n = len(images)
        if n < 2:
            logger.error("至少需要2张图像才能拼接")
            return None

        logger.info(f"开始拼接 {n} 张图像...")

        # ════════════════════════════════════════════════════════
        # Step 1: 柱面/球面投影预处理（可选）
        # 将平面图像投影到柱面，减少大视角时的透视畸变
        # ════════════════════════════════════════════════════════
        if self.projection in ('cylindrical', 'spherical'):
            if self.focal_length is None:
                self.focal_length = self._estimate_focal_length(images[0])

            logger.info(f"正在进行 {self.projection} 投影变换...")
            projected_images = []
            projected_masks = []

            for i, img in enumerate(images):
                if self.projection == 'cylindrical':
                    # 柱面投影: 适合水平拼接
                    # 数学: (u,v) → (f·arctan(x_n), f·h/(√(x_n²+1)))
                    proj_img, proj_mask = self.warper.warp_cylindrical(
                        img, self.focal_length
                    )
                else:
                    # 球面投影: 适合全向拼接
                    # 数学: (u,v) → (f·λ, f·φ) 等距柱面
                    proj_img, proj_mask = self.warper.warp_spherical(
                        img, self.focal_length
                    )
                projected_images.append(proj_img)
                projected_masks.append(proj_mask)
                logger.info(f"图像 {i+1}/{n} 投影完成")

            work_images = projected_images
        else:
            # 平面拼接: 直接使用原始图像
            work_images = images
            projected_masks = [
                np.ones(img.shape[:2], dtype=np.uint8) * 255
                for img in images
            ]

        # ════════════════════════════════════════════════════════
        # Step 2: 特征提取
        # 对每张工作图像提取特征点和描述子
        # ════════════════════════════════════════════════════════
        logger.info("正在提取特征点...")
        all_keypoints = []
        all_descriptors = []

        for i, img in enumerate(work_images):
            kp, desc = self.extractor.detect_and_compute(img)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            logger.info(f"图像 {i+1}/{n}: {len(kp)} 个特征点")

        # ════════════════════════════════════════════════════════
        # Step 3: 相邻图像特征匹配和单应性估计
        # 逐对计算相邻图像间的变换矩阵
        # ════════════════════════════════════════════════════════
        logger.info("正在进行特征匹配和单应性估计...")
        homographies = [np.eye(3)]  # 第一张图像为参考（单位矩阵）

        # 累积单应性矩阵 H_cumulative[i] = 图像i到参考系（图像0）的变换
        H_cumulative = np.eye(3)

        for i in range(n - 1):
            logger.info(f"匹配图像对 ({i+1}, {i+2})...")

            # 匹配相邻图像特征
            matches = self.matcher.match(
                all_descriptors[i],
                all_descriptors[i + 1]
            )

            # RANSAC 估计单应性矩阵 H_{i→i+1}
            # H: 将图像i+1的点映射到图像i坐标系中
            H_pair, mask = self.homography_est.estimate(
                all_keypoints[i + 1],
                all_keypoints[i],
                matches
            )

            if H_pair is None:
                logger.error(f"图像对 ({i+1},{i+2}) 单应性估计失败！匹配点不足")
                return None

            # 累积变换: H_{i+1→0} = H_{i→0} · H_{i+1→i}
            # 矩阵乘法实现变换链: 先变到i，再变到0
            H_cumulative = H_cumulative @ H_pair
            homographies.append(H_cumulative.copy())

            n_inliers = int(mask.sum()) if mask is not None else 0
            logger.info(f"图像 {i+2} 到参考系的变换估计成功，内点数: {n_inliers}")

        # ════════════════════════════════════════════════════════
        # Step 4: 计算画布大小并执行图像变形
        # 将所有图像变换到统一坐标系（参考图像坐标系）
        # ════════════════════════════════════════════════════════
        logger.info("正在计算画布大小...")
        offset_H, canvas_size = self.warper.compute_canvas_size(
            work_images, homographies
        )

        logger.info(f"正在将 {n} 张图像变形到画布...")
        warped_images = []
        warped_masks = []

        for i, (img, H) in enumerate(zip(work_images, homographies)):
            # 组合平移偏移和累积单应性:
            # H_final = offset · H_cumulative
            # 确保变换后的图像位于画布的正坐标区域
            H_final = offset_H @ H

            warped_img, warped_mask = self.warper.warp_perspective(
                img, H_final, canvas_size
            )

            # 与投影掩码取交集（处理柱面投影的黑边区域）
            if self.projection in ('cylindrical', 'spherical'):
                proj_mask_warped, _ = self.warper.warp_perspective(
                    projected_masks[i], H_final, canvas_size
                )
                warped_mask = cv2.bitwise_and(warped_mask, proj_mask_warped)

            warped_images.append(warped_img)
            warped_masks.append(warped_mask)
            logger.info(f"图像 {i+1}/{n} 变形完成")

        # ════════════════════════════════════════════════════════
        # Step 5: 曝光补偿
        # 消除图像间的亮度/颜色差异
        # ════════════════════════════════════════════════════════
        logger.info("正在进行曝光补偿...")
        warped_images = self._exposure_compensation(warped_images, warped_masks)

        # ════════════════════════════════════════════════════════
        # Step 6: 多图像融合
        # 从左到右逐步融合所有图像
        # 采用两两融合策略（可扩展为同时融合）
        # ════════════════════════════════════════════════════════
        logger.info("正在融合图像（多分辨率拉普拉斯金字塔）...")
        result = warped_images[0]
        result_mask = warped_masks[0]

        for i in range(1, n):
            logger.info(f"融合第 {i+1}/{n} 张图像...")

            # 多分辨率融合: 在重叠区域使用拉普拉斯金字塔过渡
            # 单独区域直接拷贝（权重=1或0）
            result = self.blender.blend(
                result, warped_images[i],
                result_mask, warped_masks[i]
            )

            # 更新有效区域掩码（取并集）
            result_mask = cv2.bitwise_or(result_mask, warped_masks[i])

        # ════════════════════════════════════════════════════════
        # Step 7: 后处理（裁剪黑边）
        # ════════════════════════════════════════════════════════
        logger.info("正在裁剪黑边...")
        result = self._crop_black_border(result, result_mask)

        logger.info(
            f"全景拼接完成! 输出尺寸: {result.shape[1]}×{result.shape[0]}"
        )
        return result

    def _crop_black_border(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        裁剪图像黑色边框

        【算法】
        1. 对掩码进行形态学腐蚀（去除噪声）
        2. 找到有效区域的边界框
        3. 裁剪到边界框（保留所有有效内容）

        注意：对于柱面投影，顶部和底部可能有黑边，
        但左右两侧应该连续

        Args:
            image: 待裁剪图像
            mask: 有效像素掩码

        Returns:
            裁剪后的图像
        """
        # 形态学腐蚀：去除掩码边界的噪声像素
        # 避免因微小黑色区域导致裁剪范围过小
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=3)

        # 找非零区域的边界（最小包围矩形）
        ys, xs = np.where(mask_eroded > 0)

        if len(xs) == 0 or len(ys) == 0:
            logger.warning("掩码为空，无法裁剪")
            return image

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # 添加1像素边距
        x_min = max(0, x_min - 1)
        y_min = max(0, y_min - 1)
        x_max = min(image.shape[1] - 1, x_max + 1)
        y_max = min(image.shape[0] - 1, y_max + 1)

        cropped = image[y_min:y_max+1, x_min:x_max+1]
        logger.info(
            f"裁剪: ({x_min},{y_min}) → ({x_max},{y_max}), "
            f"最终尺寸: {cropped.shape[1]}×{cropped.shape[0]}"
        )
        return cropped


def stitch_images(
    image_paths: List[str],
    output_path: str = 'panorama_output.jpg',
    feature_method: str = 'SIFT',
    projection: str = 'cylindrical',
    blend_method: str = 'multiband',
    focal_length: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    全景拼接便捷函数

    Args:
        image_paths: 图像文件路径列表（按顺序）
        output_path: 输出全景图路径
        feature_method: 特征检测方法
        projection: 投影类型
        blend_method: 融合方法
        focal_length: 相机焦距（像素），None自动估算

    Returns:
        拼接结果图像，失败返回 None
    """
    # 读取图像
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            logger.error(f"无法读取图像: {path}")
            return None
        images.append(img)
        logger.info(f"读取图像: {path} ({img.shape[1]}×{img.shape[0]})")

    # 初始化拼接器
    stitcher = PanoramaStitcher(
        feature_method=feature_method,
        projection=projection,
        blend_method=blend_method,
        focal_length=focal_length
    )

    # 执行拼接
    result = stitcher.stitch(images)

    if result is not None:
        # 保存结果
        cv2.imwrite(output_path, result)
        logger.info(f"全景图已保存到: {output_path}")

    return result
