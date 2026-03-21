"""
常规镜头畸变矫正完整案例（合成数据）

本脚本演示如何对常规镜头（非鱼眼）的径向畸变和切向畸变进行矫正。
包含三种矫正方法的对比：
1. cv2.undistort() —— 一步完成，简单直接
2. cv2.initUndistortRectifyMap() + cv2.remap() —— 预计算映射表，适合视频流
3. cv2.undistortPoints() —— 仅矫正特征点坐标，不生成完整图像

数据来源:
    本脚本使用程序自动生成的合成畸变图像，无需外部数据集。
    如需使用真实数据，可参考:
    - OpenCV 官方标定教程图片: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    - 使用 camera_calibration.py 标定后获得的内参和畸变系数

运行方式:
    cd calibration_algorithms
    python distortion_correction/standard_undistortion.py

作者: Claude (AI Assistant)
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os


# ============================================================================
# 第一步：定义相机参数（模拟已标定完成的结果）
# ============================================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 相机内参矩阵
CAMERA_MATRIX = np.array([
    [500.0,   0.0, 320.0],
    [  0.0, 500.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

# ============================================================================
# 定义三组不同程度的畸变系数，用于对比演示
# ============================================================================
# 格式: [k1, k2, p1, p2, k3]

# 轻微畸变（高质量镜头）
DIST_LIGHT = np.array([0.05, -0.02, 0.001, 0.0, 0.0], dtype=np.float64)

# 中等畸变（普通镜头）
DIST_MEDIUM = np.array([0.2, -0.5, 0.002, -0.003, 0.3], dtype=np.float64)

# 严重畸变（广角/低成本镜头）
DIST_HEAVY = np.array([-0.4, 0.2, 0.005, -0.005, -0.1], dtype=np.float64)


def generate_grid_image():
    """
    第二步：生成一个标准网格图像

    网格图像是验证畸变矫正最直观的方式：
    - 无畸变时，所有线条应该是笔直的
    - 桶形畸变会使线条向外弯曲
    - 枕形畸变会使线条向内弯曲

    返回:
        img: (H, W, 3) 标准网格图像
    """
    img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 255

    # 绘制网格线
    grid_spacing = 40
    for y in range(0, IMAGE_HEIGHT, grid_spacing):
        cv2.line(img, (0, y), (IMAGE_WIDTH - 1, y), (180, 180, 180), 1)
    for x in range(0, IMAGE_WIDTH, grid_spacing):
        cv2.line(img, (x, 0), (x, IMAGE_HEIGHT - 1), (180, 180, 180), 1)

    # 绘制十字中心标记
    cx, cy = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
    cv2.line(img, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 2)
    cv2.line(img, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 2)

    # 在交叉点绘制圆点
    for y in range(grid_spacing, IMAGE_HEIGHT - grid_spacing + 1, grid_spacing):
        for x in range(grid_spacing, IMAGE_WIDTH - grid_spacing + 1, grid_spacing):
            cv2.circle(img, (x, y), 3, (0, 120, 0), -1)

    return img


def apply_distortion(img, camera_matrix, dist_coeffs):
    """
    第三步：对图像施加畸变

    正向畸变过程：对输出（畸变）图像的每个像素位置，
    计算它在无畸变图像中的对应位置，然后采样。

    畸变数学模型:
        x_d = x(1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2+2*x^2)
        y_d = y(1 + k1*r^2 + k2*r^4 + k3*r^6) + p1*(r^2+2*y^2) + 2*p2*x*y

    参数:
        img: 无畸变的原始图像
        camera_matrix: 内参矩阵
        dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3]

    返回:
        distorted: 施加畸变后的图像
    """
    h, w = img.shape[:2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]

    # 构建映射表: 对畸变图像的每个像素，找到在原始图像中的对应位置
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    # 生成归一化坐标网格（向量化计算，比逐像素循环快很多）
    u_coords = np.arange(w, dtype=np.float64)
    v_coords = np.arange(h, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    # 像素坐标 → 归一化相机坐标
    x = (u_grid - cx) / fx
    y = (v_grid - cy) / fy

    # 径向距离的平方
    r2 = x * x + y * y

    # 径向畸变因子
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2

    # 施加完整畸变模型
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    # 归一化坐标 → 像素坐标
    map_x = (x_distorted * fx + cx).astype(np.float32)
    map_y = (y_distorted * fy + cy).astype(np.float32)

    # 使用 remap 生成畸变图像
    distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))
    return distorted


def method1_undistort(distorted_img, camera_matrix, dist_coeffs):
    """
    第四步 方法1：cv2.undistort() —— 一步去畸变

    原理:
        cv2.undistort() 内部执行以下操作:
        1. 对输出图像的每个像素 (u_dst, v_dst)，计算归一化坐标
        2. 反向施加畸变模型，找到在畸变图像中的对应位置
        3. 使用双线性插值采样得到像素值

    优点: 使用简单，一行代码完成
    缺点: 每次调用都要重新计算映射表，对视频流效率低

    参数:
        distorted_img: 畸变图像
        camera_matrix: 内参矩阵 K
        dist_coeffs: 畸变系数

    返回:
        undistorted: 去畸变图像
        elapsed_ms: 耗时（毫秒）
    """
    # =============================================
    # cv2.getOptimalNewCameraMatrix() 计算新的内参矩阵
    # =============================================
    # alpha 参数控制保留区域:
    #   alpha=0: 裁剪掉所有黑色区域（可能丢失边缘信息）
    #   alpha=1: 保留所有原始像素（会有黑色边框）
    h, w = distorted_img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h),
        alpha=0.5,  # 折中选择
        newImgSize=(w, h)
    )

    start = time.time()

    # =============================================
    # 核心：cv2.undistort()
    # =============================================
    undistorted = cv2.undistort(
        distorted_img,       # 输入畸变图像
        camera_matrix,       # 原始内参矩阵
        dist_coeffs,         # 畸变系数
        None,                # 输出图像（None 自动创建）
        new_camera_matrix    # 新的内参矩阵（控制输出视野）
    )

    elapsed_ms = (time.time() - start) * 1000

    # 可选：根据 ROI 裁剪黑色边框
    # x, y, w_roi, h_roi = roi
    # undistorted = undistorted[y:y+h_roi, x:x+w_roi]

    return undistorted, elapsed_ms


def method2_remap(distorted_img, camera_matrix, dist_coeffs):
    """
    第四步 方法2：initUndistortRectifyMap() + remap() —— 映射表方法

    原理:
        分两步进行：
        1. initUndistortRectifyMap(): 预计算 (u_src, v_src) = f(u_dst, v_dst)
           映射表，只需算一次
        2. remap(): 使用映射表对每帧图像进行重映射

    优点: 映射表只需计算一次，之后每帧只需 remap，适合视频流处理
    缺点: 需要额外存储映射表

    参数/返回: 同 method1
    """
    h, w = distorted_img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0.5
    )

    # =============================================
    # 步骤1：预计算映射表（只需执行一次）
    # =============================================
    # initUndistortRectifyMap() 参数:
    #   cameraMatrix:    原始内参
    #   distCoeffs:      畸变系数
    #   R:               可选的旋转矩阵（用于立体矫正，这里为 None）
    #   newCameraMatrix: 新内参（控制输出视野）
    #   size:            输出图像尺寸
    #   m1type:          映射表数据类型
    #     CV_32FC1: 两个 float32 映射表（map_x, map_y）
    #     CV_16SC2: 一个 int16 映射表 + 一个插值表（更快但精度稍低）
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,                # R（无立体矫正）
        new_camera_matrix,   # 新内参
        (w, h),              # 输出尺寸
        cv2.CV_32FC1         # 映射表类型
    )

    start = time.time()

    # =============================================
    # 步骤2：使用映射表进行重映射（每帧执行）
    # =============================================
    # remap() 参数:
    #   src:           输入畸变图像
    #   map1, map2:    映射表
    #   interpolation: 插值方法
    #     INTER_LINEAR:  双线性插值（速度和质量的平衡）
    #     INTER_CUBIC:   双三次插值（更高质量，更慢）
    #     INTER_NEAREST: 最近邻（最快，但有锯齿）
    undistorted = cv2.remap(
        distorted_img,
        map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    elapsed_ms = (time.time() - start) * 1000

    return undistorted, elapsed_ms


def method3_undistort_points(camera_matrix, dist_coeffs):
    """
    第四步 方法3：cv2.undistortPoints() —— 仅矫正特征点

    原理:
        不生成完整的去畸变图像，而是直接将畸变的 2D 点坐标
        转换为无畸变的归一化坐标或像素坐标。

    适用场景:
        - 只需要特征点的精确坐标（如 SLAM、视觉里程计）
        - 不需要完整的去畸变图像
        - 计算效率要求极高

    数学过程:
        给定畸变像素坐标 (u_d, v_d)，求解无畸变归一化坐标 (x, y)
        这需要求解畸变方程的逆：
        (x_d, y_d) = distort(x, y) → 求 (x, y) = distort^(-1)(x_d, y_d)
        OpenCV 使用迭代法（固定点迭代）求解

    参数:
        camera_matrix: 内参矩阵
        dist_coeffs: 畸变系数

    返回:
        results: 对比结果字典
    """
    # 生成一组模拟的特征点（在图像不同区域）
    h, w = IMAGE_HEIGHT, IMAGE_WIDTH
    feature_points = np.array([
        [100, 100], [320, 100], [540, 100],     # 上排
        [100, 240], [320, 240], [540, 240],     # 中排
        [100, 380], [320, 380], [540, 380],     # 下排
        [50, 50], [590, 50], [50, 430], [590, 430],  # 四角
    ], dtype=np.float64)

    # 先施加畸变（模拟从畸变图像中检测到的点）
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]

    distorted_points = np.zeros_like(feature_points)
    for i, (u, v) in enumerate(feature_points):
        x = (u - cx) / fx
        y = (v - cy) / fy
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        distorted_points[i] = [xd * fx + cx, yd * fy + cy]

    # =============================================
    # 核心：cv2.undistortPoints()
    # =============================================
    # 参数:
    #   src: 畸变的 2D 点坐标 (N, 1, 2)
    #   cameraMatrix: 内参矩阵
    #   distCoeffs:   畸变系数
    #   R:            可选旋转矩阵
    #   P:            新的投影矩阵（若提供，输出为像素坐标；否则为归一化坐标）
    #
    # 当 P=cameraMatrix 时，输出为无畸变的像素坐标
    undistorted_pts = cv2.undistortPoints(
        distorted_points.reshape(-1, 1, 2).astype(np.float64),
        camera_matrix,
        dist_coeffs,
        R=None,
        P=camera_matrix   # 输出像素坐标而非归一化坐标
    ).reshape(-1, 2)

    # 计算矫正精度
    errors = np.sqrt(np.sum((undistorted_pts - feature_points) ** 2, axis=1))

    results = {
        'original': feature_points,
        'distorted': distorted_points,
        'undistorted': undistorted_pts,
        'errors': errors
    }

    return results


def compare_distortion_levels(grid_img):
    """
    第五步：对比不同畸变程度的效果

    分别展示轻微、中等、严重三种畸变以及矫正后的效果。
    """
    print("\n" + "=" * 60)
    print("[步骤5] 对比不同畸变程度的矫正效果")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    distortions = {
        'Light (k1=0.05)': DIST_LIGHT,
        'Medium (k1=0.2)': DIST_MEDIUM,
        'Heavy (k1=-0.4)': DIST_HEAVY,
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    for row, (name, dist) in enumerate(distortions.items()):
        # 生成畸变图像
        distorted = apply_distortion(grid_img, CAMERA_MATRIX, dist)

        # 使用方法1矫正
        undistorted, _ = method1_undistort(distorted, CAMERA_MATRIX, dist)

        # 显示
        axes[row, 0].imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title('Original' if row == 0 else '')
        axes[row, 0].set_ylabel(name, fontsize=11)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_title('Distorted' if row == 0 else '')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title('Undistorted' if row == 0 else '')
        axes[row, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'distortion_levels_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  不同畸变程度对比图已保存: {save_path}")


def compare_methods(grid_img):
    """
    第六步：对比三种矫正方法的性能

    1. cv2.undistort(): 简单但每帧都重新计算映射
    2. remap(): 预计算映射表，适合视频
    3. undistortPoints(): 仅矫正点坐标
    """
    print("\n" + "=" * 60)
    print("[步骤6] 对比三种矫正方法")
    print("=" * 60)

    dist = DIST_MEDIUM
    distorted = apply_distortion(grid_img, CAMERA_MATRIX, dist)

    # 方法1: undistort
    undist1, time1 = method1_undistort(distorted, CAMERA_MATRIX, dist)
    print(f"  方法1 cv2.undistort():   {time1:.2f} ms")

    # 方法2: remap（测量仅 remap 步骤的时间）
    undist2, time2 = method2_remap(distorted, CAMERA_MATRIX, dist)
    print(f"  方法2 remap():           {time2:.2f} ms")

    # 多次运行取平均
    n_runs = 100
    t1_total, t2_total = 0, 0
    for _ in range(n_runs):
        _, t1 = method1_undistort(distorted, CAMERA_MATRIX, dist)
        _, t2 = method2_remap(distorted, CAMERA_MATRIX, dist)
        t1_total += t1
        t2_total += t2

    print(f"\n  平均耗时 ({n_runs} 次):")
    print(f"    方法1 undistort: {t1_total/n_runs:.3f} ms/frame")
    print(f"    方法2 remap:     {t2_total/n_runs:.3f} ms/frame")
    print(f"    remap 相对加速:  {t1_total/t2_total:.2f}x")

    # 方法3: undistortPoints
    pts_results = method3_undistort_points(CAMERA_MATRIX, dist)
    print(f"\n  方法3 undistortPoints() 精度:")
    print(f"    平均矫正误差: {np.mean(pts_results['errors']):.6f} 像素")
    print(f"    最大矫正误差: {np.max(pts_results['errors']):.6f} 像素")

    # 可视化方法对比
    output_dir = os.path.dirname(os.path.abspath(__file__))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Distorted Input')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(undist1, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Method 1: cv2.undistort()')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(undist2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Method 2: remap()')
    axes[1, 0].axis('off')

    # 方法3: 点矫正可视化
    ax = axes[1, 1]
    orig = pts_results['original']
    dist_pts = pts_results['distorted']
    undist_pts = pts_results['undistorted']

    ax.scatter(orig[:, 0], orig[:, 1], c='green', s=60, marker='o',
               label='Original', zorder=3)
    ax.scatter(dist_pts[:, 0], dist_pts[:, 1], c='red', s=40, marker='x',
               label='Distorted', zorder=3)
    ax.scatter(undist_pts[:, 0], undist_pts[:, 1], c='blue', s=30, marker='+',
               label='Undistorted', zorder=3)

    # 画箭头显示畸变方向
    for i in range(len(orig)):
        ax.annotate('', xy=dist_pts[i], xytext=orig[i],
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.4))

    ax.set_xlim(0, IMAGE_WIDTH)
    ax.set_ylim(IMAGE_HEIGHT, 0)
    ax.set_title('Method 3: undistortPoints()')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'methods_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  方法对比图已保存: {save_path}")


def demonstrate_alpha_effect(grid_img):
    """
    第七步：演示 getOptimalNewCameraMatrix 的 alpha 参数效果

    alpha 参数控制去畸变后的裁剪范围:
    - alpha=0: 裁剪掉所有无效（黑色）区域，输出的图像中所有像素都是有效的
    - alpha=1: 保留所有像素，包括黑色边框区域
    - alpha=0.5: 折中方案
    """
    print("\n" + "=" * 60)
    print("[步骤7] 演示 alpha 参数对去畸变结果的影响")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    dist = DIST_MEDIUM
    distorted = apply_distortion(grid_img, CAMERA_MATRIX, dist)
    h, w = distorted.shape[:2]

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(alphas), figsize=(20, 4))

    for i, alpha in enumerate(alphas):
        new_cam, roi = cv2.getOptimalNewCameraMatrix(
            CAMERA_MATRIX, dist, (w, h), alpha=alpha
        )
        undist = cv2.undistort(distorted, CAMERA_MATRIX, dist, None, new_cam)

        axes[i].imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'alpha={alpha}')
        axes[i].axis('off')

        # 绘制 ROI 矩形
        if roi[2] > 0 and roi[3] > 0:
            x, y, w_roi, h_roi = roi
            rect = plt.Rectangle((x, y), w_roi, h_roi,
                                 linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)

    plt.suptitle('Effect of alpha on getOptimalNewCameraMatrix (red=ROI)', fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'alpha_effect.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  alpha 参数效果图已保存: {save_path}")


def main():
    print("=" * 60)
    print("      常规镜头畸变矫正演示（合成数据）")
    print("=" * 60)

    print(f"\n[步骤1] 相机参数:")
    print(f"  图像尺寸: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"  焦距: fx={CAMERA_MATRIX[0,0]}, fy={CAMERA_MATRIX[1,1]}")

    print(f"\n  畸变系数组:")
    print(f"    轻微: k1={DIST_LIGHT[0]}, k2={DIST_LIGHT[1]}")
    print(f"    中等: k1={DIST_MEDIUM[0]}, k2={DIST_MEDIUM[1]}")
    print(f"    严重: k1={DIST_HEAVY[0]}, k2={DIST_HEAVY[1]}")

    # 生成网格图像
    print("\n[步骤2] 生成标准网格图像")
    grid_img = generate_grid_image()

    # 对比不同畸变程度
    compare_distortion_levels(grid_img)

    # 对比三种矫正方法
    compare_methods(grid_img)

    # 演示 alpha 参数
    demonstrate_alpha_effect(grid_img)

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)
    print("  三种去畸变方法适用场景:")
    print("    cv2.undistort():        单张图像，简单快速")
    print("    remap():                视频流，预计算映射表复用")
    print("    undistortPoints():      仅需特征点坐标（SLAM/VO）")
    print("  alpha 参数:")
    print("    alpha=0 裁剪黑边，alpha=1 保留全部，建议 0.5 折中")


if __name__ == '__main__':
    main()
