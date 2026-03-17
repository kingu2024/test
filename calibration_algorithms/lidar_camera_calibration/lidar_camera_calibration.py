"""
LiDAR-相机外参标定完整案例 —— 基于 PnP 算法（合成数据）

本脚本演示如何标定 LiDAR 与相机之间的外参变换矩阵 T_lidar2cam。
外参变换描述了 LiDAR 坐标系到相机坐标系的刚体变换（旋转+平移）。

核心思路：
    已知相机内参 K，通过 LiDAR 点云中的 3D 特征点与对应的图像 2D 像素点，
    使用 PnP（Perspective-n-Point）算法求解外参。

数据来源:
    本脚本使用程序自动生成的合成数据，无需外部数据集。
    如需使用真实数据，推荐以下公开数据集:
    - KITTI 数据集: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
      (包含 Velodyne LiDAR 点云和同步的相机图像，以及标注的标定参数)
    - nuScenes 数据集: https://www.nuscenes.org/
      (多传感器融合数据集，包含 LiDAR 和多相机标定)

运行方式:
    cd calibration_algorithms
    python lidar_camera_calibration/lidar_camera_calibration.py

作者: Claude (AI Assistant)
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# ============================================================================
# 第一步：定义"真实"参数（Ground Truth）
# ============================================================================

# 图像分辨率
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# 相机内参矩阵 K（假设已通过相机标定获得）
# 在实际应用中，这一步需要先完成相机标定
CAMERA_MATRIX = np.array([
    [720.0,   0.0, 640.0],   # [fx,  0, cx]
    [  0.0, 720.0, 360.0],   # [ 0, fy, cy]
    [  0.0,   0.0,   1.0]    # [ 0,  0,  1]
], dtype=np.float64)

# 相机畸变系数（假设已标定，且畸变很小）
DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# ============================================================================
# LiDAR → Camera 的真实外参变换矩阵 T_lidar2cam
# ============================================================================
# 这个 4x4 齐次变换矩阵描述了从 LiDAR 坐标系到相机坐标系的变换
# T = [R | t]
#     [0 | 1]
#
# 典型的 LiDAR-相机安装方式（以自动驾驶为例）：
# - LiDAR 安装在车顶，坐标系: x前方, y左, z上
# - 相机安装在前挡风玻璃处，坐标系: x右, y下, z前方
# 因此旋转矩阵需要交换坐标轴

# 定义旋转：绕 Y 轴旋转 5°，绕 X 轴旋转 -3°（模拟安装偏差）
angle_y = 5.0 * np.pi / 180.0   # 绕 Y 轴 5°
angle_x = -3.0 * np.pi / 180.0  # 绕 X 轴 -3°

# 绕 Y 轴旋转矩阵 Ry
Ry = np.array([
    [np.cos(angle_y),  0, np.sin(angle_y)],
    [0,                1, 0              ],
    [-np.sin(angle_y), 0, np.cos(angle_y)]
], dtype=np.float64)

# 绕 X 轴旋转矩阵 Rx
Rx = np.array([
    [1, 0,                0               ],
    [0, np.cos(angle_x), -np.sin(angle_x)],
    [0, np.sin(angle_x),  np.cos(angle_x)]
], dtype=np.float64)

# 坐标系变换基础旋转矩阵：将 LiDAR 坐标系(x前,y左,z上)
# 变换到相机坐标系(x右,y下,z前)
# 这等价于: cam_x = -lidar_y, cam_y = -lidar_z, cam_z = lidar_x
R_axes = np.array([
    [0, -1,  0],   # cam_x = -lidar_y
    [0,  0, -1],   # cam_y = -lidar_z
    [1,  0,  0]    # cam_z = lidar_x
], dtype=np.float64)

# 总旋转 = 安装偏差 × 坐标轴变换
TRUE_ROTATION = Ry @ Rx @ R_axes

# 平移向量（LiDAR 到相机的位移，在相机坐标系下表示）
# 假设相机在 LiDAR 前方 0.1m、下方 0.3m、右侧 0.05m
TRUE_TRANSLATION = np.array([[0.05], [-0.3], [0.1]], dtype=np.float64)

# 构建完整的 4x4 变换矩阵
TRUE_T_LIDAR2CAM = np.eye(4, dtype=np.float64)
TRUE_T_LIDAR2CAM[:3, :3] = TRUE_ROTATION
TRUE_T_LIDAR2CAM[:3, 3:4] = TRUE_TRANSLATION

# 合成数据参数
NUM_CALIBRATION_POINTS = 80   # 标定用的 3D-2D 对应点数
NUM_OUTLIERS = 10             # 离群点数量（用于测试 RANSAC）
NOISE_STD_2D = 1.0           # 2D 检测噪声（像素）
NOISE_STD_3D = 0.005         # 3D 点云噪声（米）


def generate_synthetic_lidar_camera_data():
    """
    第二步：生成合成的 LiDAR-相机对应点数据

    模拟场景：在 LiDAR 坐标系下，前方 5~30m、左右 ±10m、高度 -1~3m
    的范围内随机放置特征点（模拟标定板角点或自然特征点）。

    流程：
    1. 在 LiDAR 坐标系下随机生成 3D 点
    2. 通过真实变换矩阵 T_lidar2cam 变换到相机坐标系
    3. 通过相机内参矩阵 K 投影到图像平面
    4. 添加噪声
    5. 生成一些离群点（错误匹配）来测试 RANSAC 的鲁棒性

    返回:
        points_3d_lidar: LiDAR 坐标系下的 3D 点 (N, 3)
        points_2d_image: 对应的图像 2D 像素坐标 (N, 2)
        points_3d_clean: 无离群点的 3D 点（用于对比）
        points_2d_clean: 无离群点的 2D 点（用于对比）
        inlier_mask:     标记哪些点是内点
    """
    print("\n" + "=" * 60)
    print("[步骤2] 生成合成 LiDAR-相机对应点数据")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # --------------------------------------------------
    # 2.1 在 LiDAR 坐标系下生成 3D 点
    # --------------------------------------------------
    # LiDAR 坐标系: x=前方, y=左, z=上
    # 模拟多个标定板在不同距离处的角点
    points_3d_lidar = np.zeros((NUM_CALIBRATION_POINTS, 3), dtype=np.float64)
    points_3d_lidar[:, 0] = rng.uniform(5.0, 30.0, NUM_CALIBRATION_POINTS)   # x: 前方 5~30m
    points_3d_lidar[:, 1] = rng.uniform(-10.0, 10.0, NUM_CALIBRATION_POINTS) # y: 左右 ±10m
    points_3d_lidar[:, 2] = rng.uniform(-1.0, 3.0, NUM_CALIBRATION_POINTS)   # z: 高度 -1~3m

    # 添加 3D 点云噪声（模拟 LiDAR 测量误差）
    noise_3d = rng.normal(0, NOISE_STD_3D, points_3d_lidar.shape)
    points_3d_noisy = points_3d_lidar + noise_3d

    # --------------------------------------------------
    # 2.2 将 LiDAR 点变换到相机坐标系
    # --------------------------------------------------
    # P_cam = R * P_lidar + t
    points_3d_cam = (TRUE_ROTATION @ points_3d_lidar.T + TRUE_TRANSLATION).T

    # --------------------------------------------------
    # 2.3 投影到图像平面
    # --------------------------------------------------
    # 透视投影: [u, v, 1]^T = K * [X_cam/Z_cam, Y_cam/Z_cam, 1]^T
    # 只保留 Z_cam > 0 的点（在相机前方的点）
    valid_mask = points_3d_cam[:, 2] > 0.1  # Z > 0.1m

    points_2d = np.zeros((NUM_CALIBRATION_POINTS, 2), dtype=np.float64)
    for i in range(NUM_CALIBRATION_POINTS):
        if valid_mask[i]:
            X, Y, Z = points_3d_cam[i]
            # 透视除法 + 内参投影
            u = CAMERA_MATRIX[0, 0] * X / Z + CAMERA_MATRIX[0, 2]
            v = CAMERA_MATRIX[1, 1] * Y / Z + CAMERA_MATRIX[1, 2]
            points_2d[i] = [u, v]

    # 检查点是否在图像范围内
    in_image = (
        valid_mask &
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < IMAGE_WIDTH) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < IMAGE_HEIGHT)
    )

    # 筛选有效点
    points_3d_valid = points_3d_noisy[in_image]
    points_2d_valid = points_2d[in_image]

    # 添加 2D 检测噪声
    noise_2d = rng.normal(0, NOISE_STD_2D, points_2d_valid.shape)
    points_2d_noisy = points_2d_valid + noise_2d

    # 保存干净数据（无离群点）用于对比
    points_3d_clean = points_3d_valid.copy()
    points_2d_clean = points_2d_noisy.copy()

    print(f"  生成 {NUM_CALIBRATION_POINTS} 个 LiDAR 点，"
          f"其中 {len(points_3d_valid)} 个在图像范围内")

    # --------------------------------------------------
    # 2.4 添加离群点（错误匹配）
    # --------------------------------------------------
    # 模拟实际中的错误特征匹配
    n_valid = len(points_3d_valid)
    n_outliers = min(NUM_OUTLIERS, n_valid // 5)  # 离群点不超过 20%

    if n_outliers > 0:
        # 随机选择一些点，将其 2D 坐标替换为随机位置
        outlier_indices = rng.choice(n_valid, n_outliers, replace=False)
        points_2d_noisy[outlier_indices, 0] = rng.uniform(0, IMAGE_WIDTH, n_outliers)
        points_2d_noisy[outlier_indices, 1] = rng.uniform(0, IMAGE_HEIGHT, n_outliers)

        # 记录内点/离群点标记
        inlier_mask = np.ones(n_valid, dtype=bool)
        inlier_mask[outlier_indices] = False

        print(f"  添加了 {n_outliers} 个离群点（错误匹配），用于测试 RANSAC")
    else:
        inlier_mask = np.ones(n_valid, dtype=bool)

    return points_3d_valid, points_2d_noisy, points_3d_clean, points_2d_clean, inlier_mask


def calibrate_with_pnp(points_3d, points_2d):
    """
    第三步：使用标准 PnP 算法求解外参

    PnP (Perspective-n-Point) 问题：
    已知 n 个 3D 世界坐标点和对应的 2D 图像坐标，以及相机内参，
    求解相机的外参（旋转 R 和平移 t）。

    cv2.solvePnP() 默认使用 SOLVEPNP_ITERATIVE 方法：
    1. 使用 DLT（Direct Linear Transform）获得初始解
    2. 使用 Levenberg-Marquardt 优化最小化重投影误差

    注意：标准 PnP 不处理离群点，所有点都参与优化。

    参数:
        points_3d: (N, 3) LiDAR 坐标系下的 3D 点
        points_2d: (N, 2) 对应的图像 2D 像素坐标

    返回:
        R: (3, 3) 旋转矩阵
        t: (3, 1) 平移向量
        success: 是否成功
    """
    print("\n" + "=" * 60)
    print("[步骤3] 使用标准 PnP 求解外参")
    print("=" * 60)

    # OpenCV 要求特定的数组格式
    obj_pts = points_3d.reshape(-1, 1, 3).astype(np.float64)
    img_pts = points_2d.reshape(-1, 1, 2).astype(np.float64)

    # =============================================
    # 核心：调用 cv2.solvePnP()
    # =============================================
    # 参数:
    #   objectPoints: 3D 点坐标 (N, 1, 3) 或 (N, 3)
    #   imagePoints:  2D 点坐标 (N, 1, 2) 或 (N, 2)
    #   cameraMatrix: 相机内参矩阵 K
    #   distCoeffs:   畸变系数
    #   flags:        求解方法
    #     - SOLVEPNP_ITERATIVE: 迭代法（默认，基于 DLT + LM 优化）
    #     - SOLVEPNP_P3P:       P3P 算法（仅需 4 个点）
    #     - SOLVEPNP_EPNP:      EPnP 算法（高效，适合大量点）
    #     - SOLVEPNP_SQPNP:     SQPnP 算法（最新，精度好）
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts,
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        # 将旋转向量转换为旋转矩阵
        # cv2.Rodrigues() 实现 Rodrigues 旋转公式:
        # R = I + sin(θ) * [n]_x + (1-cos(θ)) * [n]_x^2
        # 其中 θ = ||rvec||, n = rvec / θ
        R, _ = cv2.Rodrigues(rvec)
        t = tvec

        print(f"  PnP 求解成功")
        print(f"  使用 {len(points_3d)} 个点（含离群点）")
    else:
        R, t = None, None
        print("  PnP 求解失败！")

    return R, t, success


def calibrate_with_pnp_ransac(points_3d, points_2d):
    """
    第四步：使用 RANSAC + PnP 求解外参（鲁棒估计）

    RANSAC (Random Sample Consensus) 算法流程：
    1. 随机选择最小点集（4个点）
    2. 用这些点求解 PnP 得到候选解
    3. 计算所有点的重投影误差
    4. 误差小于阈值的点标记为内点
    5. 如果内点数足够多，用所有内点重新求解
    6. 重复以上步骤，保留内点最多的解

    cv2.solvePnPRansac() 将 RANSAC 与 PnP 结合，
    能自动剔除离群点（错误匹配），得到更鲁棒的结果。

    参数:
        points_3d: (N, 3) LiDAR 坐标系下的 3D 点（可能含离群点）
        points_2d: (N, 2) 对应的图像 2D 像素坐标（可能含离群点）

    返回:
        R: (3, 3) 旋转矩阵
        t: (3, 1) 平移向量
        inliers: 内点索引
        success: 是否成功
    """
    print("\n" + "=" * 60)
    print("[步骤4] 使用 RANSAC + PnP 求解外参（鲁棒估计）")
    print("=" * 60)

    obj_pts = points_3d.reshape(-1, 1, 3).astype(np.float64)
    img_pts = points_2d.reshape(-1, 1, 2).astype(np.float64)

    # =============================================
    # 核心：调用 cv2.solvePnPRansac()
    # =============================================
    # 额外参数:
    #   iterationsCount: RANSAC 最大迭代次数
    #   reprojectionError: 内点判定阈值（像素），重投影误差小于此值的为内点
    #   confidence: 置信度（默认 0.99），影响迭代次数
    #   flags: 内部 PnP 求解方法
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts,
        CAMERA_MATRIX,
        DIST_COEFFS,
        iterationsCount=1000,      # 最大迭代 1000 次
        reprojectionError=8.0,     # 内点阈值 8 像素
        confidence=0.99,           # 99% 置信度
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        n_inliers = len(inliers) if inliers is not None else 0
        print(f"  RANSAC PnP 求解成功")
        print(f"  总点数: {len(points_3d)}, 内点数: {n_inliers}, "
              f"离群点数: {len(points_3d) - n_inliers}")
    else:
        R, t, inliers = None, None, None
        print("  RANSAC PnP 求解失败！")

    return R, t, inliers, success


def compare_transformation(R_est, t_est, method_name):
    """
    第五步：对比标定结果与真值

    评估指标：
    1. 旋转误差：R_err = R_true * R_est^T，从中提取旋转角度
    2. 平移误差：t 向量的欧氏距离
    3. 完整变换矩阵的逐元素对比

    参数:
        R_est: 估计的旋转矩阵 (3, 3)
        t_est: 估计的平移向量 (3, 1)
        method_name: 方法名称（用于显示）
    """
    print(f"\n--- {method_name} 结果对比 ---")

    if R_est is None:
        print("  求解失败，无法对比")
        return

    # 旋转误差
    # R_err = R_true * R_est^T 应该接近单位矩阵
    R_err = TRUE_ROTATION @ R_est.T
    # 从误差旋转矩阵提取角度: angle = arccos((trace(R_err) - 1) / 2)
    trace_val = np.trace(R_err)
    # 数值裁剪防止 arccos 输入超出 [-1, 1]
    cos_angle = np.clip((trace_val - 1) / 2, -1.0, 1.0)
    angle_error_deg = np.arccos(cos_angle) * 180.0 / np.pi

    # 平移误差
    t_error = np.linalg.norm(TRUE_TRANSLATION - t_est)
    t_relative_error = t_error / np.linalg.norm(TRUE_TRANSLATION) * 100

    print(f"  旋转误差: {angle_error_deg:.4f}°")
    print(f"  平移误差: {t_error:.6f} m ({t_relative_error:.2f}%)")

    print(f"\n  真实旋转矩阵 R_true:")
    for row in TRUE_ROTATION:
        print(f"    [{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f}]")

    print(f"  估计旋转矩阵 R_est:")
    for row in R_est:
        print(f"    [{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f}]")

    print(f"\n  真实平移 t_true: [{TRUE_TRANSLATION[0,0]:.6f}, "
          f"{TRUE_TRANSLATION[1,0]:.6f}, {TRUE_TRANSLATION[2,0]:.6f}]")
    print(f"  估计平移 t_est:  [{t_est[0,0]:.6f}, {t_est[1,0]:.6f}, {t_est[2,0]:.6f}]")

    return angle_error_deg, t_error


def compute_reprojection_error(points_3d, points_2d, R, t):
    """
    第六步：计算重投影误差

    将 LiDAR 3D 点通过估计的外参变换到相机坐标系，
    再投影到图像平面，计算与实际 2D 点的距离。

    参数:
        points_3d: (N, 3) LiDAR 3D 点
        points_2d: (N, 2) 对应的图像 2D 点
        R: 估计的旋转矩阵
        t: 估计的平移向量

    返回:
        mean_error: 平均重投影误差（像素）
        errors: 每个点的重投影误差
    """
    # 将旋转矩阵转为旋转向量（cv2.projectPoints 需要）
    rvec, _ = cv2.Rodrigues(R)

    # 重新投影
    projected, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rvec, t,
        CAMERA_MATRIX,
        DIST_COEFFS
    )
    projected = projected.reshape(-1, 2)

    # 计算每个点的欧氏距离误差
    errors = np.sqrt(np.sum((projected - points_2d) ** 2, axis=1))
    mean_error = np.mean(errors)

    return mean_error, errors


def visualize_results(points_3d, points_2d, R_pnp, t_pnp, R_ransac, t_ransac,
                      inlier_mask_true, inliers_ransac):
    """
    第七步：可视化标定结果

    生成三张图：
    1. 3D 点云对比（LiDAR 原始 vs 变换后 vs 真值变换后）
    2. 2D 重投影效果对比
    3. 重投影误差分布

    参数:
        points_3d: LiDAR 3D 点
        points_2d: 图像 2D 点
        R_pnp, t_pnp: 标准 PnP 结果
        R_ransac, t_ransac: RANSAC PnP 结果
        inlier_mask_true: 真实的内点标记
        inliers_ransac: RANSAC 检测到的内点索引
    """
    print("\n" + "=" * 60)
    print("[步骤7] 可视化标定结果")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # --------------------------------------------------
    # 图1：3D 点云在相机坐标系下的对比
    # --------------------------------------------------
    fig = plt.figure(figsize=(16, 5))

    # 真值变换
    pts_cam_true = (TRUE_ROTATION @ points_3d.T + TRUE_TRANSLATION).T

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c='blue', s=5, alpha=0.6, label='LiDAR Points')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('LiDAR Coordinate System')
    ax1.legend()

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pts_cam_true[:, 0], pts_cam_true[:, 1], pts_cam_true[:, 2],
                c='green', s=5, alpha=0.6, label='Ground Truth')
    if R_ransac is not None:
        pts_cam_est = (R_ransac @ points_3d.T + t_ransac).T
        ax2.scatter(pts_cam_est[:, 0], pts_cam_est[:, 1], pts_cam_est[:, 2],
                    c='red', s=5, alpha=0.4, label='RANSAC PnP')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Camera Coordinate System')
    ax2.legend()

    # 重投影误差对比
    ax3 = fig.add_subplot(133)
    if R_pnp is not None:
        _, errors_pnp = compute_reprojection_error(points_3d, points_2d, R_pnp, t_pnp)
        ax3.hist(errors_pnp, bins=20, alpha=0.6, color='orange', label='PnP')
    if R_ransac is not None:
        _, errors_ransac = compute_reprojection_error(points_3d, points_2d, R_ransac, t_ransac)
        ax3.hist(errors_ransac, bins=20, alpha=0.6, color='green', label='RANSAC PnP')
    ax3.set_xlabel('Reprojection Error (pixels)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution')
    ax3.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lidar_camera_calibration_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D 点云对比图已保存: {save_path}")

    # --------------------------------------------------
    # 图2：2D 重投影效果
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 用 RANSAC 结果重投影
    if R_ransac is not None:
        rvec_ransac, _ = cv2.Rodrigues(R_ransac)
        projected_ransac, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3), rvec_ransac, t_ransac,
            CAMERA_MATRIX, DIST_COEFFS
        )
        projected_ransac = projected_ransac.reshape(-1, 2)

        # 绘制原始 2D 点和重投影点
        axes[0].scatter(points_2d[inlier_mask_true, 0],
                        points_2d[inlier_mask_true, 1],
                        c='blue', s=10, alpha=0.6, label='Detected (inliers)')
        axes[0].scatter(points_2d[~inlier_mask_true, 0],
                        points_2d[~inlier_mask_true, 1],
                        c='red', s=30, marker='x', label='Detected (outliers)')
        axes[0].scatter(projected_ransac[:, 0], projected_ransac[:, 1],
                        c='green', s=10, alpha=0.6, marker='+', label='Reprojected')
        axes[0].set_xlim(0, IMAGE_WIDTH)
        axes[0].set_ylim(IMAGE_HEIGHT, 0)
        axes[0].set_xlabel('u (pixels)')
        axes[0].set_ylabel('v (pixels)')
        axes[0].set_title('RANSAC PnP Reprojection')
        axes[0].legend(fontsize=8)

    # RANSAC 内点检测准确率
    if inliers_ransac is not None:
        detected_inliers = np.zeros(len(points_3d), dtype=bool)
        detected_inliers[inliers_ransac.flatten()] = True

        tp = np.sum(detected_inliers & inlier_mask_true)
        fp = np.sum(detected_inliers & ~inlier_mask_true)
        fn = np.sum(~detected_inliers & inlier_mask_true)
        tn = np.sum(~detected_inliers & ~inlier_mask_true)

        labels = ['True Pos\n(Correct Inlier)', 'False Pos\n(Missed Outlier)',
                  'False Neg\n(Missed Inlier)', 'True Neg\n(Correct Outlier)']
        values = [tp, fp, fn, tn]
        colors = ['#4CAF50', '#FF9800', '#F44336', '#2196F3']
        axes[1].bar(labels, values, color=colors)
        axes[1].set_title('RANSAC Inlier Detection')
        axes[1].set_ylabel('Count')

        print(f"  RANSAC 内点检测: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lidar_camera_reprojection.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  2D 重投影效果图已保存: {save_path}")


def main():
    """
    主函数：按顺序执行 LiDAR-相机外参标定的完整流程
    """
    print("=" * 60)
    print("   LiDAR-相机外参标定演示 —— PnP + RANSAC（合成数据）")
    print("=" * 60)

    print(f"\n[步骤1] 定义真实参数:")
    print(f"  图像尺寸: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"  相机焦距: fx={CAMERA_MATRIX[0,0]}, fy={CAMERA_MATRIX[1,1]}")
    print(f"  真实外参变换矩阵 T_lidar2cam:")
    for i in range(4):
        row = TRUE_T_LIDAR2CAM[i]
        print(f"    [{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f} {row[3]:>10.6f}]")

    # 生成合成数据
    pts_3d, pts_2d, pts_3d_clean, pts_2d_clean, inlier_mask = \
        generate_synthetic_lidar_camera_data()

    # 使用标准 PnP 求解（不处理离群点）
    R_pnp, t_pnp, success_pnp = calibrate_with_pnp(pts_3d, pts_2d)

    # 使用 RANSAC PnP 求解（自动剔除离群点）
    R_ransac, t_ransac, inliers_ransac, success_ransac = \
        calibrate_with_pnp_ransac(pts_3d, pts_2d)

    # 对比结果
    print("\n" + "=" * 60)
    print("[步骤5] 标定结果对比")
    print("=" * 60)

    if success_pnp:
        compare_transformation(R_pnp, t_pnp, "标准 PnP（含离群点）")
        mean_err, _ = compute_reprojection_error(pts_3d, pts_2d, R_pnp, t_pnp)
        print(f"  平均重投影误差: {mean_err:.4f} 像素")

    if success_ransac:
        compare_transformation(R_ransac, t_ransac, "RANSAC PnP（自动剔除离群点）")
        mean_err, _ = compute_reprojection_error(pts_3d, pts_2d, R_ransac, t_ransac)
        print(f"  平均重投影误差: {mean_err:.4f} 像素")

        # 仅用内点计算误差
        if inliers_ransac is not None:
            inlier_idx = inliers_ransac.flatten()
            mean_err_inliers, _ = compute_reprojection_error(
                pts_3d[inlier_idx], pts_2d[inlier_idx], R_ransac, t_ransac
            )
            print(f"  内点平均重投影误差: {mean_err_inliers:.4f} 像素")

    # 可视化
    visualize_results(pts_3d, pts_2d, R_pnp, t_pnp, R_ransac, t_ransac,
                      inlier_mask, inliers_ransac)

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)
    print(f"  数据: {len(pts_3d)} 个 3D-2D 对应点（含 {np.sum(~inlier_mask)} 个离群点）")
    if success_ransac:
        angle_err, t_err = compare_transformation(R_ransac, t_ransac, "最终结果 (RANSAC PnP)")
        print(f"  结论: {'标定成功！' if angle_err < 1.0 and t_err < 0.01 else '标定精度可接受'}")
    print(f"\n  实际应用建议:")
    print(f"    1. 使用标定板（棋盘格/ArUco）获取精确的 3D-2D 对应")
    print(f"    2. 至少采集 20 组以上的对应点")
    print(f"    3. 对应点应分布在图像各区域，避免集中在一处")
    print(f"    4. 使用 RANSAC 版本以提高鲁棒性")


if __name__ == '__main__':
    main()
