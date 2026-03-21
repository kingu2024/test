"""
鱼眼镜头畸变矫正完整案例（合成数据）

本脚本演示鱼眼镜头（超广角，FOV > 120°）的标定与矫正。
鱼眼镜头的畸变模型与常规镜头不同，使用等距投影模型（Equidistant Model），
OpenCV 提供了专门的 cv2.fisheye 模块来处理。

鱼眼镜头 vs 常规镜头的核心区别:
    常规镜头: r = f * tan(θ)         (透视投影)
    鱼眼镜头: r = f * θ              (等距投影，最常见)
    其他模型: r = 2f * sin(θ/2)      (等立体角投影)
              r = 2f * tan(θ/2)      (体视投影)
              r = f * sin(θ)         (正交投影)

    其中 θ 是入射光线与光轴的夹角，r 是像点到主点的距离。
    常规镜头当 θ→90° 时 r→∞，而鱼眼镜头可以捕捉 180° 甚至更大的视场。

数据来源:
    本脚本使用程序自动生成的合成数据，无需外部数据集。
    如需真实鱼眼数据:
    - OpenCV 鱼眼标定样例:
      https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
    - GoPro / Insta360 等运动相机拍摄的棋盘格图像
    - KITTI-360 数据集 (含鱼眼相机): https://www.cvlibs.net/datasets/kitti-360/

运行方式:
    cd calibration_algorithms
    python distortion_correction/fisheye_undistortion.py

作者: Claude (AI Assistant)
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ============================================================================
# 第一步：定义鱼眼相机参数（Ground Truth）
# ============================================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 鱼眼相机内参矩阵
# 鱼眼镜头的焦距通常较短（对应更大的 FOV）
FISHEYE_K = np.array([
    [280.0,   0.0, 320.0],
    [  0.0, 280.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

# ============================================================================
# 鱼眼畸变系数 [k1, k2, k3, k4]
# ============================================================================
# OpenCV 鱼眼模型使用 Kannala-Brandt 等距投影模型:
#   θ_d = θ * (1 + k1*θ^2 + k2*θ^4 + k3*θ^6 + k4*θ^8)
# 其中 θ 是入射角，θ_d 是畸变后的角度
# 注意：鱼眼模型没有切向畸变参数（p1, p2），只有径向畸变 k1~k4
FISHEYE_D = np.array([[-0.05], [0.01], [-0.005], [0.001]], dtype=np.float64)

# 棋盘格参数（用于鱼眼标定）
BOARD_ROWS = 6
BOARD_COLS = 9
SQUARE_SIZE = 0.025  # 25mm

# 合成数据参数
NUM_VIEWS = 25
NOISE_STD = 0.3  # 像素噪声


def generate_fisheye_object_points():
    """
    第二步：生成棋盘格 3D 世界坐标

    与常规标定相同，棋盘放在 Z=0 平面。

    返回:
        objp: (N, 1, 3) float64 数组 —— 注意鱼眼标定要求 (N,1,3) 格式
    """
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 1, 3), dtype=np.float64)
    pts = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2).astype(np.float64)
    objp[:, 0, :2] = pts * SQUARE_SIZE
    return objp


def fisheye_project_points(objp_flat, rvec, tvec, K, D):
    """
    第三步：鱼眼投影模型 —— 将 3D 点投影到鱼眼图像

    鱼眼投影与常规针孔投影的区别:
    1. 常规针孔: 透视投影 r = f * tan(θ)
    2. 鱼眼等距: r = f * θ_d，其中 θ_d = θ(1 + k1*θ^2 + k2*θ^4 + ...)

    完整投影流程:
    1. 世界坐标 → 相机坐标: P_cam = R * P_w + t
    2. 计算入射角: θ = arctan(sqrt(x^2+y^2) / z)
    3. 施加鱼眼畸变: θ_d = θ(1 + k1*θ^2 + k2*θ^4 + k3*θ^6 + k4*θ^8)
    4. 计算畸变归一化坐标:
       x_d = (θ_d / r) * x,  y_d = (θ_d / r) * y,  其中 r = sqrt(x^2+y^2)
    5. 像素坐标: u = fx*x_d + cx, v = fy*y_d + cy

    参数:
        objp_flat: (N, 3) 3D 点
        rvec: (3,1) 旋转向量
        tvec: (3,1) 平移向量
        K: 内参矩阵
        D: 鱼眼畸变系数 [k1, k2, k3, k4]

    返回:
        img_pts: (N, 1, 2) 投影的 2D 像素坐标
    """
    # 旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 世界坐标 → 相机坐标
    P_cam = (R @ objp_flat.T + tvec).T  # (N, 3)

    x = P_cam[:, 0] / P_cam[:, 2]  # 归一化 x
    y = P_cam[:, 1] / P_cam[:, 2]  # 归一化 y

    # 计算入射角
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)  # 入射角

    # 鱼眼畸变模型: θ_d = θ(1 + k1*θ^2 + k2*θ^4 + k3*θ^6 + k4*θ^8)
    k1, k2, k3, k4 = D.flatten()
    theta2 = theta * theta
    theta_d = theta * (1 + k1 * theta2 + k2 * theta2 ** 2 +
                        k3 * theta2 ** 3 + k4 * theta2 ** 4)

    # 计算畸变后的归一化坐标
    # 当 r → 0 时，scale → 1（避免除零）
    scale = np.where(r > 1e-8, theta_d / r, 1.0)
    x_d = x * scale
    y_d = y * scale

    # 像素坐标
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x_d + cx
    v = fy * y_d + cy

    img_pts = np.stack([u, v], axis=1).reshape(-1, 1, 2)
    return img_pts


def generate_synthetic_fisheye_data():
    """
    第四步：生成鱼眼相机的合成标定数据

    类似常规相机标定，但使用鱼眼投影模型。
    鱼眼镜头的视场角更大，因此棋盘可以在更大的角度范围内出现。

    返回:
        object_points: 3D 点列表
        image_points:  2D 投影点列表
    """
    print("\n" + "=" * 60)
    print("[步骤4] 生成鱼眼合成标定数据")
    print("=" * 60)

    objp = generate_fisheye_object_points()
    objp_flat = objp.reshape(-1, 3)

    object_points = []
    image_points = []

    rng = np.random.RandomState(42)

    for i in range(NUM_VIEWS):
        # 生成随机位姿（鱼眼视场大，允许更大的角度范围）
        angles = rng.uniform(-45, 45, size=3) * np.pi / 180.0
        rvec = angles.reshape(3, 1).astype(np.float64)

        tx = rng.uniform(-0.08, 0.08)
        ty = rng.uniform(-0.08, 0.08)
        tz = rng.uniform(0.2, 0.6)
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

        # 使用鱼眼模型投影
        img_pts = fisheye_project_points(objp_flat, rvec, tvec, FISHEYE_K, FISHEYE_D)

        # 添加噪声
        noise = rng.normal(0, NOISE_STD, img_pts.shape).astype(np.float64)
        img_pts_noisy = img_pts + noise

        # 检查是否在图像内
        pts = img_pts_noisy.reshape(-1, 2)
        in_bounds = np.all(
            (pts[:, 0] >= 0) & (pts[:, 0] < IMAGE_WIDTH) &
            (pts[:, 1] >= 0) & (pts[:, 1] < IMAGE_HEIGHT)
        )

        if in_bounds:
            object_points.append(objp.copy())
            image_points.append(img_pts_noisy.astype(np.float64))

    print(f"  生成 {NUM_VIEWS} 个视角，{len(object_points)} 个有效")
    return object_points, image_points


def run_fisheye_calibration(object_points, image_points):
    """
    第五步：执行鱼眼相机标定

    使用 cv2.fisheye.calibrate() 函数，这是专门为鱼眼镜头设计的标定函数。
    与常规 calibrateCamera() 的区别:
    1. 使用 Kannala-Brandt 畸变模型（4 个参数 k1~k4）
    2. 不包含切向畸变
    3. 对大视场角更准确

    参数:
        object_points: 3D 点列表
        image_points:  2D 点列表

    返回:
        K_cal: 标定得到的内参
        D_cal: 标定得到的畸变系数
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
        rms:   RMS 重投影误差
    """
    print("\n" + "=" * 60)
    print("[步骤5] 执行鱼眼相机标定")
    print("=" * 60)

    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    N = len(object_points)

    # 初始化输出变量
    K_cal = np.zeros((3, 3), dtype=np.float64)
    D_cal = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]

    # =============================================
    # 核心：cv2.fisheye.calibrate()
    # =============================================
    # 参数:
    #   objectPoints: 3D 点列表，每个元素 shape=(N,1,3)
    #   imagePoints:  2D 点列表，每个元素 shape=(N,1,2)
    #   image_size:   图像尺寸
    #   K:            输出内参矩阵 (3x3)
    #   D:            输出畸变系数 (4x1) [k1,k2,k3,k4]
    #   rvecs:        输出旋转向量
    #   tvecs:        输出平移向量
    #   flags:        标定选项
    #     CALIB_RECOMPUTE_EXTRINSIC: 每次内参更新后重算外参
    #     CALIB_CHECK_COND:          检查条件数，避免数值不稳定
    #     CALIB_FIX_SKEW:            固定倾斜因子为 0
    #   criteria:     迭代终止条件
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_FIX_SKEW
    )

    rms, K_cal, D_cal, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        K_cal,
        D_cal,
        rvecs,
        tvecs,
        flags=calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    print(f"  鱼眼标定完成！RMS 重投影误差: {rms:.4f} 像素")
    return K_cal, D_cal, rvecs, tvecs, rms


def compare_fisheye_calibration(K_cal, D_cal):
    """
    第六步：对比鱼眼标定结果与真值
    """
    print("\n" + "=" * 60)
    print("[步骤6] 鱼眼标定结果对比")
    print("=" * 60)

    print("\n--- 内参矩阵 K ---")
    param_names = ['fx', 'fy', 'cx', 'cy']
    true_vals = [FISHEYE_K[0, 0], FISHEYE_K[1, 1], FISHEYE_K[0, 2], FISHEYE_K[1, 2]]
    cal_vals = [K_cal[0, 0], K_cal[1, 1], K_cal[0, 2], K_cal[1, 2]]

    print(f"{'Param':<6} {'True':>10} {'Calibrated':>12} {'Error':>10} {'Rel%':>10}")
    print("-" * 50)
    for name, tv, cv_val in zip(param_names, true_vals, cal_vals):
        err = abs(cv_val - tv)
        rel = err / abs(tv) * 100 if tv != 0 else 0
        print(f"{name:<6} {tv:>10.4f} {cv_val:>12.4f} {err:>10.4f} {rel:>9.4f}%")

    print("\n--- Fisheye Distortion [k1, k2, k3, k4] ---")
    d_names = ['k1', 'k2', 'k3', 'k4']
    print(f"{'Param':<6} {'True':>12} {'Calibrated':>12} {'Error':>12}")
    print("-" * 44)
    for name, tv, cv_val in zip(d_names, FISHEYE_D.flatten(), D_cal.flatten()):
        print(f"{name:<6} {tv:>12.6f} {cv_val:>12.6f} {abs(cv_val-tv):>12.6f}")


def generate_fisheye_distorted_image(K, D):
    """
    第七步：生成鱼眼畸变的合成图像

    创建一个标准网格，然后使用鱼眼模型对其施加畸变。
    鱼眼畸变的典型特征：图像中心几乎不变形，
    但越靠近边缘弯曲越明显（强烈的桶形畸变）。

    参数:
        K: 鱼眼内参
        D: 鱼眼畸变系数

    返回:
        original: 原始网格图像
        distorted: 鱼眼畸变后的图像
    """
    h, w = IMAGE_HEIGHT, IMAGE_WIDTH
    original = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 绘制网格
    for y in range(0, h, 30):
        cv2.line(original, (0, y), (w - 1, y), (180, 180, 180), 1)
    for x in range(0, w, 30):
        cv2.line(original, (x, 0), (x, h - 1), (180, 180, 180), 1)

    # 绘制同心圆（鱼眼特征明显）
    cx_i, cy_i = w // 2, h // 2
    for r in range(30, max(w, h), 40):
        cv2.circle(original, (cx_i, cy_i), r, (200, 200, 200), 1)

    # 绘制径向线
    for angle in range(0, 360, 15):
        rad = angle * np.pi / 180
        x2 = int(cx_i + max(w, h) * np.cos(rad))
        y2 = int(cy_i + max(w, h) * np.sin(rad))
        cv2.line(original, (cx_i, cy_i), (x2, y2), (220, 220, 220), 1)

    # 标记点
    for y in range(30, h, 60):
        for x in range(30, w, 60):
            cv2.circle(original, (x, y), 3, (0, 100, 200), -1)

    # 施加鱼眼畸变
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, k3, k4 = D.flatten()

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    u_coords = np.arange(w, dtype=np.float64)
    v_coords = np.arange(h, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    x_norm = (u_grid - cx) / fx
    y_norm = (v_grid - cy) / fy

    r = np.sqrt(x_norm ** 2 + y_norm ** 2)
    theta = np.arctan(r)

    theta2 = theta * theta
    theta_d = theta * (1 + k1 * theta2 + k2 * theta2 ** 2 +
                        k3 * theta2 ** 3 + k4 * theta2 ** 4)

    scale = np.where(r > 1e-8, theta_d / r, 1.0)
    x_d = x_norm * scale
    y_d = y_norm * scale

    map_x = (x_d * fx + cx).astype(np.float32)
    map_y = (y_d * fy + cy).astype(np.float32)

    distorted = cv2.remap(original, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))

    return original, distorted


def undistort_fisheye_methods(distorted, K_cal, D_cal):
    """
    第八步：鱼眼去畸变的三种方法

    鱼眼去畸变有三种常用策略:
    1. cv2.fisheye.undistortImage() —— 直接去畸变（保持鱼眼视场）
    2. cv2.fisheye.initUndistortRectifyMap() + remap() —— 映射表法
    3. 调整 new_K 的焦距实现不同的去畸变视场效果

    参数:
        distorted: 鱼眼畸变图像
        K_cal: 标定内参
        D_cal: 标定畸变系数

    返回:
        results: 各方法的去畸变结果字典
    """
    print("\n" + "=" * 60)
    print("[步骤8] 鱼眼去畸变 —— 三种策略")
    print("=" * 60)

    h, w = distorted.shape[:2]
    results = {}

    # =============================================
    # 策略1：使用原始焦距去畸变
    # =============================================
    # 去畸变后视场角会缩小，但直线会恢复
    new_K1 = K_cal.copy()

    # cv2.fisheye.initUndistortRectifyMap() 参数:
    #   K:      原始内参
    #   D:      畸变系数
    #   R:      旋转矩阵（通常为单位矩阵）
    #   P:      新的投影矩阵（控制输出内参）
    #   size:   输出尺寸
    #   m1type: 映射表类型
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_cal, D_cal, np.eye(3), new_K1, (w, h), cv2.CV_32FC1
    )
    undist1 = cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    results['Original FOV'] = undist1
    print("  Strategy 1: Original focal length undistortion done")

    # =============================================
    # 策略2：缩小焦距以保留更多视场
    # =============================================
    # 减小焦距 = 输出图像覆盖更大的视场角
    # 但会牺牲分辨率
    new_K2 = K_cal.copy()
    new_K2[0, 0] *= 0.5  # fx 减半
    new_K2[1, 1] *= 0.5  # fy 减半

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_cal, D_cal, np.eye(3), new_K2, (w, h), cv2.CV_32FC1
    )
    undist2 = cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    results['Wide FOV (0.5x focal)'] = undist2
    print("  Strategy 2: Reduced focal length (wider FOV) done")

    # =============================================
    # 策略3：增大焦距进行局部放大
    # =============================================
    # 增大焦距 = 只保留中心区域，但分辨率更高
    new_K3 = K_cal.copy()
    new_K3[0, 0] *= 1.5
    new_K3[1, 1] *= 1.5

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_cal, D_cal, np.eye(3), new_K3, (w, h), cv2.CV_32FC1
    )
    undist3 = cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    results['Narrow FOV (1.5x focal)'] = undist3
    print("  Strategy 3: Increased focal length (narrower FOV) done")

    # =============================================
    # 策略4：cv2.fisheye.undistortImage() 一步完成
    # =============================================
    undist4 = cv2.fisheye.undistortImage(distorted, K_cal, D_cal, Knew=new_K1)
    results['undistortImage()'] = undist4
    print("  Strategy 4: cv2.fisheye.undistortImage() done")

    return results


def undistort_fisheye_points(K_cal, D_cal):
    """
    第九步：鱼眼特征点去畸变

    cv2.fisheye.undistortPoints() 将鱼眼畸变的 2D 点坐标
    转换为无畸变的归一化坐标或像素坐标。

    与常规 undistortPoints 类似，但使用鱼眼畸变逆模型。
    """
    print("\n" + "=" * 60)
    print("[步骤9] 鱼眼特征点去畸变")
    print("=" * 60)

    # 生成测试点
    test_points = np.array([
        [320, 240],  # 中心
        [100, 100],  # 左上
        [540, 100],  # 右上
        [100, 380],  # 左下
        [540, 380],  # 右下
        [50, 240],   # 最左
        [590, 240],  # 最右
    ], dtype=np.float64).reshape(-1, 1, 2)

    # =============================================
    # cv2.fisheye.undistortPoints()
    # =============================================
    # 参数:
    #   distorted: 畸变的 2D 点 (N,1,2)
    #   K: 内参矩阵
    #   D: 鱼眼畸变系数 (4x1)
    #   R: 旋转矩阵 (可选)
    #   P: 新的投影矩阵 (可选，提供时输出像素坐标)
    undist_normalized = cv2.fisheye.undistortPoints(
        test_points, K_cal, D_cal
    )  # 输出归一化坐标

    undist_pixel = cv2.fisheye.undistortPoints(
        test_points, K_cal, D_cal, R=np.eye(3), P=K_cal
    )  # 输出像素坐标

    print(f"  输入点 (畸变像素坐标) → 输出 (去畸变像素坐标):")
    for i in range(len(test_points)):
        pt_in = test_points[i, 0]
        pt_out = undist_pixel[i, 0]
        print(f"    ({pt_in[0]:>6.1f}, {pt_in[1]:>6.1f}) → "
              f"({pt_out[0]:>8.2f}, {pt_out[1]:>8.2f})")

    return test_points, undist_pixel


def visualize_results(original, distorted, undist_results):
    """
    第十步：可视化所有结果
    """
    print("\n" + "=" * 60)
    print("[步骤10] 可视化结果")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # 图1: 鱼眼畸变 vs 各种去畸变策略
    n_results = len(undist_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original (No Distortion)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Fisheye Distorted')
    axes[0, 1].axis('off')

    for idx, (name, img) in enumerate(undist_results.items()):
        row = (idx + 2) // 3
        col = (idx + 2) % 3
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'Undistorted: {name}')
        axes[row, col].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fisheye_undistortion_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  鱼眼去畸变对比图已保存: {save_path}")

    # 图2: 常规 vs 鱼眼畸变模型对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制畸变曲线: 入射角 θ vs 像点距离 r
    theta_range = np.linspace(0, 80, 200) * np.pi / 180  # 0~80°

    # 常规针孔（透视投影）
    f = 280  # 与鱼眼焦距相同
    r_pinhole = f * np.tan(theta_range)

    # 理想鱼眼（等距投影，无畸变）
    r_equidist = f * theta_range

    # 带畸变的鱼眼
    k1, k2, k3, k4 = FISHEYE_D.flatten()
    theta2 = theta_range ** 2
    theta_d = theta_range * (1 + k1 * theta2 + k2 * theta2 ** 2 +
                              k3 * theta2 ** 3 + k4 * theta2 ** 4)
    r_fisheye = f * theta_d

    ax1.plot(theta_range * 180 / np.pi, r_pinhole, 'b-', label='Pinhole: r = f*tan(theta)', linewidth=2)
    ax1.plot(theta_range * 180 / np.pi, r_equidist, 'g--', label='Equidistant: r = f*theta', linewidth=2)
    ax1.plot(theta_range * 180 / np.pi, r_fisheye, 'r-.', label='Fisheye (with distortion)', linewidth=2)
    ax1.set_xlabel('Incident Angle theta (degrees)')
    ax1.set_ylabel('Image Radius r (pixels)')
    ax1.set_title('Projection Model Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 600)

    # 畸变量曲线
    with np.errstate(invalid='ignore'):
        distortion_amount = np.where(
            theta_range > 1e-10,
            (theta_d - theta_range) / theta_range * 100,
            0.0
        )

    ax2.plot(theta_range * 180 / np.pi, distortion_amount, 'r-', linewidth=2)
    ax2.set_xlabel('Incident Angle theta (degrees)')
    ax2.set_ylabel('Distortion Amount (%)')
    ax2.set_title('Fisheye Distortion Amount vs Incident Angle')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fisheye_model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  投影模型对比图已保存: {save_path}")


def main():
    print("=" * 60)
    print("       鱼眼镜头畸变矫正演示（合成数据）")
    print("=" * 60)

    print(f"\n[步骤1] 鱼眼相机参数:")
    print(f"  图像尺寸: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"  焦距: fx={FISHEYE_K[0,0]}, fy={FISHEYE_K[1,1]}")
    print(f"  畸变系数: k1={FISHEYE_D[0,0]}, k2={FISHEYE_D[1,0]}, "
          f"k3={FISHEYE_D[2,0]}, k4={FISHEYE_D[3,0]}")

    # 生成合成标定数据
    obj_pts, img_pts = generate_synthetic_fisheye_data()

    if len(obj_pts) < 5:
        print("有效视角不足，调整参数后重试")
        return

    # 鱼眼标定
    K_cal, D_cal, rvecs, tvecs, rms = run_fisheye_calibration(obj_pts, img_pts)

    # 对比标定结果
    compare_fisheye_calibration(K_cal, D_cal)

    # 生成鱼眼畸变图像
    print("\n" + "=" * 60)
    print("[步骤7] 生成鱼眼畸变合成图像")
    print("=" * 60)
    original, distorted = generate_fisheye_distorted_image(K_cal, D_cal)
    print("  鱼眼畸变图像生成完成")

    # 多种去畸变策略
    undist_results = undistort_fisheye_methods(distorted, K_cal, D_cal)

    # 特征点去畸变
    undistort_fisheye_points(K_cal, D_cal)

    # 可视化
    visualize_results(original, distorted, undist_results)

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)
    print(f"  鱼眼标定 RMS 重投影误差: {rms:.4f} 像素")
    print(f"  焦距误差: fx={abs(K_cal[0,0]-FISHEYE_K[0,0])/FISHEYE_K[0,0]*100:.4f}%, "
          f"fy={abs(K_cal[1,1]-FISHEYE_K[1,1])/FISHEYE_K[1,1]*100:.4f}%")
    print(f"\n  鱼眼去畸变策略选择:")
    print(f"    原始焦距:     保留中心区域，直线恢复好")
    print(f"    缩小焦距:     保留更多视场，适合全景应用")
    print(f"    增大焦距:     中心区域放大，适合目标检测")
    print(f"    undistortImage: 一步完成，最简单")
    print(f"\n  鱼眼 vs 常规镜头:")
    print(f"    畸变模型不同: 鱼眼用 θ_d = θ(1+k1θ²+...), 常规用 r_d = r(1+k1r²+...)")
    print(f"    API 不同: 鱼眼用 cv2.fisheye.*, 常规用 cv2.*")
    print(f"    视场角: 鱼眼 >120° (可达 180°+), 常规通常 <90°")


if __name__ == '__main__':
    main()
