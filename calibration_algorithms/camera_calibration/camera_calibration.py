"""
相机标定完整案例 —— 基于 Zhang 标定法（棋盘格合成数据）

本脚本演示如何使用 OpenCV 的 calibrateCamera() 函数进行相机内参标定。
为了让用户无需真实相机即可运行，脚本会自动生成合成的棋盘格角点数据。

运行方式:
    cd calibration_algorithms
    python camera_calibration/camera_calibration.py

数据来源:
    本脚本使用程序自动生成的合成数据，无需外部数据集。
    如需使用真实数据，可参考 OpenCV 官方棋盘格图片:
    https://github.com/opencv/opencv/blob/master/doc/pattern.png
    打印后用相机从不同角度拍摄 15-20 张照片即可。

作者: Claude (AI Assistant)
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合无显示器环境
import matplotlib.pyplot as plt
import os


# ============================================================================
# 第一步：定义相机的"真实"参数（Ground Truth）
# ============================================================================
# 这些参数在真实场景中是未知的，是我们要标定出来的目标。
# 这里我们预先定义它们，用于生成合成数据，并在标定完成后验证精度。

# 图像分辨率
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 相机内参矩阵 K (3x3)
# fx, fy: 焦距（单位：像素），等于物理焦距 f 除以像素物理尺寸
# cx, cy: 主点坐标，理想情况下在图像中心
TRUE_CAMERA_MATRIX = np.array([
    [800.0,   0.0, 320.0],   # [fx,  0, cx]
    [  0.0, 800.0, 240.0],   # [ 0, fy, cy]
    [  0.0,   0.0,   1.0]    # [ 0,  0,  1]
], dtype=np.float64)

# 畸变系数 [k1, k2, p1, p2, k3]
# k1, k2, k3: 径向畸变系数（桶形/枕形畸变）
# p1, p2: 切向畸变系数（由镜头与传感器不平行引起）
TRUE_DIST_COEFFS = np.array([0.1, -0.25, 0.001, -0.001, 0.05], dtype=np.float64)

# 棋盘格参数
BOARD_ROWS = 6       # 棋盘格内角点行数
BOARD_COLS = 9       # 棋盘格内角点列数
SQUARE_SIZE = 0.025  # 每个方格的边长（单位：米，这里是 25mm）

# 合成数据参数
NUM_VIEWS = 20       # 模拟的拍摄视角数量（建议 >= 15）
NOISE_STD = 0.5      # 角点检测噪声的标准差（单位：像素）


def generate_chessboard_object_points():
    """
    第二步：生成棋盘格的 3D 世界坐标点

    棋盘格放在世界坐标系的 Z=0 平面上。
    每个角点的坐标为 (col * SQUARE_SIZE, row * SQUARE_SIZE, 0)。

    返回:
        objp: shape=(BOARD_ROWS*BOARD_COLS, 3) 的 float32 数组
              每行是一个角点的 (X, Y, Z) 世界坐标
    """
    # 创建角点网格坐标
    # 例如对于 6x9 棋盘: (0,0,0), (0.025,0,0), (0.05,0,0), ...
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # 乘以方格实际尺寸，得到真实物理坐标

    print(f"[步骤2] 棋盘格角点数: {len(objp)}")
    print(f"         坐标范围: X=[0, {(BOARD_COLS-1)*SQUARE_SIZE:.3f}]m, "
          f"Y=[0, {(BOARD_ROWS-1)*SQUARE_SIZE:.3f}]m, Z=0")
    return objp


def generate_random_pose(index):
    """
    第三步：为每个视角生成随机的旋转向量和平移向量

    模拟相机从不同角度观察棋盘格的场景。
    - 旋转角度范围：每个轴 ±30°（确保足够的角度变化）
    - 平移范围：棋盘在相机前方 0.3m~0.8m 处

    参数:
        index: 视角编号（用于设置随机种子保证可重复性）

    返回:
        rvec: (3,1) 旋转向量（Rodrigues 表示）
        tvec: (3,1) 平移向量
    """
    rng = np.random.RandomState(42 + index)  # 固定种子保证可重复

    # 生成随机旋转（每个轴 ±30°）
    # 旋转向量的模长 = 旋转角度（弧度），方向 = 旋转轴
    angles = rng.uniform(-30, 30, size=3) * np.pi / 180.0
    rvec = angles.reshape(3, 1).astype(np.float64)

    # 生成随机平移
    # tx, ty: 小范围偏移，模拟棋盘不在画面正中央
    # tz: 棋盘到相机的距离（0.3~0.8m）
    tx = rng.uniform(-0.05, 0.05)
    ty = rng.uniform(-0.05, 0.05)
    tz = rng.uniform(0.3, 0.8)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    return rvec, tvec


def generate_synthetic_data():
    """
    第四步：生成合成的标定数据

    流程:
    1. 创建棋盘格 3D 坐标
    2. 对每个视角：
       a. 生成随机位姿（旋转+平移）
       b. 使用 cv2.projectPoints() 将 3D 点投影到 2D 图像平面
       c. 添加高斯噪声模拟真实角点检测的不确定性
       d. 检查投影点是否在图像范围内
    3. 只保留所有角点都在图像内的视角

    返回:
        object_points: 每个有效视角对应的 3D 点列表
        image_points:  每个有效视角对应的 2D 投影点列表
        valid_rvecs:   有效视角的旋转向量列表
        valid_tvecs:   有效视角的平移向量列表
    """
    print("\n" + "=" * 60)
    print("[步骤4] 生成合成标定数据")
    print("=" * 60)

    # 获取棋盘格 3D 坐标
    objp = generate_chessboard_object_points()

    object_points = []  # 存储所有有效视角的 3D 点
    image_points = []   # 存储所有有效视角的 2D 投影点
    valid_rvecs = []
    valid_tvecs = []

    rng = np.random.RandomState(123)  # 噪声的随机种子

    for i in range(NUM_VIEWS):
        # 生成该视角的位姿
        rvec, tvec = generate_random_pose(i)

        # =============================================
        # 核心：使用 cv2.projectPoints() 进行 3D→2D 投影
        # =============================================
        # 该函数实现了完整的相机投影模型：
        # 1. 将世界坐标系的点通过 [R|t] 变换到相机坐标系
        # 2. 进行透视除法得到归一化坐标
        # 3. 施加畸变模型
        # 4. 通过内参矩阵 K 得到像素坐标
        #
        # 数学公式: s * [u, v, 1]^T = K * (distort(R * P + t))
        img_pts, _ = cv2.projectPoints(
            objp,                # 3D 世界坐标点
            rvec,                # 旋转向量（3x1，Rodrigues 表示）
            tvec,                # 平移向量（3x1）
            TRUE_CAMERA_MATRIX,  # 相机内参矩阵 K（3x3）
            TRUE_DIST_COEFFS     # 畸变系数 [k1,k2,p1,p2,k3]
        )

        # img_pts 的形状为 (N, 1, 2)，即每个点是 (u, v) 像素坐标
        img_pts = img_pts.reshape(-1, 2)

        # 添加高斯噪声，模拟亚像素角点检测的误差
        noise = rng.normal(0, NOISE_STD, img_pts.shape)
        img_pts_noisy = img_pts + noise

        # 检查所有点是否在图像范围内
        in_bounds = np.all(
            (img_pts_noisy[:, 0] >= 0) & (img_pts_noisy[:, 0] < IMAGE_WIDTH) &
            (img_pts_noisy[:, 1] >= 0) & (img_pts_noisy[:, 1] < IMAGE_HEIGHT)
        )

        if in_bounds:
            object_points.append(objp.copy())
            # OpenCV 要求 image_points 的形状为 (N, 1, 2) 且为 float32
            image_points.append(img_pts_noisy.reshape(-1, 1, 2).astype(np.float32))
            valid_rvecs.append(rvec)
            valid_tvecs.append(tvec)

    print(f"  生成了 {NUM_VIEWS} 个视角，其中 {len(object_points)} 个有效")
    print(f"  噪声标准差: {NOISE_STD} 像素")
    return object_points, image_points, valid_rvecs, valid_tvecs


def run_calibration(object_points, image_points):
    """
    第五步：执行相机标定（Zhang 方法）

    使用 cv2.calibrateCamera() 函数，该函数内部实现了 Zhang 标定法：
    1. 对每个视角，通过 2D-3D 对应关系计算单应性矩阵 H
    2. 从多个 H 中提取内参矩阵 K 的初始估计
    3. 估计每个视角的外参 [R|t]
    4. 使用 Levenberg-Marquardt 非线性优化，最小化重投影误差
    5. 同时优化内参、畸变系数和所有外参

    参数:
        object_points: 3D 世界坐标点列表
        image_points:  2D 图像坐标点列表

    返回:
        ret:         总体重投影误差（RMS，单位：像素）
        mtx:         标定得到的内参矩阵 (3x3)
        dist:        标定得到的畸变系数
        rvecs_cal:   标定得到的每个视角的旋转向量
        tvecs_cal:   标定得到的每个视角的平移向量
    """
    print("\n" + "=" * 60)
    print("[步骤5] 执行 Zhang 标定法")
    print("=" * 60)

    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

    # =============================================
    # 核心：调用 cv2.calibrateCamera()
    # =============================================
    # 参数说明：
    #   objectPoints: 多个视角的 3D 点（列表中每个元素是一个视角的点）
    #   imagePoints:  多个视角的 2D 点（与 objectPoints 一一对应）
    #   imageSize:    图像尺寸 (width, height)
    #   cameraMatrix: 可选的初始内参（None 表示从头估计）
    #   distCoeffs:   可选的初始畸变系数（None 表示从头估计）
    #
    # 返回值：
    #   ret:    总 RMS 重投影误差
    #   mtx:    优化后的内参矩阵
    #   dist:   优化后的畸变系数
    #   rvecs:  每个视角的旋转向量
    #   tvecs:  每个视角的平移向量
    ret, mtx, dist, rvecs_cal, tvecs_cal = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,    # cameraMatrix 初始值（None=自动估计）
        None     # distCoeffs 初始值（None=自动估计）
    )

    print(f"  标定完成！RMS 重投影误差: {ret:.4f} 像素")
    return ret, mtx, dist, rvecs_cal, tvecs_cal


def compare_results(mtx, dist):
    """
    第六步：对比标定结果与真值

    将标定得到的内参矩阵和畸变系数与预设的真实值进行对比，
    计算各参数的绝对误差和相对误差。

    参数:
        mtx:  标定得到的内参矩阵
        dist: 标定得到的畸变系数
    """
    print("\n" + "=" * 60)
    print("[步骤6] 标定结果 vs 真实值对比")
    print("=" * 60)

    # 内参矩阵对比
    print("\n--- 内参矩阵 K ---")
    param_names = ['fx', 'fy', 'cx', 'cy']
    true_vals = [TRUE_CAMERA_MATRIX[0, 0], TRUE_CAMERA_MATRIX[1, 1],
                 TRUE_CAMERA_MATRIX[0, 2], TRUE_CAMERA_MATRIX[1, 2]]
    cal_vals = [mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]]

    print(f"{'参数':<6} {'真实值':>12} {'标定值':>12} {'绝对误差':>12} {'相对误差':>12}")
    print("-" * 56)
    for name, true_v, cal_v in zip(param_names, true_vals, cal_vals):
        abs_err = abs(cal_v - true_v)
        rel_err = abs_err / abs(true_v) * 100 if true_v != 0 else float('inf')
        print(f"{name:<6} {true_v:>12.4f} {cal_v:>12.4f} {abs_err:>12.4f} {rel_err:>11.4f}%")

    # 畸变系数对比
    print("\n--- 畸变系数 ---")
    dist_names = ['k1', 'k2', 'p1', 'p2', 'k3']
    dist_flat = dist.flatten()

    print(f"{'参数':<6} {'真实值':>12} {'标定值':>12} {'绝对误差':>12}")
    print("-" * 44)
    for name, true_v, cal_v in zip(dist_names, TRUE_DIST_COEFFS, dist_flat[:5]):
        abs_err = abs(cal_v - true_v)
        print(f"{name:<6} {true_v:>12.6f} {cal_v:>12.6f} {abs_err:>12.6f}")


def compute_reprojection_errors(object_points, image_points, rvecs, tvecs, mtx, dist):
    """
    第七步：计算每个视角的重投影误差

    重投影误差 = 标定后用估计的参数重新投影 3D 点到 2D，
    与原始检测到的 2D 点之间的欧氏距离。

    这是评估标定质量最重要的指标：
    - < 0.5 像素：优秀
    - 0.5~1.0 像素：良好
    - > 1.0 像素：可能需要重新标定

    参数:
        object_points: 3D 世界坐标点列表
        image_points:  检测到的 2D 图像坐标点列表
        rvecs, tvecs:  标定得到的外参
        mtx:           标定得到的内参矩阵
        dist:          标定得到的畸变系数

    返回:
        per_view_errors: 每个视角的平均重投影误差列表
    """
    print("\n" + "=" * 60)
    print("[步骤7] 计算重投影误差")
    print("=" * 60)

    per_view_errors = []
    total_error = 0
    total_points = 0

    for i in range(len(object_points)):
        # 使用标定得到的参数重新投影 3D 点
        img_pts_reproj, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], mtx, dist
        )

        # 计算重投影点与原始检测点之间的欧氏距离
        error = cv2.norm(
            image_points[i],          # 原始检测到的 2D 点
            img_pts_reproj,           # 重新投影的 2D 点
            cv2.NORM_L2              # 使用 L2 范数（欧氏距离）
        ) / len(img_pts_reproj)      # 除以点数得到平均误差

        per_view_errors.append(error)
        total_error += error
        total_points += 1

    mean_error = total_error / total_points
    print(f"  平均重投影误差: {mean_error:.4f} 像素")
    print(f"  最小误差: {min(per_view_errors):.4f} 像素 (视角 {np.argmin(per_view_errors)})")
    print(f"  最大误差: {max(per_view_errors):.4f} 像素 (视角 {np.argmax(per_view_errors)})")

    return per_view_errors


def demonstrate_undistortion(mtx, dist):
    """
    第八步：演示去畸变效果

    生成一个带棋盘格图案的合成图像，分别展示：
    1. 原始图像（带畸变）
    2. 使用 cv2.undistort() 去畸变后的图像

    cv2.undistort() 的工作原理：
    - 对输出图像的每个像素，反向计算其在畸变图像中的对应位置
    - 使用插值方法获取像素值
    - 这等效于求解畸变模型的逆变换

    参数:
        mtx:  标定得到的内参矩阵
        dist: 标定得到的畸变系数
    """
    print("\n" + "=" * 60)
    print("[步骤8] 演示去畸变效果")
    print("=" * 60)

    # 创建一个合成的网格图像（模拟直线场景）
    img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 255

    # 绘制水平和垂直网格线
    for y in range(0, IMAGE_HEIGHT, 30):
        cv2.line(img, (0, y), (IMAGE_WIDTH, y), (200, 200, 200), 1)
    for x in range(0, IMAGE_WIDTH, 30):
        cv2.line(img, (x, 0), (x, IMAGE_HEIGHT), (200, 200, 200), 1)

    # 在网格上绘制一些圆点作为标记
    for y in range(30, IMAGE_HEIGHT - 30, 60):
        for x in range(30, IMAGE_WIDTH - 30, 60):
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    # =============================================
    # 模拟畸变效果
    # =============================================
    # 生成像素坐标网格
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    fx, fy = TRUE_CAMERA_MATRIX[0, 0], TRUE_CAMERA_MATRIX[1, 1]
    cx, cy = TRUE_CAMERA_MATRIX[0, 2], TRUE_CAMERA_MATRIX[1, 2]
    k1, k2, p1, p2, k3 = TRUE_DIST_COEFFS

    for v in range(h):
        for u in range(w):
            # 像素坐标 → 归一化坐标
            x = (u - cx) / fx
            y = (v - cy) / fy

            # 计算径向距离的平方
            r2 = x * x + y * y

            # 施加畸变模型
            # 径向畸变: (1 + k1*r^2 + k2*r^4 + k3*r^6)
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            # 切向畸变
            x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            y_distorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

            # 归一化坐标 → 像素坐标
            map_x[v, u] = x_distorted * fx + cx
            map_y[v, u] = y_distorted * fy + cy

    # 使用 remap 生成畸变图像
    distorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    # =============================================
    # 使用 cv2.undistort() 去畸变
    # =============================================
    # cv2.getOptimalNewCameraMatrix() 计算去畸变后的最优内参
    # alpha=1: 保留所有像素（会有黑色边框）
    # alpha=0: 裁剪掉黑色边框
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), alpha=1, newImgSize=(w, h)
    )

    # 执行去畸变
    undistorted_img = cv2.undistort(distorted_img, mtx, dist, None, new_camera_mtx)

    # 保存对比图
    output_dir = os.path.dirname(os.path.abspath(__file__))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (No Distortion)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Distorted', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Undistorted (cv2.undistort)', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'undistortion_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  去畸变对比图已保存: {save_path}")


def plot_reprojection_errors(per_view_errors):
    """
    第九步：绘制重投影误差分布图

    参数:
        per_view_errors: 每个视角的平均重投影误差列表
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 柱状图：每个视角的重投影误差
    views = range(len(per_view_errors))
    ax1.bar(views, per_view_errors, color='steelblue', alpha=0.8)
    ax1.axhline(y=np.mean(per_view_errors), color='red', linestyle='--',
                label=f'Mean = {np.mean(per_view_errors):.4f} px')
    ax1.set_xlabel('View Index')
    ax1.set_ylabel('Reprojection Error (pixels)')
    ax1.set_title('Per-View Reprojection Error')
    ax1.legend()

    # 直方图：误差分布
    ax2.hist(per_view_errors, bins=10, color='steelblue', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Reprojection Error (pixels)')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'reprojection_errors.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  重投影误差图已保存: {save_path}")


def main():
    """
    主函数：按顺序执行相机标定的完整流程
    """
    print("=" * 60)
    print("       相机标定演示 —— Zhang 标定法（合成数据）")
    print("=" * 60)
    print(f"\n[步骤1] 定义真实相机参数:")
    print(f"  图像尺寸: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"  焦距: fx={TRUE_CAMERA_MATRIX[0,0]}, fy={TRUE_CAMERA_MATRIX[1,1]}")
    print(f"  主点: cx={TRUE_CAMERA_MATRIX[0,2]}, cy={TRUE_CAMERA_MATRIX[1,2]}")
    print(f"  畸变: k1={TRUE_DIST_COEFFS[0]}, k2={TRUE_DIST_COEFFS[1]}, "
          f"p1={TRUE_DIST_COEFFS[2]}, p2={TRUE_DIST_COEFFS[3]}, k3={TRUE_DIST_COEFFS[4]}")

    # 生成合成数据
    obj_pts, img_pts, true_rvecs, true_tvecs = generate_synthetic_data()

    if len(obj_pts) < 3:
        print("错误：有效视角数量不足（至少需要 3 个），请调整参数")
        return

    # 执行标定
    ret, mtx, dist, rvecs_cal, tvecs_cal = run_calibration(obj_pts, img_pts)

    # 对比结果
    compare_results(mtx, dist)

    # 计算重投影误差
    per_view_errors = compute_reprojection_errors(
        obj_pts, img_pts, rvecs_cal, tvecs_cal, mtx, dist
    )

    # 绘制重投影误差图
    plot_reprojection_errors(per_view_errors)

    # 演示去畸变
    demonstrate_undistortion(mtx, dist)

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)
    print(f"  使用 {len(obj_pts)} 个视角的合成数据完成了相机标定")
    print(f"  RMS 重投影误差: {ret:.4f} 像素")
    print(f"  标定精度评估:")
    fx_err = abs(mtx[0, 0] - TRUE_CAMERA_MATRIX[0, 0]) / TRUE_CAMERA_MATRIX[0, 0] * 100
    fy_err = abs(mtx[1, 1] - TRUE_CAMERA_MATRIX[1, 1]) / TRUE_CAMERA_MATRIX[1, 1] * 100
    print(f"    焦距误差: fx={fx_err:.4f}%, fy={fy_err:.4f}%")
    print(f"  结论: {'标定成功！' if ret < 1.0 else '重投影误差偏大，建议增加视角数或降低噪声'}")


if __name__ == '__main__':
    main()
