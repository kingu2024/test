"""
手眼标定完整案例 —— AX=XB 问题（合成数据）

本脚本演示机器人手眼标定（Hand-Eye Calibration），即求解
安装在机器人末端执行器（gripper）上的相机与末端之间的相对位姿关系。

手眼标定问题分为两种配置：
1. Eye-in-Hand（眼在手上）：相机固定在机器人末端
   - 已知: T_gripper2base（机器人正运动学），T_target2cam（标定板检测）
   - 求解: X = T_cam2gripper（相机到末端的变换）
   - 方程: A_i * X = X * B_i
     其中 A_i = inv(T_gripper2base_i) * T_gripper2base_j
          B_i = T_target2cam_i * inv(T_target2cam_j)

2. Eye-to-Hand（眼到手）：相机固定在外部（如工作台上）
   - 本脚本以 Eye-in-Hand 配置为例

数据来源:
    本脚本使用程序自动生成的合成数据，无需外部数据集。
    如需真实数据参考:
    - easy_handeye (ROS 包): https://github.com/IFL-CAMP/easy_handeye
      提供了完整的手眼标定 ROS 框架，支持多种机器人和相机
    - UR5 机器人 + RealSense 相机标定数据集:
      https://github.com/IFL-CAMP/easy_handeye/wiki/Example-datasets

运行方式:
    cd calibration_algorithms
    python hand_eye_calibration/hand_eye_calibration.py

作者: Claude (AI Assistant)
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os


# ============================================================================
# 第一步：定义真实的手眼变换参数（Ground Truth）
# ============================================================================

# 手眼变换矩阵 X = T_cam2gripper（相机坐标系到末端坐标系的变换）
# 模拟实际安装情况：相机安装在末端法兰上，有一定的偏移和旋转
#
# 假设：
# - 相机相对于末端绕 Z 轴旋转了 10°（安装时的角度偏差）
# - 相机相对于末端绕 X 轴倾斜了 5°
# - 平移：相机在末端前方 0.05m、上方 0.03m、右侧 0.02m

# 构建真实的手眼变换矩阵
angle_z = 10.0 * np.pi / 180.0  # 绕 Z 轴 10°
angle_x = 5.0 * np.pi / 180.0   # 绕 X 轴 5°

Rz = np.array([
    [np.cos(angle_z), -np.sin(angle_z), 0],
    [np.sin(angle_z),  np.cos(angle_z), 0],
    [0,                0,               1]
], dtype=np.float64)

Rx = np.array([
    [1, 0,                0               ],
    [0, np.cos(angle_x), -np.sin(angle_x)],
    [0, np.sin(angle_x),  np.cos(angle_x)]
], dtype=np.float64)

TRUE_R_CAM2GRIPPER = Rz @ Rx
TRUE_T_CAM2GRIPPER = np.array([[0.05], [0.02], [0.03]], dtype=np.float64)

# 完整的 4x4 手眼变换矩阵
TRUE_X = np.eye(4, dtype=np.float64)
TRUE_X[:3, :3] = TRUE_R_CAM2GRIPPER
TRUE_X[:3, 3:4] = TRUE_T_CAM2GRIPPER

# 标定板在世界坐标系（基座坐标系）中的位姿
# 假设标定板放在机器人前方约 0.5m 处
TRUE_T_TARGET2BASE = np.eye(4, dtype=np.float64)
TRUE_T_TARGET2BASE[:3, :3] = Rotation.from_euler('xyz', [5, -10, 3], degrees=True).as_matrix()
TRUE_T_TARGET2BASE[:3, 3] = [0.5, 0.0, 0.2]

# 合成数据参数
NUM_POSES = 25              # 机器人位姿数量（建议 >= 15，越多越好）
ROTATION_NOISE_DEG = 0.3    # 旋转噪声（度）—— 模拟标定板检测误差
TRANSLATION_NOISE_M = 0.001 # 平移噪声（米）—— 模拟标定板检测误差


def make_homogeneous(R, t):
    """
    辅助函数：将旋转矩阵和平移向量组合为 4x4 齐次变换矩阵

    T = [R  t]
        [0  1]

    参数:
        R: (3, 3) 旋转矩阵
        t: (3,) 或 (3, 1) 平移向量

    返回:
        T: (4, 4) 齐次变换矩阵
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def generate_robot_poses():
    """
    第二步：生成合成的机器人末端位姿数据

    模拟机器人在标定板前做不同姿态运动的场景。
    关键要求：
    - 每个位姿的旋转角度变化应 >= 30°（OpenCV 建议）
    - 位姿应覆盖尽可能多的方向（半球形分布最佳）
    - 避免所有位姿的旋转轴平行（会导致标定退化）

    返回:
        T_gripper2base_list: 机器人末端在基座坐标系下的位姿列表 (4x4)
    """
    print("\n" + "=" * 60)
    print("[步骤2] 生成合成机器人位姿")
    print("=" * 60)

    rng = np.random.RandomState(42)
    T_gripper2base_list = []

    for i in range(NUM_POSES):
        # --------------------------------------------------
        # 生成多样化的旋转
        # --------------------------------------------------
        # 使用欧拉角生成旋转，确保各轴都有足够的变化
        # 旋转范围：每个轴 ±60°（远超 30° 的最低要求）
        roll = rng.uniform(-60, 60)    # 绕 X 轴
        pitch = rng.uniform(-60, 60)   # 绕 Y 轴
        yaw = rng.uniform(-60, 60)     # 绕 Z 轴

        R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

        # --------------------------------------------------
        # 生成平移
        # --------------------------------------------------
        # 模拟末端在工作空间内的位置变化
        tx = rng.uniform(0.3, 0.7)   # 前方 0.3~0.7m
        ty = rng.uniform(-0.2, 0.2)  # 左右 ±0.2m
        tz = rng.uniform(0.1, 0.5)   # 高度 0.1~0.5m

        t = np.array([tx, ty, tz], dtype=np.float64)

        T_gripper2base = make_homogeneous(R, t)
        T_gripper2base_list.append(T_gripper2base)

    print(f"  生成了 {NUM_POSES} 个机器人位姿")
    print(f"  位置范围: X=[0.3, 0.7]m, Y=[-0.2, 0.2]m, Z=[0.1, 0.5]m")
    print(f"  旋转范围: 各轴 ±60°")

    return T_gripper2base_list


def compute_target2cam_poses(T_gripper2base_list):
    """
    第三步：计算标定板在相机坐标系下的位姿

    根据运动链关系:
        T_target2cam = inv(T_cam2gripper) * inv(T_gripper2base) * T_target2base
                     = inv(X) * inv(T_gripper2base) * T_target2base

    其中:
    - T_cam2gripper = X (我们要标定的手眼变换)
    - T_gripper2base: 机器人正运动学给出
    - T_target2base:  标定板在基座坐标系的位姿（固定不动）

    添加噪声模拟实际标定板检测的不确定性。

    参数:
        T_gripper2base_list: 机器人末端位姿列表

    返回:
        T_target2cam_list: 标定板在相机坐标系下的位姿列表 (4x4)
    """
    print("\n" + "=" * 60)
    print("[步骤3] 计算标定板在相机坐标系下的位姿")
    print("=" * 60)

    rng = np.random.RandomState(123)
    T_target2cam_list = []

    # X 的逆矩阵
    X_inv = np.linalg.inv(TRUE_X)

    for T_g2b in T_gripper2base_list:
        # 运动链计算
        # T_target2cam = inv(X) * inv(T_gripper2base) * T_target2base
        T_g2b_inv = np.linalg.inv(T_g2b)
        T_target2cam = X_inv @ T_g2b_inv @ TRUE_T_TARGET2BASE

        # --------------------------------------------------
        # 添加噪声（模拟标定板角点检测 + 位姿估计的误差）
        # --------------------------------------------------
        # 旋转噪声：小角度扰动
        noise_angles = rng.normal(0, ROTATION_NOISE_DEG, 3)
        R_noise = Rotation.from_euler('xyz', noise_angles, degrees=True).as_matrix()
        T_target2cam[:3, :3] = R_noise @ T_target2cam[:3, :3]

        # 平移噪声
        t_noise = rng.normal(0, TRANSLATION_NOISE_M, 3)
        T_target2cam[:3, 3] += t_noise

        T_target2cam_list.append(T_target2cam)

    print(f"  计算了 {len(T_target2cam_list)} 个标定板位姿")
    print(f"  添加噪声: 旋转 σ={ROTATION_NOISE_DEG}°, 平移 σ={TRANSLATION_NOISE_M*1000:.1f}mm")

    return T_target2cam_list


def prepare_calibration_input(T_gripper2base_list, T_target2cam_list):
    """
    第四步：准备 cv2.calibrateHandEye() 的输入数据

    OpenCV 的 calibrateHandEye() 要求输入为旋转矩阵和平移向量的列表，
    而不是 4x4 齐次矩阵。需要将齐次矩阵分解为 R 和 t。

    参数:
        T_gripper2base_list: (N,) 列表，每个元素是 4x4 矩阵
        T_target2cam_list:   (N,) 列表，每个元素是 4x4 矩阵

    返回:
        R_gripper2base: (N,) 列表，每个元素是 3x3 旋转矩阵
        t_gripper2base: (N,) 列表，每个元素是 3x1 平移向量
        R_target2cam:   (N,) 列表，每个元素是 3x3 旋转矩阵
        t_target2cam:   (N,) 列表，每个元素是 3x1 平移向量
    """
    print("\n" + "=" * 60)
    print("[步骤4] 准备 calibrateHandEye 输入数据")
    print("=" * 60)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for T_g2b, T_t2c in zip(T_gripper2base_list, T_target2cam_list):
        # 从 4x4 齐次矩阵中提取 R 和 t
        R_gripper2base.append(T_g2b[:3, :3])
        t_gripper2base.append(T_g2b[:3, 3:4])  # 必须是 (3,1) 形状
        R_target2cam.append(T_t2c[:3, :3])
        t_target2cam.append(T_t2c[:3, 3:4])

    print(f"  准备了 {len(R_gripper2base)} 组输入数据")
    print(f"  R_gripper2base[0] shape: {R_gripper2base[0].shape}")
    print(f"  t_gripper2base[0] shape: {t_gripper2base[0].shape}")

    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def run_hand_eye_calibration(R_g2b, t_g2b, R_t2c, t_t2c):
    """
    第五步：使用 OpenCV 的五种方法进行手眼标定

    cv2.calibrateHandEye() 实现了求解 AX=XB 问题的五种经典算法：

    1. TSAI (1989):
       - 分两步：先解旋转（利用旋转向量的性质），再解平移
       - 优点：简单高效
       - 缺点：旋转误差会累积到平移

    2. PARK (1994):
       - 基于李群/李代数，在 SO(3) 上直接优化
       - 优点：理论上更准确
       - 通常被认为是最可靠的方法之一

    3. HORAUD (1995):
       - 使用四元数表示旋转
       - 将问题转化为特征值问题

    4. ANDREFF (2001):
       - 使用克罗内克积（Kronecker product）将问题线性化
       - 同时求解旋转和平移

    5. DANIILIDIS (1999):
       - 使用对偶四元数（dual quaternion）
       - 同时求解旋转和平移，避免误差累积

    参数:
        R_g2b, t_g2b: 机器人末端位姿的旋转和平移列表
        R_t2c, t_t2c: 标定板在相机坐标系下的位姿的旋转和平移列表

    返回:
        results: 字典，键为方法名，值为 (R_cam2gripper, t_cam2gripper)
    """
    print("\n" + "=" * 60)
    print("[步骤5] 使用五种方法进行手眼标定")
    print("=" * 60)

    # 定义五种标定方法
    methods = {
        'TSAI':      cv2.CALIB_HAND_EYE_TSAI,
        'PARK':      cv2.CALIB_HAND_EYE_PARK,
        'HORAUD':    cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF':   cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}

    for name, method in methods.items():
        # =============================================
        # 核心：调用 cv2.calibrateHandEye()
        # =============================================
        # 参数:
        #   R_gripper2base: 每个位姿的末端到基座旋转矩阵列表
        #   t_gripper2base: 每个位姿的末端到基座平移向量列表
        #   R_target2cam:   每个位姿的标定板到相机旋转矩阵列表
        #   t_target2cam:   每个位姿的标定板到相机平移向量列表
        #   method:         求解方法
        #
        # 返回:
        #   R_cam2gripper: 相机到末端的旋转矩阵 (3x3)
        #   t_cam2gripper: 相机到末端的平移向量 (3x1)
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_g2b, t_g2b,
            R_t2c, t_t2c,
            method=method
        )

        results[name] = (R_cam2gripper, t_cam2gripper)
        print(f"  {name:12s} 方法完成")

    return results


def evaluate_results(results):
    """
    第六步：评估各方法的标定精度

    评估指标：
    1. 旋转误差：R_err = R_true * R_est^T 的旋转角度
    2. 平移误差：t_true 与 t_est 之间的欧氏距离
    3. AX=XB 残差：验证标定结果是否满足运动链约束

    参数:
        results: 字典，键为方法名，值为 (R, t)

    返回:
        eval_data: 字典，包含各方法的误差数据
    """
    print("\n" + "=" * 60)
    print("[步骤6] 评估标定精度")
    print("=" * 60)

    eval_data = {}

    print(f"\n{'方法':<14} {'旋转误差(°)':>12} {'平移误差(mm)':>14} {'旋转矩阵行列式':>16}")
    print("-" * 60)

    for name, (R_est, t_est) in results.items():
        # --------------------------------------------------
        # 旋转误差
        # --------------------------------------------------
        R_err = TRUE_R_CAM2GRIPPER @ R_est.T
        trace_val = np.trace(R_err)
        cos_angle = np.clip((trace_val - 1) / 2, -1.0, 1.0)
        angle_error_deg = np.arccos(cos_angle) * 180.0 / np.pi

        # --------------------------------------------------
        # 平移误差
        # --------------------------------------------------
        t_error_m = np.linalg.norm(TRUE_T_CAM2GRIPPER - t_est)
        t_error_mm = t_error_m * 1000  # 转换为毫米

        # --------------------------------------------------
        # 旋转矩阵有效性检查
        # --------------------------------------------------
        # 有效的旋转矩阵应满足: det(R) = 1, R^T * R = I
        det_R = np.linalg.det(R_est)

        eval_data[name] = {
            'angle_error': angle_error_deg,
            't_error_mm': t_error_mm,
            'det_R': det_R,
            'R': R_est,
            't': t_est
        }

        print(f"{name:<14} {angle_error_deg:>12.4f} {t_error_mm:>14.4f} {det_R:>16.6f}")

    # 打印最佳方法
    best_method = min(eval_data, key=lambda x: eval_data[x]['angle_error'])
    print(f"\n  最佳方法（旋转误差最小）: {best_method}")
    print(f"    旋转误差: {eval_data[best_method]['angle_error']:.4f}°")
    print(f"    平移误差: {eval_data[best_method]['t_error_mm']:.4f} mm")

    # 打印真值与最佳结果的详细对比
    R_best = eval_data[best_method]['R']
    t_best = eval_data[best_method]['t']

    print(f"\n  真实手眼变换 X_true (T_cam2gripper):")
    print(f"    旋转矩阵 R:")
    for row in TRUE_R_CAM2GRIPPER:
        print(f"      [{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f}]")
    print(f"    平移向量 t: [{TRUE_T_CAM2GRIPPER[0,0]:.6f}, "
          f"{TRUE_T_CAM2GRIPPER[1,0]:.6f}, {TRUE_T_CAM2GRIPPER[2,0]:.6f}] m")

    print(f"\n  最佳标定结果 ({best_method}):")
    print(f"    旋转矩阵 R:")
    for row in R_best:
        print(f"      [{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f}]")
    print(f"    平移向量 t: [{t_best[0,0]:.6f}, {t_best[1,0]:.6f}, {t_best[2,0]:.6f}] m")

    return eval_data


def verify_ax_xb(T_gripper2base_list, T_target2cam_list, R_est, t_est, method_name):
    """
    第七步：验证 AX=XB 等式残差

    对于 Eye-in-Hand 配置，运动链约束为：
        T_gripper2base_i * X = X * T_target2cam_i * T_target2base^(-1) ... (不完全正确)

    更准确地说，对于任意两个位姿 i 和 j：
        A_ij * X = X * B_ij
    其中：
        A_ij = inv(T_gripper2base_j) * T_gripper2base_i
        B_ij = T_target2cam_j * inv(T_target2cam_i)

    残差 = ||A * X - X * B||_F  (Frobenius 范数)

    参数:
        T_gripper2base_list: 末端位姿列表
        T_target2cam_list:   标定板位姿列表
        R_est, t_est:        标定结果
        method_name:         方法名称

    返回:
        mean_residual: 平均残差
    """
    X_est = make_homogeneous(R_est, t_est)
    residuals = []

    n = len(T_gripper2base_list)
    for i in range(n):
        for j in range(i + 1, n):
            # 计算 A_ij 和 B_ij
            A = np.linalg.inv(T_gripper2base_list[j]) @ T_gripper2base_list[i]
            B = T_target2cam_list[j] @ np.linalg.inv(T_target2cam_list[i])

            # 计算残差: ||A*X - X*B||_F
            AX = A @ X_est
            XB = X_est @ B
            residual = np.linalg.norm(AX - XB, 'fro')
            residuals.append(residual)

    mean_residual = np.mean(residuals)
    return mean_residual


def visualize_evaluation(eval_data, T_gripper2base_list, T_target2cam_list):
    """
    第八步：可视化评估结果

    生成图表：
    1. 各方法的旋转误差和平移误差柱状图
    2. AX=XB 残差对比

    参数:
        eval_data: 各方法的评估数据
        T_gripper2base_list: 末端位姿列表
        T_target2cam_list:   标定板位姿列表
    """
    print("\n" + "=" * 60)
    print("[步骤8] 可视化评估结果")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    methods = list(eval_data.keys())
    angle_errors = [eval_data[m]['angle_error'] for m in methods]
    t_errors = [eval_data[m]['t_error_mm'] for m in methods]

    # 计算 AX=XB 残差
    ax_xb_residuals = []
    for m in methods:
        R_est = eval_data[m]['R']
        t_est = eval_data[m]['t']
        residual = verify_ax_xb(
            T_gripper2base_list, T_target2cam_list,
            R_est, t_est, m
        )
        ax_xb_residuals.append(residual)

    # --------------------------------------------------
    # 图1：误差对比柱状图
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']

    # 旋转误差
    bars1 = axes[0].bar(methods, angle_errors, color=colors, alpha=0.8)
    axes[0].set_ylabel('Rotation Error (degrees)')
    axes[0].set_title('Rotation Error by Method')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, angle_errors):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 平移误差
    bars2 = axes[1].bar(methods, t_errors, color=colors, alpha=0.8)
    axes[1].set_ylabel('Translation Error (mm)')
    axes[1].set_title('Translation Error by Method')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, t_errors):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # AX=XB 残差
    bars3 = axes[2].bar(methods, ax_xb_residuals, color=colors, alpha=0.8)
    axes[2].set_ylabel('Mean AX=XB Residual')
    axes[2].set_title('AX=XB Constraint Residual')
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, ax_xb_residuals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'hand_eye_calibration_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  误差对比图已保存: {save_path}")

    # --------------------------------------------------
    # 图2：不同位姿数量对精度的影响
    # --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 测试不同位姿数量
    pose_counts = [5, 8, 10, 15, 20, 25]
    pose_counts = [p for p in pose_counts if p <= NUM_POSES]

    # 使用 PARK 方法测试
    method_test = cv2.CALIB_HAND_EYE_PARK
    angle_errs_by_count = []
    t_errs_by_count = []

    # 准备完整数据
    R_g2b_all = [T[:3, :3] for T in T_gripper2base_list]
    t_g2b_all = [T[:3, 3:4] for T in T_gripper2base_list]
    R_t2c_all = [T[:3, :3] for T in T_target2cam_list]
    t_t2c_all = [T[:3, 3:4] for T in T_target2cam_list]

    for count in pose_counts:
        R_result, t_result = cv2.calibrateHandEye(
            R_g2b_all[:count], t_g2b_all[:count],
            R_t2c_all[:count], t_t2c_all[:count],
            method=method_test
        )

        # 旋转误差
        R_err = TRUE_R_CAM2GRIPPER @ R_result.T
        cos_a = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        angle_errs_by_count.append(np.arccos(cos_a) * 180.0 / np.pi)

        # 平移误差
        t_errs_by_count.append(np.linalg.norm(TRUE_T_CAM2GRIPPER - t_result) * 1000)

    ax1.plot(pose_counts, angle_errs_by_count, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Poses')
    ax1.set_ylabel('Rotation Error (degrees)')
    ax1.set_title('PARK Method: Rotation Error vs Number of Poses')
    ax1.grid(True, alpha=0.3)

    ax2.plot(pose_counts, t_errs_by_count, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Poses')
    ax2.set_ylabel('Translation Error (mm)')
    ax2.set_title('PARK Method: Translation Error vs Number of Poses')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'hand_eye_pose_count_effect.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  位姿数量影响图已保存: {save_path}")


def main():
    """
    主函数：按顺序执行手眼标定的完整流程
    """
    print("=" * 60)
    print("       手眼标定演示 —— AX=XB（合成数据）")
    print("=" * 60)

    print(f"\n[步骤1] 定义真实手眼变换参数:")
    print(f"  配置: Eye-in-Hand（相机固定在机器人末端）")
    print(f"  位姿数量: {NUM_POSES}")
    print(f"  真实手眼变换矩阵 X = T_cam2gripper:")
    for i in range(4):
        row = TRUE_X[i]
        print(f"    [{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f} {row[3]:>10.6f}]")

    # 生成机器人位姿
    T_gripper2base_list = generate_robot_poses()

    # 计算标定板在相机坐标系下的位姿
    T_target2cam_list = compute_target2cam_poses(T_gripper2base_list)

    # 准备输入数据
    R_g2b, t_g2b, R_t2c, t_t2c = prepare_calibration_input(
        T_gripper2base_list, T_target2cam_list
    )

    # 使用五种方法标定
    results = run_hand_eye_calibration(R_g2b, t_g2b, R_t2c, t_t2c)

    # 评估结果
    eval_data = evaluate_results(results)

    # 验证 AX=XB 残差
    print("\n" + "=" * 60)
    print("[步骤7] 验证 AX=XB 等式残差")
    print("=" * 60)

    for name in results:
        R_est, t_est = results[name]
        residual = verify_ax_xb(
            T_gripper2base_list, T_target2cam_list,
            R_est, t_est, name
        )
        print(f"  {name:<14} 平均 AX=XB 残差: {residual:.6f}")

    # 可视化
    visualize_evaluation(eval_data, T_gripper2base_list, T_target2cam_list)

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)
    best = min(eval_data, key=lambda x: eval_data[x]['angle_error'])
    print(f"  使用 {NUM_POSES} 个位姿完成了手眼标定")
    print(f"  最佳方法: {best}")
    print(f"    旋转误差: {eval_data[best]['angle_error']:.4f}°")
    print(f"    平移误差: {eval_data[best]['t_error_mm']:.4f} mm")
    print(f"\n  实际应用建议:")
    print(f"    1. 采集 >= 15 个位姿，旋转角度变化 >= 30°")
    print(f"    2. 位姿应覆盖半球形分布，避免退化配置")
    print(f"    3. 优先使用 PARK 方法（通常最稳定）")
    print(f"    4. 对比多种方法结果，取一致性最好的")
    print(f"    5. 检查 AX=XB 残差来验证标定质量")


if __name__ == '__main__':
    main()
