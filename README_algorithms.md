# 图像全景拼接 & 视频防抖算法实现

> 基于2024-2025最新方法，融合经典算法与现代优化技术，附完整中文注释与数学公式推导

---

## 目录结构

```
.
├── panorama_stitching/          # 全景拼接算法包
│   ├── __init__.py
│   ├── feature_extraction.py   # SIFT/ORB/AKAZE特征提取与匹配
│   ├── homography.py           # RANSAC单应性矩阵估计
│   ├── warping.py              # 柱面/球面/平面投影变换
│   ├── blending.py             # 多分辨率拉普拉斯金字塔融合
│   └── stitcher.py             # 全景拼接主控流程
│
├── video_stabilization/         # 视频防抖算法包
│   ├── __init__.py
│   ├── optical_flow.py         # LK/Farneback光流估计
│   ├── trajectory_smoother.py  # 移动平均/高斯/卡尔曼/L1平滑
│   └── stabilizer.py           # 视频防抖主控流程
│
├── demo_panorama.py            # 全景拼接演示脚本
├── demo_stabilization.py       # 视频防抖演示脚本
└── requirements.txt            # 依赖包列表
```

---

## 算法原理概览

### 一、图像全景拼接

#### 完整流程

```
输入图像序列
    ↓
[Step 1] 柱面/球面投影预处理
         减少大视角透视畸变
         数学: (u,v) → (f·arctan(x/z), f·y/√(x²+z²))
    ↓
[Step 2] 特征提取 (SIFT/ORB/AKAZE)
         SIFT: 128维描述子，尺度/旋转不变
         ORB: 256位二进制描述子，快速计算
    ↓
[Step 3] 特征匹配 + Lowe's比值测试
         筛选条件: d₁/d₂ < 0.75
    ↓
[Step 4] RANSAC单应性矩阵估计
         H: 3×3投影变换矩阵（8自由度）
         DLT求解: min ||A·h||, s.t. ||h||=1
    ↓
[Step 5] 画布计算 + 图像变形
         累积变换: H_total = H₀·H₁·...·Hₙ
    ↓
[Step 6] 曝光补偿
         增益: g = median(I_ref) / median(I_cur)
    ↓
[Step 7] 多分辨率拉普拉斯金字塔融合
         消除接缝和鬼影
    ↓
全景输出
```

#### 核心数学公式

**单应性矩阵 (Homography)**
```
[x']     [h₁₁ h₁₂ h₁₃] [x]
[y'] = H·[h₂₁ h₂₂ h₂₃]·[y]    (齐次坐标)
[w']     [h₃₁ h₃₂ h₃₃] [1]

实际坐标: x' = (h₁₁x+h₁₂y+h₁₃)/(h₃₁x+h₃₂y+h₃₃)
```

**柱面投影**
```
θ = arctan(x_n)               (水平角)
h = y_n / √(x_n² + 1)        (归一化高度)
u_cyl = f·θ + cx
v_cyl = f·h + cy
```

**拉普拉斯金字塔融合**
```
构建: Lₖ = Gₖ - EXPAND(Gₖ₊₁)
融合: L̃ₖ = αₖ·L¹ₖ + (1-αₖ)·L²ₖ
重建: Ĝₖ₋₁ = Lₖ₋₁ + EXPAND(Ĝₖ)
```

---

### 二、视频防抖

#### 完整流程

```
输入视频
    ↓
[Step 1] 帧间运动估计
         LK稀疏光流追踪角点
         RANSAC估计仿射变换: M = [a b tx; c d ty]
    ↓
[Step 2] 运动参数提取
         dx = M[0,2], dy = M[1,2]
         dα = arctan2(M[1,0], M[0,0])
         ds = √(M[0,0]² + M[1,0]²)
    ↓
[Step 3] 轨迹积分
         T[k] = Σᵢ₌₀ᵏ Δm[i]   (前缀和)
    ↓
[Step 4] 轨迹平滑
         可选方法:
         - 移动平均: p̂[k] = (1/2R+1)·Σp[j]
         - 卡尔曼滤波 + RTS后向平滑
         - L1优化: min λ||p̂-p||² + ||Dp̂||₁
    ↓
[Step 5] 补偿计算
         C[k] = T̂[k] - T[k]
    ↓
[Step 6] 帧变形补偿
         M_comp = [ds·cos(dα) -ds·sin(dα) dx; ...]
         I_stable = WarpAffine(I, M_comp)
    ↓
[Step 7] 裁剪黑边
稳定视频输出
```

#### 卡尔曼滤波公式

```
状态: x[k] = [位置, 速度]ᵀ

预测:  x̂⁻[k] = F·x̂[k-1]
       P⁻[k]  = F·P[k-1]·Fᵀ + Q

更新:  K[k] = P⁻[k]·Hᵀ·(H·P⁻[k]·Hᵀ + R)⁻¹
       x̂[k] = x̂⁻[k] + K[k]·(z[k] - H·x̂⁻[k])
       P[k] = (I - K[k]·H)·P⁻[k]
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 全景拼接

```python
from panorama_stitching import PanoramaStitcher

stitcher = PanoramaStitcher(
    feature_method='SIFT',      # 特征算法: SIFT/ORB/AKAZE
    projection='cylindrical',   # 投影: planar/cylindrical/spherical
    blend_method='multiband'    # 融合: simple/feather/multiband
)

panorama = stitcher.stitch([img1, img2, img3])
```

### 视频防抖

```python
from video_stabilization import VideoStabilizer

stabilizer = VideoStabilizer(
    flow_method='lk',           # 光流: lk/farneback
    smooth_method='kalman',     # 平滑: kalman/gaussian/moving_avg/l1
    crop_ratio=0.1
)

stable_frames, metrics = stabilizer.stabilize(frames)
```

### 演示脚本

```bash
# 全景拼接演示（自动生成测试图像）
python demo_panorama.py --test --visualize

# 视频防抖演示（自动生成抖动视频）
python demo_stabilization.py --test

# 对比所有防抖方法
python demo_stabilization.py --test --compare

# 使用真实视频
python demo_stabilization.py --input my_video.mp4 --smooth kalman
```

---

## 参数调优指南

### 全景拼接

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `feature_method` | `SIFT` | 精度最高；纹理少时用`AKAZE` |
| `max_features` | 1000-3000 | 更多特征点=更准但更慢 |
| `ratio_threshold` | 0.70-0.80 | 越小匹配越严格（减少误匹配） |
| `ransac_threshold` | 3.0-5.0 | 像素误差阈值 |
| `projection` | `cylindrical` | 水平拼接首选；球形用`spherical` |
| `blend_method` | `multiband` | 质量最高；快速用`feather` |

### 视频防抖

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `smooth_method` | `kalman` | 综合最优；保留有意运动用`l1` |
| `smooth_radius` | 15-30 | 窗口越大平滑越强但越多裁剪 |
| `kalman_process_noise` | 1e-4~1e-2 | 越小路径越平滑 |
| `kalman_obs_noise` | 0.1~10 | 越大越不相信测量 |
| `crop_ratio` | 0.05-0.15 | 根据抖动幅度调整 |

---

## 参考文献

### 全景拼接
- [PAIC-Net] Progressive Alignment & Interwoven Composition Network (2024)
- [CAWarpNet] Cross-Attention Warp-equivariance for Stereo Panorama (2025)
- [LiftProj] Space Lifting and Projection-Based Panorama Stitching (2025)
- Burt & Adelson, "A Multiresolution Spline" (1983) - 拉普拉斯金字塔融合基础
- Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (2004) - SIFT

### 视频防抖
- [SOFT] Self-supervised Sparse Optical Flow Transformer (2024)
- [Gyroflow+] Gyroscope-guided Unsupervised Deep Stabilization (2024)
- Liu et al., "Subspace Video Stabilization" SIGGRAPH (2011) - L1优化
- Rauch-Tung-Striebel Smoother - 双向卡尔曼平滑理论

---

## HDR 高动态范围成像

HDR 成像算法的完整文档请参见 [README_HDR.md](README_HDR.md)。

Sources:
- [Real-Time Industrial Panorama Stitching (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12788332/)
- [PAIC-Net (Springer 2024)](https://link.springer.com/article/10.1007/s40747-024-01702-x)
- [LiftProj (arXiv 2025)](https://arxiv.org/html/2512.24276)
- [Gyroscope-Image Fusion Video Stabilization (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0097849324002899)
- [Video Stabilization Survey (Preprints 2025)](https://www.preprints.org/manuscript/202505.0819)
- [Meta-Learning Video Stabilization (arXiv 2024)](https://arxiv.org/html/2403.03662v1)
