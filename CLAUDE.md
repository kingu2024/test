# CLAUDE.md — Codebase Guide

## Repository Overview

This is a Python computer vision / imaging algorithms library covering five major subsystems:

| Module | Purpose |
|--------|---------|
| `panorama_stitching/` | Multi-image panorama stitching |
| `video_stabilization/` | Video stabilization via optical flow |
| `hdr_imaging/` | HDR imaging pipeline (multi-exposure + tone mapping) |
| `isp_pipeline/` | ISP pipeline: RAW Bayer → sRGB |
| `distillation/` | Multi-task model knowledge distillation (PyTorch) |
| `calibration_algorithms/` | Camera, LiDAR-camera, and hand-eye calibration |

Demo entry points: `demo_panorama.py`, `demo_stabilization.py`, `demo_hdr.py`.

---

## Directory Structure

```
.
├── panorama_stitching/         # Panorama stitching algorithms
│   ├── feature_extraction.py   # SIFT/ORB/AKAZE feature detect & match
│   ├── homography.py           # RANSAC homography estimation
│   ├── warping.py              # Cylindrical/spherical/planar projection
│   ├── blending.py             # Laplacian pyramid multi-band blending
│   └── stitcher.py             # Main pipeline controller
│
├── video_stabilization/        # Video stabilization
│   ├── optical_flow.py         # LK sparse / Farneback dense optical flow
│   ├── trajectory_smoother.py  # Moving avg / Gaussian / Kalman / L1 smoothing
│   └── stabilizer.py           # Main pipeline controller
│
├── hdr_imaging/                # HDR imaging (6 sub-packages)
│   ├── alignment/              # MTB alignment (Ward 2003) + feature alignment
│   ├── calibration/            # Debevec 1997 + Robertson 2003 CRF recovery
│   ├── merge/                  # Weighted irradiance → HDR radiance map
│   ├── tone_mapping/           # 10 tone mapping operators
│   │   ├── global_operators.py     # Reinhard global, Drago, AdaptiveLog
│   │   ├── local_operators.py      # Reinhard local, Durand, Fattal
│   │   └── perceptual_operators.py # ACES, Filmic, Mantiuk, Histogram
│   ├── exposure_fusion/        # Mertens 2007 exposure fusion
│   ├── single_image/           # CLAHE + multi-scale detail enhancement
│   └── hdr_pipeline.py         # HDRPipeline controller; HDRResult NamedTuple
│
├── isp_pipeline/               # Image Signal Processor
│   ├── modules/                # BLC, BPC, LSC, AWB, Demosaic, CCM, Gamma,
│   │   └── ...                 # NoiseReduction, Sharpening, ToneMapping
│   └── pipeline.py             # ISP orchestrator (5-stage Bayer→sRGB flow)
│
├── distillation/               # Knowledge distillation (PyTorch)
│   ├── backbones/              # ResNet, MobileNet (Registry-based)
│   ├── heads/                  # Seg head, Det head
│   ├── losses/                 # KD loss, feature loss, task loss
│   ├── distillers/             # Feature distiller, multi-task distiller
│   ├── models/multi_task_model.py
│   ├── utils/registry.py       # Registry pattern for component lookup
│   ├── configs/default_config.yaml
│   └── train.py                # Training entry point
│
├── calibration_algorithms/     # Camera / sensor calibration
│   ├── camera_calibration/     # Zhang's method (synthetic + real checkerboard)
│   ├── distortion_correction/  # Standard + fisheye undistortion
│   ├── hand_eye_calibration/   # Robot hand-eye calibration
│   ├── lidar_camera_calibration/
│   └── docs/
│
├── docs/superpowers/           # Implementation plans and design specs
├── demo_hdr.py
├── demo_panorama.py
├── demo_stabilization.py
└── requirements.txt
```

---

## Development Setup

```bash
# Install core dependencies
pip install -r requirements.txt
# opencv-python>=4.8.0, opencv-contrib-python>=4.8.0, numpy>=1.24.0,
# matplotlib>=3.7.0, scipy>=1.10.0

# For distillation module (not in root requirements.txt)
pip install torch torchvision

# calibration_algorithms has its own lighter requirements
pip install -r calibration_algorithms/requirements.txt
```

No test runner or CI configuration is present. Verification is done by running demo scripts with synthetic data.

---

## Running Demos

```bash
# Panorama stitching (auto-generates test images)
python demo_panorama.py --test --visualize

# Video stabilization (auto-generates shaky video)
python demo_stabilization.py --test
python demo_stabilization.py --test --compare          # compare all smoothers
python demo_stabilization.py --input my_video.mp4 --smooth kalman

# HDR imaging (auto-generates synthetic exposures)
python demo_hdr.py --test
python demo_hdr.py --test --compare                    # compare tone mappers

# Camera calibration (uses synthetic checkerboard data)
cd calibration_algorithms
python camera_calibration/camera_calibration.py
```

Output directories (git-ignored): `output_panorama/`, `output_stabilization/`, `output_hdr/`

---

## Key Conventions

### Bilingual Comments
All source files use bilingual (Chinese + English) comments. Match this style when adding code.

### Dual-Implementation Pattern
Every algorithm class in `hdr_imaging/`, `panorama_stitching/`, and `video_stabilization/` exposes two methods:
- `process(...)` — hand-written NumPy implementation
- `process_opencv(...)` — OpenCV reference implementation

Both paths must be kept in sync when modifying algorithms.

### Registry Pattern (distillation)
Components (backbones, heads, losses) are registered via `distillation/utils/registry.py`:
```python
BACKBONES = Registry("backbones")

@BACKBONES.register("resnet18")
class ResNet18(nn.Module): ...

model = BACKBONES.build("resnet18", pretrained=True)
```
New distillation components must be registered and imported in the corresponding `__init__.py` to be discoverable at training time.

### Named Tuple Results
Pipeline outputs use `collections.namedtuple`:
- `HDRResult = namedtuple('HDRResult', ['ldr_result', 'hdr_radiance_map', 'response_curve'])`

Follow this pattern for new pipelines.

### Logging
All modules use `logging.getLogger(__name__)`. Do not use `print()` for runtime messages.

### ISP Pipeline Stage Order
The 5-stage ISP ordering must be preserved:
1. Bayer preprocessing: BLC → BPC → LSC → AWB
2. Demosaicing (Bayer → RGB)
3. Linear RGB processing: CCM
4. Tone processing (linear light space): ToneMapping → Gamma
5. Spatial enhancement (gamma-encoded space): NoiseReduction → Sharpening

### Distillation Configuration
Training is driven by `distillation/configs/default_config.yaml`. Key sections: `teacher`, `student`, `feature_distill`, `logit_distill`, `cross_head`, `task_loss`, `training`, `data`. Add new loss types or heads via the registry; do not hard-code class names.

---

## Git Conventions

Commit messages follow Conventional Commits:
- `feat(module):` new feature
- `docs(module):` documentation
- `fix(module):` bug fix

Branch for AI-driven work: `claude/add-claude-documentation-V7x4S`

---

## Architecture Notes

### panorama_stitching
`PanoramaStitcher` → `FeatureExtractor` → `HomographyEstimator` → `ImageWarper` → `ImageBlender`. Multi-image alignment uses a maximum spanning tree to determine transform ordering and accumulates homographies: `H_total = H₀ · H₁ · … · Hₙ`.

### video_stabilization
`VideoStabilizer` → `OpticalFlow` → integrate motion → `TrajectorySmoother` → compute compensation → `WarpAffine`. Kalman smoother uses RTS backward pass for full-sequence smoothing.

### hdr_imaging
`HDRPipeline` supports two paths:
1. **Multi-exposure HDR**: align → calibrate CRF → merge → tone map → LDR
2. **Exposure fusion** (Mertens): align → fuse → LDR (no CRF needed)
3. **Single-image**: CLAHE + detail enhancement

`TONE_MAPPING_METHODS` dict in `hdr_pipeline.py` maps string keys to operator classes; extend it when adding new tone mappers.

### isp_pipeline
`ISPPipeline` accepts RAW Bayer input (`numpy.ndarray`) and returns sRGB. Each module in `isp_pipeline/modules/` is stateless and operates on `numpy.ndarray`. Use `@dataclass` for module configs.

### distillation
Multi-task teacher→student flow: feature-level distillation (MSE on intermediate feature maps) + logit-level KD (KL divergence with temperature scaling) + supervised task losses. `MultiTaskDistiller` computes a weighted sum of all loss terms defined in the YAML config.

---

## Gitignored Outputs

```
__pycache__/  *.pyc  *.pyo
output_panorama/  output_stabilization/  output_hdr/
*.mp4
```

Do not commit generated outputs or compiled Python files.
