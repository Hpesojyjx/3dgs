# Custom Extensions to 3D Gaussian Splatting

This repository extends the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) with several training improvements drawn from recent research. All extensions are implemented in `train_custom.py` and `scene/gaussian_model.py` and are backward-compatible with the original `train.py`.

## Overview of Changes

| Component | What changed | Source |
|-----------|-------------|--------|
| `train_custom.py` | New training script with multi-resolution curriculum, perceptual loss, budget control | This repo |
| `scene/gaussian_model.py` | PixelGS gradient accumulation, elongation filter, budget/prune utilities | PixelGS, CityGaussianV2 |
| `arguments/__init__.py` | New CLI flags for all extensions | This repo |
| `scene/dataset_readers.py` | `np.byte` → `np.uint8` type fix | Bug fix |

---

## Features

### 1. DashGaussian — Multi-Resolution Curriculum

**Paper**: [DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds](https://arxiv.org/abs/2503.18402) (CVPR 2025)

Trains progressively from low resolution to full resolution. Resolution switching points are determined automatically by **frequency energy** (mean DFT amplitude) of each resolution's training cameras — scales with more high-frequency detail get more training budget.

**Key flags:**
```
--use_dash               Enable multi-resolution curriculum
--dash_r_stages 3        Number of resolution stages (default 3: 4×→2×→1×)
--dash_r_min 4           Coarsest downscale factor
```

**How it works:**
1. At startup, cameras are loaded at `[1×, 2×, 4×]` scales.
2. `χ(F_scale)` = mean FFT energy of sample cameras at that scale.
3. `switching_iter[scale] = total_iters × χ(scale) / χ(full_res)` — lower-energy (blurrier) scales get fewer iters.
4. On each scale switch, `max_radii2D` and gradient accumulators are reset to prevent stale cross-scale statistics.

---

### 2. PixelGS — Pixel-Area Weighted Gradient Accumulation

**Paper**: [PixelGS](https://arxiv.org/abs/2403.12520)

The original 3DGS densification threshold `τk = Σ‖grad‖ / count` is biased toward large Gaussians that cover more pixels. PixelGS replaces per-visit counts with **pixel area** weights:

```
τk′ = Σ(radii² · ‖grad‖) / Σ(radii²)
```

This is always active in `train_custom.py` (via `pixel_count_accum` in `GaussianModel`). It makes densification decisions scale-invariant with respect to projected Gaussian size.

---

### 3. AH-GS — VGG Perceptual Loss with Decay

Adds a VGG-19 perceptual loss term on top of the standard L1 + D-SSIM loss, with a **linear decay schedule** so it contributes strongly early (when geometry is rough) and fades out as training converges.

```
L = L1 + λ_ssim · (1 - SSIM) + ahgs_loss(pred, gt, iter)
```

**With `--use_dash`**: The perceptual loss is suppressed during low-resolution warm-up stages (VGG features on blurry inputs are not meaningful) and only activates once full-resolution training begins, decaying linearly within that phase.

**Memory control**: Images are downsampled to `--vgg_max_size` (default 2048) before entering VGG to avoid OOM on 4K inputs.

**Flag:** `--loss_type ahgs`

---

### 4. Gaussian Budget Control

Caps the maximum number of Gaussians and optionally locks densification once the budget is reached:

```
--max_gaussians 7700000   Hard cap on Gaussian count (0 = unlimited)
--lock_after_budget       Freeze densification/opacity-reset once cap is hit
```

When `--lock_after_budget` is set, the training enters a **pure refinement phase** once either:
- The Gaussian count reaches `--max_gaussians`, or
- The iteration exceeds `--iterations` (entering the `--refine_extra_iters` tail).

During this phase only color/opacity/covariance are optimized; no new Gaussians are added or removed.

---

### 5. Extended Refinement Tail (`--refine_extra_iters`)

Appends extra iterations **after** the main training phase where Gaussians are frozen (no densification, no opacity reset) and only appearance parameters continue to be optimized:

```
--iterations 100000         Main training (DashGaussian / LR schedule reference)
--refine_extra_iters 60000  Extra refinement on top (total = 160k)
```

This separates the geometry-building phase from the appearance-polishing phase.

---

### 6. Alternative Loss Modes (`--loss_type`)

| Value | Description |
|-------|-------------|
| `baseline` | Original 3DGS: L1 + D-SSIM (default) |
| `ahgs` | + VGG perceptual loss with linear decay |
| `fregs` | + FreGS frequency-domain regularization (progressive low→high frequency supervision) |
| `depth_reg` | + Depth map total-variation regularization |
| `normal_reg` | + Normal consistency regularization derived from depth gradients |

---

### 7. CityGaussianV2 — Elongation Filter

**Paper**: [CityGaussianV2](https://arxiv.org/abs/2411.00771)

Prevents needle-like Gaussians from triggering densification, which can cascade into artifact explosions:

```python
elongation = max_scale / min_scale
# Skip densification if elongation > 10
```

Applied in both `densify_and_split` and `densify_and_clone` paths.

---

### 8. Bug Fixes

- **`scene/dataset_readers.py`**: `np.byte` (signed) → `np.uint8` (unsigned) when converting NeRF Synthetic RGBA images. `np.byte` caused pixel value wrapping for values > 127.
- **`scene/gaussian_model.py` PLY saving**: Replaced bulk `list(map(tuple, attributes))` assignment with per-attribute column indexing, fixing silent data corruption when attribute ordering differed from NumPy structured array field order.
- **`big_points_vs` in pruning**: `max_screen_size` is now passed as `None` during Dash training to avoid scale-dependent pixel-radius thresholds pruning valid large geometry when switching resolutions. World-space pruning (`big_points_ws`) is used instead.

---

## Quick Start

### Environment Setup

```bash
bash shell/setup_env.sh          # creates 'gaussian_splatting' conda env and builds CUDA extensions
# or with a custom env name/path:
ENV_PATH=/path/to/env bash shell/setup_env.sh
```

### COLMAP Reconstruction (custom video/images)

```bash
# Put your images in <data_dir>/input/
bash shell/colmap.sh <data_dir>
```

### Training

**Standard training (original behavior):**
```bash
python train.py -s <data_path> -m <output_path>
```

**Training with all extensions (recommended configuration):**
```bash
python train_custom.py \
    -s <data_path> \
    -m <output_path> \
    --loss_type ahgs \
    --use_dash \
    --dash_r_min 4 \
    --dash_r_stages 3 \
    --iterations 100000 \
    --refine_extra_iters 60000 \
    --max_gaussians 7700000 \
    --lock_after_budget \
    --antialiasing
```

Using the helper script:
```bash
bash shell/train.sh <data_path> [output_path]
# Override env: ENV_PATH=my_env bash shell/train.sh <data_path>
# Override script: TRAIN_SCRIPT=train_custom.py bash shell/train.sh <data_path>
```

### Evaluation

```bash
bash shell/eval.sh <model_path>
# With specific resolution and downsampled metrics:
bash shell/eval.sh <model_path> -1 1920
```

---

## New Arguments Reference

All new arguments are added to `OptimizationParams` in `arguments/__init__.py` and accepted by `train_custom.py`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--loss_type` | str | `baseline` | Loss variant: `baseline`, `ahgs`, `fregs`, `depth_reg`, `normal_reg` |
| `--use_dash` | flag | off | Enable DashGaussian multi-resolution curriculum |
| `--dash_r_min` | int | `4` | Coarsest downscale factor (e.g. 4 = quarter resolution) |
| `--dash_r_stages` | int | `3` | Number of resolution stages |
| `--max_gaussians` | int | `0` | Maximum Gaussian count (0 = unlimited) |
| `--lock_after_budget` | flag | off | Freeze densification once budget is reached |
| `--refine_extra_iters` | int | `0` | Extra refinement iterations after main training |
| `--wandb_project` | str | — | W&B project name (optional) |
| `--wandb_name` | str | — | W&B run name (optional) |

---

## Repository Structure

```
gaussian-splatting/
├── train.py              # Original training script (unchanged)
├── train_custom.py       # Extended training script (this repo)
├── render.py             # Rendering (unchanged)
├── metrics.py            # Metrics (unchanged)
├── convert.py            # COLMAP conversion (unchanged)
├── arguments/
│   └── __init__.py       # + 6 new OptimizationParams flags
├── scene/
│   ├── gaussian_model.py # + PixelGS accum, budget/prune utils, elongation filter
│   └── dataset_readers.py # np.byte bugfix
├── shell/
│   ├── setup_env.sh      # Environment setup
│   ├── train.sh          # Training helper
│   ├── eval.sh           # Render + metrics
│   ├── colmap.sh         # COLMAP reconstruction
│   ├── fix_undistort.sh  # Re-run undistortion on best COLMAP sub-model
│   └── download_nerf_synthetic.sh
└── submodules/           # diff-gaussian-rasterization, simple-knn, fused-ssim
```

---

## References

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {42}, number = {4}, month = {July}, year = {2023},
}

@inproceedings{dashgaussian2025,
  title   = {DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.18402},
}

@article{pixelgs2024,
  title   = {PixelGS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting},
  year    = {2024},
  url     = {https://arxiv.org/abs/2403.12520},
}

@article{citygaussianv2,
  title   = {CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.00771},
}
```
