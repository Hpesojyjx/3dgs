#!/bin/bash
# 创建 conda 环境并编译 CUDA 扩展
# 用法: bash shell/setup_env.sh [env_name_or_prefix]
#   env_name_or_prefix: conda 环境名或路径，默认 gaussian_splatting
set -e

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH=${1:-${ENV_PATH:-gaussian_splatting}}

# 非交互环境下接受 conda 默认 channel 的 ToS（可选，部分版本需要）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 如果环境已存在则跳过创建
if conda env list | grep -qE "(^|/)${ENV_PATH}( |$)"; then
    echo "[INFO] Conda env '$ENV_PATH' already exists, skipping creation"
else
    echo "[INFO] Creating conda env: $ENV_PATH"
    conda create --name "$ENV_PATH" python=3.10 -y
fi

conda activate "$ENV_PATH"

echo "[INFO] Installing PyTorch with CUDA 11.8"
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "[INFO] Installing base dependencies"
pip install plyfile tqdm opencv-python joblib tensorboard

echo "[INFO] Compiling CUDA extensions (this may take several minutes)"
cd "$CODE_DIR"

pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/fused-ssim

echo "[INFO] Installing COLMAP"
conda install --override-channels -c conda-forge colmap -y

echo "[INFO] Verifying installation"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
from diff_gaussian_rasterization import GaussianRasterizationSettings
print('diff-gaussian-rasterization: OK')
from simple_knn._C import distCUDA2
print('simple-knn: OK')
from fused_ssim import fused_ssim
print('fused-ssim: OK')
"
colmap --version && echo "colmap: OK"

echo "[DONE] Environment ready: $ENV_PATH"
