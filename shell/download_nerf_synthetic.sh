#!/bin/bash
# 下载 NeRF Synthetic 数据集 (Blender 格式)，用于测试 Gaussian Splatting 流程
# 用法: bash shell/download_nerf_synthetic.sh [target_dir]
# 默认下载到 ./data/nerf_synthetic/
set -e

TARGET_DIR=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/nerf_synthetic"}
mkdir -p "$TARGET_DIR"

# NeRF Synthetic Blender 格式（含 transforms_train.json），gaussian-splatting 支持此格式
# 注意：Synthetic_NSVF.zip 是另一种格式，不可用
DATASET_URL="https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip"

# 如果已有 lego 场景则跳过
if [ -d "$TARGET_DIR/Synthetic_NeRF/lego" ] || [ -d "$TARGET_DIR/lego" ]; then
    echo "[INFO] NeRF Synthetic dataset already exists at $TARGET_DIR"
    exit 0
fi

TMP_ZIP="/tmp/nerf_synthetic.zip"
echo "[INFO] Downloading NeRF Synthetic dataset (~700MB) to $TARGET_DIR"
echo "[INFO] This may take a few minutes..."

# 尝试下载（支持续传）
wget -c "$DATASET_URL" -O "$TMP_ZIP" || {
    echo "[WARN] wget failed, trying curl..."
    curl -L "$DATASET_URL" -o "$TMP_ZIP"
}

echo "[INFO] Extracting to $TARGET_DIR"
unzip -q "$TMP_ZIP" -d "$TARGET_DIR"
rm -f "$TMP_ZIP"

echo "[DONE] Dataset extracted to $TARGET_DIR"
echo "Available scenes:"
ls "$TARGET_DIR/"
echo ""
echo "To train on lego scene:"
echo "  bash shell/train.sh $TARGET_DIR/lego"
