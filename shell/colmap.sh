#!/bin/bash
# 在计算节点容器内运行 COLMAP 稀疏重建
# 用法: bash shell/colmap.sh <source_path>
#   source_path: 数据集目录，需含 input/ 子目录
set -e

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH=${ENV_PATH:-gaussian_splatting}

SOURCE_PATH=${1:-""}
if [ -z "$SOURCE_PATH" ]; then
    echo "[ERROR] Usage: bash shell/colmap.sh <source_path>"
    exit 1
fi

if [ ! -d "$SOURCE_PATH/input" ]; then
    echo "[ERROR] $SOURCE_PATH/input not found"
    exit 1
fi

conda activate "$ENV_PATH"

COLMAP_BIN=${COLMAP_BIN:-$(which colmap 2>/dev/null || echo colmap)}
echo "[INFO] Using colmap: $COLMAP_BIN"

cd "$CODE_DIR"
echo "[INFO] Running COLMAP sparse reconstruction on: $SOURCE_PATH"
python convert.py -s "$SOURCE_PATH" --colmap_executable "$COLMAP_BIN"

echo "[DONE] COLMAP finished: $SOURCE_PATH"
