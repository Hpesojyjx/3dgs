#!/bin/bash
# 在计算节点容器内运行训练
# 入参: $1 = 数据集路径, $2 = 输出路径 (可选)
# 用法: bash shell/train.sh <data_path> [output_path]
set -e

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH=${ENV_PATH:-gaussian_splatting}

DATA_PATH=${1:-""}
OUTPUT_PATH=${2:-"$CODE_DIR/output/$(date +%Y%m%d_%H%M%S)"}

if [ -z "$DATA_PATH" ]; then
    echo "[ERROR] Usage: bash shell/train.sh <data_path> [output_path]"
    echo "  data_path: COLMAP 或 NeRF Synthetic 格式的数据集目录"
    exit 1
fi

conda activate "$ENV_PATH"

cd "$CODE_DIR"

echo "[INFO] Training on: $DATA_PATH"
echo "[INFO] Output:      $OUTPUT_PATH"
echo "[INFO] GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"

RESOLUTION=${RESOLUTION:-1}
DATA_DEVICE=${DATA_DEVICE:-cpu}

python ${TRAIN_SCRIPT:-train.py} \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --disable_viewer \
    -r $RESOLUTION \
    --data_device $DATA_DEVICE \
    --test_iterations 7000 30000 \
    --save_iterations 7000 30000 \
    --checkpoint_iterations 7000 30000 \
    ${EXTRA_ARGS:-}

echo "[DONE] Training complete. Results at $OUTPUT_PATH"
