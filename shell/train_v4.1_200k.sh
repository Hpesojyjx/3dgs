#!/bin/bash
# v4.1 ultra-long config: DashGaussian (4×→2×→1×) + PixelGS + AH-GS + 10M budget lock
#   100k main training + 100k refinement tail = 200k total
#   Maximum quality setting; allows up to 10M Gaussians before locking.
# Usage: bash shell/train_v4.1_200k.sh <data_path> [output_path]
set -e

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH=${ENV_PATH:-gaussian_splatting}

DATA_PATH=${1:-""}
SCENE_NAME=$(basename "${DATA_PATH:-unknown}")
OUTPUT_PATH=${2:-"$CODE_DIR/output/ahgs_v4.1_200k_${SCENE_NAME}_$(date +%m%d-%H%M)"}

if [ -z "$DATA_PATH" ]; then
    echo "[ERROR] Usage: bash shell/train_v4.1_200k.sh <data_path> [output_path]"
    echo "  data_path must be a COLMAP dataset directory containing sparse/0/"
    exit 1
fi

RESOLUTION=${RESOLUTION:-1}
DATA_DEVICE=${DATA_DEVICE:-cpu}

conda activate "$ENV_PATH"
mkdir -p "$OUTPUT_PATH"
cd "$CODE_DIR"

echo "Training v4.1-200k: DashGaussian + PixelGS + AH-GS + 10M lock"
echo "  Data:    $DATA_PATH"
echo "  Output:  $OUTPUT_PATH"

python train_custom.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --disable_viewer \
    --loss_type ahgs \
    -r $RESOLUTION \
    --data_device $DATA_DEVICE \
    --iterations 100000 \
    --refine_extra_iters 100000 \
    --densify_grad_threshold 0.0002 \
    --position_lr_max_steps 100000 \
    --opacity_reset_interval 5000 \
    --test_iterations 30000 50000 100000 150000 200000 \
    --save_iterations 50000 100000 150000 200000 \
    --use_dash \
    --dash_r_min 4 \
    --dash_r_stages 3 \
    --max_gaussians 10000000 \
    --lock_after_budget \
    --antialiasing \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
    ${WANDB_NAME:+--wandb_name "$WANDB_NAME"}

echo "[DONE] Results at $OUTPUT_PATH"
