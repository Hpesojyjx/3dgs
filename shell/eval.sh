#!/bin/bash
# 对指定模型目录运行 render + metrics
# 用法: bash shell/eval.sh <model_path> [render_resolution] [eval_width]
#   render_resolution: 传给 render.py 的 -r 参数，默认 -1
#   eval_width:        render 后 downsample 到的宽度，0 表示不 downsample，默认 0
set -e

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH=${ENV_PATH:-gaussian_splatting}

MODEL_PATH=${1:-""}
RESOLUTION=${2:-"-1"}
EVAL_WIDTH=${3:-"0"}
if [ -z "$MODEL_PATH" ]; then
    echo "[ERROR] Usage: bash shell/eval.sh <model_path> [render_resolution] [eval_width]"
    exit 1
fi

conda activate "$ENV_PATH"

cd "$CODE_DIR"

# 自动检测已有的 iteration
ITERS=$(ls $MODEL_PATH/point_cloud/ | grep iteration_ | sed 's/iteration_//' | sort -n)
echo "[INFO] Found iterations: $ITERS"

for ITER in $ITERS; do
    echo "[INFO] Rendering iteration $ITER at resolution -r $RESOLUTION"
    python render.py -m "$MODEL_PATH" -s "$CODE_DIR" --iteration $ITER --skip_train --eval -r $RESOLUTION --data_device cpu

    if [ "$EVAL_WIDTH" -gt 0 ]; then
        echo "[INFO] Downsampling renders and gt to width=$EVAL_WIDTH for metrics"
        python - <<PYEOF
import os
from pathlib import Path
from PIL import Image

base = Path("$MODEL_PATH") / "test" / "ours_$ITER"
for subdir in ["renders", "gt"]:
    d = base / subdir
    for p in sorted(d.glob("*.png")):
        img = Image.open(p)
        w, h = img.size
        new_w = $EVAL_WIDTH
        new_h = round(h * new_w / w)
        img.resize((new_w, new_h), Image.LANCZOS).save(p)
print(f"  Resized to {new_w}x{new_h}")
PYEOF
    fi
done

echo "[INFO] Computing metrics"
python metrics.py -m $MODEL_PATH

echo "[DONE] $(basename $MODEL_PATH)"
cat $MODEL_PATH/results.json
