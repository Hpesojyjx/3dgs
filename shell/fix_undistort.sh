#!/bin/bash
# 用最大的 COLMAP 子模型重新跑去畸变
# 用法: bash shell/fix_undistort.sh <source_path> <best_model_id>
#   source_path:   数据集目录 (含 distorted/sparse/)
#   best_model_id: 最大子模型编号 (如 2)
set -e

ENV_PATH=${ENV_PATH:-gaussian_splatting}

SOURCE_PATH=${1:-""}
BEST_MODEL=${2:-""}

if [ -z "$SOURCE_PATH" ] || [ -z "$BEST_MODEL" ]; then
    echo "[ERROR] Usage: bash shell/fix_undistort.sh <source_path> <best_model_id>"
    exit 1
fi

conda activate "$ENV_PATH"

COLMAP_BIN=${COLMAP_BIN:-$(which colmap 2>/dev/null || echo colmap)}

echo "[INFO] Swapping model $BEST_MODEL -> model 0 in $SOURCE_PATH"
rm -rf $SOURCE_PATH/distorted/sparse/0
cp -r $SOURCE_PATH/distorted/sparse/$BEST_MODEL $SOURCE_PATH/distorted/sparse/0

rm -rf $SOURCE_PATH/images $SOURCE_PATH/sparse $SOURCE_PATH/stereo

echo "[INFO] Running image undistortion"
$COLMAP_BIN image_undistorter \
    --image_path $SOURCE_PATH/input \
    --input_path $SOURCE_PATH/distorted/sparse/0 \
    --output_path $SOURCE_PATH \
    --output_type COLMAP

mkdir -p $SOURCE_PATH/sparse/0
for f in $SOURCE_PATH/sparse/*; do
    [ "$(basename $f)" = "0" ] && continue
    mv $f $SOURCE_PATH/sparse/0/
done

echo "[INFO] Images in final reconstruction: $(ls $SOURCE_PATH/images | wc -l)"
echo "[DONE] Undistortion complete: $SOURCE_PATH"
