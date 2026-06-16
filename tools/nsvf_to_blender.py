"""
将 NSVF 格式数据集转换为 NeRF Blender 格式（gaussian-splatting 支持的格式）

NSVF 格式:
  intrinsics.txt: fx cx cy 0. / ... / W H
  pose/{split}_{idx}.txt: 4x4 c2w 矩阵
  rgb/{split}_{idx}.png: 图片 (0_=train, 1_=val, 2_=test)

输出 Blender 格式:
  transforms_train/val/test.json
  train/, val/, test/ 目录（软链接图片）
"""
import os
import json
import math
import shutil
import argparse
import numpy as np

SPLIT_MAP = {"0": "train", "1": "val", "2": "test"}


def read_intrinsics(path):
    with open(path) as f:
        lines = f.read().strip().split("\n")
    # 第一行: fx cx cy 0.
    first = lines[0].split()
    fx = float(first[0])
    # 最后一行: W H
    last = lines[-1].split()
    W, H = int(last[0]), int(last[1])
    return fx, W, H


def read_pose(path):
    with open(path) as f:
        rows = [list(map(float, line.split())) for line in f.read().strip().split("\n")]
    return rows  # 4x4 c2w


def convert(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    fx, W, H = read_intrinsics(os.path.join(src_dir, "intrinsics.txt"))
    camera_angle_x = 2 * math.atan(W / (2 * fx))

    pose_dir = os.path.join(src_dir, "pose")
    rgb_dir = os.path.join(src_dir, "rgb")

    frames_by_split = {"train": [], "val": [], "test": []}

    for fname in sorted(os.listdir(pose_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]          # e.g. "0_0042"
        split_id, idx = stem.split("_", 1)
        split = SPLIT_MAP.get(split_id)
        if split is None:
            continue

        c2w = read_pose(os.path.join(pose_dir, fname))

        img_src = os.path.join(rgb_dir, f"{stem}.png")
        split_img_dir = os.path.join(dst_dir, split)
        os.makedirs(split_img_dir, exist_ok=True)
        img_dst = os.path.join(split_img_dir, f"r_{idx}.png")
        if not os.path.exists(img_dst):
            os.symlink(os.path.abspath(img_src), img_dst)

        frames_by_split[split].append({
            "file_path": f"./{split}/r_{idx}",
            "transform_matrix": c2w,
        })

    for split, frames in frames_by_split.items():
        out = {"camera_angle_x": camera_angle_x, "frames": frames}
        with open(os.path.join(dst_dir, f"transforms_{split}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{split}] {len(frames)} frames written")

    print(f"[DONE] Converted to {dst_dir}")
    print(f"  camera_angle_x = {camera_angle_x:.6f}  ({W}x{H}, fx={fx:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="NSVF 场景目录 (含 intrinsics.txt / pose / rgb)")
    parser.add_argument("dst", help="输出目录")
    args = parser.parse_args()
    convert(args.src, args.dst)
