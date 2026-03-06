#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHS = ['rny002_gsf', 'rny004_gsf', 'rny006_gsf', 'rny008_gsf']


def main():
    parser = argparse.ArgumentParser(
        description="Extract features cho tất cả kiến trúc RegNet vào feature_output/<arch>/"
    )
    parser.add_argument("--feature-output", type=str, default="feature_output",
                        help="Thư mục chính chứa features (default: feature_output)")
    parser.add_argument("--frame-dir", type=str, default="data/soccernetball/224p",
                        help="Thư mục frames")
    parser.add_argument("--store-dir", type=str, default=None,
                        help="Thư mục store (default: cùng frame-dir)")
    parser.add_argument("--dataset-path", type=str, default="data",
                        help="Đường dẫn dataset")
    parser.add_argument("--archs", nargs="+", default=ARCHS,
                        help=f"Danh sách arch để extract (default: {ARCHS})")
    parser.add_argument("--build-clips", action="store_true",
                        help="Build clip index trước khi extract")
    args = parser.parse_args()

    store_dir = args.store_dir or args.frame_dir
    os.makedirs(args.feature_output, exist_ok=True)

    extract_script = os.path.join(SCRIPT_DIR, "extract_features.py")
    if not os.path.exists(extract_script):
        print(f"Lỗi: Không tìm thấy {extract_script}")
        sys.exit(1)

    for arch in args.archs:
        feature_dir = os.path.join(args.feature_output, arch)
        print(f"\n=== Extracting features với {arch} -> {feature_dir}/ ===")
        cmd = [
            sys.executable, extract_script,
            "--frame-dir", args.frame_dir,
            "--store-dir", store_dir,
            "--feature-dir", feature_dir,
            "--dataset-path", args.dataset_path,
            "--feature-arch", arch,
        ]
        if args.build_clips:
            cmd.append("--build-clips")

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Lỗi khi extract {arch}")
            sys.exit(ret.returncode)

    print(f"\nXong. Features lưu tại: {args.feature_output}/")
    for arch in args.archs:
        p = os.path.join(args.feature_output, arch)
        if os.path.exists(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            print(f"  {arch}/ : {subdirs}")


if __name__ == "__main__":
    main()
