#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import pickle

import dotenv
import numpy as np
import torch
import torch.nn as nn
import timm
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.io import load_json, load_text
from model.T_Deed_Modules.shift import make_temporal_shift

FPS_SN = 25
OVERLAP_SNBA = 0.9
STRIDE_SNBA = 4
ARCHS = []
INTERNVIDEO2_ARCHS = {'internvideo2_1b'}
SPLITS = ['train', 'test']


def load_classes(file_name):
    ret = {x: i + 1 for i, x in enumerate(load_text(file_name))}
    ret["BACKGROUND"] = 0
    return ret


def build_clips_for_split(split, frame_dir, store_dir, clip_len, stride, dataset_path):
    label_file = os.path.join(dataset_path, 'soccernetballanticipation', f'{split}.json')
    if not os.path.exists(label_file):
        print(f"Skip build clips {split}: {label_file} not found")
        return False

    store_path = os.path.join(store_dir, f'LEN{clip_len}DIS0SPLIT{split}')
    if os.path.exists(os.path.join(store_path, 'frame_paths.pkl')):
        print(f"Clips already built for {split}")
        return True

    classes = load_classes(os.path.join(dataset_path, 'soccernetballanticipation', 'class.txt'))
    labels = load_json(label_file)
    overlap = int((1 - OVERLAP_SNBA) * clip_len)

    frame_paths = []
    labels_store = []

    for video in tqdm(labels, desc=f"Building clips {split}"):
        num_clips = int(video['num_clips'])
        full_video_len = int(video['num_frames'] // video['num_clips'])
        video_len = int(full_video_len * ((clip_len * stride / FPS_SN) + 5) / 30)
        labels_files = load_json(os.path.join(frame_dir, video['video'] + '/Labels-ball.json'))['videos']

        for clip_idx in range(num_clips):
            labels_file = labels_files[clip_idx]['annotations']['observation'] + labels_files[clip_idx]['annotations']['anticipation']
            for base_idx in range(0, max(1, video_len - 1 + (0 - clip_len) * stride), overlap):
                path = os.path.join(frame_dir, video['video'] + f"/clip_{clip_idx+1}")
                found_start = -1
                pad_start = 0
                pad_end = 0
                for frame_num in range(base_idx, base_idx + clip_len * stride, stride):
                    if frame_num < 0:
                        pad_start += 1
                        continue
                    if pad_end > 0:
                        pad_end += 1
                        continue
                    frame_path = os.path.join(path, f'frame{frame_num}.jpg')
                    if os.path.exists(frame_path) and found_start == -1:
                        found_start = frame_num
                    if not os.path.exists(frame_path):
                        pad_end += 1

                length = (base_idx + clip_len * stride - base_idx) // stride
                ret = [path, found_start, pad_start, pad_end, -1, length]

                if found_start != -1:
                    clip_labels = []
                    for event in labels_file:
                        event_frame = int(int(event['position']) / 1000 * FPS_SN)
                        label_idx = (event_frame - base_idx) // stride
                        if 0 <= label_idx < clip_len:
                            label = classes[event['label']]
                            clip_labels.append({'label': label, 'label_idx': label_idx})
                    frame_paths.append(ret)
                    labels_store.append(clip_labels)

    os.makedirs(store_path, exist_ok=True)
    with open(os.path.join(store_path, 'frame_paths.pkl'), 'wb') as f:
        pickle.dump(frame_paths, f)
    with open(os.path.join(store_path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels_store, f)
    print(f"Stored {len(frame_paths)} clips to {store_path}")
    return True


def create_feature_extractor(feature_arch, clip_len, device='cuda'):
    arch_map = {
        'rny002': 'regnety_002',
        'rny004': 'regnety_004',
        'rny006': 'regnety_006',
        'rny008': 'regnety_008',
    }
    arch_base = feature_arch.rsplit('_', 1)[0]
    model = timm.create_model(arch_map[arch_base], pretrained=True)
    feat_dim = model.head.fc.in_features
    model.head.fc = nn.Identity()

    max_obs_len = int(clip_len * 0.5)
    if feature_arch.endswith('_gsm'):
        make_temporal_shift(model, max_obs_len, mode='gsm')
    elif feature_arch.endswith('_gsf'):
        make_temporal_shift(model, max_obs_len, mode='gsf')

    model = model.to(device).eval()
    return model, feat_dim


def load_frames_for_clip(frame_path_info, stride):
    import torchvision
    base_path, start, pad_start, pad_end, ndigits, length = frame_path_info
    actual_len = length - pad_start - pad_end

    path_prefix = os.path.join(base_path, 'frame')
    zero_img = None
    frames = []

    for j in range(actual_len):
        fp = path_prefix + str(start + j * stride) + '.jpg'
        try:
            img = torchvision.io.read_image(fp)
            if zero_img is None:
                zero_img = torch.zeros_like(img)
            frames.append(img)
        except Exception:
            if zero_img is None:
                zero_img = torch.zeros(3, 224, 224, dtype=torch.uint8)
            frames.append(zero_img)

    if not frames:
        return None

    result = torch.stack(frames, dim=0)

    if pad_start > 0 or pad_end > 0:
        result = torch.nn.functional.pad(
            result, (0, 0, 0, 0, 0, 0, pad_start, pad_end))

    return result


def _n_segment_for_clip_len(clip_len):
    """Must match create_feature_extractor (max_obs_len for GSM/GSF)."""
    return int(clip_len * 0.5)


def _load_and_pad_clips(indices, frame_paths, stride, n_segment, fixed_t, num_workers=4):
    """Load clips from disk with parallel JPEG decoding, pad to fixed_t (constant shape for cudnn.benchmark)."""
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(idx):
        frames = load_frames_for_clip(frame_paths[idx], stride)
        return idx, frames

    raw = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for idx, frames in pool.map(_load_one, indices):
            if frames is not None:
                raw.append((idx, frames))

    if not raw:
        return None, [], [], 0

    valid_indices = [r[0] for r in raw]
    batch_frames = [r[1] for r in raw]
    orig_lens = [f.shape[0] for f in batch_frames]

    chunks = []
    for f in batch_frames:
        if f.shape[0] < fixed_t:
            pad = torch.zeros(fixed_t - f.shape[0], *f.shape[1:], dtype=f.dtype)
            f = torch.cat([f, pad], dim=0)
        elif f.shape[0] > fixed_t:
            f = f[:fixed_t]
        chunks.append(f)

    mega = torch.cat(chunks, dim=0)
    return mega, valid_indices, orig_lens, fixed_t


@torch.no_grad()
def _gpu_forward(model, mega_uint8, device, mean_t, std_t, use_amp):
    """Transfer uint8 tensor to GPU, normalize in-place, run model with optional AMP."""
    mega_pin = mega_uint8.pin_memory() if device.startswith("cuda") else mega_uint8
    gpu = mega_pin.to(device, dtype=torch.float32, non_blocking=True)
    gpu.div_(255.0).sub_(mean_t).div_(std_t)
    if use_amp:
        with torch.amp.autocast("cuda"):
            feats = model(gpu)
    else:
        feats = model(gpu)
    return feats


def _save_features_async(save_pool, feat_store_path, valid_indices, feats_np, orig_lens, t_padded):
    """Submit np.save calls to a thread pool so disk I/O doesn't block the next GPU batch."""
    d = feats_np.shape[-1]
    futures = []
    for i, idx in enumerate(valid_indices):
        sl = slice(i * t_padded, i * t_padded + orig_lens[i])
        feat = feats_np[sl]
        if feat.ndim == 1:
            feat = feat.reshape(1, d)
        path = os.path.join(feat_store_path, f'{idx:06d}.npy')
        futures.append(save_pool.submit(np.save, path, feat))
    return futures


def extract_features_for_split(split, store_dir, feature_dir, model, clip_len, stride, device, batch_size):
    from concurrent.futures import ThreadPoolExecutor

    store_path = os.path.join(store_dir, f'LEN{clip_len}DIS0SPLIT{split}')
    frame_paths_file = os.path.join(store_path, 'frame_paths.pkl')

    if not os.path.exists(frame_paths_file):
        print(f"Skip extract {split}: no frame_paths.pkl")
        return

    with open(frame_paths_file, 'rb') as f:
        frame_paths = pickle.load(f)

    feat_store_path = os.path.join(feature_dir, f'LEN{clip_len}DIS0SPLIT{split}')
    os.makedirs(feat_store_path, exist_ok=True)

    already = set()
    for fn in os.listdir(feat_store_path):
        if fn.endswith('.npy'):
            already.add(int(fn.replace('.npy', '')))

    todo_indices = [i for i in range(len(frame_paths)) if i not in already]
    if not todo_indices:
        print(f"{split}: all {len(frame_paths)} clips already extracted")
        return

    n_segment = _n_segment_for_clip_len(clip_len)
    rem = clip_len % n_segment
    fixed_t = clip_len + (n_segment - rem) if rem else clip_len
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    io_workers = min(8, os.cpu_count() or 4)
    print(
        f"Extracting {split}: {len(todo_indices)}/{len(frame_paths)} clips "
        f"(batch={batch_size}, fixed_t={fixed_t}, n_seg={n_segment}, amp={use_amp}, io_workers={io_workers})"
    )

    save_pool = ThreadPoolExecutor(max_workers=2)
    pending_saves = []
    prefetch_pool = ThreadPoolExecutor(max_workers=1)

    batches = [
        todo_indices[i:i + batch_size]
        for i in range(0, len(todo_indices), batch_size)
    ]

    prefetch_future = prefetch_pool.submit(
        _load_and_pad_clips, batches[0], frame_paths, stride, n_segment, fixed_t, io_workers
    ) if batches else None

    for b_idx in tqdm(range(len(batches)), desc=f"{split}"):
        mega, valid_indices, orig_lens, t_padded = prefetch_future.result()

        if b_idx + 1 < len(batches):
            prefetch_future = prefetch_pool.submit(
                _load_and_pad_clips, batches[b_idx + 1], frame_paths, stride, n_segment, fixed_t, io_workers
            )

        if mega is None:
            continue

        feats_gpu = _gpu_forward(model, mega, device, mean_t, std_t, use_amp)
        feats_np = feats_gpu.cpu().numpy()
        del feats_gpu

        for f in pending_saves:
            f.result()
        pending_saves = _save_features_async(
            save_pool, feat_store_path, valid_indices, feats_np, orig_lens, t_padded
        )

    for f in pending_saves:
        f.result()
    save_pool.shutdown(wait=True)
    prefetch_pool.shutdown(wait=False)

    print(f"Saved {split} features -> {feat_store_path}")


def _split_has_frames(download_path, frame_size, split):
    """Check if a split already has exported frame*.jpg files."""
    size_dir = "224p" if frame_size == "224p" else "720p"
    split_dir = os.path.join(download_path, size_dir, split)
    if not os.path.isdir(split_dir):
        return False
    for game_dir in os.scandir(split_dir):
        if not game_dir.is_dir():
            continue
        for clip_dir in os.scandir(game_dir.path):
            if not clip_dir.is_dir():
                continue
            for entry in os.scandir(clip_dir.path):
                if entry.name.startswith("frame") and entry.name.endswith(".jpg"):
                    return True
    return False


def download_data(download_path, download_key, frame_size, splits, export_only,
                   delete_videos, cpus):
    print(f"=== Step 0: Download & export data via setup_dataset_BAA.py ===")

    needed_splits = []
    for split in splits:
        if _split_has_frames(download_path, frame_size, split):
            print(f"Frames already exist for {split}, skipping.")
        else:
            needed_splits.append(split)

    if not needed_splits:
        print("All splits already have frames. Nothing to download/export.\n")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_script = os.path.join(script_dir, "setup_dataset_BAA.py")
    if not os.path.exists(setup_script):
        raise FileNotFoundError(f"setup_dataset_BAA.py not found at {setup_script}")

    for split in needed_splits:
        cmd = [
            sys.executable, setup_script,
            "--download-path", download_path,
            "--frame-size", frame_size,
            "--cpus", str(cpus),
            "--one-split", split,
        ]
        if export_only:
            cmd.append("--export-only")
        else:
            if download_key is None:
                raise ValueError(
                    "NDA key required: set NDA_KEY in .env or pass --download-key "
                    "(when not using --export-only or --skip-download)"
                )
            cmd.extend(["--download-key", download_key])
        if delete_videos:
            cmd.append("--delete-videos")

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("Data download/export complete.\n")


def main():
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-output", type=str, default="feature_output")
    parser.add_argument("--frame-dir", type=str, default="data/soccernetball/224p")
    parser.add_argument("--store-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default="data")
    parser.add_argument("--archs", nargs="+", default=ARCHS)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--clip-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=STRIDE_SNBA)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Số clip gộp trong một lần forward GPU (tăng dần đến khi gần đầy VRAM; giá trị cũ 256 "
        "chỉ đọc JPEG theo nhóm, không dùng hết GPU).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download/export and go straight to clip building & feature extraction")
    parser.add_argument("--download-path", type=str, default="./data/soccernetball",
                        help="Directory for dataset download (passed to setup_dataset_BAA.py)")
    parser.add_argument(
        "--download-key",
        type=str,
        default=os.environ.get("NDA_KEY") or None,
        help="NDA download key for unzipping (default: NDA_KEY from .env; passed to setup_dataset_BAA.py)",
    )
    parser.add_argument("--export-only", action="store_true",
                        help="Only export frames from already-downloaded zips (passed to setup_dataset_BAA.py)")
    parser.add_argument("--frame-size", type=str, default="224p", choices=["224p", "448p", "720p"],
                        help="Frame export resolution (passed to setup_dataset_BAA.py)")
    parser.add_argument("--delete-videos", action="store_true",
                        help="Delete zip/video files after export to save space")
    parser.add_argument("--cpus", type=int, default=12,
                        help="Number of CPUs for frame export")
    parser.add_argument(
        "--upload-drive",
        action="store_true",
        help="After each arch finishes, upload that arch folder under --feature-output via upload_drive.py",
    )
    parser.add_argument(
        "--drive-folder-id",
        type=str,
        default="1AXx6CwC8OdGAv8PnIgPrWkJ_JK1i4xYv",
        help="Drive parent folder id (same default as upload_drive.py)",
    )
    args = parser.parse_args()

    store_dir = args.store_dir or args.frame_dir
    os.makedirs(args.feature_output, exist_ok=True)

    if not args.skip_download:
        download_data(
            download_path=args.download_path,
            download_key=args.download_key,
            frame_size=args.frame_size,
            splits=args.splits,
            export_only=args.export_only,
            delete_videos=args.delete_videos,
            cpus=args.cpus,
        )

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=== Step 1: Build clip indices ===")
    for split in args.splits:
        build_clips_for_split(split, args.frame_dir, store_dir, args.clip_len, args.stride, args.dataset_path)

    print("\n=== Step 2: Extract features for each arch ===")
    upload_script = None
    if args.upload_drive:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        upload_script = os.path.join(script_dir, "upload_drive.py")
        if not os.path.isfile(upload_script):
            raise FileNotFoundError(f"upload_drive.py not found at {upload_script}")

    def _upload_split_if_needed(args, upload_script, feature_dir, arch, split):
        if not args.upload_drive:
            return
        split_dir = os.path.join(
            feature_dir, f'LEN{args.clip_len}DIS0SPLIT{split}'
        )
        if os.path.isdir(split_dir):
            cmd = [
                sys.executable, upload_script,
                "--path", os.path.abspath(split_dir),
                "--folder-id", args.drive_folder_id,
            ]
            print(f"\n=== Upload {arch}/{split} to Drive: {' '.join(cmd)} ===")
            subprocess.run(cmd, check=True)

    for arch in args.archs:
        feature_dir = os.path.join(args.feature_output, arch)
        os.makedirs(feature_dir, exist_ok=True)
        print(f"\n--- {arch} -> {feature_dir}/ ---")

        if arch in INTERNVIDEO2_ARCHS:
            from InternVideo2_extract import (
                load_internvideo2,
                extract_features_for_split as iv2_extract,
                INTERNVIDEO2_FEAT_DIM,
            )
            iv2_model = load_internvideo2(device=args.device)
            print(f"Feature dim: {INTERNVIDEO2_FEAT_DIM}")
            for split in args.splits:
                iv2_extract(
                    split, store_dir, feature_dir, iv2_model,
                    args.clip_len, args.stride, args.device, args.batch_size,
                )
                _upload_split_if_needed(
                    args, upload_script, feature_dir, arch, split
                )
            del iv2_model
            torch.cuda.empty_cache()
        else:
            model, feat_dim = create_feature_extractor(arch, args.clip_len, device=args.device)
            print(f"Feature dim: {feat_dim}")
            for split in args.splits:
                extract_features_for_split(
                    split, store_dir, feature_dir, model,
                    args.clip_len, args.stride, args.device, args.batch_size
                )
                _upload_split_if_needed(
                    args, upload_script, feature_dir, arch, split
                )
            del model
            torch.cuda.empty_cache()

    print(f"\nDone. Features saved at: {args.feature_output}/")
    for arch in args.archs:
        p = os.path.join(args.feature_output, arch)
        if os.path.exists(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            npy_count = sum(
                len([f for f in os.listdir(os.path.join(p, d)) if f.endswith('.npy')])
                for d in subdirs
            )
            print(f"  {arch}/: {subdirs} ({npy_count} .npy files)")


if __name__ == "__main__":
    main()