#!/usr/bin/env python3
"""
Feature extractor using InternVideo2-CLIP-1B-224p-f8.

Produces per-frame features from the vision encoder, compatible with the
clip structure built by auto_extract.py (frame_paths.pkl / labels.pkl).

Usage (standalone):
    python InternVideo2_extract.py \
        --frame-dir data/soccernetball/224p \
        --store-dir data/soccernetball/224p \
        --feature-output feature_output/internvideo2_1b \
        --splits train test \
        --batch-size 4

Called from auto_extract.py when --archs includes 'internvideo2_1b'.
"""

import argparse
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

INTERNVIDEO2_MODEL_ID = "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"
INTERNVIDEO2_NUM_FRAMES = 8
INTERNVIDEO2_IMG_SIZE = 224
INTERNVIDEO2_FEAT_DIM = 768

V_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
V_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def load_internvideo2(device="cuda"):
    from transformers import AutoModel

    print(f"Loading {INTERNVIDEO2_MODEL_ID} ...")
    model = AutoModel.from_pretrained(
        INTERNVIDEO2_MODEL_ID, trust_remote_code=True
    )
    model = model.to(device).eval()
    print(f"InternVideo2-CLIP-1B loaded on {device}, dtype={model.dtype}")
    return model


def load_frames_for_clip(frame_path_info, stride):
    """Reuse the same JPEG loading logic as auto_extract."""
    import torchvision

    base_path, start, pad_start, pad_end, _ndigits, length = frame_path_info
    actual_len = length - pad_start - pad_end

    path_prefix = os.path.join(base_path, "frame")
    zero_img = None
    frames = []

    for j in range(actual_len):
        fp = path_prefix + str(start + j * stride) + ".jpg"
        try:
            img = torchvision.io.read_image(fp)
            if zero_img is None:
                zero_img = torch.zeros_like(img)
            frames.append(img)
        except Exception:
            if zero_img is None:
                zero_img = torch.zeros(3, INTERNVIDEO2_IMG_SIZE, INTERNVIDEO2_IMG_SIZE, dtype=torch.uint8)
            frames.append(zero_img)

    if not frames:
        return None

    result = torch.stack(frames, dim=0)

    if pad_start > 0 or pad_end > 0:
        result = torch.nn.functional.pad(
            result, (0, 0, 0, 0, 0, 0, pad_start, pad_end)
        )

    return result


def _subsample_frames(frames_t, num_frames):
    """Uniformly subsample T frames down to num_frames (InternVideo2 expects 8).
    Returns (subsampled_tensor [num_frames, C, H, W], original_T)."""
    T = frames_t.shape[0]
    if T <= num_frames:
        if T < num_frames:
            pad = torch.zeros(num_frames - T, *frames_t.shape[1:], dtype=frames_t.dtype)
            frames_t = torch.cat([frames_t, pad], dim=0)
        return frames_t, T
    indices = torch.linspace(0, T - 1, num_frames).long()
    return frames_t[indices], T


def _load_batch(indices, frame_paths, stride, num_frames, num_workers=4):
    """Load and subsample clips in parallel. Returns (tensor [B,T,C,H,W], valid_indices, orig_lens)."""

    def _load_one(idx):
        frames = load_frames_for_clip(frame_paths[idx], stride)
        return idx, frames

    raw = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for idx, frames in pool.map(_load_one, indices):
            if frames is not None:
                raw.append((idx, frames))

    if not raw:
        return None, [], []

    valid_indices = [r[0] for r in raw]
    orig_lens = [r[1].shape[0] for r in raw]

    subsampled = []
    for _, frames in raw:
        sub, _ = _subsample_frames(frames, num_frames)
        subsampled.append(sub)

    batch = torch.stack(subsampled, dim=0)
    return batch, valid_indices, orig_lens


@torch.no_grad()
def _extract_features(model, batch_uint8, device, mean_t, std_t, use_amp):
    """Run InternVideo2 vision encoder on a batch [B, T, C, H, W] of uint8 frames.
    Returns pooled features [B, D] as numpy."""
    batch_f = batch_uint8.float().div_(255.0)
    batch_f = batch_f.unsqueeze(2) if batch_f.ndim == 4 else batch_f
    # batch_f: [B, T, C, H, W]
    batch_f = (batch_f - mean_t) / std_t
    batch_f = batch_f.to(device, non_blocking=True)

    if use_amp:
        with torch.amp.autocast("cuda"):
            vfeat = model.get_vid_feat(batch_f)
    else:
        vfeat = model.get_vid_feat(batch_f)

    return vfeat.float().cpu().numpy()


def _save_async(save_pool, feat_store_path, valid_indices, feats_np):
    futures = []
    for i, idx in enumerate(valid_indices):
        feat = feats_np[i]
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        path = os.path.join(feat_store_path, f"{idx:06d}.npy")
        futures.append(save_pool.submit(np.save, path, feat))
    return futures


def extract_features_for_split(
    split, store_dir, feature_dir, model, clip_len, stride, device, batch_size
):
    store_path = os.path.join(store_dir, f"LEN{clip_len}DIS0SPLIT{split}")
    frame_paths_file = os.path.join(store_path, "frame_paths.pkl")

    if not os.path.exists(frame_paths_file):
        print(f"Skip extract {split}: no frame_paths.pkl")
        return

    with open(frame_paths_file, "rb") as f:
        frame_paths = pickle.load(f)

    feat_store_path = os.path.join(feature_dir, f"LEN{clip_len}DIS0SPLIT{split}")
    os.makedirs(feat_store_path, exist_ok=True)

    already = set()
    for fn in os.listdir(feat_store_path):
        if fn.endswith(".npy"):
            already.add(int(fn.replace(".npy", "")))

    todo_indices = [i for i in range(len(frame_paths)) if i not in already]
    if not todo_indices:
        print(f"{split}: all {len(frame_paths)} clips already extracted")
        return

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    mean_t = V_MEAN.clone()
    std_t = V_STD.clone()
    io_workers = min(8, os.cpu_count() or 4)
    num_frames = INTERNVIDEO2_NUM_FRAMES

    print(
        f"Extracting {split}: {len(todo_indices)}/{len(frame_paths)} clips "
        f"(batch={batch_size}, num_frames={num_frames}, amp={use_amp}, feat_dim={INTERNVIDEO2_FEAT_DIM})"
    )

    save_pool = ThreadPoolExecutor(max_workers=2)
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    pending_saves = []

    batches = [
        todo_indices[i : i + batch_size]
        for i in range(0, len(todo_indices), batch_size)
    ]

    prefetch_future = (
        prefetch_pool.submit(
            _load_batch, batches[0], frame_paths, stride, num_frames, io_workers
        )
        if batches
        else None
    )

    for b_idx in tqdm(range(len(batches)), desc=f"{split}"):
        batch_tensor, valid_indices, orig_lens = prefetch_future.result()

        if b_idx + 1 < len(batches):
            prefetch_future = prefetch_pool.submit(
                _load_batch, batches[b_idx + 1], frame_paths, stride, num_frames, io_workers
            )

        if batch_tensor is None:
            continue

        feats_np = _extract_features(model, batch_tensor, device, mean_t, std_t, use_amp)

        for f in pending_saves:
            f.result()
        pending_saves = _save_async(save_pool, feat_store_path, valid_indices, feats_np)

    for f in pending_saves:
        f.result()
    save_pool.shutdown(wait=True)
    prefetch_pool.shutdown(wait=False)

    print(f"Saved {split} features -> {feat_store_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract InternVideo2-CLIP-1B features")
    parser.add_argument("--frame-dir", type=str, default="data/soccernetball/224p")
    parser.add_argument("--store-dir", type=str, default=None)
    parser.add_argument("--feature-output", type=str, default="feature_output/internvideo2_1b")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--clip-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    store_dir = args.store_dir or args.frame_dir
    os.makedirs(args.feature_output, exist_ok=True)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_internvideo2(device=args.device)

    for split in args.splits:
        extract_features_for_split(
            split, store_dir, args.feature_output, model,
            args.clip_len, args.stride, args.device, args.batch_size,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone. Features at: {args.feature_output}/")


if __name__ == "__main__":
    main()
