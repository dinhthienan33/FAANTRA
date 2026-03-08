#!/usr/bin/env python3

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_clip_dirs(frame_dir: str) -> List[str]:
    """Find clip directories by looking for directories that contain frame*.jpg files."""
    clip_dirs = []
    for root, dirs, files in os.walk(frame_dir):
        # Heuristic: a clip directory contains at least one frame*.jpg
        has_frame = any(f.startswith("frame") and f.endswith(".jpg") for f in files)
        if has_frame:
            clip_dirs.append(root)
    clip_dirs.sort(key=_natural_key)
    return clip_dirs


def list_frame_paths(clip_dir: str) -> List[str]:
    frames = [f for f in os.listdir(clip_dir) if f.startswith("frame") and f.endswith(".jpg")]
    frames.sort(key=_natural_key)
    return [os.path.join(clip_dir, f) for f in frames]


def load_and_preprocess_frames(
    frame_paths: List[str],
    image_size_hw: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """Load frames as tensor [T, 3, H, W], float32 normalized."""
    T = len(frame_paths)
    if T == 0:
        raise ValueError("No frames found")

    out = torch.empty((T, 3, image_size_hw[0], image_size_hw[1]), dtype=torch.float32)
    for i, p in enumerate(frame_paths):
        img = torchvision.io.read_image(p)  # uint8 [C,H,W]
        # Resize to (H,W)
        try:
            img = TF.resize(img, size=list(image_size_hw), antialias=True)
        except TypeError:
            img = TF.resize(img, size=list(image_size_hw))
        img = img.float().div(255.0)
        img = TF.normalize(img, mean=mean, std=std)
        out[i] = img
    return out


def uniform_indices(length: int, n: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be > 0")
    if n <= 0:
        raise ValueError("n must be > 0")
    if length == 1:
        return np.zeros((n,), dtype=np.int64)
    # pick center of each segment
    seg = (length - 1) / float(n)
    idx = np.array([int(seg * (i + 0.5)) for i in range(n)], dtype=np.int64)
    return np.clip(idx, 0, length - 1)


@dataclass
class WindowPlan:
    starts: List[int]
    ends: List[int]


def build_sliding_windows(total_frames: int, window: int, stride: int) -> WindowPlan:
    if total_frames <= 0:
        return WindowPlan([], [])
    if total_frames <= window:
        return WindowPlan([0], [total_frames])

    starts = []
    ends = []
    s = 0
    while True:
        e = min(s + window, total_frames)
        if e - s < window:
            s = max(0, e - window)
            e = min(s + window, total_frames)
        starts.append(s)
        ends.append(e)
        if e >= total_frames:
            break
        s = s + stride
    return WindowPlan(starts, ends)


def interpolate_window_features(
    window_feats: torch.Tensor,  # [N, D]
    starts: List[int],
    ends: List[int],
    total_frames: int,
) -> torch.Tensor:
    """Interpolate window-level features to per-frame features [T, D] by averaging overlaps."""
    device = window_feats.device
    N, D = window_feats.shape
    out = torch.zeros((total_frames, D), device=device, dtype=window_feats.dtype)
    cnt = torch.zeros((total_frames, 1), device=device, dtype=window_feats.dtype)
    for i in range(N):
        s = starts[i]
        e = ends[i]
        out[s:e] += window_feats[i].unsqueeze(0)
        cnt[s:e] += 1
    out = out / cnt.clamp(min=1.0)
    return out


def load_internvideo2_model(model_id: str, device: str, dtype: torch.dtype):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    if not hasattr(model, "get_vid_feat"):
        raise AttributeError(
            f"Model '{model_id}' does not expose get_vid_feat(). "
            "Try a Stage2 checkpoint that supports retrieval/feature extraction."
        )
    return model


def get_video_embedding_stage2(model, frames_tchw: torch.Tensor) -> torch.Tensor:
    """Return window-level embedding [D] using model.get_vid_feat().

    frames_tchw: [T, 3, H, W] normalized float32/float16
    """
    # InternVideo2 Stage2 HF code expects [B, T, C, H, W] for get_vid_feat in many examples.
    vid = frames_tchw.unsqueeze(0)  # [1, T, C, H, W]
    with torch.no_grad():
        feat = model.get_vid_feat(vid)
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if feat.ndim == 2:
        feat = feat[0]
    return feat


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_features(out_path: str, features_td: torch.Tensor, meta: dict):
    ensure_dir(os.path.dirname(out_path))
    payload = {"features": features_td.cpu(), "meta": meta}
    torch.save(payload, out_path)


# Presets: (model_id, fnum, embedding_dim). Embedding dim is model-dependent; set in config when training.
INTERNVIDEO2_PRESETS = {
    "stage2_6b": {
        "model_id": "OpenGVLab/InternVideo2-Stage2_6B-224p-f4",
        "fnum": 4,
        "image_size": 224,
        "embedding_dim": 1024,
    },
    "stage2_1b": {
        "model_id": "OpenGVLab/InternVideo2-CLIP-1B-224p-f8",
        "fnum": 8,
        "image_size": 224,
        "embedding_dim": 768,
    },
}


def main():
    ap = argparse.ArgumentParser(
        description="Extract InternVideo2 features for FAANTRA pre-extracted feature training."
    )
    ap.add_argument("--frame_dir", type=str, required=True, help="Root frames directory (same as config frame_dir).")
    ap.add_argument("--output_dir", type=str, required=True, help="Root directory to store features (preextracted_feature_dir).")
    ap.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(INTERNVIDEO2_PRESETS.keys()),
        help="Use preset: stage2_6b (heavy, 1024-d) or stage2_1b (lighter, 768-d). Overrides model_id/fnum/image_size if set.",
    )
    ap.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HuggingFace model id (requires trust_remote_code). Ignored if --preset is set.",
    )
    ap.add_argument("--image_size", type=int, default=None, help="Resize frames to image_size x image_size. Ignored if --preset is set.")
    ap.add_argument("--fnum", type=int, default=None, help="Frames per window fed to model. Ignored if --preset is set.")
    ap.add_argument("--window", type=int, default=16, help="How many frames to cover per window before sampling fnum.")
    ap.add_argument("--stride", type=int, default=8, help="Sliding window stride (frames).")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of clips to process (0=all).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    if args.preset:
        p = INTERNVIDEO2_PRESETS[args.preset]
        args.model_id = p["model_id"]
        args.fnum = p["fnum"]
        args.image_size = p["image_size"]
        print(f"Using preset '{args.preset}': model_id={args.model_id}, fnum={args.fnum}, image_size={args.image_size}")
    if args.model_id is None:
        ap.error("Either --preset or --model_id is required.")
    if args.fnum is None:
        args.fnum = 4
    if args.image_size is None:
        args.image_size = 224

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    clip_dirs = list_clip_dirs(args.frame_dir)
    if args.limit and args.limit > 0:
        clip_dirs = clip_dirs[: args.limit]
    if len(clip_dirs) == 0:
        raise SystemExit(f"No clip directories found under frame_dir='{args.frame_dir}'")

    model = load_internvideo2_model(args.model_id, device=args.device, dtype=dtype)

    image_size_hw = (args.image_size, args.image_size)

    for clip_dir in clip_dirs:
        rel = os.path.relpath(clip_dir, args.frame_dir)
        out_path = os.path.join(args.output_dir, rel, "features.pt")

        frame_paths = list_frame_paths(clip_dir)
        total = len(frame_paths)
        if total == 0:
            continue

        frames = load_and_preprocess_frames(frame_paths, image_size_hw=image_size_hw, mean=mean, std=std)
        frames = frames.to(device=args.device, dtype=dtype)

        plan = build_sliding_windows(total_frames=total, window=args.window, stride=args.stride)
        window_embs = []
        for s, e in zip(plan.starts, plan.ends):
            segment = frames[s:e]  # [seg, 3, H, W]
            idx = uniform_indices(len(segment), args.fnum)
            sampled = segment[idx]  # [fnum, 3, H, W]
            emb = get_video_embedding_stage2(model, sampled)  # [D]
            window_embs.append(emb)

        window_embs = torch.stack(window_embs, dim=0)  # [N, D]
        per_frame = interpolate_window_features(window_embs, plan.starts, plan.ends, total_frames=total)  # [T, D]

        meta = {
            "model_id": args.model_id,
            "image_size": args.image_size,
            "fnum": args.fnum,
            "window": args.window,
            "stride": args.stride,
            "total_frames": total,
            "embedding_dim": int(per_frame.shape[1]),
        }
        write_features(out_path, per_frame, meta)
        print(f"[ok] {rel} -> {out_path}  shape={tuple(per_frame.shape)}")


if __name__ == "__main__":
    main()

