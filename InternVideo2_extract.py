#!/usr/bin/env python3
"""
Feature extractor using InternVideo2-Stage2_1B (1B vision encoder, 224p, f8).

This script bootstraps the InternVideo2 codebase from GitHub and loads
the pretrained weights from HuggingFace, then extracts per-clip pooled
features (768-dim) for each clip built by auto_extract.py.

Requirements:
    pip install flash-attn --no-build-isolation   # optional but recommended
    pip install transformers einops timm

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

INTERNVIDEO2_REPO = "OpenGVLab/InternVideo"
INTERNVIDEO2_HF_REPO_STAGE2 = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
INTERNVIDEO2_HF_REPO_CLIP = "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"
INTERNVIDEO2_NUM_FRAMES = 8
INTERNVIDEO2_IMG_SIZE = 224
INTERNVIDEO2_FEAT_DIM = 768

V_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
V_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _ensure_internvideo2_repo():
    """Clone InternVideo2 multi_modality code if not present, patch for optional
    flash_attn, add to sys.path."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "internvideo2_repo")
    mm_dir = os.path.join(cache_dir, "InternVideo", "InternVideo2", "multi_modality")
    patched_marker = os.path.join(mm_dir, ".patched_noflash")

    if not os.path.isdir(mm_dir):
        import subprocess
        print(f"Cloning InternVideo2 repo to {cache_dir} ...")
        os.makedirs(cache_dir, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
             f"https://github.com/{INTERNVIDEO2_REPO}.git"],
            cwd=cache_dir, check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "InternVideo2/multi_modality"],
            cwd=os.path.join(cache_dir, "InternVideo"), check=True,
        )
        print("Clone done.")

    if not os.path.exists(patched_marker):
        _patch_internvideo2_for_optional_flash_attn(mm_dir)
        with open(patched_marker, "w") as f:
            f.write("patched")

    if mm_dir not in sys.path:
        sys.path.insert(0, mm_dir)
    demo_dir = os.path.join(mm_dir, "demo")
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

    return mm_dir


def _patch_internvideo2_for_optional_flash_attn(mm_dir):
    """Patch InternVideo2 source files so flash_attn is optional (fallback to
    PyTorch SDPA) and __init__.py only imports what we need."""
    iv2_pkg = os.path.join(mm_dir, "models", "backbones", "internvideo2")

    # 1) Patch __init__.py — only import what 1B needs, skip 6B/LLaMA/MobileCLIP
    init_path = os.path.join(iv2_pkg, "__init__.py")
    with open(init_path, "w") as f:
        f.write(
            "from .internvideo2 import pretrain_internvideo2_1b_patch14_224\n"
            "from .pos_embed import interpolate_pos_embed_internvideo2_new\n"
        )

    # 2) Patch flash_attention_class.py — fallback to PyTorch SDPA
    fa_path = os.path.join(iv2_pkg, "flash_attention_class.py")
    with open(fa_path, "w") as f:
        f.write('''\
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class FlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        assert not need_weights
        if HAS_FLASH_ATTN and qkv.dtype in [torch.float16, torch.bfloat16] and qkv.is_cuda:
            return self._flash_forward(qkv, key_padding_mask, causal, cu_seqlens, max_s)
        return self._sdpa_forward(qkv, key_padding_mask, causal)

    def _flash_forward(self, qkv, key_padding_mask, causal, cu_seqlens, max_s):
        batch_size = qkv.shape[0]
        seqlen = qkv.shape[1]
        if cu_seqlens is None:
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen,
                                          dtype=torch.int32, device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal)
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal)
                output = rearrange(
                    pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                              indices, batch_size, seqlen),
                    'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal)
        return output, None

    def _sdpa_forward(self, qkv, key_padding_mask, causal):
        """Fallback using torch.nn.functional.scaled_dot_product_attention."""
        B, S, three, H, D = qkv.shape
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = ~attn_mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=causal if attn_mask is None else False,
            scale=self.softmax_scale,
        )
        out = out.transpose(1, 2).contiguous()
        return out, None
''')

    # 3) Patch internvideo2.py — wrap flash_attn imports in try/except
    iv2_main = os.path.join(iv2_pkg, "internvideo2.py")
    with open(iv2_main, "r") as f:
        src = f.read()
    src = src.replace(
        "from .flash_attention_class import FlashAttention",
        "try:\n    from .flash_attention_class import FlashAttention\n"
        "except ImportError:\n    FlashAttention = None"
    )
    with open(iv2_main, "w") as f:
        f.write(src)

    # 4) Patch models/__init__.py so it doesn't import internvideo2_clip/LLaMA/6B
    models_init = os.path.join(mm_dir, "models", "__init__.py")
    with open(models_init, "w") as f:
        f.write("# patched: only import backbones on demand\n")

    print("Patched InternVideo2 for optional flash_attn.")


def _download_weights():
    """Download checkpoint .pt/.pth from HuggingFace Hub. Returns (stage2_path, clip_path).

    Both repos are gated — you must:
      1. Visit the HF model pages and accept the license terms.
      2. Set HF_TOKEN env var (or run `huggingface-cli login`).
    """
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")

    try:
        stage2_path = hf_hub_download(
            repo_id=INTERNVIDEO2_HF_REPO_STAGE2,
            filename="InternVideo2-stage2_1b-224p-f4.pt",
            token=token,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot download {INTERNVIDEO2_HF_REPO_STAGE2}. "
            f"Visit https://huggingface.co/{INTERNVIDEO2_HF_REPO_STAGE2} to accept terms, "
            f"then set HF_TOKEN env var. Error: {e}"
        ) from e

    try:
        clip_path = hf_hub_download(
            repo_id=INTERNVIDEO2_HF_REPO_CLIP,
            filename="1B_clip.pth",
            token=token,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot download {INTERNVIDEO2_HF_REPO_CLIP}. "
            f"Visit https://huggingface.co/{INTERNVIDEO2_HF_REPO_CLIP} to accept terms, "
            f"then set HF_TOKEN env var. Error: {e}"
        ) from e

    return stage2_path, clip_path



class _InternVideo2VisionOnly(torch.nn.Module):
    """Minimal wrapper: vision encoder + projection for feature extraction only.
    No text encoder / BERT needed."""

    def __init__(self, vision_encoder, vision_proj, embed_dim, num_frames):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.vision_proj = vision_proj
        self.embed_dim = embed_dim
        self.num_frames = num_frames

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    @torch.no_grad()
    def get_vid_feat(self, frames):
        """frames: [B, T, C, H, W] -> pooled features [B, embed_dim]."""
        T = frames.shape[1]
        use_image = T == 1
        x = frames.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,C,T,H,W]
        _vision_embeds, pooled = self.vision_encoder(x, None, use_image)[:2]
        vfeat = self.vision_proj(pooled)
        vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
        return vfeat


def load_internvideo2(device="cuda"):
    """Load InternVideo2-Stage2_1B vision encoder + CLIP projection weights."""
    mm_dir = _ensure_internvideo2_repo()

    stage2_path, clip_path = _download_weights()
    print(f"Stage2 weights: {stage2_path}")
    print(f"CLIP add-on weights: {clip_path}")

    from models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
    from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new

    class _Cfg:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, _Cfg(v) if isinstance(v, dict) else v)
        def get(self, key, default=None):
            return getattr(self, key, default)

    try:
        import flash_attn  # noqa: F401
        use_half = True
    except ImportError:
        use_half = False

    model_cfg = _Cfg({
        "vision_encoder": {
            "name": "pretrain_internvideo2_1b_patch14_224",
            "img_size": INTERNVIDEO2_IMG_SIZE,
            "num_frames": INTERNVIDEO2_NUM_FRAMES,
            "tubelet_size": 1,
            "patch_size": 14,
            "d_model": 1408,
            "clip_embed_dim": INTERNVIDEO2_FEAT_DIM,
            "clip_teacher_embed_dim": 3200,
            "clip_teacher_final_dim": 768,
            "clip_norm_type": "l2",
            "clip_return_layer": 6,
            "clip_student_return_interval": 1,
            "pretrained": "",
            "use_checkpoint": True,
            "checkpoint_num": 40,
            "use_flash_attn": use_half,
            "use_fused_rmsnorm": use_half,
            "use_fused_mlp": use_half,
            "clip_teacher": None,
            "clip_input_resolution": INTERNVIDEO2_IMG_SIZE,
            "clip_teacher_return_interval": 1,
            "video_mask_type": "random",
            "video_mask_ratio": 0.8,
            "image_mask_type": "random",
            "image_mask_ratio": 0.5,
            "sep_image_video_pos_embed": True,
            "keep_temporal": False,
            "only_mask": True,
        },
        "embed_dim": 512,
    })

    print("Building InternVideo2-1B vision encoder...")
    vision_encoder = pretrain_internvideo2_1b_patch14_224(model_cfg)
    vision_proj = torch.nn.Linear(INTERNVIDEO2_FEAT_DIM, model_cfg.embed_dim)

    # Load Stage2 weights
    print(f"Loading Stage2 base weights from {stage2_path}")
    ckpt = torch.load(stage2_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("module", ckpt))

    interpolate_pos_embed_internvideo2_new(
        state_dict, vision_encoder, orig_t_size=4
    )

    ve_prefix = "vision_encoder."
    ve_sd = {k[len(ve_prefix):]: v for k, v in state_dict.items() if k.startswith(ve_prefix)}
    vp_sd = {k.replace("vision_proj.", ""): v for k, v in state_dict.items() if k.startswith("vision_proj.")}

    msg = vision_encoder.load_state_dict(ve_sd, strict=False)
    print(f"  vision_encoder: {len(ve_sd)} keys loaded, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    if vp_sd:
        vision_proj.load_state_dict(vp_sd, strict=False)
        print(f"  vision_proj: {len(vp_sd)} keys loaded")

    # Load CLIP add-on weights on top
    print(f"Loading CLIP add-on weights from {clip_path}")
    clip_ckpt = torch.load(clip_path, map_location="cpu", weights_only=False)
    clip_sd = clip_ckpt.get("model", clip_ckpt.get("module", clip_ckpt))

    ve_clip = {k[len(ve_prefix):]: v for k, v in clip_sd.items() if k.startswith(ve_prefix)}
    vp_clip = {k.replace("vision_proj.", ""): v for k, v in clip_sd.items() if k.startswith("vision_proj.")}
    if ve_clip:
        msg = vision_encoder.load_state_dict(ve_clip, strict=False)
        print(f"  CLIP vision_encoder: {len(ve_clip)} keys, missing={len(msg.missing_keys)}")
    if vp_clip:
        vision_proj.load_state_dict(vp_clip, strict=False)
        print(f"  CLIP vision_proj: {len(vp_clip)} keys")

    model = _InternVideo2VisionOnly(
        vision_encoder, vision_proj, model_cfg.embed_dim, INTERNVIDEO2_NUM_FRAMES
    )
    model = model.to(device).eval().to(torch.float32)
    print(f"InternVideo2-1B vision-only loaded on {device}")
    return model


# --------------- frame loading (shared with auto_extract.py) ---------------

def load_frames_for_clip(frame_path_info, stride):
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
    """Uniformly subsample T frames to num_frames. Returns (tensor [num_frames,C,H,W], orig_T)."""
    T = frames_t.shape[0]
    if T <= num_frames:
        if T < num_frames:
            pad = torch.zeros(num_frames - T, *frames_t.shape[1:], dtype=frames_t.dtype)
            frames_t = torch.cat([frames_t, pad], dim=0)
        return frames_t, T
    indices = torch.linspace(0, T - 1, num_frames).long()
    return frames_t[indices], T


def _load_batch(indices, frame_paths, stride, num_frames, num_workers=4):
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

    subsampled = []
    for _, frames in raw:
        sub, _ = _subsample_frames(frames, num_frames)
        subsampled.append(sub)

    batch = torch.stack(subsampled, dim=0)  # [B, T, C, H, W]
    return batch, valid_indices, []


# --------------- GPU forward ---------------

@torch.no_grad()
def _extract_features(model, batch_uint8, device, mean_t, std_t, use_amp):
    """Run model.get_vid_feat on [B,T,C,H,W] uint8 frames. Returns [B, D] numpy."""
    batch_f = batch_uint8.float().div_(255.0)
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
        batch_tensor, valid_indices, _ = prefetch_future.result()

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
    import dotenv
    dotenv.load_dotenv()

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