#!/usr/bin/env python3
"""
Extract RegNet features for BAA dataset and save to disk.
Features are saved per-clip for later use with dataset load_mode.
"""

import argparse
import os
import sys
import pickle

import torch
import torch.nn as nn
import timm
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, FrameReader
from model.T_Deed_Modules.shift import make_temporal_shift


# Constants matching dataset
FPS_SN = 25
OVERLAP_SNBA = 0.9
STRIDE_SNBA = 4


def create_feature_extractor(feature_arch, clip_len, obs_perc_list=None, device='cuda'):
    """Create RegNet feature extractor matching FUTR backbone."""
    arch_map = {
        'rny002': 'regnety_002',
        'rny004': 'regnety_004',
        'rny006': 'regnety_006',
        'rny008': 'regnety_008',
    }
    if obs_perc_list is None:
        obs_perc_list = [0.5]
    arch_base = feature_arch.rsplit('_', 1)[0]
    if arch_base not in arch_map:
        raise ValueError(f"Unsupported feature_arch: {feature_arch}. Use rny002, rny004, rny006, or rny008 with _gsf or _gsm suffix.")
    
    model = timm.create_model(arch_map[arch_base], pretrained=True)
    feat_dim = model.head.fc.in_features
    model.head.fc = nn.Identity()
    
    max_obs_len = int(clip_len * max(obs_perc_list))
    if feature_arch.endswith('_gsm'):
        make_temporal_shift(model, max_obs_len, mode='gsm')
    elif feature_arch.endswith('_gsf'):
        make_temporal_shift(model, max_obs_len, mode='gsf')
    else:
        raise ValueError("feature_arch must end with _gsf or _gsm")
    
    model = model.to(device).eval()
    return model, feat_dim


def extract_features_for_clip(frame_reader, feature_extractor, frames_path, stride, device,
                              normalize_mean=(0.485, 0.456, 0.406),
                              normalize_std=(0.229, 0.224, 0.225)):
    """Load frames for a clip and extract RegNet features."""
    _, _, pad_start, pad_end, _, length = frames_path
    actual_len = length - pad_start - pad_end
    
    frames = frame_reader.load_frames(
        frames_path, 0, actual_len, pad=True, stride=stride
    )
    # frames: [S, C, H, W]
    frames = frames.float() / 255.0
    # Normalize (ImageNet)
    mean = torch.tensor(normalize_mean).view(1, 3, 1, 1).to(frames.device)
    std = torch.tensor(normalize_std).view(1, 3, 1, 1).to(frames.device)
    frames = (frames - mean) / std
    
    with torch.no_grad():
        frames = frames.to(device)
        # Feature extractor expects [N, C, H, W]
        features = feature_extractor(frames)
    
    return features.cpu()


def extract_split(split, store_dir, feature_dir, frame_reader, feature_extractor,
                  clip_len, stride, radi_displacement, device, dataset_path):
    """Extract features for one split (train/valid/test)."""
    label_file = os.path.join(dataset_path, 'soccernetballanticipation', f'{split}.json')
    if not os.path.exists(label_file):
        print(f"Skipping {split}: {label_file} not found")
        return
    
    store_path = os.path.join(store_dir, f'LEN{clip_len}DIS{radi_displacement}SPLIT{split}')
    frame_paths_file = os.path.join(store_path, 'frame_paths.pkl')
    
    if not os.path.exists(frame_paths_file):
        print(f"Building clip index for {split} (run main.py with store_mode='store' first, or use --build-clips)")
        return
    
    with open(frame_paths_file, 'rb') as f:
        frame_paths = pickle.load(f)
    
    feat_store_path = os.path.join(feature_dir, f'LEN{clip_len}DIS{radi_displacement}SPLIT{split}')
    os.makedirs(feat_store_path, exist_ok=True)
    
    print(f"Extracting features for {split}: {len(frame_paths)} clips")
    for idx in tqdm(range(len(frame_paths)), desc=f"{split}"):
        feat_path = os.path.join(feat_store_path, f'{idx:06d}.pt')
        if os.path.exists(feat_path):
            continue
        try:
            features = extract_features_for_clip(
                frame_reader, feature_extractor,
                frame_paths[idx], stride, device
            )
            torch.save(features, feat_path)
        except Exception as e:
            print(f"Error processing clip {idx} in {split}: {e}")
            raise
    
    print(f"Saved features to {feat_store_path}")


def build_clips_for_split(split, frame_dir, store_dir, clip_len, stride, radi_displacement,
                          dataset_path):
    """Build frame_paths.pkl for a split by creating minimal ActionSpotDataset with store_mode='store'."""
    label_file = os.path.join(dataset_path, 'soccernetballanticipation', f'{split}.json')
    if not os.path.exists(label_file):
        print(f"Cannot build clips for {split}: {label_file} not found")
        return False
    
    classes = load_classes(os.path.join(dataset_path, 'soccernetballanticipation', 'class.txt'))
    n_class = len(classes)
    
    dataset = ActionSpotDataset(
        classes, label_file, frame_dir, store_dir, 'store',
        clip_len=clip_len, dataset_len=1000000,  # Large number for store
        label_pad_idx=255, n_class=n_class,
        stride=stride, overlap=OVERLAP_SNBA,
        dataset='soccernetballanticipation',
        obs_perc=[0.5], pred_perc=0.5, n_query=8,
    )
    return True  # dataset is used for side effect (storing clips)


def main():
    parser = argparse.ArgumentParser(description="Extract RegNet features for BAA dataset")
    parser.add_argument("--frame-dir", type=str, required=True,
                        help="Directory where frames are stored (e.g. data/soccernetball/224p)")
    parser.add_argument("--store-dir", type=str, default=None,
                        help="Directory with clip index (frame_paths.pkl). Default: same as frame-dir")
    parser.add_argument("--feature-dir", type=str, default=None,
                        help="Directory to save features. Default: store_dir/RegNet_features")
    parser.add_argument("--dataset-path", type=str, default="data",
                        help="Path to data folder containing soccernetballanticipation/")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="Splits to process")
    parser.add_argument("--clip-len", type=int, default=64, help="Clip length (frames)")
    parser.add_argument("--stride", type=int, default=STRIDE_SNBA, help="Frame stride")
    parser.add_argument("--radi-displacement", type=int, default=0)
    parser.add_argument("--feature-arch", type=str, default="rny006_gsf",
                        help="RegNet architecture: rny002_gsf, rny004_gsf, rny006_gsf, rny008_gsf")
    parser.add_argument("--build-clips", action="store_true",
                        help="Build clip index (frame_paths.pkl) if not exists")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    store_dir = args.store_dir or args.frame_dir
    feature_dir = args.feature_dir or os.path.join(store_dir, "RegNet_features")
    
    print(f"Frame dir: {args.frame_dir}")
    print(f"Store dir: {store_dir}")
    print(f"Feature dir: {feature_dir}")
    print(f"Feature arch: {args.feature_arch}")
    
    if args.build_clips:
        for split in args.splits:
            build_clips_for_split(
                split, args.frame_dir, store_dir,
                args.clip_len, args.stride, args.radi_displacement,
                args.dataset_path
            )
    
    feature_extractor, _ = create_feature_extractor(
        args.feature_arch, args.clip_len, device=args.device
    )
    frame_reader = FrameReader(args.frame_dir, dataset='soccernetballanticipation')
    
    for split in args.splits:
        extract_split(
            split, store_dir, feature_dir,
            frame_reader, feature_extractor,
            args.clip_len, args.stride, args.radi_displacement,
            args.device, args.dataset_path
        )


if __name__ == "__main__":
    main()
