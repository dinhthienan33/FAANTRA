#!/usr/bin/env python3
"""
Convert Labels-ball.json (test) to SoccerNet Ball submission format.
- observation: [] (empty)
- anticipation: label, position, confidence=1, confidence_vect (one-hot)
- path: clip name only (e.g. "clip_1" from "clip_1/224p.mp4")
"""

import json
import os
import sys
from pathlib import Path

# Add project root for util imports
sys.path.insert(0, str(Path(__file__).parent))
from util.dataset import load_classes


def get_action_classes(classes_dict, exclude_goal=True):
    """Get 0-indexed mapping for 11 BAA action classes (exclude BACKGROUND and GOAL)."""
    excluded = {"BACKGROUND", "GOAL"} if exclude_goal else {"BACKGROUND"}
    filtered = {label: idx - 1 for label, idx in classes_dict.items()
                if label not in excluded}
    # Re-index 0..10 for 11 classes
    labels_sorted = sorted(filtered.keys(), key=lambda L: classes_dict[L])
    return {label: i for i, label in enumerate(labels_sorted)}


def make_confidence_vect(label: str, label_to_idx: dict, n_classes: int) -> list:
    """Create one-hot confidence vector for ground truth (confidence=1)."""
    vec = [0.0] * n_classes
    if label in label_to_idx:
        idx = label_to_idx[label]
        vec[idx] = 1.0
    return vec


def convert_video(video: dict, label_to_idx: dict, n_classes: int) -> dict:
    """Convert one video to submission format."""
    annotations = video.get("annotations", {})
    anticipation = annotations.get("anticipation", [])

    path_full = video.get("path", "")
    # "clip_1/224p.mp4" -> "clip_1"
    path_short = path_full.split("/")[0] if "/" in path_full else path_full

    new_anticipation = []
    for ann in anticipation:
        label = ann.get("label", "")
        if label not in label_to_idx:
            continue  # Skip GOAL and other excluded classes
        position = int(ann.get("position", 0))
        conf_vect = make_confidence_vect(label, label_to_idx, n_classes)
        new_anticipation.append({
            "label": label,
            "position": position,
            "confidence": 1.0,
            "confidence_vect": conf_vect,
        })

    return {
        "annotations": {
            "observation": [],
            "anticipation": new_anticipation,
        },
        "path": path_short,
    }


def main():
    base_dir = Path("/workspace/FAANTRA/data/soccernetball")
    input_path = base_dir / "224p" / "test" / "Labels-ball.json"
    output_path = base_dir / "test_submission_format.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Load classes (11 BAA action classes for confidence_vect: exclude BACKGROUND and GOAL)
    class_file = base_dir / "class.txt"
    classes_dict = load_classes(str(class_file))
    label_to_idx = get_action_classes(classes_dict)
    n_classes = 11

    # Load Labels-ball.json
    with open(input_path, "r") as f:
        data = json.load(f)

    videos = data.get("videos", [])
    converted = [convert_video(v, label_to_idx, n_classes) for v in videos]

    output = {"videos": converted}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Converted {len(converted)} videos")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
