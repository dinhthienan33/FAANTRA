#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Labels-ball.json in SoccerNet Ball dataset.
Analyzes train, valid, test, and challenge splits.
"""

import json
import os
from collections import Counter
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

BASE_DIR = Path("/workspace/FAANTRA/data/soccernetball/224p")
SPLITS = ["train", "valid", "test", "challenge"]
FPS = 25
MS_PER_SEC = 1000


def load_labels(split: str) -> dict:
    path = BASE_DIR / split / "Labels-ball.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def flatten_annotations(data: dict, split: str):
    rows = []
    for vid in data.get("videos", []):
        path = vid.get("path", "unknown")
        for ann_type in ["observation", "anticipation"]:
            for ann in vid.get("annotations", {}).get(ann_type, []):
                rows.append({
                    "split": split,
                    "video_path": path,
                    "type": ann_type,
                    "label": ann.get("label", "unknown"),
                    "position_ms": int(ann.get("position", 0)),
                    "position_sec": int(ann.get("position", 0)) / MS_PER_SEC,
                    "team": ann.get("team", "unknown"),
                    "visibility": ann.get("visibility", "unknown"),
                })
    return rows


def compute_stats(data: dict, split: str) -> dict:
    videos = data.get("videos", [])
    n_videos = len(videos)

    n_obs = 0
    n_ant = 0
    videos_with_anticipation = 0
    label_counts_obs = Counter()
    label_counts_ant = Counter()
    team_counts = Counter()
    obs_positions = []
    ant_positions = []
    obs_per_video = []
    ant_per_video = []

    for vid in videos:
        obs = vid.get("annotations", {}).get("observation", [])
        ant = vid.get("annotations", {}).get("anticipation", [])

        n_obs += len(obs)
        n_ant += len(ant)
        if len(ant) > 0:
            videos_with_anticipation += 1

        obs_per_video.append(len(obs))
        ant_per_video.append(len(ant))

        for a in obs:
            label_counts_obs[a.get("label", "unknown")] += 1
            team_counts[a.get("team", "unknown")] += 1
            obs_positions.append(int(a.get("position", 0)))

        for a in ant:
            label_counts_ant[a.get("label", "unknown")] += 1
            ant_positions.append(int(a.get("position", 0)))

    return {
        "split": split,
        "n_videos": n_videos,
        "n_observation": n_obs,
        "n_anticipation": n_ant,
        "videos_with_anticipation": videos_with_anticipation,
        "label_counts_obs": label_counts_obs,
        "label_counts_ant": label_counts_ant,
        "team_counts": team_counts,
        "obs_positions": obs_positions,
        "ant_positions": ant_positions,
        "obs_per_video": obs_per_video,
        "ant_per_video": ant_per_video,
    }


def write_summary(all_stats: dict, log_file: Path):
    lines = []
    lines.append("=" * 80)
    lines.append("LABELS-BALL.json EDA - SUMMARY")
    lines.append("=" * 80)

    total_videos = sum(s["n_videos"] for s in all_stats.values())
    total_obs = sum(s["n_observation"] for s in all_stats.values())
    total_ant = sum(s["n_anticipation"] for s in all_stats.values())

    lines.append(f"\nTotal videos: {total_videos}")
    lines.append(f"Total observation events: {total_obs}")
    lines.append(f"Total anticipation events: {total_ant}")

    lines.append("\n--- Per-split stats ---")
    for split, stats in all_stats.items():
        pct_ant = 100 * stats["videos_with_anticipation"] / max(1, stats["n_videos"])
        lines.append(f"\n{split.upper()}:")
        lines.append(f"  Videos: {stats['n_videos']}")
        lines.append(f"  Observation events: {stats['n_observation']}")
        lines.append(f"  Anticipation events: {stats['n_anticipation']}")
        lines.append(f"  Videos with anticipation: {stats['videos_with_anticipation']} ({pct_ant:.1f}%)")

        if stats["obs_per_video"]:
            lines.append(
                f"  Obs per video: mean={np.mean(stats['obs_per_video']):.1f}, "
                f"min={min(stats['obs_per_video'])}, max={max(stats['obs_per_video'])}"
            )

        non_zero = [x for x in stats["ant_per_video"] if x > 0]
        if non_zero:
            lines.append(
                f"  Ant per video (when >0): mean={np.mean(non_zero):.1f}, "
                f"min={min(non_zero)}, max={max(non_zero)}"
            )

    lines.append("\n--- Label distribution (observation, all splits) ---")
    all_labels_obs = Counter()
    for stats in all_stats.values():
        all_labels_obs.update(stats["label_counts_obs"])
    for label, count in all_labels_obs.most_common():
        pct = 100 * count / total_obs
        lines.append(f"{label}: {count} ({pct:.1f}%)")

    lines.append("\n--- Label distribution (anticipation, all splits) ---")
    all_labels_ant = Counter()
    for stats in all_stats.values():
        all_labels_ant.update(stats["label_counts_ant"])
    for label, count in all_labels_ant.most_common():
        pct = 100 * count / max(1, total_ant)
        lines.append(f"{label}: {count} ({pct:.1f}%)")

    with open(log_file, "w") as f:
        f.write("\n".join(lines))


def main():
    all_stats = {}
    all_rows = []

    for split in SPLITS:
        data = load_labels(split)
        if data is None:
            continue
        stats = compute_stats(data, split)
        all_stats[split] = stats
        rows = flatten_annotations(data, split)
        all_rows.extend(rows)

    if not all_stats:
        return

    output_dir = BASE_DIR / "eda_output"
    os.makedirs(output_dir, exist_ok=True)

    log_file = output_dir / "eda_summary.txt"
    write_summary(all_stats, log_file)

    if HAS_VIZ and all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_dir / "labels_flat.csv", index=False)


if __name__ == "__main__":
    main()