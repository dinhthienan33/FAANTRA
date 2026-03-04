#!/usr/bin/env python3
"""
Visualize Labels-ball.json annotations on videos.
Usage:
  python visualize.py path/to/Labels-ball.json path/to/video_folder [--output out.mp4] [--video-idx 0] [--fps 25]
"""

import argparse
import json
import os

import cv2


FPS = 25


def ms_to_frame(position_ms: int, fps: float = FPS) -> int:
    return int(int(position_ms) / 1000 * fps)


def load_labels(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def visualize_video(
    labels_path: str,
    video_folder: str,
    output_path: str | None = None,
    video_idx: int = 0,
    fps: float = FPS,
):
    data = load_labels(labels_path)
    videos = data.get("videos", [])
    if video_idx >= len(videos):
        raise ValueError(f"video_idx {video_idx} >= {len(videos)} videos")
    video_info = videos[video_idx]
    rel_path = video_info.get("path", "")
    video_path = os.path.join(video_folder, rel_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    annotations = video_info.get("annotations", {})
    obs = annotations.get("observation", [])
    ant = annotations.get("anticipation", [])

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    obs_by_frame = {}
    for a in obs:
        f = ms_to_frame(a["position"], fps)
        obs_by_frame.setdefault(f, []).append(a)
    ant_by_frame = {}
    for a in ant:
        f = ms_to_frame(a["position"], fps)
        ant_by_frame.setdefault(f, []).append(a)

    print(f"Visualizing: {video_path}")
    print(f"Observation events: {len(obs)}, Anticipation events: {len(ant)}")
    print("Press 'q' to quit, 's' to save frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame.copy()
        ts_ms = frame_idx * 1000 / fps
        cv2.putText(
            frame, f"Frame {frame_idx} | {ts_ms:.0f}ms",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Frame {frame_idx} | {ts_ms:.0f}ms",
            (22, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA
        )

        if frame_idx in obs_by_frame:
            for i, ann in enumerate(obs_by_frame[frame_idx]):
                label = ann.get("label", "")
                team = ann.get("team", "")
                text = f"OBS: {label} ({team})"
                y = 70 + i * 35
                cv2.rectangle(frame, (45, y - 28), (width - 20, y + 5), (0, 165, 255), -1)
                cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        if frame_idx in ant_by_frame:
            for i, ann in enumerate(ant_by_frame[frame_idx]):
                label = ann.get("label", "")
                team = ann.get("team", "")
                text = f"ANT: {label} ({team})"
                y = 70 + (len(obs_by_frame.get(frame_idx, [])) + i) * 35
                cv2.rectangle(frame, (45, y - 28), (width - 20, y + 5), (0, 255, 0), -1)
                cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        if out:
            out.write(frame)

        cv2.imshow("Labels-ball", frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_path = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"Saved {save_path}")
        frame_idx += 1

    cap.release()
    if out:
        out.release()
        print(f"Saved video to {output_path}")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize Labels-ball.json on videos")
    parser.add_argument("labels", type=str, help="Path to Labels-ball.json")
    parser.add_argument("video_folder", type=str, help="Path to folder containing videos (e.g. train/ or valid/)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output video path")
    parser.add_argument("--video-idx", "-i", type=int, default=0, help="Video index in JSON (default: 0)")
    parser.add_argument("--fps", type=float, default=FPS, help=f"FPS (default: {FPS})")
    args = parser.parse_args()
    visualize_video(
        args.labels,
        args.video_folder,
        args.output,
        args.video_idx,
        args.fps,
    )


if __name__ == "__main__":
    main()
