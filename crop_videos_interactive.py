# -*- coding: utf-8 -*-
"""
Interactive video cropper.

For each input video, opens a frame from the middle, asks you to click
two points (top-left corner and bottom-right corner of the desired crop),
then crops the entire video to that rectangle and writes it to OUTPUT_DIR.

Edit the parameters block below, then run the script.

NOTE: requires a GUI matplotlib backend (Qt or Tk), not 'inline'.
In Spyder, set Tools -> Preferences -> IPython console -> Graphics ->
Backend = "Qt5" (or "Automatic"), then restart the kernel.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================
INPUT_DIR       = r"C:\Users\ys2605\Videos\Desktop"
OUTPUT_DIR      = r"C:\Users\ys2605\Videos\Desktop\cropped"
FRAME_POS_RATIO = 0.5            # which frame to show (0=first, 0.5=middle, 1=last)
OUTPUT_SUFFIX   = "_cropped"     # appended before file extension
OUTPUT_CODEC    = "mp4v"         # cv2 fourcc; 'mp4v' is the safe Windows default
VIDEO_EXTS      = ('.mp4', '.avi', '.mov', '.mkv')

# ============================================================

def check_backend():
    if matplotlib.get_backend().lower() in ('agg', 'module://matplotlib_inline.backend_inline'):
        print("WARNING: matplotlib backend is non-interactive (%s)." % matplotlib.get_backend())
        print("         Clicks will not register. Switch to Qt5Agg / TkAgg first.")
        print("         In Spyder: Tools -> Preferences -> IPython console -> Graphics -> Backend = Qt5")
        return False
    return True


def find_videos(directory):
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in VIDEO_EXTS)


def get_crop_from_clicks(video_path, frame_pos_ratio=0.5):
    cap = cv2.VideoCapture(str(video_path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nframes <= 0:
        cap.release()
        raise RuntimeError(f"No frames found in {video_path}")
    target_frame = int(nframes * frame_pos_ratio)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {target_frame} from {video_path}")

    H, W = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{video_path.name}\n"
                 f"CLICK 1: top-left corner of crop. CLICK 2: bottom-right corner.")
    plt.tight_layout()
    plt.show(block=False)

    print(f"\n  {video_path.name}: click two corners (top-left, then bottom-right)...")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) != 2:
        raise RuntimeError(f"Expected 2 clicks, got {len(pts)}; aborting")

    (cx1, cy1), (cx2, cy2) = pts
    x_min, x_max = sorted([int(round(cx1)), int(round(cx2))])
    y_min, y_max = sorted([int(round(cy1)), int(round(cy2))])
    # clamp to frame bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)
    if x_max - x_min < 2 or y_max - y_min < 2:
        raise RuntimeError(f"Crop region too small: ({x_min},{y_min})-({x_max},{y_max})")
    return x_min, y_min, x_max, y_max


def crop_video(input_path, output_path, x1, y1, x2, y2, codec="mp4v"):
    cap = cv2.VideoCapture(str(input_path))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width  = x2 - x1
    height = y2 - y1
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    print(f"  Writing {nframes} frames @ {fps:.1f} fps to {output_path.name} ...")
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame[y1:y2, x1:x2])
        written += 1
        if written % 500 == 0:
            print(f"    {written} / {nframes}")

    cap.release()
    writer.release()
    print(f"  Done: wrote {written} frames -> {output_path}")


def main():
    check_backend()

    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if not videos:
        print(f"No videos found in {input_dir}")
        return

    print(f"\nFound {len(videos)} video(s) in {input_dir}:")
    for i, v in enumerate(videos):
        print(f"  [{i}] {v.name}")

    raw = input(f"\nHow many videos to process? (1-{len(videos)}, or 'all'): ").strip()
    n = len(videos) if raw.lower() == 'all' else max(1, min(len(videos), int(raw)))

    for v in videos[:n]:
        print(f"\n=== {v.name} ===")
        try:
            x1, y1, x2, y2 = get_crop_from_clicks(v, frame_pos_ratio=FRAME_POS_RATIO)
        except RuntimeError as e:
            print(f"  Skipping ({e})")
            continue
        print(f"  Crop: x={x1}..{x2}, y={y1}..{y2}  (w={x2-x1}, h={y2-y1})")
        out_path = output_dir / f"{v.stem}{OUTPUT_SUFFIX}{v.suffix}"
        crop_video(v, out_path, x1, y1, x2, y2, codec=OUTPUT_CODEC)

    print("\nAll done.")


if __name__ == "__main__":
    main()
