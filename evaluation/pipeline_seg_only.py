# Setup 1 — segmentation-only baseline pipeline. Works on a single image or a video file.

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
import seg_utils

import termios
import tty

#? ---
#? USER SETTINGS
#? Set IMAGE_PATH for a single image or VIDEO_PATH for a video — leave both empty to get the selector.
#? ---

IMAGE_PATH = "recordings/images/Generated Image April 08, 2026 - 2_51PM.jpg"  # path to a single image file (.jpg / .png / etc.)
VIDEO_PATH = ""  # path to a video file — leave empty to pick from the recordings folder
RUN_NAME = ""  # name for the output folder — leave empty for auto timestamp
FRAME_EVERY = 0  # video only: frames to skip between samples — 0 uses FRAME_SAMPLE_EVERY from config.py
SAVE_ANNOTATED = True  # set to False to skip saving annotated frames (faster)

#? ---
#? LOGGING
#? Standard logging setup used across the pipeline.
#? ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


#? ---
#? ENTRY POINT
#? Routes to image mode or video mode depending on which path is set.
#? ---

# Entry point — runs image mode if IMAGE_PATH is set, otherwise video mode
def main():
    if IMAGE_PATH:
        print("-------------------------------")
        print("---- RUNNING ON IMAGE MODE ----")
        print("-------------------------------")
        _run_image()
    else:
        print("-------------------------------")
        print("---- RUNNING ON VIDEO MODE ----")
        print("-------------------------------")
        _run_video()


#? ---
#? IMAGE MODE
#? Processes a single image file — same output structure as video mode.
#? ---

# Runs the segmentation pipeline on a single image file
def _run_image():
    img_path = Path(IMAGE_PATH).expanduser().resolve()
    if not img_path.exists():
        logger.error("Image file not found: %s", img_path)
        sys.exit(1)

    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.error("Could not read image: %s", img_path)
        sys.exit(1)

    height, width = frame.shape[:2]
    run_name = RUN_NAME or f"seg_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir, frames_dir, annotated_dir, logs_dir = _make_run_dirs(run_name)

    seg_utils.load_seg_model()
    logger.info("Image: %s  |  %dx%d  |  Output -> %s", img_path.name, width, height, run_dir)

    run_meta = {
        "setup": "seg_only",
        "mode": "image",
        "source": str(img_path),
        "run_name": run_name,
        "seg_model": config.SEG_MODEL_NAME,
        "started_at": datetime.now().isoformat(),
        "resolution": [width, height],
    }
    (run_dir / "config.json").write_text(json.dumps(run_meta, indent=2))

    log_fh = (logs_dir / "seg_results.jsonl").open("w", encoding="utf-8")
    t0 = time.time()

    record = _process_frame(frame, 0, 0.0, frames_dir, annotated_dir, log_fh)

    log_fh.close()
    t_elapsed = time.time() - t0

    summary = {
        "frames_processed": 1,
        "frames_with_obstacles": 1 if record["danger_detected"] else 0,
        "obstacle_rate": 1.0 if record["danger_detected"] else 0.0,
        "total_time_s": round(t_elapsed, 2),
        "avg_inference_s": round(record["inference_time_s"], 3),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    status = "DANGER" if record["danger_detected"] else ("CONTEXT" if record["context_detected"] else "CLEAR")
    logger.info("Done.  Result: %s  |  %.2f s  |  Saved to: %s", status, t_elapsed, run_dir)


#? ---
#? VIDEO MODE
#? Processes a video file frame by frame — same selector and loop as before.
#? ---

# Runs the segmentation pipeline on a video file, with interactive file selector if VIDEO_PATH is empty
def _run_video():
    global VIDEO_PATH
    if not VIDEO_PATH:
        VIDEO_PATH = _pick_video_file()

    video_path = Path(VIDEO_PATH).expanduser().resolve()
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    run_name = RUN_NAME or f"seg_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir, frames_dir, annotated_dir, logs_dir = _make_run_dirs(run_name)

    every = FRAME_EVERY or config.FRAME_SAMPLE_EVERY

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Video: %s  |  %d frames  |  %.1f fps  |  %dx%d", video_path.name, total_frames, fps, width, height)
    logger.info("Sampling every %d frame(s). Output -> %s", every, run_dir)

    seg_utils.load_seg_model()

    run_meta = {
        "setup": "seg_only",
        "mode": "video",
        "source": str(video_path),
        "run_name": run_name,
        "seg_model": config.SEG_MODEL_NAME,
        "frame_sample_every": every,
        "operating_zone_bottom_exclude": config.OPERATING_ZONE_BOTTOM_EXCLUDE,
        "obstacle_classes": config.SEG_OBSTACLE_CLASSES,
        "obstacle_min_pixel_fraction": config.SEG_OBSTACLE_MIN_PIXEL_FRACTION,
        "started_at": datetime.now().isoformat(),
        "video_fps": fps,
        "video_resolution": [width, height],
    }
    (run_dir / "config.json").write_text(json.dumps(run_meta, indent=2))

    log_fh = (logs_dir / "seg_results.jsonl").open("w", encoding="utf-8")

    frame_idx = 0
    processed = 0
    obstacles_detected = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every != 0:
                frame_idx += 1
                continue

            record = _process_frame(frame, frame_idx, frame_idx / fps, frames_dir, annotated_dir, log_fh)

            processed += 1
            if record["danger_detected"]:
                obstacles_detected += 1

            if processed % 20 == 0 or record["danger_detected"]:
                pct = frame_idx / max(total_frames, 1) * 100
                status = "  OBSTACLE" if record["danger_detected"] else ""
                logger.info("[%5.1f%%] frame %06d  |  %.2f s/frame%s",
                            pct, frame_idx, record["inference_time_s"], status)

            frame_idx += 1

    finally:
        cap.release()
        log_fh.close()

    total_time = time.time() - t_start
    summary = {
        "frames_processed": processed,
        "frames_with_obstacles": obstacles_detected,
        "obstacle_rate": round(obstacles_detected / max(processed, 1), 4),
        "total_time_s": round(total_time, 2),
        "avg_inference_s": round(total_time / max(processed, 1), 3),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Done.  %d/%d frames had obstacles (%.1f%%)",
                obstacles_detected, processed, summary["obstacle_rate"] * 100)
    logger.info("Results saved to: %s", run_dir)


#? ---
#? FRAME PROCESSING
#? Shared logic — runs segmentation on one frame, saves outputs, writes log entry.
#? ---

# Runs segmentation on one frame, saves JPEG outputs, appends a record to the log, returns the record
def _process_frame(frame: np.ndarray, frame_idx: int, video_ts: float, frames_dir: Path, annotated_dir: Path, log_fh):
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    frame_stem = f"frame_{frame_idx:06d}"

    t0 = time.time()
    mask, classes_present, annotated = seg_utils.run_segmentation(frame)
    danger_detected, context_detected, detections = seg_utils.get_obstacle_info(mask)
    inference_time = time.time() - t0

    # seg_only only reacts to the central danger zone — side context zones are ignored
    cv2.imwrite(str(frames_dir / f"{frame_stem}.jpg"), frame, jpeg_params)

    if SAVE_ANNOTATED:
        ann = seg_utils.draw_obstacle_labels(annotated, detections, danger_detected, context_detected)
        if video_ts > 0:
            _draw_timestamp(ann, video_ts, frame_idx)
        cv2.imwrite(str(annotated_dir / f"{frame_stem}.jpg"), ann, jpeg_params)

    record = {
        "frame_index": frame_idx,
        "video_timestamp_s": round(video_ts, 3),
        "danger_detected": danger_detected,
        "context_detected": context_detected,
        "detections": detections,
        "classes_present": sorted(classes_present),
        "inference_time_s": round(inference_time, 3),
    }
    log_fh.write(json.dumps(record) + "\n")
    log_fh.flush()
    return record


#? ---
#? HELPERS
#? Utilities for output dirs, frame annotation, and the interactive video selector.
#? ---

# Creates and returns the four run output directories
def _make_run_dirs(run_name: str):
    run_dir = Path(config.RUNS_DIR) / run_name
    frames_dir = run_dir / "frames"
    annotated_dir = run_dir / "annotated"
    logs_dir = run_dir / "logs"
    for d in (frames_dir, annotated_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return run_dir, frames_dir, annotated_dir, logs_dir

# Draws the video timestamp and frame index in the bottom-left corner of the frame
def _draw_timestamp(frame: np.ndarray, video_ts: float, frame_idx: int):
    minutes = int(video_ts // 60)
    seconds = video_ts % 60
    text = f"t={minutes:02d}:{seconds:05.2f}  frame={frame_idx}"
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

# Shows an interactive terminal selector and returns the chosen video path as a string
def _pick_video_file():
    d_path = Path(__file__).parent / "recordings" / "first_tests_inside"
    if not d_path.exists():
        logger.error("Folder not found: %s", d_path)
        sys.exit(1)

    files = sorted([f for f in d_path.iterdir()
                    if f.is_file() and f.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov')])
    if not files:
        logger.error("No video files found in %s", d_path)
        sys.exit(1)

    options = []
    for f in files:
        dt = datetime.fromtimestamp(f.stat().st_mtime)
        options.append(f"{f.name} ({dt.strftime('%A %-d %B - %Hh%M')})")

    print("Select a video file to test:")
    selected_idx = -1

    if sys.stdout.isatty() and sys.stdin.isatty():
        print("Use UP/DOWN arrows to select, ENTER to confirm:")
        for _ in options:
            print()
        sys.stdout.write(f"\033[{len(options)}A")
        sys.stdout.flush()

        current_idx = 0
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while True:
                sys.stdout.write("\r")
                for i, opt in enumerate(options):
                    prefix = " > " if i == current_idx else "   "
                    sys.stdout.write(f"\033[K{prefix}{opt}\n")
                sys.stdout.write(f"\033[{len(options)}A")
                sys.stdout.flush()

                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    ch2 = sys.stdin.read(2)
                    if ch2 == '[A':
                        current_idx = max(0, current_idx - 1)
                    elif ch2 == '[B':
                        current_idx = min(len(options) - 1, current_idx + 1)
                elif ch in ('\n', '\r'):
                    sys.stdout.write(f"\033[{len(options)}B")
                    sys.stdout.flush()
                    selected_idx = current_idx
                    break
                elif ch == '\x03':
                    sys.stdout.write(f"\033[{len(options)}B")
                    sys.stdout.flush()
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    sys.exit(1)
        except Exception:
            selected_idx = -1
        finally:
            if 'old_settings' in locals():
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if selected_idx == -1:
        for i, opt in enumerate(options):
            print(f" {i+1}) {opt}")
        while True:
            try:
                choice = int(input("Enter number: "))
                if 1 <= choice <= len(options):
                    selected_idx = choice - 1
                    break
                print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")
            except EOFError:
                sys.exit(1)

    print(f"\nSelected: {files[selected_idx].name}\n")
    return str(files[selected_idx])


if __name__ == "__main__":
    main()
