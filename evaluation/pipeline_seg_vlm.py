#? segmentation and VLM pipeline, sends obstacle frames to a VLM running via Ollama

from __future__ import annotations

import base64
import json
import logging
import sys
import termios
import time
import tty
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional

import cv2
import numpy as np
import requests

import config
import seg_utils

#? user settings
#? set IMAGE_PATH for a single image or VIDEO_PATH for a video, leave both empty to require VIDEO_PATH

IMAGE_PATH = ""
#? path to a video file
VIDEO_PATH = ""
#? name for the output folder, leave empty for auto timestamp
RUN_NAME = ""
#? video only: frames to skip between samples, 0 uses FRAME_SAMPLE_EVERY from config.py
FRAME_EVERY = 0
#? set to False to skip saving annotated frames (faster)
SAVE_ANNOTATED = True
#? Ollama endpoint, leave empty to use OLLAMA_BASE_URL from config.py
OLLAMA_URL = ""
#? experimental: pass recent detections as context to the next VLM call
CONTEXT_MEMORY = False
#? send two frames separated by DUAL_FRAME_DELTA_S to the VLM instead of one, overridden at runtime
DUAL_FRAME_MODE = False
#? seconds between the earlier and the current frame sent to the VLM in dual frame mode
DUAL_FRAME_DELTA_S = 1.5

#? logging
#? standard logging setup used across the pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


#? vlm client
#? builds prompts, calls the Ollama API, and parses the structured VLM response

#? prompt for obstacles in the central danger zone, requires an immediate action decision
_PROMPT_DANGER = """\
You are the vision system of an autonomous robotic lawn mower operating in a garden.
Your role is to analyse each camera frame and decide the safest immediate action for the mower.

A segmentation model has already confirmed an obstacle is directly in the mower's vision.

STOP is reserved for two cases only: something is moving, or a child/small animal is present.
A stationary obstacle will not move into the mower - the correct response is always to turn around it, never to stop.

Rules:
- All living beings (people, animals, pets) are obstacles regardless of their behaviour or apparent intent.
- A child or small animal anywhere in the frame always means ACTION: STOP.
- A person or animal that is moving in any direction means ACTION: STOP - any motion is unpredictable.
- A stationary person, animal, or object must never produce ACTION: STOP - turn away from the side they occupy: if the obstacle is on the right side of the frame use ACTION: TURN_LEFT, if on the left side use ACTION: TURN_RIGHT, if centred prefer TURN_LEFT.
- If you see multiple people or people engaged in an activity (picnic, playing, sunbathing, gardening), use SCENE_TYPE: group_activity - if the group is on the right side use ACTION: TURN_LEFT, if on the left side use ACTION: TURN_RIGHT, if centred prefer TURN_LEFT.
- Unattended objects like garden furniture, toys, or bags use SCENE_TYPE: unattended_object and apply the same turn rule.
- Only use ACTION: CONTINUE if OBSTACLE_TYPE is none and the path is visibly and completely clear.

{context_block}\
Respond in exactly this format - no extra text, no markdown, all tokens lowercase:

DESCRIPTION: <one or two sentences: what is it, where exactly in the frame, what is happening>
OBSTACLE_TYPE: <person | dog | cat | bicycle | car | unknown | none>
SCENE_TYPE: <solo_obstacle | group_activity | unattended_object | none>
MOVEMENT: <stationary | moving | unclear>
IF MOVING: <toward_mower | away_from_mower | left | right | unclear>
ACTION: <STOP | CONTINUE | TURN_LEFT | TURN_RIGHT>
REASONING: <one sentence explaining the action>
CONFIDENCE: <high | medium | low>
"""

#? prompt for obstacles detected only in the left/right context margins
#? focus: is this obstacle likely to enter the driving path? CONTINUE unless there is clear threat
_PROMPT_CONTEXT = """\
You are the vision system of an autonomous robotic lawn mower operating in a residential garden.
Your role is to analyse each camera frame and assess whether an obstacle at the side of the frame is a future threat.
Safety is the absolute priority - when in doubt, always choose STOP over CONTINUE.

A segmentation model detected an obstacle in the {side} peripheral zone - outside the mower's immediate driving path.

Rules:
- All living beings (people, animals, pets) are obstacles regardless of their behaviour or apparent intent.
- If you see multiple people or people engaged in an activity (picnic, playing, sunbathing, gardening) on the {side} side, use SCENE_TYPE: group_activity - the mower must steer away from that side.
- For a group activity on the {side} side: use ACTION: TURN_RIGHT if the activity is on the left, or ACTION: TURN_LEFT if it is on the right, so the mower steers toward the empty side.
- A child or small animal visible anywhere in the frame always means THREAT: likely and ACTION: STOP.
- A person or animal moving toward the mower's path means THREAT: likely and ACTION: STOP.
- A stationary person or animal on the side is not going to enter the path - use THREAT: possible and ACTION: CONTINUE.
- A person facing away and walking away from the mower's path means THREAT: none - CONTINUE is acceptable.
- Unattended objects like garden furniture or toys use SCENE_TYPE: unattended_object - assess trajectory threat normally.
- Default to CONTINUE only if the obstacle shows no signs of moving toward the path.

{context_block}\
Respond in exactly this format - no extra text, no markdown, all tokens lowercase:

DESCRIPTION: <one or two sentences: what is it, where exactly, what is it doing>
OBSTACLE_TYPE: <person | dog | cat | bicycle | car | unknown | none>
SCENE_TYPE: <solo_obstacle | group_activity | unattended_object | none>
MOVEMENT: <stationary | moving | unclear>
IF MOVING: <toward_mower | away_from_mower | left | right | unclear>
ORIENTATION: <facing_mower | facing_away | sideways | unclear>
THREAT: <none | possible | likely>
ACTION: <CONTINUE | STOP | TURN_LEFT | TURN_RIGHT>
REASONING: <one sentence - only recommend STOP or TURN if the obstacle is clearly moving toward the path or is very likely to enter it>
CONFIDENCE: <high | medium | low>
"""

_CONTEXT_PREFIX = """\
Context from recent frames (treat as hints, not certainty, verify independently):
{entries}

"""

#? preamble injected before the standard prompts when two frames are sent to the VLM
_DUAL_FRAME_PREFIX = """\
You are given TWO frames from the same camera, captured approximately {delta:.1f} seconds apart.
IMAGE 1 is the earlier frame. IMAGE 2 is the current frame.
Compare the position and state of any obstacle across both frames to determine whether it is stationary or moving, and in which direction.

"""

#? builds the danger zone VLM prompt, optionally prepending rolling context
def build_prompt_danger(context_memory: Optional[Deque[dict]]):
    context_block = _build_context_block(context_memory)
    return _PROMPT_DANGER.format(context_block=context_block)

#? builds the context zone VLM prompt for the given side ("left" or "right")
def build_prompt_context(side: str, context_memory: Optional[Deque[dict]]):
    context_block = _build_context_block(context_memory)
    return _PROMPT_CONTEXT.format(side=side, context_block=context_block)

#? returns the context block string from memory, or empty string if disabled
def _build_context_block(context_memory: Optional[Deque[dict]]):
    if context_memory and config.VLM_USE_CONTEXT_MEMORY:
        entries = "\n".join(
            f"  - {m['obstacle_type']} ({m['zone']}, action={m['action']}, conf={m['confidence']})"
            for m in list(context_memory)
        )
        return _CONTEXT_PREFIX.format(entries=entries)
    return ""

#? builds the danger zone dual frame prompt with the temporal preamble
def build_prompt_danger_dual(delta: float, context_memory: Optional[Deque[dict]]):
    prefix = _DUAL_FRAME_PREFIX.format(delta=delta)
    context_block = _build_context_block(context_memory)
    return prefix + _PROMPT_DANGER.format(context_block=context_block)

#? builds the context zone dual frame prompt for the given side with the temporal preamble
def build_prompt_context_dual(side: str, delta: float, context_memory: Optional[Deque[dict]]):
    prefix = _DUAL_FRAME_PREFIX.format(delta=delta)
    context_block = _build_context_block(context_memory)
    return prefix + _PROMPT_CONTEXT.format(side=side, context_block=context_block)

#? sends two frames (prior then current) and a prompt to the Ollama VLM, returns (parsed_response_dict, elapsed_seconds)
def call_vlm_dual(frame_current: np.ndarray, frame_prior: np.ndarray, prompt: str, ollama_url: str):
    payload = {
        "model": config.VLM_MODEL,
        "prompt": prompt,
        "images": [frame_to_b64(frame_prior), frame_to_b64(frame_current)],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096}
    }
    t0 = time.time()
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=config.VLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("VLM dual-frame request failed: %s", exc)
        return None, time.time() - t0
    elapsed = time.time() - t0
    raw_text = resp.json().get("response", "")
    parsed = _parse_vlm_response(raw_text)
    parsed["raw_text"] = raw_text
    return parsed, elapsed

#? encodes a BGR frame as a base64 JPEG string for the Ollama API
def frame_to_b64(frame_bgr: np.ndarray):
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("utf-8")

#? sends a frame and a pre-built prompt to the Ollama VLM, returns (parsed_response_dict, elapsed_seconds)
def call_vlm(frame_bgr: np.ndarray, prompt: str, ollama_url: str):
    image_b64 = frame_to_b64(frame_bgr)
    payload = {
        "model": config.VLM_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096}
    }
    t0 = time.time()
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=config.VLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("VLM request failed: %s", exc)
        return None, time.time() - t0
    elapsed = time.time() - t0
    raw_text = resp.json().get("response", "")
    parsed = _parse_vlm_response(raw_text)
    parsed["raw_text"] = raw_text
    return parsed, elapsed

#? parses the structured key:value fields out of a free-text VLM response
def _parse_vlm_response(text: str):
    result = {
        "description": "",
        "obstacle_type": "unknown",
        "scene_type": "",
        "movement": "unclear",
        "if_moving": "",
        "orientation": "",
        "threat": "",
        "action": "STOP",
        "reasoning": "",
        "confidence": "low",
    }
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            result["description"] = line[len("DESCRIPTION:"):].strip()
        elif line.startswith("OBSTACLE_TYPE:"):
            result["obstacle_type"] = line[len("OBSTACLE_TYPE:"):].strip()
        elif line.startswith("SCENE_TYPE:"):
            result["scene_type"] = line[len("SCENE_TYPE:"):].strip().lower()
        elif line.startswith("MOVEMENT:"):
            result["movement"] = line[len("MOVEMENT:"):].strip()
        elif line.startswith("IF MOVING:"):
            result["if_moving"] = line[len("IF MOVING:"):].strip()
        elif line.startswith("ORIENTATION:"):
            result["orientation"] = line[len("ORIENTATION:"):].strip()
        elif line.startswith("THREAT:"):
            result["threat"] = line[len("THREAT:"):].strip().lower()
        elif line.startswith("ACTION:"):
            val = line[len("ACTION:"):].strip().upper()
            if val in {"STOP", "CONTINUE", "TURN_LEFT", "TURN_RIGHT"}:
                result["action"] = val
        elif line.startswith("REASONING:"):
            result["reasoning"] = line[len("REASONING:"):].strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line[len("CONFIDENCE:"):].strip().lower()
    return result


#? model selection
#? prompts the user at runtime to choose between the standard and high quality VLM

#? prompts the user to select a VLM and overrides config.VLM_MODEL accordingly
def _select_model():
    print("")
    print("Select VLM mode:")
    print(f"  [1] Standard  - {config.VLM_MODEL}")
    print(f"  [2] Strong    - {config.VLM_MODEL_STRONG}")
    print("")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            print(f"Using standard model: {config.VLM_MODEL}")
            break
        if choice == "2":
            config.VLM_MODEL = config.VLM_MODEL_STRONG
            print(f"Using strong model: {config.VLM_MODEL}")
            break
        print("Invalid input, please enter 1 or 2.")
    print("")

#? prompts the user to choose between single frame and dual frame VLM mode and sets the delta if dual
def _select_pipeline_mode():
    global DUAL_FRAME_MODE, DUAL_FRAME_DELTA_S
    print("Select pipeline mode:")
    print("  [1] Standard    - one frame per VLM call")
    print("  [2] Dual-frame  - two frames for movement detection")
    print("")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            DUAL_FRAME_MODE = False
            print("Mode: standard (single frame)")
            break
        if choice == "2":
            DUAL_FRAME_MODE = True
            while True:
                raw = input("Seconds between the two frames (e.g. 1.5): ").strip()
                try:
                    val = float(raw)
                    if val > 0:
                        DUAL_FRAME_DELTA_S = val
                        break
                    print("Value must be greater than 0.")
                except ValueError:
                    print("Please enter a valid number.")
            print(f"Mode: dual-frame  |  delta: {DUAL_FRAME_DELTA_S}s")
            break
        print("Invalid input, please enter 1 or 2.")
    print("")

#? prompts the user to enter a run name, falls back to a timestamp if left empty
def _prompt_run_name():
    name = input("Enter a name for this run (leave blank for auto timestamp): ").strip()
    if not name:
        name = f"seg_vlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Using auto name: {name}")
    print("")
    return name


#? main
#? routes to image mode or video mode depending on which path is set

#? entry point, runs image mode if IMAGE_PATH is set, otherwise asks the user to pick a mode
def main():
    _select_model()
    _select_pipeline_mode()
    if IMAGE_PATH:
        print("-------------------------------")
        print("---- RUNNING ON IMAGE MODE ----")
        print("-------------------------------")
        _run_image()
        return
    print("-------------------------------")
    print("---- RUNNING ON VIDEO MODE ----")
    print("-------------------------------")
    mode, files, prefix = _select_run_mode()
    if mode == "all":
        for f in files:
            run_name = f"{prefix}_{f.stem}" if prefix else f.stem
            print(f"\nProcessing: {f.name}")
            _run_video(video_path_override=str(f), run_name_override=run_name)
    else:
        _run_video()


#? image mode
#? processes a single image, same output structure as video mode

#? runs the seg and VLM pipeline on a single image file
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
    ollama_url = OLLAMA_URL or config.OLLAMA_BASE_URL
    use_context = CONTEXT_MEMORY or config.VLM_USE_CONTEXT_MEMORY

    _check_ollama(ollama_url)

    run_name = RUN_NAME or _prompt_run_name()
    run_dir, frames_dir, annotated_dir, logs_dir = _make_run_dirs(run_name)

    seg_utils.load_seg_model()
    logger.info("Image: %s  |  %dx%d  |  VLM: %s  |  Output -> %s", img_path.name, width, height, config.VLM_MODEL, run_dir)

    run_meta = {
        "setup": "seg_vlm",
        "mode": "image",
        "source": str(img_path),
        "run_name": run_name,
        "seg_model": config.SEG_MODEL_NAME,
        "vlm_model": config.VLM_MODEL,
        "ollama_url": ollama_url,
        "started_at": datetime.now().isoformat(),
        "resolution": [width, height],
    }
    (run_dir / "config.json").write_text(json.dumps(run_meta, indent=2))

    seg_log = (logs_dir / "seg_results.jsonl").open("w", encoding="utf-8")
    vlm_log = (logs_dir / "vlm_results.jsonl").open("w", encoding="utf-8")
    context_memory: Deque[dict] = deque(maxlen=config.VLM_CONTEXT_MEMORY_MAX_FRAMES)

    t0 = time.time()
    seg_rec, vlm_rec = _process_frame(frame, 0, 0.0, frames_dir, annotated_dir, seg_log, vlm_log, context_memory, ollama_url, use_context)
    seg_log.close()
    vlm_log.close()
    t_elapsed = time.time() - t0

    summary = {
        "frames_processed": 1,
        "frames_danger_zone": 1 if seg_rec["danger_detected"] else 0,
        "frames_context_zone": 1 if seg_rec["context_detected"] else 0,
        "vlm_calls": 1 if vlm_rec else 0,
        "vlm_skipped_blurry": 1 if seg_rec["blurry"] and (seg_rec["danger_detected"] or seg_rec["context_detected"]) else 0,
        "total_time_s": round(t_elapsed, 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    status = "DANGER" if seg_rec["danger_detected"] else ("CONTEXT" if seg_rec["context_detected"] else "CLEAR")
    action = vlm_rec.get("action", "-") if vlm_rec else "-"
    logger.info("Done.  Status: %s  |  VLM action: %s  |  %.2f s  |  Saved to: %s", status, action, t_elapsed, run_dir)


#? video mode
#? processes a video file frame by frame using the segmentation and VLM pipeline

#? runs the seg and VLM pipeline on every sampled frame of a video, uses selector and name prompt if no overrides given
def _run_video(video_path_override=None, run_name_override=None):
    global VIDEO_PATH
    if video_path_override:
        video_path = Path(video_path_override).expanduser().resolve()
    else:
        if not VIDEO_PATH:
            VIDEO_PATH = _pick_video_file()
        video_path = Path(VIDEO_PATH).expanduser().resolve()

    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    ollama_url = OLLAMA_URL or config.OLLAMA_BASE_URL
    use_context = CONTEXT_MEMORY or config.VLM_USE_CONTEXT_MEMORY
    every = FRAME_EVERY or config.FRAME_SAMPLE_EVERY

    _check_ollama(ollama_url)

    run_name = run_name_override or RUN_NAME or _prompt_run_name()
    run_dir, frames_dir, annotated_dir, logs_dir = _make_run_dirs(run_name, dual_mode=DUAL_FRAME_MODE)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Video: %s  |  %d frames  |  %.1f fps  |  %dx%d", video_path.name, total_frames, fps, width, height)
    logger.info("VLM endpoint: %s  |  model: %s  |  Context memory: %s", ollama_url, config.VLM_MODEL, use_context)
    logger.info("Output -> %s", run_dir)

    seg_utils.load_seg_model()

    run_meta = {
        "setup": "seg_vlm_dual" if DUAL_FRAME_MODE else "seg_vlm",
        "mode": "video",
        "source": str(video_path),
        "run_name": run_name,
        "seg_model": config.SEG_MODEL_NAME,
        "vlm_model": config.VLM_MODEL,
        "ollama_url": ollama_url,
        "frame_sample_every": every,
        "operating_zone_bottom_exclude": config.OPERATING_ZONE_BOTTOM_EXCLUDE,
        "obstacle_classes": config.SEG_OBSTACLE_CLASSES,
        "obstacle_min_pixel_fraction": config.SEG_OBSTACLE_MIN_PIXEL_FRACTION,
        "context_memory_enabled": use_context,
        "context_memory_max_frames": config.VLM_CONTEXT_MEMORY_MAX_FRAMES,
        "dual_frame_mode": DUAL_FRAME_MODE,
        "dual_frame_delta_s": DUAL_FRAME_DELTA_S if DUAL_FRAME_MODE else None,
        "started_at": datetime.now().isoformat(),
        "video_fps": fps,
        "video_resolution": [width, height],
    }
    (run_dir / "config.json").write_text(json.dumps(run_meta, indent=2))

    seg_log = (logs_dir / "seg_results.jsonl").open("w", encoding="utf-8")
    vlm_log = (logs_dir / "vlm_results.jsonl").open("w", encoding="utf-8")
    context_memory: Deque[dict] = deque(maxlen=config.VLM_CONTEXT_MEMORY_MAX_FRAMES)

    frame_idx = 0
    processed = 0
    danger_count = 0
    context_count = 0
    vlm_calls = 0
    vlm_skipped_blur = 0
    #? rolling buffer of sampled frames for dual frame lookups: (frame_bgr, frame_idx, video_ts)
    frame_buffer: deque = deque(maxlen=200)
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every != 0:
                frame_idx += 1
                continue

            current_ts = frame_idx / fps

            #? look up the best prior frame before appending the current one
            prior_frame = None
            prior_frame_idx = None
            actual_delta = None
            if DUAL_FRAME_MODE:
                target_ts = current_ts - DUAL_FRAME_DELTA_S
                for buf_f, buf_i, buf_ts in reversed(list(frame_buffer)):
                    if buf_ts <= target_ts:
                        prior_frame = buf_f
                        prior_frame_idx = buf_i
                        actual_delta = round(current_ts - buf_ts, 3)
                        break

            frame_buffer.append((frame.copy(), frame_idx, current_ts))

            seg_rec, vlm_rec = _process_frame(frame, frame_idx, current_ts,
                                               frames_dir, annotated_dir,
                                               seg_log, vlm_log, context_memory, ollama_url, use_context,
                                               prior_frame=prior_frame, prior_frame_idx=prior_frame_idx,
                                               temporal_delta_s=actual_delta)

            processed += 1
            if seg_rec["danger_detected"]:
                danger_count += 1
            if seg_rec["context_detected"]:
                context_count += 1
            if vlm_rec:
                vlm_calls += 1
            if seg_rec["blurry"] and (seg_rec["danger_detected"] or seg_rec["context_detected"]):
                vlm_skipped_blur += 1

            if processed % 20 == 0:
                pct = frame_idx / max(total_frames, 1) * 100
                logger.info("[%5.1f%%] frame %06d", pct, frame_idx)

            frame_idx += 1

    finally:
        cap.release()
        seg_log.close()
        vlm_log.close()

    total_time = time.time() - t_start
    summary = {
        "frames_processed": processed,
        "frames_danger_zone": danger_count,
        "frames_context_zone": context_count,
        "danger_rate": round(danger_count / max(processed, 1), 4),
        "vlm_calls": vlm_calls,
        "vlm_skipped_blurry": vlm_skipped_blur,
        "vlm_avg_time_s": 0,
        "total_time_s": round(total_time, 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Done.  danger=%d  context=%d  |  %d VLM calls  |  %d skipped (blurry)",
                danger_count, context_count, vlm_calls, vlm_skipped_blur)
    logger.info("Results saved to: %s", run_dir)


#? frame processing
#? shared logic, runs segmentation and VLM on one frame, saves outputs, writes log entries

#? runs seg and VLM on one frame, writes to logs, returns (seg_record, vlm_record or None)
def _process_frame(frame: np.ndarray, frame_idx: int, video_ts: float, frames_dir: Path, annotated_dir: Path, seg_log, vlm_log, context_memory: Deque, ollama_url: str, use_context: bool, prior_frame: Optional[np.ndarray] = None, prior_frame_idx: Optional[int] = None, temporal_delta_s: Optional[float] = None):
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    frame_stem = f"frame_{frame_idx:06d}"

    t_seg = time.time()
    mask, classes_present, annotated = seg_utils.run_segmentation(frame)
    danger_detected, context_detected, detections = seg_utils.get_obstacle_info(mask)
    seg_time = time.time() - t_seg

    blurry, lap_variance = seg_utils.is_blurry(frame)

    cv2.imwrite(str(frames_dir / f"{frame_stem}.jpg"), frame, jpeg_params)

    seg_record = {
        "frame_index": frame_idx,
        "video_timestamp_s": round(video_ts, 3),
        "danger_detected": danger_detected,
        "context_detected": context_detected,
        "detections": detections,
        "classes_present": sorted(classes_present),
        "seg_time_s": round(seg_time, 3),
        "laplacian_variance": lap_variance,
        "blurry": blurry,
    }
    seg_log.write(json.dumps(seg_record) + "\n")
    seg_log.flush()

    vlm_result = None
    trigger_zone = ""

    if danger_detected or context_detected:
        if blurry:
            logger.info("[frame %06d] Obstacle detected but frame is blurry (var=%.1f) - skipping VLM.", frame_idx, lap_variance)
        else:
            if danger_detected:
                trigger_zone = "danger"
                if prior_frame is not None:
                    prompt = build_prompt_danger_dual(temporal_delta_s, context_memory if use_context else None)
                else:
                    prompt = build_prompt_danger(context_memory if use_context else None)
            else:
                ctx_labels = [l for l, d in detections.items() if d["zone"] != seg_utils.ZONE_DANGER]
                side = "left" if detections[ctx_labels[0]]["zone"] == seg_utils.ZONE_CONTEXT_LEFT else "right"
                trigger_zone = f"context_{side}"
                if prior_frame is not None:
                    prompt = build_prompt_context_dual(side, temporal_delta_s, context_memory if use_context else None)
                else:
                    prompt = build_prompt_context(side, context_memory if use_context else None)

            vlm_mode_label = f"dual-frame (delta={temporal_delta_s}s)" if prior_frame is not None else "single-frame (cold start)"
            logger.info("[frame %06d] %s obstacle (var=%.1f) - calling VLM [%s] ...", frame_idx, trigger_zone, lap_variance, vlm_mode_label)

            if prior_frame is not None:
                vlm_result, vlm_time = call_vlm_dual(frame, prior_frame, prompt, ollama_url)
            else:
                vlm_result, vlm_time = call_vlm(frame, prompt, ollama_url)

            if vlm_result:
                vlm_record = {
                    "frame_index": frame_idx,
                    "video_timestamp_s": round(video_ts, 3),
                    "trigger_zone": trigger_zone,
                    "detections": detections,
                    "vlm_time_s": round(vlm_time, 3),
                    "dual_frame": prior_frame is not None,
                    "prior_frame_index": prior_frame_idx,
                    "temporal_delta_s": temporal_delta_s,
                    **vlm_result,
                }
                vlm_log.write(json.dumps(vlm_record) + "\n")
                vlm_log.flush()

                if use_context:
                    context_memory.append({
                        "obstacle_type": vlm_result["obstacle_type"],
                        "zone": trigger_zone,
                        "action": vlm_result["action"],
                        "confidence": vlm_result["confidence"],
                    })

                logger.info("  VLM -> %s | zone: %s | action: %s | %.1fs",
                            vlm_result.get("obstacle_type", "?"),
                            trigger_zone,
                            vlm_result.get("action", "?"),
                            vlm_time)

    if SAVE_ANNOTATED:
        ann = seg_utils.draw_obstacle_labels(annotated, detections, danger_detected, context_detected)
        if vlm_result:
            _draw_vlm_overlay(ann, vlm_result)
        if video_ts > 0:
            _draw_timestamp(ann, video_ts, frame_idx)
        cv2.imwrite(str(annotated_dir / f"{frame_stem}.jpg"), ann, jpeg_params)

    return seg_record, vlm_result


#? helpers
#? frame annotation utilities and Ollama health check

#? draws the VLM obstacle type, action, and confidence onto the annotated frame
def _draw_vlm_overlay(frame: np.ndarray, vlm_result: dict):
    action = vlm_result.get("action", "")
    obstacle_type = vlm_result.get("obstacle_type", "")
    confidence = vlm_result.get("confidence", "")
    action_colour = {
        "STOP": (0, 0, 255),
        "CONTINUE": (0, 200, 0),
        "TURN_LEFT": (0, 165, 255),
        "TURN_RIGHT": (0, 165, 255),
    }.get(action, (255, 255, 255))
    h = frame.shape[0]
    cv2.putText(frame, f"VLM: {obstacle_type}  [{confidence}]",
                (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACTION: {action}",
                (10, h - 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, action_colour, 2, cv2.LINE_AA)

#? draws the video timestamp and frame index in the bottom-left corner of the frame
def _draw_timestamp(frame: np.ndarray, video_ts: float, frame_idx: int):
    minutes = int(video_ts // 60)
    seconds = video_ts % 60
    text = f"t={minutes:02d}:{seconds:05.2f}  frame={frame_idx}"
    cv2.putText(frame, text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

#? creates and returns the four run output directories, routing dual frame runs to a separate folder
def _make_run_dirs(run_name: str, dual_mode: bool = False):
    base = config.RUNS_DIR_SEG_VLM_DUAL if dual_mode else config.RUNS_DIR_SEG_VLM
    run_dir = base / run_name
    frames_dir = run_dir / "frames"
    annotated_dir = run_dir / "annotated"
    logs_dir = run_dir / "logs"
    for d in (frames_dir, annotated_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return run_dir, frames_dir, annotated_dir, logs_dir


#? returns a sorted list of video files from outside_data_collection
def _list_video_files():
    d_path = Path(__file__).parent / "recordings" / "outside_data_collection"
    if not d_path.exists():
        logger.error("Folder not found: %s", d_path)
        sys.exit(1)
    files = sorted([f for f in d_path.iterdir()
                    if f.is_file() and f.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov')])
    if not files:
        logger.error("No video files found in %s", d_path)
        sys.exit(1)
    return files

#? lists available videos then asks the user to process all or pick one, returns (mode, files, prefix)
def _select_run_mode():
    files = _list_video_files()
    print("\nAvailable videos:")
    for i, f in enumerate(files):
        print(f"  {i + 1}) {f.name}")
    print("")
    while True:
        choice = input("Process [all] videos or [pick] one? ").strip().lower()
        if choice in ("all", "pick"):
            break
        print("Please enter 'all' or 'pick'.")
    if choice == "all":
        prefix = input("Enter a keyword prefix for the run names: ").strip()
        print("")
        return "all", files, prefix
    return "pick", files, ""

#? shows an interactive terminal selector and returns the chosen video path as a string
def _pick_video_file():
    files = _list_video_files()
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


#? checks that Ollama is reachable and that the configured VLM model is available
def _check_ollama(url: str):
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if config.VLM_MODEL not in models:
            logger.warning("Model '%s' not found in Ollama. Available: %s", config.VLM_MODEL, models)
            logger.warning("Run:  ollama pull %s", config.VLM_MODEL)
        else:
            logger.info("Ollama OK - model '%s' is available.", config.VLM_MODEL)
    except requests.RequestException as exc:
        logger.error("Cannot reach Ollama at %s: %s", url, exc)
        logger.error("Make sure Ollama is running:  ollama serve")
        sys.exit(1)


if __name__ == "__main__":
    main()
