# Camera server: streams Pi camera live and records AVI files via a browser UI.

from __future__ import annotations

import os
import sys
import time
import platform
import threading
import logging
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

#? ---
#? ENVIRONMENT DETECTION
#? Detects whether the server is running on a Raspberry Pi, macOS, or generic Linux.
#? ---

_IS_MAC = platform.system() == "Darwin"
_IS_LINUX = platform.system() == "Linux"
_IS_PI = _IS_LINUX and os.path.exists("/proc/device-tree/model") and \
         "raspberry" in open("/proc/device-tree/model", "r", errors="ignore").read().lower()

if _IS_PI:
    ENV_NAME = "Raspberry Pi"
    RECORDINGS_DIR = "/home/pi/recordings"
    SCREENSHOTS_DIR = "/home/pi/screenshots"
    CAM_BACKEND = cv2.CAP_V4L2
elif _IS_MAC:
    ENV_NAME = "macOS"
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "recordings")
    SCREENSHOTS_DIR = os.path.join(_SCRIPT_DIR, "screenshots")
    CAM_BACKEND = cv2.CAP_AVFOUNDATION
else:
    ENV_NAME = "Linux (generic)"
    RECORDINGS_DIR = os.path.expanduser("~/recordings")
    SCREENSHOTS_DIR = os.path.expanduser("~/screenshots")
    CAM_BACKEND = cv2.CAP_V4L2

# Safety guard - catch any misconfiguration early
assert isinstance(RECORDINGS_DIR, str), f"RECORDINGS_DIR is not a string: {RECORDINGS_DIR!r}"
assert isinstance(SCREENSHOTS_DIR, str), f"SCREENSHOTS_DIR is not a string: {SCREENSHOTS_DIR!r}"

#? ---
#? FIXED CONFIG
#? Static camera and server configuration values.
#? ---

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 30
RECORD_FPS = 2
JPEG_QUALITY = 80
SERVER_PORT = 5555
CAMERA_INDEX = 0

#? ---
#? TERMINAL BANNER
#? Prints a startup summary to the terminal with environment details and the server URL.
#? ---

# Prints a formatted startup banner showing environment, paths, and server URL
def _print_banner():
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    DIM = "\033[2m"

    if _IS_PI:
        env_colour = GREEN
        env_details = [
            f"  {DIM}Camera backend :{RESET} V4L2  (native Linux video layer)",
            f"  {DIM}Recordings     :{RESET} {RECORDINGS_DIR}",
            f"  {DIM}Screenshots    :{RESET} {SCREENSHOTS_DIR}",
            f"  {DIM}Note           :{RESET} Run  v4l2-ctl --list-devices  if camera index fails",
        ]
    elif _IS_MAC:
        env_colour = BLUE
        env_details = [
            f"  {DIM}Camera backend :{RESET} AVFoundation  (native macOS)",
            f"  {DIM}Recordings     :{RESET} {RECORDINGS_DIR}",
            f"  {DIM}Screenshots    :{RESET} {SCREENSHOTS_DIR}",
        ]
    else:
        env_colour = YELLOW
        env_details = [
            f"  {DIM}Camera backend :{RESET} V4L2",
            f"  {DIM}Recordings     :{RESET} {RECORDINGS_DIR}",
            f"  {DIM}Screenshots    :{RESET} {SCREENSHOTS_DIR}",
        ]

    width = 62
    line = "-" * width
    print(f"\n{CYAN}{BOLD}  ╔{'=' * (width - 2)}╗{RESET}")
    print(f"{CYAN}{BOLD}  ║{'Camera Server':^{width - 2}}║{RESET}")
    print(f"{CYAN}{BOLD}  ╚{'=' * (width - 2)}╝{RESET}")
    print(f"  {line}")
    print(f"  {DIM}Environment  :{RESET}  {env_colour}{BOLD}{ENV_NAME}{RESET}")
    print(f"  {line}")
    for detail in env_details:
        print(detail)
    print(f"  {line}")
    print(f"  {DIM}Web UI       :{RESET}  {CYAN}http://192.168.4.1:{SERVER_PORT}/{RESET}  (open in your browser)")
    print(f"  {line}\n")
    sys.stdout.flush()


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

app = Flask(__name__)

#? ---
#? SHARED STATE
#? Threading locks and mutable globals for the camera frame, recording state, and screenshots.
#? ---

_latest_frame = None
_frame_lock = threading.Lock()

_is_recording = False
_video_writer = None
_recording_path = None
_recording_start = None
_recording_fps = RECORD_FPS
_state_lock = threading.Lock()

_rec_frame_interval = max(1, round(STREAM_FPS / RECORD_FPS))
_rec_frame_counter = 0
_rec_frame_lock = threading.Lock()

_screenshots = []
_screenshots_lock = threading.Lock()

#? ---
#? CAMERA LOOP
#? Background thread that reads frames from the camera and writes to the active recording.
#? ---

# Runs in a background thread - grabs frames, updates the shared frame, and writes to the recorder
def camera_loop():
    global _latest_frame, _is_recording, _video_writer
    global _rec_frame_counter, _rec_frame_interval

    log.info("Starting OpenCV VideoCapture - index %d, backend: %s", CAMERA_INDEX, ENV_NAME)
    cap = cv2.VideoCapture(CAMERA_INDEX, CAM_BACKEND)

    if not cap.isOpened():
        log.error("Could not open camera at index %d.", CAMERA_INDEX)
        if _IS_PI:
            log.error("Run  v4l2-ctl --list-devices  to confirm the correct index.")
        elif _IS_MAC:
            log.error("Check System Settings - Privacy & Security - Camera.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, STREAM_FPS)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log.info("Camera initialised at %dx%d", actual_w, actual_h)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to grab frame. Retrying...")
                time.sleep(0.1)
                continue

            with _frame_lock:
                _latest_frame = frame.copy()

            # Sub-sample: write only every Nth frame to keep recording at chosen FPS
            with _rec_frame_lock:
                _rec_frame_counter += 1
                should_write = (_rec_frame_counter >= _rec_frame_interval)
                if should_write:
                    _rec_frame_counter = 0

            if should_write:
                with _state_lock:
                    if _is_recording and _video_writer is not None:
                        _video_writer.write(frame)

    except Exception as e:
        log.error("Camera loop error: %s", e)
    finally:
        cap.release()
        log.info("Camera pipeline stopped.")

#? ---
#? MJPEG STREAM
#? Yields MJPEG-encoded frames for the browser live preview - does not write to disk.
#? ---

# Generator that yields MJPEG-encoded frames for the /video_feed route
def _generate_mjpeg():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            placeholder = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "Waiting for camera ...",
                (STREAM_WIDTH // 2 - 200, STREAM_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2,
            )
            frame = placeholder

        _, buf = cv2.imencode(".jpg", frame, encode_params)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1.0 / STREAM_FPS)

#? ---
#? INTERNAL HELPERS
#? Shared utility for stopping an active recording safely from any thread.
#? ---

# Stops the active recording and returns (saved_path, size_mb) - thread-safe
def _do_stop_recording():
    global _is_recording, _video_writer, _recording_path, _recording_start
    with _state_lock:
        if not _is_recording:
            return None, 0
        _is_recording = False
        _recording_start = None
        if _video_writer:
            _video_writer.release()
            _video_writer = None
        saved = _recording_path
        _recording_path = None

    if saved and os.path.exists(saved):
        size_mb = os.path.getsize(saved) / (1024 * 1024)
        log.info("Recording saved -> %s (%.1f MB)", saved, size_mb)
        return saved, round(size_mb, 2)
    return saved, 0

#? ---
#? FLASK ROUTES
#? API and page routes served to the browser UI.
#? ---

# Serves the main HTML page
@app.route("/")
def index():
    return _HTML_PAGE

# Serves the MJPEG camera stream
@app.route("/video_feed")
def video_feed():
    return Response(_generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Starts a new recording at the requested FPS, optionally stopping automatically after a duration
@app.route("/start_recording", methods=["POST"])
def start_recording():
    global _is_recording, _video_writer, _recording_path, _recording_start
    global _recording_fps, _rec_frame_interval, _rec_frame_counter

    body = request.get_json(force=True, silent=True) or {}
    duration = int(body.get("duration", 0))
    req_fps = int(body.get("fps", RECORD_FPS))
    req_fps = max(1, min(req_fps, STREAM_FPS))

    with _state_lock:
        if _is_recording:
            return jsonify({"status": "already_recording", "file": _recording_path})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}_{req_fps}fps.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        _video_writer = cv2.VideoWriter(
            path, fourcc, req_fps,
            (int(STREAM_WIDTH), int(STREAM_HEIGHT))
        )

        if not _video_writer.isOpened():
            log.error("VideoWriter failed to open.")
            _video_writer = None
            return jsonify({"status": "error", "message": "Failed to create video file."})

        _recording_path = path
        _is_recording = True
        _recording_start = time.time()
        _recording_fps = req_fps

    with _rec_frame_lock:
        _rec_frame_interval = max(1, round(STREAM_FPS / req_fps))
        _rec_frame_counter = 0

    log.info("Recording started -> %s  (%d fps)%s", path, req_fps,
             f"  auto-stop in {duration}s" if duration else "")

    if duration > 0:
        def _auto_stop():
            time.sleep(duration)
            _do_stop_recording()
        threading.Thread(target=_auto_stop, daemon=True, name="auto-stop").start()

    return jsonify({"status": "started", "file": path, "duration": duration, "fps": req_fps})

# Stops the current recording and returns the saved file path and size
@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    saved, size_mb = _do_stop_recording()
    if saved:
        return jsonify({"status": "stopped", "saved_to": saved, "size_mb": size_mb})
    return jsonify({"status": "not_recording"})

# Captures a single JPEG screenshot from the current frame and saves it to disk
@app.route("/take_screenshot", methods=["POST"])
def take_screenshot():
    with _frame_lock:
        frame = _latest_frame

    if frame is None:
        return jsonify({"status": "error", "message": "No frame available yet."})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.jpg"
    path = os.path.join(SCREENSHOTS_DIR, filename)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({"status": "error", "message": "Failed to encode screenshot."})

    with open(path, "wb") as f:
        f.write(buf.tobytes())

    thumb = frame.copy()
    h, w = thumb.shape[:2]
    scale = min(1.0, 240 / w)
    thumb = cv2.resize(thumb, (int(w * scale), int(h * scale)))
    _, tbuf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    thumb_b64 = base64.b64encode(tbuf.tobytes()).decode("utf-8")

    entry = {"filename": filename, "path": path, "thumb_b64": thumb_b64}
    with _screenshots_lock:
        _screenshots.append(entry)

    log.info("Screenshot saved -> %s", path)
    return jsonify({"status": "ok", "filename": filename, "thumb_b64": thumb_b64})

# Renames a saved screenshot file on disk and updates the in-memory list
@app.route("/rename_screenshot", methods=["POST"])
def rename_screenshot():
    data = request.get_json(force=True)
    old_name = os.path.basename(data.get("old_name", "").strip())
    new_name = os.path.basename(data.get("new_name", "").strip())

    if not old_name or not new_name:
        return jsonify({"status": "error", "message": "old_name and new_name required."})
    if not new_name.lower().endswith(".jpg"):
        new_name += ".jpg"

    old_path = os.path.join(SCREENSHOTS_DIR, old_name)
    new_path = os.path.join(SCREENSHOTS_DIR, new_name)

    if not os.path.exists(old_path):
        return jsonify({"status": "error", "message": "File not found."})
    if os.path.exists(new_path):
        return jsonify({"status": "error", "message": "Target name already exists."})

    os.rename(old_path, new_path)
    with _screenshots_lock:
        for entry in _screenshots:
            if entry["filename"] == old_name:
                entry["filename"] = new_name
                entry["path"] = new_path
                break

    log.info("Screenshot renamed: %s -> %s", old_name, new_name)
    return jsonify({"status": "ok", "new_name": new_name})

# Returns the current server configuration as JSON
@app.route("/config")
def config():
    return jsonify({
        "env": ENV_NAME,
        "recordings_dir": RECORDINGS_DIR,
        "screenshots_dir": SCREENSHOTS_DIR,
        "record_fps": RECORD_FPS,
        "stream_fps": STREAM_FPS,
    })

# Returns recording status, elapsed time, recent recordings list, and screenshots
@app.route("/status")
def status():
    with _state_lock:
        recording = _is_recording
        path = _recording_path
        started = _recording_start
        current_fps = _recording_fps

    elapsed_s = round(time.time() - started, 1) if (recording and started) else 0
    est_size_mb = round(elapsed_s * current_fps * 115 / 1024, 2) if elapsed_s else 0

    recordings = []
    try:
        for fname in sorted(
            [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".avi")],
            reverse=True
        )[:10]:
            fpath = os.path.join(RECORDINGS_DIR, fname)
            try:
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            except OSError:
                size_mb = 0
            try:
                parts = fname.replace("recording_", "").replace(".avi", "")
                if parts.endswith("fps"):
                    parts = "_".join(parts.split("_")[:-1])
                dt = datetime.strptime(parts, "%Y%m%d_%H%M%S")
                when = dt.strftime("%Y-%m-%d  %H:%M:%S")
            except Exception:
                when = ""
            recordings.append({"name": fname, "size_mb": size_mb, "when": when})
    except Exception:
        pass

    with _screenshots_lock:
        shots = [{"filename": s["filename"], "thumb_b64": s["thumb_b64"]}
                 for s in _screenshots]

    return jsonify({
        "recording": recording,
        "file": path,
        "elapsed_s": elapsed_s,
        "est_size_mb": est_size_mb,
        "current_fps": current_fps,
        "recordings": recordings,
        "screenshots": shots,
    })

#? ---
#? HTML PAGE
#? Single-file browser UI served at the root route.
#? ---

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Camera — Live Stream</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #f0f0f0;
    color: #222;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px 12px 32px;
    min-height: 100vh;
    -webkit-text-size-adjust: 100%;
  }

  h1 {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: #111;
    letter-spacing: 0.02em;
    text-align: center;
  }

  /* -- Desktop: side-by-side; Mobile: stacked -- */
  #layout {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    width: 100%;
    max-width: 980px;
  }

  #left {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
  }

  /* Stream */
  #stream-container {
    position: relative;
    border: 2px solid #ccc;
    border-radius: 8px;
    overflow: hidden;
    background: #ddd;
    width: 100%;
    max-width: 1280px;
    /* Enforce 16:9 aspect ratio so it scales correctly on all screen sizes */
    aspect-ratio: 16 / 9;
  }
  #stream {
    display: block;
    width: 100%;
    height: auto;
  }

  #rec-badge {
    display: none;
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(200,0,0,0.88);
    color: #fff;
    padding: 4px 9px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    animation: blink 1.2s step-start infinite;
  }
  #rec-overlay {
    display: none;
    position: absolute;
    top: 10px; right: 10px;
    background: rgba(0,0,0,0.55);
    color: #fff;
    padding: 4px 9px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
    text-align: right;
    white-space: pre;
    line-height: 1.4;
  }
  @keyframes blink { 50% { opacity: 0.3; } }

  /* -- FPS selector -- */
  #fps-row {
    display: flex;
    gap: 8px;
    margin-top: 14px;
    align-items: center;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 1280px;
  }
  #fps-row label {
    font-size: 0.82rem;
    color: #555;
    font-weight: 500;
  }
  #fps-select {
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #bbb;
    background: #fff;
    font-size: 0.88rem;
    font-weight: 600;
    color: #222;
    cursor: pointer;
    /* larger touch target */
    min-height: 40px;
  }
  #fps-hint {
    font-size: 0.74rem;
    color: #888;
  }

  /* -- Controls -- */
  #controls {
    display: flex;
    gap: 10px;
    margin-top: 12px;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 1280px;
  }

  button {
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    /* minimum touch target */
    min-height: 44px;
    -webkit-tap-highlight-color: transparent;
  }
  button:active  { transform: scale(0.96); }
  button:disabled { opacity: 0.35; cursor: not-allowed; }

  #btn-start      { background: #e03030; color: #fff; flex: 1 1 auto; }
  #btn-stop       { background: #555;    color: #eee; flex: 1 1 auto; }
  #btn-screenshot { background: #2a7ae2; color: #fff; flex: 1 1 auto; }

  /* Timed buttons */
  #timed-controls {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    width: 100%;
    max-width: 1280px;
  }
  #timed-label {
    font-size: 0.78rem;
    color: #666;
    width: 100%;
    text-align: center;
    margin-bottom: 2px;
  }
  .btn-timed {
    background: #d97706;
    color: #fff;
    flex: 1 1 60px;
    min-width: 60px;
    padding: 11px 10px;
    font-size: 0.85rem;
  }

  /* Status bar */
  #status-bar {
    margin-top: 12px;
    padding: 10px 16px;
    background: #e4e4e4;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 0.82rem;
    color: #555;
    width: 100%;
    max-width: 1280px;
    min-height: 38px;
    text-align: center;
    line-height: 1.4;
  }
  #status-bar span { color: #111; font-weight: 600; }

  /* Recordings list */
  #recordings {
    margin-top: 14px;
    width: 100%;
    max-width: 1280px;
  }
  #recordings h2 {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 8px;
    font-weight: 500;
  }
  #rec-list { list-style: none; }
  #rec-list li {
    background: #e8e8e8;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.78rem;
    color: #333;
    font-family: monospace;
    display: flex;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .rec-name { flex: 1 1 auto; word-break: break-all; }
  .rec-meta { color: #777; white-space: nowrap; font-size: 0.74rem; }

  /* Screenshot section — inline below recordings on mobile */
  #screenshots-section {
    margin-top: 14px;
    width: 100%;
    max-width: 1280px;
  }
  #screenshots-section h2 {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 8px;
    font-weight: 500;
  }
  #shot-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 8px;
  }
  .shot-card {
    background: #e8e8e8;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 5px;
    cursor: pointer;
    transition: box-shadow 0.15s;
    -webkit-tap-highlight-color: transparent;
  }
  .shot-card:active { box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
  .shot-card img    { width: 100%; border-radius: 4px; display: block; }
  .shot-name {
    font-size: 0.65rem;
    color: #444;
    margin-top: 3px;
    word-break: break-all;
    font-family: monospace;
  }

  /* Desktop: put screenshots in right column */
  @media (min-width: 760px) {
    #layout { flex-direction: row; }
    #left   { max-width: 660px; }
    #screenshots-section {
      flex: 1 1 200px;
      min-width: 160px;
      max-width: 240px;
      margin-top: 0;
    }
    #shot-list {
      grid-template-columns: 1fr;
      max-height: 600px;
      overflow-y: auto;
    }
    .shot-name { font-size: 0.70rem; }
  }

  /* Lightbox */
  #lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.80);
    z-index: 200;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    gap: 12px;
    padding: 16px;
  }
  #lightbox.open { display: flex; }
  #lightbox img  {
    max-width: 100%;
    max-height: 65vh;
    border-radius: 6px;
    object-fit: contain;
  }
  #lb-filename {
    color: #fff;
    font-size: 0.82rem;
    font-family: monospace;
    text-align: center;
    word-break: break-all;
  }
  #lb-controls {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 420px;
  }
  #lb-rename-input {
    padding: 10px 12px;
    border-radius: 6px;
    border: none;
    font-size: 0.9rem;
    flex: 1 1 160px;
    min-height: 44px;
  }
  #lb-rename-btn { background: #2a7ae2; color: #fff; flex: 0 0 auto; }
  #lb-close-btn  { background: #666;    color: #fff; flex: 0 0 auto; }
  #lb-msg { color: #aef; font-size: 0.80rem; min-height: 1.2em; text-align: center; }
</style>
</head>
<body>

<h1>Robot Camera — Live Stream</h1>

<div id="layout">

  <!-- Main column -->
  <div id="left">

    <!-- Stream -->
    <div id="stream-container">
      <img id="stream" src="/video_feed" width="1280" height="720" alt="stream">
      <div id="rec-badge">&#9679; REC</div>
      <div id="rec-overlay"></div>
    </div>

    <!-- FPS selector -->
    <div id="fps-row">
      <label for="fps-select">Recording FPS:</label>
      <select id="fps-select">
        <option value="1">1 fps — 1 frame/s</option>
        <option value="2" selected>2 fps — 1 frame/0.5s  (default)</option>
        <option value="5">5 fps — 1 frame/0.2s</option>
        <option value="10">10 fps — smooth</option>
        <option value="30">30 fps — full smooth</option>
      </select>
      <span id="fps-hint"></span>
    </div>

    <!-- Main buttons -->
    <div id="controls">
      <button id="btn-start"      onclick="startRec(0)">&#9679; Start Recording</button>
      <button id="btn-stop"       onclick="stopRec()">&#9632; Stop</button>
      <button id="btn-screenshot" onclick="takeShot()">&#128247; Screenshot</button>
    </div>

    <!-- Timed buttons — NOT disabled in HTML; JS manages state -->
    <div id="timed-controls">
      <div id="timed-label">Timed recording:</div>
      <button class="btn-timed" id="btn-t5"  onclick="startRec(5)">5 s</button>
      <button class="btn-timed" id="btn-t15" onclick="startRec(15)">15 s</button>
      <button class="btn-timed" id="btn-t30" onclick="startRec(30)">30 s</button>
      <button class="btn-timed" id="btn-t60" onclick="startRec(60)">1 min</button>
    </div>

    <div id="status-bar">Ready — choose FPS and hit Start Recording.</div>

    <!-- Recordings list -->
    <div id="recordings">
      <h2 id="rec-dir-label">Saved recordings</h2>
      <ul id="rec-list"></ul>
    </div>

    <!-- Screenshots -->
    <div id="screenshots-section">
      <h2>Screenshots</h2>
      <div id="shot-list"></div>
    </div>

  </div><!-- /#left -->

</div><!-- /#layout -->

<!-- Lightbox -->
<div id="lightbox">
  <img id="lb-img" src="" alt="screenshot">
  <div id="lb-filename"></div>
  <div id="lb-controls">
    <input id="lb-rename-input" type="text" placeholder="New filename (without .jpg)" autocorrect="off" autocapitalize="off">
    <button id="lb-rename-btn" onclick="renameShot()">Rename</button>
    <button id="lb-close-btn"  onclick="closeLightbox()">&#10005; Close</button>
  </div>
  <div id="lb-msg"></div>
</div>

<script>
  const statusBar       = document.getElementById('status-bar');
  const recBadge        = document.getElementById('rec-badge');
  const recOverlay      = document.getElementById('rec-overlay');
  const btnStart        = document.getElementById('btn-start');
  const btnStop         = document.getElementById('btn-stop');
  const recList         = document.getElementById('rec-list');
  const shotList        = document.getElementById('shot-list');
  const fpsSelect       = document.getElementById('fps-select');
  const fpsHint         = document.getElementById('fps-hint');
  const allTimedBtns    = document.querySelectorAll('.btn-timed');

  // FPS hint text
  const fpsHints = {
    '1':  'Very sparse — 1 sample/s  (~7 MB/min)',
    '2':  'Recommended for VLM (~5 s cadence)  (~15 MB/min)',
    '5':  'Good temporal detail  (~29 MB/min)',
    '10': 'Smooth  (~77 MB/min)',
    '30': 'Full smooth — very large files  (~215 MB/min)',
  };
  fpsSelect.addEventListener('change', () => {
    fpsHint.textContent = fpsHints[fpsSelect.value] || '';
  });
  fpsHint.textContent = fpsHints[fpsSelect.value] || '';

  // -- Elapsed timer --
  let _recStartEpoch = null;
  let _elapsedTimer  = null;
  let _timedDuration = 0;
  let _currentFps    = 2;

  function _fmtTime(s) {
    s = Math.max(0, s);
    const m   = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return (m > 0 ? m + 'm ' : '') + String(sec).padStart(2, '0') + 's';
  }

  function _startElapsedTimer(durationSec, fps) {
    _timedDuration = durationSec || 0;
    _currentFps    = fps || 2;
    if (_elapsedTimer) clearInterval(_elapsedTimer);
    _elapsedTimer = setInterval(() => {
      if (!_recStartEpoch) return;
      const elapsed = (Date.now() - _recStartEpoch) / 1000;
      const estKB   = elapsed * _currentFps * 115;  // ~115 KB/frame at 1280×720 MJPG
      const estStr  = estKB < 1024
        ? estKB.toFixed(0) + ' KB'
        : (estKB / 1024).toFixed(1) + ' MB';
      let overlay = _fmtTime(elapsed) + '  ~' + estStr;
      if (_timedDuration > 0) overlay += '\\n-' + _fmtTime(_timedDuration - elapsed);
      recOverlay.textContent = overlay;
      statusBar.innerHTML =
        'Recording  <span>' + _fmtTime(elapsed) + '</span>' +
        '  ·  ~<span>' + estStr + '</span>' +
        (_timedDuration > 0 ? '  ·  <span>-' + _fmtTime(_timedDuration - elapsed) + '</span>' : '');
    }, 500);
  }

  function _stopElapsedTimer() {
    if (_elapsedTimer) { clearInterval(_elapsedTimer); _elapsedTimer = null; }
    _recStartEpoch = null;
    _timedDuration = 0;
    recOverlay.textContent  = '';
    recOverlay.style.display = 'none';
  }

  // -- UI state: enable/disable buttons based on recording status --
  function setRecordingUI(active) {
    recBadge.style.display   = active ? 'block' : 'none';
    recOverlay.style.display = active ? 'block' : 'none';
    btnStart.disabled  = active;
    btnStop.disabled   = !active;
    fpsSelect.disabled = active;
    // Timed buttons: enabled only when NOT recording
    allTimedBtns.forEach(b => { b.disabled = active; });
    if (!active) _stopElapsedTimer();
  }

  async function startRec(duration) {
    const fps = parseInt(fpsSelect.value, 10);
    const r   = await fetch('/start_recording', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ duration, fps })
    });
    const d = await r.json();
    if (d.status === 'started') {
      _recStartEpoch = Date.now();
      setRecordingUI(true);
      _startElapsedTimer(duration, fps);
      const label = duration
        ? ' (' + (duration >= 60 ? (duration / 60) + ' min' : duration + 's') + ')'
        : '';
      statusBar.innerHTML = 'Recording' + label + ' at <span>' + fps + ' fps</span>';
    } else {
      statusBar.textContent = 'Could not start: ' + (d.message || d.status);
    }
  }

  async function stopRec() {
    const r = await fetch('/stop_recording', { method: 'POST' });
    const d = await r.json();
    setRecordingUI(false);
    const name = d.saved_to ? d.saved_to.split('/').pop() : '';
    const size = d.size_mb  ? ' (' + d.size_mb + ' MB)'   : '';
    statusBar.innerHTML = name
      ? 'Saved: <span>' + name + size + '</span>'
      : 'Stopped.';
    refreshStatus();
  }

  // -- Screenshot --
  let lbCurrentName = '';

  function openLightbox(thumb_b64, filename) {
    lbCurrentName = filename;
    document.getElementById('lb-img').src              = 'data:image/jpeg;base64,' + thumb_b64;
    document.getElementById('lb-filename').textContent = filename;
    document.getElementById('lb-rename-input').value   = filename.replace(/\\.jpg$/i, '');
    document.getElementById('lb-msg').textContent      = '';
    document.getElementById('lightbox').classList.add('open');
  }

  function closeLightbox() {
    document.getElementById('lightbox').classList.remove('open');
  }

  async function renameShot() {
    const input = document.getElementById('lb-rename-input').value.trim();
    const msgEl = document.getElementById('lb-msg');
    if (!input) { msgEl.textContent = 'Please enter a name.'; return; }
    const r = await fetch('/rename_screenshot', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ old_name: lbCurrentName, new_name: input })
    });
    const d = await r.json();
    if (d.status === 'ok') {
      lbCurrentName = d.new_name;
      document.getElementById('lb-filename').textContent = d.new_name;
      document.getElementById('lb-rename-input').value   = d.new_name.replace(/\\.jpg$/i, '');
      msgEl.textContent = 'Renamed successfully.';
      document.querySelectorAll('.shot-card').forEach(card => {
        if (card.dataset.filename === lbCurrentName) {
          card.querySelector('.shot-name').textContent = d.new_name;
          card.dataset.filename = d.new_name;
        }
      });
      refreshStatus();
    } else {
      msgEl.textContent = 'Error: ' + (d.message || 'unknown');
    }
  }

  async function takeShot() {
    statusBar.textContent = 'Capturing screenshot...';
    const r = await fetch('/take_screenshot', { method: 'POST' });
    const d = await r.json();
    if (d.status === 'ok') {
      addShotCard(d.filename, d.thumb_b64);
      statusBar.innerHTML = 'Screenshot: <span>' + d.filename + '</span>';
    } else {
      statusBar.textContent = 'Screenshot failed: ' + (d.message || '');
    }
  }

  function addShotCard(filename, thumb_b64) {
    if (document.querySelector('[data-filename="' + CSS.escape(filename) + '"]')) return;
    const card = document.createElement('div');
    card.className        = 'shot-card';
    card.dataset.filename = filename;
    card.innerHTML =
      '<img src="data:image/jpeg;base64,' + thumb_b64 + '" alt="' + filename + '" loading="lazy">' +
      '<div class="shot-name">' + filename + '</div>';
    card.addEventListener('click', () => openLightbox(thumb_b64, card.dataset.filename));
    shotList.prepend(card);
  }

  // -- Status polling --
  async function refreshStatus() {
    try {
      const r = await fetch('/status');
      const d = await r.json();

      if (d.recording && !_recStartEpoch && d.elapsed_s > 0) {
        _recStartEpoch = Date.now() - d.elapsed_s * 1000;
        _startElapsedTimer(0, d.current_fps || 2);
      }
      if (!d.recording && _elapsedTimer) {
        _stopElapsedTimer();
      }

      // Always sync button state from server truth
      setRecordingUI(d.recording);

      if (!d.recording &&
          !statusBar.innerHTML.includes('Saved') &&
          !statusBar.innerHTML.includes('Screenshot') &&
          !statusBar.innerHTML.includes('Recording')) {
        statusBar.textContent = 'Ready — choose FPS and hit Start Recording.';
      }

      recList.innerHTML = '';
      (d.recordings || []).forEach(rec => {
        const li   = document.createElement('li');
        const name = document.createElement('span');
        name.className   = 'rec-name';
        name.textContent = rec.name;
        const meta = document.createElement('span');
        meta.className   = 'rec-meta';
        meta.textContent = (rec.when ? rec.when + '  ' : '') + rec.size_mb + ' MB';
        li.appendChild(name);
        li.appendChild(meta);
        recList.appendChild(li);
      });

      (d.screenshots || []).forEach(s => addShotCard(s.filename, s.thumb_b64));
    } catch (_) {}
  }

  document.getElementById('lightbox').addEventListener('click', function(e) {
    if (e.target === this) closeLightbox();
  });

  // Config fetch on load
  (async () => {
    try {
      const r = await fetch('/config');
      const d = await r.json();
      document.getElementById('rec-dir-label').textContent =
        'Saved recordings (' + d.recordings_dir + '/)';
      document.title = 'Camera — ' + d.env;
      document.querySelector('h1').textContent = 'Camera  [' + d.env + ']';
    } catch (_) {}
  })();

  setRecordingUI(false);

  refreshStatus();
  setInterval(refreshStatus, 2000);
</script>

</body>
</html>
"""

if __name__ == "__main__":
    _print_banner()
    cam_thread = threading.Thread(target=camera_loop, daemon=True, name="camera")
    cam_thread.start()
    time.sleep(2)
    log.info("Server starting on http://0.0.0.0:%d", SERVER_PORT)
    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True, use_reloader=False)
