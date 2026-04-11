# Flask web viewer — serves run data and frame images to the browser UI at localhost:5050.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, render_template, send_file

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as pipeline_config

app = Flask(__name__)

RUNS_DIR = Path(pipeline_config.RUNS_DIR)


#? ---
#? HELPERS
#? Functions that read run folders and build data structures for the API routes.
#? ---

# Lists all run folders with their metadata and summary stats
def _list_runs():
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        runs.append({
            "name": run_dir.name,
            "setup": meta.get("setup", "unknown"),
            "video": Path(meta.get("video", "")).name,
            "started_at": meta.get("started_at", ""),
            "frames_processed": summary.get("frames_processed", 0),
            "obstacle_rate": summary.get("obstacle_rate", 0),
            "vlm_calls": summary.get("vlm_calls"),
        })
    return runs

# Reads the seg_results.jsonl log for a run and returns it as a list of dicts
def _load_seg_log(run_name: str):
    path = RUNS_DIR / run_name / "logs" / "seg_results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

# Reads the vlm_results.jsonl log and returns a dict keyed by frame_index
def _load_vlm_log(run_name: str):
    path = RUNS_DIR / run_name / "logs" / "vlm_results.jsonl"
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            result[rec["frame_index"]] = rec
    return result

# Returns the path to a saved frame JPEG, falling back to original if annotated is missing
def _frame_path(run_name: str, frame_index: int, annotated: bool):
    stem = f"frame_{frame_index:06d}.jpg"
    subdir = "annotated" if annotated else "frames"
    path = RUNS_DIR / run_name / subdir / stem
    return path if path.exists() else None


#? ---
#? ROUTES
#? Flask API and page routes served to the browser viewer.
#? ---

# Serves the main viewer HTML page
@app.route("/")
def index():
    return render_template("index.html")

# Returns a JSON list of all available runs
@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())

# Returns all frame records for a run, merged with any VLM results
@app.route("/api/runs/<run_name>/frames")
def api_frames(run_name: str):
    seg_log = _load_seg_log(run_name)
    vlm_log = _load_vlm_log(run_name)
    frames = []
    for rec in seg_log:
        fi = rec["frame_index"]
        entry = {
            "frame_index": fi,
            "video_timestamp_s": rec.get("video_timestamp_s", 0),
            "danger_detected": rec.get("danger_detected", rec.get("obstacle_detected", False)),
            "context_detected": rec.get("context_detected", False),
            "detections": rec.get("detections", {}),
            "classes_present": rec.get("classes_present", []),
            "seg_time_s": rec.get("seg_time_s") or rec.get("inference_time_s"),
            "laplacian_variance": rec.get("laplacian_variance"),
            "blurry": rec.get("blurry", False),
            "has_annotated": (RUNS_DIR / run_name / "annotated" / f"frame_{fi:06d}.jpg").exists(),
            "vlm": vlm_log.get(fi),
        }
        frames.append(entry)
    return jsonify(frames)

# Returns the config.json and summary.json for a run
@app.route("/api/runs/<run_name>/config")
def api_run_config(run_name: str):
    meta_path = RUNS_DIR / run_name / "config.json"
    summary_path = RUNS_DIR / run_name / "summary.json"
    if not meta_path.exists():
        abort(404)
    meta = json.loads(meta_path.read_text())
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return jsonify({"meta": meta, "summary": summary})

# Serves a frame JPEG (original or annotated) for display in the viewer
@app.route("/api/runs/<run_name>/image/<int:frame_index>/<string:kind>")
def api_image(run_name: str, frame_index: int, kind: str):
    annotated = kind == "annotated"
    path = _frame_path(run_name, frame_index, annotated)
    if path is None and annotated:
        path = _frame_path(run_name, frame_index, False)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    print(f"Viewer starting - runs directory: {RUNS_DIR}")
    print("Open http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
