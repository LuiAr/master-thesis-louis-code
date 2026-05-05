from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, render_template, send_file

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as pipeline_config

app = Flask(__name__)

RUNS_DIR_SEG_ONLY = Path(pipeline_config.RUNS_DIR_SEG_ONLY)
RUNS_DIR_SEG_VLM = Path(pipeline_config.RUNS_DIR_SEG_VLM)
_ALL_RUN_ROOTS = [RUNS_DIR_SEG_ONLY, RUNS_DIR_SEG_VLM]


def _resolve_run_dir(run_name: str):
    for root in _ALL_RUN_ROOTS:
        candidate = root / run_name
        if candidate.is_dir():
            return candidate
    return None

#? lists all run folders from both pipeline subdirectories
def _list_runs():
    runs = []
    for root in _ALL_RUN_ROOTS:
        if not root.exists():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "config.json"
            summary_path = run_dir / "summary.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            started_at = meta.get("started_at", "")
            sort_key = started_at if started_at else datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat()
            runs.append({
                "name": run_dir.name,
                "setup": meta.get("setup", "unknown"),
                "video": Path(meta.get("video", "")).name,
                "started_at": started_at,
                "frames_processed": summary.get("frames_processed", 0),
                "obstacle_rate": summary.get("obstacle_rate", 0),
                "vlm_calls": summary.get("vlm_calls"),
                "_sort_key": sort_key,
            })
    runs.sort(key=lambda r: r["_sort_key"], reverse=True)
    for r in runs:
        del r["_sort_key"]
    return runs

def _load_seg_log(run_name: str):
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        return []
    path = run_dir / "logs" / "seg_results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

#? reads the vlm_results.jsonl log and returns a dict keyed by frame_index
def _load_vlm_log(run_name: str):
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        return {}
    path = run_dir / "logs" / "vlm_results.jsonl"
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            result[rec["frame_index"]] = rec
    return result

def _frame_path(run_name: str, frame_index: int, annotated: bool):
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        return None
    stem = f"frame_{frame_index:06d}.jpg"
    subdir = "annotated" if annotated else "frames"
    path = run_dir / subdir / stem
    return path if path.exists() else None


#? routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())
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
            "has_annotated": ((_resolve_run_dir(run_name) or Path()) / "annotated" / f"frame_{fi:06d}.jpg").exists(),
            "vlm": vlm_log.get(fi),
        }
        frames.append(entry)
    return jsonify(frames)

@app.route("/api/runs/<run_name>/config")
def api_run_config(run_name: str):
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        abort(404)
    meta_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    if not meta_path.exists():
        abort(404)
    meta = json.loads(meta_path.read_text())
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return jsonify({"meta": meta, "summary": summary})

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
    print(f"Viewer starting - seg_only: {RUNS_DIR_SEG_ONLY} | seg_vlm: {RUNS_DIR_SEG_VLM}")
    print("Open http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
