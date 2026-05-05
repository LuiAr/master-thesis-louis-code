#? results evaluator, manual review of pipeline decisions frame by frame

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as pipeline_config

RUNS_DIR_SEG_ONLY = Path(pipeline_config.RUNS_DIR_SEG_ONLY)
RUNS_DIR_SEG_VLM = Path(pipeline_config.RUNS_DIR_SEG_VLM)
RUNS_DIR_SEG_VLM_DUAL = Path(pipeline_config.RUNS_DIR_SEG_VLM_DUAL)
_ALL_RUN_ROOTS = [RUNS_DIR_SEG_ONLY, RUNS_DIR_SEG_VLM, RUNS_DIR_SEG_VLM_DUAL]
EVALUATIONS_DIR = Path(__file__).parent.parent / "evaluations"
EVALUATIONS_DIR.mkdir(exist_ok=True)
THESIS_RESULTS_DIR = Path(__file__).parent.parent / "thesis_results"
THESIS_RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


#? data loaders
def _resolve_run_dir(run_name: str):
    for root in _ALL_RUN_ROOTS:
        candidate = root / run_name
        if candidate.is_dir():
            return candidate
    return None

#? returns a list of all pipeline run folders from both subdirectories
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
                "vlm_calls": summary.get("vlm_calls"),
                "_sort_key": sort_key,
            })
    runs.sort(key=lambda r: r["_sort_key"], reverse=True)
    for r in runs:
        del r["_sort_key"]
    return runs

#? loads VLM decisions as a sorted list of dicts
def _load_vlm_decisions(run_name: str):
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        return []
    path = run_dir / "logs" / "vlm_results.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(records, key=lambda r: r["frame_index"])

#? loads segmentation detections for runs without a VLM log (seg_only setup)
def _load_seg_decisions(run_name: str):
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        return []
    path = run_dir / "logs" / "seg_results.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    danger = [r for r in records if r.get("danger_detected") or r.get("obstacle_detected")]
    return sorted(danger, key=lambda r: r["frame_index"])

#? returns the path for the manual evaluation JSON file
def _eval_path(run_name: str):
    return EVALUATIONS_DIR / f"manual_eval_{run_name}.json"

#? returns the best available frame image path
def _frame_path(run_name: str, frame_index: int):
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        return None
    stem = f"frame_{frame_index:06d}.jpg"
    annotated = run_dir / "annotated" / stem
    original = run_dir / "frames" / stem
    if annotated.exists():
        return annotated
    if original.exists():
        return original
    return None


#? evaluation state
#? builds a clean assessment entry
def _init_assessment(record: dict, source: str):
    return {
        "frame_index": record["frame_index"],
        "video_timestamp_s": record.get("video_timestamp_s"),
        "source": source,
        "pipeline_action": record.get("action") or record.get("pipeline_action"),
        "correct": None,
        "note": "",
    }

#? loads existing evaluation or builds a fresh one from the run's decision logs
def _load_or_init_eval(run_name: str):
    path = _eval_path(run_name)
    vlm = _load_vlm_decisions(run_name)
    seg = _load_seg_decisions(run_name) if not vlm else []
    decisions = vlm if vlm else seg
    source = "vlm" if vlm else "seg"
    if path.exists():
        ev = json.loads(path.read_text())
        for d in decisions:
            key = str(d["frame_index"])
            if key not in ev["assessments"]:
                ev["assessments"][key] = _init_assessment(d, source)
        path.write_text(json.dumps(ev, indent=2))
        return ev, decisions
    assessments = {str(d["frame_index"]): _init_assessment(d, source) for d in decisions}
    ev = {
        "run_name": run_name,
        "source": source,
        "created_at": datetime.now().isoformat(),
        "assessments": assessments,
    }
    path.write_text(json.dumps(ev, indent=2))
    return ev, decisions

#? persists the evaluation dict to disk
def _save_eval(ev: dict, run_name: str):
    _eval_path(run_name).write_text(json.dumps(ev, indent=2))


#? routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())

@app.route("/api/load/<run_name>")
def api_load(run_name: str):
    ev, decisions = _load_or_init_eval(run_name)
    result = []
    for d in decisions:
        key = str(d["frame_index"])
        assessment = ev["assessments"].get(key, _init_assessment(d, ev["source"]))
        result.append({
            "frame_index": d["frame_index"],
            "video_timestamp_s": d.get("video_timestamp_s"),
            "source": ev["source"],
            #? VLM fields
            "action": d.get("action"),
            "description": d.get("description", ""),
            "obstacle_type": d.get("obstacle_type", ""),
            "movement": d.get("movement", ""),
            "threat": d.get("threat", ""),
            "confidence": d.get("confidence", ""),
            "reasoning": d.get("reasoning", ""),
            "trigger_zone": d.get("trigger_zone", ""),
            #? segmentation fields
            "classes_present": [str(c) for c in d.get("classes_present", [])],
            "detections": d.get("detections", {}),
            #? assessment
            "correct": assessment["correct"],
            "note": assessment.get("note", ""),
        })
    reviewed = sum(1 for a in ev["assessments"].values() if a["correct"] is not None)
    return jsonify({
        "run_name": run_name,
        "source": ev["source"],
        "decisions": result,
        "total": len(result),
        "reviewed": reviewed,
    })

@app.route("/api/frame_image/<run_name>/<int:frame_index>")
def api_frame_image(run_name: str, frame_index: int):
    path = _frame_path(run_name, frame_index)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/jpeg")

@app.route("/api/assess/<run_name>/<int:frame_index>", methods=["POST"])
def api_assess(run_name: str, frame_index: int):
    body = request.get_json()
    correct = body.get("correct")
    note = body.get("note", "")
    if correct not in (True, False, None):
        return jsonify({"error": "correct must be true, false, or null"}), 400
    ev, _ = _load_or_init_eval(run_name)
    key = str(frame_index)
    if key not in ev["assessments"]:
        return jsonify({"error": "Frame not found"}), 404
    ev["assessments"][key]["correct"] = correct
    ev["assessments"][key]["note"] = note
    _save_eval(ev, run_name)
    reviewed = sum(1 for a in ev["assessments"].values() if a["correct"] is not None)
    return jsonify({"ok": True, "reviewed": reviewed, "total": len(ev["assessments"])})

@app.route("/api/stats/<run_name>")
def api_stats(run_name: str):
    path = _eval_path(run_name)
    if not path.exists():
        return jsonify({}), 404
    ev = json.loads(path.read_text())
    correct = sum(1 for a in ev["assessments"].values() if a["correct"] is True)
    wrong = sum(1 for a in ev["assessments"].values() if a["correct"] is False)
    pending = sum(1 for a in ev["assessments"].values() if a["correct"] is None)
    total = len(ev["assessments"])
    reviewed = correct + wrong
    accuracy = correct / reviewed if reviewed > 0 else None
    return jsonify({
        "correct": correct,
        "wrong": wrong,
        "pending": pending,
        "total": total,
        "reviewed": reviewed,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
    })

@app.route("/api/export/<run_name>")
def api_export(run_name: str):
    path = _eval_path(run_name)
    if not path.exists():
        return jsonify({"error": "Evaluation not found"}), 404
    ev = json.loads(path.read_text())
    _, decisions = _load_or_init_eval(run_name)
    dec_map = {str(d["frame_index"]): d for d in decisions}
    rows = sorted(ev["assessments"].values(), key=lambda a: a["frame_index"])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "frame_index", "video_timestamp_s", "source",
        "action", "obstacle_type", "movement", "confidence", "reasoning",
        "correct", "note",
    ])
    for a in rows:
        d = dec_map.get(str(a["frame_index"]), {})
        writer.writerow([
            a["frame_index"],
            a.get("video_timestamp_s", ""),
            a.get("source", ""),
            d.get("action", ""),
            d.get("obstacle_type", ""),
            d.get("movement", ""),
            d.get("confidence", ""),
            d.get("reasoning", ""),
            a["correct"],
            a.get("note", ""),
        ])
    filename = f"manual_eval_{run_name}.csv"
    return Response(
        output.getvalue().encode(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/save_to_thesis/<run_name>", methods=["POST"])
def api_save_to_thesis(run_name: str):
    path = _eval_path(run_name)
    if not path.exists():
        return jsonify({"error": "Evaluation not found"}), 404
    ev = json.loads(path.read_text())
    _, decisions = _load_or_init_eval(run_name)
    dec_map = {str(d["frame_index"]): d for d in decisions}
    rows = sorted(ev["assessments"].values(), key=lambda a: a["frame_index"])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "frame_index", "video_timestamp_s", "source",
        "action", "obstacle_type", "movement", "confidence", "reasoning",
        "correct", "note",
    ])
    for a in rows:
        d = dec_map.get(str(a["frame_index"]), {})
        writer.writerow([
            a["frame_index"],
            a.get("video_timestamp_s", ""),
            a.get("source", ""),
            d.get("action", ""),
            d.get("obstacle_type", ""),
            d.get("movement", ""),
            d.get("confidence", ""),
            d.get("reasoning", ""),
            a["correct"],
            a.get("note", ""),
        ])
    filename = f"manual_eval_{run_name}.csv"
    run_dir = _resolve_run_dir(run_name)
    setup = ""
    if run_dir:
        meta_path = run_dir / "config.json"
        if meta_path.exists():
            setup = json.loads(meta_path.read_text()).get("setup", "")
    if setup == "seg_vlm_dual":
        sub = THESIS_RESULTS_DIR / "vlm dual frame"
    elif setup == "seg_vlm":
        sub = THESIS_RESULTS_DIR / "vlm"
    elif setup == "seg_only":
        sub = THESIS_RESULTS_DIR / "seg only"
    else:
        sub = THESIS_RESULTS_DIR
    sub.mkdir(exist_ok=True)
    dest = sub / filename
    dest.write_text(output.getvalue(), encoding="utf-8")
    return jsonify({"ok": True, "saved_to": str(dest)})

@app.route("/api/reset/<run_name>", methods=["POST"])
def api_reset(run_name: str):
    path = _eval_path(run_name)
    if not path.exists():
        return jsonify({"error": "Evaluation not found"}), 404
    ev = json.loads(path.read_text())
    for key in ev["assessments"]:
        ev["assessments"][key]["correct"] = None
        ev["assessments"][key]["note"] = ""
    _save_eval(ev, run_name)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Results evaluator starting - open http://localhost:5052")
    app.run(host="0.0.0.0", port=5052, debug=False)
