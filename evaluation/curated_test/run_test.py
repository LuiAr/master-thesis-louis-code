# Curated test runner: runs the VLM on every image in images/, scores against ground_truth.csv, writes results.

import sys
import csv
import json
import base64
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

#? ---
#? PATHS
#? Curated_test folder layout resolved from this script's location.
#? ---

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
GROUND_TRUTH_PATH = ROOT / "ground_truth.csv"

VALID_ACTIONS = {"CONTINUE", "STOP", "REROUTE"}

#? ---
#? VLM CALL
#? Sends an image and prompt to Ollama and parses the structured response with REROUTE support.
#? ---

# Encodes a BGR frame to a base64 JPEG string for the Ollama payload
def frame_to_b64(frame_bgr):
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# Parses the structured key:value VLM response and validates the action against the new schema
def parse_vlm_response(text):
    result = {
        "description": "",
        "obstacle_type": "unknown",
        "movement": "unclear",
        "if_moving": "",
        "action": "STOP",
        "reasoning": "",
        "confidence": "low",
    }
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            result["description"] = line[len("DESCRIPTION:"):].strip()
        elif line.startswith("OBSTACLE_TYPE:"):
            result["obstacle_type"] = line[len("OBSTACLE_TYPE:"):].strip().lower()
        elif line.startswith("MOVEMENT:"):
            result["movement"] = line[len("MOVEMENT:"):].strip().lower()
        elif line.startswith("IF MOVING:"):
            result["if_moving"] = line[len("IF MOVING:"):].strip().lower()
        elif line.startswith("ACTION:"):
            val = line[len("ACTION:"):].strip().upper()
            if val in VALID_ACTIONS:
                result["action"] = val
        elif line.startswith("REASONING:"):
            result["reasoning"] = line[len("REASONING:"):].strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line[len("CONFIDENCE:"):].strip().lower()
    return result

# Calls Ollama with the image and prompt, returns the parsed dict and elapsed time in seconds
def call_vlm(frame_bgr, prompt, model_name, ollama_url):
    image_b64 = frame_to_b64(frame_bgr)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
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
        print(f"  VLM request failed: {exc}")
        return None, time.time() - t0
    elapsed = time.time() - t0
    raw_text = resp.json().get("response", "")
    parsed = parse_vlm_response(raw_text)
    parsed["raw_text"] = raw_text
    return parsed, elapsed

#? ---
#? INTERACTIVE SELECTION
#? Prompts the user at runtime for the model, prompt variant, and run label.
#? ---

# Asks the user to choose between standard and strong VLM
def select_model():
    print("")
    print("Select VLM:")
    print(f"  [1] Standard - {config.VLM_MODEL}")
    print(f"  [2] Strong   - {config.VLM_MODEL_STRONG}")
    print("")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return config.VLM_MODEL
        if choice == "2":
            return config.VLM_MODEL_STRONG
        print("Invalid input.")

# Lists the .txt files in prompts/ and asks the user to pick one
def select_prompt():
    files = sorted(p for p in PROMPTS_DIR.iterdir() if p.suffix == ".txt")
    if not files:
        raise FileNotFoundError(f"No prompt .txt files in {PROMPTS_DIR}")
    print("")
    print("Select prompt variant:")
    for i, p in enumerate(files, 1):
        print(f"  [{i}] {p.name}")
    print("")
    while True:
        choice = input(f"Enter 1 to {len(files)}: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("Invalid input.")

# Asks the user for an optional short run label, returns the cleaned string or empty
def prompt_run_label():
    label = input("Optional short label for this run (press enter to skip): ").strip()
    return label

#? ---
#? GROUND TRUTH AND SCORING
#? Loads expected actions and classifies wrong predictions as safe or unsafe.
#? ---

# Loads ground_truth.csv into a dict keyed by image filename stem (so .jpg or .png both match)
def load_ground_truth():
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Missing {GROUND_TRUTH_PATH}")
    truth = {}
    with open(GROUND_TRUTH_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stem = Path(row["filename"]).stem
            truth[stem] = row
    return truth

# Classifies a prediction outcome: correct, unsafe (CONTINUE when STOP/REROUTE expected), or safe (any other mismatch)
def classify_outcome(expected, predicted):
    if predicted == "ERROR":
        return "error"
    if expected == predicted:
        return "correct"
    if predicted == "CONTINUE" and expected in {"STOP", "REROUTE"}:
        return "unsafe"
    return "safe"

#? ---
#? OUTPUT
#? Builds the run output directory and writes results.csv and summary.json.
#? ---

# Builds the timestamped per-run output directory under results/
def make_run_dir(model_name, prompt_path, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(":", "-").replace("/", "-")
    parts = [timestamp, safe_model, prompt_path.stem]
    if label:
        parts.append(label.replace(" ", "_"))
    out_dir = RESULTS_DIR / "_".join(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# Writes the per-image results CSV
def write_results_csv(out_dir, rows):
    if not rows:
        return
    fieldnames = [
        "filename",
        "expected_action",
        "predicted_action",
        "outcome",
        "elapsed_s",
        "obstacle_type",
        "movement",
        "if_moving",
        "confidence",
        "reasoning",
        "description",
        "raw_text",
    ]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

# Writes the aggregate summary JSON: confusion matrix, accuracy, safe vs unsafe error counts
def write_summary_json(out_dir, rows, model_name, prompt_name, label):
    actions = ["CONTINUE", "STOP", "REROUTE"]
    matrix = {a: {b: 0 for b in actions + ["ERROR"]} for a in actions}
    correct = 0
    safe_errors = 0
    unsafe_errors = 0
    request_errors = 0
    for r in rows:
        exp = r["expected_action"]
        pred = r["predicted_action"]
        if exp in matrix and pred in matrix[exp]:
            matrix[exp][pred] += 1
        outcome = r["outcome"]
        if outcome == "correct":
            correct += 1
        elif outcome == "unsafe":
            unsafe_errors += 1
        elif outcome == "safe":
            safe_errors += 1
        elif outcome == "error":
            request_errors += 1
    total = len(rows)
    summary = {
        "model": model_name,
        "prompt": prompt_name,
        "label": label,
        "total_images": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "safe_errors": safe_errors,
        "unsafe_errors": unsafe_errors,
        "request_errors": request_errors,
        "confusion_matrix": matrix,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

#? ---
#? MAIN
#? Iterates over the curated images, runs the VLM, scores against ground truth, writes files.
#? ---

# Main entry point
def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing {IMAGES_DIR}")
    truth = load_ground_truth()
    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No images in {IMAGES_DIR}")

    model_name = select_model()
    prompt_path = select_prompt()
    label = prompt_run_label()
    prompt_text = prompt_path.read_text(encoding="utf-8")

    out_dir = make_run_dir(model_name, prompt_path, label)
    print(f"\nWriting results to: {out_dir}\n")

    rows = []
    for img_path in images:
        fname = img_path.name
        truth_row = truth.get(img_path.stem)
        if truth_row is None:
            print(f"Skipping {fname}: no ground truth entry")
            continue
        expected = truth_row.get("expected_action", "").strip().upper()
        if not expected:
            print(f"Skipping {fname}: empty expected_action")
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Skipping {fname}: could not read image")
            continue
        print(f"Running {fname} ...")
        parsed, elapsed = call_vlm(frame, prompt_text, model_name, config.OLLAMA_BASE_URL)
        if parsed is None:
            predicted = "ERROR"
            row = {
                "filename": fname,
                "expected_action": expected,
                "predicted_action": predicted,
                "outcome": classify_outcome(expected, predicted),
                "elapsed_s": round(elapsed, 2),
                "obstacle_type": "",
                "movement": "",
                "if_moving": "",
                "confidence": "",
                "reasoning": "",
                "description": "",
                "raw_text": "",
            }
        else:
            predicted = parsed["action"]
            row = {
                "filename": fname,
                "expected_action": expected,
                "predicted_action": predicted,
                "outcome": classify_outcome(expected, predicted),
                "elapsed_s": round(elapsed, 2),
                "obstacle_type": parsed.get("obstacle_type", ""),
                "movement": parsed.get("movement", ""),
                "if_moving": parsed.get("if_moving", ""),
                "confidence": parsed.get("confidence", ""),
                "reasoning": parsed.get("reasoning", ""),
                "description": parsed.get("description", ""),
                "raw_text": parsed.get("raw_text", ""),
            }
        rows.append(row)

    write_results_csv(out_dir, rows)
    write_summary_json(out_dir, rows, model_name, prompt_path.name, label)
    print(f"\nDone. {len(rows)} images processed. Results in {out_dir}")


if __name__ == "__main__":
    main()
