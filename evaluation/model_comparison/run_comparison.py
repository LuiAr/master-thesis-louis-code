#? compares VLM models and prompt strategies on a set of images, results saved to results/

from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"

#? full candidate list, the script automatically detects which ones are installed in Ollama and only runs those
#* uncomment the ones to use, during testing different subset has been tested to not have too many at same time
CANDIDATE_MODELS = [
    "gemma3:4b",
    # "richardyoung/smolvlm2-2.2b-instruct",
    "llava-phi3",
    "gemma4:e2b",
    # "llava:7b",
    # "moondream",
    # "qwen3-vl:2b",
    # "granite3.2-vision:2b",
]

#? images to test, paths relative to the evaluation/ folder
IMAGES = [
    "",
]

#? prompt strategies to compare
STRATEGIES = ["full", "describe_only", "rule_based"]

#? number of timed runs per combination, first run is warmup (otherwise VLM inference time is biased), rest are recorded
N_RUNS = 3

#? seconds to wait after unloading a model before loading the next one
UNLOAD_WAIT_S = 60

TIMEOUT = 120

#? prompts
#? three strategies, full asks for an action decision, describe_only does not, rule_based assigns action by code

PROMPTS = {
    "full": """\
You are the vision system of an autonomous lawn mower navigating a garden.
Scan the ENTIRE frame including the sides and edges — do not focus only on the centre.
All living beings (people, animals, pets) are obstacles regardless of their behaviour or apparent intent.
Describe what you see and decide the action.
Respond in exactly this format — no extra text, no markdown, all tokens lowercase:

DESCRIPTION: <one or two sentences: what is it, where exactly in the frame>
OBSTACLE_TYPE: <person | dog | cat | bicycle | car | unknown | none>
MOVEMENT: <stationary | moving | unclear>
IF MOVING: <toward_mower | away_from_mower | left | right | unclear>
ACTION: <STOP | CONTINUE | TURN_LEFT | TURN_RIGHT>
REASONING: <one sentence>
CONFIDENCE: <high | medium | low>
""",

    "describe_only": """\
You are the vision system of an autonomous lawn mower navigating a garden.
Scan the ENTIRE frame including the sides and edges — do not focus only on the centre.
All living beings (people, animals, pets) are obstacles regardless of their behaviour.
Describe what you observe. Do not suggest any action.
Respond in exactly this format — no extra text, no markdown, all tokens lowercase:

DESCRIPTION: <one or two sentences: what is it, where exactly in the frame>
OBSTACLE_TYPE: <person | dog | cat | bicycle | car | unknown | none>
MOVEMENT: <stationary | moving | unclear>
IF MOVING: <toward_mower | away_from_mower | left | right | unclear>
CONFIDENCE: <high | medium | low>
""",

    #? rule_based uses the same prompt as describe_only, action is assigned by _rule_action()
    "rule_based": """\
You are the vision system of an autonomous lawn mower navigating a garden.
Scan the ENTIRE frame including the sides and edges — do not focus only on the centre.
All living beings (people, animals, pets) are obstacles regardless of their behaviour.
Describe what you observe. Do not suggest any action.
Respond in exactly this format — no extra text, no markdown, all tokens lowercase:

DESCRIPTION: <one or two sentences: what is it, where exactly in the frame>
OBSTACLE_TYPE: <person | dog | cat | bicycle | car | unknown | none>
MOVEMENT: <stationary | moving | unclear>
IF MOVING: <toward_mower | away_from_mower | left | right | unclear>
CONFIDENCE: <high | medium | low>
""",
}

REQUIRED_FIELDS = {
    "full":          {"description", "obstacle_type", "movement", "action", "confidence"},
    "describe_only": {"description", "obstacle_type", "movement", "confidence"},
    "rule_based":    {"description", "obstacle_type", "movement", "confidence"},
}

_ALWAYS_STOP = {"person", "dog", "cat"}

#? returns a deterministic action string from VLM description fields
def _rule_action(parsed: dict):
    obs = parsed.get("obstacle_type", "unknown").lower()
    movement = parsed.get("movement", "unclear").lower()
    direction = parsed.get("if_moving", "unclear").lower()

    if obs in _ALWAYS_STOP:
        if movement == "stationary":
            return "STOP"
        if direction == "toward_mower":
            return "STOP"
        if direction in ("left", "right", "away_from_mower"):
            return "CONTINUE"
        return "STOP"

    if obs in ("bicycle", "car", "motorbike"):
        return "STOP"

    if obs == "none":
        return "CONTINUE"

    return "STOP"


#? ollama helpers
#? model loading, unloading, and inference calls

def _encode_image(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _call_vlm(model: str, image_b64: str, prompt: str):
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0},
    }
    t0 = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", ""), round(time.time() - t0, 2)
    except requests.RequestException as exc:
        print(f"    ERROR: {exc}")
        return None, round(time.time() - t0, 2)

#? returns True if a candidate name matches any installed Ollama model name
def _matches_installed(candidate: str, installed: set):
    if candidate in installed:
        return True
    if ":" not in candidate.split("/")[-1] and f"{candidate}:latest" in installed:
        return True
    if ":" in candidate.split("/")[-1]:
        base = candidate.rsplit(":", 1)[0]
        if base in installed or f"{base}:latest" in installed:
            return True
    return False

def _detect_installed_models():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        installed = {m["name"] for m in resp.json().get("models", [])}
        found = [m for m in CANDIDATE_MODELS if _matches_installed(m, installed)]
        missing = [m for m in CANDIDATE_MODELS if not _matches_installed(m, installed)]
        print("Installed candidates:")
        for m in found:
            print(f"  ✓  {m}")
        if missing:
            print("Not installed (skipped):")
            for m in missing:
                print(f"  -  {m}  (ollama pull {m})")
        if not found:
            print("None of the candidate models are installed. Pull at least one and retry.")
        return found
    except requests.RequestException as exc:
        print(f"Could not reach Ollama at {OLLAMA_URL}: {exc}")
        return []

#? unloads a model from Ollama memory
def _unload_model(model: str):
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=10,
        )
        print(f"  Unloaded {model}. Waiting {UNLOAD_WAIT_S}s ...")
        time.sleep(UNLOAD_WAIT_S)
    except requests.RequestException:
        pass


#? response parsing
def _parse(text: str):
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        result[key.strip().lower().replace(" ", "_")] = val.strip()
    return result

#? returns True if all required fields for the strategy are present and non-empty
def _well_formed(parsed: dict, strategy: str):
    return REQUIRED_FIELDS[strategy].issubset({k for k, v in parsed.items() if v})


#? main
def main():
    eval_dir = Path(__file__).parent.parent
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    print(f"\n{'='*72}")
    models = _detect_installed_models()
    if not models:
        return

    print(f"\nImages     : {[Path(i).name for i in IMAGES]}")
    print(f"Strategies : {STRATEGIES}")
    print(f"Runs each  : {N_RUNS} (first is warmup{',' if N_RUNS > 1 else ' , no warmup'} timing = last run)")
    print(f"{'='*72}\n")

    encoded_images = {}
    for image_rel in IMAGES:
        image_path = eval_dir / image_rel
        if not image_path.exists():
            print(f"[SKIP] Image not found: {image_path}")
            continue
        encoded_images[image_rel] = (_encode_image(image_path), image_path.name)

    if not encoded_images:
        print("No valid images found. Exiting.")
        return

    for model_idx, model in enumerate(models):
        print(f"--- Model: {model} ---")

        for image_rel, (image_b64, image_name) in encoded_images.items():
            for strategy in STRATEGIES:
                prompt = PROMPTS[strategy]
                label = f"  {strategy:<15}  {image_name[:28]}"

                raw_last = None
                elapsed_last = None

                for run_num in range(N_RUNS):
                    is_warmup = (run_num == 0 and N_RUNS > 1)
                    tag = "warmup" if is_warmup else f"run {run_num}"
                    print(f"{label}  [{tag}] ...", end=" ", flush=True)

                    raw, elapsed = _call_vlm(model, image_b64, prompt)

                    if raw is None:
                        print(f"TIMEOUT/ERROR ({elapsed}s)")
                        if not is_warmup:
                            raw_last = None
                            elapsed_last = elapsed
                    else:
                        print(f"{elapsed}s")
                        if not is_warmup:
                            raw_last = raw
                            elapsed_last = elapsed

                if N_RUNS == 1:
                    raw_last = raw
                    elapsed_last = elapsed

                if raw_last is None:
                    result = {
                        "model": model, "strategy": strategy, "image": image_name,
                        "prompt": prompt,
                        "elapsed_s": elapsed_last, "success": False,
                        "raw": None, "parsed": {}, "well_formed": False, "action": None,
                        "n_runs": N_RUNS, "timing_note": "error",
                    }
                else:
                    parsed = _parse(raw_last)
                    wf = _well_formed(parsed, strategy)

                    if strategy == "rule_based":
                        action = _rule_action(parsed)
                        parsed["action_rule"] = action
                    else:
                        action = parsed.get("action", "—").upper()

                    timing_note = "last of N runs" if N_RUNS > 1 else "single run"
                    result = {
                        "model": model, "strategy": strategy, "image": image_name,
                        "prompt": prompt,
                        "elapsed_s": elapsed_last, "success": True,
                        "raw": raw_last, "parsed": parsed, "well_formed": wf, "action": action,
                        "n_runs": N_RUNS, "timing_note": timing_note,
                    }

                all_results.append(result)

        if model_idx < len(models) - 1:
            _unload_model(model)

        print()

    _print_summary(all_results)

    out_path = results_dir / f"comparison_{run_id}.json"
    out_path.write_text(json.dumps({"run_id": run_id, "results": all_results}, indent=2))
    print(f"\nFull results saved to: {out_path.relative_to(Path(__file__).parent.parent)}")


#? prints a compact summary table grouped by model and strategy
def _print_summary(results: list):
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  {'model':<22}  {'strategy':<15}  {'elapsed':>8}  {'wf':>4}  action")
    print(f"  {'-'*65}")
    for r in results:
        wf = "yes" if r["well_formed"] else ("no" if r["success"] else "err")
        elapsed = f"{r['elapsed_s']:.1f}s" if r["elapsed_s"] is not None else "—"
        action = r["action"] or "—"
        print(f"  {r['model']:<22}  {r['strategy']:<15}  {elapsed:>8}  {wf:>4}  {action}")


if __name__ == "__main__":
    main()
