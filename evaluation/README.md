# Evaluation Pipelines

Two offline evaluation setups for the autonomous lawn mower vision pipeline,
running on pre-recorded video from the robot's camera.

---

## Setup 1 — Segmentation only (baseline)

Runs DeepLabV3 semantic segmentation on every Nth frame. Flags frames where
obstacle-class pixels (person, dog, cat, bicycle, car, …) appear in the
operating zone (frame minus bottom 20 % where the mower body is visible).

```bash
cd evaluation
python pipeline_seg_only.py /path/to/recording.mp4
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--run-name` | auto timestamp | Name of the output folder under `runs/` |
| `--every N` | 5 (config) | Process every Nth frame |
| `--no-annotated` | off | Skip saving annotated frames (faster) |

---

## Setup 2 — Segmentation + VLM (proposed method)

Same segmentation step as Setup 1. When an obstacle is detected, the frame is
forwarded to a VLM (qwen2.5vl:3b via Ollama) which returns a description and a
recommended action (STOP / CONTINUE / TURN\_LEFT / TURN\_RIGHT).

**Prerequisites:** Ollama must be running with the model pulled.

```bash
ollama pull qwen2.5vl:3b
ollama serve          # keeps running in the background
```

Then run the pipeline:

```bash
cd evaluation
python pipeline_seg_vlm.py /path/to/recording.mp4
```

Additional options beyond Setup 1:

| Flag | Description |
|---|---|
| `--context-memory` | Enable experimental rolling VLM context (last 4 detections prepended to prompt) |
| `--ollama-url URL` | Override Ollama base URL (default: `http://localhost:11434`) |

For Pi-to-MacBook offloading, pass `--ollama-url http://Luis-MacBook-Pro.local:11434`.

---

## Web Viewer

Frame-by-frame browser UI for reviewing run output.

```bash
cd evaluation/viewer
python app.py
```

Open **http://localhost:5050** — select a run from the sidebar, navigate with
arrow keys or the filmstrip. Toggle between the original frame and the
segmentation overlay. VLM responses (action, description, confidence) are shown
in the detail panel for frames that triggered the VLM.

---

## Installation

```bash
cd evaluation
pip install -r requirements.txt
```

On Apple Silicon, PyTorch will automatically use the MPS backend. On CPU-only
machines the pipeline still works but segmentation inference will be slower.

---

## Output structure

Each run produces a folder under `evaluation/runs/<run-name>/`:

```
runs/my_run/
├-- config.json          metadata (model, video path, thresholds, …)
├-- summary.json         aggregate stats (obstacle rate, VLM call count, …)
├-- frames/              original JPEG frames
├-- annotated/           frames with segmentation colour overlay
└-- logs/
    ├-- seg_results.jsonl    one JSON record per processed frame
    └-- vlm_results.jsonl    one JSON record per VLM-triggered frame (Setup 2)
```

Each line in `seg_results.jsonl`:

```json
{
  "frame_index": 150,
  "video_timestamp_s": 5.0,
  "obstacle_detected": true,
  "pixel_fractions": {"person": 0.03412},
  "classes_present": [0, 15],
  "seg_time_s": 0.21
}
```

Each line in `vlm_results.jsonl`:

```json
{
  "frame_index": 150,
  "video_timestamp_s": 5.0,
  "pixel_fractions": {"person": 0.03412},
  "vlm_time_s": 4.8,
  "description": "A person is standing in the centre of the frame facing the camera.",
  "obstacle_type": "person",
  "movement": "stationary",
  "action": "STOP",
  "reasoning": "A person directly in the mower's path requires an immediate stop.",
  "confidence": "high",
  "raw_text": "..."
}
```

---

## Config

All tuneable parameters are in `evaluation/config.py`. Key values:

| Parameter | Default | Effect |
|---|---|---|
| `FRAME_SAMPLE_EVERY` | 5 | Frame sampling rate |
| `OPERATING_ZONE_BOTTOM_EXCLUDE` | 0.20 | Bottom fraction excluded from detection |
| `SEG_OBSTACLE_MIN_PIXEL_FRACTION` | 0.005 | Min pixel coverage to flag an obstacle |
| `VLM_USE_CONTEXT_MEMORY` | False | Rolling memory across frames |
| `VLM_CONTEXT_MEMORY_MAX_FRAMES` | 4 | How many past detections to pass as context |
| `OLLAMA_BASE_URL` | localhost:11434 | Ollama endpoint |
