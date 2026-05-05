# VLM-Based Obstacle Detection for Robotic Lawnmowers

Master thesis project — Louis Arbey, Redfield

This repository contains the full implementation and evaluation of a vision system for autonomous robotic lawnmowers. The system combines semantic segmentation (DeepLabV3) with a Vision-Language Model (VLM via Ollama) to detect and classify obstacles in camera frames and decide whether the mower should stop, continue, or turn.

## Repository structure

```
data_collection/    camera server running on the Raspberry Pi mounted on the mower
evaluation/         pipeline code, config, tooling, and results
```

## Quick start

**Requirements**: Python 3.10+, [Ollama](https://ollama.com) running locally with a vision model pulled.

```bash
cd evaluation
pip install -r requirements.txt
```

Pull the VLM model used by the pipeline:

```bash
ollama pull gemma3:4b
```

Edit `evaluation/config.py` to set your Ollama endpoint and any pipeline parameters, then run a pipeline:

```bash
python pipeline_seg_only.py   # segmentation-only baseline
python pipeline_seg_vlm.py    # segmentation + VLM
```

Both scripts prompt interactively at runtime — no arguments needed. See `evaluation/README.md` for the full guide.

## Extras

- `evaluation/flower_test/` — standalone demo: upload two plant images and compare VLM decisions side by side. Requires additional dependencies listed in `evaluation/flower_test/requirements.txt`.
- `evaluation/model_comparison/` — scripts used during model selection to benchmark candidate VLMs against each other.
- `data_collection/` — camera server (Flask + OpenCV) for streaming and recording from the Pi. See `data_collection/README.md`.
