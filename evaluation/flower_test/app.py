# Web app: upload two images, analyse both with VLM, show side-by-side reasoning and decision

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as pipeline_config


#? ---
#? CONFIGURATION
#? Model references and VLM endpoint
#? ---

OLLAMA_URL = pipeline_config.OLLAMA_BASE_URL
VLM_MODEL = pipeline_config.VLM_MODEL_STRONG
SEGFORMER_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"

VEGETATION_KEYWORDS = ["grass", "plant", "flower", "shrub", "tree", "vegetation", "bush"]
OVERLAY_EXCLUDE_KEYWORDS = {"grass", "tree", "flower"}
VEGETATION_COLOURS = {
    "flower": (255, 80, 180),
    "plant": (50, 200, 80),
    "shrub": (100, 160, 50),
    "bush": (100, 160, 50),
    "tree": (30, 120, 30),
    "grass": (140, 220, 100),
    "vegetation": (80, 180, 80),
}

RESPONSE_FIELDS = ["OBSERVATION", "PLANT_TYPE", "REASONING", "DECISION", "CONFIDENCE"]

VLM_PROMPT = (
    "You are the vision system of an autonomous robotic lawnmower. "
    "You are looking at a camera image from the mower's perspective.\n\n"
    "FOCUS: Direct your entire analysis at the plant or vegetation visible in the CENTRE of the image. "
    "Ignore background elements, buildings, or vegetation at the edges. "
    "Your decision must be based solely on what is growing in the centre of the frame.\n\n"
    "Your task is to inspect that central plant and decide whether the mower should stop or continue.\n\n"
    "CRITICAL DISTINCTION - you must reason explicitly about this difference:\n\n"
    "1. ORNAMENTAL / PLANTED FLOWERS: Plants that have been deliberately placed in a garden, "
    "such as roses, tulips, daisies, lavender, or any other cultivated ornamental flower. "
    "These have structured petals with distinct colouring, grow in defined clusters or beds, "
    "and are clearly not random wild vegetation. "
    "If any planted or ornamental flowers are present in the centre, the mower MUST STOP.\n\n"
    "2. TALL GRASS / WEEDS / WILD VEGETATION: Grass that has grown too long, wild weeds, "
    "or any unstructured overgrown vegetation spreading across the lawn surface. "
    "These lack defined petal structures, grow randomly, and are exactly what the mower is designed to cut. "
    "If only this type of vegetation is present in the centre, the mower should CONTINUE.\n\n"
    "Respond using exactly this format and no other text:\n\n"
    "OBSERVATION: [Describe precisely the plant in the centre of the image - colours, petal structure, arrangement, height]\n"
    "PLANT_TYPE: [ORNAMENTAL_FLOWER | TALL_GRASS | MIXED | NONE]\n"
    "REASONING: [Step-by-step explanation focused on the central plant - why it is a planted ornamental flower "
    "or just grass/weeds. Refer to specific visual clues such as petal definition, colour contrast, "
    "clustered arrangement, and signs of deliberate cultivation versus wild growth]\n"
    "DECISION: [STOP | CONTINUE]\n"
    "CONFIDENCE: [HIGH | MEDIUM | LOW]"
)


#? ---
#? MODEL STATE
#? Global references to the segmentation model, loaded once at startup
#? ---

_seg_extractor = None
_seg_model = None
_seg_id2label = None
_seg_available = False


#? ---
#? SEGMENTATION
#? Loads SegFormer once and provides helpers to segment and overlay images
#? ---

# Loads the SegFormer model into globals; sets _seg_available to False if unavailable
def load_segformer():
    global _seg_extractor, _seg_model, _seg_id2label, _seg_available
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        print(f"  Loading segmentation model ({SEGFORMER_MODEL}) ...")
        _seg_extractor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
        _seg_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
        _seg_model.eval()
        _seg_id2label = _seg_model.config.id2label
        _seg_available = True
        print("  Segmentation model ready.")
    except Exception as e:
        print(f"  Segmentation model unavailable: {e}")
        _seg_available = False

# Runs SegFormer on a PIL image and returns (label_map, id2label) or (None, None) if unavailable
def _run_segmentation(pil_image):
    if not _seg_available:
        return None, None
    inputs = _seg_extractor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = _seg_model(**inputs)
    upsampled = torch.nn.functional.interpolate(
        outputs.logits,
        size=(pil_image.size[1], pil_image.size[0]),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze().numpy(), _seg_id2label

# Returns a PIL image with flower/plant overlay applied (grass and tree excluded)
def _build_overlay(pil_image, label_map, id2label):
    img_array = np.array(pil_image).copy()
    overlay = img_array.copy()
    for class_id in np.unique(label_map):
        name = id2label.get(int(class_id), "").lower()
        for keyword in VEGETATION_KEYWORDS:
            if keyword in name:
                if keyword in OVERLAY_EXCLUDE_KEYWORDS:
                    break
                colour = VEGETATION_COLOURS.get(keyword, (0, 200, 0))
                overlay[label_map == class_id] = colour
                break
    blended = (img_array * 0.55 + overlay * 0.45).astype(np.uint8)
    return Image.fromarray(blended)

# Encodes a PIL image as a base64 JPEG string
def _encode_pil(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


#? ---
#? VLM
#? Sends an image to Ollama and parses the structured response
#? ---

# Sends the image to the VLM and returns the raw response text
def _query_vlm(pil_image):
    payload = {
        "model": VLM_MODEL,
        "prompt": VLM_PROMPT,
        "images": [_encode_pil(pil_image)],
        "stream": False,
    }
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=pipeline_config.VLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")

# Parses structured VLM output into a dict of field -> value
def _parse_response(text):
    result = {}
    for field in RESPONSE_FIELDS:
        marker = f"{field}:"
        if marker not in text:
            continue
        start = text.index(marker) + len(marker)
        end = len(text)
        for other in RESPONSE_FIELDS:
            other_marker = f"{other}:"
            if other_marker != marker and other_marker in text:
                pos = text.index(other_marker)
                if pos > start:
                    end = min(end, pos)
        result[field] = text[start:end].strip()
    return result


#? ---
#? ANALYSIS PIPELINE
#? Runs full segmentation + VLM pipeline on a single image and returns result dict
#? ---

# Runs segmentation overlay and VLM on one image; returns dict with base64 image and parsed fields
def analyse_image(pil_image):
    label_map, id2label = _run_segmentation(pil_image)
    if label_map is not None:
        display_image = _build_overlay(pil_image, label_map, id2label)
    else:
        display_image = pil_image
    vlm_raw = _query_vlm(pil_image)
    parsed = _parse_response(vlm_raw)
    return {
        "image_b64": _encode_pil(display_image),
        "observation": parsed.get("OBSERVATION", ""),
        "reasoning": parsed.get("REASONING", ""),
        "decision": parsed.get("DECISION", "UNKNOWN"),
        "confidence": parsed.get("CONFIDENCE", ""),
    }


#? ---
#? ROUTES
#? Flask routes serving the UI and analysis API
#? ---

app = Flask(__name__)

# Serves the main comparison UI
@app.route("/")
def index():
    return render_template("index.html")

# Accepts two uploaded images, runs analysis on each, and returns combined JSON results
@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    file_a = request.files.get("image_a")
    file_b = request.files.get("image_b")
    if not file_a or not file_b:
        return jsonify({"error": "Both images are required"}), 400
    try:
        img_a = Image.open(file_a.stream).convert("RGB")
        img_b = Image.open(file_b.stream).convert("RGB")
        result_a = analyse_image(img_a)
        result_b = analyse_image(img_b)
        return jsonify({"a": result_a, "b": result_b})
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Could not connect to Ollama at {OLLAMA_URL}. Make sure it is running."}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    load_segformer()
    print("Flower comparison app starting - open http://localhost:5053")
    app.run(host="0.0.0.0", port=5053, debug=False)
