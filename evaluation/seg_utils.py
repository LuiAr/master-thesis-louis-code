#? semantic segmentation helpers using torchvision DeepLabV3

from __future__ import annotations

import logging
from typing import Dict

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

#? zone labels: danger = central driving path, context_left/right = peripheral margins
ZONE_DANGER = "danger"
ZONE_CONTEXT_LEFT = "context_left"
ZONE_CONTEXT_RIGHT = "context_right"

_model_cache = {}

#? loads and returns the segmentation model, cached after the first call
def load_seg_model():
    if "model" in _model_cache:
        return _model_cache["model"]
    model = _load_torch_model()
    _model_cache["model"] = model
    return model

#? loads DeepLabV3 with pretrained COCO weights onto the best available device
def _load_torch_model():
    import torch
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights,
        DeepLabV3_ResNet50_Weights,
        deeplabv3_mobilenet_v3_large,
        deeplabv3_resnet50,
    )
    device = _best_device()
    logger.info("Loading segmentation model '%s' on %s ...", config.SEG_MODEL_NAME, device)
    if config.SEG_MODEL_NAME == "deeplabv3_mobilenet_v3_large":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    elif config.SEG_MODEL_NAME == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown SEG_MODEL_NAME: {config.SEG_MODEL_NAME!r}")
    model.eval()
    model.to(device)
    logger.info("Segmentation model ready.")
    return model

#? returns the best available torch device: MPS > CUDA > CPU
def _best_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


#? preprocessing constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


#? inference
#? runs DeepLabV3 on a BGR frame, returns (mask, classes_present, annotated_frame)
def run_segmentation(frame_bgr: np.ndarray):
    import torch
    from torchvision import transforms

    model = load_seg_model()
    device = next(model.parameters()).device
    h, w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tensor = preprocess(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)["out"]

    mask = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    classes_present = set(np.unique(mask).tolist())
    annotated = _draw_overlay(frame_bgr, mask)
    return mask, classes_present, annotated


#? blur detection
def is_blurry(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(variance < config.BLUR_LAPLACIAN_THRESHOLD), round(float(variance), 2)


#? obstacle detection
def get_obstacle_info(mask: np.ndarray):
    h, w = mask.shape
    bottom_row = int(h * (1.0 - config.OPERATING_ZONE_BOTTOM_EXCLUDE))
    left_col = int(w * config.OPERATING_ZONE_SIDE_MARGIN)
    right_col = int(w * (1.0 - config.OPERATING_ZONE_SIDE_MARGIN))

    zone_slices = {
        ZONE_CONTEXT_LEFT:  mask[:bottom_row, :left_col],
        ZONE_DANGER:        mask[:bottom_row, left_col:right_col],
        ZONE_CONTEXT_RIGHT: mask[:bottom_row, right_col:],
    }
    total_op_pixels = mask[:bottom_row, :].size
    if total_op_pixels == 0:
        return False, False, {}

    detections: Dict[str, dict] = {}
    for class_idx, label in config.SEG_OBSTACLE_CLASSES.items():
        zone_counts = {z: int(np.sum(sl == class_idx)) for z, sl in zone_slices.items()}
        total_count = sum(zone_counts.values())
        fraction = total_count / total_op_pixels
        if fraction < config.SEG_OBSTACLE_MIN_PIXEL_FRACTION:
            continue
        #? danger zone checked first, any object overlapping it by >= threshold is treated as danger
        danger_overlap = zone_counts[ZONE_DANGER] / max(total_count, 1)
        if danger_overlap >= config.SEG_ZONE_OVERLAP_THRESHOLD:
            primary_zone = ZONE_DANGER
        else:
            primary_zone = max(zone_counts, key=lambda z: zone_counts[z])
        detections[label] = {
            "fraction": round(fraction, 5),
            "zone": primary_zone,
            "zone_fractions": {z: round(n / total_op_pixels, 5) for z, n in zone_counts.items()},
        }

    danger_detected = any(d["zone"] == ZONE_DANGER for d in detections.values())
    context_detected = any(d["zone"] in (ZONE_CONTEXT_LEFT, ZONE_CONTEXT_RIGHT) for d in detections.values())
    return danger_detected, context_detected, detections


#? visualisation
#? draws the colour overlay, zone boundaries, and obstacle labels on frames
def _draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45):
    palette = np.array(config.SEG_PALETTE, dtype=np.uint8)
    colour_mask = palette[mask]
    blended = cv2.addWeighted(frame_bgr, 1 - alpha, colour_mask, alpha, 0)

    h, w = frame_bgr.shape[:2]

    bottom_y = int(h * (1.0 - config.OPERATING_ZONE_BOTTOM_EXCLUDE))
    cv2.line(blended, (0, bottom_y), (w, bottom_y), (0, 255, 255), 2)
    cv2.putText(blended, "operating zone", (8, bottom_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    left_x = int(w * config.OPERATING_ZONE_SIDE_MARGIN)
    right_x = int(w * (1.0 - config.OPERATING_ZONE_SIDE_MARGIN))
    cv2.line(blended, (left_x, 0), (left_x, bottom_y), (0, 220, 255), 2)
    cv2.line(blended, (right_x, 0), (right_x, bottom_y), (0, 220, 255), 2)
    cv2.putText(blended, "ctx", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, "ctx", (right_x + 4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, "DANGER", (left_x + 6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return blended

#? overlays status label and per-detection zone and fraction info on the frame
def draw_obstacle_labels(annotated: np.ndarray, detections: dict, danger_detected: bool, context_detected: bool):
    frame = annotated.copy()
    if danger_detected:
        status_text, status_colour = "OBSTACLE - DANGER", (0, 0, 220)
    elif context_detected:
        status_text, status_colour = "OBSTACLE - CONTEXT", (0, 165, 255)
    else:
        status_text, status_colour = "CLEAR", (0, 200, 0)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_colour, 2, cv2.LINE_AA)
    y = 56
    for label, info in detections.items():
        zone_short = {"danger": "DNG", "context_left": "CTX-L", "context_right": "CTX-R"}.get(info["zone"], info["zone"])
        colour = (0, 0, 220) if info["zone"] == ZONE_DANGER else (0, 165, 255)
        cv2.putText(frame, f"{label}: {info['fraction'] * 100:.2f}%  [{zone_short}]", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
        y += 22
    return frame
