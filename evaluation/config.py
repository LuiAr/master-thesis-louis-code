# Shared configuration — edit values here to apply them across both pipelines.

import os

#? ---
#? PATHS
#? Root directories used by all pipeline scripts.
#? ---

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


#? ---
#? FRAME SAMPLING
#? Controls how many frames are skipped between processed frames.
#? ---

# Process every Nth frame (1 = every frame).
# 5 at 30 fps gives ~6 effective fps — fast enough while keeping run times short.
FRAME_SAMPLE_EVERY = 5

#? ---
#? OPERATING ZONE
#? Excludes the mower body at the bottom of the frame from obstacle detection.
#? ---

# Strip the bottom fraction of each frame before checking for obstacles.
OPERATING_ZONE_BOTTOM_EXCLUDE = 0.25

# Left and right margin as a fraction of frame width.
# In seg_only: obstacles whose majority pixels fall in these margins are fully ignored.
# In seg_vlm:  the same margins become "context zones" — the VLM is still called but
#              with a different prompt focused on movement direction and threat trajectory.
OPERATING_ZONE_SIDE_MARGIN = 0.25

#? ---
#? SEGMENTATION MODEL
#? DeepLabV3 with MobileNetV3-Large backbone, COCO pretrained (21 VOC classes).
#? ---

# Swap to "deeplabv3_resnet50" for higher accuracy at the cost of speed.
SEG_MODEL_NAME = "deeplabv3_mobilenet_v3_large"

# Minimum fraction of operating-zone pixels belonging to an obstacle class to flag the frame.
SEG_OBSTACLE_MIN_PIXEL_FRACTION = 0.005

# Minimum fraction of an object's own pixels that must overlap a zone for it to be assigned to that zone.
# Danger zone is checked first — if >= this fraction of the object's pixels are in the danger zone,
# it is classified as danger even if the majority of the object is in a context zone.
# Example: 0.05 means 5% of the object touching the danger zone is enough to trigger DANGER.
SEG_ZONE_OVERLAP_THRESHOLD = 0.05

# COCO/PASCAL VOC class indices that count as obstacles.
SEG_OBSTACLE_CLASSES = {
    2:  "bicycle",
    3:  "bird",
    6:  "bus",
    7:  "car",
    8:  "cat",
    10: "cow",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    17: "sheep",
}

# BGR colour palette for the segmentation overlay, one entry per VOC class 0-20.
SEG_PALETTE = [
    (0,   0,   0),    # 0  background
    (128, 0,   0),    # 1  aeroplane
    (0,   128, 0),    # 2  bicycle
    (128, 128, 0),    # 3  bird
    (0,   0,   128),  # 4  boat
    (128, 0,   128),  # 5  bottle
    (0,   128, 128),  # 6  bus
    (128, 128, 128),  # 7  car
    (64,  0,   0),    # 8  cat
    (192, 0,   0),    # 9  chair
    (64,  128, 0),    # 10 cow
    (192, 128, 0),    # 11 dining table
    (64,  0,   128),  # 12 dog
    (192, 0,   128),  # 13 horse
    (64,  128, 128),  # 14 motorbike
    (192, 128, 128),  # 15 person
    (0,   64,  0),    # 16 potted plant
    (128, 64,  0),    # 17 sheep
    (0,   192, 0),    # 18 sofa
    (128, 192, 0),    # 19 train
    (0,   64,  128),  # 20 tv/monitor
]

#? ---
#? BLUR DETECTION
#? Laplacian variance threshold — frames below this are skipped before VLM inference.
#? ---

# Frames with a Laplacian variance below this value are considered too blurry to analyse.
# 100 is a reasonable starting point — lower = more tolerant, higher = stricter.
BLUR_LAPLACIAN_THRESHOLD = 100

#? ---
#? VLM SETTINGS
#? Controls the VLM endpoint and model used in Setup 2.
#? ---

# Ollama endpoint. On the Pi set this to http://Luis-MacBook-Pro.local:11434.
OLLAMA_BASE_URL = "http://localhost:11434"
VLM_MODEL = "gemma3:4b"
VLM_TIMEOUT_SECONDS = 60

# Experimental: pass recent VLM detections as context to the next VLM call.
VLM_USE_CONTEXT_MEMORY = False
VLM_CONTEXT_MEMORY_MAX_FRAMES = 4

#? ---
#? OUTPUT QUALITY
#? JPEG compression level for saved frames.
#? ---

JPEG_QUALITY = 85
