#? shared configuration, edit values here to apply them across both pipelines

from pathlib import Path

#? root directories used by all pipeline scripts
RUNS_DIR = Path(__file__).parent / "runs"
RUNS_DIR_SEG_ONLY = RUNS_DIR / "seg_only"
RUNS_DIR_SEG_VLM = RUNS_DIR / "seg_and_vlm"
RUNS_DIR_SEG_VLM_DUAL = RUNS_DIR / "seg_and_vlm_dual"


#? controls how many frames are skipped between processed frames
FRAME_SAMPLE_EVERY = 5

#? excludes the mower body at the bottom of the frame from obstacle detection
OPERATING_ZONE_BOTTOM_EXCLUDE = 0.25

#? side margins as a fraction of frame width, used as context zones in seg_vlm mode
OPERATING_ZONE_SIDE_MARGIN = 0.25

#? name of the torchvision segmentation model to load
SEG_MODEL_NAME = "deeplabv3_mobilenet_v3_large"

#? minimum fraction of operating zone pixels belonging to an obstacle class to flag the frame
SEG_OBSTACLE_MIN_PIXEL_FRACTION = 0.005

#? minimum object pixel overlap fraction needed to assign it to a zone, danger zone checked first
SEG_ZONE_OVERLAP_THRESHOLD = 0.05

#? COCO/PASCAL VOC class indices that count as obstacles
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

#? BGR colour palette for the segmentation overlay, one entry per VOC class 0 to 20
SEG_PALETTE = [
    #? 0 background
    (0,   0,   0),
    #? 1 aeroplane
    (128, 0,   0),
    #? 2 bicycle
    (0,   128, 0),
    #? 3 bird
    (128, 128, 0),
    #? 4 boat
    (0,   0,   128),
    #? 5 bottle
    (128, 0,   128),
    #? 6 bus
    (0,   128, 128),
    #? 7 car
    (128, 128, 128),
    #? 8 cat
    (64,  0,   0),
    #? 9 chair
    (192, 0,   0),
    #? 10 cow
    (64,  128, 0),
    #? 11 dining table
    (192, 128, 0),
    #? 12 dog
    (64,  0,   128),
    #? 13 horse
    (192, 0,   128),
    #? 14 motorbike
    (64,  128, 128),
    #? 15 person
    (192, 128, 128),
    #? 16 potted plant
    (0,   64,  0),
    #? 17 sheep
    (128, 64,  0),
    #? 18 sofa
    (0,   192, 0),
    #? 19 train
    (128, 192, 0),
    #? 20 tv/monitor
    (0,   64,  128),
]

#? blur detection
#? frames below this Laplacian variance threshold are skipped before VLM inference
BLUR_LAPLACIAN_THRESHOLD = 100

#? vlm settings
#? Ollama base URL, set to the remote host when running on the Pi
OLLAMA_BASE_URL = "http://localhost:11434"

#? default VLM model
VLM_MODEL = "gemma3:4b"
#? stronger model used when the user selects high quality mode at runtime
VLM_MODEL_STRONG = "gemma4:e2b"
VLM_TIMEOUT_SECONDS = 120

#? frames to suppress VLM calls after a detection, 0 disables cooldown
VLM_COOLDOWN_FRAMES = 15

#! EXPERIMENTAL
#* pass recent VLM detections as context to the next VLM call 
VLM_USE_CONTEXT_MEMORY = False
VLM_CONTEXT_MEMORY_MAX_FRAMES = 4
#! EXPERIMENTAL

#? output quality
#? JPEG compression level for saved frames
JPEG_QUALITY = 85
