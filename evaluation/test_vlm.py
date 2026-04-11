# Quick one-shot test: send an image to any Ollama VLM and print the raw response.

import base64
import time
import requests

#? ---
#? USER SETTINGS
#? Set the model and image to test, then run: python test_vlm.py
#? ---

MODEL = "llava:7b"
IMAGE_PATH = "recordings/images/Generated Image April 08, 2026 - 2_51PM.jpg"
OLLAMA_URL = "http://localhost:11434"

PROMPT = """\
You are the vision system of an autonomous lawn mower navigating a garden.
A segmentation model detected an obstacle in the {side} peripheral zone of the frame — \
outside the mower's immediate driving path.

Assess whether this obstacle is a future threat to the mower's path. \
Respond in exactly this format - no extra text, no markdown:

DESCRIPTION: <one or two sentences: what is it, where exactly, what is it doing>
OBSTACLE_TYPE: <person | dog | cat | bicycle | car | unknown | other>
MOVEMENT: <stationary | moving | unclear>
IF MOVING: <toward_mower | away_from_mower | left | right | unclear>
ORIENTATION: <facing_mower | facing_away | sideways | unclear>
THREAT: <none | possible | likely>
ACTION: <CONTINUE | STOP | TURN_LEFT | TURN_RIGHT>
REASONING: <one sentence — only recommend STOP or TURN if the obstacle is clearly moving toward the path or is very likely to enter it>
CONFIDENCE: <high | medium | low>
"""

#? ---
#? RUN
#? Encodes the image, calls Ollama, prints the raw response and elapsed time.
#? ---

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": MODEL,
    "prompt": PROMPT,
    "images": [image_b64],
    "stream": False,
    "options": {"temperature": 0},
}

print(f"Model : {MODEL}")
print(f"Image : {IMAGE_PATH}")
print(f"Sending request to {OLLAMA_URL} ...\n")

t0 = time.time()
resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
elapsed = time.time() - t0

resp.raise_for_status()
response_text = resp.json().get("response", "")

print("--- RAW RESPONSE ---")
print(response_text)
print("--------------------")
print(f"\nElapsed: {elapsed:.1f}s")
