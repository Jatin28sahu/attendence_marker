# utils.py
import os
import json
import re
import numpy as np

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def cosine_sim_norm(a: np.ndarray, b: np.ndarray) -> float:
    # assumes a and b are already L2-normalized
    return float(np.dot(a, b))

# NEW: safe filename helper
def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

# NEW: crop helper with optional padding (in pixels)
def crop_face(img, bbox, pad: int = 10):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]
