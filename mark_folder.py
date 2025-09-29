# mark_folder.py
import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from insightface.app import FaceAnalysis
from utils import (
    l2_normalize, load_json, cosine_sim_norm,
    crop_face, safe_name
)

DATA_DIR = "data"
OUT_DIR = "out"
CROPS_DIR = os.path.join(OUT_DIR, "crops")
EMB_PATH = os.path.join(DATA_DIR, "arcface_encodings.json")
NAMES_PATH = os.path.join(DATA_DIR, "arcface_roll_numbers.json")

def main():
    if len(sys.argv) < 2:
        print("Usage: python mark_folder.py path/to/photos_folder [threshold]")
        print("Example: python mark_folder.py photos 0.3")
        sys.exit(1)

    photos_dir = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    if not (os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH)):
        print("❌ Enrollment data not found. Run: python enroll.py")
        sys.exit(1)

    # Load known encodings and names
    known_enc = [np.array(x, dtype=np.float32) for x in load_json(EMB_PATH)]
    names = load_json(NAMES_PATH)

    # Collect all image files in photos_dir
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    photo_paths = []
    for p in patterns:
        photo_paths.extend(glob.glob(os.path.join(photos_dir, p)))
    photo_paths = sorted(photo_paths)

    if not photo_paths:
        print(f"⚠️ No class photos found in: {photos_dir}")
        sys.exit(0)

    # Init face app (CPU provider is Codespaces-friendly)
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)

    rows = []
    already_marked = set()  # ensures each person is marked once across ALL photos
    crop_counter = {}

    time_str = datetime.now().strftime("%H:%M:%S")
    date_str = datetime.now().strftime("%d-%m-%Y")

    print(f"🖼️ Found {len(photo_paths)} photo(s) in {photos_dir}")
    for photo_path in photo_paths:
        img = cv2.imread(photo_path)
        if img is None:
            print(f"❌ Cannot read {photo_path}, skipping.")
            continue

        faces = app.get(img)
        print(f"📷 {os.path.basename(photo_path)} → detected {len(faces)} face(s).")

        for f in faces:
            emb = l2_normalize(f.embedding)

            # cosine on normalized vectors = dot product
            scores = np.array([cosine_sim_norm(emb, k) for k in known_enc], dtype=np.float32)
            j = int(np.argmax(scores))
            best = float(scores[j])

            # Decide name (or Unknown)
            if best >= threshold:
                person = names[j]
            else:
                person = "Unknown"

            # Save crop for new marks and unknowns
            crop = crop_face(img, f.bbox, pad=10)
            if crop is not None:
                key = safe_name(person)
                crop_counter[key] = crop_counter.get(key, 0) + 1
                crop_name = f"{key}_{crop_counter[key]:03d}.jpg"
                cv2.imwrite(os.path.join(CROPS_DIR, crop_name), crop)

            # Only add attendance once per person across the whole folder
            if person != "Unknown" and person not in already_marked and best >= threshold:
                already_marked.add(person)
                rows.append([person, time_str, date_str, round(best, 4)])
                print(f"✅ {person} marked present (sim={best:.4f})")
            elif person != "Unknown" and person in already_marked:
                # Skip marking again per your requirement
                print(f"↩️  {person} already present — skipping re-mark.")
            else:
                print(f"❌ Unknown (sim={best:.4f})")

    out_csv = os.path.join(OUT_DIR, "Attendance_ArcFace.csv")
    pd.DataFrame(rows, columns=["Name", "Time", "Date", "Similarity"]).to_csv(out_csv, index=False)
    print(f"💾 Saved → {out_csv}")
    print(f"🖼️ Crops saved in → {CROPS_DIR}")

if __name__ == "__main__":
    main()
