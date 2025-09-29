# enroll.py
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils import l2_normalize, save_json

FACES_DIR = "faces"
DATA_DIR = "data"
EMB_PATH = os.path.join(DATA_DIR, "arcface_encodings.json")
NAMES_PATH = os.path.join(DATA_DIR, "arcface_roll_numbers.json")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # CPU by default (Codespaces-friendly). Switch to CUDAExecutionProvider if you have GPU.
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    student_dirs = [
        os.path.join(FACES_DIR, d) for d in os.listdir(FACES_DIR)
        if os.path.isdir(os.path.join(FACES_DIR, d))
    ]

    if not student_dirs:
        print(f"⚠️ Put student folders with images inside ./{FACES_DIR}")
        return

    encodings = []
    names = []

    for sdir in sorted(student_dirs):
        student_name = os.path.basename(sdir)
        img_files = [os.path.join(sdir, f) for f in os.listdir(sdir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if len(img_files) < 3:
            print(f"⚠️ {student_name}: found {len(img_files)} images (<3). Add more for better accuracy.")
        if not img_files:
            print(f"❌ {student_name}: no images found, skipping.")
            continue

        emb_list = []

        for p in img_files:
            img = cv2.imread(p)
            if img is None:
                print(f"   - Cannot read {p}, skipping.")
                continue

            faces = app.get(img)
            if not faces:
                print(f"   - No face in {os.path.basename(p)}, skipping.")
                continue

            # pick the largest face in case there are multiple
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            emb = l2_normalize(f.embedding)
            emb_list.append(emb)

        if not emb_list:
            print(f"❌ {student_name}: no usable faces across images, skipping.")
            continue

        # average then re-normalize → single centroid embedding per student
        mean_emb = l2_normalize(np.mean(np.stack(emb_list, axis=0), axis=0))
        encodings.append(mean_emb.tolist())
        names.append(student_name)
        print(f"✅ Enrolled {student_name} with {len(emb_list)} face(s)")

    save_json(EMB_PATH, encodings)
    save_json(NAMES_PATH, names)
    print(f"💾 Saved {len(names)} students to {DATA_DIR}/")

if __name__ == "__main__":
    main()
