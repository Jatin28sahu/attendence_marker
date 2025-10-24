from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
import os
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from database import init_db, save_student, get_students, save_attendance ,delete_student_by_name,delete_class_data
from utils import l2_normalize
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "faces")
ATTENDANCE_CROPS_DIR = os.path.join(DATA_DIR, "attendance_crops")
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_CROPS_DIR, exist_ok=True)

# Initialize face analysis app
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize database
init_db()

@app.post("/enroll/")
async def enroll_students(
    class_name: str = Form(...),
    section: str = Form(...),
    subject: Optional[str] = Form(None),
    faces_zip: UploadFile = File(...)
):
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    zip_path = os.path.join(temp_dir, "upload.zip")
    
    # Save and extract zip file
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(faces_zip.file, f)
    
    # Extract to faces directory with class name and section
    class_dir = os.path.join(FACES_DIR, f"{class_name}_{section}")
    os.makedirs(class_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(class_dir)
    
    enrolled = []
    # Process each student directory
    for student_dir in os.listdir(class_dir):
        full_student_path = os.path.join(class_dir, student_dir)
        if not os.path.isdir(full_student_path):
            continue
            
        emb_list = []
        # Use full path when listing files
        for img_file in os.listdir(full_student_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(full_student_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            faces = face_app.get(img)
            if not faces:
                continue
                
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            emb = l2_normalize(f.embedding)
            emb_list.append(emb)
            
        if emb_list:
            # Ensure proper stacking and normalization of embeddings
            emb_stack = np.stack([np.array(e, dtype=np.float32) for e in emb_list], axis=0)
            mean_emb = l2_normalize(np.mean(emb_stack, axis=0))
            save_student(student_dir, class_name, section, subject, full_student_path, mean_emb)
            enrolled.append(student_dir)
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    
    return {"enrolled_students": enrolled}

def get_current_datetime():
    now = datetime.now()
    return {
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S.%f')[:-3],  # Include milliseconds
        'timestamp': now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
    }

def get_attendance_crop_path(class_name: str, section: str, subject: Optional[str] = None) -> str:
    dt = get_current_datetime()
    base_path = os.path.join(ATTENDANCE_CROPS_DIR, dt['date'], class_name, section)
    if subject:
        base_path = os.path.join(base_path, subject)
    os.makedirs(base_path, exist_ok=True)
    return base_path, dt['timestamp']

@app.post("/mark-attendance/")
async def mark_attendance_endpoint(
    class_name: str = Form(...),
    section: str = Form(...),
    subject: Optional[str] = Form(None),
    photos_zip: UploadFile = File(...),
    threshold: float = Form(0.3)
):
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    zip_path = os.path.join(temp_dir, "photos.zip")
    photos_dir = os.path.join(temp_dir, "extracted")
    
    # Save and extract zip file
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(photos_zip.file, f)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(photos_dir)
    
    # Get enrolled students for this class/section
    names, known_encodings = get_students(class_name, section, subject)
    


    marked = []
    already_marked = set()
    
    # Process each photo in extracted directory
    for root, _, files in os.walk(photos_dir):
        for photo_name in files:
            if not photo_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(root, photo_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            faces = face_app.get(img)
            for f in faces:
                # Ensure proper embedding format and normalization
                emb = np.array(f.embedding, dtype=np.float32)
                emb = l2_normalize(emb)
                scores = np.array([np.dot(emb, k) for k in known_encodings])
                
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    
                    if best_score >= threshold:
                        student_name = names[best_idx]
                        if student_name not in already_marked:
                            marked.append({
                                "name": student_name,
                                "similarity": float(best_score)
                            })
                            
                            # Save cropped face and attendance with precise timestamp
                            crops_dir, timestamp = get_attendance_crop_path(class_name, section, subject)
                            bbox = f.bbox.astype(int)
                            face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            crop_filename = f"{student_name}_{timestamp}.jpg"
                            crop_path = os.path.join(crops_dir, crop_filename)
                            cv2.imwrite(crop_path, face_crop)
                            
                            # Get current date and time for attendance
                            dt = get_current_datetime()
                            save_attendance(student_name, class_name, section, subject, float(best_score), dt['date'], dt['time'])
                            
                            already_marked.add(student_name)

    # Cleanup
    shutil.rmtree(temp_dir)
    
    return {"marked_students": marked}


@app.delete("/delete-student/")
async def delete_student(
    student_name: str,
    class_name: str,
    section: Optional[str] = None
):
    # Get student data
    if section:
        names, _ = get_students(class_name, section)
    else:
        names, _ = get_students(class_name, None)
    
    # Check if student exists
    if student_name not in names:
        return {"error": "Student not found"}
    
    # Delete student
    success = delete_student_by_name(student_name)
    print(f"Delete operation result: {success}")  # Debug print
    
    if success:
        return {"message": f"Student {student_name} deleted successfully"}
    else:
        return {"error": "Failed to delete student"}

@app.delete("/delete-class/")
async def delete_class(
    class_name: str,
    section: Optional[str] = None,
    subject: Optional[str] = None
):
    # Delete class data
    success = delete_class_data(class_name, section)
    
    if success:
        message = f"Deleted data for class {class_name}"
        if section:
            message += f" section {section}"
        if subject:
            message += f" subject {subject}"
        return {"message": message}
    else:
        return {"error": "No matching data found to delete"}