from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
import os
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from database import init_db, save_student, get_students, save_attendance, delete_student_by_roll_no, delete_class_data
from utils import l2_normalize
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

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

def validate_roll_no(roll_no: str) -> bool:
    """Validate that roll_no contains only alphanumeric characters and is not empty."""
    if not roll_no:
        return False
    # Allow alphanumeric characters, hyphens, and underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', roll_no))

def parse_student_folder_name(folder_name: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse student folder name to extract roll_no and name.
    Expected format: rollno_name (e.g., 21045001_aman_meena)
    
    Returns:
        tuple: (roll_no, name) or (None, None) if parsing fails
    """
    # Split on first underscore
    parts = folder_name.split('_', 1)
    
    if len(parts) < 2:
        logger.warning(f"Skipping folder '{folder_name}': Does not contain underscore separator")
        return None, None
    
    roll_no = parts[0].strip()
    name = parts[1].strip()
    
    # Validate roll_no
    if not validate_roll_no(roll_no):
        logger.warning(f"Skipping folder '{folder_name}': Invalid roll number format '{roll_no}'")
        return None, None
    
    if not name:
        logger.warning(f"Skipping folder '{folder_name}': Empty name after roll number")
        return None, None
    
    return roll_no, name

@app.post("/enroll/")
async def enroll_students(
    class_name: str = Form(...),
    section: str = Form(...),
    subject: Optional[str] = Form(None),
    faces_zip: UploadFile = File(...)
):
    logger.info(f"Starting enrollment for class={class_name}, section={section}, subject={subject}")
    
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
    skipped = []
    processed_roll_nos = set()
    
    # Process each student directory
    for student_dir in os.listdir(class_dir):
        full_student_path = os.path.join(class_dir, student_dir)
        if not os.path.isdir(full_student_path):
            continue
        
        # Parse and validate folder name
        roll_no, name = parse_student_folder_name(student_dir)
        if roll_no is None or name is None:
            skipped.append({"folder": student_dir, "reason": "Invalid folder name format"})
            continue
        
        # Check for duplicate roll numbers in this upload
        if roll_no in processed_roll_nos:
            logger.warning(f"Duplicate roll number '{roll_no}' found in folder '{student_dir}'. Skipping.")
            skipped.append({"folder": student_dir, "reason": f"Duplicate roll number: {roll_no}"})
            continue
        
        processed_roll_nos.add(roll_no)
            
        emb_list = []
        img_count = 0
        
        # Use full path when listing files
        for img_file in os.listdir(full_student_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_count += 1
            img_path = os.path.join(full_student_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            faces = face_app.get(img)
            if not faces:
                logger.warning(f"No face detected in image: {img_path}")
                continue
                
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            emb = l2_normalize(f.embedding)
            emb_list.append(emb)
        
        if not emb_list:
            logger.warning(f"No valid face embeddings for student '{roll_no}' ({name}). Processed {img_count} images.")
            skipped.append({"folder": student_dir, "reason": "No valid face embeddings found"})
            continue
            
        # Ensure proper stacking and normalization of embeddings
        emb_stack = np.stack([np.array(e, dtype=np.float32) for e in emb_list], axis=0)
        mean_emb = l2_normalize(np.mean(emb_stack, axis=0))
        save_student(roll_no, name, class_name, section, subject, full_student_path, mean_emb)
        
        logger.info(f"Successfully enrolled: {roll_no} - {name} (from {len(emb_list)} images)")
        enrolled.append({"roll_no": roll_no, "name": name, "images_processed": len(emb_list)})
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    
    logger.info(f"Enrollment complete: {len(enrolled)} students enrolled, {len(skipped)} skipped")
    
    result = {"enrolled_students": enrolled}
    if skipped:
        result["skipped"] = skipped
    
    return result

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
    roll_nos, names, known_encodings = get_students(class_name, section, subject)
    


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
                        roll_no = roll_nos[best_idx]
                        student_name = names[best_idx]
                        if roll_no not in already_marked:
                            marked.append({
                                "roll_no": roll_no,
                                "name": student_name,
                                "similarity": float(best_score)
                            })
                            
                            # Save cropped face and attendance with precise timestamp
                            crops_dir, timestamp = get_attendance_crop_path(class_name, section, subject)
                            bbox = f.bbox.astype(int)
                            face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            crop_filename = f"{roll_no}_{student_name}_{timestamp}.jpg"
                            crop_path = os.path.join(crops_dir, crop_filename)
                            cv2.imwrite(crop_path, face_crop)
                            
                            # Get current date and time for attendance
                            dt = get_current_datetime()
                            save_attendance(roll_no, student_name, class_name, section, subject, float(best_score), dt['date'], dt['time'])
                            
                            already_marked.add(roll_no)

    # Cleanup
    shutil.rmtree(temp_dir)
    
    return {"marked_students": marked}


@app.delete("/delete-student/")
async def delete_student(
    roll_no: str
):
    # Delete student by roll number
    success = delete_student_by_roll_no(roll_no)
    print(f"Delete operation result: {success}")  # Debug print
    
    if success:
        return {"message": f"Student with roll number {roll_no} deleted successfully"}
    else:
        return {"error": "Student not found"}

@app.delete("/delete-class/")
async def delete_class(
    class_name: str,
    section: Optional[str] = None,
    subject: Optional[str] = None
):
    # Delete class data with optional subject parameter
    success = delete_class_data(class_name, section, subject)
    
    if success:
        message = f"Deleted data for class {class_name}"
        if section:
            message += f" section {section}"
        if subject:
            message += f" subject {subject}"
        return {"message": message}
    else:
        return {"error": "No matching data found to delete"}
