import sqlite3
import os
import numpy as np
from datetime import datetime
from utils import l2_normalize  # Add this import

# Define database path
DB_PATH = 'attendance.db'

# Only try to create directory if DB_PATH includes a directory component
db_dir = os.path.dirname(DB_PATH)
if db_dir:  # Only create if there's a directory component
    os.makedirs(db_dir, exist_ok=True)

def get_db():
    return sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)

def init_db():
    """Initialize database tables if they don't exist. Does NOT drop existing data."""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Create students table with roll_no as primary key (only if not exists)
    # roll_no is the primary key to ensure uniqueness across all classes
    # Note: If you need the same roll_no in different classes, consider changing this
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (roll_no TEXT PRIMARY KEY,
                  name TEXT,
                  class_name TEXT,
                  section TEXT,
                  subject TEXT,
                  face_path TEXT,
                  embedding BLOB)''')
    
    # Create attendance table with roll_no reference (only if not exists)
    # No unique constraint here to allow multiple attendance records per student
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  roll_no TEXT,
                  student_name TEXT,
                  class_name TEXT,
                  section TEXT,
                  subject TEXT,
                  similarity_score REAL,
                  date TEXT,
                  time TEXT)''')
    
    # Create index on attendance table for faster queries
    c.execute('''CREATE INDEX IF NOT EXISTS idx_attendance_roll_no 
                 ON attendance(roll_no)''')
    
    c.execute('''CREATE INDEX IF NOT EXISTS idx_attendance_class 
                 ON attendance(class_name, section, date)''')
    
    conn.commit()
    conn.close()

def adapt_array(arr):
    """Converts numpy array to binary blob for database storage"""
    out = np.array(arr, dtype=np.float32)  # Ensure float32 type
    return out.tobytes()

def convert_array(blob):
    """Converts binary blob back to numpy array"""
    out = np.frombuffer(blob, dtype=np.float32)  # Specify dtype when converting back
    if len(out) == 512:  # InsightFace embeddings are 512-dimensional
        return l2_normalize(out)  # Ensure normalized
    return out

# Register adapters
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("BLOB", convert_array)

def save_student(roll_no, name, class_name, section, subject, face_path, face_encoding):
    # Ensure face_encoding is float32 and normalized before saving
    face_encoding = np.array(face_encoding, dtype=np.float32)
    face_encoding = l2_normalize(face_encoding)
    
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO students 
            (roll_no, name, class_name, section, subject, face_path, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (roll_no, name, class_name, section, subject, face_path, face_encoding))
        conn.commit()
    finally:
        conn.close()

def save_attendance(roll_no, student_name, class_name, section, subject, similarity_score, date, time):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    c.execute('''INSERT INTO attendance 
                 (roll_no, student_name, class_name, section, subject, similarity_score, date, time)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (roll_no, student_name, class_name, section, subject, similarity_score, date, time))
    
    conn.commit()
    conn.close()

def get_students(class_name, section, subject=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        if subject:
            query = '''
                SELECT roll_no, name, embedding 
                FROM students 
                WHERE class_name=? AND section=? AND subject=?
            '''
            params = (class_name, section, subject)
        else:
            query = '''
                SELECT roll_no, name, embedding 
                FROM students 
                WHERE class_name=? AND section=?
            '''
            params = (class_name, section)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        # Ensure embeddings are properly normalized when retrieved
        roll_nos = [row[0] for row in rows]
        names = [row[1] for row in rows]
        encodings = [l2_normalize(row[2]) for row in rows]
        return roll_nos, names, encodings
    finally:
        conn.close()

def delete_student_by_roll_no(roll_no):
    conn = get_db()
    try:
        cur = conn.cursor()
        # Delete from students table
        cur.execute('DELETE FROM students WHERE roll_no = ?', (roll_no,))
        students_deleted = cur.rowcount
        
        # Delete from attendance table
        cur.execute('DELETE FROM attendance WHERE roll_no = ?', (roll_no,))
        attendance_deleted = cur.rowcount
        
        conn.commit()
        # Return True if anything was deleted from either table
        return students_deleted > 0 or attendance_deleted > 0
    finally:
        conn.close()
        
def delete_class_data(class_name, section=None, subject=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        
        # Build query based on provided parameters
        if section and subject:
            # Delete with class, section, and subject
            cur.execute('DELETE FROM students WHERE class_name = ? AND section = ? AND subject = ?', 
                       (class_name, section, subject))
            students_deleted = cur.rowcount
            
            cur.execute('DELETE FROM attendance WHERE class_name = ? AND section = ? AND subject = ?', 
                       (class_name, section, subject))
            attendance_deleted = cur.rowcount
        elif section:
            # Delete with class and section only
            cur.execute('DELETE FROM students WHERE class_name = ? AND section = ?', 
                       (class_name, section))
            students_deleted = cur.rowcount
            
            cur.execute('DELETE FROM attendance WHERE class_name = ? AND section = ?', 
                       (class_name, section))
            attendance_deleted = cur.rowcount
        else:
            # Delete with just class
            cur.execute('DELETE FROM students WHERE class_name = ?', (class_name,))
            students_deleted = cur.rowcount
            
            cur.execute('DELETE FROM attendance WHERE class_name = ?', (class_name,))
            attendance_deleted = cur.rowcount
            
        conn.commit()
        # Return True if anything was deleted from either table
        return students_deleted > 0 or attendance_deleted > 0
    finally:
        conn.close()

# Database will be initialized when app starts, not on module import
