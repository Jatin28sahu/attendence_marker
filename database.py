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
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Create students table
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (name TEXT, class_name TEXT, section TEXT, subject TEXT,
                  face_path TEXT, embedding BLOB)''')
    
    # Drop existing attendance table if exists to avoid schema conflicts
    c.execute('DROP TABLE IF EXISTS attendance')
    
    # Create attendance table with correct schema
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (student_name TEXT,
                  class_name TEXT,
                  section TEXT,
                  subject TEXT,
                  similarity_score REAL,
                  date TEXT,
                  time TEXT)''')
    
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

def save_student(name, class_name, section, subject, face_path, face_encoding):
    # Ensure face_encoding is float32 and normalized before saving
    face_encoding = np.array(face_encoding, dtype=np.float32)
    face_encoding = l2_normalize(face_encoding)
    
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO students 
            (name, class_name, section, subject, face_path, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, class_name, section, subject, face_path, face_encoding))
        conn.commit()
    finally:
        conn.close()

def save_attendance(student_name, class_name, section, subject, similarity_score, date, time):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    c.execute('''INSERT INTO attendance 
                 (student_name, class_name, section, subject, similarity_score, date, time)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (student_name, class_name, section, subject, similarity_score, date, time))
    
    conn.commit()
    conn.close()

def get_students(class_name, section, subject=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        if subject:
            query = '''
                SELECT name, embedding 
                FROM students 
                WHERE class_name=? AND section=? AND subject=?
            '''
            params = (class_name, section, subject)
        else:
            query = '''
                SELECT name, embedding 
                FROM students 
                WHERE class_name=? AND section=?
            '''
            params = (class_name, section)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        # Ensure embeddings are properly normalized when retrieved
        names = [row[0] for row in rows]
        encodings = [l2_normalize(row[1]) for row in rows]
        return names, encodings
    finally:
        conn.close()

def delete_student_by_name(student_name):
    conn = get_db()
    try:
        cur = conn.cursor()
        # Delete from students table
        cur.execute('DELETE FROM students WHERE name = ?', (student_name,))
        students_deleted = cur.rowcount
        
        # Delete from attendance table
        cur.execute('DELETE FROM attendance WHERE student_name = ?', (student_name,))
        attendance_deleted = cur.rowcount
        
        conn.commit()
        # Return True if anything was deleted from either table
        return students_deleted > 0 or attendance_deleted > 0
    finally:
        conn.close()
        
def delete_class_data(class_name, section=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        if section:
            # Delete from students table with class and section
            cur.execute('DELETE FROM students WHERE class_name = ? AND section = ?', 
                       (class_name, section))
            students_deleted = cur.rowcount
            
            # Delete from attendance table with class and section
            cur.execute('DELETE FROM attendance WHERE class_name = ? AND section = ?', 
                       (class_name, section))
            attendance_deleted = cur.rowcount
        else:
            # Delete from students table with just class
            cur.execute('DELETE FROM students WHERE class_name = ?', (class_name,))
            students_deleted = cur.rowcount
            
            # Delete from attendance table with just class
            cur.execute('DELETE FROM attendance WHERE class_name = ?', (class_name,))
            attendance_deleted = cur.rowcount
            
        conn.commit()
        # Return True if anything was deleted from either table
        return students_deleted > 0 or attendance_deleted > 0
    finally:
        conn.close()

# Initialize database when module is imported
init_db()
