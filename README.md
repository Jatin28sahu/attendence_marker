# Attendance Marker System

An automated face recognition-based attendance system built with FastAPI and InsightFace.

## Overview

This system allows you to:
- Enroll students with their facial recognition data using roll numbers
- Mark attendance automatically from group photos
- Manage student records and attendance data
- Delete students and class data with fine-grained control

## Features

- **Roll Number Based System**: Students are identified by unique roll numbers (primary key)
- **Face Recognition**: Uses InsightFace for accurate face detection and recognition
- **Batch Enrollment**: Upload ZIP files containing student face images
- **Automated Attendance**: Process group photos to mark attendance automatically
- **Flexible Deletion**: Delete students by roll number or entire class data with optional filters
- **CLI Testing Tool**: Test database operations without running the web server

## Recent Updates (November 2025)

### Critical Bug Fixes ✅
1. **Database Persistence Issue Fixed**: Database no longer resets on multiple uploads or server restarts
2. **Roll Number Validation**: Added proper validation for folder names during enrollment
3. **Duplicate Detection**: System now detects and prevents duplicate roll numbers within same upload
4. **Enhanced Logging**: Comprehensive logging added for debugging enrollment issues

### Database Schema Changes
- **Roll Number as Primary Key**: Students are now identified by roll numbers instead of names
- **Enhanced Attendance Table**: Stores both roll number and student name with auto-increment ID
- **Database Indexes**: Added indexes for faster queries on attendance records
- **Subject Support**: Optional subject parameter for better organization
- **Data Persistence**: Database now persists across server restarts and multiple uploads

### API Changes
1. **Enrollment (`POST /enroll/`)**: 
   - Now parses roll number from folder names (format: `{roll_no}_{name}`)
   - Example: `21045001_aman_meena`
   - **NEW**: Returns detailed information about skipped students
   - **NEW**: Validates roll numbers (alphanumeric, hyphens, underscores only)
   - **NEW**: Detects duplicate roll numbers within upload
   - **NEW**: Returns image processing statistics

2. **Delete Student (`DELETE /delete-student/`)**:
   - Changed from `student_name` parameter to `roll_no`
   - Simpler and more precise deletion

3. **Delete Class (`DELETE /delete-class/`)**:
   - Added optional `subject` parameter
   - Supports three modes:
     - Delete by class only
     - Delete by class + section
     - Delete by class + section + subject

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Jatin28sahu/attendence_marker.git
cd attendence_marker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python -c "from database import init_db; init_db()"
```

## Usage

### Option 1: FastAPI Server

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Access the API documentation at: `http://localhost:8000/docs`

#### API Endpoints

##### 1. Enroll Students
```http
POST /enroll/
Content-Type: multipart/form-data

Parameters:
- class_name: string (required)
- section: string (required)
- subject: string (optional)
- faces_zip: file (required)
```

**Folder Structure in ZIP:**
```
faces.zip
├── 21045001_aman_meena/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── 21045002_ambuj_yadav/
│   ├── image1.jpg
│   └── image2.jpg
└── 21045003_avinash_sharma/
    └── image1.jpg
```

**Response:**
```json
{
  "enrolled_students": [
    {
      "roll_no": "21045001", 
      "name": "aman_meena",
      "images_processed": 5
    },
    {
      "roll_no": "21045002", 
      "name": "ambuj_yadav",
      "images_processed": 3
    },
    {
      "roll_no": "21045003", 
      "name": "avinash_sharma",
      "images_processed": 4
    }
  ],
  "skipped": [
    {
      "folder": "invalid_folder_name",
      "reason": "Invalid folder name format"
    }
  ]
}
```

**Note:** The API now returns:
- Number of images successfully processed per student
- List of skipped folders with reasons (if any)
- Detailed logging in server console for debugging

##### 2. Mark Attendance
```http
POST /mark-attendance/
Content-Type: multipart/form-data

Parameters:
- class_name: string (required)
- section: string (required)
- subject: string (optional)
- photos_zip: file (required)
- threshold: float (default: 0.3)
```

**Response:**
```json
{
  "marked_students": [
    {
      "roll_no": "21045001",
      "name": "aman_meena",
      "similarity": 0.87
    }
  ]
}
```

##### 3. Delete Student
```http
DELETE /delete-student/?roll_no=21045001
```

**Response:**
```json
{
  "message": "Student with roll number 21045001 deleted successfully"
}
```

##### 4. Delete Class Data
```http
DELETE /delete-class/?class_name=CSE&section=A&subject=Math
```

**Parameters:**
- `class_name`: Required
- `section`: Optional (delete entire class if omitted)
- `subject`: Optional (delete specific subject if provided)

**Response:**
```json
{
  "message": "Deleted data for class CSE section A subject Math"
}
```

### Option 2: CLI Testing Tool

For quick testing without running the FastAPI server:

```bash
python main.py
```

**Available Operations:**
1. View all students
2. View students by class/section/subject
3. View all attendance records
4. View attendance by class/section
5. Add sample student (for testing)
6. Delete student by roll number
7. Delete class data
8. Clear all data (reset database)
9. Exit

**Example Usage:**
```bash
$ python main.py

============================================================
ATTENDANCE SYSTEM - BACKEND TESTING
============================================================
1. View all students
2. View students by class/section/subject
...

Enter your choice (1-9): 5

Add Sample Student
Enter roll number (e.g., 21045001): 21045001
Enter name (e.g., aman_meena): aman_meena
Enter class name (e.g., CSE): CSE
Enter section (e.g., A): A
Enter subject (press Enter to skip): Math

✓ Student aman_meena (Roll No: 21045001) added successfully!
```

## Database Schema

### Students Table
| Column      | Type | Description                    |
|-------------|------|--------------------------------|
| roll_no     | TEXT | Primary Key - Student roll no  |
| name        | TEXT | Student name                   |
| class_name  | TEXT | Class name                     |
| section     | TEXT | Section                        |
| subject     | TEXT | Subject (optional)             |
| face_path   | TEXT | Path to face images            |
| embedding   | BLOB | Face embedding vector (512-dim)|

### Attendance Table
| Column           | Type  | Description                  |
|------------------|-------|------------------------------|
| roll_no          | TEXT  | Student roll number          |
| student_name     | TEXT  | Student name                 |
| class_name       | TEXT  | Class name                   |
| section          | TEXT  | Section                      |
| subject          | TEXT  | Subject (optional)           |
| similarity_score | REAL  | Face match confidence        |
| date             | TEXT  | Attendance date              |
| time             | TEXT  | Attendance time              |

## File Structure

```
attendence_marker/
├── app.py                 # FastAPI application
├── database.py            # Database operations
├── main.py                # CLI testing tool
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables
├── attendance.db          # SQLite database (auto-generated)
├── data/
│   ├── faces/             # Enrolled face images
│   └── attendance_crops/  # Attendance face crops
└── temp_uploads/          # Temporary file storage
```

## Configuration

Environment variables can be set in `.env` file:
```env
# Add any configuration variables here
```

## Important Notes

### Roll Number Format
- Roll numbers are parsed from folder names using the format: `{roll_no}_{name}`
- Example: `21045001_aman_meena` → Roll No: `21045001`, Name: `aman_meena`
- Ensure folder names follow this format during enrollment

### Face Recognition
- Minimum 3 images per student recommended for better accuracy
- Images should be clear, well-lit, and show the face clearly
- Supported formats: JPG, JPEG, PNG

### Deletion Operations
- **Delete Student**: Removes student from both students and attendance tables
- **Delete Class**: 
  - With only `class_name`: Deletes all sections in that class
  - With `class_name` + `section`: Deletes specific section
  - With `class_name` + `section` + `subject`: Deletes specific subject only

### Attendance Crops
- Face crops from attendance photos are saved in `data/attendance_crops/`
- Organized by: `date/class/section/subject/`
- Filename format: `{roll_no}_{name}_{timestamp}.jpg`

## Troubleshooting

### Common Issues

1. **"Student not found" when deleting**
   - Verify the roll number exists using the CLI tool (option 1)
   - Roll numbers are case-sensitive

2. **Face not recognized during attendance**
   - Ensure good quality enrollment images
   - Adjust the `threshold` parameter (lower = more lenient)
   - Default threshold is 0.3

3. **Database locked errors**
   - Ensure only one process is accessing the database at a time
   - Close the CLI tool before running the FastAPI server

## Development

### Running Tests
Use the CLI tool (`main.py`) to test database operations:
```bash
python main.py
```

### API Testing
Use FastAPI's built-in docs:
```bash
uvicorn app:app --reload
# Visit http://localhost:8000/docs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Contact

For issues and questions, please use the GitHub issue tracker.

## Changelog

### Version 2.0 (Latest)
- Added roll number as primary key
- Enhanced delete operations with subject support
- Created CLI testing tool
- Improved folder name parsing for enrollment
- Updated API documentation

### Version 1.0
- Initial release
- Basic enrollment and attendance functionality
