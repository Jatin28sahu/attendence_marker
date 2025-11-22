# Fixes Applied to Attendance Marker System

## Date: 2025-11-22

## Issues Addressed

### Issue 1: Database Reset on Multiple Uploads ✅
**Problem:** When uploading back-to-back zip files, the database was being reset, causing loss of previously enrolled students.

**Root Cause:** The `init_db()` function was dropping and recreating tables every time the database module was imported.

**Solution:**
- Removed `DROP TABLE` statements from `init_db()`
- Changed to use only `CREATE TABLE IF NOT EXISTS` (which was already present)
- Removed automatic `init_db()` call on module import
- Added proper initialization on FastAPI startup event
- Database now persists data across multiple uploads

### Issue 2: Ambiguous Roll Number Parsing ✅
**Problem:** Folder names not following the `rollno_name` format could cause incorrect student counts and parsing failures.

**Root Cause:** Simple string split without validation could lead to:
- Invalid roll numbers being accepted
- Empty names being processed
- Duplicate roll numbers within same upload

**Solution:**
- Created `validate_roll_no()` function to ensure roll numbers are alphanumeric
- Created `parse_student_folder_name()` function with proper validation
- Added duplicate detection within each upload batch
- Added comprehensive logging for all enrollment steps
- Returns detailed error information about skipped students

## Changes Made

### 1. database.py
- **Removed:** `DROP TABLE` statements that were resetting data
- **Removed:** Automatic `init_db()` call at module import
- **Added:** Database indexes for faster queries on attendance table
- **Added:** Auto-increment ID for attendance records
- **Enhanced:** Documentation about primary key constraints

### 2. app.py
- **Added:** Logging configuration at module level
- **Added:** Database initialization on FastAPI startup event
- **Added:** `validate_roll_no()` function for roll number validation
- **Added:** `parse_student_folder_name()` function for safe parsing
- **Enhanced:** Enrollment endpoint with:
  - Comprehensive logging at each step
  - Duplicate roll number detection within uploads
  - Detailed error tracking and reporting
  - Image processing statistics
  - Skipped student reporting with reasons

### 3. New Features
- **Logging:** All enrollment operations now logged with INFO level
- **Validation:** Roll numbers validated against regex pattern `^[a-zA-Z0-9_-]+$`
- **Error Reporting:** API returns both enrolled and skipped students with reasons
- **Duplicate Detection:** Prevents duplicate roll numbers within single upload
- **Statistics:** Returns count of images processed per student

## Database Schema

### Students Table
- `roll_no` (TEXT, PRIMARY KEY) - Ensures unique roll numbers across system
- `name` (TEXT)
- `class_name` (TEXT)
- `section` (TEXT)
- `subject` (TEXT)
- `face_path` (TEXT)
- `embedding` (BLOB)

### Attendance Table
- `id` (INTEGER, PRIMARY KEY, AUTOINCREMENT) - Unique record ID
- `roll_no` (TEXT, INDEXED)
- `student_name` (TEXT)
- `class_name` (TEXT, INDEXED)
- `section` (TEXT, INDEXED)
- `subject` (TEXT)
- `similarity_score` (REAL)
- `date` (TEXT, INDEXED)
- `time` (TEXT)

## API Response Changes

### /enroll/ endpoint now returns:
```json
{
  "enrolled_students": [
    {
      "roll_no": "21045001",
      "name": "aman_meena",
      "images_processed": 5
    }
  ],
  "skipped": [
    {
      "folder": "invalid_folder",
      "reason": "Invalid folder name format"
    }
  ]
}
```

## Testing Instructions

### Test 1: Back-to-Back Uploads
1. Start the FastAPI server: `uvicorn app:app --reload`
2. Upload first zip file with 3 students
3. Verify response shows 3 enrolled students
4. Upload second zip file with 3 different students
5. **Expected:** Response shows 3 newly enrolled students
6. **Verify:** Database now contains 6 total students (not 3)

### Test 2: Duplicate Roll Numbers
1. Create zip with folders: `21045001_student1` and `21045001_student2`
2. Upload the zip
3. **Expected:** Only first student enrolled, second skipped with duplicate reason

### Test 3: Invalid Folder Names
1. Create zip with folders:
   - `student_name_only` (no roll number at start)
   - `123#invalid` (invalid characters)
   - `21045001_` (empty name)
2. Upload the zip
3. **Expected:** All three skipped with appropriate reasons in response

### Test 4: Validate Logs
1. Check terminal/console output during upload
2. **Expected logs:**
   ```
   INFO: Starting enrollment for class=CS, section=A, subject=Math
   INFO: Successfully enrolled: 21045001 - aman_meena (from 5 images)
   WARNING: Skipping folder 'invalid': Does not contain underscore separator
   INFO: Enrollment complete: 3 students enrolled, 1 skipped
   ```

### Test 5: Database Persistence
1. Stop the FastAPI server
2. Start it again
3. Query students from previous enrollment
4. **Expected:** All previously enrolled students still present

## Important Notes

1. **Roll Number Uniqueness:** The current implementation uses `roll_no` as PRIMARY KEY, meaning the same roll number cannot be used across different classes. If you need the same roll number in multiple classes, the schema needs to be modified to use a composite key.

2. **INSERT OR REPLACE:** When re-enrolling a student with the same roll number, the old data will be replaced with new data (new embeddings, class info, etc.).

3. **Folder Name Format:** Student folders MUST follow format: `rollno_name` where:
   - Roll number contains only alphanumeric characters, hyphens, or underscores
   - Name can contain any characters after the first underscore
   - Examples: `21045001_aman_meena`, `A-123_john_doe`, `2024_student`

4. **Logging Level:** Currently set to INFO. Can be changed in app.py if needed:
   ```python
   logging.basicConfig(level=logging.DEBUG)  # For more verbose output
   ```

## Rollback Instructions

If you need to rollback these changes:
1. The original behavior had tables being dropped on each module import
2. To restore: Add back `DROP TABLE` statements in `init_db()` 
3. Add back `init_db()` call at bottom of database.py

However, this is **NOT RECOMMENDED** as it will cause data loss on every restart.
