"""
Main CLI interface for testing attendance system backend operations
"""
import sqlite3
from database import (
    init_db, 
    save_student, 
    get_students, 
    save_attendance, 
    delete_student_by_roll_no, 
    delete_class_data
)
import numpy as np
from datetime import datetime

def print_menu():
    print("\n" + "="*60)
    print("ATTENDANCE SYSTEM - BACKEND TESTING")
    print("="*60)
    print("1. View all students")
    print("2. View students by class/section/subject")
    print("3. View all attendance records")
    print("4. View attendance by class/section")
    print("5. Add sample student (for testing)")
    print("6. Delete student by roll number")
    print("7. Delete class data")
    print("8. Clear all data (reset database)")
    print("9. Exit")
    print("="*60)

def view_all_students():
    """Display all students in the database"""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT roll_no, name, class_name, section, subject FROM students')
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        print("\nNo students found in database.")
        return
    
    print(f"\n{'Roll No':<15} {'Name':<20} {'Class':<10} {'Section':<10} {'Subject':<15}")
    print("-" * 80)
    for row in rows:
        roll_no, name, class_name, section, subject = row
        subject_display = subject if subject else "N/A"
        print(f"{roll_no:<15} {name:<20} {class_name:<10} {section:<10} {subject_display:<15}")
    print(f"\nTotal students: {len(rows)}")

def view_students_by_class():
    """Display students filtered by class/section/subject"""
    class_name = input("Enter class name: ").strip()
    section = input("Enter section: ").strip()
    subject = input("Enter subject (press Enter to skip): ").strip()
    
    if not subject:
        subject = None
    
    try:
        roll_nos, names, encodings = get_students(class_name, section, subject)
        
        if not roll_nos:
            print(f"\nNo students found for class {class_name}, section {section}" + 
                  (f", subject {subject}" if subject else ""))
            return
        
        print(f"\n{'Roll No':<15} {'Name':<20}")
        print("-" * 35)
        for roll_no, name in zip(roll_nos, names):
            print(f"{roll_no:<15} {name:<20}")
        print(f"\nTotal students: {len(roll_nos)}")
    except Exception as e:
        print(f"Error: {e}")

def view_all_attendance():
    """Display all attendance records"""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT roll_no, student_name, class_name, section, subject, similarity_score, date, time FROM attendance ORDER BY date DESC, time DESC')
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        print("\nNo attendance records found.")
        return
    
    print(f"\n{'Roll No':<12} {'Name':<18} {'Class':<8} {'Section':<8} {'Subject':<12} {'Score':<8} {'Date':<12} {'Time':<12}")
    print("-" * 100)
    for row in rows:
        roll_no, name, class_name, section, subject, score, date, time = row
        subject_display = subject if subject else "N/A"
        print(f"{roll_no:<12} {name:<18} {class_name:<8} {section:<8} {subject_display:<12} {score:<8.4f} {date:<12} {time:<12}")
    print(f"\nTotal records: {len(rows)}")

def view_attendance_by_class():
    """Display attendance records filtered by class/section"""
    class_name = input("Enter class name: ").strip()
    section = input("Enter section: ").strip()
    
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT roll_no, student_name, subject, similarity_score, date, time FROM attendance WHERE class_name=? AND section=? ORDER BY date DESC, time DESC',
              (class_name, section))
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        print(f"\nNo attendance records found for class {class_name}, section {section}")
        return
    
    print(f"\n{'Roll No':<12} {'Name':<20} {'Subject':<15} {'Score':<8} {'Date':<12} {'Time':<12}")
    print("-" * 85)
    for row in rows:
        roll_no, name, subject, score, date, time = row
        subject_display = subject if subject else "N/A"
        print(f"{roll_no:<12} {name:<20} {subject_display:<15} {score:<8.4f} {date:<12} {time:<12}")
    print(f"\nTotal records: {len(rows)}")

def add_sample_student():
    """Add a sample student for testing"""
    print("\nAdd Sample Student")
    roll_no = input("Enter roll number (e.g., 21045001): ").strip()
    name = input("Enter name (e.g., aman_meena): ").strip()
    class_name = input("Enter class name (e.g., CSE): ").strip()
    section = input("Enter section (e.g., A): ").strip()
    subject = input("Enter subject (press Enter to skip): ").strip()
    
    if not subject:
        subject = None
    
    # Create a dummy embedding (512-dimensional for InsightFace)
    dummy_embedding = np.random.randn(512).astype(np.float32)
    dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)  # Normalize
    
    face_path = f"data/faces/{class_name}_{section}/{roll_no}_{name}"
    
    try:
        save_student(roll_no, name, class_name, section, subject, face_path, dummy_embedding)
        print(f"\n✓ Student {name} (Roll No: {roll_no}) added successfully!")
    except Exception as e:
        print(f"\n✗ Error adding student: {e}")

def delete_student():
    """Delete a student by roll number"""
    print("\nDelete Student")
    roll_no = input("Enter roll number to delete: ").strip()
    
    confirm = input(f"Are you sure you want to delete student with roll number {roll_no}? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Delete cancelled.")
        return
    
    try:
        success = delete_student_by_roll_no(roll_no)
        if success:
            print(f"\n✓ Student with roll number {roll_no} deleted successfully!")
        else:
            print(f"\n✗ Student with roll number {roll_no} not found.")
    except Exception as e:
        print(f"\n✗ Error deleting student: {e}")

def delete_class():
    """Delete class data"""
    print("\nDelete Class Data")
    class_name = input("Enter class name: ").strip()
    section = input("Enter section (press Enter to delete entire class): ").strip()
    subject = input("Enter subject (press Enter to skip): ").strip()
    
    if not section:
        section = None
    if not subject:
        subject = None
    
    confirm_msg = f"class {class_name}"
    if section:
        confirm_msg += f", section {section}"
    if subject:
        confirm_msg += f", subject {subject}"
    
    confirm = input(f"Are you sure you want to delete data for {confirm_msg}? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Delete cancelled.")
        return
    
    try:
        success = delete_class_data(class_name, section, subject)
        if success:
            print(f"\n✓ Data for {confirm_msg} deleted successfully!")
        else:
            print(f"\n✗ No data found for {confirm_msg}.")
    except Exception as e:
        print(f"\n✗ Error deleting class data: {e}")

def clear_all_data():
    """Clear all data and reinitialize database"""
    print("\n⚠️  WARNING: This will delete ALL students and attendance records!")
    confirm = input("Type 'DELETE ALL' to confirm: ").strip()
    
    if confirm != 'DELETE ALL':
        print("Operation cancelled.")
        return
    
    try:
        init_db()  # This will drop and recreate tables
        print("\n✓ All data cleared. Database reset successfully!")
    except Exception as e:
        print(f"\n✗ Error clearing data: {e}")

def main():
    """Main CLI loop"""
    # Initialize database
    init_db()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            view_all_students()
        elif choice == '2':
            view_students_by_class()
        elif choice == '3':
            view_all_attendance()
        elif choice == '4':
            view_attendance_by_class()
        elif choice == '5':
            add_sample_student()
        elif choice == '6':
            delete_student()
        elif choice == '7':
            delete_class()
        elif choice == '8':
            clear_all_data()
        elif choice == '9':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\n✗ Invalid choice. Please enter a number between 1 and 9.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ATTENDANCE SYSTEM - Backend Testing Interface")
    print("="*60)
    print("This CLI tool allows you to test database operations directly")
    print("without using the FastAPI server.")
    print("="*60)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
