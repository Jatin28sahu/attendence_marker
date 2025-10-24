# attendence_marker

# Smart Attendance (ArcFace + RetinaFace, Codespaces-ready)

This repo enrolls students from single-face images and marks attendance from a group photo.

# Smart Attendance — Simple (ArcFace), multi-image per student


# attendence_marker

# Smart Attendance (ArcFace + RetinaFace, Codespaces-ready)

This repo enrolls students from single-face images and marks attendance from a group photo.

# Smart Attendance — Simple (ArcFace), multi-image per student



1.first activate .venv

source .venv/bin/activate

2.second activate the fast api

uvicorn main:app --reload

3. use the /docs of fastapi via link 

and test it using zip file for upload


# Steps for scratch

initilaize venv

    python -m venv new_env1

Activate venv

    source new_env/bin/activate

installing requirements

    pip install -r requirements.txt

opening fastapi port
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

open the documentation
    /docs

upload file 
    in zip format
        faces(zip)>
                    ambuj>
                            1.
                            2.
                    aman>
                            1.
                            2.
        
        class_photo(zip)>
                    1.
                    2.
