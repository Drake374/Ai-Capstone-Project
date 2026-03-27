from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from services.face_service import register_student_faces, verify_face
from db.student_repo import upsert_student, get_student_by_email, get_student
from db.student_repo import mark_registered as mark_student_registered
from db.face_repo import count_embeddings

router = APIRouter()


class Frame(BaseModel):
    imageData: str  # base64 encoded image, e.g. "data:image/jpeg;base64,..."
    timestamp: float  # timestamp as float (JavaScript number)


class RegisterFacesRequest(BaseModel):
    frames: List[Frame]
    studentId: str


class VerifyFaceRequest(BaseModel):
    imageData: str  # base64 encoded image


class RegisterStudentRequest(BaseModel):
    studentId: str
    name: str
    email: str
    photoUrl: str = ""


@router.post("/register-student")
async def register_student(body: RegisterStudentRequest):
    """Create or update a student profile."""
    try:
        student = await upsert_student(
            student_id=body.studentId,
            name=body.name,
            email=body.email,
            photo_url=body.photoUrl,
        )
        return {
            "student_id": student.student_id,
            "name": student.name,
            "email": student.email,
            "registered": student.registered,
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.get("/profile")
async def get_profile(email: str = Query(..., description="Student email")):
    """Get student profile by email, including registration status and face count."""
    student = await get_student_by_email(email)
    if not student:
        return {"found": False}

    faces = await count_embeddings(student.student_id)
    return {
        "found": True,
        "student_id": student.student_id,
        "name": student.name,
        "email": student.email,
        "photo_url": student.photo_url,
        "registered": student.registered,
        "face_count": faces,
    }


@router.post("/register-faces")
async def register_faces(body: RegisterFacesRequest):
    print(f"Received registration request for student {body.studentId} with {len(body.frames)} frames")
    if not body.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    try:
        await register_student_faces(
            student_id=body.studentId,
            frames=[f.imageData for f in body.frames],
        )
        # Mark student as registered
        await mark_student_registered(body.studentId)
        return {"message": f"Faces registered for student {body.studentId}"}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.post("/verify-face")
async def verify_face_endpoint(body: VerifyFaceRequest):
    """Verify a face against all registered embeddings using cosine similarity."""
    if not body.imageData:
        raise HTTPException(status_code=400, detail="No image data provided")

    try:
        result = await verify_face(frame_data_url=body.imageData)
        return result
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())