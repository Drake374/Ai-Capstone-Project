from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from services.face_service import register_student_faces, verify_face

router = APIRouter()


class Frame(BaseModel):
    imageData: str  # base64 encoded image, e.g. "data:image/jpeg;base64,..."
    timestamp: float  # timestamp as float (JavaScript number)


class RegisterFacesRequest(BaseModel):
    frames: List[Frame]
    studentId: str


class VerifyFaceRequest(BaseModel):
    imageData: str  # base64 encoded image


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