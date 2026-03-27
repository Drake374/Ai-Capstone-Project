from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class FaceEmbeddingCreate(BaseModel):
    student_id: str
    embedding: list[float] = Field(
        min_items=512,
        max_items=512,
        description="512-dimensional FaceNet embedding"
    )
    created_at: datetime = Field(default_factory=datetime.now)

class FaceEmbeddingResponse(BaseModel):
    student_id: str
    embedding: list[float]
    created_at: datetime


class AttendanceLog(BaseModel):
    student_id: str
    status: str  # "present" or "absent"
    similarity: float
    timestamp: datetime = Field(default_factory=datetime.now)


class AttendanceLogResponse(BaseModel):
    student_id: str
    status: str
    similarity: float
    timestamp: datetime


class Student(BaseModel):
    student_id: str
    name: str
    email: str
    photo_url: str = ""
    registered: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class StudentResponse(BaseModel):
    student_id: str
    name: str
    email: str
    photo_url: str
    registered: bool
    created_at: datetime