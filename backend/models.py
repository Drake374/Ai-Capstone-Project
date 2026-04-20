from pydantic import BaseModel, Field
from typing import Literal, Optional
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
    status: Literal["present", "absent", "late", "excused", "partial"]
    similarity: float
    timestamp: datetime = Field(default_factory=datetime.now)


class AttendanceLogResponse(BaseModel):
    student_id: str
    status: Literal["present", "absent", "late", "excused", "partial"]
    similarity: float
    timestamp: datetime


class CheckResult(BaseModel):
    student_id: str
    session_id: str
    passed: bool
    similarity: float
    timestamp: datetime = Field(default_factory=datetime.now)


class CheckResultResponse(BaseModel):
    student_id: str
    session_id: str
    passed: bool
    similarity: float
    timestamp: datetime


class Student(BaseModel):
    student_id: str
    name: str
    email: str
    photo_url: str = ""
    role: str = "student"
    registered: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class StudentResponse(BaseModel):
    student_id: str
    name: str
    email: str
    photo_url: str
    role: str = "student"
    registered: bool
    created_at: datetime


class Admin(BaseModel):
    name: str
    email: str
    photo_url: str = ""
    role: str = "admin"
    created_at: datetime = Field(default_factory=datetime.now)


class AdminResponse(BaseModel):
    name: str
    email: str
    photo_url: str
    role: str = "admin"
    created_at: datetime


class AttendanceCheck(BaseModel):
    student_id: str
    session_id: str
    passed: bool
    similarity: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)