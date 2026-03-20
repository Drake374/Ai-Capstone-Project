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