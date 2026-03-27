from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
from models import FaceEmbeddingCreate, FaceEmbeddingResponse

_client = AsyncIOMotorClient(settings.mongodb_url)
_col = _client[settings.mongodb_db]["face_embeddings"]


async def save_embeddings(student_id: str, embeddings: list[list[float]]) -> None:
    # Delete existing embeddings for this student
    await _col.delete_many({"student_id": student_id})
    
    # Validate with Pydantic before saving
    docs = [
        FaceEmbeddingCreate(student_id=student_id, embedding=emb).dict()
        for emb in embeddings
    ]
    await _col.insert_many(docs)


async def get_all_embeddings() -> list[FaceEmbeddingResponse]:
    cursor = _col.find({}, {"_id": 0, "student_id": 1, "embedding": 1, "created_at": 1})
    results = await cursor.to_list(length=None)
    return [FaceEmbeddingResponse(**doc) for doc in results]