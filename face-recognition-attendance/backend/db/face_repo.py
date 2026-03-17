from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

_client = AsyncIOMotorClient(settings.mongodb_url)
_col = _client[settings.mongodb_db]["face_embeddings"]


async def save_embeddings(student_id: str, embeddings: list[list[float]]) -> None:
    docs = [{"student_id": student_id, "embedding": emb} for emb in embeddings]
    await _col.insert_many(docs)


async def get_all_embeddings() -> list[dict]:
    cursor = _col.find({}, {"_id": 0, "student_id": 1, "embedding": 1})
    return await cursor.to_list(length=None)