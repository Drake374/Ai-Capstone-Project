from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
from models import AttendanceLog, AttendanceLogResponse
from datetime import datetime, timedelta

_client = AsyncIOMotorClient(settings.mongodb_url)
_col = _client[settings.mongodb_db]["attendance_logs"]


async def save_attendance_log(
    student_id: str, status: str, similarity: float
) -> None:
    """Save an attendance log entry."""
    log = AttendanceLog(
        student_id=student_id,
        status=status,
        similarity=similarity,
    )
    await _col.insert_one(log.dict())


async def get_attendance_logs(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[AttendanceLogResponse]:
    """
    Query attendance logs within a date range.
    Defaults to today if no dates provided.
    """
    if start_date is None:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if end_date is None:
        end_date = start_date + timedelta(days=1)

    query = {
        "timestamp": {
            "$gte": start_date,
            "$lt": end_date,
        }
    }

    cursor = _col.find(query, {"_id": 0}).sort("timestamp", -1)
    results = await cursor.to_list(length=None)
    return [AttendanceLogResponse(**doc) for doc in results]
