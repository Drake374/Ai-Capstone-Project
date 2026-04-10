from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
from models import Student, StudentResponse

_client = AsyncIOMotorClient(settings.mongodb_url)
_col = _client[settings.mongodb_db]["students"]


async def upsert_student(
    student_id: str, name: str, email: str, photo_url: str = ""
) -> StudentResponse:
    """Create a new student or update if the email already exists."""
    existing = await _col.find_one({"email": email})

    if existing:
        # Update existing student with latest info
        await _col.update_one(
            {"email": email},
            {"$set": {"name": name, "photo_url": photo_url, "student_id": student_id, "role": "student"}},
        )
        updated = await _col.find_one({"email": email}, {"_id": 0})
        return StudentResponse(**updated)
    else:
        # Create new student
        student = Student(
            student_id=student_id,
            name=name,
            email=email,
            photo_url=photo_url,
        )
        await _col.insert_one(student.dict())
        return StudentResponse(**student.dict())


async def get_student_by_email(email: str) -> StudentResponse | None:
    """Find a student by email."""
    doc = await _col.find_one({"email": email}, {"_id": 0})
    if doc:
        return StudentResponse(**doc)
    return None


async def get_student(student_id: str) -> StudentResponse | None:
    """Find a student by student_id."""
    doc = await _col.find_one({"student_id": student_id}, {"_id": 0})
    if doc:
        return StudentResponse(**doc)
    return None


async def mark_registered(student_id: str) -> None:
    """Mark a student as having registered their face."""
    await _col.update_one(
        {"student_id": student_id},
        {"$set": {"registered": True}},
    )


async def get_all_students() -> list[StudentResponse]:
    """Get all students."""
    cursor = _col.find({}, {"_id": 0})
    results = await cursor.to_list(length=None)
    return [StudentResponse(**doc) for doc in results]
