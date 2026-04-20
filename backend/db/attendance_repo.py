from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
from models import AttendanceLog, AttendanceLogResponse, CheckResult, CheckResultResponse
from datetime import datetime, timedelta
from typing import List

_client = AsyncIOMotorClient(settings.mongodb_url)
_col = _client[settings.mongodb_db]["attendance_logs"]
_check_col = _client[settings.mongodb_db]["check_results"]


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


async def save_check_result(
    student_id: str, session_id: str, passed: bool, similarity: float
) -> None:
    """Save a check result for a session."""
    check = CheckResult(
        student_id=student_id,
        session_id=session_id,
        passed=passed,
        similarity=similarity,
    )
    await _check_col.insert_one(check.dict())


async def get_checks_for_session(session_id: str) -> List[CheckResultResponse]:
    """Get all check results for a session."""
    query = {"session_id": session_id}
    cursor = _check_col.find(query, {"_id": 0}).sort("timestamp", 1)
    results = await cursor.to_list(length=None)
    return [CheckResultResponse(**doc) for doc in results]


async def calculate_attendance_for_session(session_id: str, student_id: str) -> str:
    """Calculate attendance status based on check results for a student in a session."""
    checks = await get_checks_for_session(session_id)
    student_checks = [c for c in checks if c.student_id == student_id]
    
    if not student_checks:
        return "absent"
    
    total_checks = len(student_checks)
    passed_checks = sum(1 for c in student_checks if c.passed)
    
    # Rule 1: If passes all checks -> present
    if passed_checks == total_checks:
        return "present"
    
    # Rule 2: If passes first check but not rest -> absent
    if student_checks[0].passed and passed_checks == 1:
        return "absent"
    
    # Rule 3: If passes almost half and not rest -> partial
    if passed_checks >= total_checks // 2:
        return "partial"
    
    # Rule 4: If missed first few but joined later and finished rest -> partial
    # Assuming "first few" means more than 1 missed at start
    missed_at_start = 0
    for check in student_checks:
        if not check.passed:
            missed_at_start += 1
        else:
            break
    if missed_at_start > 1 and passed_checks > missed_at_start:
        return "partial"
    
    # Rule 5: If only finished last 3 checks -> partial
    # Assuming total checks > 3
    if total_checks > 3:
        last_3_passed = sum(1 for c in student_checks[-3:] if c.passed)
        if last_3_passed == 3 and passed_checks == 3:
            return "partial"
    
    # Rule 6: If only finishes last check but failed others -> absent
    if student_checks[-1].passed and passed_checks == 1:
        return "absent"
    
    # Default to absent if none match
    return "absent"


async def finalize_attendance_for_session(session_id: str) -> None:
    """Finalize attendance for all students in a session."""
    checks = await get_checks_for_session(session_id)
    student_ids = set(c.student_id for c in checks)
    
    for student_id in student_ids:
        status = await calculate_attendance_for_session(session_id, student_id)
        # Save with average similarity or something; for now, use 0.0
        avg_similarity = sum(c.similarity for c in checks if c.student_id == student_id) / len([c for c in checks if c.student_id == student_id])
        await save_attendance_log(student_id, status, avg_similarity)
