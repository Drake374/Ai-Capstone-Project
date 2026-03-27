from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta
from db.attendance_repo import get_attendance_logs
from db.student_repo import get_student
import csv
import io

router = APIRouter()


@router.get("/attendance-logs")
async def attendance_logs(
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Get attendance logs within a date range. Defaults to today."""
    start = _parse_date(start_date)
    end = _parse_end_date(end_date)

    logs = await get_attendance_logs(start_date=start, end_date=end)

    # Lookup student names
    results = []
    for log in logs:
        student = await get_student(log.student_id)
        results.append({
            "student_id": log.student_id,
            "student_name": student.name if student else "Unknown",
            "status": log.status,
            "similarity": round(log.similarity, 4),
            "timestamp": log.timestamp.isoformat(),
        })

    return results


@router.get("/attendance-logs/export")
async def export_attendance_logs(
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Export attendance logs as a CSV file download."""
    start = _parse_date(start_date)
    end = _parse_end_date(end_date)

    logs = await get_attendance_logs(start_date=start, end_date=end)

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Student Name", "Status", "Similarity", "Timestamp"])

    for log in logs:
        student = await get_student(log.student_id)
        writer.writerow([
            log.student_id,
            student.name if student else "Unknown",
            log.status,
            f"{log.similarity:.4f}",
            log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        ])

    output.seek(0)

    # Determine filename based on date range
    start_str = (start or datetime.now()).strftime("%Y-%m-%d")
    end_str = (end or datetime.now()).strftime("%Y-%m-%d")
    filename = f"attendance_logs_{start_str}_to_{end_str}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD string to datetime (start of day), or None."""
    if not date_str:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d")


def _parse_end_date(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD string to datetime (end of day), or None."""
    if not date_str:
        return None
    # End date should be the END of that day (start of next day)
    return datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)

