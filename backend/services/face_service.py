import logging
from ml.embedding import extract_embeddings, cosine_similarity
from db.face_repo import save_embeddings, get_all_embeddings
from utils.image_utils import decode_base64_image
from datetime import datetime, time
from db.attendance_repo import save_attendance_log


logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.6

#the time for cutoff for late attendance (9:05 AM)
LATE_CUTOFF = time(9, 5)

async def register_student_faces(student_id: str, frames: list[str]) -> None:
    """
    For each frame:
      1. Decode base64 → PIL image
      2. Extract face embedding via FaceNet
      3. Persist all embeddings linked to student_id
    """
    logger.info(f"Processing {len(frames)} frames for student {student_id}")
    embeddings = []

    for i, frame_data_url in enumerate(frames):
        try:
            logger.info(f"Processing frame {i+1}/{len(frames)}")
            logger.info(f"Frame data URL length: {len(frame_data_url)}")
            image = decode_base64_image(frame_data_url)
            logger.info(f"Decoded image: {image.size}, mode: {image.mode}")

            embedding = extract_embeddings(image)
            logger.info(f"Extracted embedding: {embedding is not None}")

            if embedding is not None:
                embeddings.append(embedding)
                logger.info(f"Added embedding {len(embeddings)}")
            else:
                logger.warning(f"No face detected in frame {i+1}")

        except Exception as e:
            logger.error(f"Error processing frame {i+1}: {e}")
            raise

    logger.info(f"Total embeddings collected: {len(embeddings)}")

    if not embeddings:
        raise ValueError("No valid faces detected in any of the provided frames")

    await save_embeddings(student_id=student_id, embeddings=embeddings)

async def verify_face(
    frame_data_url: str,
    session_id: str,
    expected_student_id: str | None = None
) -> dict:

    logger.info("Starting face verification")

    # 1. Decode image
    image = decode_base64_image(frame_data_url)

    # 2. Extract embedding
    embedding = extract_embeddings(image)

    if embedding is None:
        logger.warning("No face detected")

        await save_attendance_log(
            student_id=expected_student_id or "unknown",
            status="absent",
            similarity=0.0,
        )

        return {"matched": False, "reason": "No face detected"}

    # 3. Get stored embeddings
    stored = await get_all_embeddings()

    if not stored:
        return {"matched": False, "reason": "No registered faces"}

    # 4. Find best match
    best_similarity = -1.0
    best_student_id = None

    for record in stored:
        sim = cosine_similarity(embedding, record.embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_student_id = record.student_id

    is_match = best_similarity >= SIMILARITY_THRESHOLD
    matched_expected = (
        expected_student_id is None or best_student_id == expected_student_id
    )

    passed = is_match and matched_expected

    student_id_for_check = (
        best_student_id if passed else (expected_student_id or "unknown")
    )
    await save_attendance_log(
        student_id=student_id_for_check,
        status="present" if passed else "absent",
        similarity=best_similarity,
    )
    # 5. Return result
    if passed:
        return {
            "matched": True,
            "student_id": best_student_id,
            "similarity": round(best_similarity, 4),
        }

    reason = "Face did not match any registered student"

    if is_match and expected_student_id and best_student_id != expected_student_id:
        reason = "Face matched a different registered student"

    return {
        "matched": False,
        "similarity": round(best_similarity, 4),
        "reason": reason,
    }