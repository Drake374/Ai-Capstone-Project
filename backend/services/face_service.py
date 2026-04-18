import logging
from ml.embedding import extract_embeddings, cosine_similarity
from db.face_repo import save_embeddings, get_all_embeddings
from db.attendance_repo import save_attendance_log
from utils.image_utils import decode_base64_image
from datetime import datetime, time

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


async def verify_face(frame_data_url: str, expected_student_id: str | None = None) -> dict:
    """
    Verify a face against all stored embeddings using cosine similarity.
    Returns match result with student_id and similarity score.
    """
    logger.info("Starting face verification")

    # 1. Decode base64 frame → PIL image
    image = decode_base64_image(frame_data_url)
    logger.info(f"Decoded verification image: {image.size}, mode: {image.mode}")

    # 2. Extract embedding from the captured frame
    embedding = extract_embeddings(image)
    if embedding is None:
        logger.warning("No face detected in verification frame")
        return {"matched": False, "reason": "No face detected in the frame"}

    # 3. Fetch all stored embeddings from MongoDB
    stored = await get_all_embeddings()
    if not stored:
        logger.warning("No registered faces found in database")
        return {"matched": False, "reason": "No registered faces in the system"}

    # 4. Compute cosine similarity against each stored embedding
    best_similarity = -1.0
    best_student_id = None

    for record in stored:
        sim = cosine_similarity(embedding, record.embedding)
        logger.debug(f"Similarity with {record.student_id}: {sim:.4f}")
        if sim > best_similarity:
            best_similarity = sim
            best_student_id = record.student_id

    logger.info(f"Best match: student_id={best_student_id}, similarity={best_similarity:.4f}")

    is_match = best_similarity >= SIMILARITY_THRESHOLD
    matched_expected_student = (
        expected_student_id is None or best_student_id == expected_student_id
    )

    # 5. Save attendance log and return result
    if is_match and matched_expected_student:
        current_time = datetime.now().time()

        if current_time > LATE_CUTOFF:
            status = "late"
        else:
            status = "present"

        await save_attendance_log(
            student_id=best_student_id,
            status=status,
            similarity=best_similarity,
        )
        return {
            "matched": True,
            "student_id": best_student_id,
            "similarity": round(best_similarity, 4),
        }
    student_id_for_log = expected_student_id or best_student_id or "unknown"
    reason = "Face did not match any registered student"

    if is_match and expected_student_id and best_student_id != expected_student_id:
        reason = "Face matched a different registered student"

    await save_attendance_log(
        student_id=student_id_for_log,
        status="absent",
        similarity=best_similarity,
    )
    return {
        "matched": False,
        "similarity": round(best_similarity, 4),
        "reason": reason,
    }
