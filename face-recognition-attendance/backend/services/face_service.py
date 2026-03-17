import logging
from ml.embedding import extract_embeddings
from db.face_repo import save_embeddings
from utils.image_utils import decode_base64_image

logger = logging.getLogger(__name__)


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