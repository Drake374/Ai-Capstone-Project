import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Initialise once at import time (CPU-friendly for small scale)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {_device}")
_project_root = Path(__file__).resolve().parents[2]
_custom_model_path = Path(settings.custom_model_path)

if not _custom_model_path.is_absolute():
    _custom_model_path = _project_root / _custom_model_path

_mtcnn = None
_resnet = None

try:
    _mtcnn = MTCNN(image_size=160, margin=20, device=_device, keep_all=False)
    _resnet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(_device)

    if _custom_model_path.exists():
        state_dict = torch.load(_custom_model_path, map_location=_device)
        _resnet.load_state_dict(state_dict)
        logger.info(f"Loaded finetuned FaceNet backbone from {_custom_model_path}")
    else:
        logger.warning(
            f"Finetuned backbone not found at {_custom_model_path}. "
            "Falling back to pretrained vggface2 weights."
        )

    logger.info("FaceNet models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load FaceNet models: {e}")
    logger.info("Face recognition will be disabled")


def extract_embeddings(image: Image.Image) -> list[float] | None:
    """
    Detect a face in the image and return a 512-dim embedding as a plain list,
    or None if no face is detected.
    """
    if _mtcnn is None or _resnet is None:
        logger.error("FaceNet models not loaded - cannot extract embeddings")
        raise RuntimeError("Face recognition models failed to load. Check NumPy compatibility.")

    try:
        logger.info(f"Processing image: {image.size}, mode: {image.mode}")

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.debug("Converted image to RGB")

        face_tensor = _mtcnn(image)  # returns (3, 160, 160) tensor or None
        logger.debug(f"Face detection result: {face_tensor is not None}")

        if face_tensor is None:
            return None

        face_tensor = face_tensor.unsqueeze(0).to(_device)  # (1, 3, 160, 160)

        with torch.no_grad():
            embedding = _resnet(face_tensor)  # (1, 512)
            logger.debug(f"Embedding shape: {embedding.shape}")

        # Convert to numpy and then to list
        embedding_np = embedding.squeeze().cpu().numpy()
        embedding_list = embedding_np.tolist()

        logger.debug(f"Embedding extracted successfully, length: {len(embedding_list)}")
        return embedding_list

    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    dot = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
