# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:36:31 2026

@author: admi
"""


import os
import glob
import time
import pickle
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image

# Optional (only used for webcam mode)
try:
    import cv2
except Exception:
    cv2 = None

from facenet_pytorch import MTCNN, InceptionResnetV1


# Model setup
def build_models(device: str):
    """Create face detector (MTCNN) and embedding model (FaceNet)."""
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        post_process=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet


# Embedding + similarity
def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # assumes normalized vectors
    return float(np.dot(a, b))


def get_embedding_from_pil(
    img: Image.Image,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: str
) -> Optional[np.ndarray]:
    """Return a 512-d normalized embedding or None if no face detected."""
    img = img.convert("RGB")
    face = mtcnn(img)  # tensor [3, 160, 160] or None
    if face is None:
        return None

    face = face.unsqueeze(0).to(device)  # [1, 3, 160, 160]
    with torch.no_grad():
        emb = resnet(face).cpu().numpy()[0]  # [512]
    return l2_normalize(emb)


def get_embedding_from_path(
    img_path: str,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: str
) -> Optional[np.ndarray]:
    img = Image.open(img_path)
    return get_embedding_from_pil(img, mtcnn, resnet, device)


# Enrollment
def build_template_db(
    enroll_dir: str,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: str
) -> Dict[str, np.ndarray]:
    """
    Build template DB:
      { person_folder_name: mean_embedding }
    Each person folder may contain multiple images.
    """
    db: Dict[str, np.ndarray] = {}

    if not os.path.isdir(enroll_dir):
        raise FileNotFoundError(f"Enrollment directory not found: {enroll_dir}")

    people = sorted([p for p in os.listdir(enroll_dir) if os.path.isdir(os.path.join(enroll_dir, p))])
    if not people:
        raise ValueError(f"No person folders found in: {enroll_dir}")

    for person in people:
        folder_path = os.path.join(enroll_dir, person)
        img_paths = glob.glob(os.path.join(folder_path, "*.*"))

        embs = []
        for pth in img_paths:
            try:
                emb = get_embedding_from_path(pth, mtcnn, resnet, device)
                if emb is not None:
                    embs.append(emb)
            except Exception:
                # skip unreadable images
                continue

        if len(embs) == 0:
            print(f"[SKIP] {person}: no detectable faces")
            continue

        mean_emb = np.mean(np.stack(embs), axis=0)
        mean_emb = l2_normalize(mean_emb)
        db[person] = mean_emb
        print(f"[OK] Enrolled {person} using {len(embs)} image(s)")

    if not db:
        raise ValueError("Enrollment finished but DB is empty (no faces detected).")

    return db


def save_db(db: Dict[str, np.ndarray], out_path: str) -> None:
    with open(out_path, "wb") as f:
        pickle.dump(db, f)
    print(f"[SAVED] Template DB -> {out_path}")


def load_db(db_path: str) -> Dict[str, np.ndarray]:
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    if not isinstance(db, dict) or len(db) == 0:
        raise ValueError("Loaded DB is empty or invalid.")
    return db


# Recognition
def recognize_image(
    img_path: str,
    db: Dict[str, np.ndarray],
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: str,
    threshold: float
) -> Tuple[str, float]:
    emb = get_embedding_from_path(img_path, mtcnn, resnet, device)
    if emb is None:
        return "Unknown", 0.0

    best_id = "Unknown"
    best_score = -1.0

    for person_id, template in db.items():
        score = cosine_similarity(emb, template)
        if score > best_score:
            best_score = score
            best_id = person_id

    if best_score >= threshold:
        return best_id, best_score
    return "Unknown", best_score


# Webcam mode 
def webcam_recognition(
    db: Dict[str, np.ndarray],
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: str,
    threshold: float,
    camera_index: int = 0
) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run: pip install opencv-python")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print("[INFO] Press 'q' to quit webcam mode.")
    last_print = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Convert BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        emb = get_embedding_from_pil(pil_img, mtcnn, resnet, device)
        label = "No Face"
        score = 0.0

        if emb is not None:
            best_id = "Unknown"
            best_score = -1.0
            for person_id, template in db.items():
                s = cosine_similarity(emb, template)
                if s > best_score:
                    best_score = s
                    best_id = person_id
            label = best_id if best_score >= threshold else "Unknown"
            score = best_score

        # Draw text overlay
        text = f"{label}  (score={score:.3f})"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("FaceNet Attendance - Webcam", frame)

        # Print occasionally (optional)
        if time.time() - last_print > 2.0:
            print(f"[LIVE] {label} (score={score:.3f})")
            last_print = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# CLI
def main():
    parser = argparse.ArgumentParser(description="FaceNet (facenet-pytorch) attendance pipeline in one file.")
    parser.add_argument("--enroll_dir", type=str, default=None, help="Folder with subfolders per person for enrollment.")
    parser.add_argument("--save_db", type=str, default=None, help="Path to save templates DB (pkl).")

    parser.add_argument("--db", type=str, default=None, help="Path to an existing templates DB (pkl).")
    parser.add_argument("--test_image", type=str, default=None, help="Image path to recognize.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Cosine similarity threshold (typical 0.55-0.70).")

    parser.add_argument("--webcam", action="store_true", help="Run realtime webcam recognition.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    mtcnn, resnet = build_models(device)

    # 1) Enrollment mode (optional)
    db = None
    if args.enroll_dir:
        db = build_template_db(args.enroll_dir, mtcnn, resnet, device)
        if args.save_db:
            save_db(db, args.save_db)

    # 2) Load DB (if not already built)
    if db is None:
        if not args.db:
            raise ValueError("No DB provided. Use --enroll_dir (and optionally --save_db) or --db to load.")
        db = load_db(args.db)
        print(f"[LOADED] Template DB <- {args.db} ({len(db)} identities)")

    # 3) Recognize single image
    if args.test_image:
        pred, score = recognize_image(
            args.test_image, db, mtcnn, resnet, device, args.threshold
        )
        print(f"[RESULT] {args.test_image} -> {pred} (score={score:.4f})")

    # 4) Webcam mode
    if args.webcam:
        webcam_recognition(db, mtcnn, resnet, device, args.threshold, args.camera)


if __name__ == "__main__":
    main()