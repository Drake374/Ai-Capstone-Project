# src/inference/api_fastapi.py
from datetime import datetime
import csv
from pathlib import Path
from fastapi.responses import FileResponse

import os
import cv2
import tempfile
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, Form
from facenet_pytorch import MTCNN, InceptionResnetV1

from .liveness import LivenessONNX

#newly added
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


app = FastAPI()



#newly added
BASE_DIR = Path(__file__).resolve().parents[2]   # .../codefiles/capstone
FRONTEND_DIR = BASE_DIR / "frontend"


LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CSV_PATH = LOG_DIR / "attendance_log.csv"

def append_to_csv(row: dict):
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ----------------------------
# Device + Face detector
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# ----------------------------
# Face recognition model (pretrained + finetuned)
# Loads ONCE when API starts
# ----------------------------
model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)

finetuned_path = "models/backbone_finetuned.pt"
if os.path.exists(finetuned_path):
    model.load_state_dict(torch.load(finetuned_path, map_location=device))
    model.eval()
    print(f"[OK] Loaded finetuned backbone: {finetuned_path}")
else:
    print("[WARN] Finetuned backbone not found. Using pretrained vggface2 weights only.")

# ----------------------------
# Liveness (optional)
# Safe-load: won't crash if ONNX is missing/invalid
# ----------------------------
LIVENESS_MODEL = "models/liveness/anti_spoof_model.onnx"

liveness = None
if os.path.exists(LIVENESS_MODEL):
    try:
        liveness = LivenessONNX(LIVENESS_MODEL)
        print(f"[OK] Loaded liveness ONNX: {LIVENESS_MODEL}")
    except Exception as e:
        print(f"[WARN] Liveness ONNX invalid/unloadable. Skipping liveness. Error: {e}")
        liveness = None


def extract_frames(video_path: str, every_n: int = 4, max_frames: int = 15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


# @app.get("/")
# def root():
#     return {"status": "ok", "message": "Open /docs for Swagger UI"}


@app.post("/attendance")
async def attendance(student_id: str = Form(...), video: UploadFile = File(...)):
    # ----------------------------
    # Save uploaded video to temp file
    # ----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    frames = extract_frames(video_path, every_n=4, max_frames=15)
    os.unlink(video_path)

    if len(frames) < 5:
        return {"present": False, "reason": "video_too_short", "frames_extracted": len(frames)}

    # ----------------------------
    # Load enrolled template once per request
    # ----------------------------
    template_path = os.path.join("data", "enrolled", f"{student_id}.npy")
    if not os.path.exists(template_path):
        return {"present": False, "reason": "student_not_enrolled"}

    template = np.load(template_path)  # normalized in enroll.py

    sims = []
    live_scores = []

    # ----------------------------
    # Process frames (NO double face detection)
    # 1) mtcnn(rgb) returns aligned face tensor
    # 2) embed with model
    # 3) cosine similarity with template
    # ----------------------------
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aligned face tensor directly (single detection path)
        face_tensor = mtcnn(rgb)
        if face_tensor is None:
            continue

        # Face embedding
        with torch.no_grad():
            emb = model(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        # Cosine similarity (template already normalized)
        sim = float(np.dot(template, emb))
        sims.append(sim)

        # Optional liveness: needs a cropped face image
        if liveness is not None:
            boxes, _ = mtcnn.detect(rgb)
            if boxes is not None:
                x1, y1, x2, y2 = boxes[0].astype(int)
                face_bgr = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if face_bgr.size != 0:
                    live_scores.append(liveness.score(face_bgr))

    # If too few usable frames, fail
    if len(sims) < 3:
        return {
            "present": False,
            "reason": "no_clear_face_frames",
            "frames_extracted": len(frames),
            "usable_face_frames": len(sims),
        }

    sim_med = float(np.median(sims))
    live_med = float(np.median(live_scores)) if live_scores else None

    # Decision policy
    face_ok = sim_med >= 0.55
    live_ok = (live_med is None) or (live_med >= 0.70)  # tune if you add real liveness model

    timestamp = datetime.now().isoformat(timespec="seconds")

    log_row = {
    "timestamp": timestamp,
    "student_id": student_id,
    "present": bool(face_ok and live_ok),
    "similarity_median": sim_med,
    "liveness_median": live_med,
    "frames_extracted": len(frames),
    "usable_face_frames": len(sims),
    "reason": None if (face_ok and live_ok) else "face_mismatch_or_spoof",
    }

    append_to_csv(log_row)



    if face_ok and live_ok:
        return {
            "present": True,
            "similarity_median": sim_med,
            "liveness_median": live_med,
            "frames_extracted": len(frames),
            "usable_face_frames": len(sims),
        }

    return {
        "present": False,
        "reason": "face_mismatch_or_spoof",
        "similarity_median": sim_med,
        "liveness_median": live_med,
        "frames_extracted": len(frames),
        "usable_face_frames": len(sims),
    }


@app.get("/export")
def export_csv():
    if not CSV_PATH.exists():
        return {"error": "No attendance log found yet."}
    return FileResponse(
        path=str(CSV_PATH),
        filename="attendance_log.csv",
        media_type="text/csv"
    )
