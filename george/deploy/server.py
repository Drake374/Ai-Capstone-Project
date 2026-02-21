import os
import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from facenet_pytorch import MTCNN, InceptionResnetV1
from contextlib import asynccontextmanager
from typing import List

ROOT_FOLDER = Path.cwd().parent
MODEL_PATH  = ROOT_FOLDER / "finetuned_model.pth"
DATASET_DIR = ROOT_FOLDER / "dataset" / "train"


# ─────────────────────────────────────────────
# Lifespan – load model once on startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mtcnn, resnet, idx_to_label, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mtcnn = MTCNN(image_size=160, margin=20, device=device)

    checkpoint   = torch.load(MODEL_PATH, map_location=device)
    num_classes  = checkpoint["num_classes"]
    resnet       = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    resnet.logits = nn.Linear(512, num_classes).to(device)
    resnet.load_state_dict(checkpoint["model_state_dict"])
    resnet.eval()
    idx_to_label = checkpoint["idx_to_label"]

    print(f"Model loaded. Classes: {list(checkpoint['label_to_idx'].keys())}")
    yield


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class ImagePayload(BaseModel):
    image: str  # data:image/jpeg;base64,...

class RegisterPayload(BaseModel):
    name:   str
    images: List[str]  # list of data:image/jpeg;base64,... strings


# ─────────────────────────────────────────────
# /recognize
# ─────────────────────────────────────────────
@app.post("/recognize")
async def recognize(payload: ImagePayload):
    try:
        header, encoded = payload.image.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return {"faces": []}

    results = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)

        face = mtcnn(img.crop((x1, y1, x2, y2)))
        if face is None:
            continue

        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding     = resnet(face)
            logits        = resnet.logits(embedding)
            probs_t       = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs_t, 1)
            confidence    = confidence.item()
            predicted_idx = predicted_idx.item()

        name = idx_to_label[predicted_idx] if confidence > 0.7 else "Unknown"
        results.append({"name": name, "confidence": round(confidence, 3), "box": [x1, y1, x2, y2]})

    return {"faces": results}


# ─────────────────────────────────────────────
# /register  – save images to dataset folder
# ─────────────────────────────────────────────
@app.post("/register")
async def register(payload: RegisterPayload):
    name = payload.name.strip().lower().replace(" ", "_")
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if not payload.images:
        raise HTTPException(status_code=400, detail="No images provided.")

    save_dir = DATASET_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Start numbering after any existing images
    existing = len(list(save_dir.glob("*.jpg")))
    saved = 0
    for i, data_url in enumerate(payload.images):
        try:
            _, encoded = data_url.split(",", 1)
            img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            img.save(save_dir / f"{existing + i}.jpg", "JPEG", quality=92)
            saved += 1
        except Exception as e:
            print(f"Failed to save image {i}: {e}")

    if saved == 0:
        raise HTTPException(status_code=500, detail="Failed to save any images.")

    print(f"[Register] Saved {saved} images for '{name}' → {save_dir}")
    return {"saved": saved, "path": str(save_dir)}


# ─────────────────────────────────────────────
# Serve Frontend
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/register")
def register_page():
    return FileResponse("static/register.html")

# python -m uvicorn server:app --host 0.0.0.0 --port 8000