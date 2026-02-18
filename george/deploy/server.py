import os
import io
import base64
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

MODEL_PATH = r"C:\Users\georg\OneDrive\Desktop\centennial\2026 winter sem\comp385\facenet_mtcnn\finetuned_model.pth"
# -----------------------------
# Lifespan (loads model once on startup)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mtcnn, resnet, idx_to_label, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mtcnn = MTCNN(image_size=160, margin=20, device=device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    num_classes = checkpoint['num_classes']
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    resnet.logits = nn.Linear(512, num_classes).to(device)
    resnet.load_state_dict(checkpoint['model_state_dict'])
    resnet.eval()

    idx_to_label = checkpoint['idx_to_label']

    print(f"Model loaded. Classes: {list(checkpoint['label_to_idx'].keys())}")
    yield


# -----------------------------
# App
# -----------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request Schema
# -----------------------------
class ImagePayload(BaseModel):
    image: str  # base64 encoded image (data:image/jpeg;base64,...)


# -----------------------------
# Recognize Endpoint
# -----------------------------
@app.post("/recognize")
async def recognize(payload: ImagePayload):
    try:
        # Decode base64 image
        header, encoded = payload.image.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    # Detect faces
    boxes, probs = mtcnn.detect(img)

    if boxes is None:
        return {"faces": []}

    results = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)

        face = mtcnn(img.crop((x1, y1, x2, y2)))

        if face is None:
            continue

        face = face.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(face)
            logits = resnet.logits(embedding)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

        name = idx_to_label[predicted_idx] if confidence > 0.7 else "Unknown"

        results.append({
            "name": name,
            "confidence": round(confidence, 3),
            "box": [x1, y1, x2, y2]
        })

    return {"faces": results}


# -----------------------------
# Serve Frontend
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")