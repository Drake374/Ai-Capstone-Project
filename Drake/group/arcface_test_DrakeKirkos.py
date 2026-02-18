import os
import cv2
import torch
import numpy as np
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import insightface

# =============================
# Paths
# =============================
dataset_path = r"C:\Users\drake\Desktop\comp263\group\dataset"
result_path = r"C:\Users\drake\Desktop\comp263\group\attendance_results"
os.makedirs(result_path, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# InsightFace – RetinaFace + ArcFace
# =============================
face_app = FaceAnalysis(
    name="buffalo_l",  # RetinaFace + ArcFace
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0 if device == "cuda" else -1)

# ArcFace embedding model (512-D)
arcface = get_model("arcface_r100_v1")
arcface.prepare(ctx_id=0 if device == "cuda" else -1)

# =============================
# Image preprocessing
# =============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =============================
# Training dataset
# =============================
train_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "train")
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False
)

# =============================
# Extract training embeddings
# =============================
def get_train_embeddings():
    embeddings = []
    labels = []

    print("Extracting train embeddings...")

    for img, target in train_loader:
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        faces = face_app.get(img)
        if len(faces) == 0:
            continue

        face = faces[0]  # largest face
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        embeddings.append(emb)
        labels.append(target.item())

    print("Done.\n")
    return np.array(embeddings), np.array(labels)


train_embeddings, train_labels = get_train_embeddings()

# =============================
# Attendance evaluation
# =============================
def evaluate_test_images(threshold=0.6):

    test_folder = os.path.join(dataset_path, "test")
    attendance = set()

    print("Processing test images...\n")

    for root, _, files in os.walk(test_folder):
        for file in files:

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = face_app.get(img)

            if len(faces) == 0:
                print(f"No faces detected in: {file}")
            else:
                print(f"Detected {len(faces)} face(s) in: {file}")

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)

                similarities = cosine_similarity(
                    [embedding],
                    train_embeddings
                )

                best_idx = np.argmax(similarities)
                best_score = similarities[0][best_idx]

                if best_score > threshold:
                    identity = train_dataset.classes[
                        train_labels[best_idx]
                    ]
                    attendance.add(identity)
                else:
                    identity = "Unknown"

                # Draw bounding box
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{identity} | {best_score:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{timestamp}_{file}"
            cv2.imwrite(
                os.path.join(result_path, output_name),
                img
            )
            print("Saved:", output_name)

    print("\nAttendance List:", attendance)


# =============================
# Run
# =============================
evaluate_test_images(threshold=0.6)