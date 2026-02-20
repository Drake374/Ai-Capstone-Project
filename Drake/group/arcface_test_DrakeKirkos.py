import os
import cv2
import torch
import numpy as np
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import shutil

# Paths
dataset_path = r"/content/dataset"
result_path = r"/content/attendance_results"
os.makedirs(result_path, exist_ok=True)

# Remove .ipynb_checkpoints if it exists in the dataset path
ipynb_checkpoints_path = os.path.join(dataset_path, "train", ".ipynb_checkpoints")
if os.path.exists(ipynb_checkpoints_path):
    shutil.rmtree(ipynb_checkpoints_path)
    print(f"Removed: {ipynb_checkpoints_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# InsightFace – RetinaFace + ArcFace
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0 if device == "cuda" else -1)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((112, 112)), # Removed transforms.ToPILImage() as ImageFolder already returns PIL images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Training dataset
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"])

train_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "train"),
    is_valid_file=lambda x: is_image_file(x) and '.ipynb_checkpoints' not in x,
    transform=transform # Add this line to apply the transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False
)

# Extract training embeddings
def get_train_embeddings():
    embeddings = []
    labels = []

    print("Extracting train embeddings...")

    for img, target in train_loader:
        # The image is already a tensor here due to the transform
        # No need for img.squeeze(0).permute(1, 2, 0).numpy() followed by (img * 255).astype(np.uint8)
        # Instead, convert tensor to numpy array for face_app.get
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 127.5 + 127.5).astype(np.uint8) # Denormalize to 0-255 range

        faces = face_app.get(img_np)
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

# Attendance evaluation
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


# Run
evaluate_test_images(threshold=0.6)
