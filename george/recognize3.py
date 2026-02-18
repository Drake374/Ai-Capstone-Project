# %%
#%%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime


# %%

# -----------------------------
# Device
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# %%

# -----------------------------
# Models
# -----------------------------
mtcnn = MTCNN(image_size=160, margin=20, device=device)


# %%

# -----------------------------
# Custom Dataset Class
# -----------------------------
class FaceDataset(Dataset):
    def __init__(self, dataset_path, mtcnn, transform=None):
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # Build label mappings
        person_names = sorted(os.listdir(dataset_path))
        for idx, person_name in enumerate(person_names):
            self.label_to_idx[person_name] = idx
            self.idx_to_label[idx] = person_name
        
        # Load and process all images
        print("Loading dataset...")
        for person_name in person_names:
            person_folder = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_folder):
                continue
                
            print(f"  Loading {person_name}...")
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    face = mtcnn(img)
                    
                    if face is not None:
                        self.samples.append(face)
                        self.labels.append(self.label_to_idx[person_name])
                except Exception as e:
                    print(f"    Error loading {img_name}: {e}")
        
        print(f"Loaded {len(self.samples)} face images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# %%

# -----------------------------
# Load and Modify Model
# -----------------------------
DATASET_PATH = "C:/Users/georg/OneDrive/Desktop/centennial/2026 winter sem/comp385/facenet_mtcnn/dataset/train"
TEST_PATH = "C:/Users/georg/OneDrive/Desktop/centennial/2026 winter sem/comp385/facenet_mtcnn/dataset/test"
TEST_RESULT_PATH = "C:/Users/georg/OneDrive/Desktop/centennial/2026 winter sem/comp385/facenet_mtcnn/testing_results/"

# Count number of classes
num_classes = len([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
print(f"Number of classes: {num_classes}")

# Load pretrained model
resnet = InceptionResnetV1(
    pretrained='vggface2',
    classify=False  # We'll add our own classifier
).to(device)

# Freeze early layers (optional - fine-tune only last layers)
for param in resnet.parameters():
    param.requires_grad = False

# Unfreeze last few blocks for fine-tuning
for param in resnet.block8.parameters():
    param.requires_grad = True
for param in resnet.last_linear.parameters():
    param.requires_grad = True
for param in resnet.last_bn.parameters():
    param.requires_grad = True

# Add classification head
resnet.logits = nn.Linear(512, num_classes).to(device)  # 512 is embedding size


# %%

# -----------------------------
# Fine-tune Model
# -----------------------------

# Create dataset and dataloader
dataset = FaceDataset(DATASET_PATH, mtcnn)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Store label mappings for later use
label_to_idx = dataset.label_to_idx
idx_to_label = dataset.idx_to_label

print("\nLabel mappings:")
for name, idx in label_to_idx.items():
    print(f"  {name}: {idx}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001)

# Training loop
num_epochs = 10
print(f"\nTraining for {num_epochs} epochs...")

resnet.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for faces, labels in dataloader:
        faces = faces.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = resnet(faces)  # Get embeddings
        outputs = resnet.logits(embeddings)  # Classify
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

resnet.eval()
print("Training complete!")

# Save the fine-tuned model
torch.save({
    'model_state_dict': resnet.state_dict(),
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_classes': num_classes
}, 'finetuned_model.pth')
print("Model saved to finetuned_model.pth")


# %%

# -----------------------------
# Recognition Function
# -----------------------------
def recognize_face(image_path, confidence_threshold=0.7):
    
    img = Image.open(image_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    file_name = ""

    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        print("No face detected")
        return

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        face = mtcnn(img.crop((x1, y1, x2, y2)))

        if face is None:
            continue

        face = face.unsqueeze(0).to(device)
        
        # Get prediction from fine-tuned model
        with torch.no_grad():
            embedding = resnet(face)
            logits = resnet.logits(embedding)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
        
        if confidence > confidence_threshold:
            name = idx_to_label[predicted_idx]
        else:
            name = "Unknown"

        # Draw result
        cv2.rectangle(img_cv, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img_cv, f"{name} ({confidence:.2f})",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)
        
        #file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_result_{name}_{confidence:.2f}.jpg"
        
    
    # cv2.imshow("Result", img_cv)
    cv2.imwrite(f"{TEST_RESULT_PATH+file_name}", img_cv)
    print(file_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# %%

# -----------------------------
# Run Test
# -----------------------------
recognize_face(f"{TEST_PATH}/test1.jpg")
recognize_face(f"{TEST_PATH}/test2.jpg")
recognize_face(f"{TEST_PATH}/test3.jpg")


