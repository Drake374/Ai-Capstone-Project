import os, numpy as np, torch, cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

def cosine(a,b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a,b))

def verify_image(student_id: str, image_path: str, threshold=0.55, enrolled_dir="data/enrolled"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)

    finetuned_path = "models/backbone_finetuned.pt"
    if os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path, map_location=device))
        model.eval()

    template_path = os.path.join(enrolled_dir, f"{student_id}.npy")
    if not os.path.exists(template_path):
        return {"ok": False, "reason": "student_not_enrolled"}

    template = np.load(template_path)

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    face = mtcnn(img)
    if face is None:
        return {"ok": False, "reason": "no_face_detected"}

    with torch.no_grad():
        emb = model(face.unsqueeze(0).to(device)).cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    sim = cosine(template, emb)
    return {"ok": sim >= threshold, "similarity": sim}
