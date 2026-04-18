import os, numpy as np, torch, cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

def enroll_from_images(student_id: str, image_paths: list[str], out_dir="data/enrolled"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)

    finetuned_path = "models/backbone_finetuned.pt"
    if os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path, map_location=device))
        model.eval()

    embs = []
    for p in image_paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        face = mtcnn(img)
        if face is None:
            continue
        with torch.no_grad():
            emb = model(face.unsqueeze(0).to(device)).cpu().numpy()[0]
        embs.append(emb)

    if len(embs) < 3:
        raise ValueError("Not enough detectable faces to enroll. Provide clearer images.")

    template = np.mean(np.stack(embs), axis=0)
    template = template / (np.linalg.norm(template) + 1e-9)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{student_id}.npy"), template)
    return os.path.join(out_dir, f"{student_id}.npy")
