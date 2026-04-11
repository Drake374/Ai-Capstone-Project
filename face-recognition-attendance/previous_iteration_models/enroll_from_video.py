import os, cv2, numpy as np, torch
from facenet_pytorch import MTCNN, InceptionResnetV1

def extract_frames(video_path, every_n=2, max_frames=30):
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

def enroll_from_video(student_id: str, video_path: str, out_dir="data/enrolled"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=160, margin=20, device=device)

    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    finetuned_path = "models/backbone_finetuned.pt"
    if os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path, map_location=device))
        model.eval()

    frames = extract_frames(video_path)
    embs = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is None:
            continue
        with torch.no_grad():
            emb = model(face.unsqueeze(0).to(device)).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        embs.append(emb)

    if len(embs) < 5:
        raise RuntimeError(f"Not enough usable face frames. Got {len(embs)}")

    template = np.mean(np.stack(embs), axis=0)
    template = template / (np.linalg.norm(template) + 1e-9)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{student_id}.npy")
    np.save(out_path, template)
    print("✅ Saved:", out_path, "frames_used:", len(embs))

if __name__ == "__main__":
    student_id = "numaan_1"
    video_path = r"data\samples\numaan_enroll.mp4"
    enroll_from_video(student_id, video_path)
