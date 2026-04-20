import os, cv2, numpy as np, torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics import accuracy_score

def cosine(a,b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a,b))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)

    # If you saved a finetuned backbone:
    finetuned_path = r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\models\backbone_finetuned.pt"
    if os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path, map_location=device))
        model.eval()

    # Expect a pairs file like: img1 img2 label(1 same, 0 diff)
    pairs_file = r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\data\lfw\pairs_simple.txt"
    y_true, y_pred = [], []

    threshold = 0.55  # tune on a dev split
    with open(pairs_file, "r") as f:
        for line in f:
            p1, p2, lab = line.strip().split()
            img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)

            f1 = mtcnn(img1)
            f2 = mtcnn(img2)
            if f1 is None or f2 is None:
                continue

            with torch.no_grad():
                e1 = model(f1.unsqueeze(0).to(device)).cpu().numpy()[0]
                e2 = model(f2.unsqueeze(0).to(device)).cpu().numpy()[0]

            sim = cosine(e1, e2)
            pred_same = 1 if sim >= threshold else 0
            y_true.append(int(lab)); y_pred.append(pred_same)

    acc = accuracy_score(y_true, y_pred)
    print("LFW verification accuracy:", acc)

if __name__ == "__main__":
    main()
