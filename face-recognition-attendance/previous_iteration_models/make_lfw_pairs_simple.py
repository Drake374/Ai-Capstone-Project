import os
import random
from pathlib import Path

def make_pairs(lfw_root: str, out_file: str, same_pairs=3000, diff_pairs=3000, seed=42):
    random.seed(seed)
    lfw_root = Path(lfw_root)

    # Collect people -> list of image paths
    people = {}
    for person_dir in lfw_root.iterdir():
        if person_dir.is_dir():
            imgs = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
            if len(imgs) >= 2:
                people[person_dir.name] = imgs

    names = list(people.keys())
    if len(names) < 2:
        raise ValueError("Not enough identities with >=2 images in LFW folder.")

    pairs = []

    # SAME pairs
    for _ in range(same_pairs):
        name = random.choice(names)
        a, b = random.sample(people[name], 2)
        pairs.append((str(a.resolve()), str(b.resolve()), 1))

    # DIFF pairs
    for _ in range(diff_pairs):
        n1, n2 = random.sample(names, 2)
        a = random.choice(people[n1])
        b = random.choice(people[n2])
        pairs.append((str(a.resolve()), str(b.resolve()), 0))

    random.shuffle(pairs)

    os.makedirs(Path(out_file).parent, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for p1, p2, lab in pairs:
            f.write(f"{p1} {p2} {lab}\n")

    print("✅ Saved pairs file:", out_file)
    print("Total pairs:", len(pairs))

if __name__ == "__main__":
    # Your LFW funneled path (from your message)
    LFW_ROOT = r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\lfwfunneled\lfw_funneled"
    OUT_FILE = r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\data\lfw\pairs_simple.txt"

    make_pairs(LFW_ROOT, OUT_FILE, same_pairs=3000, diff_pairs=3000, seed=42)
