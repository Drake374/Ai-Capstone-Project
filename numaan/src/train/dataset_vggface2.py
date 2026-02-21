import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class VGGFace2Folder(Dataset):
    """
    Expects:
      data/vggface2/train/<identity_id>/*.jpg
    """
    def __init__(self, root, transform=None, max_identities=None, max_images_per_id=None):
        self.root = Path(root)
        self.transform = transform

        identities = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if max_identities is not None:
            identities = identities[:max_identities]

        self.class_to_idx = {ident.name: i for i, ident in enumerate(identities)}

        self.samples = []
        for ident in identities:
            imgs = sorted(list(ident.glob("*.jpg")) + list(ident.glob("*.png")) + list(ident.glob("*.jpeg")))
            if max_images_per_id is not None and len(imgs) > max_images_per_id:
                imgs = random.sample(imgs, max_images_per_id)
            for img_path in imgs:
                self.samples.append((str(img_path), self.class_to_idx[ident.name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
