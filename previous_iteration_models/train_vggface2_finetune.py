# {
#  "cells": [],
#  "metadata": {
#   "language_info": {
#    "name": "python"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 5
# }


import os, yaml, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from facenet_pytorch import InceptionResnetV1
from dataset_vggface2 import VGGFace2Folder

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    cfg = yaml.safe_load(open(r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\configs\train.yaml", "r"))
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    img_size = cfg["data"]["img_size"]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    train_root = cfg["data"]["vggface2_root"]

    print("train_root from yaml =", train_root, "| type =", type(train_root))

    ds = VGGFace2Folder(
        train_root,
        transform=tfm,
        max_identities=cfg["data"]["max_identities"],
        max_images_per_id=cfg["data"]["max_images_per_id"],
    )
    dl = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                    num_workers=cfg["data"]["num_workers"], pin_memory=True)

    num_classes = len(ds.class_to_idx)
    print(f"Loaded samples={len(ds)} classes={num_classes}")

    # 1) Pretrained backbone (VGGFace2)
    backbone = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    # 512-d embeddings
    head = nn.Linear(512, num_classes).to(device)

    # Train head only first
    for p in backbone.parameters():
        p.requires_grad = False

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr_head"], weight_decay=cfg["train"]["weight_decay"])

    def run_epoch(train=True):
        backbone.eval()  # frozen
        head.train() if train else head.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in tqdm(dl, desc="train" if train else "eval"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                emb = backbone(x)
            logits = head(emb)
            loss = ce(logits, y)

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        return total_loss/total, total_correct/total

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    for ep in range(cfg["train"]["epochs_head"]):
        loss, acc = run_epoch(train=True)
        print(f"[HEAD] epoch={ep+1} loss={loss:.4f} acc={acc:.4f}")

    # 2) Fine-tune last blocks (unfreeze a portion)
    for name, p in backbone.named_parameters():
        if "block8" in name or "last_linear" in name or "last_bn" in name:
            p.requires_grad = True

    opt2 = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, backbone.parameters())) + list(head.parameters()),
        lr=cfg["train"]["lr_finetune"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    for ep in range(cfg["train"]["epochs_finetune"]):
        backbone.train()
        head.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in tqdm(dl, desc="finetune"):
            x, y = x.to(device), y.to(device)
            emb = backbone(x)
            logits = head(emb)
            loss = ce(logits, y)

            opt2.zero_grad()
            loss.backward()
            opt2.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        print(f"[FT] epoch={ep+1} loss={total_loss/total:.4f} acc={total_correct/total:.4f}")

    # Save only backbone for verification-style use
    torch.save(backbone.state_dict(), cfg["output"]["finetuned_backbone_path"])
    print("Saved:", cfg["output"]["finetuned_backbone_path"])

if __name__ == "__main__":
    main()



