"""
OCR Training Script
Dataset: Doctor's Handwritten Prescription BD (Kaggle)
Loss: CTC Loss
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    DEVICE, OCR_IMG_H, OCR_IMG_W, OCR_CHARSET, OCR_BLANK_IDX,
    OCR_EPOCHS, OCR_BATCH_SIZE, OCR_LR, OCR_MODEL_PATH,
)
from pipeline.stage1_ocr.crnn_model import CRNN, NUM_CLASSES


# ── Dataset ───────────────────────────────────────────────────────────────
class PrescriptionDataset(Dataset):
    """
    Expects a CSV with columns: 'filename', 'label'
    and a folder of prescription image crops.
    Compatible with Doctor's Handwritten Prescription BD dataset from Kaggle.
    """
    def __init__(self, csv_path, img_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.augment = augment
        self.char2idx = {c: i + 1 for i, c in enumerate(OCR_CHARSET)}  # 0=blank

        base = [A.Resize(OCR_IMG_H, OCR_IMG_W)]
        aug = [
            A.Rotate(limit=3, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=3, p=0.5),
        ]
        norm = [
            A.Normalize(mean=[0.5] * 3, std=[0.5] * 3, max_pixel_value=255.0),
            ToTensorV2(),
        ]
        self.transform = A.Compose(base + (aug if augment else []) + norm)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((OCR_IMG_H, OCR_IMG_W, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.transform(image=img)['image']

        label_str = str(row['label']).strip()
        label_enc = [self.char2idx.get(c, 0) for c in label_str
                     if c in self.char2idx]
        label_enc = label_enc if label_enc else [0]
        return img, torch.tensor(label_enc, dtype=torch.long), len(label_enc)


def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    labels_padded = torch.zeros(len(labels), max(label_lengths), dtype=torch.long)
    for i, lbl in enumerate(labels):
        labels_padded[i, :len(lbl)] = lbl
    return images, labels_padded, label_lengths


# ── CTC Decode (greedy) ───────────────────────────────────────────────────
def ctc_greedy_decode(logits, charset=OCR_CHARSET):
    """
    Greedy CTC decoding: collapse repeats and remove blanks.
    logits: (T, B, C)
    """
    pred_idx = logits.argmax(dim=2)  # (T, B)
    results = []
    for b in range(pred_idx.shape[1]):
        seq = pred_idx[:, b].tolist()
        decoded = []
        prev = None
        for s in seq:
            if s != prev and s != OCR_BLANK_IDX:
                if 1 <= s <= len(charset):
                    decoded.append(charset[s - 1])
            prev = s
        results.append(''.join(decoded))
    return results


# ── Training Loop ─────────────────────────────────────────────────────────
def train_ocr(train_csv, train_img_dir, val_csv=None, val_img_dir=None,
              save_path=OCR_MODEL_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_ds = PrescriptionDataset(train_csv, train_img_dir, augment=True)
    train_dl = DataLoader(train_ds, batch_size=OCR_BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    ctc = nn.CTCLoss(blank=OCR_BLANK_IDX, reduction='mean', zero_infinity=True)
    optim = torch.optim.Adam(model.parameters(), lr=OCR_LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=OCR_EPOCHS)

    best_loss = float('inf')
    for epoch in range(1, OCR_EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, labels, label_lengths in tqdm(train_dl,
                                                 desc=f"Epoch {epoch}/{OCR_EPOCHS}"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)               # (T, B, C)
            log_probs = logits.log_softmax(2)  # required by CTCLoss

            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long)

            # Flatten labels for CTC
            flat_labels = []
            for i in range(B):
                flat_labels.extend(labels[i, :label_lengths[i]].tolist())
            flat_labels = torch.tensor(flat_labels, dtype=torch.long).to(DEVICE)

            loss = ctc(log_probs.cpu(), flat_labels.cpu(),
                       input_lengths, label_lengths)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        sched.step()
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state': model.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss}, save_path)
            print(f"  Saved best model (loss={best_loss:.4f})")

    return model


if __name__ == "__main__":
    # Update these paths to your dataset location
    TRAIN_CSV = "data/prescription_train.csv"
    TRAIN_IMG_DIR = "data/prescription_images/train"
    train_ocr(TRAIN_CSV, TRAIN_IMG_DIR)
