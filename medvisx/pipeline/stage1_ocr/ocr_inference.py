"""
OCR Inference Module
- Loads trained CRNN
- Runs inference on prescription image
- Fuzzy-matches output to known medicine names
"""
import os
import sys
import torch
import cv2
import numpy as np
from fuzzywuzzy import process
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEVICE, OCR_IMG_H, OCR_IMG_W, OCR_MODEL_PATH
from pipeline.stage1_ocr.crnn_model import CRNN, NUM_CLASSES
from pipeline.stage1_ocr.medicine_db import MEDICINE_DB, MED_ABBREVIATIONS
from pipeline.stage1_ocr.train_ocr import ctc_greedy_decode


class OCRInference:
    def __init__(self, model_path=OCR_MODEL_PATH):
        self.model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
            # Handle PyTorch Lightning checkpoints
            if 'state_dict' in ckpt:
                state = ckpt['state_dict']
            elif 'model_state' in ckpt:
                state = ckpt['model_state']
            else:
                state = ckpt
            # Try strict first, fall back to non-strict
            try:
                self.model.load_state_dict(state, strict=True)
            except RuntimeError:
                self.model.load_state_dict(state, strict=False)
                print(f"[OCR] Model loaded (non-strict) from {model_path}")
            else:
                print(f"[OCR] Model loaded from {model_path}")
        else:
            print("[OCR] WARNING: Model not found. Run training first or use text input.")
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(OCR_IMG_H, OCR_IMG_W),
            A.Normalize(mean=[0.5] * 3, std=[0.5] * 3, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        self.medicine_names = list(MEDICINE_DB.keys())

    def preprocess(self, image_path_or_array):
        """Preprocess image for CRNN input."""
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_path_or_array}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image_path_or_array)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return self.transform(image=img)['image'].unsqueeze(0).to(DEVICE)

    def fuzzy_correct(self, raw_text, threshold=80):
        """Map noisy OCR output to known medicine names using edit distance."""
        words = raw_text.lower().split()
        corrected = []
        for word in words:
            # Check abbreviations first
            for abbr, full in MED_ABBREVIATIONS.items():
                if abbr in word:
                    word = full
                    break
            # Fuzzy match to medicine database
            if self.medicine_names:
                best_match, score = process.extractOne(word, self.medicine_names)
                if score >= threshold:
                    corrected.append(best_match)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def extract_text(self, image_path_or_array):
        """Run OCR and return raw + corrected text."""
        tensor = self.preprocess(image_path_or_array)
        with torch.no_grad():
            logits = self.model(tensor)  # (T, 1, C)
        raw_text = ctc_greedy_decode(logits)[0]
        corrected = self.fuzzy_correct(raw_text)

        # Look up medicine info
        medicines_found = {}
        for med in self.medicine_names:
            if med in corrected.lower():
                medicines_found[med] = MEDICINE_DB[med]

        return {
            "raw_text": raw_text,
            "corrected_text": corrected,
            "medicines_found": medicines_found,
        }
