# MedVis-X — Complete Project Master Prompt (Fully Local — No External APIs)
# Paste this ENTIRE prompt into Claude (VS Code) to build the project
# Everything runs 100% locally after a one-time HuggingFace model download.

---

## CONTEXT & PROJECT OVERVIEW

You are building **MedVis-X**, a Traceable and Explainable Multimodal Clinical Decision Support System. This is both a research project (IEEE paper) and a deployable product. You must build it as a complete, working, production-ready Python application. **Everything runs 100% locally — no cloud APIs, no API keys, no internet at inference time.** After a one-time HuggingFace model download, the system is fully offline. **Everything runs 100% locally — no cloud APIs, no API keys, no internet at inference time.** After a one-time HuggingFace model download, the system is fully offline.

**The system has 3 core AI components connected in a 6-stage pipeline:**

```
[Clinical Document / Prescription]
        ↓
  STAGE 1: OCR (CRNN Model — ResNet18 + BiLSTM + CTC)
        ↓
  STAGE 2: NER (ClinicalBERT — extract symptoms, diseases, medications)
        ↓
  STAGE 3: Hypothesis Scoring (Deterministic weighted scoring + SHAP)
        ↓
  STAGE 4: Image Generation (LoRA Stable Diffusion v1.5)
        ↓
  STAGE 5: Grad-CAM Explainability Overlay
        ↓
  STAGE 6: LLaVA-1.5-7B (4-bit quantised) — Fully Local Multimodal XAI
        ↓
  [Final Output: Report + Generated Image + Heatmap + LLaVA Explanation]
```

---

## TECHNOLOGY STACK

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch, HuggingFace Diffusers, PEFT (LoRA), Transformers
- **OCR**: Custom CRNN (ResNet18 + BiLSTM + CTC Loss)
- **NER**: ClinicalBERT (`medicalai/ClinicalBERT`)
- **Image Generation**: Stable Diffusion v1.5 with LoRA weights
- **XAI Visual**: Grad-CAM (pytorch-grad-cam) on DenseNet121 backbone — no training needed
- **XAI Text**: SHAP waterfall plots on scoring function — no training needed
- **XAI Multimodal**: LLaVA-1.5-7B-hf in 4-bit NF4 quantisation — **fully local, no API key**
- **Quantisation**: bitsandbytes (4-bit NF4) — fits LLaVA in ~5 GB VRAM
- **UI**: Gradio (web app, runs in browser)
- **Hardware target**: Single NVIDIA GPU with 8GB VRAM (RTX 4060 / T4)

---

## COMPLETE PROJECT STRUCTURE

Create this exact folder structure:

```
medvisx/
├── app.py                    ← Main Gradio web app entry point
├── requirements.txt          ← All dependencies
├── config.py                 ← All hyperparameters and paths
├── pipeline/
│   ├── __init__.py
│   ├── full_pipeline.py      ← Orchestrates all 6 stages end-to-end
│   ├── stage1_ocr/
│   │   ├── __init__.py
│   │   ├── crnn_model.py     ← CRNN architecture definition
│   │   ├── train_ocr.py      ← OCR training script
│   │   ├── ocr_inference.py  ← OCR inference + fuzzy post-processing
│   │   └── medicine_db.py    ← Medicine-to-disease mapping database
│   ├── stage2_ner/
│   │   ├── __init__.py
│   │   └── ner_extractor.py  ← ClinicalBERT NER
│   ├── stage3_scoring/
│   │   ├── __init__.py
│   │   ├── hypothesis_scorer.py  ← Weighted deterministic scoring
│   │   └── symptom_disease_db.py ← Curated symptom-disease database
│   ├── stage4_generation/
│   │   ├── __init__.py
│   │   ├── lora_generator.py ← LoRA SD v1.5 inference
│   │   └── prompt_builder.py ← Builds clinical prompts from hypotheses
│   ├── stage5_explainability/
│   │   ├── __init__.py
│   │   ├── gradcam.py        ← Grad-CAM on diffusion UNet
│   │   └── shap_explainer.py ← SHAP on scoring function
│   └── stage6_local_llm/
│       ├── __init__.py
│       └── llava_explainer.py ← LLaVA-1.5-7B (local) clinical explanation
├── training/
│   ├── train_ocr.py          ← Standalone OCR training runner
│   └── train_lora.py         ← LoRA fine-tuning script
├── models/
│   ├── ocr/                  ← OCR model weights saved here (.pth)
│   └── lora_weights/         ← LoRA adapter weights saved here
├── data/
│   └── sample_prescriptions/ ← Sample test images
└── utils/
    ├── __init__.py
    ├── image_utils.py
    └── text_utils.py
```

---

## DETAILED IMPLEMENTATION INSTRUCTIONS

### FILE: `requirements.txt`

```
torch>=2.1.0
torchvision>=0.16.0
diffusers>=0.25.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
gradio>=4.15.0
opencv-python>=4.8.0
albumentations>=1.3.1
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
shap>=0.44.0
pytorch-grad-cam>=1.4.8
bitsandbytes>=0.43.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.23.0
scikit-learn>=1.3.0
tqdm>=4.66.0
scipy>=1.11.0
pandas>=2.0.0
```

---

### FILE: `config.py`

```python
import os
import torch

# ── Hardware ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
OCR_MODEL_PATH  = os.path.join(BASE_DIR, "models", "ocr", "crnn_best.pth")
LORA_WEIGHTS    = os.path.join(BASE_DIR, "models", "lora_weights", "final_weights")
SD_BASE_MODEL   = "runwayml/stable-diffusion-v1-5"

# ── OCR ───────────────────────────────────────────────────────────────────
OCR_IMG_H       = 32
OCR_IMG_W       = 128
OCR_HIDDEN_SIZE = 64
OCR_CHARSET     = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,/-"
OCR_BLANK_IDX   = 0

# ── Training (OCR) ────────────────────────────────────────────────────────
OCR_EPOCHS      = 50
OCR_BATCH_SIZE  = 32
OCR_LR          = 1e-3
OCR_AUGMENT_FACTOR = 5   # 5x augmentation

# ── Training (LoRA) ───────────────────────────────────────────────────────
LORA_RANK       = 8
LORA_ALPHA      = 8
LORA_TRAIN_STEPS = 5000
LORA_LR         = 2e-4
LORA_BATCH_SIZE = 4
LORA_IMG_SIZE   = 512
IMAGES_PER_DATASET = 50

# ── Generation ────────────────────────────────────────────────────────────
SD_INFERENCE_STEPS = 40
SD_GUIDANCE_SCALE  = 7.5
SD_SEED            = 42

# ── Local LLM (LLaVA) ────────────────────────────────────────────────────
# Downloads ~14 GB once from HuggingFace, then runs 100% offline
LLAVA_MODEL_ID       = "llava-hf/llava-1.5-7b-hf"
LLAVA_LOAD_IN_4BIT   = True        # 4-bit NF4 — ~5 GB VRAM
LLAVA_MAX_NEW_TOKENS = 512

# ── Hypothesis Scoring ────────────────────────────────────────────────────
TOP_K_HYPOTHESES = 3
```

---

### FILE: `pipeline/stage1_ocr/crnn_model.py`

Build a CRNN model with this exact architecture:

```python
"""
CRNN for Medical Prescription OCR
Architecture: ResNet18 CNN encoder → FC projection → BiLSTM → CTC decode
"""
import torch
import torch.nn as nn
import torchvision.models as models
from config import OCR_HIDDEN_SIZE, OCR_CHARSET, OCR_BLANK_IDX

NUM_CLASSES = len(OCR_CHARSET) + 1  # +1 for CTC blank


class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=OCR_HIDDEN_SIZE):
        super().__init__()

        # ── CNN Encoder: ResNet18 backbone (pretrained) ──────────────────
        resnet = models.resnet18(pretrained=True)
        # Remove avgpool and fc, keep only feature extractor layers
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # ── Projection: CNN features → sequence ──────────────────────────
        # Input image: (B, 3, 32, 128) → after CNN: (B, 512, 1, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse height
        self.projection = nn.Sequential(
            nn.Linear(512, hidden_size * 2),
            nn.Dropout(0.3),
        )

        # ── BiLSTM Sequence Decoder ───────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.3,
        )

        # ── Output: predict character at each timestep ───────────────────
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.cnn(x)           # (B, 512, H', W')
        features = self.adaptive_pool(features)  # (B, 512, 1, W')
        features = features.squeeze(2)    # (B, 512, W')
        features = features.permute(2, 0, 1)     # (W', B, 512) — time-first
        features = self.projection(features)      # (W', B, hidden*2)
        lstm_out, _ = self.lstm(features)         # (W', B, hidden*2)
        logits = self.fc(lstm_out)                # (W', B, num_classes)
        return logits
```

---

### FILE: `pipeline/stage1_ocr/train_ocr.py`

```python
"""
OCR Training Script
Dataset: Doctor's Handwritten Prescription BD (Kaggle)
Loss: CTC Loss
"""
import os
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
from config import *
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
        self.char2idx = {c: i+1 for i, c in enumerate(OCR_CHARSET)}  # 0=blank

        base = [A.Resize(OCR_IMG_H, OCR_IMG_W)]
        aug  = [
            A.Rotate(limit=3, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=3, p=0.5),
        ]
        norm = [
            A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
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
    # logits: (T, B, C)
    pred_idx = logits.argmax(dim=2)   # (T, B)
    results = []
    for b in range(pred_idx.shape[1]):
        seq = pred_idx[:, b].tolist()
        # collapse repeats and remove blanks
        decoded = []
        prev = None
        for s in seq:
            if s != prev and s != OCR_BLANK_IDX:
                if 1 <= s <= len(charset):
                    decoded.append(charset[s-1])
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

    model  = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    ctc    = nn.CTCLoss(blank=OCR_BLANK_IDX, reduction='mean', zero_infinity=True)
    optim  = torch.optim.Adam(model.parameters(), lr=OCR_LR, weight_decay=1e-4)
    sched  = CosineAnnealingLR(optim, T_max=OCR_EPOCHS)

    best_loss = float('inf')
    for epoch in range(1, OCR_EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, labels, label_lengths in tqdm(train_dl,
                                                 desc=f"Epoch {epoch}/{OCR_EPOCHS}"):
            imgs   = imgs.to(DEVICE)
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
            print(f"  ✅ Saved best model (loss={best_loss:.4f})")

    return model


if __name__ == "__main__":
    # Update these paths to your dataset location
    TRAIN_CSV     = "data/prescription_train.csv"
    TRAIN_IMG_DIR = "data/prescription_images/train"
    train_ocr(TRAIN_CSV, TRAIN_IMG_DIR)
```

---

### FILE: `pipeline/stage1_ocr/medicine_db.py`

```python
"""
Medicine-to-Disease Mapping Database
Covers 30+ common medications with associated diseases, severity,
and symptom profiles. Used for cross-referencing OCR output.
"""

MEDICINE_DB = {
    "amoxicillin":    {"diseases": ["pneumonia", "bronchitis", "sinusitis"],
                       "severity": "moderate", "symptoms": ["fever", "cough", "congestion"]},
    "azithromycin":   {"diseases": ["pneumonia", "bronchitis", "otitis"],
                       "severity": "moderate", "symptoms": ["fever", "sore throat", "cough"]},
    "metformin":      {"diseases": ["diabetes type 2"],
                       "severity": "chronic", "symptoms": ["polyuria", "polydipsia", "fatigue"]},
    "insulin":        {"diseases": ["diabetes type 1", "diabetes type 2"],
                       "severity": "severe", "symptoms": ["hyperglycemia", "ketoacidosis"]},
    "lisinopril":     {"diseases": ["hypertension", "heart failure"],
                       "severity": "chronic", "symptoms": ["high blood pressure", "edema"]},
    "atorvastatin":   {"diseases": ["hypercholesterolemia", "cardiovascular disease"],
                       "severity": "chronic", "symptoms": ["chest pain", "fatigue"]},
    "omeprazole":     {"diseases": ["GERD", "peptic ulcer"],
                       "severity": "mild", "symptoms": ["heartburn", "acid reflux", "epigastric pain"]},
    "salbutamol":     {"diseases": ["asthma", "COPD"],
                       "severity": "moderate", "symptoms": ["wheezing", "dyspnea", "chest tightness"]},
    "prednisolone":   {"diseases": ["asthma", "rheumatoid arthritis", "lupus"],
                       "severity": "moderate", "symptoms": ["inflammation", "joint pain", "rash"]},
    "ciprofloxacin":  {"diseases": ["UTI", "pneumonia", "gastroenteritis"],
                       "severity": "moderate", "symptoms": ["dysuria", "fever", "diarrhea"]},
    "paracetamol":    {"diseases": ["fever", "pain"],
                       "severity": "mild", "symptoms": ["pyrexia", "headache", "myalgia"]},
    "ibuprofen":      {"diseases": ["pain", "inflammation", "fever"],
                       "severity": "mild", "symptoms": ["joint pain", "headache", "fever"]},
    "warfarin":       {"diseases": ["deep vein thrombosis", "atrial fibrillation", "pulmonary embolism"],
                       "severity": "severe", "symptoms": ["chest pain", "leg swelling", "palpitations"]},
    "levothyroxine":  {"diseases": ["hypothyroidism"],
                       "severity": "chronic", "symptoms": ["fatigue", "weight gain", "cold intolerance"]},
    "metoprolol":     {"diseases": ["hypertension", "angina", "heart failure"],
                       "severity": "moderate", "symptoms": ["chest pain", "palpitations", "dyspnea"]},
    "furosemide":     {"diseases": ["heart failure", "edema", "hypertension"],
                       "severity": "moderate", "symptoms": ["edema", "dyspnea", "ascites"]},
    "clopidogrel":    {"diseases": ["myocardial infarction", "stroke", "peripheral artery disease"],
                       "severity": "severe", "symptoms": ["chest pain", "TIA", "claudication"]},
    "amlodipine":     {"diseases": ["hypertension", "angina"],
                       "severity": "moderate", "symptoms": ["chest pain", "high blood pressure"]},
    "sertraline":     {"diseases": ["depression", "anxiety", "OCD"],
                       "severity": "moderate", "symptoms": ["low mood", "anxiety", "insomnia"]},
    "gabapentin":     {"diseases": ["epilepsy", "neuropathic pain"],
                       "severity": "moderate", "symptoms": ["seizures", "neuropathy", "pain"]},
    "tramadol":       {"diseases": ["moderate pain", "neuropathic pain"],
                       "severity": "moderate", "symptoms": ["chronic pain", "post-surgical pain"]},
    "doxycycline":    {"diseases": ["malaria", "Lyme disease", "acne"],
                       "severity": "mild", "symptoms": ["rash", "fever", "joint pain"]},
    "fluconazole":    {"diseases": ["candidiasis", "fungal infection"],
                       "severity": "mild", "symptoms": ["discharge", "itching", "oral thrush"]},
    "pantoprazole":   {"diseases": ["GERD", "peptic ulcer", "H. pylori"],
                       "severity": "mild", "symptoms": ["heartburn", "epigastric pain"]},
    "losartan":       {"diseases": ["hypertension", "diabetic nephropathy"],
                       "severity": "chronic", "symptoms": ["high blood pressure", "proteinuria"]},
    "methotrexate":   {"diseases": ["rheumatoid arthritis", "psoriasis", "cancer"],
                       "severity": "severe", "symptoms": ["joint pain", "skin lesions"]},
    "tamoxifen":      {"diseases": ["breast cancer"],
                       "severity": "severe", "symptoms": ["breast lump", "nipple discharge"]},
    "levodopa":       {"diseases": ["Parkinson's disease"],
                       "severity": "severe", "symptoms": ["tremor", "rigidity", "bradykinesia"]},
    "donepezil":      {"diseases": ["Alzheimer's disease", "dementia"],
                       "severity": "severe", "symptoms": ["memory loss", "confusion"]},
    "albuterol":      {"diseases": ["asthma", "COPD", "bronchospasm"],
                       "severity": "moderate", "symptoms": ["wheezing", "dyspnea", "chest tightness"]},
}

# Fuzzy match abbreviations
MED_ABBREVIATIONS = {
    "amox": "amoxicillin", "azithro": "azithromycin",
    "metf": "metformin",   "ins": "insulin",
    "lisin": "lisinopril", "atorv": "atorvastatin",
    "omep": "omeprazole",  "salb": "salbutamol",
    "pred": "prednisolone","cipro": "ciprofloxacin",
    "pcm":  "paracetamol", "ibu":  "ibuprofen",
    "warf": "warfarin",    "levo": "levothyroxine",
    "metop": "metoprolol", "furo": "furosemide",
    "clop": "clopidogrel", "amlo": "amlodipine",
    "sert": "sertraline",  "gaba": "gabapentin",
}
```

---

### FILE: `pipeline/stage1_ocr/ocr_inference.py`

```python
"""
OCR Inference Module
- Loads trained CRNN
- Runs inference on prescription image
- Fuzzy-matches output to known medicine names
"""
import torch
import cv2
import numpy as np
from fuzzywuzzy import process
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *
from pipeline.stage1_ocr.crnn_model import CRNN, NUM_CLASSES
from pipeline.stage1_ocr.medicine_db import MEDICINE_DB, MED_ABBREVIATIONS
from pipeline.stage1_ocr.train_ocr import ctc_greedy_decode


class OCRInference:
    def __init__(self, model_path=OCR_MODEL_PATH):
        self.model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=DEVICE)
            state = ckpt.get('model_state', ckpt)
            self.model.load_state_dict(state)
            print(f"✅ OCR model loaded from {model_path}")
        else:
            print("⚠️  OCR model not found. Run training first.")
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(OCR_IMG_H, OCR_IMG_W),
            A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        self.medicine_names = list(MEDICINE_DB.keys())

    def preprocess(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path_or_array
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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
            best_match, score = process.extractOne(word, self.medicine_names)
            if score >= threshold:
                corrected.append(best_match)
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
```

---

### FILE: `pipeline/stage2_ner/ner_extractor.py`

```python
"""
Named Entity Recognition using ClinicalBERT
Extracts: symptoms, diseases, medications, diagnostic_tests, anatomical_refs
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from config import DEVICE

CLINICALBERT_MODEL = "samrawal/bert-base-uncased_clinical-ner"
# Alternative: "d4data/biomedical-ner-all"

ENTITY_CATEGORIES = {
    "symptoms":        ["SYMPTOM", "SIGN", "PROBLEM"],
    "diseases":        ["DISEASE", "DISORDER", "CONDITION"],
    "medications":     ["MEDICATION", "DRUG", "CHEMICAL"],
    "diagnostic_tests":["TEST", "PROCEDURE", "LAB"],
    "anatomical_refs": ["ANATOMY", "BODY_PART", "ORGAN"],
}

class NERExtractor:
    def __init__(self):
        print("Loading ClinicalBERT NER model...")
        self.nlp = pipeline(
            "ner",
            model=CLINICALBERT_MODEL,
            tokenizer=CLINICALBERT_MODEL,
            aggregation_strategy="simple",
            device=0 if DEVICE == "cuda" else -1,
        )
        print("✅ NER model loaded")

    def extract(self, text: str) -> dict:
        """Extract medical entities from clinical text."""
        entities_raw = self.nlp(text)
        result = {cat: [] for cat in ENTITY_CATEGORIES}
        result["raw_entities"] = entities_raw

        for ent in entities_raw:
            label = ent["entity_group"].upper()
            word  = ent["word"].strip()
            score = round(ent["score"], 3)

            for category, keywords in ENTITY_CATEGORIES.items():
                if any(kw in label for kw in keywords):
                    result[category].append({"text": word, "score": score})
                    break
            else:
                # Fallback heuristic classification by keyword matching
                word_lower = word.lower()
                if any(s in word_lower for s in ["pain", "fever", "cough", "ache",
                                                   "nausea", "fatigue", "swelling"]):
                    result["symptoms"].append({"text": word, "score": score})
                elif any(d in word_lower for d in ["itis", "emia", "osis", "oma",
                                                     "diabetes", "cancer", "tumor"]):
                    result["diseases"].append({"text": word, "score": score})

        return result
```

---

### FILE: `pipeline/stage3_scoring/symptom_disease_db.py`

```python
"""
Curated Symptom-Disease Association Database
Association strengths: 0.0 (unrelated) to 1.0 (pathognomonic)
"""

SYMPTOM_DISEASE_ASSOCIATIONS = {
    "pneumonia": {
        "fever":              0.85,
        "cough":              0.90,
        "productive cough":   0.85,
        "dyspnea":            0.80,
        "crackles":           0.90,
        "shortness of breath":0.80,
        "chest pain":         0.60,
        "low oxygen saturation": 0.75,
        "sputum":             0.70,
        "tachypnea":          0.75,
        "consolidation":      0.85,
    },
    "glioma": {
        "headache":           0.70,
        "seizure":            0.80,
        "nausea":             0.60,
        "vomiting":           0.60,
        "blurred vision":     0.65,
        "cognitive changes":  0.70,
        "focal neurological": 0.85,
        "papilledema":        0.75,
        "hemiparesis":        0.75,
    },
    "melanoma": {
        "irregular mole":     0.90,
        "asymmetric lesion":  0.90,
        "color variation":    0.85,
        "border irregularity":0.85,
        "diameter >6mm":      0.80,
        "evolving lesion":    0.80,
        "skin lesion":        0.75,
        "ulceration":         0.70,
    },
    "diabetic retinopathy": {
        "floaters":           0.75,
        "blurred vision":     0.70,
        "dark spots":         0.75,
        "visual acuity loss": 0.80,
        "diabetes":           0.90,
        "high HbA1c":         0.85,
        "microaneurysm":      0.90,
        "hemorrhage":         0.85,
    },
    "tuberculosis": {
        "productive cough":   0.85,
        "hemoptysis":         0.90,
        "night sweats":       0.85,
        "weight loss":        0.80,
        "fever":              0.75,
        "fatigue":            0.70,
        "lymphadenopathy":    0.75,
    },
    "myocardial infarction": {
        "chest pain":         0.90,
        "crushing pain":      0.95,
        "radiation to arm":   0.85,
        "dyspnea":            0.75,
        "diaphoresis":        0.80,
        "nausea":             0.65,
        "ECG changes":        0.90,
        "elevated troponin":  0.95,
    },
    "appendicitis": {
        "right lower quadrant pain": 0.90,
        "rebound tenderness":        0.85,
        "nausea":                    0.70,
        "vomiting":                  0.70,
        "fever":                     0.75,
        "anorexia":                  0.70,
        "McBurney's point":          0.90,
    },
    "asthma": {
        "wheezing":           0.90,
        "dyspnea":            0.80,
        "chest tightness":    0.85,
        "cough":              0.75,
        "nocturnal symptoms": 0.80,
        "allergic history":   0.70,
        "reversible airflow": 0.90,
    },
    "urinary tract infection": {
        "dysuria":            0.90,
        "frequency":          0.85,
        "urgency":            0.80,
        "suprapubic pain":    0.75,
        "hematuria":          0.70,
        "fever":              0.60,
        "cloudy urine":       0.75,
    },
}

# Clinical importance weights per entity type
ENTITY_IMPORTANCE_WEIGHTS = {
    "symptoms":         1.0,
    "diseases":         0.9,
    "diagnostic_tests": 0.8,
    "medications":      0.6,
    "anatomical_refs":  0.4,
}

# Modality routing per top disease
DISEASE_TO_MODALITY = {
    "pneumonia":               "chest_xray",
    "tuberculosis":            "chest_xray",
    "asthma":                  "chest_xray",
    "glioma":                  "brain_mri",
    "brain tumor":             "brain_mri",
    "meningioma":              "brain_mri",
    "melanoma":                "dermoscopy",
    "skin cancer":             "dermoscopy",
    "diabetic retinopathy":    "fundus",
    "myocardial infarction":   "chest_xray",
    "lung cancer":             "chest_ct",
    "breast cancer":           "breast_mri",
    "kidney stone":            "kidney_ct",
    "appendicitis":            "abdominal_ct",
    "default":                 "chest_xray",
}
```

---

### FILE: `pipeline/stage3_scoring/hypothesis_scorer.py`

```python
"""
Deterministic Hypothesis Scoring
Formula: S(d) = Σ w_imp(e_i) * w_rec(e_i) * A(e_i, d)
Fully transparent — every score is traceable to its inputs.
"""
import numpy as np
import shap
from config import TOP_K_HYPOTHESES
from pipeline.stage3_scoring.symptom_disease_db import (
    SYMPTOM_DISEASE_ASSOCIATIONS,
    ENTITY_IMPORTANCE_WEIGHTS,
    DISEASE_TO_MODALITY,
)


class HypothesisScorer:
    def __init__(self):
        self.diseases = list(SYMPTOM_DISEASE_ASSOCIATIONS.keys())

    def score(self, extracted_entities: dict) -> dict:
        """
        Score all diseases given extracted entities.
        Returns ranked hypotheses with SHAP-style attribution.
        """
        # Flatten all entities with their type
        flat_entities = []
        for etype, items in extracted_entities.items():
            if etype in ("raw_entities", "medicines_found"):
                continue
            weight = ENTITY_IMPORTANCE_WEIGHTS.get(etype, 0.5)
            for item in items:
                flat_entities.append({
                    "text":       item["text"].lower(),
                    "type":       etype,
                    "confidence": item["score"],
                    "importance": weight,
                })

        # Score each disease
        scores = {}
        attributions = {}  # entity → contribution per disease
        for disease, assoc in SYMPTOM_DISEASE_ASSOCIATIONS.items():
            total = 0.0
            contrib = {}
            for entity in flat_entities:
                # Find best matching symptom key
                best_key, best_assoc = None, 0.0
                for sym_key, sym_assoc in assoc.items():
                    if sym_key in entity["text"] or entity["text"] in sym_key:
                        if sym_assoc > best_assoc:
                            best_assoc = sym_assoc
                            best_key   = sym_key
                if best_key:
                    contribution = (entity["importance"]
                                    * entity["confidence"]
                                    * best_assoc)
                    total += contribution
                    contrib[entity["text"]] = round(contribution, 4)
            scores[disease]       = total
            attributions[disease] = contrib

        # Normalise to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {d: round(s / max_score, 4) for d, s in scores.items()}

        # Rank
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k  = ranked[:TOP_K_HYPOTHESES]

        top_disease   = top_k[0][0] if top_k else "unknown"
        modality      = DISEASE_TO_MODALITY.get(top_disease,
                         DISEASE_TO_MODALITY["default"])
        shap_values   = attributions.get(top_disease, {})

        return {
            "top_hypotheses":  top_k,
            "top_disease":     top_disease,
            "top_score":       top_k[0][1] if top_k else 0.0,
            "modality":        modality,
            "shap_values":     shap_values,
            "all_scores":      dict(ranked),
        }
```

---

### FILE: `pipeline/stage4_generation/prompt_builder.py`

```python
"""
Build clinical prompts from hypothesis and entities.
"""

MODALITY_PROMPTS = {
    "chest_xray": "chest x-ray showing {disease}, frontal radiography, "
                  "high quality medical imaging",
    "brain_mri":  "brain MRI scan showing {disease}, T1-weighted axial "
                  "view, clear medical imaging",
    "dermoscopy": "dermoscopic image of {disease}, asymmetrical skin lesion, "
                  "irregular borders, high resolution clinical photography",
    "fundus":     "retinal fundus photograph showing {disease}, hemorrhages, "
                  "grade 4 medical imaging, high resolution",
    "chest_ct":   "chest CT scan showing {disease}, axial view, "
                  "high quality medical imaging",
    "breast_mri": "breast MRI showing {disease}, contrast-enhanced, "
                  "medical imaging",
    "kidney_ct":  "CT scan showing {disease} in kidney, axial view, "
                  "high quality medical imaging",
    "abdominal_ct":"abdominal CT scan showing {disease}, axial view, "
                   "high quality medical imaging",
}

def build_prompt(hypothesis_result: dict) -> str:
    disease  = hypothesis_result["top_disease"]
    modality = hypothesis_result["modality"]
    template = MODALITY_PROMPTS.get(modality, MODALITY_PROMPTS["chest_xray"])
    return template.format(disease=disease)
```

---

### FILE: `pipeline/stage4_generation/lora_generator.py`

```python
"""
LoRA Stable Diffusion v1.5 Image Generator
Loads fine-tuned LoRA weights and generates medical images from text prompts.
"""
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from config import *


class LoRAGenerator:
    _instance = None  # Singleton to avoid re-loading model

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("Loading Stable Diffusion v1.5 with LoRA weights...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            SD_BASE_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(DEVICE)

        # Load LoRA adapter if available
        if os.path.exists(LORA_WEIGHTS):
            self.pipeline.load_lora_weights(LORA_WEIGHTS)
            print(f"✅ LoRA weights loaded from {LORA_WEIGHTS}")
        else:
            print("⚠️  LoRA weights not found — using base SD v1.5")

        # Memory optimisation
        if DEVICE == "cuda":
            self.pipeline.enable_attention_slicing()

        print("✅ Generator ready")

    def generate(self, prompt: str, seed: int = SD_SEED) -> Image.Image:
        """Generate a medical image from a clinical text prompt."""
        generator = torch.Generator(DEVICE).manual_seed(seed)
        result = self.pipeline(
            prompt,
            num_inference_steps=SD_INFERENCE_STEPS,
            guidance_scale=SD_GUIDANCE_SCALE,
            generator=generator,
            height=512,
            width=512,
        )
        return result.images[0]
```

---

### FILE: `pipeline/stage5_explainability/gradcam.py`

```python
"""
Grad-CAM on generated medical images.
Uses a pretrained DenseNet121 (CheXNet-style) as the classification backbone
to produce spatially relevant activation maps.
The backbone is applied to the generated image to highlight regions
relevant to the predicted disease class.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import DEVICE


class GradCAMExplainer:
    def __init__(self):
        # Use DenseNet121 pretrained on ImageNet as proxy backbone
        # (In production you'd use a CheXNet-pretrained checkpoint)
        backbone = models.densenet121(pretrained=True)
        self.model = backbone.to(DEVICE).eval()

        # Target the last conv layer in DenseNet
        target_layers = [self.model.features.denseblock4.denselayer16.conv2]
        self.cam = GradCAM(model=self.model, target_layers=target_layers)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def generate_heatmap(self, pil_image: Image.Image,
                          target_class: int = None) -> Image.Image:
        """
        Generate Grad-CAM heatmap overlay on the generated medical image.
        Returns a PIL Image with the heatmap blended onto the original.
        """
        # Prepare input tensor
        img_rgb = np.array(pil_image.convert("RGB").resize((224, 224)))
        img_float = img_rgb.astype(np.float32) / 255.0
        tensor = self.preprocess(pil_image).unsqueeze(0).to(DEVICE)

        # Compute Grad-CAM
        grayscale_cam = self.cam(input_tensor=tensor,
                                  targets=None)  # None = highest scoring class
        grayscale_cam = grayscale_cam[0]  # (H, W)

        # Blend onto original image
        visualization = show_cam_on_image(img_float, grayscale_cam,
                                           use_rgb=True, colormap=cv2.COLORMAP_JET)
        return Image.fromarray(visualization)
```

---

### FILE: `pipeline/stage5_explainability/shap_explainer.py`

```python
"""
SHAP Waterfall Plot for Hypothesis Scoring Explainability
Shows which clinical entity contributed how much to the final score.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
from io import BytesIO
from PIL import Image


def generate_shap_plot(shap_values: dict, top_disease: str,
                        top_score: float) -> Image.Image:
    """
    Generate a SHAP-style waterfall plot showing entity contributions.
    Returns PIL Image of the plot.
    """
    if not shap_values:
        # Return blank image if no SHAP data
        img = Image.new("RGB", (600, 300), color=(255, 255, 255))
        return img

    items  = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    values = [i[1] for i in items]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.6)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)

    ax.set_xlabel("Contribution to Hypothesis Score", fontsize=11)
    ax.set_title(f"SHAP Feature Attribution → {top_disease.title()}\n"
                 f"Final Score: {top_score:.2f}", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"+{val:.3f}" if val >= 0 else f"{val:.3f}",
                va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
```

---

### FILE: `pipeline/stage6_local_llm/llava_explainer.py`

```python
"""
Stage 6: Natural Language Clinical Explanation via LLaVA-1.5-7B (local)
- Multimodal: receives generated image + clinical context
- Returns a structured clinical explanation for the clinician
"""
import google.generativeai as genai
from PIL import Image
from config import DEVICE, LLAVA_MODEL_ID, LLAVA_LOAD_IN_4BIT, LLAVA_MAX_NEW_TOKENS


class LLaVAExplainer:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"✅ Gemini {GEMINI_MODEL} ready")

    def explain(self, generated_image: Image.Image,
                clinical_context: dict) -> str:
        """
        Generate a structured clinical explanation.

        Args:
            generated_image: PIL Image of the generated medical scan
            clinical_context: dict with keys:
                - ocr_text: str
                - entities: dict
                - top_disease: str
                - top_score: float
                - shap_values: dict
                - prompt: str

        Returns:
            Structured explanation string for display to clinician
        """
        disease   = clinical_context.get("top_disease", "unknown")
        score     = clinical_context.get("top_score", 0.0)
        ocr_text  = clinical_context.get("ocr_text", "")
        entities  = clinical_context.get("entities", {})
        shap_vals = clinical_context.get("shap_values", {})

        # Format entity summary
        symptoms    = [e["text"] for e in entities.get("symptoms", [])]
        medications = [e["text"] for e in entities.get("medications", [])]

        prompt_text = f"""You are an expert clinical AI assistant helping a physician understand a decision support system's output.

**Clinical Input (OCR-extracted):** {ocr_text}

**Identified Symptoms:** {', '.join(symptoms) if symptoms else 'None detected'}
**Identified Medications:** {', '.join(medications) if medications else 'None detected'}
**Top Hypothesis:** {disease} (confidence: {score:.0%})
**Key Contributing Factors:** {', '.join([f'{k} (+{v:.3f})' for k, v in list(shap_vals.items())[:5]])}

**Generated Medical Image:** The attached image was synthetically generated to visually represent the top hypothesis.

Please provide a structured clinical explanation with these sections:
1. **Clinical Summary** (2-3 sentences): What the evidence suggests
2. **Image Interpretation** (2-3 sentences): What to look for in the generated image and what the highlighted regions indicate
3. **Key Evidence** (bullet list): The most clinically significant findings
4. **Differential Considerations** (1-2 sentences): What else to rule out
5. **Recommended Next Steps** (bullet list): Suggested diagnostic actions

IMPORTANT: Always end with — "⚠️ This is a decision-support tool only. All findings must be verified by a qualified clinician."
"""

        try:
            response = self.model.generate_content([prompt_text, generated_image])
            return response.text
        except Exception as e:
            return (f"⚠️ Gemini explanation unavailable: {e}\n\n"
                    f"**System Summary**: Top hypothesis is **{disease}** "
                    f"with confidence score {score:.2f} based on extracted "
                    f"clinical entities. Please review the generated image "
                    f"and SHAP attribution plot for detailed evidence tracing.")
```

---

### FILE: `pipeline/full_pipeline.py`

```python
"""
MedVis-X Full 6-Stage Pipeline
Orchestrates: OCR → NER → Scoring → Generation → Grad-CAM → Gemini
"""
from pipeline.stage1_ocr.ocr_inference      import OCRInference
from pipeline.stage2_ner.ner_extractor      import NERExtractor
from pipeline.stage3_scoring.hypothesis_scorer import HypothesisScorer
from pipeline.stage4_generation.lora_generator import LoRAGenerator
from pipeline.stage4_generation.prompt_builder  import build_prompt
from pipeline.stage5_explainability.gradcam      import GradCAMExplainer
from pipeline.stage5_explainability.shap_explainer import generate_shap_plot
from pipeline.stage6_local_llm.llava_explainer  import LLaVAExplainer


class MedVisXPipeline:
    def __init__(self):
        print("\n🔬 Initialising MedVis-X Pipeline...")
        self.ocr       = OCRInference()
        self.ner       = NERExtractor()
        self.scorer    = HypothesisScorer()
        self.generator = LoRAGenerator.get_instance()
        self.gradcam   = GradCAMExplainer()
        self.gemini    = LLaVAExplainer()
        print("✅ All pipeline components ready\n")

    def run(self, image_input, text_input: str = None):
        """
        Run the full 6-stage pipeline.

        Args:
            image_input: file path or numpy array of prescription image
                         (pass None if using text_input only)
            text_input:  optional direct text override (skips OCR)

        Returns:
            dict with all stage outputs
        """
        results = {}

        # ── Stage 1: OCR ──────────────────────────────────────────────────
        if text_input:
            ocr_result = {"raw_text": text_input,
                          "corrected_text": text_input,
                          "medicines_found": {}}
        else:
            print("Stage 1: OCR...")
            ocr_result = self.ocr.extract_text(image_input)
        results["ocr"] = ocr_result
        print(f"  OCR: {ocr_result['corrected_text'][:80]}...")

        # ── Stage 2: NER ──────────────────────────────────────────────────
        print("Stage 2: NER...")
        combined_text = ocr_result["corrected_text"]
        # Append medicine-derived context
        for med, info in ocr_result["medicines_found"].items():
            combined_text += f". Prescribed {med} for {', '.join(info['diseases'])}."
        entities = self.ner.extract(combined_text)
        results["entities"] = entities
        print(f"  Entities: {sum(len(v) for k,v in entities.items() if k!='raw_entities')} found")

        # ── Stage 3: Hypothesis Scoring ───────────────────────────────────
        print("Stage 3: Scoring...")
        scoring = self.scorer.score(entities)
        results["scoring"] = scoring
        print(f"  Top hypothesis: {scoring['top_disease']} ({scoring['top_score']:.2f})")

        # ── Stage 4: Image Generation ─────────────────────────────────────
        print("Stage 4: Generating image...")
        prompt = build_prompt(scoring)
        results["prompt"] = prompt
        generated_image = self.generator.generate(prompt)
        results["generated_image"] = generated_image
        print(f"  Generated: {prompt[:60]}...")

        # ── Stage 5: Explainability ───────────────────────────────────────
        print("Stage 5: Explainability...")
        heatmap_image = self.gradcam.generate_heatmap(generated_image)
        shap_plot     = generate_shap_plot(
            scoring["shap_values"],
            scoring["top_disease"],
            scoring["top_score"],
        )
        results["heatmap"]   = heatmap_image
        results["shap_plot"] = shap_plot

        # ── Stage 6: Gemini Explanation ───────────────────────────────────
        print("Stage 6: LLaVA-7B local explanation...")
        clinical_context = {
            "ocr_text":    ocr_result["corrected_text"],
            "entities":    entities,
            "top_disease": scoring["top_disease"],
            "top_score":   scoring["top_score"],
            "shap_values": scoring["shap_values"],
            "prompt":      prompt,
        }
        explanation = self.llava.explain(generated_image, clinical_context)
        results["explanation"] = explanation
        print("✅ Pipeline complete!")

        return results
```

---

### FILE: `app.py` — Main Gradio Application

Build a beautiful, professional Gradio web application with this EXACT design:

```python
"""
MedVis-X — Traceable Clinical Decision Support System
Main Gradio Web Application
"""
import gradio as gr
import numpy as np
from PIL import Image
from pipeline.full_pipeline import MedVisXPipeline

# Lazy-load pipeline (only once)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = MedVisXPipeline()
    return _pipeline


def run_pipeline(prescription_image, clinical_text, use_text_only):
    """Main inference function called by Gradio."""
    try:
        pipe = get_pipeline()

        if use_text_only or prescription_image is None:
            if not clinical_text.strip():
                return (None, None, None,
                        "⚠️ Please provide either an image or clinical text.")
            results = pipe.run(image_input=None, text_input=clinical_text)
        else:
            # Convert gradio image (numpy) to PIL for OCR
            if isinstance(prescription_image, np.ndarray):
                pil_img = Image.fromarray(prescription_image.astype(np.uint8))
            else:
                pil_img = prescription_image
            results = pipe.run(image_input=pil_img)

        # Format OCR + scoring summary
        scoring  = results["scoring"]
        ocr_text = results["ocr"]["corrected_text"]

        hypotheses_text = "**🔬 Top Hypotheses:**\n"
        for disease, score in scoring["top_hypotheses"]:
            bar = "█" * int(score * 20)
            hypotheses_text += f"\n• **{disease.title()}**: {score:.2f} {bar}"

        summary = f"""## 📋 Clinical Analysis Report

**OCR Extracted Text:**
> {ocr_text}

---

{hypotheses_text}

---

**🎯 Top Diagnosis:** `{scoring['top_disease'].title()}`
**📊 Confidence:** `{scoring['top_score']:.1%}`
**🖼️ Imaging Modality:** `{scoring['modality'].replace('_', ' ').title()}`
**📝 Generation Prompt:** *{results['prompt']}*
"""

        return (
            results["generated_image"],   # Generated scan
            results["heatmap"],           # Grad-CAM overlay
            results["shap_plot"],         # SHAP attribution plot
            summary + "\n---\n\n" + results["explanation"],
        )

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return None, None, None, f"❌ Pipeline Error:\n```\n{err}\n```"


# ── Gradio UI ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    max-width: 1400px !important;
}
.title-banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.title-banner h1 { font-size: 2.2rem; margin: 0; }
.title-banner p  { opacity: 0.8; margin: 0.5rem 0 0; }
.stage-label {
    font-weight: 600;
    color: #2c5364;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
"""

with gr.Blocks(css=CUSTOM_CSS, title="MedVis-X Clinical AI") as demo:

    gr.HTML("""
    <div class="title-banner">
        <h1>🔬 MedVis-X</h1>
        <p>Traceable & Explainable Multimodal Clinical Decision Support System</p>
        <p style="font-size:0.8rem; opacity:0.6;">
            OCR → NER → Hypothesis Scoring → Diffusion Generation → Grad-CAM → Gemini XAI
        </p>
    </div>
    """)

    with gr.Row():
        # ── LEFT: Inputs ──────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Clinical Input")

            prescription_img = gr.Image(
                label="Upload Prescription / Clinical Document",
                type="numpy",
                height=280,
            )

            with gr.Accordion("✍️ Or enter clinical text directly", open=False):
                clinical_text = gr.Textbox(
                    label="Clinical Notes / Symptoms",
                    placeholder=(
                        "e.g. Patient presents with high fever, productive cough, "
                        "and crackles in lower left lobe. SpO2 < 92%."
                    ),
                    lines=5,
                )
                use_text_only = gr.Checkbox(
                    label="Use text input only (skip OCR)",
                    value=False,
                )

            run_btn = gr.Button(
                "🚀 Run Full Pipeline",
                variant="primary",
                size="lg",
            )

            gr.Markdown("---")
            gr.Examples(
                examples=[
                    [None, "Patient presents with high fever, productive cough, "
                           "and crackles in lower left lobe. SpO2 < 92%.", True],
                    [None, "Severe chronic headaches, nausea, blurred vision in "
                           "right eye. Neurological exam indicates localized pressure.", True],
                    [None, "Irregularly shaped mole on upper back, asymmetric, "
                           "multiple color variations, diameter > 6mm.", True],
                    [None, "Type 2 diabetic patient. HbA1c 8.5%. Floaters and "
                           "dark spots in vision.", True],
                ],
                inputs=[prescription_img, clinical_text, use_text_only],
                label="📌 Example Patient Cases",
            )

        # ── RIGHT: Outputs ────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📤 Analysis Outputs")

            with gr.Tabs():
                with gr.Tab("🖼️ Generated Scan + Heatmap"):
                    with gr.Row():
                        generated_out = gr.Image(
                            label="Stage 4: Generated Medical Image",
                            height=350,
                        )
                        heatmap_out = gr.Image(
                            label="Stage 5: Grad-CAM Activation Map",
                            height=350,
                        )

                with gr.Tab("📊 SHAP Attribution"):
                    shap_out = gr.Image(
                        label="Stage 5: SHAP Feature Attribution",
                        height=420,
                    )

                with gr.Tab("📋 Report + LLaVA Explanation"):
                    report_out = gr.Markdown(
                        label="Full Clinical Analysis",
                        value="*Run the pipeline to see the clinical report here.*",
                    )

    # ── Bind ──────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_pipeline,
        inputs=[prescription_img, clinical_text, use_text_only],
        outputs=[generated_out, heatmap_out, shap_out, report_out],
    )

    gr.HTML("""
    <div style="text-align:center; padding:1rem; opacity:0.5; font-size:0.8rem;">
        ⚠️ MedVis-X is a research decision-support tool only.
        All outputs must be reviewed and verified by a qualified clinician.
        Generated images are synthetic and NOT real patient data.
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
```

---

## TRAINING SCRIPTS

### FILE: `training/train_ocr.py`

This is a standalone runner to retrain the OCR model:

```python
"""
Standalone OCR Training Runner
Usage: python training/train_ocr.py \
    --train_csv data/prescription_train.csv \
    --train_img_dir data/prescription_images/train \
    --val_csv data/prescription_val.csv \
    --val_img_dir data/prescription_images/val
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stage1_ocr.train_ocr import train_ocr

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv",     required=True)
parser.add_argument("--train_img_dir", required=True)
parser.add_argument("--val_csv",       default=None)
parser.add_argument("--val_img_dir",   default=None)
parser.add_argument("--save_path",     default="models/ocr/crnn_best.pth")
args = parser.parse_args()

print("🏋️ Starting OCR Training...")
train_ocr(args.train_csv, args.train_img_dir,
          args.val_csv,   args.val_img_dir,
          args.save_path)
print("✅ Training complete.")
```

### FILE: `training/train_lora.py`

```python
"""
LoRA Fine-Tuning for Stable Diffusion v1.5
Trains on 5 medical image datasets (50 images each = 250 total)
Usage: python training/train_lora.py
"""
import os, sys, glob, random, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from tqdm import tqdm
from config import *

# ── Dataset Paths — update to your Kaggle input paths ────────────────────
DATASET_PATHS = {
    "Brain_MRI":           "/kaggle/input/brain-tumor-mri-dataset",
    "Chest_CT":            "/kaggle/input/chest-ctscan-images",
    "Chest_XRay":          "/kaggle/input/chest-xray-pneumonia",
    "Diabetic_Retinopathy":"/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered",
    "Skin_Cancer":         "/kaggle/input/skin-cancer-mnist-ham10000",
}

PROMPT_TEMPLATES = {
    "Brain_MRI":           "brain MRI scan showing {class_name}, medical imaging",
    "Chest_CT":            "chest CT scan showing {class_name}, axial view",
    "Chest_XRay":          "chest x-ray showing {class_name}, frontal radiography",
    "Diabetic_Retinopathy":"retinal fundus photograph, diabetic retinopathy {class_name}",
    "Skin_Cancer":         "dermoscopic image of {class_name}, skin lesion",
}

class MedicalFewShotDataset(Dataset):
    def __init__(self, image_paths, prompts):
        self.image_paths = image_paths
        self.prompts     = prompts
        self.transform   = A.Compose([
            A.Resize(LORA_IMG_SIZE, LORA_IMG_SIZE, interpolation=cv2.INTER_LANCZOS4),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.Normalize([0.5]*3, [0.5]*3, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None: img = np.zeros((LORA_IMG_SIZE, LORA_IMG_SIZE, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return {"pixel_values": self.transform(image=img)["image"],
                "text":         self.prompts[idx]}


def collect_samples():
    all_paths, all_prompts = [], []
    for ds_name, ds_path in DATASET_PATHS.items():
        images = []
        for ext in ["*.jpg","*.jpeg","*.png"]:
            images += glob.glob(str(Path(ds_path)/"**"/ext), recursive=True)
        if not images:
            print(f"⚠️  No images found for {ds_name}")
            continue
        sampled = random.sample(images, min(IMAGES_PER_DATASET, len(images)))
        for p in sampled:
            cls  = Path(p).parent.name.replace("_"," ").lower()
            all_paths.append(p)
            all_prompts.append(PROMPT_TEMPLATES[ds_name].format(class_name=cls))
        print(f"✅ {ds_name}: {len(sampled)} images")
    return all_paths, all_prompts


def train_lora():
    random.seed(42); torch.manual_seed(42)
    accelerator = Accelerator(mixed_precision="fp16")

    paths, prompts = collect_samples()
    dataset  = MedicalFewShotDataset(paths, prompts)
    loader   = DataLoader(dataset, batch_size=LORA_BATCH_SIZE, shuffle=True, num_workers=2)

    # Load pipeline components
    pipe = StableDiffusionPipeline.from_pretrained(SD_BASE_MODEL, torch_dtype=torch.float32)
    unet = pipe.unet
    vae  = pipe.vae.to(accelerator.device)
    text_encoder = pipe.text_encoder.to(accelerator.device)
    tokenizer    = pipe.tokenizer
    noise_sched  = DDPMScheduler.from_pretrained(SD_BASE_MODEL, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Apply LoRA to UNet cross-attention
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q","to_k","to_v","to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LORA_LR, weight_decay=0.01)
    from diffusers.optimization import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, 50, LORA_TRAIN_STEPS)

    unet, optimizer, loader, scheduler = accelerator.prepare(
        unet, optimizer, loader, scheduler)

    global_step = 0
    while global_step < LORA_TRAIN_STEPS:
        unet.train()
        for batch in tqdm(loader, desc=f"Step {global_step}/{LORA_TRAIN_STEPS}"):
            with accelerator.accumulate(unet):
                # Encode images
                pixel_values = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(pixel_values.float()).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps,
                                          (latents.shape[0],), device=latents.device)
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                # Encode text
                tokens = tokenizer(batch["text"], padding="max_length",
                                   max_length=77, truncation=True,
                                   return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden = text_encoder(tokens)[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")
                ckpt_dir = f"models/lora_weights/checkpoint-{global_step}"
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir)
                print(f"  💾 Checkpoint saved: {ckpt_dir}")

            if global_step >= LORA_TRAIN_STEPS: break

    # Save final
    os.makedirs("models/lora_weights/final_weights", exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained("models/lora_weights/final_weights")
    print("✅ LoRA training complete. Final weights saved.")


if __name__ == "__main__":
    train_lora()
```

---

## EXECUTION ORDER

After building all files, run in this order:

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Train OCR model (use your Kaggle dataset path)
python training/train_ocr.py \
    --train_csv data/prescription_train.csv \
    --train_img_dir data/prescription_images

# 3. Train LoRA model (run on Kaggle T4/P100 GPU)
python training/train_lora.py

# 4. Set Gemini API key (get free key from https://aistudio.google.com)
export GEMINI_API_KEY="your_key_here"

# 5. Launch the web app
python app.py
# Open browser: http://localhost:7860
```

---

## IMPORTANT NOTES FOR IMPLEMENTATION

1. **LoRA weights** — If you already have a trained LoRA zip from Kaggle, extract it to `models/lora_weights/final_weights/` and skip step 3.

2. **OCR model** — Must be retrained. The CRNN trains in ~2 hours on GPU. Without GPU, reduce `OCR_EPOCHS` to 20 for a quick test.

3. **LLaVA** — Downloads ~14 GB on first run from HuggingFace (`llava-hf/llava-1.5-7b-hf`), then runs 100% offline. No API key ever needed.

4. **GPU memory** — If you get OOM errors, enable `pipeline.enable_model_cpu_offload()` in `lora_generator.py` instead of `.to(DEVICE)`.

5. **ClinicalBERT** — The first run downloads ~440MB from HuggingFace. Ensure internet access on first launch.

6. **The Grad-CAM backbone** — Uses DenseNet121 pretrained on ImageNet as a proxy. For production, replace with a CheXNet checkpoint from HuggingFace (`nikbearbrown/CheXNet`).

7. **Everything is modular** — Each stage can be tested independently before running the full pipeline.

---

*MedVis-X is a decision-support tool. All outputs must be reviewed by a qualified clinician.*
*Generated images are synthetic and are NOT real patient scans.*
