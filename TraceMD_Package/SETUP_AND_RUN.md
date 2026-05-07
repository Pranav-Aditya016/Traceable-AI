# TraceMD — Complete Setup & Execution Guide

**Project:** TraceMD — Fully Local Clinical Decision Support Research Prototype  
**Pipeline:** 6-Stage AI Pipeline (OCR → NER → Scoring → Image Gen → Grad-CAM → Explanation)  
**Models:** Gemma 4 · MedGemma · Stable Diffusion v1.5 · DenseNet121  
**Package Date:** 2026-05-06  

---

## Folder Structure (This Package)

```
TraceMD_Package/
├── SETUP_AND_RUN.md             ← You are here
├── SOFTWARE_LIST.md             ← All required software & dependencies
└── src/
    ├── backend/
    │   ├── main.py              ← FastAPI server (SSE streaming, 6-stage pipeline)
    │   ├── config.py            ← Model paths, VRAM utilities, device detection
    │   ├── requirements.txt     ← Python dependencies
    │   ├── download_models.py   ← Downloads HuggingFace models to models/ folder
    │   ├── verify_models.py     ← Pre-flight check before starting server
    │   ├── smoke_test_api.py    ← Quick API test
    │   ├── test_pipeline.py     ← Full pipeline test
    │   ├── scripts/
    │   │   └── download_models_git.ps1  ← Git LFS model downloader
    │   └── pipeline/
    │       ├── ollama_ocr.py    ← Stage 1: Gemma 4 OCR
    │       ├── ner.py           ← Stage 2: MedGemma NER
    │       ├── scorer.py        ← Stage 3: SHAP hypothesis scoring
    │       ├── generator.py     ← Stage 4: Stable Diffusion v1.5
    │       ├── gradcam.py       ← Stage 5: DenseNet121 Grad-CAM
    │       └── explainer.py     ← Stage 6: MedGemma clinical explanation
    └── frontend/
        ├── package.json
        ├── vite.config.ts
        ├── tailwind.config.ts
        ├── index.html
        └── src/
            ├── App.tsx
            ├── types.ts
            ├── api/analyzeStream.ts
            ├── components/
            └── three/
```

---

## How the Pipeline Works

```
Upload (JPG / PNG / PDF)
        │
        ▼
[Stage 1] Gemma 4 (gemma4:e4b via Ollama)
          → Reads entire document as image
          → Extracts all text: patient info, vitals, symptoms, meds
        │
        ▼
[Stage 2] MedGemma (medgemma:4b via Ollama)
          → Structures OCR text into typed JSON
          → Extracts: organ, conditions, confidence, reasoning
          → Fallback: deterministic heuristic parser
        │
        ▼
[Stage 3] Hypothesis Scorer (Pure Python + SHAP)
          → Scores conditions using symptom-association database
          → Computes SHAP attribution for explainability
        │
        ▼
[Stage 4] Stable Diffusion v1.5 (Stability AI — local folder)
          → Generates 512×512 grayscale DICOM-style synthetic image
          → Watermarked: "SYNTHETIC — NOT A REAL PATIENT IMAGE"
        │
        ▼
[Stage 5] DenseNet121 Grad-CAM (torchvision)
          → Applies to original uploaded image (not synthetic)
          → Produces saliency map + binary mask overlay
        │
        ▼
[Stage 6] MedGemma (medgemma:4b via Ollama)
          → Generates structured 4-section clinical narrative
          → Sections: Clinical Summary, Primary Diagnosis, Differentials, Disclaimer
        │
        ▼
Frontend receives all results via SSE (Server-Sent Events)
```

---

## STEP 1 — Install System Software

### 1.1 Python 3.11 or 3.12
```
Download: https://python.org/downloads
```
During install: **check "Add Python to PATH"**  
Verify: `python --version`

### 1.2 Node.js v22 LTS
```
Download: https://nodejs.org/en/download
```
Verify: `node --version` and `npm --version`

### 1.3 CUDA Toolkit 12.4 (NVIDIA GPU users only)
```
Download: https://developer.nvidia.com/cuda-downloads
```
Select: Windows → x86_64 → your Windows version → exe (local)  
Verify after install: `nvcc --version`

### 1.4 Ollama
```
Download: https://ollama.com/download
```
Run the installer. Ollama starts as a background Windows service automatically.  
Verify: `ollama --version`

### 1.5 Git + Git LFS
```
Git: https://git-scm.com/download/win
Git LFS: https://git-lfs.com
```
After installing both, run once:
```powershell
git lfs install
```

---

## STEP 2 — Create New Project Folder

Open **PowerShell** (or Command Prompt):

```powershell
# Create project folder anywhere you like
mkdir "C:\MyProjects\TraceMD"
cd "C:\MyProjects\TraceMD"
```

Copy the contents of `src/backend/` and `src/frontend/` from this package into the new folder:

```
C:\MyProjects\TraceMD\
├── backend\          ← copy from src/backend/
├── frontend\         ← copy from src/frontend/
└── models\           ← will be created automatically
```

Using PowerShell xcopy:
```powershell
xcopy /E /I "path\to\TraceMD_Package\src\backend\*" "C:\MyProjects\TraceMD\backend\"
xcopy /E /I "path\to\TraceMD_Package\src\frontend\*" "C:\MyProjects\TraceMD\frontend\"
```

---

## STEP 3 — Set Up Python Virtual Environment

```powershell
cd "C:\MyProjects\TraceMD\backend"

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate
# You should now see (.venv) in your prompt

# Upgrade pip first
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4 (GPU users)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt
```

> **CPU-only users:** Replace the PyTorch line with:
> ```powershell
> pip install torch torchvision
> ```

---

## STEP 4 — Pull Ollama Models

Make sure Ollama is running (it usually auto-starts on Windows). Then pull both models:

```powershell
# Gemma 4 vision model — Stage 1 OCR (~4 GB)
ollama pull gemma4:e4b

# MedGemma — Stage 2 NER + Stage 6 Explanation (~4 GB)
ollama pull medgemma:4b

# Verify both are listed
ollama list
```

Expected output:
```
NAME               SIZE    MODIFIED
gemma4:e4b         ...     ...
medgemma:4b        ...     ...
```

If `gemma4:e4b` is not found, try:
```powershell
ollama pull gemma4   # pulls the default gemma4 tag
```
Then update `OCR_MODEL` in [pipeline/ollama_ocr.py](src/backend/pipeline/ollama_ocr.py) line 17 to match.

---

## STEP 5 — Download HuggingFace Models (Stable Diffusion + Others)

### Option A — Automatic downloader (recommended)

```powershell
cd "C:\MyProjects\TraceMD\backend"
.venv\Scripts\activate

# For open (non-gated) models only — Stable Diffusion + TrOCR
python download_models.py
```

For gated models (MedGemma, PaliGemma) — these require HuggingFace account + accepted terms:

```powershell
# Step 1: Log in to HuggingFace
pip install huggingface_hub
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens

# Step 2: Download all including gated models
python download_models.py
```

### Option B — Git LFS downloader (PowerShell script)

```powershell
# Non-gated only (no token needed)
powershell -ExecutionPolicy Bypass -File "scripts\download_models_git.ps1" -Models trocr,sd15

# With token (for gated models)
powershell -ExecutionPolicy Bypass -File "scripts\download_models_git.ps1" -Models all -Token "your_hf_token_here"
```

Models are saved to `C:\MyProjects\TraceMD\models\`:
```
models\
├── stable-diffusion-v1-5\    ← Stability AI (~4 GB)
├── medgemma-4b-it\           ← Google MedGemma (~8 GB) — gated
├── paligemma2-3b-pt-896\     ← Google PaliGemma (~6 GB) — gated
└── trocr-large-handwritten\  ← Microsoft TrOCR (~1.3 GB)
```

---

## STEP 6 — Verify All Models

```powershell
cd "C:\MyProjects\TraceMD\backend"
.venv\Scripts\activate
python verify_models.py
```

Expected output:
```
=== TraceMD Model Verification ===

CUDA available: True
GPU: NVIDIA GeForce RTX 4060 ...
VRAM: 8.0 GB

  ✓ TrOCR Large Handwritten — XX files found
  ✓ PaliGemma 2 3B PT 896 — XX files found
  ✓ MedGemma 4B IT — XX files found
  ✓ Stable Diffusion v1.5 — XX files found
  ✓ DenseNet121 — torchvision weights available

✓ All models verified. Safe to start server.
```

---

## STEP 7 — Set Up Frontend

```powershell
cd "C:\MyProjects\TraceMD\frontend"

# Install Node.js dependencies
npm install
```

---

## STEP 8 — Run the Application

Open **3 separate terminal windows**:

### Terminal 1 — Ensure Ollama is running
```powershell
# Ollama usually starts automatically on Windows as a service
# If not running, start it manually:
ollama serve
```

### Terminal 2 — Start Backend
```powershell
cd "C:\MyProjects\TraceMD\backend"
.venv\Scripts\activate
python main.py
```

Expected output:
```
INFO:     Started server process [...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 3 — Start Frontend
```powershell
cd "C:\MyProjects\TraceMD\frontend"
npm run dev
```

Expected output:
```
  VITE v5.x  ready in ... ms
  ➜  Local:   http://localhost:5173/
```

---

## STEP 9 — Open the App

Open browser and go to:
```
http://localhost:5173
```

You should see the TraceMD dark-themed interface with the particle background and upload zone.

---

## STEP 10 — Test the Pipeline

### Quick API health check
```powershell
# In any terminal with .venv active
python -c "import requests; r = requests.get('http://localhost:8000/health'); print(r.json())"
```
Expected: `{"status": "ok", "cuda": true, "gpu": "NVIDIA GeForce RTX ..."}`

### Full smoke test
```powershell
cd "C:\MyProjects\TraceMD\backend"
.venv\Scripts\activate
python smoke_test_api.py
```

### Using the UI
1. Go to `http://localhost:5173`
2. Drag & drop a medical image (JPG/PNG) or a PDF prescription
3. Click **Analyze**
4. Watch the **6-stage pipeline tracker** update in real time via SSE
5. View results: extracted text, diagnosis, synthetic image, Grad-CAM, SHAP, clinical report

---

## Ports & URLs Summary

| Service | URL | Notes |
|---------|-----|-------|
| React Frontend | http://localhost:5173 | Open this in browser |
| FastAPI Backend | http://localhost:8000 | Streams via SSE |
| Backend Health | http://localhost:8000/health | JSON health check |
| Ollama | http://localhost:11434 | Runs Gemma 4 + MedGemma |

---

## Environment Variables (Optional)

Set these before starting the backend if you need custom endpoints:

```powershell
# Custom Ollama URL (default: http://localhost:11434)
$env:OLLAMA_BASE_URL = "http://localhost:11434"

# HuggingFace token for gated model downloads
$env:HF_TOKEN = "your_token_here"
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Ollama is not running` | Ollama service not started | Run `ollama serve` in a terminal |
| `404 model not found` for `gemma4:e4b` | Wrong tag | Run `ollama list` to see available tags; pull the correct one |
| `CUDA out of memory` | Two large models loaded together | The pipeline automatically clears VRAM between stages — do not run concurrent requests |
| `ModuleNotFoundError` | venv not activated | Run `.venv\Scripts\activate` before `python main.py` |
| SD model fails to load | Model folder missing or empty | Re-run `python download_models.py` |
| Frontend blank page | Backend not running | Start backend first on port 8000, then frontend |
| `pip install` errors | Outdated pip | Run `python -m pip install --upgrade pip` first |
| Gated model download fails | No HF token / terms not accepted | Accept model terms on huggingface.co, then `huggingface-cli login` |
| `pypdfium2` install fails | Compiler missing | Install Visual C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 6 GB VRAM | NVIDIA RTX 4060 8 GB |
| RAM | 16 GB | 32 GB |
| Storage | 35 GB free | 50 GB free |
| Internet | Required for first-time model download | Not needed after setup |

---

## VRAM Usage Per Stage

| Stage | Model | Peak VRAM |
|-------|-------|-----------|
| Stage 1 | Gemma 4 via Ollama | ~3 GB (Ollama manages this) |
| Stage 2 | MedGemma via Ollama | ~3 GB |
| Stage 3 | SHAP (CPU only) | 0 |
| Stage 4 | Stable Diffusion v1.5 | ~4 GB |
| Stage 5 | DenseNet121 | ~0.5 GB |
| Stage 6 | MedGemma via Ollama | ~3 GB |

Stages run **sequentially** with full VRAM cleanup between each. Total 8 GB is sufficient.

---

*Research prototype. Not for clinical diagnostic use.*
