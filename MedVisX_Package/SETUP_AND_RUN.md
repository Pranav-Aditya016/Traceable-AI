# MedVis-X — Complete Setup & Execution Guide

**Project:** MedVis-X: Traceable & Explainable Multimodal Clinical Decision Support System  
**Student:** Pranav Aditya | B.Tech Computer Science  
**Package Date:** 2026-05-06  

---

## Folder Structure (This Package)

```
MedVisX_Package/
├── SETUP_AND_RUN.md          ← You are here
├── SOFTWARE_LIST.md          ← All required software
└── src/
    ├── server.py             ← FastAPI backend entry point
    ├── config.py             ← All hyperparameters & paths
    ├── requirements.txt      ← Python dependencies
    ├── app.py                ← Legacy Gradio app (optional)
    ├── test_api.py           ← API integration tests
    ├── test_gen.py           ← Image generation tests
    ├── test_pipeline.py      ← Full pipeline tests
    ├── test_quick.py         ← Quick smoke test
    ├── test_xai.py           ← XAI (Grad-CAM + SHAP) tests
    ├── pipeline/
    │   ├── structured_analyzer.py   ← LLaVA + Mistral two-model analyzer
    │   ├── full_pipeline.py         ← Legacy 6-stage orchestrator
    │   ├── stage1_ocr/              ← CRNN OCR model
    │   ├── stage2_ner/              ← ClinicalBERT NER
    │   ├── stage3_scoring/          ← Hypothesis scorer + SHAP
    │   ├── stage4_generation/       ← SDXL image generator + prompt builder
    │   ├── stage5_explainability/   ← Grad-CAM + SHAP plots
    │   └── stage6_local_llm/        ← Ollama LLaVA explainer
    └── frontend/
        ├── package.json
        ├── vite.config.ts
        ├── tsconfig.json
        ├── index.html
        └── src/
            ├── App.tsx
            ├── main.tsx
            ├── types.ts
            ├── api.ts
            ├── index.css
            └── components/
                ├── Header.tsx
                ├── UploadPanel.tsx
                ├── ProgressBar.tsx
                ├── LoadingOverlay.tsx
                └── ResultsDashboard.tsx
```

---

## STEP 1 — Install Required Software

### 1.1 Python 3.11 or 3.13
- Download from: https://www.python.org/downloads/
- During install: **check "Add Python to PATH"**
- Verify: `python --version`

### 1.2 Node.js v22 (LTS)
- Download from: https://nodejs.org/en/download
- Verify: `node --version` and `npm --version`

### 1.3 CUDA Toolkit 12.4 (if using NVIDIA GPU)
- Download from: https://developer.nvidia.com/cuda-downloads
- Verify: `nvcc --version`

### 1.4 Ollama (Local LLM Server)
- Download from: https://ollama.com/download
- Install and run it once so it starts as a background service
- Verify: `ollama --version`

### 1.5 Git (Optional — for cloning models)
- Download from: https://git-scm.com/download/win

---

## STEP 2 — Create a New Project Folder

Open **PowerShell** or **Command Prompt** as Administrator:

```powershell
# Create your new project folder anywhere you like
mkdir "C:\MyProjects\MedVisX"
cd "C:\MyProjects\MedVisX"
```

Copy everything from the `src/` folder in this package into `C:\MyProjects\MedVisX\`:

```powershell
# Example: copy all src files
xcopy /E /I "path\to\MedVisX_Package\src\*" "C:\MyProjects\MedVisX\"
```

---

## STEP 3 — Set Up Python Backend

```powershell
# Navigate to your project folder
cd "C:\MyProjects\MedVisX"

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# You should see (.venv) in your prompt now

# Install PyTorch with CUDA 12.4 (GPU users)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt

# Also install FastAPI server dependencies (not in requirements.txt)
pip install fastapi uvicorn python-multipart sse-starlette
```

> **CPU-only users:** Replace the PyTorch install with:
> ```powershell
> pip install torch torchvision
> ```

---

## STEP 4 — Pull Ollama Models

Ollama must be running. Pull the two required models:

```powershell
# Pull LLaVA (vision + language, ~4.4 GB)
ollama pull llava

# Pull Mistral (JSON structuring, ~4.1 GB)
ollama pull mistral

# Verify both are available
ollama list
```

Expected output:
```
NAME            ID              SIZE    MODIFIED
llava:latest    ...             4.4 GB  ...
mistral:latest  ...             4.1 GB  ...
```

---

## STEP 5 — Download Image Generation Model (SDXL)

The SDXL model (~6.5 GB) downloads automatically on first run via Hugging Face.  
If you want to pre-download it:

```powershell
# With .venv activated
python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')"
```

This saves to `~/.cache/huggingface/hub/` and is reused on every subsequent run.

**Fallback:** If SDXL is not cached, the system automatically falls back to SD v1.5 (`runwayml/stable-diffusion-v1-5`, ~4 GB).

---

## STEP 6 — Set Up Frontend

```powershell
# Navigate to frontend folder
cd "C:\MyProjects\MedVisX\frontend"

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```

---

## STEP 7 — Run the Application

Open **3 separate terminal windows**:

### Terminal 1: Start Ollama (if not auto-started)
```powershell
ollama serve
```
Leave this running. (On Windows, Ollama usually auto-starts as a service.)

### Terminal 2: Start Python Backend
```powershell
cd "C:\MyProjects\MedVisX"
.venv\Scripts\activate
python server.py
```

Expected output:
```
INFO:     Started server process [...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 3: Start React Frontend
```powershell
cd "C:\MyProjects\MedVisX\frontend"
npm run dev
```

Expected output:
```
  VITE v7.x.x  ready in ... ms
  ➜  Local:   http://localhost:5173/
```

---

## STEP 8 — Open the Application

Open your browser and go to:

```
http://localhost:5173
```

You should see the MedVis-X dark medical-themed interface.

---

## STEP 9 — Verify Everything Works

### Quick Health Check
```powershell
# In any terminal (with .venv active)
python -c "import requests; print(requests.get('http://localhost:8000/api/health').json())"
```

Expected:
```json
{"status": "ok", "device": "cuda", "gpu": {"name": "NVIDIA GeForce RTX ...", ...}, "ollama": true}
```

### Quick Pipeline Test
```powershell
cd "C:\MyProjects\MedVisX"
.venv\Scripts\activate
python test_quick.py
```

---

## STEP 10 — Use the App

1. Open `http://localhost:5173` in your browser
2. **Upload** a medical image (PNG/JPEG) or type clinical notes in the text area
3. Fill in patient name, age, sex (optional)
4. Click **"Analyze & Generate"**
5. Wait 60–180 seconds for the full pipeline to complete
6. View results: diagnosis, synthetic image, Grad-CAM, SHAP, and clinical explanation

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Make sure `.venv\Scripts\activate` is active before running `python server.py` |
| `CUDA out of memory` | Close other GPU apps; ensure no other Python processes are using the GPU |
| Ollama not responding | Run `ollama serve` manually in a terminal and leave it open |
| Frontend can't reach backend | Make sure backend is running on port 8000; check `vite.config.ts` proxy |
| SDXL download hangs | Check internet connection; retry; or set `HF_HUB_OFFLINE=1` to force SD v1.5 fallback |
| `pip install` errors | Upgrade pip first: `python -m pip install --upgrade pip` |

---

## Ports Summary

| Service | URL | Notes |
|---------|-----|-------|
| React Frontend | http://localhost:5173 | Open this in browser |
| FastAPI Backend | http://localhost:8000 | API server |
| Ollama LLM | http://localhost:11434 | Auto-started service |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 6GB VRAM | NVIDIA RTX 4060 8GB+ |
| RAM | 16 GB | 32 GB |
| Storage | 30 GB free | 50 GB free |
| Internet | Required for first model download | Not needed after setup |

---

*Research prototype. Not for clinical diagnostic use.*
