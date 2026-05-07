# Traceable AI — MedVis-X & TraceMD

**Traceable & Explainable Multimodal Clinical Decision Support System**

> 100% Local Inference | No Cloud APIs | Explainable AI | Privacy by Design

B.Tech Major Project — Computer Science | Pranav Aditya

---

## What This Project Is

This repository contains two interrelated clinical AI systems built as a major project for a B.Tech degree:

| Sub-project | Description | Status |
|-------------|-------------|--------|
| **MedVis-X** (`medvisx/`) | Original 6-stage pipeline: OCR → NER → Scoring → Image Gen → Grad-CAM → LLaVA. Gradio UI, then migrated to React+FastAPI. | Sprints 1–8 complete |
| **TraceMD** (`TraceMD/`) | Evolved architecture with TrOCR + PaliGemma2 + MedGemma + Stable Diffusion + Grad-CAM, streaming SSE API, Three.js frontend. | Active development |

Both systems run **entirely on local hardware** — no data leaves the machine at inference time.

---

## The Problem

Current medical AI systems have three critical gaps:

1. **Black-box predictions** — models diagnose without explaining their reasoning, making them untrustworthy for clinical use.
2. **Cloud dependency** — sending patient data to GPT-4V or Gemini creates HIPAA/GDPR compliance risks.
3. **No traceability** — predictions cannot be audited back to specific input features.

This project addresses all three: local inference only, SHAP attribution for every prediction, Grad-CAM visual explanations, and a complete timestamped audit log.

---

## Pipeline Overview

### MedVis-X — 6-Stage Pipeline

```
[Prescription / Medical Document]
        ↓
  Stage 1: OCR
    CRNN (ResNet18 → BiLSTM → CTC) + EasyOCR fallback
        ↓
  Stage 2: NER
    ClinicalBERT — extracts symptoms, diseases, medications
        ↓
  Stage 3: Hypothesis Scoring
    Deterministic weighted formula: S(d) = Σ[w_imp × w_rec × A(entity, disease)]
    SHAP waterfall attribution per feature
        ↓
  Stage 4: Synthetic Image Generation
    Stable Diffusion XL (768×768, 15 steps, CPU offload)
    8 imaging modalities × 12+ disease finding banks
        ↓
  Stage 5: Grad-CAM Explainability
    DenseNet121 backbone — JET colormap saliency overlay
        ↓
  Stage 6: Natural Language Explanation
    LLaVA via Ollama — 5-section clinical report
        ↓
  [Report + Synthetic Image + Heatmap + SHAP Plot + LLaVA Explanation + Audit Log]
```

> **Architecture evolution (Sprint 6):** Stages 1–3 were replaced by a Two-Model Analyzer (LLaVA describes → Mistral structures to JSON) for better accuracy and reliability. The legacy OCR/NER/Scorer code is preserved in `medvisx/pipeline/`.

### TraceMD — Upgraded Pipeline (v2)

```
[Handwritten Prescription / Medical Report]
        ↓
  Stage 1: OCR Router
    TrOCR (handwritten) or PaliGemma2 (printed) — heuristic routing
        ↓
  Stage 2: Entity Extraction
    MedGemma 4B (4-bit quantised) — symptoms, medications, conditions
        ↓
  Stage 3: Hypothesis Scoring + SHAP
    Symbolic weighted scoring with SHAP attribution
        ↓
  Stage 4: Image Generation
    Stable Diffusion v1.5 (fp16, 512×512)
        ↓
  Stage 5: Grad-CAM
    DenseNet121 saliency map
        ↓
  Stage 6: Clinical Narrative
    MedGemma 4B (reused from Stage 2) — structured explanation
        ↓
  [SSE-streamed JSON response to React frontend]
```

---

## Repository Structure

```
Traceable AI/
│
├── medvisx/                        ← MedVis-X (primary research system)
│   ├── server.py                   ← FastAPI app — 3 endpoints
│   ├── config.py                   ← All hyperparameters
│   ├── app.py                      ← Gradio entry point (legacy UI)
│   ├── pipeline/
│   │   ├── structured_analyzer.py  ← Two-Model Analyzer: LLaVA + Mistral
│   │   ├── full_pipeline.py        ← Legacy 6-stage orchestrator
│   │   ├── stage1_ocr/             ← CRNN model, OCR training, inference
│   │   ├── stage2_ner/             ← ClinicalBERT NER + keyword fallback
│   │   ├── stage3_scoring/         ← Hypothesis scorer + disease DB
│   │   ├── stage4_generation/      ← SDXL generator + prompt builder
│   │   ├── stage5_explainability/  ← Grad-CAM + SHAP plots
│   │   └── stage6_local_llm/       ← Ollama LLaVA explainer
│   ├── training/
│   │   ├── train_lora.py           ← LoRA fine-tuning for SD v1.5
│   │   └── train_ocr.py            ← CRNN training runner
│   ├── frontend/                   ← React 19 + TypeScript + Vite UI
│   │   └── src/
│   │       ├── App.tsx
│   │       ├── components/         ← Header, UploadPanel, ProgressBar,
│   │       │                          LoadingOverlay, ResultsDashboard
│   │       ├── api.ts
│   │       └── types.ts            ← Full AnalysisResult TypeScript interface
│   └── test_*.py                   ← Integration test files
│
├── TraceMD/                        ← TraceMD (v2 architecture)
│   ├── backend/
│   │   ├── main.py                 ← FastAPI + SSE streaming endpoint
│   │   ├── config.py               ← Model paths + VRAM utilities
│   │   ├── download_models.py      ← HuggingFace model downloader
│   │   └── pipeline/
│   │       ├── ollama_ocr.py       ← TrOCR / PaliGemma2 OCR router
│   │       ├── ner.py              ← MedGemma 4B entity extraction
│   │       ├── scorer.py           ← Hypothesis scorer + SHAP
│   │       ├── generator.py        ← SD v1.5 image generation
│   │       ├── gradcam.py          ← DenseNet121 Grad-CAM
│   │       └── explainer.py        ← MedGemma 4B narrative explanation
│   ├── frontend/                   ← React 18 + TypeScript + Three.js
│   │   └── src/
│   │       ├── App.tsx
│   │       ├── components/         ← Hero, UploadZone, PipelineTracker,
│   │       │                          SummaryView (7 sub-components)
│   │       └── three/              ← Three.js particle field
│   └── scripts/
│       └── download_models_git.ps1 ← PowerShell model downloader
│
├── MedVisX_Package/                ← Standalone distributable (MedVis-X)
├── TraceMD_Package/                ← Standalone distributable (TraceMD)
├── TRACEMD-main/                   ← Early TypeScript prototype
│
├── MP01/                           ← Jupyter notebooks (model experiments)
│   ├── BrainCT_Model.ipynb
│   ├── BreastCancer_Model.ipynb
│   ├── ChestCT_LungCancer_Model.ipynb
│   ├── DRR_Bones_Model.ipynb
│   ├── BraTS_Preprocessing.ipynb
│   └── ...
│
├── .gitignore
└── README.md
```

> **Not in this repo (gitignored):** Model weights (`.pth`, `.bin`, `.safetensors`), downloaded HuggingFace models (`TraceMD/models/`, `medvisx/models/`), CT scan datasets, LoRA checkpoints, Python `.venv/`, Node `node_modules/`, and personal documents.

---

## Technology Stack

### MedVis-X Backend
| Component | Technology | Purpose |
|-----------|------------|---------|
| Runtime | Python 3.13.8 | Backend logic |
| API | FastAPI + Uvicorn | REST API on port 8000 |
| Deep Learning | PyTorch 2.6.0 + CUDA 12.4 | Model inference |
| Image Generation | Diffusers — SDXL | `stabilityai/stable-diffusion-xl-base-1.0` |
| Local LLM | Ollama | LLaVA 4.4 GB + Mistral 4.1 GB |
| Grad-CAM | pytorch-grad-cam | DenseNet121 saliency maps |
| SHAP | shap + matplotlib | Feature attribution waterfall plots |
| NER (legacy) | ClinicalBERT | Named entity extraction |
| OCR (legacy) | Custom CRNN | ResNet18 + BiLSTM + CTC |

### TraceMD Backend
| Component | Technology | Purpose |
|-----------|------------|---------|
| OCR | TrOCR (`microsoft/trocr-large-handwritten`) | Handwritten text |
| OCR alt | PaliGemma2 3B (`google/paligemma2-3b-pt-896`) | Printed/typed text |
| NER + Explain | MedGemma 4B (`google/medgemma-4b-it`) | Entity extraction + narrative |
| Generation | SD v1.5 (`stable-diffusion-v1-5`) | Synthetic medical images |
| API | FastAPI + `sse-starlette` | SSE streaming response |

### Frontend (both projects)
| Component | Technology |
|-----------|------------|
| Framework | React 18/19 + TypeScript |
| Build | Vite |
| Styling | Tailwind CSS |
| Charts | Recharts |
| Animations | Framer Motion |
| 3D (TraceMD) | Three.js |

### Hardware Target
- NVIDIA RTX 4060 Laptop, 8 GB VRAM, CUDA 12.4
- Windows 11
- Ollama server: `localhost:11434`

---

## Setup & Running

### Prerequisites

- Python 3.11+
- Node.js 22+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU with CUDA (8 GB VRAM minimum)

Pull the required Ollama models once:
```powershell
ollama pull llava
ollama pull mistral
```

---

### MedVis-X

#### 1. Python environment

```powershell
cd medvisx
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Start backend

```powershell
python server.py
# API available at http://localhost:8000
```

#### 3. Start frontend

```powershell
cd frontend
npm install
npm run dev
# UI available at http://localhost:5173
```

#### 4. Verify

```
GET http://localhost:8000/api/health
```
Should return `{ "status": "ok", "device": "cuda", "ollama": true }`.

---

### TraceMD

#### 1. Python environment

```powershell
cd TraceMD/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. HuggingFace access (gated models)

`google/medgemma-4b-it` and `google/paligemma2-3b-pt-896` require accepted terms on HuggingFace.

```powershell
huggingface-cli login
# or set HF_TOKEN=your_token_here
```

#### 3. Download models

```powershell
# From TraceMD/ root
powershell -ExecutionPolicy Bypass -File .\scripts\download_models_git.ps1 -Models trocr,sd15
# For gated models (requires HF access):
powershell -ExecutionPolicy Bypass -File .\scripts\download_models_git.ps1 -Models all -Token $env:HF_TOKEN
```

TraceMD auto-loads local folders under `TraceMD/models/` before falling back to HuggingFace hub IDs.

#### 4. Run backend

```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### 5. Run frontend

```powershell
cd frontend
npm install
npm run dev
# UI at http://localhost:5173, proxied to backend at :8000
```

---

## API Reference

### MedVis-X / TraceMD Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System status, GPU info, Ollama availability |
| `GET` | `/api/models` | Model load status (generator, gradcam, explainer) |
| `POST` | `/api/analyze` | Full pipeline — accepts `multipart/form-data` |

**POST `/api/analyze` — Request fields:**

| Field | Type | Description |
|-------|------|-------------|
| `files` | `UploadFile` | PNG, JPEG, TIFF medical image (optional) |
| `clinical_text` | `string` | Free-text clinical notes (optional) |
| `patient_name` | `string` | Patient name (defaults to "Anonymous") |
| `patient_age` | `string` | Patient age |
| `patient_sex` | `string` | Patient sex |

**POST `/api/analyze` — Response shape:**

```typescript
{
  report_id: string                // "rep_XXXXXXXXXXXX"
  patient: { name, age, sex }
  ocr: { final_text, method_used }
  entities: { symptoms[], medications[], conditions[] }
  prediction: {
    top_disease, confidence, modality, organ,
    hypotheses: { disease, score, model }[]
  }
  images: {
    generated: string,             // base64 PNG (768×768 SDXL)
    heatmap: string,               // base64 PNG (Grad-CAM overlay)
    shap_plot: string              // base64 PNG (SHAP waterfall)
  }
  xai: {
    shap_values: { feature, value, impact }[],
    explanation: string,           // LLaVA 5-section clinical report
    naturalLanguageSummary: string
  }
  provenance: { generator, seed, conditioning, createdAt }
  auditLog: { user, action, timestamp }[]
  timings: { analysis, generation, xai, total }  // seconds
}
```

> TraceMD streams this as `text/event-stream` (SSE). MedVis-X returns it as a single JSON response.

---

## XAI — Explainability Details

### Grad-CAM
- **Backbone:** DenseNet121 pretrained on ImageNet
- **Target layer:** `model.features.denseblock4.denselayer16.conv2`
- **Output:** JET colormap overlay blended onto the generated synthetic image
- Input resized to 224×224, ImageNet-normalized

### SHAP
- **Source:** Feature weights from the hypothesis scorer or structured analyzer
- **Plot:** Horizontal bar chart — red = positive contribution, blue = negative
- **Sanitization:** Non-medical features (email, phone, URL, address) are automatically removed before display

### LLaVA Clinical Explanation
- **Model:** `llava:latest` via Ollama (multimodal, 4.4 GB)
- **Prompt:** Requests a 5-section structured report — Clinical Summary, Image Interpretation, Key Evidence, Differential Considerations, Recommended Next Steps
- **Fallback:** Deterministic Markdown report generated from structured data when Ollama is unavailable
- **Disclaimer always appended:** *"This is a decision-support tool only. All findings must be verified by a qualified clinician."*

---

## VRAM Management

Models are loaded sequentially, never concurrently. After each stage:
- Model references are deleted (`del model`)
- `torch.cuda.empty_cache()` and `gc.collect()` are called

| Model | VRAM | Quantization |
|-------|------|-------------|
| TrOCR (TraceMD) | ~2 GB | 4-bit |
| PaliGemma2 3B (TraceMD) | ~3 GB | 4-bit |
| MedGemma 4B (TraceMD) | ~3 GB | 4-bit |
| SDXL (MedVis-X) | ~3–4 GB | fp16 + CPU offload |
| SD v1.5 (TraceMD) | ~4 GB | fp16 |
| DenseNet121 | ~0.1 GB | fp32 |
| LLaVA via Ollama | ~4.4 GB | 4-bit (managed by Ollama) |

Peak single-stage usage stays within 8 GB on an RTX 4060.

All generated images are watermarked: **"SYNTHETIC — NOT A REAL PATIENT IMAGE"**

---

## Testing

```powershell
# From medvisx/ with .venv active
python test_api.py        # Health + models endpoints
python test_quick.py      # Text-only full pipeline
python test_gen.py        # SDXL image generation
python test_xai.py        # Grad-CAM + SHAP + LLaVA explanation
python test_pipeline.py   # End-to-end integration

# From TraceMD/backend/
python test_pipeline.py
python smoke_test_api.py
```

**Test Summary (MedVis-X — 2nd Review):**

| Category | Total | Pass | Fail |
|----------|-------|------|------|
| Backend (TC-01 – TC-10) | 10 | 10 | 0 |
| Frontend (TC-11 – TC-17) | 7 | 7 | 0 |
| Planned features (TC-18 – TC-25) | 8 | 0 | 8 |
| **Total** | **25** | **17** | **8** |

The 8 failing cases are planned features: DICOM support, PDF export, concurrent requests, batch processing, authentication, analysis history, model preloading, and 3D volume rendering.

---

## Jupyter Notebooks (`MP01/`)

| Notebook | Description |
|----------|-------------|
| `BrainCT_Model.ipynb` | Brain CT tumour classification model |
| `BrainCT_Conditioned_Model.ipynb` | Conditioned diffusion for brain CT |
| `BreastCancer_Model.ipynb` | Breast cancer detection model |
| `ChestCT_LungCancer_Model.ipynb` | Lung cancer CT classification |
| `DRR_Bones_Model.ipynb` | Digitally reconstructed radiograph model |
| `BraTS_Preprocessing.ipynb` | BraTS dataset preprocessing |
| `Kaggle_BrainTumor_Training.ipynb` | Kaggle brain tumour training run |
| `DentalOPG_Xray_Model.ipynb` | Dental panoramic X-ray model |

Datasets (CT scans, BraTS, etc.) are not included in the repo. See `MP01/datasets_info/` for download instructions.

---

## Project Roadmap

| Sprint | Status | Deliverable |
|--------|--------|-------------|
| 1–2 | ✅ Done | CRNN OCR, ClinicalBERT NER, hypothesis scorer |
| 3–4 | ✅ Done | SD v1.5 generation, Grad-CAM, SHAP, LLaVA XAI, Gradio UI |
| 5–6 | ✅ Done | SDXL migration, Two-Model Analyzer (LLaVA+Mistral) |
| 7–8 | ✅ Done | React 19 frontend, FastAPI REST API, audit trail, provenance |
| 9–10 | ⬜ Planned | DICOM support, PDF export, SQLite persistence, batch processing |
| 11–12 | ⬜ Planned | Concurrent requests, 3D volume rendering, JWT auth, Docker |

---

## Disclaimer

**Research prototype. Not for clinical diagnostic use.**

This system is developed as an academic project to explore explainability and traceability in medical AI. It must not be used for actual clinical diagnosis or patient care decisions. All AI-generated images are synthetic and clearly labeled as such.

---

*B.Tech Major Project — Pranav Aditya | Computer Science*
