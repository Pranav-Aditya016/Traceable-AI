# TraceMD

TraceMD is a fully local, end-to-end clinical decision support research prototype.
It processes handwritten prescriptions or printed/typed medical reports through a 6-stage AI pipeline and returns structured entities, hypothesis scores, synthetic imaging, Grad-CAM, SHAP attribution, and a final clinical narrative.

## Stack

- Backend: Python 3.11, FastAPI, SSE (`sse-starlette`)
- Frontend: React 18 + TypeScript + Vite + Tailwind + Three.js + Framer Motion + Recharts
- Hardware target: RTX 4060 (8GB VRAM)

## Project Structure

```text
TraceMD/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── pipeline/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── package.json
└── README.md
```

## Setup

### 1) Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Hugging Face access

`google/medgemma-4b-it` and `google/paligemma2-3b-pt-896` require gated access.

```bash
huggingface-cli login
```

Ensure your account has accepted terms for:
- `google/medgemma-4b-it`

Optional token env var:

```bash
set HUGGINGFACE_HUB_TOKEN=your_token_here
```

Or (also supported):

```bash
set HF_TOKEN=your_token_here
```

### 2b) Download models with Git LFS (local folders)

If you prefer local model folders instead of runtime downloads, use:

```bash
cd ..
powershell -ExecutionPolicy Bypass -File .\scripts\download_models_git.ps1 -Models trocr,sd15
```

To include gated models (requires access + token):

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\download_models_git.ps1 -Models all -Token $env:HF_TOKEN
```

Notes:
- `medgemma-4b-it` and `paligemma2-3b-pt-896` are gated and still require approved Hugging Face access.
- TraceMD auto-loads local folders under `TraceMD/models/` when present, before falling back to hub IDs.

### 3) Run backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4) Frontend

```bash
cd ../frontend
npm install
npm run dev
```

Frontend defaults to `http://localhost:8000` for API. Override with:

```bash
set VITE_API_BASE=http://localhost:8000
```

## Endpoint

- `POST /api/analyze`
  - multipart form-data: `file` (`.jpg`, `.jpeg`, `.png`, `.pdf`)
  - response: `text/event-stream`

## Model Pipeline

1. OCR router (heuristic) → TrOCR or PaliGemma 2
2. MedGemma 4B entity extraction
3. Symbolic scoring + SHAP
4. Stable Diffusion v1.5 image generation
5. DenseNet121 Grad-CAM
6. MedGemma 4B narrative explanation (reused from Stage 2)

## VRAM & Safety Rules

- Large models are loaded sequentially, never concurrently
- Each stage unloads model references and runs `torch.cuda.empty_cache()`
- TrOCR, PaliGemma2, and MedGemma use 4-bit quantization
- Stable Diffusion runs in fp16
- Generated image watermark is always applied:
  - `SYNTHETIC — NOT A REAL PATIENT IMAGE`
- UI disclaimer is always shown:
  - `Research prototype. Not for clinical diagnostic use.`

## Disclaimer

Research prototype. Not for clinical diagnostic use.
