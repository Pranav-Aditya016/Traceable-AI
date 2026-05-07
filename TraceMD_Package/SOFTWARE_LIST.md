# TraceMD — Complete Software & Dependency List

---

## System Software (Install Manually)

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| Python | 3.11 or 3.12 | Backend runtime | https://python.org/downloads |
| Node.js | v22 LTS | Frontend dev server & build | https://nodejs.org |
| CUDA Toolkit | 12.4+ | GPU acceleration (NVIDIA only) | https://developer.nvidia.com/cuda-downloads |
| Ollama | Latest | Runs Gemma 4 + MedGemma locally | https://ollama.com/download |
| Git | Latest | Clone model weights via Git LFS | https://git-scm.com |
| Git LFS | Latest | Download large model files | https://git-lfs.com |

---

## Ollama AI Models (Pull via `ollama pull`)

| Model Tag | Size | Used In | Purpose |
|-----------|------|---------|---------|
| `gemma4:e4b` | ~4 GB | Stage 1 — OCR | Gemma 4 vision model — reads handwritten & printed medical docs |
| `medgemma:4b` | ~4 GB | Stage 2 — NER | Google MedGemma — extracts clinical entities (symptoms, organs, diagnoses) |
| `medgemma:4b` | ~4 GB | Stage 6 — Explanation | Same model — generates structured clinical narrative |

---

## HuggingFace Models (Downloaded to `TraceMD/models/`)

| Model | HuggingFace ID | Size | Gated? | Used In | Purpose |
|-------|----------------|------|--------|---------|---------|
| Stable Diffusion v1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | ~4 GB | No | Stage 4 — Generator | Stability AI — synthetic 512×512 medical images |
| DenseNet121 | torchvision built-in | ~32 MB | No | Stage 5 — Grad-CAM | Saliency map on original uploaded image |
| MedGemma 4B IT | `google/medgemma-4b-it` | ~8 GB | **Yes** | (stored locally, used via Ollama) | Google medical LLM |
| PaliGemma2 3B | `google/paligemma2-3b-pt-896` | ~6 GB | **Yes** | (stored locally, reference) | Google vision-language |
| TrOCR Large | `microsoft/trocr-large-handwritten` | ~1.3 GB | No | (stored locally, reference) | Microsoft handwriting OCR |

> **Gated models** (MedGemma, PaliGemma) require accepting terms at huggingface.co and using an HF token.

---

## Python Packages (`backend/requirements.txt`)

### API Server
| Package | Purpose |
|---------|---------|
| `fastapi` | REST API + SSE streaming framework |
| `uvicorn[standard]` | ASGI server to run FastAPI |
| `python-multipart` | File upload (multipart/form-data) |
| `sse-starlette` | Server-Sent Events for real-time pipeline progress |
| `pypdfium2` | PDF → PIL image conversion (Page 1 extraction) |

### Deep Learning
| Package | Purpose |
|---------|---------|
| `torch>=2.1.0` | PyTorch — all model inference |
| `torchvision` | DenseNet121 pretrained weights (Grad-CAM) |
| `transformers>=4.40.0` | HuggingFace model loading |
| `diffusers>=0.27.0` | Stable Diffusion pipeline (Stability AI) |
| `accelerate` | Model acceleration + device management |
| `torchcam` | Grad-CAM activation maps (`GradCAM` + `overlay_mask`) |

### Explainability
| Package | Purpose |
|---------|---------|
| `shap` | SHAP values for hypothesis scorer |
| `scipy` | Scientific computing (used by SHAP) |
| `numpy` | Array operations throughout the pipeline |

### Image & Data
| Package | Purpose |
|---------|---------|
| `pillow` | Image loading, conversion, watermarking |
| `opencv-python` | Image preprocessing |

### Tokenizer Support
| Package | Purpose |
|---------|---------|
| `sentencepiece` | Tokenizer for Gemma/MedGemma models |
| `protobuf` | Protocol buffers required by transformers |

### HTTP
| Package | Purpose |
|---------|---------|
| `requests` | HTTP calls to Ollama API (localhost:11434) |

---

## Node.js / npm Packages (`frontend/package.json`)

### Runtime Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `react` | ^18.3.1 | UI framework |
| `react-dom` | ^18.3.1 | React DOM renderer |
| `framer-motion` | ^11.11.9 | Smooth UI animations |
| `three` | ^0.169.0 | Three.js 3D particle background |
| `recharts` | ^2.13.3 | Charts for confidence/SHAP visualization |

### Dev Dependencies (Build Tools)
| Package | Version | Purpose |
|---------|---------|---------|
| `vite` | ^5.4.8 | Frontend build tool + dev server |
| `@vitejs/plugin-react` | ^4.3.1 | Vite React plugin |
| `typescript` | ^5.6.2 | TypeScript compiler |
| `tailwindcss` | ^3.4.13 | Utility-first CSS framework |
| `postcss` | ^8.4.47 | CSS processing |
| `autoprefixer` | ^10.4.20 | CSS vendor prefixes |
| `@types/react` | ^18.3.5 | TypeScript types for React |
| `@types/three` | ^0.169.0 | TypeScript types for Three.js |

---

## Frontend Component Map

```
frontend/src/
├── App.tsx                          ← Root component, state management
├── main.tsx                         ← React entry point
├── types.ts                         ← All TypeScript interfaces
├── index.css                        ← Global Tailwind styles
├── api/
│   └── analyzeStream.ts             ← SSE client — reads real-time pipeline events
├── components/
│   ├── Hero.tsx                     ← Landing hero section
│   ├── UploadZone.tsx               ← Drag-and-drop file uploader
│   ├── PipelineTracker.tsx          ← 6-stage live progress tracker
│   └── SummaryView/
│       ├── index.tsx                ← Results dashboard container
│       ├── OverviewCard.tsx         ← Primary diagnosis + confidence
│       ├── PredictionDetails.tsx    ← Differential diagnoses table
│       ├── ExtractedData.tsx        ← OCR text + entity tags
│       ├── LocalizationViewer.tsx   ← Generated image / Grad-CAM / mask toggle
│       ├── XaiVisualizations.tsx    ← SHAP table + text features
│       ├── NLExplanationTab.tsx     ← Clinical narrative explanation
│       ├── AuditLog.tsx             ← Timestamped action trail
│       ├── SummaryHeader.tsx        ← Report header with patient info
│       ├── SummaryActions.tsx       ← Download / share buttons
│       ├── PipelineExplanation.tsx  ← Pipeline provenance display
│       └── Tabs.tsx                 ← Tab navigation
└── three/
    └── ParticleField.tsx            ← Three.js animated background
```

---

## Total Disk Space Required

| Component | Size |
|-----------|------|
| Ollama (gemma4 + medgemma) | ~8 GB |
| Stable Diffusion v1.5 (local) | ~4 GB |
| MedGemma 4B (local copy) | ~8 GB |
| PaliGemma2 3B (local copy) | ~6 GB |
| TrOCR Large (local copy) | ~1.3 GB |
| Python venv + all packages | ~6 GB |
| Node.js dependencies | ~400 MB |
| **Total** | **~34 GB** |

---

## AI Model Summary

| Model | Creator | Accessed Via | Stage |
|-------|---------|-------------|-------|
| **Gemma 4 (gemma4:e4b)** | Google | Ollama | Stage 1 — OCR |
| **MedGemma (medgemma:4b)** | Google | Ollama | Stage 2 — NER & Stage 6 — Explanation |
| **Stable Diffusion v1.5** | Stability AI | HuggingFace Diffusers (local folder) | Stage 4 — Image Generation |
| DenseNet121 | Torchvision | torchvision (auto-download) | Stage 5 — Grad-CAM |

---

*All inference runs 100% locally. No cloud API keys required.*
