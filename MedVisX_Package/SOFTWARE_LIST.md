# MedVis-X — Complete Software & Dependency List

---

## System Software (Install Manually)

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| Python | 3.11 or 3.13 | Backend runtime | https://python.org/downloads |
| Node.js | v22 LTS | Frontend build tool & dev server | https://nodejs.org |
| CUDA Toolkit | 12.4 | GPU acceleration for PyTorch | https://developer.nvidia.com/cuda-downloads |
| Ollama | Latest | Local LLM inference server | https://ollama.com/download |
| Git | Latest | (Optional) Model cloning | https://git-scm.com |

---

## Ollama AI Models (Downloaded via `ollama pull`)

| Model | Size | Purpose |
|-------|------|---------|
| `llava:latest` | ~4.4 GB | Vision + language — describes medical images |
| `mistral:latest` | ~4.1 GB | Text structuring — converts descriptions to JSON |

---

## Python Packages (from requirements.txt + extras)

### Deep Learning
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1.0 (cu124) | PyTorch — neural network inference |
| `torchvision` | ≥0.16.0 | Image transforms for DenseNet121 |
| `diffusers` | ≥0.25.0 | Stable Diffusion XL image generation |
| `transformers` | ≥4.36.0 | ClinicalBERT NER, TrOCR |
| `peft` | ≥0.7.0 | LoRA fine-tuning support |
| `accelerate` | ≥0.25.0 | Model offloading (CPU offload for SDXL) |
| `safetensors` | ≥0.4.0 | Fast model weight loading |

### Explainability (XAI)
| Package | Version | Purpose |
|---------|---------|---------|
| `grad-cam` | ≥1.5.0 | Grad-CAM heatmap generation (pytorch-grad-cam) |
| `shap` | ≥0.44.0 | SHAP feature attribution waterfall plots |
| `matplotlib` | ≥3.7.0 | SHAP plot rendering |

### Image & Data Processing
| Package | Version | Purpose |
|---------|---------|---------|
| `Pillow` | ≥10.0.0 | Image loading and manipulation |
| `opencv-python` | ≥4.8.0 | Image preprocessing |
| `albumentations` | ≥1.3.1 | Data augmentation for OCR training |
| `numpy` | ≥1.24.0 | Array operations |
| `scipy` | ≥1.11.0 | Scientific computing |
| `pandas` | ≥2.0.0 | Data manipulation |
| `scikit-learn` | ≥1.3.0 | ML utilities |

### NLP & Text
| Package | Version | Purpose |
|---------|---------|---------|
| `fuzzywuzzy` | ≥0.18.0 | Fuzzy string matching for NER fallback |
| `python-Levenshtein` | ≥0.23.0 | String distance (speeds up fuzzywuzzy) |

### API Server
| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | Latest | REST API framework |
| `uvicorn` | Latest | ASGI server for FastAPI |
| `python-multipart` | Latest | File upload handling (FormData) |

### Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | ≥2.31.0 | HTTP calls to Ollama API |
| `tqdm` | ≥4.66.0 | Progress bars during training |
| `gradio` | ≥4.15.0 | Legacy Gradio UI (optional) |

---

## Node.js / npm Packages (frontend/package.json)

### Dependencies (Runtime)
| Package | Version | Purpose |
|---------|---------|---------|
| `react` | ^19.2.4 | UI component framework |
| `react-dom` | ^19.2.4 | React DOM renderer |
| `@vitejs/plugin-react` | ^5.1.4 | Vite plugin for React |
| `framer-motion` | ^12.34.5 | UI animations and transitions |
| `lucide-react` | ^0.576.0 | Medical iconography |
| `react-dropzone` | ^15.0.0 | Drag-and-drop file upload |

### Dev Dependencies (Build Tools)
| Package | Version | Purpose |
|---------|---------|---------|
| `vite` | ^7.3.1 | Build tool + dev server with HMR |
| `typescript` | ~5.9.3 | Type-safe JavaScript |
| `@types/react` | ^19.2.14 | TypeScript types for React |
| `@types/react-dom` | ^19.2.3 | TypeScript types for React DOM |

---

## AI Models (Auto-Downloaded by HuggingFace)

| Model | Size | Where Cached | Purpose |
|-------|------|-------------|---------|
| `stabilityai/stable-diffusion-xl-base-1.0` | ~6.5 GB | `~/.cache/huggingface/hub/` | Primary image generation |
| `runwayml/stable-diffusion-v1-5` | ~4 GB | `~/.cache/huggingface/hub/` | Fallback image generation |
| `densenet121` (torchvision) | ~32 MB | `~/.cache/torch/hub/` | Grad-CAM backbone |
| `samrawal/bert-base-uncased_clinical-ner` | ~440 MB | `~/.cache/huggingface/hub/` | ClinicalBERT NER (legacy stage) |

---

## Total Disk Space Required

| Component | Approx Size |
|-----------|-------------|
| Ollama (llava + mistral) | ~8.5 GB |
| SDXL model | ~6.5 GB |
| SD v1.5 fallback | ~4 GB |
| ClinicalBERT | ~440 MB |
| DenseNet121 | ~32 MB |
| Python venv + packages | ~5 GB |
| Node.js dependencies | ~500 MB |
| **Total** | **~25 GB** |

---

*All inference runs 100% locally. No cloud API keys required.*
