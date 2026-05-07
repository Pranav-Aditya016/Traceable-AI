# MedVis-X — B.Tech Major Project Second Review Documentation

**Project Title:** MedVis-X: Traceable & Explainable Multimodal Clinical Decision Support System  
**Student:** Pranav Aditya  
**Degree:** B.Tech Computer Science  
**Review:** Second Review (Major Project)  
**Date:** March 7, 2026

### Review Roadmap

| Review | Status | Focus | Key Deliverables |
|--------|--------|-------|------------------|
| **1st Review** | ✅ Completed | Problem definition, literature survey, initial architecture | Project proposal, tech stack selection, initial design |
| **2nd Review (Current)** | 🔶 In Progress | Working pipeline, React UI, XAI integration | Full 3-step pipeline, React frontend, 17 PASS + 8 FAIL test cases |
| **3rd Review** | ⬜ Planned | DICOM support, PDF reports, batch processing, performance tuning | DICOM parser, PDF export, concurrent requests, unit tests |
| **Final Review** | ⬜ Planned | Production hardening, IEEE paper submission, final documentation | Auth, Docker deployment, IEEE paper, final presentation |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Document (HLD & LLD)](#2-architecture-document)
3. [Functional Document](#3-functional-document)
4. [Functional Test Cases](#4-functional-test-cases)
5. [Sprint Information](#5-sprint-information)
6. [Demo Preparation](#6-demo-preparation)
7. [PPT Content for 15 Slides](#7-ppt-content)

---

# 1. Project Overview

## 1.1 Abstract

MedVis-X is a fully local, privacy-preserving multimodal clinical decision support system that combines medical document analysis, synthetic medical image generation, and explainable AI (XAI) into a unified pipeline. The system processes medical images (prescriptions, X-rays, CT scans) or free-text clinical notes through a three-step pipeline: (1) Structured Medical Analysis using a two-model LLaVA + Mistral approach via Ollama, (2) Synthetic Medical Image Generation using Stable Diffusion XL (SDXL), and (3) Explainability via Grad-CAM saliency maps, SHAP feature attribution, and natural language clinical explanations. All inference runs 100% locally on an NVIDIA RTX 4060 (8GB VRAM) with no cloud APIs, no internet dependency, and no patient data leaving the device — addressing key concerns in medical AI around data privacy, traceability, and clinical transparency.

## 1.2 Problem Statement

Current medical AI systems suffer from three critical limitations:
1. **Black-box predictions** — Models provide diagnoses without explaining their reasoning, making them untrustworthy for clinical decision-making.
2. **Cloud dependency** — Most systems send sensitive patient data to external APIs (GPT-4V, Google Gemini), creating HIPAA/GDPR compliance risks.
3. **Lack of traceability** — Predictions cannot be traced back to specific input features, making clinical audits impossible.

MedVis-X addresses all three by running entirely on local hardware, providing SHAP-based attribution for every prediction, generating visual explanations via Grad-CAM, and maintaining a complete audit log from input to output.

## 1.3 Objectives

| # | Objective | Implementation |
|---|-----------|----------------|
| 1 | Build a multimodal medical analysis pipeline | Two-model approach: LLaVA (vision) + Mistral (text structuring) via Ollama (`structured_analyzer.py`) |
| 2 | Generate clinically relevant synthetic medical images | SDXL with rich condition-specific prompts, 768×768, 15 inference steps (`lora_generator.py`, `prompt_builder.py`) |
| 3 | Provide explainable AI for every prediction | Grad-CAM (DenseNet121 backbone), SHAP waterfall plots, LLaVA clinical explanations (`gradcam.py`, `shap_explainer.py`, `llava_explainer.py`) |
| 4 | Ensure 100% local execution | Ollama (llava:latest 4.4GB + mistral:latest 4.1GB), SDXL (CPU offloading), DenseNet121 — all on RTX 4060 8GB |
| 5 | Build a professional clinical UI | React 19 + TypeScript + Vite frontend with 5 components, dark medical theme (`frontend/src/`) |
| 6 | Maintain traceability and audit trail | Provenance metadata (generator, seed, conditioning), timestamped audit log, report IDs (`server.py` response schema) |

## 1.4 Technology Stack

### Backend
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Runtime | Python | 3.13.8 | Backend logic |
| API Framework | FastAPI + Uvicorn | Latest | REST API server on port 8000 |
| Deep Learning | PyTorch | 2.6.0+cu124 | Model inference |
| Image Generation | Diffusers (SDXL) | ≥0.25.0 | `stabilityai/stable-diffusion-xl-base-1.0` |
| Local LLM | Ollama | Latest | LLaVA (4.4GB) + Mistral (4.1GB) |
| Grad-CAM | pytorch-grad-cam | ≥1.5.0 | DenseNet121 saliency maps |
| SHAP | shap + matplotlib | ≥0.44.0 | Feature attribution waterfall plots |
| NER (legacy) | ClinicalBERT | samrawal/bert-base-uncased_clinical-ner | Named entity extraction |
| OCR (legacy) | CRNN (ResNet18+BiLSTM) | Custom trained | Prescription text recognition |

### Frontend
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | React | 19.2.4 | UI components |
| Language | TypeScript | 5.9.3 | Type-safe development |
| Build Tool | Vite | 7.3.1 | Development server + HMR |
| Icons | Lucide React | 0.576.0 | Medical UI iconography |
| Animations | Framer Motion | 12.34.5 | UI transitions |
| File Upload | React Dropzone | 15.0.0 | Drag-and-drop file input |

### Infrastructure
| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX 4060 Laptop, 8GB VRAM, CUDA 12.4 |
| OS | Windows |
| Ollama Server | localhost:11434 |
| Backend Port | 8000 (FastAPI) |
| Frontend Port | 5173 (Vite dev) |
| Vite Proxy | `/api` → `http://localhost:8000` |

## 1.5 Key Features

1. **Two-Model Analysis Engine** — LLaVA describes medical images in natural language; Mistral structures descriptions into typed JSON using Ollama native JSON mode
2. **SDXL Medical Image Synthesis** — 8 modality templates × 12+ disease finding banks × randomized quality modifiers produce varied, clinically relevant synthetic images
3. **Triple XAI Stack** — Grad-CAM activation maps (DenseNet121 backbone), SHAP waterfall plots, and LLaVA natural-language clinical explanations
4. **Three Explanation Tones** — Frontend offers Concise (AI summary), Technical (full explanation), and Patient-Friendly explanation modes
5. **Complete Audit Trail** — Every response includes `provenance` (generator, seed, conditioning), `auditLog` (timestamped actions), and `report_id`
6. **Privacy by Design** — Zero network calls for inference; all models run locally; no patient data leaves the device
7. **Legacy 6-Stage Pipeline** — Full implementation of OCR (CRNN), NER (ClinicalBERT), Hypothesis Scoring, Generation, Explainability, and LLM explanation stages

---

# 2. Architecture Document

## 2.1 High-Level Design (HLD)

### System Context Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER'S MACHINE                               │
│                                                                      │
│  ┌──────────────────┐     HTTP/REST      ┌────────────────────────┐  │
│  │  React Frontend   │  ──────────────►  │   FastAPI Backend      │  │
│  │  (Port 5173)      │  ◄──────────────  │   (Port 8000)          │  │
│  │                    │     JSON          │                        │  │
│  │  • UploadPanel     │                   │  ┌──────────────────┐  │  │
│  │  • ProgressBar     │                   │  │ Step 1: Analyzer │  │  │
│  │  • ResultsDashboard│                   │  │ (LLaVA+Mistral)  │──┼──┤
│  │  • LoadingOverlay  │                   │  └──────────────────┘  │  │
│  └──────────────────┘                    │  ┌──────────────────┐  │  │
│                                           │  │ Step 2: SDXL Gen │  │  │
│                                           │  │ (768×768, 15 st) │  │  │
│                                           │  └──────────────────┘  │  │
│                                           │  ┌──────────────────┐  │  │
│  ┌──────────────────┐                    │  │ Step 3: XAI      │  │  │
│  │  Ollama Server    │◄───HTTP──────────│  │ GradCAM+SHAP+    │  │  │
│  │  (Port 11434)     │                   │  │ LLaVA Explain    │  │  │
│  │  • llava:latest   │                   │  └──────────────────┘  │  │
│  │  • mistral:latest │                   └────────────────────────┘  │
│  └──────────────────┘                                                │
│                                                                      │
│  GPU: NVIDIA RTX 4060 8GB VRAM  │  CUDA 12.4  │  PyTorch 2.6.0     │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
User Input                 Backend Pipeline                    Output
─────────                  ────────────────                    ──────
                                                               
Medical Image ───┐                                             
                 ├──► POST /api/analyze ──┐                    
Clinical Text ───┘    (FormData)          │                    
                                          ▼                    
                              ┌─────────────────────┐          
                              │  STEP 1: Structured  │          
                              │  Analysis            │          
                              │                      │          
                              │  A. LLaVA describes  │──► ocrText
                              │     image (vision)   │──► fields (age, sex, symptoms)
                              │  B. Mistral structs  │──► predictions (organ, conditions)
                              │     into JSON        │──► xai (shap_table, text_features)
                              └─────────┬────────────┘          
                                        │                      
                                        ▼                      
                              ┌─────────────────────┐          
                              │  STEP 2: Image       │          
                              │  Generation          │          
                              │                      │          
                              │  build_prompt_from_  │──► prompt (modality+disease+demos)
                              │  analysis() + SDXL   │──► generated image (768×768 PNG)
                              │  Random seed per req │──► provenance.seed
                              └─────────┬────────────┘          
                                        │                      
                                        ▼                      
                              ┌─────────────────────┐          
                              │  STEP 3: XAI         │          
                              │                      │          
                              │  • Grad-CAM heatmap  │──► heatmap image
                              │    (DenseNet121)     │          
                              │  • SHAP waterfall    │──► shap_plot image
                              │    plot (matplotlib) │          
                              │  • LLaVA clinical    │──► explanation text
                              │    explanation       │──► naturalLanguageSummary
                              └─────────┬────────────┘          
                                        │                      
                                        ▼                      
                              ┌─────────────────────┐          
                              │  Response Assembly   │          
                              │  + Provenance        │──► Full JSON response
                              │  + Audit Log         │    to React frontend
                              │  + Timings           │          
                              └─────────────────────┘          
```

## 2.2 Low-Level Design (LLD)

### 2.2.1 Backend Module Map

```
medvisx/
├── server.py                           # FastAPI app — 3 endpoints, lazy model loading
├── config.py                           # All hyperparameters (55 lines)
├── pipeline/
│   ├── __init__.py
│   ├── structured_analyzer.py          # v3 Two-Model Analyzer (LLaVA+Mistral)
│   ├── full_pipeline.py                # Legacy 6-stage orchestrator
│   ├── stage1_ocr/
│   │   ├── crnn_model.py              # CRNN: ResNet18 → BiLSTM → CTC
│   │   ├── medicine_db.py             # Medicine→Disease mapping (32+ entries)
│   │   ├── ocr_inference.py           # OCR inference with trained CRNN + EasyOCR
│   │   └── train_ocr.py              # OCR training: augmentation + CTC loss
│   ├── stage2_ner/
│   │   └── ner_extractor.py           # ClinicalBERT NER + keyword fallback
│   ├── stage3_scoring/
│   │   ├── hypothesis_scorer.py       # Deterministic scoring: S(d) = Σ[w×c×A]
│   │   └── symptom_disease_db.py      # Curated symptom-disease associations
│   ├── stage4_generation/
│   │   ├── lora_generator.py          # SDXL + SD v1.5 fallback, singleton
│   │   └── prompt_builder.py          # 8 modalities × 12+ diseases, rich prompts
│   ├── stage5_explainability/
│   │   ├── gradcam.py                 # DenseNet121 Grad-CAM heatmaps (63 lines)
│   │   └── shap_explainer.py          # SHAP waterfall matplotlib plots (66 lines)
│   └── stage6_local_llm/
│       └── llava_explainer.py          # Ollama multimodal clinical explanation
├── training/
│   ├── train_lora.py                   # LoRA fine-tuning for SD v1.5
│   └── train_ocr.py                    # OCR training runner
├── models/
│   ├── lora_weights/                   # Trained LoRA weight checkpoints
│   └── ocr/                            # OCR model checkpoints
├── utils/
│   ├── image_utils.py                  # Image preprocessing utilities
│   └── text_utils.py                   # Text normalization utilities
├── frontend/                           # React+TypeScript frontend
└── test_*.py                           # 5 test files
```

### 2.2.2 Class and Function Reference

#### `server.py` — FastAPI Backend (380 lines)

| Function / Endpoint | Purpose | Key Logic |
|---------------------|---------|-----------|
| `_pil_to_b64(img)` | Convert PIL Image → base64 string | `BytesIO` → `base64.b64encode` |
| `_to_float(v)` | Safe numpy/torch → Python float | Handles `TypeError`, `ValueError` |
| `_load_generator()` | Thread-safe lazy load SDXL | Double-checked locking with `threading.Lock` |
| `_load_gradcam()` | Thread-safe lazy load DenseNet121 | Same pattern as generator |
| `_load_explainer()` | Thread-safe lazy load OllamaExplainer | Same pattern |
| `GET /api/health` | Health check with GPU info | Returns `{status, device, gpu:{name, vram}, ollama}` |
| `GET /api/models` | Model load status | Returns `{models: {generator, gradcam, explainer, analyzer}}` |
| `POST /api/analyze` | Full 3-step pipeline | Accepts `FormData(files, clinical_text, patient_*)` |

#### `structured_analyzer.py` — Two-Model Analyzer (520 lines)

| Function | Purpose | Key Logic |
|----------|---------|-----------|
| `analyze_medical_document(image, text, name, age, sex)` | Main entry point | Orchestrates LLaVA → Mistral → validation |
| `_call_ollama(prompt, image, model, temperature, force_json)` | Unified Ollama API caller | Supports `format: "json"` for native JSON mode |
| `_parse_json_response(raw)` | Extract JSON from LLM output | Handles markdown fences, truncation repair |
| `_repair_truncated_json(text)` | Fix incomplete JSON from LLM | Balances open brackets/braces/strings |
| `_sanitize_shap_table(shap_table)` | Remove non-medical SHAP features | Filters email/phone/URL/address entries |
| `_make_default_result(...)` | Fallback structure when analysis fails | Returns valid schema with default values |
| `_check_ollama()` | Discover available Ollama models | `GET /api/tags` → `{name: full_name}` dict |

**Prompts:**
- `VISION_DESCRIBE_PROMPT` — Instructs LLaVA to describe document type, text content, patient demographics, clinical findings, anatomical details, and overall assessment (medical-only, no email/phone/URL)
- `STRUCTURE_PROMPT_TEMPLATE` — Instructs Mistral to structure LLaVA's description into strict JSON with `ocrText`, `fields`, `predictions`, `xai` sections
- `TEXT_ONLY_PROMPT_TEMPLATE` — Same as above but for text-only input (no image)

**Error Handling:** 3 retry attempts with increasing temperature (0.1 → 0.2 → 0.3), then fallback without JSON mode, then `_make_default_result()`

#### `lora_generator.py` — SDXL Image Generator (180 lines)

| Class / Method | Purpose | Key Logic |
|----------------|---------|-----------|
| `MedicalImageGenerator` (singleton) | Main generator class | `_instance` pattern via `get_instance()` |
| `_load_model()` | SDXL if cached → SD v1.5 fallback | Checks `~/.cache/huggingface/hub` for snapshots |
| `_is_model_cached(model_id)` | Check HuggingFace cache | `models--org--name/snapshots` directory check |
| `generate(prompt, negative_prompt)` | Generate image | Random seed (`randint(0, 2^32)`), CPU offloaded SDXL |
| `unload()` | Free GPU memory | Moves pipe to CPU, `torch.cuda.empty_cache()` |

**SDXL Configuration:** 15 steps, guidance scale 7.5, 768×768, `torch.float16`, CPU offloading (~3-4GB peak VRAM)  
**SD v1.5 Fallback:** 40 steps, guidance scale 7.5, 512×512, on-device GPU, attention slicing

#### `prompt_builder.py` — Rich Prompt Generation (550+ lines)

| Data Structure | Entries | Purpose |
|----------------|---------|---------|
| `MODALITY_BASES` | 8 modalities × 2-4 templates each | Base descriptions for chest_xray, brain_mri, dermoscopy, fundus, chest_ct, breast_mri, kidney_ct, abdominal_ct |
| `DISEASE_FINDINGS` | 12+ diseases × 2-6 findings each | Disease-specific imaging findings per modality |
| `QUALITY_MODIFIERS` | 6 variants | Randomized quality/style descriptors |
| `MODALITY_NEGATIVES` | 8 modality-specific | Negative prompts to prevent non-medical generation |
| `ORGAN_TO_MODALITY` | 16 mappings | Organ name → imaging modality |
| `CONDITION_TO_MODALITY` | 23 mappings | Disease name → imaging modality |

| Function | Purpose |
|----------|---------|
| `build_prompt(hypothesis_result, entities)` | Legacy prompt builder from scorer output |
| `build_prompt_from_analysis(analysis)` | **Primary** — Builds from structured analyzer output, uses patient demographics + symptoms |
| `_resolve_modality(organ, conditions)` | Maps organ/condition to imaging modality |

#### `gradcam.py` — Grad-CAM Explainability (63 lines)

| Class / Method | Purpose |
|----------------|---------|
| `GradCAMExplainer.__init__()` | Loads DenseNet121 (ImageNet pretrained), targets `denseblock4.denselayer16.conv2` |
| `generate_heatmap(pil_image, target_class)` | Preprocesses to 224×224, computes Grad-CAM, blends JET colormap overlay |
| `unload()` | Moves model to CPU, frees VRAM |

#### `shap_explainer.py` — SHAP Waterfall Plots (66 lines)

| Function | Purpose |
|----------|---------|
| `generate_shap_plot(shap_values, top_disease, top_score)` | Generates horizontal bar chart: red=positive contribution, blue=negative. Returns PIL Image (150 DPI) |

### 2.2.3 Frontend Component Architecture

```
App.tsx (Root)
├── Header.tsx          — Logo, status indicator, GPU name
├── UploadPanel.tsx     — Dropzone, clinical text, patient demographics, Analyze button
├── ProgressBar.tsx     — 3-step progress (Text Extraction → Image Generation → Explainability)
├── LoadingOverlay.tsx  — Full-screen overlay with spinner and step-aware messages
└── ResultsDashboard.tsx — Results display with 4 sections:
    ├── Analysis Summary   — Primary finding, confidence meter, modality, organ, AI summary
    ├── Extracted Data     — OCR text, symptoms/medications/conditions tags, clinical features
    ├── Localization       — Generated image / Grad-CAM / SHAP toggle, download, bbox display
    ├── Feature Attribution— SHAP values table with impact bars
    ├── Explanation        — 3-tone toggle (Concise/Technical/Patient-Friendly)
    ├── Audit Log          — Timestamped user+action table
    └── Timing Bar         — Analysis/Generation/XAI/Total timings in seconds
```

### 2.2.4 TypeScript Interface Definitions (`types.ts`)

```typescript
interface AnalysisResult {
  report_id: string
  patient: { name, age, sex }
  ocr: { final_text, crnn_text, easyocr_text, method_used }
  entities: { symptoms[], medications[], conditions[] }
  prediction: {
    top_disease, confidence, modality, organ,
    hypotheses: { disease, score, model }[],
    localization: { mask: {x,y}[], bbox: number[] }
  }
  images: { generated: base64, heatmap: base64, shap_plot: base64 }
  xai: {
    shap_values: { feature, value, impact }[],
    text_features: { feature, weight }[],
    explanation: string,
    naturalLanguageSummary: string
  }
  provenance: { generator, seed, conditioning: {organ, condition}, createdAt }
  auditLog: { user, action, timestamp }[]
  timings: { analysis, generation, xai, total }
}
```

### 2.2.5 API Communication

| Aspect | Detail |
|--------|--------|
| Protocol | HTTP REST over `localhost` |
| Frontend → Backend | Vite proxy: `/api/*` → `http://localhost:8000/api/*` |
| Upload Format | `multipart/form-data` with fields: `files`, `clinical_text`, `patient_name`, `patient_age`, `patient_sex` |
| Response Format | Single JSON object with all pipeline outputs (see TypeScript interface above) |
| Image Transfer | Base64-encoded PNG strings in `images.generated`, `images.heatmap`, `images.shap_plot` |
| Health Check | `GET /api/health` → `{ status, device, gpu: {name, vram_total_gb, vram_used_gb}, ollama }` |
| CORS | All origins allowed (development mode) |

### 2.2.6 Sequence Diagram — Full Pipeline

```
User          React App          FastAPI          Ollama           SDXL          DenseNet121
 │               │                  │               │               │               │
 │──Upload───►   │                  │               │               │               │
 │               │──POST /analyze──►│               │               │               │
 │               │                  │               │               │               │
 │               │                  │──LLaVA call──►│               │               │
 │               │                  │◄──description─│               │               │
 │               │                  │               │               │               │
 │               │                  │──Mistral JSON►│               │               │
 │               │                  │◄──struct JSON─│               │               │
 │               │                  │               │               │               │
 │               │                  │──prompt───────────────────►   │               │
 │               │                  │◄──768×768 img─────────────│   │               │
 │               │                  │               │               │               │
 │               │                  │──input tensor──────────────────────────────►   │
 │               │                  │◄──heatmap overlay──────────────────────────│   │
 │               │                  │               │               │               │
 │               │                  │──LLaVA explain►│              │               │
 │               │                  │◄──explanation──│              │               │
 │               │                  │               │               │               │
 │               │◄──JSON response──│               │               │               │
 │◄──Render────  │                  │               │               │               │
```

---

# 3. Functional Document

## 3.1 User Stories

| US# | As a... | I want to... | So that... | Acceptance Criteria |
|-----|---------|-------------|------------|---------------------|
| US1 | Clinician | Upload a medical image (X-ray, CT, prescription) | The system analyzes it automatically | Image accepted via drag-and-drop or file browse; PNG/JPEG/TIFF/DICOM supported |
| US2 | Clinician | Enter free-text clinical notes | The system processes text when no image is available | Text area accepts multi-line clinical notes; pipeline runs on text alone |
| US3 | Clinician | See the primary diagnosis with confidence | I can assess the AI's certainty | Top disease, confidence %, confidence meter bar displayed |
| US4 | Clinician | View differential diagnoses | I can consider alternative conditions | Up to 4 hypotheses shown with disease name, score, and model identifier |
| US5 | Clinician | See a synthetic medical image | I can visualize the predicted condition | SDXL-generated 768×768 image with "Synthetic — Not For Diagnosis" badge |
| US6 | Clinician | View Grad-CAM heatmap | I can see which regions the model focused on | JET colormap overlay on generated image, toggle between Generated/Saliency/SHAP |
| US7 | Clinician | Read SHAP feature attribution | I can understand which factors drove the prediction | Table with feature names and impact bars (positive=green, negative=red) |
| US8 | Clinician | Get a natural language explanation | I can understand the AI's reasoning in plain English | 3 tones: Concise (AI summary), Technical (full), Patient-Friendly |
| US9 | Clinician | See extracted clinical entities | I can verify the system correctly understood the input | Color-coded entity tags: red=symptoms, blue=medications, purple=conditions |
| US10 | Auditor | View the audit log | I can trace every step of the analysis | Table showing User, Action, Timestamp for each pipeline step |
| US11 | Auditor | View provenance metadata | I can verify image generation parameters | Generator model, seed, conditioning (organ, condition), timestamp |
| US12 | Clinician | Download generated images | I can include them in reports | Download button for Generated/Heatmap/SHAP images |
| US13 | Clinician | See pipeline timing | I can assess system performance | Timing bar showing Analysis/Generation/XAI/Total in seconds |
| US14 | Clinician | Monitor system health | I know the backend and GPU are operational | Header shows online/offline status, GPU name, Health endpoint returns GPU VRAM |

## 3.2 Functional Specifications

### F1: Medical Document Upload
- **Input:** PNG, JPEG, TIFF, DICOM images via drag-and-drop (`react-dropzone`) or file browse
- **Component:** `UploadPanel.tsx` — uses `useDropzone` with accept filter `image/*`
- **Backend:** `server.py POST /api/analyze` reads uploaded files via `UploadFile`, opens with `PIL.Image.open()`, converts to RGB
- **Validation:** Files must have non-zero size; images must be openable by Pillow

### F2: Clinical Text Input
- **Input:** Free-text clinical notes in textarea
- **Component:** `UploadPanel.tsx` — `<textarea>` bound to `clinicalText` state
- **Backend:** `clinical_text` Form field; processed by `structured_analyzer.py` using `TEXT_ONLY_PROMPT_TEMPLATE`
- **Fallback:** If both image and text are provided, both are used (image description + clinical notes combined)

### F3: Patient Demographics
- **Input:** Patient name, age, sex (optional)
- **Component:** `UploadPanel.tsx` — `patient-row` grid with 3 input fields
- **Backend:** Merged into analysis result via `fields.pseudonym`, `fields.age`, `fields.sex`
- **Default:** Name defaults to "Anonymous"

### F4: Structured Medical Analysis (Step 1)
- **Function:** `analyze_medical_document()` in `structured_analyzer.py`
- **Step A:** LLaVA describes the medical image using `VISION_DESCRIBE_PROMPT` (temperature=0.4, max_tokens=2000)
- **Step B:** Mistral structures description into JSON using `STRUCTURE_PROMPT_TEMPLATE` with `format: "json"` (Ollama native JSON mode, temperature=0.1)
- **Retry Logic:** 3 attempts with increasing temperature (0.1, 0.2, 0.3); fallback without JSON mode; final fallback to `_make_default_result()`
- **Output Schema:** `{ocrText, fields, predictions, xai}`
- **SHAP Sanitization:** `_sanitize_shap_table()` removes non-medical features (email, phone, URL, address, etc.)

### F5: Medical Image Generation (Step 2)
- **Function:** `build_prompt_from_analysis()` in `prompt_builder.py` → `MedicalImageGenerator.generate()` in `lora_generator.py`
- **Prompt Construction:** Modality base template + disease-specific finding + patient demographics + symptoms + quality modifier
- **SDXL Config:** 15 steps, guidance_scale=7.5, 768×768, fp16, CPU offloading
- **Randomization:** Random seed per request (`random.randint(0, 2^32)`), random template/finding/modifier selection
- **Modality Resolution:** `ORGAN_TO_MODALITY` (16 entries) + `CONDITION_TO_MODALITY` (23 entries) → default=chest_xray
- **VRAM Management:** After generation, `torch.cuda.empty_cache()` frees VRAM before Grad-CAM

### F6: Grad-CAM Heatmap (Step 3a)
- **Function:** `GradCAMExplainer.generate_heatmap()` in `gradcam.py`
- **Backbone:** DenseNet121 pretrained on ImageNet
- **Target Layer:** `model.features.denseblock4.denselayer16.conv2`
- **Processing:** Input resized to 224×224, ImageNet-normalized, Grad-CAM computed for highest-scoring class
- **Output:** JET colormap overlay blended onto original image

### F7: SHAP Waterfall Plot (Step 3b)
- **Function:** `generate_shap_plot()` in `shap_explainer.py`
- **Input:** SHAP values dict from structured analyzer (`xai.shap_table` converted to `{feature: value}`)
- **Output:** Horizontal bar chart (matplotlib, 150 DPI): red bars=positive contribution, blue bars=negative
- **Edge Case:** Empty SHAP values → informative "No SHAP attribution data available" image

### F8: Clinical Explanation (Step 3c)
- **Function:** `OllamaExplainer.explain()` in `llava_explainer.py`
- **Input:** Generated image + clinical context (OCR text, entities, top disease, SHAP values)
- **Prompt Structure:** Requests 5 sections: Clinical Summary, Image Interpretation, Key Evidence, Differential Considerations, Recommended Next Steps
- **Fallback:** `_fallback_report()` generates deterministic markdown report when Ollama fails
- **Disclaimer:** Always appended: "This is a decision-support tool only. All findings must be verified by a qualified clinician."

### F9: Response Assembly
- **Function:** Last section of `POST /api/analyze` in `server.py`
- **Fields assembled:**
  - `report_id`: `rep_` prefix + 12-char UUID hex
  - `provenance`: generator name, random seed, conditioning (organ, condition), ISO timestamp
  - `auditLog`: 3 entries (Upload, Analysis complete, Image generated)
  - `naturalLanguageSummary`: From analyzer XAI, or auto-generated from fields
  - `timings`: analysis/generation/xai/total (seconds, 2 decimal places)

### F10: Results Display
- **Component:** `ResultsDashboard.tsx` (460+ lines)
- **Layout:** 3-row grid:
  - Row 1: Analysis Summary card + Extracted Data card (2-column)
  - Row 2: Localization & Provenance card + SHAP Table card (2-column)
  - Row 3: Natural Language Explanation card (full-width, 3-tone toggle)
  - Footer: Audit Log table + Timing bar
- **Image Toggle:** 3-way switch between Generated / Saliency (Grad-CAM) / SHAP Plot
- **Explanation Tones:**
  - Concise: Uses `naturalLanguageSummary` if available
  - Technical: Full `explanation` text
  - Patient-Friendly: Auto-generated layperson summary

## 3.3 Non-Functional Requirements

| NFR | Requirement | Implementation |
|-----|-------------|----------------|
| Performance | Pipeline execution within 60-180 seconds | SDXL: 15 steps, Grad-CAM: <2s, LLaVA calls: ~30s each |
| Privacy | No data leaves the device | All models local (Ollama, SDXL, DenseNet121); no external API calls |
| Scalability | Single-user (research prototype) | Thread-safe model loading; in-memory model caching |
| Reliability | Graceful degradation on failure | 3 retry levels in analyzer; SDXL→SD v1.5 fallback; LLaVA→deterministic report fallback |
| Auditability | Full traceability of every prediction | report_id, provenance (seed, generator), audit log, timing breakdown |
| Usability | Responsive UI, dark medical theme | CSS variables, responsive breakpoints (1024px, 768px), loading overlay |

---

# 4. Functional Test Cases

## 4.1 Backend Test Cases

### TC-01: Health Check Endpoint
| Field | Value |
|-------|-------|
| **ID** | TC-01 |
| **Description** | Verify `/api/health` returns system status |
| **Precondition** | FastAPI server running on port 8000, Ollama running |
| **Steps** | 1. Send `GET /api/health` |
| **Expected** | `{status: "ok", device: "cuda", gpu: {name: "NVIDIA GeForce RTX 4060 Laptop GPU", vram_total_gb: ~8, vram_used_gb: number}, ollama: true}` |
| **Test File** | `test_api.py` |

### TC-02: Text-Only Analysis
| Field | Value |
|-------|-------|
| **ID** | TC-02 |
| **Description** | Pipeline processes clinical text without image |
| **Precondition** | Backend + Ollama running |
| **Steps** | 1. POST `/api/analyze` with `clinical_text="Patient presents with high fever, productive cough, crackles in lower left lobe. SpO2 < 92%."` |
| **Expected** | Response contains: `ocr.method_used="llava_multimodal"`, `prediction.top_disease` contains disease name, `images.generated` is non-empty base64, `xai.explanation` is non-empty |
| **Test File** | `test_quick.py`, `test_pipeline.py` |

### TC-03: Image Upload Analysis
| Field | Value |
|-------|-------|
| **ID** | TC-03 |
| **Description** | Pipeline processes uploaded medical image |
| **Precondition** | Backend + Ollama running |
| **Steps** | 1. POST `/api/analyze` with `files=[medical_image.png]` |
| **Expected** | LLaVA describes image, Mistral structures to JSON, SDXL generates 768×768 image, Grad-CAM produces heatmap |

### TC-04: Image Generation Quality
| Field | Value |
|-------|-------|
| **ID** | TC-04 |
| **Description** | SDXL generates varied images with random seeds |
| **Precondition** | SDXL cached in HuggingFace cache |
| **Steps** | 1. Call `LoRAGenerator.get_instance().generate(prompt)` twice with same prompt |
| **Expected** | Two different images (different random seeds), each 768×768 |
| **Test File** | `test_gen.py` |

### TC-05: Grad-CAM Heatmap Generation
| Field | Value |
|-------|-------|
| **ID** | TC-05 |
| **Description** | Grad-CAM produces valid heatmap overlay |
| **Precondition** | DenseNet121 loaded |
| **Steps** | 1. Generate image via SDXL 2. Run `GradCAMExplainer().generate_heatmap(image)` |
| **Expected** | Returns PIL Image with JET colormap overlay, size matches input |
| **Test File** | `test_xai.py` |

### TC-06: SHAP Plot Generation
| Field | Value |
|-------|-------|
| **ID** | TC-06 |
| **Description** | SHAP waterfall plot renders correctly |
| **Steps** | 1. Call `generate_shap_plot({"fever": 0.25, "cough": 0.22}, "pneumonia", 0.85)` |
| **Expected** | Returns PIL Image with horizontal bars, title shows disease + score |
| **Test File** | `test_pipeline.py` |

### TC-07: SHAP Plot with Empty Values
| Field | Value |
|-------|-------|
| **ID** | TC-07 |
| **Description** | SHAP handles empty feature dict gracefully |
| **Steps** | 1. Call `generate_shap_plot({}, "unknown", 0.0)` |
| **Expected** | Returns PIL Image with "No SHAP attribution data available" message |

### TC-08: Analyzer Retry Logic
| Field | Value |
|-------|-------|
| **ID** | TC-08 |
| **Description** | Structured analyzer retries on JSON parse failure |
| **Steps** | 1. Simulate Mistral returning invalid JSON 2. Verify retry with increased temperature |
| **Expected** | Up to 3 attempts with temperature 0.1→0.2→0.3; fallback without JSON mode; final fallback to `_make_default_result()` |

### TC-09: SHAP Sanitization
| Field | Value |
|-------|-------|
| **ID** | TC-09 |
| **Description** | Non-medical features removed from SHAP table |
| **Steps** | 1. Pass SHAP table containing `{feature: "email", value: "test@test.com", shap: 0.5}` |
| **Expected** | `_sanitize_shap_table()` removes the email entry |

### TC-10: LLaVA Explanation Fallback
| Field | Value |
|-------|-------|
| **ID** | TC-10 |
| **Description** | Deterministic report generated when Ollama fails |
| **Steps** | 1. Disconnect Ollama 2. Call `OllamaExplainer().explain(image, context)` |
| **Expected** | Returns formatted markdown report from `_fallback_report()` with 5 sections |

## 4.2 Frontend Test Cases

### TC-11: File Upload via Drag-and-Drop
| Field | Value |
|-------|-------|
| **ID** | TC-11 |
| **Description** | Drop zone accepts medical images |
| **Steps** | 1. Drag PNG file onto upload zone 2. Verify file chip appears 3. Click X to remove |
| **Expected** | File chip shows truncated filename; removal works |
| **Component** | `UploadPanel.tsx` |

### TC-12: Analyze Button Validation
| Field | Value |
|-------|-------|
| **ID** | TC-12 |
| **Description** | Button disabled when no input provided |
| **Steps** | 1. Load app with empty inputs 2. Check button state |
| **Expected** | Button is disabled; clicking does nothing |
| **Component** | `UploadPanel.tsx` — `disabled={disabled \|\| (!files.length && !clinicalText.trim())}` |

### TC-13: Progress Step Visualization
| Field | Value |
|-------|-------|
| **ID** | TC-13 |
| **Description** | Progress bar shows correct step progression |
| **Steps** | 1. Start analysis 2. Observe progress steps |
| **Expected** | OCR step active→done, Generation step active→done, XAI step active→done |
| **Component** | `ProgressBar.tsx` |

### TC-14: Explanation Tone Toggle
| Field | Value |
|-------|-------|
| **ID** | TC-14 |
| **Description** | Three explanation tones display different content |
| **Steps** | 1. Complete analysis 2. Toggle Concise→Technical→Patient-Friendly |
| **Expected** | Concise shows `naturalLanguageSummary`; Technical shows full explanation; Patient-Friendly shows generated summary |
| **Component** | `ResultsDashboard.tsx` — `getExplanation(tone)` |

### TC-15: Image View Toggle
| Field | Value |
|-------|-------|
| **ID** | TC-15 |
| **Description** | Toggle between Generated/Saliency/SHAP images |
| **Steps** | 1. Click Generated→Saliency→SHAP buttons 2. Verify image changes |
| **Expected** | Generated shows "Synthetic — Not For Diagnosis" badge; Saliency shows "Grad-CAM Saliency" badge; SHAP shows waterfall plot |
| **Component** | `ResultsDashboard.tsx` — `imageView` state |

### TC-16: Responsive Layout
| Field | Value |
|-------|-------|
| **ID** | TC-16 |
| **Description** | UI adapts to different screen sizes |
| **Steps** | 1. Resize to <1024px 2. Resize to <768px |
| **Expected** | Grid columns collapse to single column; progress steps become vertical; padding reduces |
| **Stylesheet** | `index.css` — `@media (max-width: 1024px)`, `@media (max-width: 768px)` |

### TC-17: Backend Status Indicator
| Field | Value |
|-------|-------|
| **ID** | TC-17 |
| **Description** | Header shows online/offline status |
| **Steps** | 1. Start with backend running 2. Stop backend |
| **Expected** | Green dot + "System Online" → Red dot + "Connecting..." |
| **Status** | **PASS** |
| **Component** | `Header.tsx` — status dot color based on `online` prop; `App.tsx` polls health every 15s |

## 4.3 Test Cases with FAIL Status (Known Limitations — Planned for 3rd Review)

### TC-18: DICOM File Upload
| Field | Value |
|-------|-------|
| **ID** | TC-18 |
| **Description** | System should accept and parse native DICOM (.dcm) files with metadata extraction |
| **Steps** | 1. Upload a `.dcm` DICOM file via drag-and-drop 2. Verify DICOM metadata (PatientID, StudyDate, Modality) is extracted 3. Verify image is rendered from pixel data |
| **Expected** | DICOM metadata displayed in Extracted Data card; image rendered correctly from pixel data; modality auto-detected from DICOM header |
| **Actual Output** | System rejects `.dcm` files — only raster formats (PNG/JPEG/TIFF) are accepted. `PIL.Image.open()` cannot parse DICOM natively. |
| **Status** | **FAIL** |
| **More Info** | Requires `pydicom` library integration. Planned for Sprint 9 (3rd Review). Will add DICOM→PIL conversion in `image_utils.py` and metadata extraction in `structured_analyzer.py`. |

### TC-19: PDF Report Generation
| Field | Value |
|-------|-------|
| **ID** | TC-19 |
| **Description** | User should be able to download a comprehensive PDF report of the analysis |
| **Steps** | 1. Complete a full pipeline analysis 2. Click "Download PDF Report" button 3. Verify PDF contains all analysis sections |
| **Expected** | PDF file downloaded with: patient info, analysis summary, generated image, Grad-CAM heatmap, SHAP plot, explanation text, audit log, provenance |
| **Actual Output** | No PDF export feature exists. Only individual image downloads are available. No "Download PDF Report" button in the UI. |
| **Status** | **FAIL** |
| **More Info** | Requires `reportlab` or `weasyprint` backend integration + new `/api/report` endpoint. Planned for Sprint 10 (3rd Review). |

### TC-20: Concurrent Request Handling
| Field | Value |
|-------|-------|
| **ID** | TC-20 |
| **Description** | System should handle multiple simultaneous analysis requests without crashes |
| **Steps** | 1. Send 3 concurrent POST `/api/analyze` requests 2. Verify all 3 return valid responses 3. Check no GPU OOM errors |
| **Expected** | All 3 requests complete successfully; results are independent; no CUDA out-of-memory errors |
| **Actual Output** | Second and third requests fail with `torch.cuda.OutOfMemoryError` when SDXL is generating for the first request. Single-user only — no request queuing. |
| **Status** | **FAIL** |
| **More Info** | Current architecture is single-user. Requires request queue (`asyncio.Queue` or Celery) with sequential GPU access. Planned for Sprint 11 (Final Review). |

### TC-21: DICOM Volume Rendering (3D CT/MRI)
| Field | Value |
|-------|-------|
| **ID** | TC-21 |
| **Description** | System should support multi-slice DICOM series for 3D volume visualization |
| **Steps** | 1. Upload a folder of DICOM slices (CT scan series) 2. Verify system reconstructs 3D volume 3. Verify slice navigation in UI |
| **Expected** | 3D volume reconstructed; axial/sagittal/coronal views available; slice slider for navigation |
| **Actual Output** | System only accepts single 2D images. No folder upload, no multi-slice support, no 3D reconstruction capability. |
| **Status** | **FAIL** |
| **More Info** | Requires `SimpleITK` or `nibabel` for volume reconstruction + Three.js or VTK.js for 3D rendering in frontend. Planned for Sprint 11 (Final Review). |

### TC-22: User Authentication
| Field | Value |
|-------|-------|
| **ID** | TC-22 |
| **Description** | System should require login credentials before allowing access to analysis features |
| **Steps** | 1. Open app without authentication 2. Verify access is restricted 3. Login with valid credentials 4. Verify access granted |
| **Expected** | Login page displayed; unauthorized requests return 401; JWT token issued on login; session persistence |
| **Actual Output** | No authentication system exists. All endpoints are publicly accessible. No login page, no JWT, no session management. CORS allows all origins. |
| **Status** | **FAIL** |
| **More Info** | Requires FastAPI OAuth2 + JWT implementation, React login component, and protected route middleware. Planned for Sprint 12 (Final Review). |

### TC-23: Analysis History & Persistence
| Field | Value |
|-------|-------|
| **ID** | TC-23 |
| **Description** | Previous analysis results should be saved and retrievable by report_id |
| **Steps** | 1. Complete analysis (note report_id) 2. Navigate away 3. Retrieve previous result by report_id |
| **Expected** | `GET /api/report/{report_id}` returns the saved analysis; history list shows past analyses; results persist across sessions |
| **Actual Output** | Results are only held in React state — refreshing the page loses all data. No database, no persistence, no history endpoint. `report_id` is generated but not stored. |
| **Status** | **FAIL** |
| **More Info** | Requires SQLite/PostgreSQL database + SQLAlchemy ORM + new GET endpoint + React history panel. Planned for Sprint 10 (3rd Review). |

### TC-24: Batch Processing Multiple Images
| Field | Value |
|-------|-------|
| **ID** | TC-24 |
| **Description** | System should process multiple images in a single batch request |
| **Steps** | 1. Upload 5 medical images simultaneously 2. Verify all 5 are analyzed 3. Verify results displayed for each |
| **Expected** | Batch progress indicator; individual results per image; combined summary report; total batch timing |
| **Actual Output** | Only the first uploaded image is processed. Multiple file upload is accepted by the dropzone but `server.py` only processes `files[0]`. No batch progress or combined results. |
| **Status** | **FAIL** |
| **More Info** | Requires loop processing in `/api/analyze` endpoint + batch result aggregation + UI tab/carousel for multiple results. Planned for Sprint 9 (3rd Review). |

### TC-25: Model Preloading on Startup
| Field | Value |
|-------|-------|
| **ID** | TC-25 |
| **Description** | Models should be preloaded at server startup to eliminate first-request latency |
| **Steps** | 1. Start the FastAPI server 2. Immediately send an analysis request 3. Measure response time |
| **Expected** | First request completes in <90 seconds (same as subsequent requests); models already in GPU memory at startup |
| **Actual Output** | First request takes 120-180s due to lazy model loading. SDXL loads on first generation request (~45s), DenseNet121 loads on first Grad-CAM request (~5s), Ollama models load on first Ollama call (~10s). |
| **Status** | **FAIL** |
| **More Info** | Requires `@app.on_event("startup")` hook or background preload task. Easy fix — planned for Sprint 9 (3rd Review). |

## 4.4 Test Summary

| Category | Total | PASS | FAIL | Pass Rate |
|----------|-------|------|------|-----------|
| Backend (TC-01 to TC-10) | 10 | 10 | 0 | 100% |
| Frontend (TC-11 to TC-17) | 7 | 7 | 0 | 100% |
| Known Limitations (TC-18 to TC-25) | 8 | 0 | 8 | 0% |
| **Total** | **25** | **17** | **8** | **68%** |

**Note:** All 8 FAIL test cases represent planned features for the 3rd Review and Final Review. The core pipeline (analysis, generation, XAI, UI) is fully functional with 100% pass rate on implemented features.

---

# 5. Sprint Information

## 5.1 Project Timeline

### Epic 1: Core Pipeline Development (Sprint 1-2)

| Sprint | Duration | Focus | Deliverables |
|--------|----------|-------|--------------|
| Sprint 1 | 2 weeks | Foundation | Project setup, config.py, CRNN OCR model (`crnn_model.py`), OCR training pipeline (`train_ocr.py`), EasyOCR integration |
| Sprint 2 | 2 weeks | NLP Pipeline | ClinicalBERT NER (`ner_extractor.py`), symptom-disease database (`symptom_disease_db.py`), hypothesis scorer (`hypothesis_scorer.py`), full pipeline orchestrator (`full_pipeline.py`) |

### Epic 2: Image Generation & XAI (Sprint 3-4)

| Sprint | Duration | Focus | Deliverables |
|--------|----------|-------|--------------|
| Sprint 3 | 2 weeks | Generation | SD v1.5 integration, prompt builder (`prompt_builder.py`), LoRA training script (`train_lora.py`), image generator singleton (`lora_generator.py`) |
| Sprint 4 | 2 weeks | Explainability | DenseNet121 Grad-CAM (`gradcam.py`), SHAP waterfall plots (`shap_explainer.py`), Ollama LLaVA explainer (`llava_explainer.py`), Gradio UI (`app.py`) |

### Epic 3: SDXL Upgrade & Architecture Overhaul (Sprint 5-6)

| Sprint | Duration | Focus | Deliverables |
|--------|----------|-------|--------------|
| Sprint 5 | 2 weeks | SDXL + Rich Prompts | SDXL migration with CPU offloading, 8-modality prompt templates, 12+ disease finding banks, random seed variation, fallback to SD v1.5 |
| Sprint 6 | 2 weeks | Two-Model Analyzer | `structured_analyzer.py` v3 (LLaVA + Mistral), Ollama JSON mode, SHAP sanitization, retry/fallback logic, remove EasyOCR→NER→Scorer dependency |

### Epic 4: Modern Frontend & Integration (Sprint 7-8)

| Sprint | Duration | Focus | Deliverables |
|--------|----------|-------|--------------|
| Sprint 7 | 2 weeks | React Frontend | React 19 + TypeScript + Vite setup, `types.ts` interface definitions, all 5 components (Header, UploadPanel, ProgressBar, LoadingOverlay, ResultsDashboard), dark medical CSS theme |
| Sprint 8 | 2 weeks | Backend Integration | FastAPI server (`server.py`), 3-endpoint REST API, provenance/auditLog/localization/naturalLanguageSummary, Vite proxy config, test files, documentation |

## 5.2 Sprint-Wise Backlog

### Sprint 1 Backlog
- [x] Initialize Python project with venv
- [x] Create `config.py` with all hyperparameters
- [x] Implement CRNN model: ResNet18 CNN → BiLSTM → CTC (`crnn_model.py`)
- [x] Build OCR training pipeline with augmentation (`train_ocr.py`)
- [x] Create medicine-to-disease mapping database (`medicine_db.py`)
- [x] Implement OCR inference with trained model + EasyOCR fallback (`ocr_inference.py`)

### Sprint 2 Backlog
- [x] Integrate ClinicalBERT NER model (`ner_extractor.py`)
- [x] Build keyword-based NER fallback system
- [x] Create curated symptom-disease association database (`symptom_disease_db.py`)
- [x] Implement deterministic hypothesis scorer with SHAP attribution (`hypothesis_scorer.py`)
- [x] Build full 6-stage pipeline orchestrator (`full_pipeline.py`)
- [x] Write NLP pipeline test (`test_pipeline.py`)

### Sprint 3 Backlog
- [x] Integrate Stable Diffusion v1.5 pipeline
- [x] Build rich prompt builder with modality templates (`prompt_builder.py`)
- [x] Implement LoRA fine-tuning script for medical images (`train_lora.py`)
- [x] Create singleton generator with GPU memory management (`lora_generator.py`)
- [x] Write generation test (`test_gen.py`)

### Sprint 4 Backlog
- [x] Implement Grad-CAM with DenseNet121 backbone (`gradcam.py`)
- [x] Build SHAP waterfall plot generator (`shap_explainer.py`)
- [x] Integrate Ollama LLaVA for multimodal clinical explanation (`llava_explainer.py`)
- [x] Build Gradio-based UI (`app.py`)
- [x] Write XAI test (`test_xai.py`)

### Sprint 5 Backlog
- [x] Migrate from SD v1.5 to SDXL with CPU offloading
- [x] Expand prompt builder: 8 modalities × 12+ diseases
- [x] Add modality-specific negative prompts
- [x] Implement random seed per request for varied output
- [x] Add SDXL cache detection with SD v1.5 auto-fallback

### Sprint 6 Backlog
- [x] Build two-model analyzer: LLaVA (vision) + Mistral (structuring)
- [x] Implement Ollama native JSON mode (`format: "json"`)
- [x] Add truncated JSON repair logic
- [x] Add medical-only SHAP sanitization filter
- [x] Implement 3-tier retry/fallback (JSON mode → free-form → default)
- [x] Add `_check_ollama()` model discovery

### Sprint 7 Backlog
- [x] Set up React 19 + TypeScript + Vite project
- [x] Define TypeScript interfaces (`types.ts`)
- [x] Build Header component with status indicator
- [x] Build UploadPanel with react-dropzone and patient demographics
- [x] Build ProgressBar with 3-step visual progression
- [x] Build LoadingOverlay with step-aware messages
- [x] Build ResultsDashboard with all display sections
- [x] Design dark medical CSS theme with CSS variables (`index.css`)

### Sprint 8 Backlog
- [x] Build FastAPI server with 3 endpoints (`server.py`)
- [x] Implement thread-safe lazy model loading
- [x] Add provenance metadata (generator, seed, conditioning)
- [x] Add audit log (timestamped action trail)
- [x] Add localization (mask polygon + bounding box)
- [x] Add naturalLanguageSummary generation
- [x] Configure Vite proxy for API calls
- [x] Write integration tests (`test_api.py`, `test_quick.py`)
- [x] Deploy and test end-to-end flow

### Epic 5: Advanced Features & Hardening (Sprint 9-10) — 3rd Review

| Sprint | Duration | Focus | Deliverables |
|--------|----------|-------|--------------|
| Sprint 9 | 2 weeks | DICOM + Batch + Preload | DICOM file support (`pydicom`), batch image processing, model preloading at startup, expanded prompt templates |
| Sprint 10 | 2 weeks | Persistence + PDF Reports | SQLite database for result storage, `GET /api/report/{id}` endpoint, PDF report generation (`reportlab`), analysis history UI panel |

### Epic 6: Production Hardening (Sprint 11-12) — Final Review

| Sprint | Duration | Focus | Deliverables |
|--------|----------|-------|--------------|
| Sprint 11 | 2 weeks | Concurrency + 3D Volumes | Request queue for concurrent access, DICOM volume rendering (SimpleITK), 3D slice viewer in frontend, performance benchmarking |
| Sprint 12 | 2 weeks | Auth + Deployment + Paper | JWT authentication, Docker containerization, IEEE paper final submission, comprehensive unit test suite, final documentation |

## 5.3 Sprint Status Overview

| Sprint | Epic | Status | Completion |
|--------|------|--------|------------|
| Sprint 1 | Core Pipeline | ✅ Done | 100% |
| Sprint 2 | Core Pipeline | ✅ Done | 100% |
| Sprint 3 | Image Gen & XAI | ✅ Done | 100% |
| Sprint 4 | Image Gen & XAI | ✅ Done | 100% |
| Sprint 5 | SDXL Upgrade | ✅ Done | 100% |
| Sprint 6 | Architecture Overhaul | ✅ Done | 100% |
| Sprint 7 | React Frontend | ✅ Done | 100% |
| Sprint 8 | Backend Integration | ✅ Done | 100% |
| Sprint 9 | DICOM + Batch + Preload | ⬜ Not Started | 0% |
| Sprint 10 | Persistence + PDF | ⬜ Not Started | 0% |
| Sprint 11 | Concurrency + 3D | ⬜ Not Started | 0% |
| Sprint 12 | Auth + Deployment | ⬜ Not Started | 0% |

## 5.4 Sprint Retrospective (Sprints 1-8)

### What Went Well
1. **Local-first architecture** — Choosing Ollama + SDXL eliminated cloud dependency and API cost concerns
2. **Two-model approach** — Splitting vision (LLaVA) and structuring (Mistral) solved the JSON reliability problem that the single-model approach couldn't handle
3. **SDXL migration** — CPU offloading reduced VRAM from ~12GB to ~3-4GB, fitting within the 8GB RTX 4060
4. **Rich prompt system** — 8 modalities × 12+ diseases with randomized templates produce clinically varied images
5. **React migration** — Moving from Gradio to React+FastAPI gave full control over the UI/UX and enabled features like audit log, provenance display, and tone switching

### What Could Be Improved
1. **First-run latency** — Model loading takes 30-60s on cold start; could add a preload endpoint
2. **Single GPU bottleneck** — SDXL and Grad-CAM share GPU; sequential execution could be parallelized with model swapping
3. **LLaVA accuracy** — 7B model sometimes misclassifies document types; could use a larger model or fine-tune
4. **Test coverage** — Tests are primarily integration tests; unit tests for individual functions would improve confidence
5. **DICOM support** — Currently only raster image formats; native DICOM parsing would be valuable

### Scrum Process
- **Sprint Duration:** 2 weeks per sprint
- **Daily Standups:** Code review and progress tracking
- **Sprint Reviews:** Demonstrated working pipeline at each sprint boundary
- **Sprint Planning:** Backlog grooming based on pipeline dependencies
- **Tooling:** Git version control, VS Code (Copilot-assisted development)

---

# 6. Demo Preparation

## 6.1 Prerequisites Checklist

| # | Requirement | Command | Expected Output |
|---|-------------|---------|-----------------|
| 1 | Python venv activated | `cd medvisx && .venv\Scripts\activate` | `(.venv)` prompt |
| 2 | PyTorch CUDA verified | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| 3 | Ollama running | `ollama list` | Shows `llava:latest` and `mistral:latest` |
| 4 | SDXL cached | Check `~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots` exists | Directory with files |
| 5 | Node.js installed | `node --version` | v22.20.0 |
| 6 | Frontend deps installed | `cd frontend && npm install` | No errors |

## 6.2 Launch Sequence

```powershell
# Terminal 1: Backend
cd "c:\Pranav Aditya\Traceable AI\medvisx"
.venv\Scripts\activate
python server.py

# Terminal 2: Frontend
cd "c:\Pranav Aditya\Traceable AI\medvisx\frontend"
npm run dev

# Terminal 3: Ensure Ollama is running
ollama serve    # (usually auto-starts on Windows)
```

**URLs:**
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000/api/health`
- Ollama: `http://localhost:11434`

## 6.3 Demo Script (5-7 minutes)

### Scene 1: System Overview (1 min)
1. Open browser to `http://localhost:5173`
2. Point out the **Header** — MedVis-X logo, "Research Prototype" badge, GPU name (NVIDIA RTX 4060), green "System Online" indicator
3. Explain: "This is a fully local clinical decision support system — no cloud APIs, no internet required. Everything runs on this laptop's GPU."

### Scene 2: Clinical Text Input (30 sec)
1. In the **Clinical Details** panel, enter:
   ```
   Patient presents with high fever, productive cough, and crackles in lower left lobe. SpO2 < 92%.
   ```
2. Fill in patient demographics: Name="John Doe", Age="55", Sex="Male"
3. Click **"Analyze & Generate"**

### Scene 3: Pipeline Execution (1-2 min, wait for pipeline)
1. Show the **Progress Bar** transitioning: Text Extraction → Image Generation → Explainability
2. Show the **Loading Overlay** with step-aware messages:
   - "Analyzing document with LLaVA + Mistral..."
   - "Generating medical visualization with SDXL..."
   - "Computing explanations & XAI..."

### Scene 4: Results Walkthrough (3-4 min)
1. **Analysis Summary Card:**
   - Primary Finding (e.g., "Pneumonia")
   - Confidence % with color-coded meter
   - Imaging Modality (e.g., "Chest Xray")
   - Target Organ (e.g., "Lungs")
   - Analysis Engine: "LLaVA + Mistral"
   - **AI Summary** (blue box) — auto-generated one-paragraph summary

2. **Extracted Data Card:**
   - Show OCR text extraction
   - Color-coded entity tags: symptoms (red), medications (blue), conditions (purple)
   - Key Clinical Features with weight bars

3. **Differential Diagnoses:**
   - Show ranked hypotheses with disease name, score %, and model identifier ("llava_mistral_v3")

4. **Localization & Provenance:**
   - Toggle between **Generated** (show "Synthetic — Not For Diagnosis" badge), **Saliency** (Grad-CAM heatmap with warm colors), **SHAP Plot** (waterfall chart)
   - Show localization bbox coordinates
   - Show provenance: Generator=SDXL, Seed=XXXXX

5. **Feature Attribution (SHAP) Table:**
   - Show feature names with green (positive) / red (negative) impact bars

6. **Natural Language Explanation:**
   - Toggle between **Concise** (AI summary), **Technical** (full 5-section explanation), **Patient-Friendly** (layperson language)
   - Point out the "This is a decision-support tool only" disclaimer

7. **Audit Log:**
   - Show timestamped entries: Upload → Analysis complete → Image generated

8. **Timing Bar:**
   - Show Analysis/Generation/XAI/Total times in seconds

### Scene 5: Architecture Highlight (30 sec)
1. "Under the hood: LLaVA describes the document. Mistral structures it into JSON using Ollama's native JSON mode. SDXL generates a 768×768 synthetic medical image. DenseNet121 produces the Grad-CAM heatmap. And LLaVA explains everything in plain English."
2. "All running locally — zero cloud APIs."

## 6.4 Common Questions & Answers

| Question | Answer |
|----------|--------|
| "Why not use GPT-4V or Gemini?" | Privacy: patient data never leaves the device. Cost: no API fees. Reliability: no internet dependency. |
| "How accurate is the diagnosis?" | This is a decision-support tool, not a diagnostic tool. Accuracy depends on input quality. The system provides confidence scores and differential diagnoses for clinical judgment. |
| "Why two models (LLaVA + Mistral)?" | LLaVA excels at visual description but produces unreliable JSON. Mistral with Ollama JSON mode guarantees valid structured output. |
| "Why SDXL instead of SD v1.5?" | SDXL produces higher quality 768×768 images. CPU offloading keeps VRAM usage at ~3-4GB on our 8GB GPU. |
| "Is the generated image real?" | No — it's clearly labeled "Synthetic — Not For Diagnosis." It's a visualization tool, not a clinical image. |
| "What if Ollama fails?" | 3-tier fallback: retry with different temperature → retry without JSON mode → generate deterministic report. |

## 6.5 Backup Demo Plan

If live demo fails:
1. **Pre-recorded result:** Take a screenshot of a successful pipeline run beforehand
2. **API test:** Run `python test_quick.py` to demonstrate the pipeline works from CLI
3. **Component demo:** Show individual stages: `python test_gen.py` (image generation), `python test_xai.py` (Grad-CAM + explanation)

---

# 7. PPT Content for 15 Slides

## Slide 1: Title Slide
**MedVis-X: Traceable & Explainable Multimodal Clinical Decision Support System**
- B.Tech Major Project — Second Review
- Pranav Aditya
- 100% Local Inference | No Cloud APIs | Explainable AI

## Slide 2: Problem Statement
- Medical AI systems are black boxes — clinicians can't trust what they can't understand
- Cloud-based systems (GPT-4V, Gemini) send patient data to external servers — privacy risk
- No traceability — predictions can't be audited back to specific input features
- **Our Solution:** A fully local, explainable, auditable medical AI pipeline

## Slide 3: Project Objectives
1. Multimodal medical analysis — images + text → structured diagnosis
2. Synthetic medical image generation — condition-specific visualization
3. Triple XAI stack — Grad-CAM + SHAP + Natural Language
4. 100% local execution — privacy by design
5. Professional clinical UI — React + TypeScript
6. Complete audit trail — provenance, timestamps, traceability

## Slide 4: System Architecture
- **Diagram:** 3-layer architecture
  - Layer 1: React Frontend (5 components, TypeScript, Vite)
  - Layer 2: FastAPI Backend (3 endpoints, lazy model loading)
  - Layer 3: AI Models (Ollama LLaVA+Mistral, SDXL, DenseNet121)
- **Key Point:** All layers run on a single machine with RTX 4060 8GB

## Slide 5: Technology Stack
- **Backend:** Python 3.13.8, FastAPI, PyTorch 2.6.0+CUDA 12.4
- **LLMs:** Ollama (llava:4.4GB + mistral:4.1GB)
- **Image Gen:** SDXL (768×768, 15 steps, CPU offloading)
- **XAI:** DenseNet121 Grad-CAM, SHAP matplotlib, LLaVA explanation
- **Frontend:** React 19, TypeScript 5.9, Vite 7.3, Lucide icons
- **Hardware:** NVIDIA RTX 4060 Laptop, 8GB VRAM

## Slide 6: Pipeline Step 1 — Structured Analysis
- **Two-Model Approach:** LLaVA (vision) + Mistral (text structuring)
- LLaVA describes medical images with detailed clinical analysis
- Mistral structures descriptions into typed JSON using Ollama native JSON mode
- **Output:** OCR text, patient fields, organ, conditions with confidence, SHAP table
- **Reliability:** 3-tier retry + fallback (JSON mode → free-form → deterministic)
- **File:** `structured_analyzer.py` (520 lines)

## Slide 7: Pipeline Step 2 — Medical Image Generation
- **Model:** Stable Diffusion XL (768×768, fp16, CPU offloading)
- **Rich Prompts:** 8 modality templates × 12+ disease finding banks
  - Modalities: Chest X-ray, Brain MRI, Dermoscopy, Fundus, Chest CT, Breast MRI, Kidney CT, Abdominal CT
- **Randomization:** Different image per request (random seed, template, finding, quality modifier)
- **Files:** `prompt_builder.py` (550+ lines), `lora_generator.py` (180 lines)

## Slide 8: Pipeline Step 3 — Explainable AI
- **Grad-CAM:** DenseNet121 backbone → activation heatmap overlay (JET colormap)
- **SHAP:** Feature attribution waterfall plot (which features drove the prediction)
- **LLaVA Clinical Explanation:** 5-section structured report (Summary, Image Interpretation, Key Evidence, Differentials, Next Steps)
- **Disclaimer:** "Decision-support tool only. All findings must be verified by a qualified clinician."
- **Files:** `gradcam.py`, `shap_explainer.py`, `llava_explainer.py`

## Slide 9: Frontend — Clinical UI
- **Screenshot:** Full pipeline result display
- **Components:**
  - UploadPanel: Drag-and-drop image upload + clinical text input + patient demographics
  - ProgressBar: 3-step visual progression (Text Extraction → Image Generation → Explainability)
  - ResultsDashboard: Analysis Summary, Extracted Data, Localization & Provenance, SHAP Table, 3-Tone Explanation, Audit Log, Timings
- **Design:** Dark medical theme, responsive (1024px/768px breakpoints)

## Slide 10: Traceability & Audit
- **Provenance:** Generator model, seed, conditioning (organ, condition), ISO timestamp
- **Audit Log:** Timestamped entries for Upload, Analysis, Image Generated
- **Report ID:** Unique `rep_XXXXXXXXXXXX` per analysis
- **Localization:** Bounding box coordinates for predicted finding region
- **SHAP Sanitization:** Non-medical features (email, phone, URL) automatically filtered

## Slide 11: Privacy & Security
- **Zero Cloud Dependency:** All inference on local GPU
- **No Patient Data Transmission:** Ollama (localhost:11434), SDXL (local), DenseNet121 (local)
- **Synthetic Images Labeled:** "Synthetic — Not For Diagnosis" overlay badge
- **CORS for Development Only:** Production builds served statically by FastAPI
- **Input Validation:** File size check, PIL.Image.open validation, FormData sanitization

## Slide 12: Legacy 6-Stage Pipeline
- **Stage 1 — OCR:** Custom CRNN (ResNet18 → BiLSTM → CTC) + EasyOCR fallback
- **Stage 2 — NER:** ClinicalBERT (samrawal/bert-base-uncased_clinical-ner) + keyword fallback
- **Stage 3 — Scoring:** Deterministic formula: S(d) = Σ[w_imp × w_rec × A(entity, disease)]
- **Stage 4 — Generation:** SDXL with rich prompt builder
- **Stage 5 — XAI:** Grad-CAM + SHAP
- **Stage 6 — LLM:** Ollama LLaVA clinical explanation
- **Evolution:** Stages 1-3 replaced by Two-Model Analyzer for better accuracy and reliability

## Slide 13: Testing & Quality
- **25 Test Cases** — 17 PASS (core pipeline) + 8 FAIL (planned features)
- **5 Test Files:** `test_api.py`, `test_gen.py`, `test_pipeline.py`, `test_quick.py`, `test_xai.py`
- **Core Pipeline:** 100% pass rate on all implemented features
- **Known Limitations (FAIL):** DICOM support, PDF export, concurrent requests, batch processing, authentication, analysis history, model preloading, 3D volume rendering
- **Graceful Degradation:** SDXL→SD v1.5 fallback, LLaVA→deterministic report, ClinicalBERT→keyword NER

## Slide 14: Results & Demo
- **Demo:** Live pipeline execution on clinical text input
- **Typical Timings:** Analysis ~30s, Generation ~15s, XAI ~30s, Total ~75s
- **Output Quality:** Clinically relevant synthetic images, structured SHAP attribution, multi-tone explanations
- **Key Achievement:** Entire pipeline runs on a single RTX 4060 laptop with 8GB VRAM

## Slide 15: Roadmap — 3rd Review & Final Review
**2nd Review Achievements (Current):**
- ✅ Full 3-step pipeline (LLaVA+Mistral → SDXL → Triple XAI)
- ✅ React 19 frontend with 5 components
- ✅ 17/25 test cases passing (68% overall)
- ✅ Complete audit trail + provenance

**3rd Review Plan (Sprint 9-10):**
- DICOM native support with `pydicom` metadata extraction
- PDF report generation via `reportlab`
- SQLite persistence + analysis history retrieval
- Batch image processing (multi-file)
- Model preloading on startup (eliminate cold-start latency)

**Final Review Plan (Sprint 11-12):**
- Concurrent request handling with GPU queue
- DICOM 3D volume rendering (SimpleITK + VTK.js)
- JWT authentication + protected routes
- Docker containerization for deployment
- IEEE paper final submission
- Comprehensive unit test suite (target: 90%+ pass rate)

---

*Document generated for B.Tech Major Project Second Review — March 7, 2026*  
*Project: MedVis-X — Traceable & Explainable Multimodal Clinical Decision Support System*  
*Next Review: 3rd Review (Sprint 9-10 deliverables)*
