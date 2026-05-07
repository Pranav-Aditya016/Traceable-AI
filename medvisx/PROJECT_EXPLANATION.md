# MedVis-X — Complete Project Explanation (Study Guide)

> **Author:** Pranav Aditya | **Degree:** B.Tech Computer Science, SRM University  
> **Project:** Major Project — Traceable & Explainable Multimodal Clinical Decision Support System  
> **Hardware:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), CUDA 12.4

---

## What Is MedVis-X?

MedVis-X is a **medical imaging AI system** that takes a **medical prescription or medical report** (handwritten or printed) from a doctor, **analyzes** it using AI models to extract symptoms/diseases/medications, **generates** a realistic synthetic medical image showing the detected condition, and then **explains** its reasoning in plain English — completely locally on your laptop, no cloud, no internet, no patient data ever leaves your machine.

Think of it like this:

```
Doctor uploads prescription/report  →  AI reads & extracts medical info  →  AI generates a reference image of the condition  →  AI explains WHY it thinks so
```

---

## The 3-Step Pipeline (How It Actually Works)

The entire system runs in **3 steps**. When a doctor clicks "Analyze & Generate", this is what happens behind the scenes:

### Step 1: Structured Medical Analysis (LLaVA + Mistral via Ollama)

**What happens:** The uploaded medical prescription or report is sent to two AI models, one after the other.

**Model 1 — LLaVA (Large Language and Vision Assistant)**
- **What it is:** A multimodal AI model — it can "see" images AND understand text (like GPT-4 Vision, but runs locally)
- **Size:** 4.4 GB
- **Why we use it:** LLaVA reads the uploaded prescription/report and extracts medical information from it — symptoms, diseases, medications, patient details. Example: *"The prescription mentions bilateral infiltrates in lower lobes, patient presenting with cough, fever, and dyspnea consistent with pneumonia..."*
- **Why not just use LLaVA for everything?** LLaVA is great at reading and understanding documents, but it can't reliably output perfectly structured JSON data. It sometimes adds extra text, forgets commas, or changes the format.

**Model 2 — Mistral (7B text model)**
- **What it is:** A pure text LLM (no vision) — fast, accurate, good at following instructions
- **Size:** 4.1 GB
- **Why we use it:** We take LLaVA's natural language extraction and ask Mistral to convert it into strict JSON format with fields like: patient info, organ detected, disease predictions with confidence scores, symptoms, localization coordinates, and SHAP-like feature attributions.
- **Why not just use Mistral alone?** Mistral can't see images/documents. It needs LLaVA to read the prescription/report first.

**Why two models instead of one?**
- LLaVA = good at READING prescriptions/reports (handwritten or printed), bad at structured output
- Mistral = good at STRUCTURING text into JSON, can't see documents
- Together = best of both worlds

**How they communicate — Ollama:**
- **What is Ollama?** A local AI model server. Think of it like a mini ChatGPT server running on your laptop at `http://localhost:11434`.
- **Why Ollama?** No internet needed. No API keys. No cloud costs. Patient data never leaves the machine. We just send HTTP requests to localhost.
- **Mistral's JSON mode:** Ollama has a feature called `"format": "json"` which forces Mistral to output valid JSON. This is critical — without it, the LLM sometimes adds explanatory text around the JSON.

**Retry logic:** If Mistral fails to produce valid JSON (it happens ~5% of the time), the system retries up to 3 times with increasing temperature (0.1 → 0.2 → 0.3). If all retries fail, it falls back to non-JSON mode and tries to extract JSON from the text. If even that fails, it returns a safe default result.

**Output of Step 1:** A structured JSON object like:
```json
{
  "ocrText": "Bilateral infiltrates in lower lobes...",
  "fields": {
    "pseudonym": "Patient_A",
    "age": 45,
    "sex": "Male",
    "symptoms": ["cough", "fever", "dyspnea"]
  },
  "predictions": {
    "organ": "lungs",
    "conditions": [
      {"label": "pneumonia", "confidence": 0.87, "model": "llava_mistral_v3"},
      {"label": "tuberculosis", "confidence": 0.45, "model": "llava_mistral_v3"}
    ],
    "localization": {"mask": [...], "bbox": [50, 100, 400, 350]}
  },
  "xai": {
    "text_features": [{"feature": "bilateral infiltrates", "weight": 0.35}],
    "shap_table": [{"feature": "infiltrates", "value": "bilateral", "shap": 0.42}],
    "naturalLanguageSummary": "The analysis suggests pneumonia based on..."
  }
}
```

---

### Step 2: Medical Image Generation (Stable Diffusion XL)

**What happens:** Using the diagnosis from Step 1, the system generates a **synthetic reference medical image** showing what that condition typically looks like.

**Model — Stable Diffusion XL (SDXL)**
- **What it is:** A state-of-the-art image generation model by Stability AI. It creates images from text descriptions.
- **Model ID:** `stabilityai/stable-diffusion-xl-base-1.0`
- **Resolution:** 768 × 768 pixels
- **Inference Steps:** 15 (how many times the model refines the image — more steps = better quality but slower)
- **Guidance Scale:** 7.5 (how closely the image follows the text prompt — higher = more literal, lower = more creative)
- **Size:** ~6.5 GB on disk

**Why SDXL instead of SD v1.5?**
- SDXL produces 768×768 images (vs 512×512 for v1.5) — more detail
- SDXL has better anatomical understanding
- SDXL follows prompts more accurately
- The system still has SD v1.5 as a **fallback** — if SDXL isn't downloaded, it auto-falls back to v1.5 (512×512, 40 steps)

**How the prompt is built (`prompt_builder.py`):**
The system doesn't just say "generate a chest X-ray of pneumonia." It constructs a detailed, randomized prompt like:
> *"Posteroanterior chest radiograph, standard clinical imaging, showing bilateral patchy airspace opacities with air bronchograms predominantly in lower lobes, 45-year-old Male patient, patient presenting with cough, fever, dyspnea, high-resolution diagnostic quality medical scan"*

This is built from:
- **8 modality bases** — templates for chest X-ray, brain MRI, dermoscopy, fundus photography, chest CT, breast MRI, kidney CT, abdominal CT
- **12 disease findings** — specific radiological descriptions for pneumonia, tuberculosis, glioma, melanoma, diabetic retinopathy, myocardial infarction, lung cancer, breast cancer, kidney stone, appendicitis, asthma, UTI
- **6 quality modifiers** — random style/quality suffixes to add variety
- **26 organ-to-modality mappings** — so "lungs" → "chest_xray", "brain" → "brain_mri", etc.
- **25 condition-to-modality mappings** — so "pneumonia" → "chest_xray", "glioma" → "brain_mri", etc.

Each time you run it, `random.choice()` picks different template variants, so you get different (but medically accurate) images each time.

**Negative prompt:** Also sent to SDXL to avoid bad outputs:
> *"blurry, low quality, text, watermark, signature, cartoon, illustration, 3d render, painting, sketch, artifacts, noise, distorted anatomy..."*

**VRAM management (critical for 8 GB GPU):**
- `enable_model_cpu_offload()` — keeps most of the model on CPU RAM, only moves the active layer to GPU during inference. This means SDXL (which normally needs ~12 GB VRAM) can run on our 8 GB card.
- `enable_vae_slicing()` — processes the VAE (image decoder) in slices instead of all-at-once, further reducing peak VRAM usage.
- After generation, the model can be `unload()`ed to CPU to free VRAM for the next step.

**Singleton pattern:** The generator uses a singleton (`get_instance()`) so it's loaded once and reused across requests. Loading SDXL takes ~30-45 seconds; we only pay that cost once.

**Output of Step 2:** A 768×768 PIL Image of the synthetic medical image showing the detected condition.

---

### Step 3: Explainability (Grad-CAM + SHAP + LLaVA)

**What happens:** The system now explains its reasoning using three different explainability methods. This is the "traceable" part — the doctor can see WHY the AI made its decision.

#### 3a. Grad-CAM (Gradient-weighted Class Activation Mapping)

**Model — DenseNet121**
- **What it is:** A convolutional neural network (CNN) pretrained on ImageNet (1.2M images, 1000 classes)
- **Why DenseNet121?** It's compact (~8M parameters), each layer connects to every other layer (dense connections), and it's widely used in medical imaging (CheXNet, the famous chest X-ray model, uses DenseNet121)
- **Target layer:** `features.denseblock4.denselayer16.conv2` — the deepest convolutional layer, which captures the highest-level features

**What is Grad-CAM?**
- It answers: "Which pixels in the image did the model focus on to make its decision?"
- It works by: (1) Forward pass through DenseNet121, (2) Compute gradients of the predicted class w.r.t. the target layer, (3) Weight the layer's activations by these gradients, (4) Produce a heatmap showing important regions
- The heatmap is overlaid on the original image using the JET colormap (blue = unimportant, red = very important)

**Output:** A heatmap overlay image showing which regions the model thinks are diagnostically relevant.

#### 3b. SHAP (SHapley Additive exPlanations)

**What is SHAP?**
- It answers: "Which clinical features (symptoms, findings, patient demographics) contributed most to the diagnosis?"
- Named after Lloyd Shapley's game theory concept — it calculates the "fair contribution" of each feature to the final prediction

**How we use it:**
- The SHAP values come from Step 1's Mistral output — each text feature gets a weight (e.g., "bilateral infiltrates" → +0.42, "no fever" → -0.15)
- `generate_shap_plot()` creates a horizontal waterfall bar chart:
  - **Red bars** = features that pushed TOWARD the diagnosis (positive contribution)
  - **Blue bars** = features that pushed AWAY from the diagnosis (negative contribution)
  - Sorted by absolute magnitude

**Output:** A matplotlib bar chart image + a table of features with their SHAP values.

#### 3c. LLaVA Natural Language Explanation

**What happens:** The generated image from Step 2 is sent back to LLaVA (same model from Step 1) with the clinical context, and LLaVA writes a detailed **clinical report** in Markdown format with 5 sections:

1. **Clinical Summary** — one-paragraph overview
2. **Image Interpretation** — what the AI sees in the generated image
3. **Key Evidence** — specific radiological signs supporting the diagnosis
4. **Differential Considerations** — other possible diagnoses
5. **Recommended Next Steps** — suggested follow-up tests/treatments

A disclaimer is always appended: *"AI-generated report for research purposes only. Not a substitute for professional medical judgment."*

**Three tone modes in the frontend:**
- **Concise** — bullet points, short
- **Technical** — full medical jargon
- **Patient-Friendly** — simple language a patient could understand

**Output of Step 3:** Grad-CAM heatmap + SHAP plot + full text explanation.

---

## The Frontend (What the Doctor Sees)

### Technology Stack
| Tech | Version | Why |
|------|---------|-----|
| **React** | 19 | Modern component-based UI framework |
| **TypeScript** | 5.9 | Type safety — catches bugs before runtime |
| **Vite** | 7.3 | Blazing fast dev server + HMR (hot module replacement) |
| **lucide-react** | 0.576 | Clean medical-friendly icon library |
| **framer-motion** | 12.34 | Smooth animations and transitions |
| **react-dropzone** | 15.0 | Drag-and-drop file upload |

### Frontend Architecture

```
frontend/
├── src/
│   ├── App.tsx                    ← Root component (state machine)
│   ├── api.ts                     ← HTTP calls to FastAPI backend
│   ├── types.ts                   ← TypeScript interfaces
│   ├── main.tsx                   ← React entry point
│   ├── index.css                  ← Full dark-theme CSS (~940 lines)
│   └── components/
│       ├── Header.tsx             ← Top navigation bar
│       ├── UploadPanel.tsx        ← Drag-drop + patient form
│       ├── ProgressBar.tsx        ← 3-step pipeline progress
│       ├── LoadingOverlay.tsx     ← Full-screen loading spinner
│       └── ResultsDashboard.tsx   ← All results (512 lines, biggest component)
├── package.json
└── vite.config.ts                 ← Proxies /api → localhost:8000
```

### Component Breakdown

**`Header.tsx` (43 lines)** — Top bar showing:
- MedVis-X logo
- "Research Prototype" badge
- GPU name (e.g., "NVIDIA RTX 4060")
- Online/Offline status dot (green = backend running)

**`UploadPanel.tsx` (~175 lines)** — The upload screen:
- Drag-and-drop zone for medical prescriptions/reports (accepts .png, .jpg, .jpeg, .bmp, .tiff, .dcm)
- File thumbnail chips with remove button
- Patient info form: Name, Age, Sex dropdown (Male/Female/Other)
- Clinical notes textarea for symptoms/history
- "Analyze & Generate" button (disabled until file uploaded + backend online)

**`ProgressBar.tsx` (43 lines)** — Shows the 3 pipeline steps:
- Step 1: Medical Analysis (LLaVA + Mistral)
- Step 2: Image Generation (SDXL)
- Step 3: Explainability (Grad-CAM + SHAP + LLaVA)
- Active step pulses, completed steps show checkmark

**`LoadingOverlay.tsx` (32 lines)** — Full-screen translucent overlay with spinner during processing.

**`ResultsDashboard.tsx` (~512 lines)** — The results page (biggest and most complex):
- **Analysis Summary Card:** Top prediction with confidence meter (color-coded: green > 70%, yellow > 40%, red ≤ 40%), modality, organ, engine name, NLS summary, differential diagnoses list
- **Extracted Data Card:** OCR text display, symptom/medication/condition tags, text feature attribution bar chart
- **Image Viewer:** Toggle between Generated Image / Grad-CAM Saliency Map / SHAP Plot, with download button
- **SHAP Feature Table:** Sortable table of features with positive/negative contribution indicators
- **Natural Language Explanation:** The LLaVA-generated clinical report with 3 tone mode toggle (Concise/Technical/Patient-Friendly) and copy button
- **Audit Log:** Table showing every pipeline step with timestamp, action, and result
- **Timing Bar:** Footer showing how long each step took in seconds

**`App.tsx` (120 lines)** — The brain of the frontend:
- State machine with steps: `idle` → `ocr` → `generation` → `xai` → `done` (or `error`)
- Health check every 15 seconds via `GET /api/health` (returns GPU name + status)
- On "Analyze", sends `FormData` (files + patient info) via `POST /api/analyze`
- Simulates step progression with timers while waiting for backend response
- When backend responds, jumps to `done` and shows `ResultsDashboard`

**`api.ts` (49 lines)** — Two fetch wrappers:
- `analyzeImages(files, clinicalText, patientName, patientAge, patientSex)` → POST to `/api/analyze`
- `checkHealth()` → GET `/api/health`

**`types.ts` (66 lines)** — TypeScript interfaces:
- `AnalysisResult` — the full response object from the backend
- `PipelineStep` — union type: `"idle" | "ocr" | "generation" | "xai" | "done" | "error"`
- `Localization`, `Provenance`, `AuditLogEntry` — sub-types

### How Frontend Talks to Backend

Vite's dev server (port 5173) proxies all `/api/*` requests to FastAPI (port 8000):

```
Browser (localhost:5173)  →  /api/analyze  →  Vite proxy  →  FastAPI (localhost:8000)
```

This avoids CORS issues and means the frontend just calls `/api/analyze` without knowing the backend's port.

---

## The Backend (FastAPI Server)

### `server.py` (262 lines)

**What is FastAPI?**
- A modern Python web framework — fast (async), auto-generates API docs, built-in request validation
- Runs with `uvicorn` on port 8000

**Three endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Returns `{"status": "online", "gpu": "NVIDIA RTX 4060 Laptop GPU", ...}` |
| `/api/models` | GET | Returns which AI models are loaded and their memory usage |
| `/api/analyze` | POST | **The main endpoint** — receives image + patient info, runs full 3-step pipeline, returns all results |

**Lazy loading:** Models are NOT loaded when the server starts. They're loaded on first request:
- `_load_generator()` → loads SDXL/SD v1.5 (first call takes ~30-45s)
- `_load_gradcam()` → loads DenseNet121 (fast, ~2s)
- `_load_explainer()` → loads OllamaExplainer (just checks Ollama is running)

This means the server starts instantly. The first analysis takes longer, but subsequent ones are fast.

**The `/api/analyze` flow:**

```python
# 1. Receive uploaded prescription/report image + patient info from React frontend
image = Image.open(uploaded_file)  # prescription or medical report photo

# 2. Step 1: LLaVA + Mistral analysis
analysis = analyze_medical_document(image, clinical_text, patient_name, patient_age, patient_sex)

# 3. Step 2: Build SDXL prompt from analysis results
prompt, negative = build_prompt_from_analysis(analysis)
generated_image = generator.generate(prompt, negative)

# 4. Step 3a: Grad-CAM heatmap
heatmap = gradcam.generate_heatmap(generated_image)

# 5. Step 3b: SHAP plot
shap_plot = generate_shap_plot(analysis["xai"]["shap_table"], top_disease, top_confidence)

# 6. Step 3c: LLaVA clinical explanation
explanation = explainer.explain(generated_image, clinical_context_string)

# 7. Convert images to base64 and return everything as JSON
return {
    "report_id": uuid,
    "patient": {...},
    "ocr": {...},
    "entities": {...},
    "prediction": {...},
    "images": {"generated": base64, "heatmap": base64, "shap": base64},
    "xai": {"explanation": explanation, "shap_values": [...], ...},
    "provenance": {...},
    "auditLog": [...],
    "timings": {"step1": 12.3, "step2": 8.5, "step3": 5.2}
}
```

**Images are sent as base64:** The backend converts PIL Images to base64 strings and embeds them in the JSON response. The frontend decodes these with `data:image/png;base64,...` to display them.

---

## Every File Explained

### Backend Files

| File | Lines | What It Does |
|------|-------|--------------|
| `config.py` | 57 | All settings in one place: device (CUDA/CPU), model paths, hyperparameters, Ollama URL. Has some dead code (SD v1.5 defined first, then overwritten by SDXL). |
| `server.py` | 262 | FastAPI web server. 3 endpoints. Orchestrates the full pipeline. Lazy-loads models. Returns JSON with base64 images. |
| `pipeline/structured_analyzer.py` | ~510 | **The brain.** Two-model LLaVA→Mistral pipeline. Calls Ollama API. Handles JSON parsing, truncation repair, retries, fallbacks. |
| `pipeline/stage4_generation/lora_generator.py` | 160 | Loads SDXL (or SD v1.5 fallback). Singleton pattern. CPU offloading for 8GB VRAM. Generates 768×768 images. |
| `pipeline/stage4_generation/prompt_builder.py` | ~560 | Builds detailed medical image prompts. 8 modalities × 12 diseases × 6 quality modifiers. Maps organs → modalities. |
| `pipeline/stage5_explainability/gradcam.py` | 62 | DenseNet121 + Grad-CAM. Generates attention heatmaps. JET colormap overlay. |
| `pipeline/stage5_explainability/shap_explainer.py` | 66 | Creates SHAP waterfall bar charts. Red = positive, Blue = negative contribution. |
| `pipeline/stage6_local_llm/llava_explainer.py` | 157 | Sends generated image + context back to LLaVA for a 5-section clinical report. Has deterministic fallback if Ollama is down. |

### Frontend Files

| File | Lines | What It Does |
|------|-------|--------------|
| `App.tsx` | 120 | Root component. State machine (idle→ocr→generation→xai→done). Health checks every 15s. |
| `api.ts` | 49 | HTTP wrappers for `/api/analyze` and `/api/health`. Uses `fetch` + `FormData`. |
| `types.ts` | 66 | TypeScript interfaces for `AnalysisResult`, `PipelineStep`, etc. |
| `Header.tsx` | 43 | Logo + GPU name + online/offline status. |
| `UploadPanel.tsx` | ~175 | Drag-drop image upload + patient info form + clinical notes. |
| `ProgressBar.tsx` | 43 | 3-step visual pipeline progress indicator. |
| `LoadingOverlay.tsx` | 32 | Full-screen spinner during processing. |
| `ResultsDashboard.tsx` | ~512 | **The biggest component.** Shows everything: predictions, images (toggle between 3), SHAP table, explanation, audit log, timings. |
| `index.css` | ~940 | Complete dark-theme CSS. No CSS framework — hand-written. |
| `vite.config.ts` | 14 | Vite dev server + API proxy to port 8000. |
| `package.json` | 24 | Dependencies: React 19, Vite 7.3, TS 5.9, lucide, framer-motion, react-dropzone. |

### Legacy Files (Old Pipeline — No Longer Active)

| File | What It Was |
|------|-------------|
| `pipeline/full_pipeline.py` | Old 6-stage orchestrator: OCR → NER → Scoring → Generation → Grad-CAM → LLaVA |
| `app.py` | Old Gradio UI (replaced by React) |
| `pipeline/stage1_ocr/crnn_model.py` | CRNN (CNN+BiLSTM+CTC) for reading handwritten prescriptions |
| `pipeline/stage2_ner/ner_extractor.py` | ClinicalBERT NER for extracting symptoms, diseases, medications |
| `pipeline/stage3_scoring/hypothesis_scorer.py` | Deterministic disease scoring formula: S(d) = Σ[w_imp × w_rec × A(e,d)] |
| `pipeline/stage3_scoring/symptom_disease_db.py` | Hand-curated symptom → disease association database |

---

## Every Model Explained

### Models Currently Used in the Active Pipeline

| # | Model | Type | Size | Where It Runs | What It Does | Why This Model |
|---|-------|------|------|---------------|--------------|----------------|
| 1 | **LLaVA** (llava:latest) | Vision+Language LLM | 4.4 GB | Ollama (localhost:11434) | Reads medical prescriptions/reports and extracts clinical information in natural language | Only locally-runnable multimodal model that fits in 8 GB VRAM alongside other models |
| 2 | **Mistral** (mistral:latest) | Text-only LLM (7B) | 4.1 GB | Ollama (localhost:11434) | Structures LLaVA's description into clean JSON with predictions, confidence scores, SHAP values | Fast, accurate JSON mode, smallest model that reliably follows formatting instructions |
| 3 | **SDXL** (stable-diffusion-xl-base-1.0) | Diffusion image generator | ~6.5 GB | PyTorch + CUDA (GPU) | Generates realistic synthetic medical images from text prompts | Best open-source image gen model; 768×768 resolution; good anatomical detail |
| 4 | **SD v1.5** (stable-diffusion-v1-5) | Diffusion image generator (fallback) | ~4 GB | PyTorch + CUDA (GPU) | Fallback generator if SDXL not downloaded | Smaller, well-established, runs on any GPU |
| 5 | **DenseNet121** | CNN classifier | ~30 MB | PyTorch (GPU/CPU) | Provides feature maps for Grad-CAM heatmap generation | Compact, dense connections, widely used in medical imaging (CheXNet architecture) |

### Models From Legacy Pipeline (Code Retained, Not Active)

| # | Model | What It Did | Why Replaced |
|---|-------|-------------|--------------|
| 6 | **CRNN** (custom trained) | Read handwritten medical prescriptions using CNN (ResNet18) + BiLSTM + CTC decoding | LLaVA does this better — it can read printed AND handwritten text in prescriptions/reports |
| 7 | **ClinicalBERT** (samrawal/bert-base-uncased_clinical-ner) | Extracted medical entities (symptoms, diseases, medications) from OCR text using Named Entity Recognition | Mistral does this in one shot while also scoring confidence — no need for a separate NER model |

### Models Used for Training (MP01 Notebook Collection)

| # | Notebook | What Was Trained | Dataset Used |
|---|----------|------------------|--------------|
| 8 | `BrainCT_Model.ipynb` | Diffusion model on brain CT scans | Computed Tomography of the Brain (~3,367 images) |
| 9 | `BreastCancer_Model.ipynb` | Diffusion model on breast cancer imaging | Breast Cancer (~30,836 images) |
| 10 | `ChestCT_LungCancer_Model.ipynb` | Diffusion model on chest CT lung cancer | Chest CT-Scan images (~4,500 images) |
| 11 | `DentalOPG_Xray_Model.ipynb` | Diffusion model on dental X-rays | Dental OPG X-ray (~300+ images) |
| 12 | `DRR_Bones_Model.ipynb` | Diffusion model on bone radiographs | DRR Bones (~350+ images) |
| 13 | `IQ_OTH_NCCD_LungCancer_Model.ipynb` | Diffusion model on lung cancer CT | IQ-OTH/NCCD Lung Cancer (~14,261 images) |
| 14 | `Kaggle_BrainTumor_Training.ipynb` | Diffusion model on brain tumor MRI | Brain Tumor MRI (7,023 images) |
| 15 | `BraTS_Preprocessing.ipynb` | Preprocessing pipeline for BraTS 2020 | BraTS 2020 (369 patients, multi-modal MRI) |
| 16 | `BrainCT_Conditioned_Model.ipynb` | Conditioned diffusion with class labels | Brain CT with conditioning |

---

## Every Technology Explained

| Technology | What It Is | Why We Use It |
|------------|-----------|---------------|
| **Python 3.13** | Programming language | Backend, AI models, data processing — the industry standard for AI/ML |
| **PyTorch 2.6** | Deep learning framework | Runs SDXL, DenseNet121; supports CUDA GPU acceleration; most popular for research |
| **CUDA 12.4** | NVIDIA GPU computing toolkit | Lets PyTorch use the GPU for 10-100× faster inference than CPU |
| **FastAPI** | Python web framework | Serves the backend API; async, fast, auto-generates docs at `/docs` |
| **Uvicorn** | ASGI server | Runs FastAPI; handles HTTP requests efficiently |
| **Ollama** | Local LLM server | Serves LLaVA + Mistral locally; no cloud, no API keys, no cost |
| **Diffusers** (HuggingFace) | Diffusion model library | Loads and runs SDXL/SD v1.5 pipelines with easy-to-use Python API |
| **Transformers** (HuggingFace) | NLP/Vision model library | Provides DenseNet121 and other pretrained models |
| **PEFT** (HuggingFace) | LoRA training library | Used for fine-tuning SDXL with Low-Rank Adaptation on medical images |
| **grad-cam** (python package) | Grad-CAM implementation | Generates class activation maps from CNN models |
| **SHAP** (python package) | Explainability library | Conceptual basis for our feature attribution values |
| **OpenCV** | Computer vision library | Image preprocessing, colormap overlays for Grad-CAM |
| **Pillow (PIL)** | Image manipulation | Loading, resizing, converting images; base64 encoding |
| **Matplotlib** | Plotting library | Creates SHAP waterfall charts |
| **React 19** | JavaScript UI framework | Builds the doctor-facing web interface with components |
| **TypeScript 5.9** | Typed JavaScript | Catches frontend bugs at compile time; better IDE support |
| **Vite 7.3** | Frontend build tool | Lightning-fast dev server with HMR; proxies API calls to backend |
| **lucide-react** | Icon library | Clean, medical-appropriate icons for the UI |
| **framer-motion** | Animation library | Smooth transitions and loading animations |
| **react-dropzone** | File upload component | Drag-and-drop medical prescription/report upload |
| **Node.js 22** | JavaScript runtime | Runs the Vite dev server and frontend tooling |

---

## LoRA Training Explained

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a technique to fine-tune large AI models without retraining all their parameters.

Normally, SDXL has ~3.5 billion parameters. Fine-tuning all of them would need >24 GB VRAM and days of training. LoRA instead adds tiny "adapter" matrices (rank 8 in our case) to specific layers, only training ~0.1% of parameters.

### Our LoRA Setup

| Setting | Value | Meaning |
|---------|-------|---------|
| `LORA_RANK` | 8 | Size of the low-rank matrices (lower = less memory, higher = more capacity) |
| `LORA_ALPHA` | 8 | Scaling factor (alpha/rank = 1.0, meaning full adaptation strength) |
| `LORA_TRAIN_STEPS` | 5000 | Total training iterations |
| `LORA_LR` | 2e-4 | Learning rate (0.0002) |
| `LORA_BATCH_SIZE` | 4 | Images processed per training step |
| `LORA_IMG_SIZE` | 512 | Training image resolution |
| `IMAGES_PER_DATASET` | 50 | We use 50 images per medical domain |

### Training Domains (7 domains × 50 images = 350 total)

| Domain | Dataset Source | # Images Used |
|--------|---------------|---------------|
| Brain MRI | Brain Tumor MRI | 50 |
| Chest CT | Chest CT-Scan Lung Cancer | 50 |
| Breast Cancer | Breast Cancer Dataset | 50 |
| Dental OPG | Dental OPG X-ray | 50 |
| DRR Bones | DRR Bones | 50 |
| Lung Cancer | IQ-OTH/NCCD | 50 |
| Kidney | Kidney CT Dataset | 50 |

The trained LoRA weights are saved as `pytorch_lora_weights.safetensors` and loaded at inference time.

---

## All 16 Datasets

| # | Dataset | # Images | Modality | Used For |
|---|---------|----------|----------|----------|
| 1 | Computed Tomography (CT) of the Brain | ~3,367 | CT | LoRA training + evaluation |
| 2 | Breast Cancer | ~30,836 | Histopathology/CT | LoRA training + evaluation |
| 3 | Chest CT-Scan Lung Cancer | ~4,500 | CT | LoRA training + evaluation |
| 4 | Dental OPG X-ray | ~300+ | X-ray (panoramic) | LoRA training |
| 5 | DRR Bones | ~350+ | DRR (X-ray) | LoRA training |
| 6 | IQ-OTH/NCCD Lung Cancer | ~14,261 | CT | LoRA training + evaluation |
| 7 | Kidney CT (Cyst/Normal/Stone/Tumor) | ~229,957 | CT | LoRA training + classification |
| 8 | Brain Tumor MRI | 7,023 | MRI | LoRA training + evaluation |
| 9 | Chest X-ray Pneumonia | 5,863 | X-ray | Evaluation / testing |
| 10 | NIH Chest X-Ray | 300 (subset) | X-ray | Evaluation / testing |
| 11 | Skin Cancer HAM10000 | ~10,015 | Dermoscopy | Evaluation |
| 12 | Diabetic Retinopathy | ~35,126 | Fundus | Evaluation |
| 13 | Doctor's Handwritten Prescription (BD) | varies | Handwriting | OCR training (legacy CRNN) |
| 14 | IRMA X-Ray | varies | Mixed X-ray | Classification reference |
| 15 | BraTS 2020 | 369 patients | MRI (multi-modal) | Brain tumor segmentation |
| 16 | Kaggle Alternative Datasets | varies | Mixed | Supplementary training |

---

## VRAM Management (How to Run SDXL on 8 GB)

This is one of the most important engineering decisions in the project. SDXL normally needs ~12 GB VRAM. Here's how we make it work on 8 GB:

| Technique | What It Does | VRAM Saved |
|-----------|-------------|------------|
| `enable_model_cpu_offload()` | Keeps model weights on CPU RAM; only moves the currently active layer to GPU | ~4 GB |
| `enable_vae_slicing()` | Decodes the generated image in slices instead of all at once | ~1 GB |
| `torch.float16` | Uses half-precision floating point instead of float32 | ~50% of model size |
| **15 inference steps** (instead of 50) | Fewer steps = less intermediate VRAM | Moderate |
| **Singleton pattern** | Loads model once, reuses; no duplicate loads | Prevents OOM |
| **Sequential pipeline** | Only one model on GPU at a time (Ollama → SDXL → DenseNet121) | Total control |

---

## How to Run the Project

### Prerequisites
- NVIDIA GPU with ≥8 GB VRAM
- CUDA 12.x installed
- Python 3.10+
- Node.js 18+
- Ollama installed and running

### Step-by-step

```bash
# 1. Start Ollama and pull models (one-time)
ollama serve
ollama pull llava
ollama pull mistral

# 2. Start the backend
cd medvisx
pip install -r requirements.txt
python server.py
# → FastAPI running at http://localhost:8000

# 3. Start the frontend (in a new terminal)
cd medvisx/frontend
npm install
npm run dev
# → React app at http://localhost:5173

# 4. Open browser → http://localhost:5173
# Upload a medical prescription/report → Fill patient info → Click "Analyze & Generate"
```

---

## Old vs New Pipeline — Why We Changed

### Old Pipeline (6 stages, Gradio UI)

```
Prescription Image
    ↓
[Stage 1] CRNN OCR → reads handwritten text
    ↓
[Stage 2] ClinicalBERT NER → extracts entities (symptoms, diseases, medications)
    ↓
[Stage 3] Hypothesis Scorer → scores diseases using formula: S(d) = Σ[w_imp × w_rec × A(e,d)]
    ↓
[Stage 4] SD v1.5 → generates 512×512 medical image
    ↓
[Stage 5] Grad-CAM (DenseNet121) → heatmap
    ↓
[Stage 6] LLaVA (via HuggingFace + bitsandbytes) → explanation
    ↓
Gradio Dashboard
```

**Problems with the old pipeline:**
1. CRNN only worked on certain handwritten prescription styles — failed on printed reports or messy handwriting
2. ClinicalBERT NER needed clean OCR text — garbage in → garbage out
3. Hypothesis Scorer was deterministic (hand-coded weights) — not AI-based
4. SD v1.5 at 512×512 was low resolution
5. LLaVA via HuggingFace + bitsandbytes was SLOW and used tons of VRAM
6. Gradio UI was limited — no dark theme, no patient forms, no audit log

### New Pipeline (3 steps, React + FastAPI)

```
Medical Prescription / Report Image + Clinical Notes
    ↓
[Step 1] LLaVA → reads prescription/report, Mistral → structures into JSON (via Ollama)
    ↓
[Step 2] SDXL → generates 768×768 reference image
    ↓
[Step 3] Grad-CAM + SHAP + LLaVA → explains everything
    ↓
React Dashboard (dark theme, audit log, 3 tone modes, download)
```

**Why the new pipeline is better:**
1. Works on ANY medical prescription or report (handwritten or printed) — much more robust than CRNN OCR
2. LLaVA reads the prescription/report directly — no separate OCR step needed
3. Mistral replaces ClinicalBERT NER + Hypothesis Scorer — one model does both
4. SDXL at 768×768 is much higher quality
5. Ollama is 10× faster than HuggingFace + bitsandbytes for LLaVA
6. React frontend is professional, responsive, dark-themed, with full audit trail

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Two models (LLaVA + Mistral) instead of one** | LLaVA reads prescriptions/reports but can't output reliable JSON. Mistral outputs perfect JSON but can't read documents. Together they cover both needs. |
| **Ollama instead of HuggingFace** | 10× faster inference. No bitsandbytes quantization needed. Native JSON mode. Simple HTTP API. |
| **SDXL instead of SD v1.5** | 768×768 vs 512×512. Better anatomical detail. Better prompt following. |
| **CPU offloading instead of a bigger GPU** | Makes the project runnable on a laptop with 8 GB VRAM — no expensive cloud GPUs needed. |
| **React instead of Gradio** | Professional UI. Dark theme. Patient forms. Audit log. Tone mode switching. Image download. Gradio couldn't do all this. |
| **FastAPI instead of Flask** | Async support. Auto-generates API docs. Type validation. Better performance. |
| **DenseNet121 for Grad-CAM** | Compact (~8M params). Dense connections preserve gradients. CheXNet (famous medical AI) uses the same architecture. |
| **Singleton model loading** | Prevents loading 6.5 GB SDXL model multiple times. Load once, reuse forever. |
| **Randomized prompts** | Each generation uses `random.choice()` from template variants — produces diverse but medically accurate images. |
| **Base64 images in JSON** | Avoids file system I/O. Everything in one API response. Frontend displays directly from base64. |
| **Retry with increasing temperature** | If Mistral fails at temperature 0.1, try 0.2, then 0.3. Higher temperature = more creative = more likely to produce parseable output. |
| **Deterministic fallback reports** | If Ollama is completely down, `llava_explainer.py` has a template-based fallback that generates a report from the data it already has. The system never crashes. |

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| SDXL resolution | 768 × 768 pixels |
| SDXL inference steps | 15 |
| SDXL guidance scale | 7.5 |
| SD v1.5 resolution (fallback) | 512 × 512 pixels |
| SD v1.5 inference steps (fallback) | 40 |
| LLaVA model size | 4.4 GB |
| Mistral model size | 4.1 GB |
| SDXL model size | ~6.5 GB |
| DenseNet121 parameters | ~8 million |
| Grad-CAM target layer | denseblock4.denselayer16.conv2 |
| Grad-CAM input size | 224 × 224 pixels |
| Modalities supported | 8 |
| Diseases supported | 12 |
| LoRA rank | 8 |
| LoRA training steps | 5,000 |
| LoRA images per domain | 50 |
| LoRA training domains | 7 |
| Total LoRA training images | 350 |
| Total datasets | 16 |
| Frontend port | 5173 |
| Backend port | 8000 |
| Ollama port | 11434 |
| Ollama retry attempts | 3 |
| Health check interval | 15 seconds |
| VRAM required | 8 GB minimum |
| GPU used | NVIDIA RTX 4060 Laptop |
| CUDA version | 12.4 |

---

## Glossary

| Term | Meaning |
|------|---------|
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — shows which image regions influenced the AI's decision |
| **SHAP** | SHapley Additive exPlanations — shows which features contributed to the prediction |
| **LoRA** | Low-Rank Adaptation — fine-tune large models by training only tiny adapter matrices |
| **SDXL** | Stable Diffusion XL — latest text-to-image generation model |
| **LLaVA** | Large Language and Vision Assistant — multimodal model that sees images + understands text |
| **Mistral** | 7B parameter text model — fast, accurate, good at structured output |
| **Ollama** | Local LLM server — runs AI models on your machine, no cloud needed |
| **FastAPI** | Python web framework for building APIs |
| **DenseNet121** | Dense Convolutional Network with 121 layers — each layer is connected to every other layer |
| **CTC** | Connectionist Temporal Classification — loss function for sequence recognition (used in CRNN) |
| **BiLSTM** | Bidirectional Long Short-Term Memory — reads sequences in both directions |
| **NER** | Named Entity Recognition — extracts structured entities from text |
| **ClinicalBERT** | BERT model fine-tuned on clinical text for medical NER |
| **CRNN** | Convolutional Recurrent Neural Network — combines CNN (for features) + RNN (for sequences) |
| **XAI** | Explainable AI — making AI decisions interpretable to humans |
| **VRAM** | Video RAM — GPU memory used for AI model inference |
| **CPU offloading** | Keeping model weights in CPU RAM, moving only active layers to GPU during inference |
| **VAE slicing** | Processing the image decoder in smaller slices to reduce peak VRAM |
| **Inference steps** | How many times the diffusion model refines the image (more = better but slower) |
| **Guidance scale** | How closely the generated image follows the text prompt (higher = more literal) |
| **Singleton** | Design pattern — only one instance of a class exists, shared across all requests |
| **HMR** | Hot Module Replacement — Vite updates the browser instantly when you save a file |
| **CORS** | Cross-Origin Resource Sharing — browser security policy; Vite proxy avoids this issue |
| **Base64** | Text encoding for binary data — lets us embed images directly in JSON |

---

*Last updated for 2nd Review — March 2026*
