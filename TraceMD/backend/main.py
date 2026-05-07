"""
TraceMD — FastAPI Server
POST /api/analyze → Streams pipeline progress via SSE.
GET  /health      → Health check.

CRITICAL RULES:
1. Never load two large models in VRAM simultaneously
2. Every stage prints start/progress/completion
3. Every error returns meaningful message — never silent 500
"""

import sys
import os
import uuid
import json
import re
import traceback
from datetime import datetime
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from PIL import Image
import pypdfium2 as pdfium

from pipeline.ollama_ocr import run_ollama_ocr
from pipeline.ner        import run_ner
from pipeline.scorer     import run_scorer
from pipeline.generator  import run_generator
from pipeline.gradcam    import run_gradcam
from pipeline.explainer  import run_explainer

app = FastAPI(title="TraceMD API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse_event(stage: int, status: str, label: str = "",
               result: str = "", payload: dict = None) -> dict:
    data = {"stage": stage, "status": status, "label": label, "result": result}
    if payload:
        data["payload"] = payload
    return {"event": "message", "data": json.dumps(data, default=str)}


def _load_upload_as_image(contents: bytes, is_pdf: bool) -> Image.Image:
    if is_pdf:
        try:
            pdf   = pdfium.PdfDocument(contents)
            if len(pdf) < 1:
                raise RuntimeError("PDF has no pages")
            page  = pdf[0]
            image = page.render(scale=2).to_pil().convert("RGB")
            page.close()
            pdf.close()
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")

    try:
        return Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")


def _is_low_information_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) < 20:
        return True
    tokens       = re.findall(r"[A-Za-z]{2,}", cleaned.lower())
    unique_tokens = len(set(tokens))
    return unique_tokens < 3


@app.get("/health")
def health():
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    return {
        "status": "ok",
        "cuda":   torch.cuda.is_available(),
        "gpu":    gpu_name,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    filename     = file.filename or "unknown"
    content_type = (file.content_type or "").lower()
    lowered_name = filename.lower()

    is_pdf   = content_type == "application/pdf" or lowered_name.endswith(".pdf")
    is_image = content_type.startswith("image/") or lowered_name.endswith((".jpg", ".jpeg", ".png"))

    if not is_image and not is_pdf:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Only images (JPG, PNG) and PDFs are supported.",
        )

    contents  = await file.read()
    image     = _load_upload_as_image(contents, is_pdf)
    report_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    async def stream():
        audit_log = []

        def log(action: str):
            audit_log.append({"action": action, "by": "System", "timestamp": datetime.now().isoformat()})

        try:
            # ── STAGE 1: OCR ──────────────────────────────────
            print(f"\n{'='*50}")
            print(f"[Stage 1] OCR — Ollama gemma4:e4b — {filename}")
            print(f"{'='*50}")

            yield _sse_event(1, "running", "Extracting text via Gemma 4...")
            ocr_text = run_ollama_ocr(image)

            log("Document OCR complete via Gemma 4 (Ollama)")
            yield _sse_event(
                1, "complete",
                result=f"{len(ocr_text)} characters extracted",
                payload={"input_type": "gemma4"},
            )

            # ── STAGE 2: NER ──────────────────────────────────
            print(f"\n{'='*50}")
            print(f"[Stage 2] Named Entity Recognition — Ollama medgemma:4b")
            print(f"{'='*50}")

            yield _sse_event(2, "running", "Extracting clinical entities via MedGemma...")
            ner_result = run_ner(ocr_text, image)
            log("Entity extraction complete")
            yield _sse_event(
                2, "complete",
                result=f"Organ: {ner_result.get('organ', '?')} — "
                       f"{len(ner_result.get('conditions', []))} condition(s)",
            )

            # ── STAGE 3: SCORING ──────────────────────────────
            print(f"\n{'='*50}")
            print(f"[Stage 3] Hypothesis Scoring")
            print(f"{'='*50}")

            yield _sse_event(3, "running", "Scoring hypotheses with SHAP...")
            scored_result = run_scorer(ner_result)
            top_conds     = scored_result.get("conditions", [])
            result_str    = " | ".join(f"{c['label']}: {c['confidence']}" for c in top_conds[:2])
            log("Hypothesis scoring complete")
            yield _sse_event(3, "complete", result=result_str)

            # ── STAGE 4: IMAGE GENERATION ────────────────────
            print(f"\n{'='*50}")
            print(f"[Stage 4] Synthetic Image Generation — Stable Diffusion v1.5")
            print(f"{'='*50}")

            yield _sse_event(4, "running", "Generating synthetic medical image...")
            seed      = int(report_id[:4], 16) % 10000
            image_b64 = run_generator(scored_result, seed=seed)
            log("Synthetic image generated")
            yield _sse_event(
                4, "complete",
                result=f"Image generated for: {scored_result.get('organ', 'organ')}",
            )

            # ── STAGE 5: GRAD-CAM ────────────────────────────
            print(f"\n{'='*50}")
            print(f"[Stage 5] Grad-CAM Explainability — DenseNet121")
            print(f"{'='*50}")

            yield _sse_event(5, "running", "Computing Grad-CAM saliency and mask on original image...")
            gradcam_result = run_gradcam(image)
            log("Grad-CAM complete")
            yield _sse_event(5, "complete", result="Saliency map and binary mask generated")

            # ── STAGE 6: NL EXPLANATION ──────────────────────
            print(f"\n{'='*50}")
            print(f"[Stage 6] Clinical Explanation — Ollama medgemma:4b")
            print(f"{'='*50}")

            yield _sse_event(6, "running", "Generating clinical explanation via MedGemma...")

            report = {
                "id":        report_id,
                "fileName":  filename,
                "createdAt": timestamp,
                "inputType": "gemma4",
                "ocrText":   ocr_text,
                "fields": {
                    "pseudonym": scored_result.get("pseudonym", "Unknown"),
                    "age":       scored_result.get("age", 0),
                    "sex":       scored_result.get("sex", "Unknown"),
                    "symptoms":  scored_result.get("symptoms", []),
                },
                "predictions": {
                    "organ":              scored_result.get("organ", ""),
                    "overall_confidence": scored_result.get("overall_confidence", 0.0),
                    "conditions":         scored_result.get("conditions", []),
                    "localization":       scored_result.get("localization", {}),
                },
                "xai": {
                    "text_features":          scored_result.get("xai", {}).get("text_features", []),
                    "shap_table":             scored_result.get("xai", {}).get("shap_table", []),
                    "saliencyImageB64":       gradcam_result["saliency_b64"],
                    "maskImageB64":           gradcam_result["mask_b64"],
                    "naturalLanguageSummary": "",
                    "provenance": {
                        "ocr_model":   "gemma4:e4b (Ollama)",
                        "ner_model":   "medgemma:4b (Ollama)",
                        "generator":   "Stable Diffusion v1.5",
                        "xai_model":   "DenseNet121 Grad-CAM",
                        "explain_model": "medgemma:4b (Ollama)",
                    },
                },
                "generatedImageB64": image_b64,
                "auditLog":          audit_log,
                "status":            "predicted",
            }

            nl_summary = run_explainer(report)
            report["xai"]["naturalLanguageSummary"] = nl_summary
            report["auditLog"].append({
                "action":    "Analysis complete",
                "by":        "System",
                "timestamp": datetime.now().isoformat(),
            })

            log("Natural language explanation complete")
            yield _sse_event(
                6, "complete",
                result="Pipeline complete",
                payload=report,
            )

            print(f"\n{'='*50}")
            print(f"[TraceMD] Pipeline complete for {filename}")
            print(f"{'='*50}\n")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[TraceMD ERROR] {e}\n{tb}")
            yield _sse_event(-1, "error", label="Pipeline failed", result=str(e))

    return EventSourceResponse(stream())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
