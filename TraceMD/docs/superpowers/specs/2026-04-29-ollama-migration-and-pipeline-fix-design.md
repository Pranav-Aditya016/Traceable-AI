# TraceMD — Ollama Migration & Pipeline Fix

**Date:** 2026-04-29

## Summary

Migrate all LLM inference to Ollama (local HTTP API). Fix Stable Diffusion and Grad-CAM crashes.

## Stage Changes

| Stage | Before | After |
|-------|--------|-------|
| 1 OCR | TrOCR + PaliGemma 2 (HuggingFace) | `gemma4:e4b` via Ollama |
| 2 NER | MedGemma 4B (HuggingFace, in-memory) | `medgemma:4b` via Ollama |
| 3 Scoring | SHAP (unchanged) | SHAP (unchanged) |
| 4 Image Gen | SD v1.5 — crashes | SD v1.5 — fixed |
| 5 Grad-CAM | DenseNet121 — crashes | DenseNet121 — fixed |
| 6 XAI | MedGemma 4B (HuggingFace, reloaded) | `medgemma:4b` via Ollama |

## Ollama Integration

- Base URL: `http://localhost:11434` (env: `OLLAMA_BASE_URL`)
- OCR model: `gemma4:e4b` — multimodal, handles handwritten + printed in one call
- NER + XAI model: `medgemma:4b`
- All calls use `POST /api/generate` with `stream: false`
- Images sent as base64 PNG in `images` array

## Files Deleted

`pipeline/trocr.py`, `pipeline/paligemma.py`, `pipeline/ocr_router.py`

## Files Created

`pipeline/ollama_ocr.py`

## SD Fixes

1. Model ID: `runwayml/stable-diffusion-v1-5` → `stable-diffusion-v1-5/stable-diffusion-v1-5`
2. Add `requires_safety_checker=False` to `from_pretrained`
3. Improve prompts: add `grayscale, monochrome` prefix; strengthen negative prompt with color rejection terms

## Grad-CAM Fixes

1. Wrap forward pass in `with GradCAM(...) as cam_extractor:` context manager (torchcam 0.4.x)
2. Guard `torch.cuda.empty_cache()` with `if torch.cuda.is_available()`
