"""
Stage 1 — OCR via Ollama gemma4:e4b (multimodal).
Handles handwritten and printed medical documents in a single call.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import re
import requests
from io import BytesIO
from PIL import Image

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OCR_MODEL   = "gemma4:e4b"
TIMEOUT     = 120


def run_ollama_ocr(image: Image.Image) -> str:
    image = image.convert("RGB")
    buf   = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "You are a specialized medical document OCR system. "
        "Extract ALL text visible in this image exactly as written.\n\n"
        "Capture every piece of text including:\n"
        "- Patient name, ID, age, sex, date of birth, date of visit\n"
        "- Chief complaint and history of presenting illness\n"
        "- Vital signs: BP, HR, SpO2, temperature, respiratory rate, weight\n"
        "- Symptoms and their duration, onset, severity\n"
        "- Physical examination findings\n"
        "- Laboratory values and units\n"
        "- Imaging findings and radiology reports\n"
        "- Diagnoses, differential diagnoses, ICD codes\n"
        "- Medications with doses, frequency, route\n"
        "- Allergies and past medical history\n\n"
        "Preserve all numbers, units, abbreviations, and punctuation exactly. "
        "For handwritten text, transcribe as accurately as possible. "
        "Return ONLY the extracted plain text — no commentary, no markdown, no explanations."
    )

    print(f"  [OllamaOCR] Sending image to {OCR_MODEL}...")
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OCR_MODEL, "prompt": prompt, "images": [b64], "stream": False},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama OCR timed out after {TIMEOUT}s")
    except Exception as e:
        raise RuntimeError(f"Ollama OCR failed: {e}")

    result = re.sub(r"\s+", " ", result).strip()
    print(f"  [OllamaOCR] Extracted {len(result)} characters")
    return result if result else "OCR extraction produced no readable text."
