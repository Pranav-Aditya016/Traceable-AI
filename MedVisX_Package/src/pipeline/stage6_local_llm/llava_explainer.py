"""
Stage 6: Natural Language Clinical Explanation via Ollama (100% Local)

Uses Ollama's local API with the `llava` model for multimodal explanation.
- Receives generated image + clinical context
- Returns a structured clinical explanation for the clinician
- No API keys, no internet, no cloud services needed at inference time

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull the llava model: ollama pull llava
    3. Ollama server runs at http://localhost:11434 by default
"""
import os
import sys
import json
import base64
import requests
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS, OLLAMA_TIMEOUT


class OllamaExplainer:
    """
    Local multimodal explainer using Ollama with llava model.
    Runs 100% locally — no API keys needed.
    """

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.max_tokens = OLLAMA_MAX_TOKENS
        self.timeout = OLLAMA_TIMEOUT

        # Check if Ollama is running
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                model_base = self.model.split(":")[0]
                has_model = any(model_base in m for m in models)
                if has_model:
                    print(f"[Ollama] Connected - model '{self.model}' available")
                else:
                    print(f"[Ollama] Connected but '{self.model}' not found.")
                    print(f"[Ollama] Available models: {models}")
                    print(f"[Ollama] Run: ollama pull {self.model}")
                self._available = True
            else:
                print("[Ollama] WARNING: Server responded with error")
                self._available = False
        except requests.ConnectionError:
            print("[Ollama] WARNING: Cannot connect to Ollama server")
            print("[Ollama] Make sure Ollama is running: https://ollama.com")
            self._available = False

    def _image_to_base64(self, pil_image: Image.Image) -> str:
        """Convert PIL Image to base64 string for Ollama API."""
        buffered = BytesIO()
        # Resize to reduce payload (Ollama llava expects reasonable sizes)
        img = pil_image.convert("RGB").resize((512, 512))
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def explain(self, generated_image: Image.Image,
                clinical_context: dict) -> str:
        """
        Generate a structured clinical explanation.

        Args:
            generated_image: PIL Image of the generated medical scan
            clinical_context: dict with keys:
                - ocr_text: str
                - entities: dict
                - top_disease: str
                - top_score: float
                - shap_values: dict
                - prompt: str

        Returns:
            Structured explanation string for display to clinician
        """
        disease   = clinical_context.get("top_disease", "unknown")
        score     = clinical_context.get("top_score", 0.0)
        ocr_text  = clinical_context.get("ocr_text", "")
        entities  = clinical_context.get("entities", {})
        shap_vals = clinical_context.get("shap_values", {})

        # Format entity summary
        symptoms    = [e["text"] for e in entities.get("symptoms", [])]
        medications = [e["text"] for e in entities.get("medications", [])]

        prompt_text = f"""You are an expert clinical AI assistant helping a physician understand a decision support system's output.

**Clinical Input (OCR-extracted):** {ocr_text}

**Identified Symptoms:** {', '.join(symptoms) if symptoms else 'None detected'}
**Identified Medications:** {', '.join(medications) if medications else 'None detected'}
**Top Hypothesis:** {disease} (confidence: {score:.0%})
**Key Contributing Factors:** {', '.join([f'{k} (+{v:.3f})' for k, v in list(shap_vals.items())[:5]])}

**Generated Medical Image:** The attached image was synthetically generated to visually represent the top hypothesis.

Please provide a structured clinical explanation with these sections:
1. **Clinical Summary** (2-3 sentences): What the evidence suggests
2. **Image Interpretation** (2-3 sentences): What to look for in the generated image and what the highlighted regions indicate
3. **Key Evidence** (bullet list): The most clinically significant findings
4. **Differential Considerations** (1-2 sentences): What else to rule out
5. **Recommended Next Steps** (bullet list): Suggested diagnostic actions

IMPORTANT: Always end with — "This is a decision-support tool only. All findings must be verified by a qualified clinician."
"""

        # Try Ollama API
        if self._available:
            try:
                return self._call_ollama(prompt_text, generated_image)
            except Exception as e:
                print(f"[Ollama] API call failed: {e}")

        # Fallback: generate a deterministic report
        return self._fallback_report(disease, score, symptoms, medications, shap_vals)

    def _call_ollama(self, prompt: str, image: Image.Image) -> str:
        """Call Ollama API with multimodal input."""
        img_b64 = self._image_to_base64(image)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.3,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        return result.get("response", "No response generated.")

    def _fallback_report(self, disease: str, score: float,
                          symptoms: list, medications: list,
                          shap_vals: dict) -> str:
        """Generate a deterministic report when Ollama is unavailable."""
        symptom_str = ", ".join(symptoms) if symptoms else "not specified"
        med_str = ", ".join(medications) if medications else "not specified"
        shap_str = "\n".join(
            [f"  - **{k}**: +{v:.3f}" for k, v in list(shap_vals.items())[:5]]
        ) if shap_vals else "  - No attribution data available"

        return f"""## Clinical Analysis Report

### 1. Clinical Summary
Based on the extracted clinical entities, the system's top hypothesis is **{disease.title()}** with a confidence score of **{score:.0%}**. The identified symptoms ({symptom_str}) are consistent with this diagnostic hypothesis.

### 2. Image Interpretation
The generated medical image represents a synthetic visualization of {disease}. The Grad-CAM heatmap overlay highlights regions of highest diagnostic relevance — warmer colors (red/yellow) indicate areas the model considers most significant for this condition.

### 3. Key Evidence
{shap_str}

### 4. Differential Considerations
Other conditions with overlapping symptom profiles should be considered. A thorough clinical workup is recommended before finalizing the diagnosis.

### 5. Recommended Next Steps
- Correlate with patient history and physical examination
- Order confirmatory imaging/laboratory investigations
- Consider differential diagnoses with similar presentations
- Consult relevant specialists as appropriate

---
*This is a decision-support tool only. All findings must be verified by a qualified clinician.*
*Generated images are synthetic and NOT real patient data.*
"""
