"""
Structured Medical Document Analyzer - Two-Model Approach (v3)
===============================================================
Mirrors the Gemini prototype analysis locally using Ollama:

  1. LLaVA (multimodal) - Describe the medical image thoroughly
  2. Mistral (text-only) - Structure the description into strict JSON
     using Ollama native JSON mode ("format": "json")

Key improvements over v2:
  - Uses Ollama JSON mode for guaranteed valid JSON output
  - Medical-ONLY analysis (no email, phone, address, etc.)
  - Better vision prompt for varied per-image descriptions
  - Adds model field to conditions (matches reference)
  - Adds localization with bbox (matches reference)
  - Adds naturalLanguageSummary in XAI

100% local - no API keys, no cloud services.
"""
import json
import base64
import random
import requests
from io import BytesIO
from PIL import Image

# -- Ollama settings --
OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "llava"
TEXT_MODEL = "mistral"
OLLAMA_TIMEOUT = 300


# -- STEP 1 PROMPT: LLaVA describes the medical image --
VISION_DESCRIBE_PROMPT = """You are a specialist radiologist analyzing a medical document or scan.

Look at this medical image very carefully and provide a DETAILED analysis:

1. **Document Type**: What kind of medical document is this? (X-ray, CT scan, MRI, lab report, prescription, discharge summary, pathology report, ultrasound, ECG, etc.)

2. **Text Content**: Read and transcribe ALL visible text in the image, word by word. Include patient details, dates, values, findings, and any other text.

3. **Patient Demographics**: If visible - patient name/ID, age, sex/gender.

4. **Clinical Findings**: List every medical finding, symptom, diagnosis, abnormality, or condition mentioned or visible. Be specific about:
   - What organ or body system is affected
   - What abnormalities are present
   - Severity indicators
   - Measurements or values

5. **Anatomical Details**: Describe the anatomical structures visible. Note any abnormal areas - their location, size, and appearance.

6. **Overall Assessment**: What is the most likely primary diagnosis or finding?

IMPORTANT: Focus ONLY on medical/clinical information. Do NOT mention email addresses, phone numbers, website URLs, or other non-medical contact information. Those are irrelevant to clinical analysis.

Be thorough, specific, and detailed. Every medical image is unique - describe what makes THIS specific image different."""


# -- STEP 2 PROMPT: Mistral structures into strict JSON --
STRUCTURE_PROMPT_TEMPLATE = """You are a medical data extraction AI. Your task is to analyze the clinical information below and extract structured medical data.

=== CLINICAL INFORMATION ===
{text}
=== END CLINICAL INFORMATION ===

Patient info provided by user: Name="{name}", Age="{age}", Sex="{sex}"

INSTRUCTIONS:
1. Extract ONLY medical/clinical data. Ignore any email, phone, URL, or non-medical info.
2. Identify the primary organ system affected.
3. Predict the top 1-2 medical conditions with realistic confidence scores (0.0 to 1.0).
4. Identify the top 3 most important clinical features that drive the diagnosis.
5. Create a SHAP attribution table with medical features ONLY (symptoms, findings, demographics - NOT email/phone/URL).
6. For localization, estimate a bounding box [x1, y1, x2, y2] in a 512x512 pixel image where the primary finding would be located.
7. Write a short one-paragraph natural language summary for a clinician.

You MUST respond with valid JSON in exactly this format:
{{"ocrText": "all extracted clinical text here", "fields": {{"pseudonym": "patient name or Anonymous", "age": 0, "sex": "M or F or Unknown", "symptoms": ["symptom1", "symptom2"]}}, "predictions": {{"organ": "primary organ", "conditions": [{{"label": "primary condition", "confidence": 0.85, "model": "llava_mistral_v3"}}, {{"label": "differential diagnosis", "confidence": 0.45, "model": "llava_mistral_v3"}}], "localization": {{"mask": [{{"x": 200, "y": 180}}, {{"x": 350, "y": 180}}, {{"x": 350, "y": 330}}, {{"x": 200, "y": 330}}], "bbox": [200, 180, 350, 330]}}}}, "xai": {{"text_features": [{{"feature": "most important finding", "weight": 0.92}}, {{"feature": "second finding", "weight": 0.75}}, {{"feature": "third finding", "weight": 0.58}}], "shap_table": [{{"feature": "age", "value": "patient age", "shap": 0.12}}, {{"feature": "sex", "value": "patient sex", "shap": 0.05}}, {{"feature": "primary symptom", "value": "present/absent", "shap": 0.45}}, {{"feature": "key finding", "value": "description", "shap": 0.38}}, {{"feature": "secondary symptom", "value": "present/absent", "shap": 0.22}}], "naturalLanguageSummary": "A brief clinical summary paragraph for the physician."}}}}

CRITICAL RULES:
- Output ONLY the JSON object, nothing else.
- All confidence values must be between 0.0 and 1.0.
- The "model" field must always be "llava_mistral_v3".
- SHAP table must contain ONLY medical features (age, sex, symptoms, findings, measurements). NEVER include email, phone, URL, address, or non-medical fields.
- The bbox must be 4 integers [x1, y1, x2, y2] within a 512x512 image.
- The "symptoms" array must contain actual medical symptoms, not generic text.
- The "naturalLanguageSummary" should be a concise clinical summary in 2-3 sentences."""


# -- TEXT-ONLY PROMPT (no image involved) --
TEXT_ONLY_PROMPT_TEMPLATE = """You are a medical data extraction AI. Analyze this clinical text and extract structured medical data.

=== CLINICAL TEXT ===
{text}
=== END CLINICAL TEXT ===

Patient info: Name="{name}", Age="{age}", Sex="{sex}"

Extract ONLY medical/clinical information. Ignore any email, phone, URL, or non-medical data.

You MUST respond with valid JSON in exactly this format:
{{"ocrText": "the clinical text", "fields": {{"pseudonym": "patient name", "age": 0, "sex": "M or F or Unknown", "symptoms": ["symptom1", "symptom2"]}}, "predictions": {{"organ": "primary organ affected", "conditions": [{{"label": "primary condition", "confidence": 0.85, "model": "mistral_text_v3"}}, {{"label": "differential", "confidence": 0.4, "model": "mistral_text_v3"}}], "localization": {{"mask": [{{"x": 150, "y": 150}}, {{"x": 350, "y": 150}}, {{"x": 350, "y": 350}}, {{"x": 150, "y": 350}}], "bbox": [150, 150, 350, 350]}}}}, "xai": {{"text_features": [{{"feature": "key finding 1", "weight": 0.9}}, {{"feature": "key finding 2", "weight": 0.7}}, {{"feature": "key finding 3", "weight": 0.5}}], "shap_table": [{{"feature": "age", "value": "value", "shap": 0.1}}, {{"feature": "sex", "value": "value", "shap": 0.05}}, {{"feature": "primary symptom", "value": "present", "shap": 0.4}}, {{"feature": "key finding", "value": "value", "shap": 0.35}}, {{"feature": "secondary finding", "value": "value", "shap": 0.2}}], "naturalLanguageSummary": "Brief clinical summary for the physician."}}}}

CRITICAL: Output ONLY the JSON. No markdown, no explanation. SHAP table must contain ONLY medical features - never email, phone, or non-medical fields."""


def _image_to_base64(pil_image):
    """Convert PIL Image to base64 for Ollama API."""
    buf = BytesIO()
    img = pil_image.convert("RGB").resize((512, 512))
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_ollama(prompt, image=None, model=None, temperature=0.1,
                 max_tokens=2048, force_json=False):
    """
    Call Ollama API (with optional image for multimodal).
    force_json: use Ollama native JSON mode for guaranteed valid JSON.
    """
    if model is None:
        model = TEXT_MODEL

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    # Ollama native JSON mode - forces valid JSON output
    if force_json:
        payload["format"] = "json"

    if image is not None:
        payload["images"] = [_image_to_base64(image)]

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _parse_json_response(raw):
    """Extract JSON from LLM response."""
    text = raw.strip()

    # Remove markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Fix common LLM habits
    text = text.replace("\\_", "_")

    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try repairing truncated JSON
    if start >= 0:
        fragment = text[start:]
        repaired = _repair_truncated_json(fragment)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    return None


def _repair_truncated_json(text):
    """Attempt to repair truncated JSON by closing open brackets/braces."""
    in_string = False
    escape_next = False
    stack = []

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    if not stack:
        return text

    if in_string:
        text += '"'

    text = text.rstrip()
    if text.endswith(','):
        text = text[:-1]

    for opener in reversed(stack):
        text = text.rstrip().rstrip(',')
        if opener == '{':
            text += '}'
        elif opener == '[':
            text += ']'

    return text


def _make_default_result(text="", patient_name="Anonymous",
                         patient_age=0, patient_sex="Unknown"):
    """Return a valid result structure with defaults."""
    return {
        "ocrText": text,
        "fields": {
            "pseudonym": patient_name,
            "age": patient_age,
            "sex": patient_sex,
            "symptoms": [],
        },
        "predictions": {
            "organ": "unknown",
            "conditions": [{"label": "unknown", "confidence": 0.0, "model": "fallback"}],
            "localization": {
                "mask": [{"x": 150, "y": 150}, {"x": 350, "y": 150},
                         {"x": 350, "y": 350}, {"x": 150, "y": 350}],
                "bbox": [150, 150, 350, 350],
            },
        },
        "xai": {
            "text_features": [],
            "shap_table": [],
            "naturalLanguageSummary": "Unable to analyze the document. Please try with a clearer medical image.",
        },
    }


def _check_ollama():
    """Check which models are available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return {}
        models = {m["name"].split(":")[0]: m["name"]
                  for m in resp.json().get("models", [])}
        return models
    except Exception:
        return {}


def _sanitize_shap_table(shap_table):
    """Remove non-medical features from SHAP table."""
    NON_MEDICAL = [
        "email", "phone", "telephone", "fax", "url", "website", "http",
        "address", "zip", "postal", "street", "city", "state",
        "insurance", "billing", "payment", "policy",
        "mrn", "account", "ssn", "social security",
    ]
    filtered = []
    for entry in shap_table:
        feature = str(entry.get("feature", "")).lower()
        if not any(kw in feature for kw in NON_MEDICAL):
            filtered.append(entry)
    return filtered if filtered else shap_table


def analyze_medical_document(image=None, clinical_text="",
                              patient_name="Anonymous",
                              patient_age="",
                              patient_sex=""):
    """
    Two-model analysis: LLaVA (vision) + Mistral (structuring).

    Mirrors the Gemini prototype analysis step:
      Step A: LLaVA describes the medical image in natural language
      Step B: Mistral structures the description into strict JSON
              using Ollama native JSON mode for guaranteed valid output

    Returns structured dict matching the reference schema.
    """
    print("\n" + "=" * 60)
    print("[Analyzer] Starting medical document analysis (v3)...")
    print("=" * 60)

    available = _check_ollama()
    if not available:
        print("[Analyzer] ERROR: Ollama unavailable!")
        return _make_default_result(clinical_text, patient_name,
                                     int(patient_age) if patient_age.isdigit() else 0,
                                     patient_sex or "Unknown")

    has_vision = "llava" in available
    has_text = "mistral" in available
    print(f"[Analyzer] Available models: {list(available.keys())}")

    structuring_model = "mistral" if has_text else list(available.keys())[0]

    # -- STEP A: Image description with LLaVA --
    image_description = ""
    if image is not None and has_vision:
        print("[Analyzer] Step 1: LLaVA analyzing medical image...")
        try:
            raw_desc = _call_ollama(
                VISION_DESCRIBE_PROMPT,
                image=image,
                model="llava",
                temperature=0.4,
                max_tokens=2000,
            )
            image_description = raw_desc.strip()
            print(f"[Analyzer] Image description: {len(image_description)} chars")
            preview = image_description[:300].replace('\n', ' ')
            print(f"[Analyzer] Preview: {preview}...")
        except Exception as e:
            print(f"[Analyzer] LLaVA image description failed: {e}")

    # -- Combine all available text --
    all_text_parts = []
    if image_description:
        all_text_parts.append(f"[MEDICAL IMAGE ANALYSIS]\n{image_description}")
    if clinical_text and clinical_text.strip():
        all_text_parts.append(f"[CLINICAL NOTES]\n{clinical_text.strip()}")

    combined_text = "\n\n".join(all_text_parts)

    if not combined_text.strip():
        print("[Analyzer] No text available - returning defaults")
        return _make_default_result("", patient_name,
                                     int(patient_age) if patient_age.isdigit() else 0,
                                     patient_sex or "Unknown")

    # -- STEP B: Structure with Mistral (JSON mode) --
    print(f"[Analyzer] Step 2: Structuring with {structuring_model} (JSON mode)...")
    result = None

    if image_description:
        prompt = STRUCTURE_PROMPT_TEMPLATE.format(
            text=combined_text,
            name=patient_name or "Anonymous",
            age=patient_age or "unknown",
            sex=patient_sex or "Unknown",
        )
    else:
        prompt = TEXT_ONLY_PROMPT_TEMPLATE.format(
            text=combined_text,
            name=patient_name or "Anonymous",
            age=patient_age or "unknown",
            sex=patient_sex or "Unknown",
        )

    # Try with JSON mode (guaranteed valid JSON from Ollama)
    for attempt in range(3):
        try:
            raw_response = _call_ollama(
                prompt,
                model=structuring_model,
                temperature=0.1 + (attempt * 0.1),
                max_tokens=3000,
                force_json=True,
            )
            print(f"[Analyzer] {structuring_model} response: {len(raw_response)} chars (attempt {attempt + 1})")

            result = _parse_json_response(raw_response)
            if result and "fields" in result and "predictions" in result:
                print("[Analyzer] Structured extraction successful!")
                break
            elif result:
                print(f"[Analyzer] JSON parsed but missing required keys (attempt {attempt + 1})")
                if attempt == 2:
                    break
                result = None
            else:
                print(f"[Analyzer] JSON parse failed (attempt {attempt + 1})")
                if raw_response:
                    print(f"[Analyzer] Raw preview: {raw_response[:200]}...")
        except Exception as e:
            print(f"[Analyzer] {structuring_model} failed (attempt {attempt + 1}): {e}")

    # -- Fallback: try without JSON mode --
    if result is None:
        print("[Analyzer] JSON mode failed - trying without format constraint...")
        try:
            raw_response = _call_ollama(
                prompt, model=structuring_model,
                temperature=0.15, max_tokens=3000,
                force_json=False,
            )
            result = _parse_json_response(raw_response)
            if result:
                print("[Analyzer] Fallback structuring succeeded")
        except Exception as e:
            print(f"[Analyzer] Fallback structuring failed: {e}")

    # -- If all structuring failed, build from raw text --
    if result is None:
        print("[Analyzer] All structuring failed - building from raw text")
        result = _make_default_result(
            combined_text, patient_name,
            int(patient_age) if patient_age.isdigit() else 0,
            patient_sex or "Unknown"
        )

    # -- Merge user-provided patient info --
    fields = result.get("fields", {})
    if patient_name and patient_name != "Anonymous":
        fields["pseudonym"] = patient_name
    if patient_age and patient_age.isdigit():
        fields["age"] = int(patient_age)
    if patient_sex:
        fields["sex"] = patient_sex
    result["fields"] = fields

    # -- Validate and fill missing keys --
    result.setdefault("ocrText", combined_text)
    result["fields"].setdefault("pseudonym", patient_name or "Anonymous")
    result["fields"].setdefault("age", 0)
    result["fields"].setdefault("sex", "Unknown")
    result["fields"].setdefault("symptoms", [])

    # Ensure symptoms is a list of strings
    symptoms = result["fields"]["symptoms"]
    if symptoms and isinstance(symptoms[0], dict):
        result["fields"]["symptoms"] = [s.get("text", str(s)) for s in symptoms]

    result.setdefault("predictions", {
        "organ": "unknown",
        "conditions": [{"label": "unknown", "confidence": 0.0, "model": "fallback"}],
    })
    result["predictions"].setdefault("organ", "unknown")
    result["predictions"].setdefault("conditions", [])
    if not result["predictions"]["conditions"]:
        result["predictions"]["conditions"] = [{"label": "unknown", "confidence": 0.0, "model": "fallback"}]

    # Ensure conditions have model field
    for cond in result["predictions"]["conditions"]:
        cond.setdefault("model", "llava_mistral_v3")

    # Ensure localization exists
    result["predictions"].setdefault("localization", {
        "mask": [{"x": 150, "y": 150}, {"x": 350, "y": 150},
                 {"x": 350, "y": 350}, {"x": 150, "y": 350}],
        "bbox": [150, 150, 350, 350],
    })
    loc = result["predictions"]["localization"]
    loc.setdefault("mask", [{"x": 150, "y": 150}, {"x": 350, "y": 150},
                            {"x": 350, "y": 350}, {"x": 150, "y": 350}])
    loc.setdefault("bbox", [150, 150, 350, 350])

    # Ensure XAI section
    result.setdefault("xai", {"text_features": [], "shap_table": [], "naturalLanguageSummary": ""})
    result["xai"].setdefault("text_features", [])
    result["xai"].setdefault("shap_table", [])
    result["xai"].setdefault("naturalLanguageSummary", "")

    # Sanitize SHAP table (remove non-medical features)
    result["xai"]["shap_table"] = _sanitize_shap_table(result["xai"]["shap_table"])

    # Generate summary if missing
    if not result["xai"]["naturalLanguageSummary"]:
        organ = result["predictions"]["organ"]
        conds = result["predictions"]["conditions"]
        syms = result["fields"]["symptoms"]
        primary = conds[0] if conds else {"label": "unknown", "confidence": 0}
        result["xai"]["naturalLanguageSummary"] = (
            f"The system analyzed the medical document and identified findings related to the {organ}. "
            f"The primary predicted condition is {primary['label']} with "
            f"{primary.get('confidence', 0) * 100:.0f}% confidence. "
            f"Key symptoms include: {', '.join(syms[:4]) if syms else 'none explicitly identified'}."
        )

    organ = result['predictions']['organ']
    conditions = [c['label'] for c in result['predictions']['conditions']]
    symptoms_out = result['fields']['symptoms']
    print(f"\n[Analyzer] RESULT: organ={organ}")
    print(f"[Analyzer] RESULT: conditions={conditions}")
    print(f"[Analyzer] RESULT: symptoms={symptoms_out}")
    print(f"[Analyzer] RESULT: text_features={len(result['xai']['text_features'])}")
    print(f"[Analyzer] RESULT: shap_table={len(result['xai']['shap_table'])}")
    print("=" * 60 + "\n")

    return result
