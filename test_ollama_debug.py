"""Debug: test LLaVA structured analysis directly."""
import requests, json

clinical_text = (
    "Patient John Doe, 58 year old male, presenting with persistent dry cough "
    "for 3 weeks, mild hemoptysis, weight loss of 5kg over 2 months. CT chest "
    "shows 3cm spiculated mass in the right upper lobe with mediastinal "
    "lymphadenopathy. History of 30 pack-year smoking."
)

prompt = f"""You are an expert medical text analysis AI. Analyze the following clinical text carefully.

Clinical Text:
\"\"\"
{clinical_text}
\"\"\"

Perform ALL of the following tasks:

1. **Field Extraction**: From the text, identify:
   - Patient pseudonym/name (use "Anonymous" if not found)
   - Patient age (integer, use 0 if not found)
   - Patient sex (M/F/Other/Unknown)
   - List of ALL symptoms and clinical findings mentioned
2. **Clinical Prediction**: Based on the extracted information:
   - Predict the primary organ system affected
   - Predict up to 2 most likely conditions with confidence (0.0 to 1.0)
3. **XAI (Explainable AI)**: Generate:
   - Top 3 text features that most influenced your prediction, with importance weights (0.0 to 1.0)
   - SHAP table: for each relevant clinical feature, its value and SHAP contribution score (-1.0 to 1.0)

You MUST respond with ONLY valid JSON matching this exact structure (no markdown, no extra text):
{{
  "ocrText": "{clinical_text}",
  "fields": {{
    "pseudonym": "patient name or Anonymous",
    "age": 45,
    "sex": "M",
    "symptoms": ["symptom1", "symptom2"]
  }},
  "predictions": {{
    "organ": "lungs",
    "conditions": [
      {{"label": "pneumonia", "confidence": 0.85}},
      {{"label": "bronchitis", "confidence": 0.45}}
    ]
  }},
  "xai": {{
    "text_features": [
      {{"feature": "productive cough with fever", "weight": 0.92}},
      {{"feature": "bilateral infiltrates", "weight": 0.87}},
      {{"feature": "elevated WBC count", "weight": 0.76}}
    ],
    "shap_table": [
      {{"feature": "age", "value": "45", "shap": 0.15}},
      {{"feature": "sex", "value": "M", "shap": 0.05}},
      {{"feature": "symptom: cough", "value": "present", "shap": 0.45}},
      {{"feature": "symptom: fever", "value": "present", "shap": 0.38}}
    ]
  }}
}}"""

payload = {
    "model": "llava",
    "prompt": prompt,
    "stream": False,
    "options": {
        "num_predict": 4096,
        "temperature": 0.1,
    },
}

print("Sending to Ollama...")
resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
raw = resp.json().get("response", "")
print(f"Response length: {len(raw)} chars")
print("--- FULL RAW RESPONSE ---")
print(raw)
print("--- END ---")

# Try parsing
try:
    parsed = json.loads(raw.strip())
    print("\nDirect JSON parse: SUCCESS")
    print(json.dumps(parsed, indent=2)[:500])
except json.JSONDecodeError as e:
    print(f"\nDirect JSON parse FAILED: {e}")
    # Try finding boundaries
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end])
            print("Boundary parse: SUCCESS")
            print(json.dumps(parsed, indent=2)[:500])
        except json.JSONDecodeError as e2:
            print(f"Boundary parse FAILED: {e2}")
            print(f"Fragment: ...{raw[max(0,end-100):end+50]}...")
