"""Quick end-to-end test of the new MedVisX two-model pipeline."""
import requests, json, time

url = "http://localhost:8000/api/analyze"
data = {
    "clinical_text": (
        "Patient John Doe, 58 year old male, persistent dry cough for 3 weeks, "
        "mild hemoptysis, 5kg weight loss over 2 months. CT shows 3cm spiculated "
        "mass in right upper lobe with mediastinal lymphadenopathy. 30 pack-year smoker."
    ),
    "patient_name": "John Doe",
    "patient_age": "58",
    "patient_sex": "M",
}

print("Sending request...")
t0 = time.time()
try:
    resp = requests.post(url, data=data, timeout=600)
    elapsed = time.time() - t0
    print(f"Status: {resp.status_code} in {elapsed:.1f}s")
    if resp.status_code == 200:
        r = resp.json()
        print(f"Report: {r['report_id']}")
        print(f"Patient: {r['patient']}")
        print(f"OCR method: {r['ocr']['method_used']}")
        print(f"OCR text (first 200): {r['ocr']['final_text'][:200]}")
        print(f"Prediction: {r['prediction']['top_disease']} ({r['prediction']['confidence']*100:.1f}%)")
        print(f"Organ: {r['prediction'].get('organ', 'N/A')}")
        print(f"Modality: {r['prediction']['modality']}")
        print(f"Hypotheses: {r['prediction']['hypotheses']}")
        print(f"Symptoms: {r['entities']['symptoms']}")
        print(f"Conditions: {r['entities']['conditions']}")
        print(f"Text features: {r['xai'].get('text_features', [])}")
        print(f"SHAP values: {r['xai']['shap_values']}")
        print(f"Explanation (first 300): {r['xai']['explanation'][:300]}")
        print(f"Timings: {r['timings']}")
        print(f"Has generated image: {len(r['images']['generated']) > 100}")
        print(f"Has heatmap: {len(r['images']['heatmap']) > 100}")
        print(f"Has SHAP plot: {len(r['images']['shap_plot']) > 100}")
    else:
        print(f"Error: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
