import requests, sys

url = "http://localhost:8000/api/analyze"
filepath = r"MedVisX_Paper_Samples\Patient_101\1_NLP_Extraction_Report.png"

with open(filepath, "rb") as f:
    files = [("files", ("test.png", f, "image/png"))]
    data = {
        "clinical_text": "Patient has persistent cough and fever for 5 days",
        "patient_name": "Test Patient",
        "patient_age": "45",
        "patient_sex": "Male",
    }
    print("Sending request...")
    r = requests.post(url, files=files, data=data, timeout=600)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        print(f"Report ID: {j.get('report_id', 'N/A')}")
        pred = j.get("prediction", {})
        conds = pred.get("conditions", [])
        if conds:
            print(f"Top prediction: {conds[0]['label']} ({conds[0]['confidence']})")
        imgs = j.get("images", {})
        print(f"Generated image: {'YES' if imgs.get('generated') else 'NO'} ({len(imgs.get('generated',''))} chars)")
        print(f"Heatmap: {'YES' if imgs.get('heatmap') else 'NO'} ({len(imgs.get('heatmap',''))} chars)")
        print(f"SHAP plot: {'YES' if imgs.get('shap') else 'NO'} ({len(imgs.get('shap',''))} chars)")
        xai = j.get("xai", {})
        print(f"Explanation: {'YES' if xai.get('explanation') else 'NO'} ({len(xai.get('explanation',''))} chars)")
        print(f"Timings: {j.get('timings', {})}")
        print("=== FULL PIPELINE PASSED! ===")
    else:
        print(f"ERROR: {r.text[:500]}")
        sys.exit(1)
