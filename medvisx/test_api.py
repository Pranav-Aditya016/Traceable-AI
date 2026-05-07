"""Quick end-to-end test of MedVis-X via the Gradio app's HTTP API."""
import requests
import json

BASE = "http://localhost:7860"

# Test the app is responding
try:
    r = requests.get(BASE, timeout=5)
    print(f"App status: {r.status_code} OK" if r.status_code == 200 else f"App status: {r.status_code}")
except Exception as e:
    print(f"App not reachable: {e}")
    exit(1)

# Test the API endpoint
print("\nTesting pipeline via API...")
payload = {
    "data": [
        None,  # no image
        "Patient presents with high fever, productive cough, and crackles in lower left lobe. SpO2 < 92%.",
        True,  # use text only
    ]
}

try:
    r = requests.post(f"{BASE}/api/predict", json=payload, timeout=300)
    if r.status_code == 200:
        result = r.json()
        print("API Response received!")
        # Check what we got back
        data = result.get("data", [])
        print(f"  Outputs: {len(data)} items")
        for i, item in enumerate(data):
            if item is None:
                print(f"  [{i}] None (model loading or image gen pending)")
            elif isinstance(item, str):
                print(f"  [{i}] Text: {item[:100]}...")
            elif isinstance(item, dict):
                print(f"  [{i}] Image/data dict")
            else:
                print(f"  [{i}] Type: {type(item).__name__}")
    else:
        print(f"API error: {r.status_code}")
        print(r.text[:500])
except requests.Timeout:
    print("API timeout (pipeline may still be loading models)")
except Exception as e:
    print(f"API test failed: {e}")
