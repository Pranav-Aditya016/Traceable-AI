"""Test: Image generation + Grad-CAM + Ollama XAI"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- 1. Generate image ---
print("=" * 50)
print("TEST 1: Stable Diffusion + LoRA Generation")
print("=" * 50)
from pipeline.stage4_generation.lora_generator import LoRAGenerator
gen = LoRAGenerator.get_instance()
prompt = "high resolution chest X-ray showing bilateral lower lobe consolidation consistent with pneumonia, medical imaging, grayscale"
img = gen.generate(prompt)
img.save("test_xai_generated.png")
print(f"Generated: {img.size}\n")

# --- 2. Grad-CAM ---
print("=" * 50)
print("TEST 2: Grad-CAM Heatmap")
print("=" * 50)
from pipeline.stage5_explainability.gradcam import GradCAMExplainer
gcam = GradCAMExplainer()
heatmap = gcam.generate_heatmap(img)
heatmap.save("test_xai_heatmap.png")
print(f"Heatmap: {heatmap.size}\n")

# --- 3. Ollama XAI ---
print("=" * 50)
print("TEST 3: Ollama llava Explanation")
print("=" * 50)
from pipeline.stage6_local_llm.llava_explainer import OllamaExplainer
explainer = OllamaExplainer()
context = {
    "ocr_text": "Patient presents with high fever, productive cough, crackles in lower left lobe. SpO2 < 92%.",
    "entities": {"symptoms": [{"text": "fever", "score": 0.9}, {"text": "cough", "score": 0.9}, {"text": "crackles", "score": 0.85}]},
    "top_disease": "pneumonia",
    "top_score": 1.0,
    "shap_values": {"fever": 0.25, "cough": 0.22, "crackles": 0.20},
    "prompt": prompt,
}
explanation = explainer.explain(img, context)
print(f"\n--- Explanation ({len(explanation)} chars) ---")
print(explanation[:500])
print("...\n")

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
