"""Quick smoke test: run the pipeline with text-only input."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.full_pipeline import MedVisXPipeline

pipe = MedVisXPipeline()

print("\n\n===== Running pipeline with text input =====\n")
results = pipe.run(
    text_input="Patient presents with high fever, productive cough, "
               "and crackles in lower left lobe. SpO2 < 92%."
)

print("\n===== RESULTS =====")
print(f"OCR text:      {results['ocr']['corrected_text'][:60]}...")
print(f"Top disease:   {results['scoring']['top_disease']}")
print(f"Top score:     {results['scoring']['top_score']:.2f}")
print(f"Gen image:     {results['generated_image'].size}")
print(f"Heatmap:       {results['heatmap'].size}")
print(f"SHAP plot:     {type(results['shap_plot'])}")
print(f"Explanation:   {results['explanation'][:100]}...")
print("\n===== ALL STAGES PASSED =====")
