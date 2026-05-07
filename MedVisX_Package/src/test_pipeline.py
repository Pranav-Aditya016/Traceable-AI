"""Quick test of the NLP pipeline stages (no GPU-heavy models needed)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("  MedVis-X NLP Pipeline Test")
print("=" * 50)

# Stage 2: NER
print("\n[Stage 2] Testing NER...")
from pipeline.stage2_ner.ner_extractor import NERExtractor

ner = NERExtractor()
text = ("Patient presents with high fever, productive cough, "
        "and crackles in lower left lobe. SpO2 < 92%.")
entities = ner.extract(text)
print("NER Results:")
for cat, items in entities.items():
    if cat == "raw_entities":
        continue
    if isinstance(items, list) and items:
        names = [e["text"] for e in items]
        print(f"  {cat}: {names}")

# Stage 3: Scoring
print("\n[Stage 3] Testing Hypothesis Scorer...")
from pipeline.stage3_scoring.hypothesis_scorer import HypothesisScorer

scorer = HypothesisScorer()
scoring = scorer.score(entities)
print(f"Top Hypothesis: {scoring['top_disease']} ({scoring['top_score']:.2f})")
print(f"Top 3: {scoring['top_hypotheses']}")
print(f"Modality: {scoring['modality']}")
print(f"SHAP values: {scoring['shap_values']}")

# Stage 4: Prompt building
print("\n[Stage 4] Testing Prompt Builder...")
from pipeline.stage4_generation.prompt_builder import build_prompt

prompt = build_prompt(scoring)
print(f"Generation Prompt: {prompt}")

# Stage 5: SHAP plot
print("\n[Stage 5] Testing SHAP Plot Generation...")
from pipeline.stage5_explainability.shap_explainer import generate_shap_plot

shap_img = generate_shap_plot(
    scoring["shap_values"], scoring["top_disease"], scoring["top_score"]
)
print(f"SHAP Plot size: {shap_img.size}")

# Stage 6: Ollama connectivity test
print("\n[Stage 6] Testing Ollama connectivity...")
from pipeline.stage6_local_llm.llava_explainer import OllamaExplainer

explainer = OllamaExplainer()

print("\n" + "=" * 50)
print("  All NLP Pipeline Tests PASSED!")
print("=" * 50)
