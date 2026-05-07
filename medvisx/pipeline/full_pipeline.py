"""
MedVis-X Full 6-Stage Pipeline
Orchestrates: OCR -> NER -> Scoring -> Generation -> Grad-CAM -> Ollama XAI

Pipeline stages:
  Stage 1: OCR (CRNN Model - ResNet18 + BiLSTM + CTC)
  Stage 2: NER (ClinicalBERT - extract symptoms, diseases, medications)
  Stage 3: Hypothesis Scoring (Deterministic weighted scoring + SHAP)
  Stage 4: Image Generation (LoRA Stable Diffusion v1.5)
  Stage 5: Grad-CAM Explainability Overlay + SHAP Attribution
  Stage 6: Ollama Local Multimodal XAI (llava model)

Design:
  - Lightweight stages (OCR, NER, Scorer, Ollama) are loaded at init
  - Heavy GPU stages (SD, Grad-CAM) are lazy-loaded on first inference
    to avoid OOM on 8 GB VRAM cards
"""
import os
import sys
import time
import traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE


class MedVisXPipeline:
    def __init__(self):
        """
        Initialise lightweight pipeline components only.
        Heavy GPU models (SD, Grad-CAM) are lazy-loaded on first run().
        """
        print("\n" + "=" * 60)
        print("  MedVis-X Pipeline Initialisation")
        print("=" * 60)
        print(f"  Device: {DEVICE}")
        if DEVICE == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            except Exception as e:
                print(f"  GPU info unavailable: {e}")
        print("=" * 60 + "\n")

        t0 = time.time()

        # ── Lightweight stages (loaded eagerly) ──────────────────────
        # Stage 1: OCR  (small CRNN — ~5 MB)
        print("[1/6] Loading OCR model...")
        from pipeline.stage1_ocr.ocr_inference import OCRInference
        self.ocr = OCRInference()

        # Stage 2: NER  (ClinicalBERT — loaded on **CPU** to save VRAM)
        print("[2/6] Loading NER model...")
        from pipeline.stage2_ner.ner_extractor import NERExtractor
        self.ner = NERExtractor()

        # Stage 3: Scorer (pure Python — no model)
        print("[3/6] Initialising Hypothesis Scorer...")
        from pipeline.stage3_scoring.hypothesis_scorer import HypothesisScorer
        self.scorer = HypothesisScorer()

        # Stage 6: Ollama (just validates connectivity — no GPU memory)
        print("[6/6] Connecting to Ollama...")
        from pipeline.stage6_local_llm.llava_explainer import OllamaExplainer
        self.explainer = OllamaExplainer()

        # ── Heavy stages (deferred — loaded on first inference) ──────
        self._generator = None   # Stage 4
        self._gradcam = None     # Stage 5

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"  Lightweight stages ready ({elapsed:.1f}s)")
        print(f"  SD + Grad-CAM will load on first inference")
        print(f"{'=' * 60}\n")

    # ── Lazy accessors ────────────────────────────────────────────────
    def _get_generator(self):
        if self._generator is None:
            print("[4/6] Loading Stable Diffusion + LoRA (first run)...")
            from pipeline.stage4_generation.lora_generator import LoRAGenerator
            self._generator = LoRAGenerator.get_instance()
        return self._generator

    def _get_gradcam(self):
        if self._gradcam is None:
            print("[5/6] Loading Grad-CAM backbone (first run)...")
            from pipeline.stage5_explainability.gradcam import GradCAMExplainer
            self._gradcam = GradCAMExplainer()
        return self._gradcam

    def run(self, image_input=None, text_input: str = None):
        """
        Run the full 6-stage pipeline.

        Args:
            image_input: file path, numpy array, or PIL Image of prescription
                         (pass None if using text_input only)
            text_input:  optional direct text override (skips OCR)

        Returns:
            dict with all stage outputs
        """
        from pipeline.stage4_generation.prompt_builder import build_prompt
        from pipeline.stage5_explainability.shap_explainer import generate_shap_plot

        results = {}
        timings = {}

        # ── Stage 1: OCR ──────────────────────────────────────────────────
        t = time.time()
        if text_input:
            ocr_result = {
                "raw_text": text_input,
                "corrected_text": text_input,
                "medicines_found": {},
            }
        else:
            print("  Stage 1: Running OCR...")
            ocr_result = self.ocr.extract_text(image_input)
        results["ocr"] = ocr_result
        timings["ocr"] = time.time() - t
        print(f"  [OCR] Text: {ocr_result['corrected_text'][:80]}...")

        # ── Stage 2: NER ──────────────────────────────────────────────────
        t = time.time()
        print("  Stage 2: Extracting entities (NER)...")
        combined_text = ocr_result["corrected_text"]
        for med, info in ocr_result.get("medicines_found", {}).items():
            combined_text += f". Prescribed {med} for {', '.join(info['diseases'])}."
        entities = self.ner.extract(combined_text)
        results["entities"] = entities
        timings["ner"] = time.time() - t
        entity_count = sum(
            len(v) for k, v in entities.items()
            if k != "raw_entities" and isinstance(v, list)
        )
        print(f"  [NER] Found {entity_count} entities")

        # ── Stage 3: Hypothesis Scoring ───────────────────────────────────
        t = time.time()
        print("  Stage 3: Scoring hypotheses...")
        scoring = self.scorer.score(entities)
        results["scoring"] = scoring
        timings["scoring"] = time.time() - t
        print(f"  [Scoring] Top: {scoring['top_disease']} "
              f"({scoring['top_score']:.2f})")

        # ── Stage 4: Image Generation (lazy-loaded) ──────────────────────
        t = time.time()
        prompt = build_prompt(scoring)
        results["prompt"] = prompt
        try:
            generator = self._get_generator()
            print("  Stage 4: Generating image...")
            generated_image = generator.generate(prompt)
            results["generated_image"] = generated_image
            timings["generation"] = time.time() - t
            print(f"  [Generator] Done ({timings['generation']:.1f}s)")
        except Exception as e:
            print(f"  [Generator] ERROR: {e}")
            traceback.print_exc()
            from PIL import Image as _Img
            results["generated_image"] = _Img.new("RGB", (512, 512), (30, 30, 30))
            timings["generation"] = time.time() - t

        # Free SD VRAM before Grad-CAM if both use GPU
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # ── Stage 5: Explainability (lazy-loaded) ────────────────────────
        t = time.time()
        print("  Stage 5: Generating explainability overlays...")
        try:
            gradcam = self._get_gradcam()
            heatmap_image = gradcam.generate_heatmap(results["generated_image"])
        except Exception as e:
            print(f"  [Grad-CAM] ERROR: {e}")
            traceback.print_exc()
            heatmap_image = results["generated_image"]

        shap_plot = generate_shap_plot(
            scoring["shap_values"],
            scoring["top_disease"],
            scoring["top_score"],
        )
        results["heatmap"] = heatmap_image
        results["shap_plot"] = shap_plot
        timings["explainability"] = time.time() - t
        print(f"  [XAI] Grad-CAM + SHAP done ({timings['explainability']:.1f}s)")

        # ── Stage 6: Ollama Explanation ───────────────────────────────────
        t = time.time()
        print("  Stage 6: Generating clinical explanation (Ollama)...")
        clinical_context = {
            "ocr_text":    ocr_result["corrected_text"],
            "entities":    entities,
            "top_disease": scoring["top_disease"],
            "top_score":   scoring["top_score"],
            "shap_values": scoring["shap_values"],
            "prompt":      prompt,
        }
        try:
            explanation = self.explainer.explain(
                results["generated_image"], clinical_context
            )
        except Exception as e:
            print(f"  [Ollama] ERROR: {e}")
            explanation = self.explainer._fallback_report(
                scoring["top_disease"], scoring["top_score"],
                [e["text"] for e in entities.get("symptoms", [])],
                [e["text"] for e in entities.get("medications", [])],
                scoring["shap_values"],
            )
        results["explanation"] = explanation
        timings["explanation"] = time.time() - t
        print(f"  [Ollama] Done ({timings['explanation']:.1f}s)")

        results["timings"] = timings
        total = sum(timings.values())
        print(f"\n  Pipeline complete! Total: {total:.1f}s\n")

        return results
