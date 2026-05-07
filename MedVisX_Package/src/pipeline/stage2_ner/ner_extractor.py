"""
Named Entity Recognition using ClinicalBERT
Extracts: symptoms, diseases, medications, diagnostic_tests, anatomical_refs
Downloads ~440MB on first run from HuggingFace, then runs offline.
"""
import os
import sys
from transformers import pipeline as hf_pipeline
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEVICE

# Use a clinical NER model from HuggingFace
CLINICALBERT_MODEL = "samrawal/bert-base-uncased_clinical-ner"

ENTITY_CATEGORIES = {
    "symptoms":         ["SYMPTOM", "SIGN", "PROBLEM"],
    "diseases":         ["DISEASE", "DISORDER", "CONDITION"],
    "medications":      ["MEDICATION", "DRUG", "CHEMICAL"],
    "diagnostic_tests": ["TEST", "PROCEDURE", "LAB"],
    "anatomical_refs":  ["ANATOMY", "BODY_PART", "ORGAN"],
}


class NERExtractor:
    def __init__(self):
        print("[NER] Loading ClinicalBERT NER model...")
        try:
            # Always use CPU for NER to leave GPU VRAM free for SD + GradCAM
            self.nlp = hf_pipeline(
                "ner",
                model=CLINICALBERT_MODEL,
                tokenizer=CLINICALBERT_MODEL,
                aggregation_strategy="simple",
                device=-1,          # force CPU
            )
            self._model_loaded = True
            print("[NER] ClinicalBERT model loaded (CPU)")
        except Exception as e:
            print(f"[NER] WARNING: Could not load ClinicalBERT: {e}")
            print("[NER] Falling back to keyword-based NER extraction")
            self._model_loaded = False
            self.nlp = None

    def _keyword_extract(self, text: str) -> dict:
        """Fallback keyword-based extraction when model unavailable."""
        text_lower = text.lower()
        result = {cat: [] for cat in ENTITY_CATEGORIES}
        result["raw_entities"] = []

        # Symptom keywords
        symptom_keywords = [
            "fever", "cough", "headache", "pain", "nausea", "vomiting",
            "fatigue", "dyspnea", "shortness of breath", "wheezing",
            "chest tightness", "chest pain", "diarrhea", "swelling",
            "edema", "rash", "blurred vision", "seizure", "tremor",
            "crackles", "sputum", "hemoptysis", "night sweats",
            "weight loss", "polyuria", "polydipsia", "palpitations",
            "dysuria", "hematuria", "confusion", "memory loss",
            "insomnia", "anxiety", "joint pain", "bradykinesia",
            "rigidity", "floaters", "dark spots", "low oxygen",
            "spo2", "tachypnea", "productive cough", "crushing pain",
            "diaphoresis", "rebound tenderness", "anorexia",
            "irregular mole", "asymmetric lesion", "color variation",
            "border irregularity", "ulceration", "skin lesion",
            "focal neurological", "papilledema", "hemiparesis",
            "cognitive changes", "visual acuity loss", "microaneurysm",
            "hemorrhage", "consolidation", "claudication",
        ]
        for kw in symptom_keywords:
            if kw in text_lower:
                result["symptoms"].append({"text": kw, "score": 0.85})

        # Disease keywords
        disease_keywords = [
            "pneumonia", "diabetes", "hypertension", "asthma", "copd",
            "tuberculosis", "cancer", "tumor", "glioma", "melanoma",
            "retinopathy", "bronchitis", "appendicitis", "epilepsy",
            "alzheimer", "parkinson", "dementia", "heart failure",
            "myocardial infarction", "stroke", "uti",
            "urinary tract infection", "depression", "anxiety",
        ]
        for kw in disease_keywords:
            if kw in text_lower:
                result["diseases"].append({"text": kw, "score": 0.80})

        # Medication keywords
        medication_keywords = [
            "amoxicillin", "azithromycin", "metformin", "insulin",
            "lisinopril", "atorvastatin", "omeprazole", "salbutamol",
            "prednisolone", "ciprofloxacin", "paracetamol", "ibuprofen",
            "warfarin", "levothyroxine", "metoprolol", "furosemide",
            "clopidogrel", "amlodipine", "sertraline", "gabapentin",
        ]
        for kw in medication_keywords:
            if kw in text_lower:
                result["medications"].append({"text": kw, "score": 0.90})

        return result

    def extract(self, text: str) -> dict:
        """Extract medical entities from clinical text."""
        # If model not loaded, use keyword fallback
        if not self._model_loaded or self.nlp is None:
            return self._keyword_extract(text)

        try:
            entities_raw = self.nlp(text)
        except Exception as e:
            print(f"[NER] Model inference failed: {e}, using keyword fallback")
            return self._keyword_extract(text)

        result = {cat: [] for cat in ENTITY_CATEGORIES}
        result["raw_entities"] = entities_raw

        for ent in entities_raw:
            label = ent["entity_group"].upper()
            word = ent["word"].strip()
            score = round(ent["score"], 3)

            matched = False
            for category, keywords in ENTITY_CATEGORIES.items():
                if any(kw in label for kw in keywords):
                    result[category].append({"text": word, "score": score})
                    matched = True
                    break

            if not matched:
                # Fallback heuristic classification by keyword matching
                word_lower = word.lower()
                if any(s in word_lower for s in ["pain", "fever", "cough", "ache",
                                                   "nausea", "fatigue", "swelling"]):
                    result["symptoms"].append({"text": word, "score": score})
                elif any(d in word_lower for d in ["itis", "emia", "osis", "oma",
                                                     "diabetes", "cancer", "tumor"]):
                    result["diseases"].append({"text": word, "score": score})

        # Also run keyword extraction to supplement model output
        keyword_result = self._keyword_extract(text)
        for cat in ENTITY_CATEGORIES:
            existing_texts = {e["text"].lower() for e in result[cat]}
            for item in keyword_result[cat]:
                if item["text"].lower() not in existing_texts:
                    result[cat].append(item)

        return result
