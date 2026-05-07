"""
Deterministic Hypothesis Scoring
Formula: S(d) = Sum[ w_imp(e_i) * w_rec(e_i) * A(e_i, d) ]
Fully transparent — every score is traceable to its inputs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TOP_K_HYPOTHESES
from pipeline.stage3_scoring.symptom_disease_db import (
    SYMPTOM_DISEASE_ASSOCIATIONS,
    ENTITY_IMPORTANCE_WEIGHTS,
    DISEASE_TO_MODALITY,
)


class HypothesisScorer:
    def __init__(self):
        self.diseases = list(SYMPTOM_DISEASE_ASSOCIATIONS.keys())

    def score(self, extracted_entities: dict) -> dict:
        """
        Score all diseases given extracted entities.
        Returns ranked hypotheses with SHAP-style attribution.
        """
        # Flatten all entities with their type
        flat_entities = []
        for etype, items in extracted_entities.items():
            if etype in ("raw_entities", "medicines_found"):
                continue
            if not isinstance(items, list):
                continue
            weight = ENTITY_IMPORTANCE_WEIGHTS.get(etype, 0.5)
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    flat_entities.append({
                        "text":       item["text"].lower(),
                        "type":       etype,
                        "confidence": item.get("score", 0.8),
                        "importance": weight,
                    })

        # Score each disease
        scores = {}
        attributions = {}  # entity -> contribution per disease
        for disease, assoc in SYMPTOM_DISEASE_ASSOCIATIONS.items():
            total = 0.0
            contrib = {}
            for entity in flat_entities:
                # Find best matching symptom key
                best_key, best_assoc = None, 0.0
                for sym_key, sym_assoc in assoc.items():
                    if sym_key in entity["text"] or entity["text"] in sym_key:
                        if sym_assoc > best_assoc:
                            best_assoc = sym_assoc
                            best_key = sym_key
                if best_key:
                    contribution = (entity["importance"]
                                    * entity["confidence"]
                                    * best_assoc)
                    total += contribution
                    contrib[entity["text"]] = round(contribution, 4)
            scores[disease] = total
            attributions[disease] = contrib

        # Normalise to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {d: round(s / max_score, 4) for d, s in scores.items()}

        # Rank
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:TOP_K_HYPOTHESES]

        top_disease = top_k[0][0] if top_k else "unknown"
        modality = DISEASE_TO_MODALITY.get(
            top_disease, DISEASE_TO_MODALITY["default"]
        )
        shap_values = attributions.get(top_disease, {})

        return {
            "top_hypotheses": top_k,
            "top_disease":    top_disease,
            "top_score":      top_k[0][1] if top_k else 0.0,
            "modality":       modality,
            "shap_values":    shap_values,
            "all_scores":     dict(ranked),
        }
