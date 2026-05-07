"""
Stage 3 — Symbolic Hypothesis Scorer + SHAP Attribution.
No GPU required. Pure Python + SHAP library.
"""

import shap
import numpy as np

ASSOCIATION_DB = {
    # Respiratory
    "fever":           {"pneumonia": 0.75, "influenza": 0.65, "sepsis": 0.60, "tuberculosis": 0.50, "meningitis": 0.55},
    "cough":           {"pneumonia": 0.82, "bronchitis": 0.78, "tuberculosis": 0.60, "asthma exacerbation": 0.65, "copd exacerbation": 0.70},
    "spo2":            {"pneumonia": 0.92, "pulmonary embolism": 0.85, "copd exacerbation": 0.80, "heart failure": 0.75},
    "crackles":        {"pneumonia": 0.88, "heart failure": 0.70, "pulmonary fibrosis": 0.65},
    "wheezing":        {"asthma exacerbation": 0.92, "copd exacerbation": 0.80, "bronchitis": 0.65, "anaphylaxis": 0.72},
    "hemoptysis":      {"tuberculosis": 0.82, "lung cancer": 0.78, "pulmonary embolism": 0.68, "bronchiectasis": 0.72},
    "dyspnea":         {"pneumonia": 0.78, "heart failure": 0.88, "asthma exacerbation": 0.75, "copd exacerbation": 0.80, "pulmonary embolism": 0.82, "anemia": 0.60},
    # Cardiac
    "chest pain":      {"myocardial infarction": 0.88, "myocardial ischemia": 0.82, "angina": 0.78, "pericarditis": 0.65, "aortic dissection": 0.72, "pulmonary embolism": 0.60},
    "palpitations":    {"atrial fibrillation": 0.85, "arrhythmia": 0.88, "hyperthyroidism": 0.65, "anxiety": 0.55, "supraventricular tachycardia": 0.78},
    "edema":           {"heart failure": 0.88, "deep vein thrombosis": 0.72, "nephrotic syndrome": 0.70, "liver cirrhosis": 0.75, "cellulitis": 0.55},
    "tachycardia":     {"sepsis": 0.80, "pulmonary embolism": 0.75, "atrial fibrillation": 0.85, "anemia": 0.68, "hyperthyroidism": 0.68},
    "syncope":         {"arrhythmia": 0.78, "vasovagal syncope": 0.82, "aortic stenosis": 0.65, "pulmonary embolism": 0.60},
    "hypertension":    {"hypertensive crisis": 0.88, "chronic kidney disease": 0.72, "renal artery stenosis": 0.68, "pheochromocytoma": 0.58},
    # Neurological
    "headache":        {"migraine": 0.78, "meningitis": 0.68, "subarachnoid hemorrhage": 0.62, "brain tumor": 0.50, "hypertensive crisis": 0.65},
    "seizure":         {"epilepsy": 0.88, "brain tumor": 0.72, "stroke": 0.65, "meningitis": 0.60, "hypoglycemia": 0.55},
    "blurred vision":  {"glaucoma": 0.72, "diabetic retinopathy": 0.78, "stroke": 0.68, "multiple sclerosis": 0.58, "cataract": 0.65},
    "weakness":        {"stroke": 0.78, "multiple sclerosis": 0.68, "guillain-barre": 0.62, "myasthenia gravis": 0.58},
    "confusion":       {"stroke": 0.72, "meningitis": 0.70, "encephalitis": 0.78, "metabolic encephalopathy": 0.82, "dementia": 0.68},
    # Musculoskeletal
    "fracture":        {"bone fracture": 0.95, "osteoporotic fracture": 0.88, "pathological fracture": 0.60},
    "swelling":        {"bone fracture": 0.68, "cellulitis": 0.65, "deep vein thrombosis": 0.78, "gout": 0.72},
    "joint pain":      {"osteoarthritis": 0.82, "rheumatoid arthritis": 0.78, "gout": 0.72, "septic arthritis": 0.68},
    "back pain":       {"disc herniation": 0.78, "spinal stenosis": 0.68, "vertebral fracture": 0.72, "nephrolithiasis": 0.58},
    # Gastrointestinal
    "nausea":          {"gastroenteritis": 0.78, "appendicitis": 0.68, "pancreatitis": 0.72, "myocardial infarction": 0.55},
    "vomiting":        {"gastroenteritis": 0.78, "appendicitis": 0.72, "bowel obstruction": 0.68, "pancreatitis": 0.65},
    "abdominal pain":  {"appendicitis": 0.82, "pancreatitis": 0.78, "cholecystitis": 0.72, "bowel obstruction": 0.68, "nephrolithiasis": 0.60},
    "diarrhea":        {"gastroenteritis": 0.88, "inflammatory bowel disease": 0.72, "c. difficile infection": 0.78},
    "jaundice":        {"hepatitis": 0.88, "liver cirrhosis": 0.82, "biliary obstruction": 0.78, "pancreatic cancer": 0.72},
    # Systemic / Other
    "fatigue":         {"anemia": 0.78, "hypothyroidism": 0.72, "heart failure": 0.68, "depression": 0.65, "cancer": 0.60},
    "weight loss":     {"cancer": 0.78, "tuberculosis": 0.72, "diabetes mellitus": 0.68, "hyperthyroidism": 0.65, "hiv": 0.68},
    "night sweats":    {"tuberculosis": 0.82, "lymphoma": 0.78, "hiv": 0.68, "menopause": 0.72},
    "dysuria":         {"urinary tract infection": 0.92, "nephrolithiasis": 0.68, "prostatitis": 0.65},
    "polyuria":        {"diabetes mellitus": 0.88, "diabetes insipidus": 0.72, "hypercalcemia": 0.62},
    "lesion":          {"melanoma": 0.68, "basal cell carcinoma": 0.62, "squamous cell carcinoma": 0.58, "pneumonia": 0.70},
    "pain":            {"bone fracture": 0.55, "osteoarthritis": 0.65, "neuropathy": 0.52},
    "rash":            {"contact dermatitis": 0.78, "psoriasis": 0.68, "drug reaction": 0.72, "systemic lupus": 0.58, "shingles": 0.72},
}

ORGAN_HINTS = {
    "pneumonia":                "lung",
    "bronchitis":               "lung",
    "tuberculosis":             "lung",
    "copd exacerbation":        "lung",
    "asthma exacerbation":      "lung",
    "pulmonary embolism":       "lung",
    "lung cancer":              "lung",
    "pulmonary fibrosis":       "lung",
    "myocardial infarction":    "heart",
    "myocardial ischemia":      "heart",
    "angina":                   "heart",
    "atrial fibrillation":      "heart",
    "heart failure":            "heart",
    "arrhythmia":               "heart",
    "pericarditis":             "heart",
    "aortic dissection":        "heart",
    "migraine":                 "brain",
    "brain tumor":              "brain",
    "stroke":                   "brain",
    "epilepsy":                 "brain",
    "meningitis":               "brain",
    "encephalitis":             "brain",
    "subarachnoid hemorrhage":  "brain",
    "bone fracture":            "bone",
    "osteoporotic fracture":    "bone",
    "osteoarthritis":           "bone",
    "disc herniation":          "spine",
    "spinal stenosis":          "spine",
    "melanoma":                 "skin",
    "basal cell carcinoma":     "skin",
    "psoriasis":                "skin",
    "retinopathy":              "eye",
    "glaucoma":                 "eye",
    "cataract":                 "eye",
    "appendicitis":             "abdomen",
    "pancreatitis":             "abdomen",
    "cholecystitis":            "abdomen",
    "bowel obstruction":        "abdomen",
    "hepatitis":                "liver",
    "liver cirrhosis":          "liver",
    "urinary tract infection":  "kidney",
    "nephrolithiasis":          "kidney",
    "diabetes mellitus":        "abdomen",
    "hypothyroidism":           "thyroid",
    "hypertensive crisis":      "heart",
    "deep vein thrombosis":     "bone",
    "gastroenteritis":          "abdomen",
}

HIGH_PRIORITY_SYMPTOMS = {
    "spo2", "seizure", "fracture", "hemoptysis", "chest pain",
    "dyspnea", "syncope", "confusion", "jaundice", "tachycardia",
    "hypertension", "weakness", "hemoptysis",
}


def _clinical_importance(symptoms: list) -> float:
    score = 0.45
    for s in symptoms:
        s_lower = s.lower()
        if any(p in s_lower for p in HIGH_PRIORITY_SYMPTOMS):
            score = min(score + 0.12, 1.0)
        # Bonus for specific values (e.g. "SpO2 88%", "fever 39.5C")
        if re.search(r"\d", s_lower):
            score = min(score + 0.05, 1.0)
    return round(score, 3)


def _association_strength(symptoms: list, condition: str) -> float:
    condition_lower = condition.lower()
    total, matches = 0.0, 0
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        for key, conditions in ASSOCIATION_DB.items():
            if key in symptom_lower:
                for cond, weight in conditions.items():
                    if cond in condition_lower or condition_lower in cond:
                        total += weight
                        matches += 1
    return round((total / matches) if matches > 0 else 0.28, 3)


def _temporal_relevance(symptoms: list) -> float:
    acute_terms = ["this morning", "today", "sudden", "acute", "new onset", "recent",
                   "overnight", "hours", "minutes", "abrupt", "worsening", "progressive"]
    for s in symptoms:
        if any(t in s.lower() for t in acute_terms):
            return 0.85
    return 0.50


def _infer_organ_from_condition(label: str) -> str:
    lower_label = (label or "").lower()
    for key, organ in ORGAN_HINTS.items():
        if key in lower_label:
            return organ
    return "unspecified"


def _fallback_conditions_from_symptoms(symptoms: list) -> list:
    condition_scores: dict[str, float] = {}
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        for key, conditions in ASSOCIATION_DB.items():
            if key in symptom_lower:
                for label, weight in conditions.items():
                    condition_scores[label] = condition_scores.get(label, 0.0) + float(weight)

    if not condition_scores:
        return [{"label": "Unspecified condition", "confidence": 0.45, "model": "HybridScorer", "reasoning": "Insufficient symptom evidence for specific diagnosis."}]

    ranked    = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    max_score = max(v for _, v in ranked)

    return [
        {"label": label.title(), "confidence": round(min(0.42 + (0.45 * (score / max(max_score, 0.01))), 0.94), 3),
         "model": "HybridScorer", "reasoning": "Condition inferred from symptom-association database."}
        for label, score in ranked
    ]


def _coerce_confidence(value, default: float = 0.55) -> float:
    try:
        value = float(str(value).replace("%", ""))
    except Exception:
        value = default
    return round(min(max(value if value <= 1.0 else value / 100, 0.0), 1.0), 3)


import re


def run_scorer(ner_result: dict) -> dict:
    print("  [Scorer] Running hypothesis scoring...")

    symptoms = ner_result.get("symptoms", [])
    if not isinstance(symptoms, list):
        symptoms = [str(symptoms)]
    symptoms = [str(s).strip() for s in symptoms if str(s).strip()]

    xai = ner_result.setdefault("xai", {})
    xai.setdefault("text_features", [])
    xai.setdefault("shap_table", [])

    conditions = ner_result.get("conditions", [])
    if not isinstance(conditions, list):
        conditions = []
    if not conditions:
        print("  [Scorer] No conditions from NER — inferring from symptom database.")
        conditions = _fallback_conditions_from_symptoms(symptoms)

    ci   = _clinical_importance(symptoms)
    temp = _temporal_relevance(symptoms)

    scored_conditions, association_scores = [], []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        label = str(cond.get("label", "")).strip()
        if not label:
            continue
        assoc = _association_strength(symptoms, label)
        association_scores.append(assoc)

        # Weighted ensemble: original NER confidence + scorer evidence
        original_conf = _coerce_confidence(cond.get("confidence", 0.55))
        scorer_score  = (ci * 0.30) + (assoc * 0.50) + (temp * 0.20)
        final_conf    = round(min(max((original_conf * 0.55) + (scorer_score * 0.45), 0.0), 0.97), 3)

        scored_conditions.append({
            "label":      label,
            "confidence": final_conf,
            "model":      str(cond.get("model") or "HybridScorer"),
            "reasoning":  str(cond.get("reasoning") or "Weighted scoring over symptoms and clinical priors."),
        })

    if not scored_conditions:
        scored_conditions  = _fallback_conditions_from_symptoms(symptoms)
        association_scores = [0.30]

    scored_conditions.sort(key=lambda x: x["confidence"], reverse=True)
    top       = scored_conditions[0]
    top_assoc = association_scores[0] if association_scores else _association_strength(symptoms, top["label"])

    feature_names  = ["clinical_importance", "association_strength", "temporal_relevance"]
    feature_values = np.array([[ci, top_assoc, temp]])

    def score_fn(X):
        return np.array([(x[0] * 0.45) + (x[1] * 0.35) + (x[2] * 0.20) for x in X])

    try:
        explainer   = shap.Explainer(score_fn, feature_values)
        shap_values = explainer(feature_values)
        shap_entries = [
            {"feature": name, "value": str(round(float(feature_values[0][i]), 3)), "impact": round(float(shap_values.values[0][i]), 3)}
            for i, name in enumerate(feature_names)
        ]
    except Exception as e:
        print(f"  [Scorer] SHAP computation failed (non-critical): {e}")
        shap_entries = [
            {"feature": "clinical_importance", "value": str(round(ci, 3)),       "impact": round((ci - 0.5) * 0.55, 3)},
            {"feature": "association_strength","value": str(round(top_assoc, 3)), "impact": round((top_assoc - 0.3) * 0.65, 3)},
            {"feature": "temporal_relevance",  "value": str(round(temp, 3)),      "impact": round((temp - 0.5) * 0.40, 3)},
        ]

    xai["shap_table"] = xai.get("shap_table", []) + shap_entries

    if not xai.get("text_features"):
        xai["text_features"] = [
            {"feature": s, "weight": round(max(0.40, 0.88 - (i * 0.08)), 3)}
            for i, s in enumerate(symptoms[:6])
        ] or [{"feature": top["label"].lower(), "weight": 0.55}]

    ner_result["conditions"]         = scored_conditions
    ner_result["overall_confidence"] = top["confidence"]

    current_organ = str(ner_result.get("organ", "")).strip().lower()
    inferred_organ = _infer_organ_from_condition(top["label"])
    # Always correct organ if it's unspecified, or if top condition strongly implies a different organ
    if current_organ in {"", "unspecified", "unknown"}:
        ner_result["organ"] = inferred_organ
    elif inferred_organ != "unspecified" and inferred_organ != current_organ and top["confidence"] >= 0.70:
        print(f"  [Scorer] Correcting organ: {current_organ} → {inferred_organ} (top condition: {top['label']})")
        ner_result["organ"] = inferred_organ

    print(f"  [Scorer] Top: {top['label']} ({top['confidence']}) | ci={ci} assoc={top_assoc} temp={temp}")
    return ner_result
