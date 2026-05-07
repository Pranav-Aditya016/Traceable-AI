"""
Stage 2 — Named Entity Recognition via Ollama medgemma:4b.
Falls back to deterministic heuristic parser if Ollama is unavailable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import requests
from PIL import Image

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NER_MODEL   = "medgemma:4b"
TIMEOUT     = 120

SYMPTOM_ALIASES = {
    "fever":           ["fever", "pyrexia", "high temperature", "febrile", "hyperthermia", "temp >38", "temp>38"],
    "cough":           ["cough", "productive cough", "dry cough", "persistent cough", "nonproductive cough"],
    "dyspnea":         ["dyspnea", "shortness of breath", "breathlessness", "sob", "difficulty breathing", "respiratory distress", "breathless"],
    "chest pain":      ["chest pain", "chest discomfort", "tight chest", "chest tightness", "chest pressure", "chest heaviness", "retrosternal pain"],
    "headache":        ["headache", "head pain", "cephalgia", "head ache", "migraine headache"],
    "seizure":         ["seizure", "convulsion", "fits", "tonic-clonic", "epileptic"],
    "blurred vision":  ["blurred vision", "vision loss", "double vision", "diplopia", "visual disturbance", "visual changes", "blurring of vision"],
    "swelling":        ["swelling", "edema", "oedema", "puffiness", "engorgement", "swollen"],
    "pain":            ["pain", "tenderness", "ache", "discomfort", "soreness", "painful"],
    "fracture":        ["fracture", "broken bone", "hairline crack", "cortical discontinuity", "break"],
    "lesion":          ["lesion", "mass", "nodule", "lump", "growth", "tumor", "tumour", "opacity", "consolidation"],
    "fatigue":         ["fatigue", "tiredness", "weakness", "lethargy", "malaise", "exhaustion", "tired"],
    "nausea":          ["nausea", "nauseated", "queasy", "feeling sick", "nauseous"],
    "vomiting":        ["vomiting", "vomit", "emesis", "throwing up", "vomited"],
    "abdominal pain":  ["abdominal pain", "stomach pain", "belly pain", "epigastric pain", "abdominal cramps", "stomach ache", "abdominal tenderness"],
    "diarrhea":        ["diarrhea", "diarrhoea", "loose stools", "watery stools", "frequent stools"],
    "weight loss":     ["weight loss", "losing weight", "unintentional weight loss", "cachexia", "loss of weight"],
    "night sweats":    ["night sweats", "nocturnal sweating", "sweating at night"],
    "palpitations":    ["palpitations", "heart racing", "rapid heartbeat", "irregular heartbeat", "heart pounding", "racing heart"],
    "syncope":         ["syncope", "fainting", "faint", "loss of consciousness", "blackout", "fainted"],
    "edema":           ["edema", "oedema", "swollen legs", "ankle swelling", "peripheral edema", "pitting edema", "pedal edema"],
    "wheezing":        ["wheezing", "wheeze", "chest wheeze", "stridor", "wheezy"],
    "hemoptysis":      ["hemoptysis", "coughing blood", "blood in sputum", "bloody cough", "blood-tinged sputum"],
    "joint pain":      ["joint pain", "arthralgia", "arthritis", "joint swelling", "joint stiffness", "articular pain"],
    "back pain":       ["back pain", "backache", "lumbar pain", "spine pain", "low back pain", "flank pain", "loin pain"],
    "weakness":        ["weakness", "muscle weakness", "limb weakness", "hemiparesis", "hemiplegia", "paresis"],
    "confusion":       ["confusion", "confused", "disorientation", "altered mental status", "encephalopathy", "delirium"],
    "rash":            ["rash", "skin rash", "eruption", "erythema", "skin lesions", "urticaria", "hives"],
    "tachycardia":     ["tachycardia", "rapid heart rate", "fast heart rate", "heart racing", "hr >100", "hr>100"],
    "hypertension":    ["hypertension", "high blood pressure", "elevated bp", "htn", "bp >140", "bp>140"],
    "polyuria":        ["polyuria", "frequent urination", "excessive urination", "nocturia", "urinary frequency"],
    "dysuria":         ["dysuria", "painful urination", "burning urination", "burning on urination", "pain on urination"],
    "jaundice":        ["jaundice", "yellowing", "yellow skin", "yellow eyes", "icterus", "scleral icterus"],
    "spo2":            ["spo2", "oxygen saturation", "o2 sat", "sao2", "pulse ox", "saturation <95", "spo2 <95"],
}

ORGAN_ALIASES = {
    "lung":    ["lung", "pulmonary", "respiratory", "pleural", "bronch", "trachea", "lobe", "alveol", "pneumo"],
    "brain":   ["brain", "cerebral", "cranial", "neuro", "intracranial", "cortex", "cerebrum", "meninges", "cns"],
    "heart":   ["heart", "cardiac", "myocardial", "coronary", "pericardial", "atrial", "ventricular", "aortic", "troponin", "ecg", "ekg", "stemi", "nstemi", "st elevation", "st depression"],
    "kidney":  ["kidney", "renal", "nephro", "urinary", "glomerul", "tubular"],
    "liver":   ["liver", "hepatic", "hepato", "biliary", "gallbladder"],
    "bone":    ["bone", "skeletal", "femur", "humerus", "spine", "vertebra", "cortex", "periosteal", "orthopedic"],
    "skin":    ["skin", "dermal", "cutaneous", "dermis", "subcutaneous", "dermatology"],
    "eye":     ["eye", "ocular", "retina", "vision", "optic", "cornea", "intraocular"],
    "abdomen": ["abdomen", "abdominal", "gastrointestinal", "bowel", "intestine", "colon", "stomach", "gastric", "pancreas", "appendix"],
    "thyroid": ["thyroid", "thyro", "goiter", "thyroxine", "tsh"],
    "spine":   ["spine", "spinal", "vertebral", "disc", "lumbar", "cervical", "thoracic"],
    "bladder": ["bladder", "vesical", "cystitis", "urethra"],
    "prostate":["prostate", "prostatic"],
}

CONDITION_RULES = [
    # Respiratory
    {"label": "Pneumonia",              "organ": "lung",    "symptoms": ["fever", "cough", "dyspnea", "chest pain", "lesion"],    "reasoning": "Fever with productive cough, dyspnea, and pulmonary infiltrates is classic pneumonia."},
    {"label": "Bronchitis",             "organ": "lung",    "symptoms": ["cough", "fever", "chest pain"],                          "reasoning": "Lower respiratory tract irritation without consolidation suggests bronchitis."},
    {"label": "Asthma Exacerbation",    "organ": "lung",    "symptoms": ["wheezing", "dyspnea", "cough"],                          "reasoning": "Reversible expiratory wheeze with dyspnea suggests asthma."},
    {"label": "COPD Exacerbation",      "organ": "lung",    "symptoms": ["dyspnea", "cough", "wheezing", "fatigue"],               "reasoning": "Chronic obstructive pattern with acute worsening."},
    {"label": "Pulmonary Embolism",     "organ": "lung",    "symptoms": ["dyspnea", "chest pain", "tachycardia", "syncope"],       "reasoning": "Sudden cardiopulmonary symptoms raise pulmonary embolism concern."},
    {"label": "Tuberculosis",           "organ": "lung",    "symptoms": ["cough", "hemoptysis", "night sweats", "weight loss"],    "reasoning": "Constitutional TB triad with hemoptysis."},
    # Cardiac
    {"label": "Myocardial Infarction",  "organ": "heart",   "symptoms": ["chest pain", "dyspnea", "fatigue", "nausea"],           "reasoning": "Acute ischemic pattern with typical chest pain and systemic symptoms."},
    {"label": "Heart Failure",          "organ": "heart",   "symptoms": ["dyspnea", "edema", "fatigue", "wheezing"],              "reasoning": "Pulmonary congestion and peripheral edema indicate heart failure."},
    {"label": "Atrial Fibrillation",    "organ": "heart",   "symptoms": ["palpitations", "dyspnea", "fatigue", "syncope"],        "reasoning": "Irregular palpitations with hemodynamic symptoms suggest AF."},
    {"label": "Myocardial Ischemia",    "organ": "heart",   "symptoms": ["chest pain", "dyspnea", "fatigue"],                     "reasoning": "Cardiopulmonary symptoms raise concern for ischemic heart disease."},
    # Neurological
    {"label": "Migraine",               "organ": "brain",   "symptoms": ["headache", "blurred vision", "nausea", "vomiting"],     "reasoning": "Unilateral headache with visual aura and GI symptoms is migraine."},
    {"label": "Brain Tumor",            "organ": "brain",   "symptoms": ["headache", "seizure", "blurred vision", "weakness"],    "reasoning": "Neurologic red-flag triad: headache, seizure, focal deficit."},
    {"label": "Stroke",                 "organ": "brain",   "symptoms": ["weakness", "blurred vision", "headache", "confusion"],  "reasoning": "Acute focal neurological deficit with sudden onset suggests stroke."},
    {"label": "Meningitis",             "organ": "brain",   "symptoms": ["headache", "fever", "confusion", "seizure"],            "reasoning": "Fever with meningeal signs and neurological compromise."},
    {"label": "Epilepsy",               "organ": "brain",   "symptoms": ["seizure", "confusion", "weakness"],                     "reasoning": "Recurrent seizure pattern with post-ictal confusion."},
    # Musculoskeletal
    {"label": "Bone Fracture",          "organ": "bone",    "symptoms": ["fracture", "pain", "swelling"],                         "reasoning": "Traumatic musculoskeletal injury with cortical discontinuity."},
    {"label": "Osteoarthritis",         "organ": "bone",    "symptoms": ["joint pain", "pain", "swelling"],                       "reasoning": "Chronic degenerative joint disease with mechanical symptoms."},
    # Gastrointestinal
    {"label": "Appendicitis",           "organ": "abdomen", "symptoms": ["abdominal pain", "nausea", "vomiting", "fever"],        "reasoning": "RIF tenderness with systemic inflammation is classic appendicitis."},
    {"label": "Pancreatitis",           "organ": "abdomen", "symptoms": ["abdominal pain", "nausea", "vomiting", "fever"],        "reasoning": "Epigastric pain radiating to back with raised amylase/lipase."},
    {"label": "Cholecystitis",          "organ": "abdomen", "symptoms": ["abdominal pain", "nausea", "fever", "jaundice"],        "reasoning": "RUQ pain with systemic inflammation and biliary signs."},
    # Renal
    {"label": "Urinary Tract Infection","organ": "kidney",  "symptoms": ["dysuria", "fever", "pain", "polyuria"],                 "reasoning": "Urinary symptoms with systemic inflammation indicate UTI/pyelonephritis."},
    {"label": "Nephrolithiasis",        "organ": "kidney",  "symptoms": ["back pain", "pain", "dysuria", "nausea"],               "reasoning": "Severe colicky flank pain with urinary symptoms suggests kidney stones."},
    # Metabolic
    {"label": "Diabetes Mellitus",      "organ": "abdomen", "symptoms": ["polyuria", "fatigue", "weight loss", "blurred vision"], "reasoning": "Classic hyperglycemic presentation with osmotic and metabolic symptoms."},
    {"label": "Hypothyroidism",         "organ": "thyroid", "symptoms": ["fatigue", "weight loss", "edema", "weakness"],          "reasoning": "Metabolic slowdown with constitutional symptoms suggests hypothyroidism."},
]

ORGAN_FALLBACK_CONDITIONS = {
    "lung":    "Pulmonary pathology",
    "brain":   "Neurologic pathology",
    "heart":   "Cardiac pathology",
    "kidney":  "Renal pathology",
    "liver":   "Hepatic pathology",
    "bone":    "Musculoskeletal pathology",
    "skin":    "Dermatologic pathology",
    "eye":     "Ophthalmic pathology",
    "abdomen": "Abdominal pathology",
    "thyroid": "Thyroid pathology",
    "spine":   "Spinal pathology",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_age(text: str) -> int:
    for pattern in [r"\bage\s*[:\-]?\s*(\d{1,3})\b", r"\b(\d{1,3})\s*(?:years?|yrs?)\s*(?:old)?\b", r"\b(\d{1,3})\s*/\s*(?:M|F|male|female)\b"]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 0 < age <= 120:
                return age
    return 45


def _extract_sex(text: str) -> str:
    lower_text = text.lower()
    if re.search(r"\b(female|woman|girl|\bf\b|sex\s*:\s*f)\b", lower_text):
        return "F"
    if re.search(r"\b(male|man|boy|\bm\b|sex\s*:\s*m)\b", lower_text):
        return "M"
    return "Unknown"


def _extract_pseudonym(text: str) -> str:
    for pattern in [
        r"\bpatient\s*(?:name)?\s*[:\-]?\s*([A-Z][a-zA-Z\s]{1,30}?)(?:\s*\n|\s*,|\s*age|\s*\d)",
        r"\bname\s*[:\-]\s*([A-Z][a-zA-Z\s]{1,30}?)(?:\s*\n|\s*,|\s*age|\s*\d)",
        r"\bpt\s*[:\-]\s*([A-Z][a-zA-Z\s]{1,20}?)(?:\s*\n|\s*,|\s*\d)",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            token = match.group(1).strip()
            if token.lower() not in {"male", "female", "m", "f", "unknown"} and len(token) > 1:
                return f"Patient-{token.split()[0].capitalize()}"
    return "Patient-A"


def _extract_symptoms(text: str) -> list:
    symptoms, lower_text = [], text.lower()
    for canonical, aliases in SYMPTOM_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lower_text):
                symptoms.append(canonical)
                break
    deduped, seen = [], set()
    for s in symptoms:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped[:10]


def _infer_organ(text: str, symptoms: list) -> str:
    lower_text = text.lower()
    organ_scores: dict[str, int] = {}
    for organ, aliases in ORGAN_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lower_text):
                organ_scores[organ] = organ_scores.get(organ, 0) + 1
    if organ_scores:
        return max(organ_scores, key=lambda k: organ_scores[k])

    symptom_set = set(symptoms)
    if symptom_set & {"cough", "dyspnea", "chest pain", "fever", "wheezing", "hemoptysis", "spo2", "lesion"}:
        return "lung"
    if symptom_set & {"headache", "seizure", "blurred vision", "confusion", "weakness"}:
        return "brain"
    if symptom_set & {"chest pain", "palpitations", "edema", "tachycardia", "syncope"}:
        return "heart"
    if symptom_set & {"fracture", "joint pain", "back pain", "swelling", "pain"}:
        return "bone"
    if symptom_set & {"abdominal pain", "nausea", "vomiting", "jaundice", "diarrhea"}:
        return "abdomen"
    if symptom_set & {"dysuria", "polyuria", "back pain"}:
        return "kidney"
    return "unspecified"


def _infer_conditions(symptoms: list, organ: str) -> list:
    symptom_set = set(symptoms)
    candidates = []
    for rule in CONDITION_RULES:
        matched = [s for s in rule["symptoms"] if s in symptom_set]
        if not matched:
            continue
        support = len(matched) / max(len(rule["symptoms"]), 1)
        if organ != "unspecified" and organ == rule["organ"]:
            support += 0.15
        confidence = min(0.50 + (0.42 * support), 0.93)
        candidates.append({"label": rule["label"], "confidence": round(confidence, 3), "model": "Hybrid-NER", "reasoning": rule["reasoning"]})

    if not candidates:
        fallback_label = ORGAN_FALLBACK_CONDITIONS.get(organ, "Unspecified condition")
        candidates = [{"label": fallback_label, "confidence": 0.55 if organ != "unspecified" else 0.45, "model": "Hybrid-NER", "reasoning": "Heuristic inference from extracted context."}]

    deduped, seen = [], set()
    for cond in sorted(candidates, key=lambda x: x["confidence"], reverse=True):
        if cond["label"].lower() not in seen:
            deduped.append(cond)
            seen.add(cond["label"].lower())
    return deduped[:3]


def _build_text_features(symptoms: list, organ: str, conditions: list, ocr_text: str = "") -> list:
    features = []
    # High-value clinical indicators from OCR text
    vitals_patterns = [
        (r"spo2\s*[:\-]?\s*(\d+)\s*%", "SpO2 {}%"),
        (r"o2\s*sat\s*[:\-]?\s*(\d+)\s*%", "O2 Sat {}%"),
        (r"bp\s*[:\-]?\s*(\d+/\d+)", "BP {}"),
        (r"hr\s*[:\-]?\s*(\d+)", "HR {} bpm"),
        (r"temp\s*[:\-]?\s*([\d.]+)", "Temp {}°"),
        (r"rr\s*[:\-]?\s*(\d+)", "RR {} /min"),
    ]
    lower_ocr = ocr_text.lower()
    for pattern, label_tpl in vitals_patterns:
        match = re.search(pattern, lower_ocr)
        if match:
            features.append({"feature": label_tpl.format(match.group(1)), "weight": round(min(0.95, 0.75 + len(features) * 0.01), 3)})
    for i, symptom in enumerate(symptoms[:5]):
        features.append({"feature": symptom, "weight": round(max(0.40, 0.88 - (i * 0.08)), 3)})
    if organ and organ != "unspecified":
        features.append({"feature": f"organ:{organ}", "weight": 0.65})
    if conditions:
        features.append({"feature": f"dx:{conditions[0]['label']}", "weight": 0.60})
    seen, deduped = set(), []
    for f in features:
        if f["feature"] not in seen:
            deduped.append(f)
            seen.add(f["feature"])
    return deduped[:8]


def _heuristic_extract(ocr_text: str) -> dict:
    clean_text = _normalize_text(ocr_text)
    symptoms   = _extract_symptoms(clean_text)
    organ      = _infer_organ(clean_text, symptoms)
    conditions = _infer_conditions(symptoms, organ)
    return {
        "pseudonym":  _extract_pseudonym(clean_text),
        "age":        _extract_age(clean_text),
        "sex":        _extract_sex(clean_text),
        "symptoms":   symptoms or ["unspecified"],
        "organ":      organ,
        "conditions": conditions,
        "localization": {"bbox": [50, 50, 450, 450], "mask": [], "region_description": organ if organ != "unspecified" else "unknown region"},
        "xai": {"text_features": _build_text_features(symptoms, organ, conditions, ocr_text), "shap_table": []},
        "overall_confidence": conditions[0]["confidence"] if conditions else 0.5,
    }


def _build_ner_prompt(ocr_text: str) -> str:
    clipped = _normalize_text(ocr_text)[:3000]
    return f"""You are a board-certified physician AI. Extract ALL clinical entities from the medical text below.

Return ONLY valid JSON matching this schema exactly:
{{
  "pseudonym": "Patient-X",
  "age": 52,
  "sex": "M",
  "symptoms": ["fever 39.1C", "productive cough", "dyspnea", "SpO2 91%", "chest pain"],
  "organ": "lung",
  "conditions": [
    {{
      "label": "Community-Acquired Pneumonia",
      "confidence": 0.88,
      "model": "MedGemma-NER",
      "reasoning": "Fever, productive cough, SpO2 91%, and right lower lobe opacity are classical pneumonia."
    }},
    {{
      "label": "Pulmonary Embolism",
      "confidence": 0.38,
      "model": "MedGemma-NER",
      "reasoning": "Dyspnea and chest pain warrant PE on differential, but infectious picture is dominant."
    }}
  ],
  "localization": {{
    "bbox": [80, 120, 420, 390],
    "mask": [],
    "region_description": "Right lower lobe consolidation"
  }},
  "xai": {{
    "text_features": [
      {{"feature": "SpO2 91%", "weight": 0.93}},
      {{"feature": "fever 39.1C", "weight": 0.85}},
      {{"feature": "productive cough", "weight": 0.82}},
      {{"feature": "right lower lobe opacity", "weight": 0.88}},
      {{"feature": "organ:lung", "weight": 0.72}}
    ],
    "shap_table": []
  }},
  "overall_confidence": 0.88
}}

Extraction rules — follow precisely:
1. symptoms: include SPECIFIC values when mentioned (e.g. "fever 39.2C", "SpO2 88%", "BP 180/110", "HR 118") — do NOT just write "fever"
2. organ: single most-affected body system from: lung, brain, heart, kidney, liver, bone, skin, eye, abdomen, thyroid, spine
3. conditions: list 2-3 in differential order; most likely FIRST; include specific medical labels (not generic)
4. confidence: 0.85-0.95 = very strong evidence; 0.65-0.84 = moderate; 0.40-0.64 = possible; <0.40 = unlikely
5. reasoning: 1-2 sentences citing specific findings from the text that support/refute each condition
6. text_features: the specific OCR terms / vitals / imaging findings most diagnostic — higher weight = more specific to diagnosis
7. localization bbox: estimate [x1,y1,x2,y2] for the key anatomical region in a 512x512 image
8. Return ONLY the JSON — no preamble, no explanation, no markdown fences

MEDICAL TEXT:
{clipped}"""


def _extract_json_object(raw_text: str):
    if not raw_text:
        return None
    candidates = []
    for fence_match in re.finditer(r"```(?:json)?\s*(\{{.*?\}})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(fence_match.group(1))
    for start in [m.start() for m in re.finditer(r"\{", raw_text)]:
        depth = 0
        for idx in range(start, len(raw_text)):
            if raw_text[idx] == "{":
                depth += 1
            elif raw_text[idx] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw_text[start:idx + 1])
                    break
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except Exception:
            continue
    return None


def _normalize_result(model_result: dict, fallback: dict) -> dict:
    merged = json.loads(json.dumps(fallback))
    if not isinstance(model_result, dict):
        return merged

    if isinstance(model_result.get("pseudonym"), str) and model_result["pseudonym"].strip():
        merged["pseudonym"] = model_result["pseudonym"].strip()

    age = model_result.get("age")
    if isinstance(age, str) and age.isdigit():
        age = int(age)
    if isinstance(age, (int, float)) and 0 < int(age) <= 120:
        merged["age"] = int(age)

    sex = str(model_result.get("sex", "")).strip().upper()
    if sex in {"M", "MALE"}:
        merged["sex"] = "M"
    elif sex in {"F", "FEMALE"}:
        merged["sex"] = "F"

    symptoms = model_result.get("symptoms")
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    if isinstance(symptoms, list):
        cleaned = [str(s).strip() for s in symptoms if str(s).strip()]
        if cleaned:
            merged["symptoms"] = cleaned[:10]

    if isinstance(model_result.get("organ"), str) and model_result["organ"].strip():
        merged["organ"] = model_result["organ"].strip().lower()

    conditions = model_result.get("conditions")
    if isinstance(conditions, list):
        parsed = []
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            label = str(cond.get("label", "")).strip()
            if not label:
                continue
            try:
                conf = float(cond.get("confidence", 0.55))
            except Exception:
                conf = 0.55
            parsed.append({"label": label, "confidence": round(min(max(conf, 0.0), 1.0), 3), "model": str(cond.get("model") or "MedGemma-NER"), "reasoning": str(cond.get("reasoning") or "Clinical inference from OCR text.")})
        if parsed:
            parsed.sort(key=lambda x: x["confidence"], reverse=True)
            merged["conditions"] = parsed[:3]

    localization = model_result.get("localization")
    if isinstance(localization, dict):
        bbox = localization.get("bbox", merged["localization"]["bbox"])
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                merged["localization"]["bbox"] = [int(float(v)) for v in bbox]
            except Exception:
                pass
        if isinstance(localization.get("region_description"), str) and localization["region_description"].strip():
            merged["localization"]["region_description"] = localization["region_description"].strip()

    xai = model_result.get("xai")
    if isinstance(xai, dict):
        feats = xai.get("text_features")
        if isinstance(feats, list):
            parsed_feats = []
            for feat in feats:
                if not isinstance(feat, dict):
                    continue
                name = str(feat.get("feature", "")).strip()
                if not name:
                    continue
                try:
                    weight = float(feat.get("weight", 0.5))
                except Exception:
                    weight = 0.5
                parsed_feats.append({"feature": name, "weight": round(min(max(weight, 0.0), 1.0), 3)})
            if parsed_feats:
                merged["xai"]["text_features"] = parsed_feats[:8]
        if isinstance(xai.get("shap_table"), list):
            merged["xai"]["shap_table"] = xai["shap_table"]

    try:
        oc = float(model_result.get("overall_confidence"))
        merged["overall_confidence"] = round(min(max(oc, 0.0), 1.0), 3)
    except Exception:
        if merged.get("conditions"):
            merged["overall_confidence"] = merged["conditions"][0]["confidence"]

    if not merged.get("xai", {}).get("text_features"):
        merged["xai"]["text_features"] = _build_text_features(merged.get("symptoms", []), merged.get("organ", "unspecified"), merged.get("conditions", []))
    if not merged.get("conditions"):
        merged["conditions"] = _infer_conditions(merged.get("symptoms", []), merged.get("organ", "unspecified"))
    if not merged.get("overall_confidence") and merged.get("conditions"):
        merged["overall_confidence"] = merged["conditions"][0]["confidence"]

    return merged


def run_ner(ocr_text: str, image: Image.Image = None) -> dict:
    fallback_result = _heuristic_extract(ocr_text)
    prompt = _build_ner_prompt(ocr_text)

    print("  [MedGemma NER] Sending to Ollama medgemma:4b...")
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": NER_MODEL, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  [MedGemma NER] WARNING: Ollama unavailable — using heuristic fallback")
        return fallback_result
    except Exception as e:
        print(f"  [MedGemma NER] WARNING: {e} — using heuristic fallback")
        return fallback_result

    print(f"  [MedGemma NER] Response: {len(raw)} chars | preview: {raw[:200]}")
    model_result = _extract_json_object(raw)
    if model_result is None:
        print("  [MedGemma NER] No valid JSON — using heuristic fallback")
        return fallback_result

    result = _normalize_result(model_result, fallback_result)
    print(f"  [MedGemma NER] organ={result.get('organ')}, "
          f"conditions={[c['label'] for c in result.get('conditions', [])]}")
    return result
