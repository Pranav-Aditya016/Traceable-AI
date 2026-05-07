"""
Rich Medical Image Prompt Builder
=================================
Generates detailed, varied prompts for SDXL medical image generation.
Uses diagnosis, modality, AND NER entities to create specific clinical prompts.
Each call produces a DIFFERENT prompt through randomized selection.
"""
import random

# ── Modality base templates (randomly selected per call) ─────────────────

MODALITY_BASES = {
    "chest_xray": [
        (
            "a professional posteroanterior chest radiograph displayed on a PACS lightbox viewer, "
            "grayscale medical X-ray film, ribs and cardiac silhouette clearly visible, "
            "lung fields with vascular markings, costophrenic angles sharp"
        ),
        (
            "a clinical anteroposterior chest X-ray on radiology workstation, "
            "high-contrast grayscale radiological scan, thoracic anatomy visible, "
            "mediastinum and lung parenchyma well delineated"
        ),
        (
            "a diagnostic quality PA chest radiograph, monochrome X-ray image, "
            "clear ribcage structure, bilateral lung fields visible, "
            "trachea midline, medical imaging DICOM display"
        ),
        (
            "a standard upright PA chest X-ray on digital detector, "
            "grayscale thoracic radiograph, pulmonary vasculature visible, "
            "well-penetrated film, adequate inspiration"
        ),
    ],
    "brain_mri": [
        (
            "a professional brain MRI T1-weighted axial cross-section, "
            "grayscale neuroimaging scan, brain tissue anatomy visible, "
            "ventricles and sulci delineated, clinical radiology scan"
        ),
        (
            "a clinical brain MRI T2-weighted axial slice, "
            "grayscale magnetic resonance image, cerebral hemispheres visible, "
            "high resolution neurological scan on PACS viewer"
        ),
        (
            "a diagnostic brain MRI FLAIR sequence axial image, "
            "grayscale cross-sectional neuroimaging, brain parenchyma visible, "
            "gray-white matter differentiation, clinical MRI scan"
        ),
        (
            "a gadolinium-enhanced brain MRI T1 post-contrast axial slice, "
            "grayscale neuroimaging, cortical and subcortical structures visible, "
            "basal ganglia and thalami delineated"
        ),
    ],
    "dermoscopy": [
        (
            "a clinical dermoscopy photograph, magnified polarized dermatoscope view, "
            "skin lesion close-up with dermoscopic structures visible, "
            "medical dermatology imaging, sharp color photograph"
        ),
        (
            "a professional dermatoscopic image, high magnification skin surface microscopy, "
            "pigment patterns and vascular structures visible, "
            "clinical dermatology photography, 10x magnification"
        ),
        (
            "a contact dermoscopy image of a skin lesion, "
            "immersion fluid interface visible, structural patterns clear, "
            "dermatological close-up photography, clinical grade"
        ),
    ],
    "fundus": [
        (
            "a medical retinal fundus photograph, wide-angle ophthalmoscopy image, "
            "optic disc and macula visible, retinal vessels clearly delineated, "
            "clinical ophthalmology imaging, orange-red retinal background"
        ),
        (
            "a professional fundus camera photograph of the retina, "
            "optic nerve head and peripapillary area visible, "
            "retinal vasculature network, color fundus photography"
        ),
        (
            "a high-resolution digital fundus photograph, "
            "fovea centralis and optic cup visible, retinal arterioles and venules, "
            "posterior pole of the eye, clinical ophthalmic imaging"
        ),
    ],
    "chest_ct": [
        (
            "a professional chest CT scan axial slice on radiology workstation, "
            "grayscale computed tomography image, lung window setting, "
            "pulmonary vasculature and airways visible, DICOM display"
        ),
        (
            "a clinical high-resolution chest CT axial cross-section, "
            "grayscale computed tomography, mediastinal and lung structures visible, "
            "diagnostic quality thin-slice CT imaging"
        ),
        (
            "a chest CT scan in soft tissue window, axial plane, "
            "grayscale computed tomography, bronchial tree and great vessels visible, "
            "clinical radiology HRCT scan"
        ),
    ],
    "breast_mri": [
        (
            "a clinical breast MRI contrast-enhanced axial slice, "
            "grayscale magnetic resonance image of breast tissue, "
            "glandular and fibrous structures visible, dynamic contrast imaging"
        ),
        (
            "a breast MRI DCE sequence axial image, "
            "grayscale radiological scan showing breast parenchyma, "
            "fibroglandular tissue with enhancement, clinical radiology"
        ),
    ],
    "kidney_ct": [
        (
            "a clinical abdominal CT scan axial slice focused on renal structures, "
            "grayscale computed tomography, kidneys and collecting system visible, "
            "high contrast resolution CT imaging on PACS viewer"
        ),
        (
            "a contrast-enhanced CT scan of the abdomen at renal level, "
            "grayscale computed tomography, bilateral kidneys in corticomedullary phase, "
            "renal parenchyma and pelvis visible, diagnostic CT"
        ),
    ],
    "abdominal_ct": [
        (
            "a professional abdominal CT scan axial cross-section, "
            "grayscale computed tomography image, abdominal organs visible, "
            "bowel loops and retroperitoneal structures, diagnostic quality CT"
        ),
        (
            "a contrast-enhanced abdominal CT axial slice, "
            "grayscale computed tomography, liver and mesenteric structures visible, "
            "peritoneal cavity well delineated, clinical MDCT scan"
        ),
    ],
}


# ── Disease-specific imaging findings (randomly selected) ────────────────

DISEASE_FINDINGS = {
    "pneumonia": {
        "chest_xray": [
            "showing bilateral patchy alveolar infiltrates with air bronchograms in the lower lobes",
            "demonstrating right lower lobe consolidation with meniscus sign and blunted costophrenic angle",
            "revealing diffuse ground-glass opacities bilaterally consistent with atypical pneumonia pattern",
            "showing focal area of consolidation with silhouette sign obscuring the right heart border",
            "demonstrating bilateral perihilar opacities with air space disease in the mid and lower zones",
            "showing left lower lobe opacification with visible air bronchograms and small pleural effusion",
        ],
    },
    "tuberculosis": {
        "chest_xray": [
            "showing upper lobe cavitary lesion with thick irregular walls and surrounding fibrotic changes",
            "demonstrating miliary pattern with numerous tiny nodular opacities diffusely throughout both lung fields",
            "revealing right upper lobe consolidation with calcified hilar lymphadenopathy and Ghon lesion",
            "showing bilateral apical fibrocavitary disease with volume loss and tracheal deviation",
            "demonstrating tree-in-bud opacities in the right upper lobe with satellite nodules",
        ],
    },
    "glioma": {
        "brain_mri": [
            "showing heterogeneous enhancing mass in the right temporal lobe with surrounding vasogenic edema and mass effect",
            "demonstrating large ring-enhancing lesion in the left frontal lobe with central necrosis and midline shift",
            "revealing diffuse infiltrating tumor in the corpus callosum crossing to both hemispheres butterfly pattern",
            "showing irregular enhancing mass with central necrosis in the right parietal lobe compressing the lateral ventricle",
            "demonstrating high-grade glioma with heterogeneous enhancement and perilesional T2 hyperintensity",
        ],
    },
    "melanoma": {
        "dermoscopy": [
            "showing asymmetric pigmented lesion with irregular border and multiple colors including brown black and blue-white structures",
            "demonstrating atypical pigment network with regression structures blue-white veil and irregular globules",
            "revealing multicomponent lesion with pseudopods streaks and off-center blotch of dark pigmentation",
            "showing polymorphous vascular pattern with irregular dots and structureless blue-white areas suggestive of invasion",
        ],
    },
    "diabetic retinopathy": {
        "fundus": [
            "showing scattered dot and blot hemorrhages with hard exudates in circinate pattern surrounding the macula",
            "demonstrating numerous microaneurysms with cotton-wool spots venous beading and IRMA in all four quadrants",
            "revealing proliferative neovascularization at the optic disc with preretinal hemorrhage and fibrovascular proliferation",
            "showing moderate non-proliferative changes with multiple flame-shaped hemorrhages and macular edema",
        ],
    },
    "myocardial infarction": {
        "chest_xray": [
            "showing cardiomegaly with increased cardiothoracic ratio pulmonary vascular congestion and bilateral pleural effusions",
            "demonstrating enlarged cardiac silhouette with upper lobe pulmonary venous distension and Kerley B lines",
            "revealing cephalization of pulmonary vessels with bilateral interstitial edema pattern and bat-wing opacities",
            "showing prominent cardiac silhouette with bilateral alveolar edema and perihilar haziness",
        ],
    },
    "lung cancer": {
        "chest_ct": [
            "showing a spiculated pulmonary nodule in the right upper lobe with mediastinal lymphadenopathy",
            "demonstrating large central mass encasing the left main bronchus with post-obstructive atelectasis",
            "revealing peripheral solid mass with irregular margins and pleural tethering in the left lower lobe",
            "showing heterogeneous enhancing mass in the right hilum with satellite nodules and pleural effusion",
        ],
    },
    "breast cancer": {
        "breast_mri": [
            "showing irregular enhancing mass with spiculated margins in the upper outer quadrant of the left breast",
            "demonstrating heterogeneous enhancement with washout kinetics and architectural distortion suggesting malignancy",
            "revealing multicentric enhancing masses with suspicious morphology and plateau enhancement curve",
        ],
    },
    "kidney stone": {
        "kidney_ct": [
            "showing hyperdense calculus in the right ureteropelvic junction with mild hydronephrosis and perinephric stranding",
            "demonstrating staghorn calculus filling the left renal pelvis and multiple calyces with cortical thinning",
            "revealing obstructing calculus in the distal left ureter with proximal ureteral dilatation and hydroureteronephrosis",
        ],
    },
    "appendicitis": {
        "abdominal_ct": [
            "showing dilated fluid-filled appendix with periappendiceal fat stranding wall thickening and enhancement",
            "demonstrating appendicolith with surrounding inflammatory changes and reactive lymph nodes in the right lower quadrant",
            "revealing enlarged appendix measuring over 10mm with periappendiceal abscess and extraluminal gas",
        ],
    },
    "asthma": {
        "chest_xray": [
            "showing hyperinflated lungs with flattened diaphragms increased retrosternal airspace and normal heart size",
            "demonstrating bilateral hyperinflation with peribronchial wall thickening and increased AP diameter",
            "revealing hyperexpanded lung fields with flattened hemidiaphragms subtle peribronchial cuffing",
        ],
    },
    "urinary tract infection": {
        "abdominal_ct": [
            "showing perinephric stranding with thickened urothelium and mild hydronephrosis suggestive of pyelonephritis",
            "demonstrating renal cortical enhancement with focal areas of hypoenhancement and urothelial thickening",
        ],
    },
}


# ── Quality/style modifiers (randomly selected for variation) ────────────

QUALITY_MODIFIERS = [
    "clinical diagnostic quality, sharp anatomical detail, high contrast imaging",
    "professional radiology scan, precise medical imaging, crisp resolution",
    "high resolution medical scan, detailed anatomical structures, expert quality",
    "diagnostic reference quality, excellent visualization, professional grade imaging",
    "medical textbook quality, superb structural detail, clinical accuracy",
    "research quality medical imaging, exceptional detail and resolution",
]


# ── Modality-specific negative prompts ───────────────────────────────────

MODALITY_NEGATIVES = {
    "chest_xray": (
        "color photograph, colored image, RGB photo, photograph of a person, "
        "human body skin, nude, nsfw, face, portrait, selfie, "
        "blurry, low quality, distorted, artifacts, noise, jpeg compression, "
        "text overlay, watermark, logo, labels, timestamp, annotation, "
        "cartoon, illustration, drawing, painting, anime, comic book, 3d render, CGI, "
        "natural photo, landscape, food, animal, colorful, saturated colors, neon, "
        "deformed structures, extra anatomy, bad medical image, unrealistic"
    ),
    "brain_mri": (
        "color photograph, human skin, face, portrait, nude, "
        "blurry, low quality, distorted, artifacts, noise, "
        "text overlay, watermark, logo, cartoon, illustration, painting, "
        "anatomically incorrect brain, extra structures, deformed, "
        "natural photo, landscape, colorful, saturated, X-ray, ultrasound"
    ),
    "dermoscopy": (
        "blurry, low quality, distorted, artifacts, noise, out of focus, "
        "text overlay, watermark, logo, cartoon, illustration, painting, "
        "landscape, non-medical, X-ray, CT scan, MRI scan, "
        "anatomically incorrect, deformed skin, unrealistic texture, "
        "full body photo, face, portrait"
    ),
    "fundus": (
        "blurry, low quality, distorted, artifacts, noise, "
        "text overlay, watermark, logo, cartoon, illustration, painting, "
        "landscape, non-medical, X-ray, CT scan, MRI scan, "
        "anatomically incorrect eye, deformed retina, "
        "face, portrait, full body"
    ),
    "chest_ct": (
        "color photograph, human skin, face, portrait, nude, "
        "blurry, low quality, distorted, artifacts, noise, "
        "text overlay, watermark, logo, cartoon, illustration, painting, "
        "natural photo, landscape, colorful, saturated, "
        "anatomically incorrect, extra structures, flat X-ray film"
    ),
    "breast_mri": (
        "color photograph, human skin, face, portrait, nude body, nsfw, "
        "blurry, low quality, distorted, artifacts, "
        "text overlay, watermark, logo, cartoon, illustration, "
        "non-medical, landscape, deformed, unrealistic anatomy"
    ),
    "kidney_ct": (
        "color photograph, human skin, face, portrait, "
        "blurry, low quality, distorted, artifacts, "
        "text overlay, watermark, logo, cartoon, illustration, "
        "non-medical, landscape, colorful, flat X-ray film"
    ),
    "abdominal_ct": (
        "color photograph, human skin, face, portrait, "
        "blurry, low quality, distorted, artifacts, "
        "text overlay, watermark, logo, cartoon, illustration, "
        "non-medical, landscape, colorful, flat X-ray film"
    ),
}

DEFAULT_NEGATIVE = (
    "photograph of a person, human body, skin, nude, nsfw, face, portrait, "
    "blurry, low quality, distorted, text overlay, watermark, "
    "cartoon, illustration, drawing, painting, anime, 3d render, "
    "natural photo, landscape, colorful, deformed, unrealistic"
)


def build_prompt(hypothesis_result: dict, entities: dict = None) -> dict:
    """
    Build rich, varied text-to-image prompts from diagnosis + NER entities.

    Returns dict with 'prompt' and 'negative_prompt' keys.
    Each call produces a DIFFERENT prompt through randomized selection of:
      - modality description base
      - disease-specific finding
      - quality modifier
      - clinical entity details
    """
    disease = hypothesis_result["top_disease"]
    modality = hypothesis_result["modality"]

    # 1. Pick a random modality base description
    bases = MODALITY_BASES.get(modality, MODALITY_BASES.get("chest_xray", [""]))
    base = random.choice(bases)

    # 2. Pick disease-specific findings
    disease_pool = DISEASE_FINDINGS.get(disease, {}).get(modality, [])
    if disease_pool:
        finding = random.choice(disease_pool)
    else:
        # Fallback: construct a generic finding description
        finding = f"showing imaging findings consistent with {disease}"

    # 3. Extract clinical details from NER entities for prompt enrichment
    entity_details = ""
    if entities:
        parts = []
        symptoms = [e["text"] for e in entities.get("symptoms", []) if isinstance(e, dict)]
        if symptoms:
            selected = random.sample(symptoms, min(3, len(symptoms)))
            parts.append(f"clinical findings of {', '.join(selected)}")
        conditions = [e["text"] for e in entities.get("diseases", []) if isinstance(e, dict)]
        if conditions:
            selected = random.sample(conditions, min(2, len(conditions)))
            parts.append(f"suggestive of {', '.join(selected)}")
        medications = [e["text"] for e in entities.get("medications", []) if isinstance(e, dict)]
        if medications:
            parts.append(f"patient on {medications[0]}")
        if parts:
            entity_details = ", " + ", ".join(parts)

    # 4. Random quality modifier
    quality = random.choice(QUALITY_MODIFIERS)

    # 5. Compose final prompt
    prompt = f"{base}, {finding}{entity_details}, {quality}"

    # 6. Modality-specific negative prompt
    negative = MODALITY_NEGATIVES.get(modality, DEFAULT_NEGATIVE)

    return {
        "prompt": prompt,
        "negative_prompt": negative,
    }


# ── Organ → modality mapping (for structured analyzer output) ────────────

ORGAN_TO_MODALITY = {
    "lungs": "chest_xray",
    "lung": "chest_xray",
    "chest": "chest_xray",
    "respiratory": "chest_xray",
    "pulmonary": "chest_xray",
    "brain": "brain_mri",
    "neurological": "brain_mri",
    "nervous system": "brain_mri",
    "skin": "dermoscopy",
    "dermatological": "dermoscopy",
    "eye": "fundus",
    "retina": "fundus",
    "ophthalmic": "fundus",
    "heart": "chest_xray",
    "cardiac": "chest_xray",
    "cardiovascular": "chest_xray",
    "kidney": "kidney_ct",
    "renal": "kidney_ct",
    "urinary": "kidney_ct",
    "breast": "breast_mri",
    "abdomen": "abdominal_ct",
    "abdominal": "abdominal_ct",
    "gastrointestinal": "abdominal_ct",
    "appendix": "abdominal_ct",
    "liver": "abdominal_ct",
}

# ── Disease name → modality (for common conditions) ──────────────────────

CONDITION_TO_MODALITY = {
    "pneumonia": "chest_xray",
    "tuberculosis": "chest_xray",
    "tb": "chest_xray",
    "asthma": "chest_xray",
    "copd": "chest_xray",
    "bronchitis": "chest_xray",
    "pleural effusion": "chest_xray",
    "lung cancer": "chest_ct",
    "pulmonary embolism": "chest_ct",
    "glioma": "brain_mri",
    "brain tumor": "brain_mri",
    "meningioma": "brain_mri",
    "stroke": "brain_mri",
    "alzheimer": "brain_mri",
    "melanoma": "dermoscopy",
    "skin cancer": "dermoscopy",
    "diabetic retinopathy": "fundus",
    "glaucoma": "fundus",
    "macular degeneration": "fundus",
    "myocardial infarction": "chest_xray",
    "heart failure": "chest_xray",
    "breast cancer": "breast_mri",
    "kidney stone": "kidney_ct",
    "renal calculus": "kidney_ct",
    "appendicitis": "abdominal_ct",
    "urinary tract infection": "abdominal_ct",
}


def _resolve_modality(organ: str, conditions: list) -> str:
    """Determine imaging modality from organ and condition labels."""
    # Try condition first (more specific)
    for cond in conditions:
        label = cond.get("label", "").lower().strip()
        for key, mod in CONDITION_TO_MODALITY.items():
            if key in label or label in key:
                return mod

    # Then try organ
    organ_lower = organ.lower().strip()
    for key, mod in ORGAN_TO_MODALITY.items():
        if key in organ_lower or organ_lower in key:
            return mod

    return "chest_xray"  # default


def build_prompt_from_analysis(analysis: dict) -> dict:
    """
    Build rich image generation prompt from structured analyzer output.
    This is the NEW primary function matching the Gemini prototype workflow.

    Prompt template (mirrors Gemini prototype):
      "Create a highly detailed, medically accurate synthetic image of a CT scan
       of a {organ}. The patient is a {age}-year-old {sex} presenting with
       symptoms of {symptoms}. The image must clearly show a well-defined
       {condition}. Focus on realistic tissue texture and anatomical accuracy."

    Args:
        analysis: dict from structured_analyzer with keys:
            - fields: {pseudonym, age, sex, symptoms}
            - predictions: {organ, conditions: [{label, confidence}]}

    Returns:
        dict with 'prompt', 'negative_prompt', 'modality'
    """
    fields = analysis.get("fields", {})
    predictions = analysis.get("predictions", {})

    organ = predictions.get("organ", "lungs")
    conditions = predictions.get("conditions", [])
    age = fields.get("age", 0)
    sex = fields.get("sex", "Unknown")
    symptoms = fields.get("symptoms", [])

    # Primary condition
    primary_condition = conditions[0]["label"] if conditions else "unknown"
    confidence = conditions[0].get("confidence", 0) if conditions else 0

    # Resolve modality
    modality = _resolve_modality(organ, conditions)

    # Pick base template for this modality
    bases = MODALITY_BASES.get(modality, MODALITY_BASES.get("chest_xray", [""]))
    base = random.choice(bases)

    # Pick disease-specific finding if available
    disease_key = primary_condition.lower().strip()
    disease_pool = DISEASE_FINDINGS.get(disease_key, {}).get(modality, [])
    # Also try partial matching
    if not disease_pool:
        for dk, findings_by_mod in DISEASE_FINDINGS.items():
            if dk in disease_key or disease_key in dk:
                disease_pool = findings_by_mod.get(modality, [])
                if disease_pool:
                    break

    if disease_pool:
        finding = random.choice(disease_pool)
    else:
        finding = f"showing imaging findings consistent with {primary_condition}"

    # Build symptom description
    symptom_str = ""
    if symptoms:
        selected = random.sample(symptoms, min(4, len(symptoms)))
        symptom_str = f", patient presenting with {', '.join(selected)}"

    # Patient demographics
    demo_str = ""
    if age and age > 0:
        demo_str += f", {age}-year-old"
    if sex and sex != "Unknown":
        sex_word = "male" if sex.upper() == "M" else "female" if sex.upper() == "F" else sex
        demo_str += f" {sex_word} patient"
    elif demo_str:
        demo_str += " patient"

    # Quality modifier
    quality = random.choice(QUALITY_MODIFIERS)

    # Compose prompt
    prompt = f"{base}, {finding}{demo_str}{symptom_str}, {quality}"

    # Negative prompt
    negative = MODALITY_NEGATIVES.get(modality, DEFAULT_NEGATIVE)

    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "modality": modality,
        "primary_condition": primary_condition,
    }
