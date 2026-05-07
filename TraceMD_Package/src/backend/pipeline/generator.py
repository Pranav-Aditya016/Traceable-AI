"""
Stage 4 — Synthetic Medical Image Generation.
Uses stable-diffusion-v1-5 in fp16.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from config import (SD_MODEL_ID, SD_STEPS, SD_GUIDANCE_SCALE,
                    SD_WIDTH, SD_HEIGHT, DEVICE, print_vram_usage,
                    resolve_model_source, SYNTHETIC_WATERMARK)

# Modality-specific prompt prefixes — forces grayscale radiograph aesthetic
MODALITY_PROMPTS = {
    "lung":    "grayscale chest X-ray radiograph, PA view, high contrast, DICOM style",
    "chest":   "grayscale chest CT scan, axial slice, lung window, DICOM style",
    "brain":   "grayscale brain MRI, axial T2-weighted, high resolution, DICOM style",
    "skull":   "grayscale skull CT scan, axial bone window, DICOM style",
    "spine":   "grayscale spinal MRI, sagittal T2-weighted, DICOM style",
    "kidney":  "grayscale abdominal CT scan, axial slice, soft tissue window, DICOM style",
    "liver":   "grayscale abdominal CT scan, hepatic phase, DICOM style",
    "heart":   "grayscale cardiac MRI, 4-chamber view, DICOM style",
    "bone":    "grayscale plain X-ray radiograph, bone detail, high contrast",
    "fracture":"grayscale plain X-ray showing cortical discontinuity, bone detail",
    "knee":    "grayscale knee MRI, sagittal view, DICOM style",
    "hip":     "grayscale hip X-ray radiograph, AP view, bone detail",
    "femur":   "grayscale femur X-ray radiograph, AP view, bone detail",
    "skin":    "clinical dermoscopic photograph, polarized light, high detail",
    "eye":     "grayscale fundus retinal photograph, optic disc visible",
    "breast":  "grayscale mammography scan, MLO view, high contrast",
    "abdomen": "grayscale abdominal CT scan, axial slice, soft tissue window, DICOM style",
}

ORGAN_FALLBACK_CONDITION = {
    "lung":    "patchy pulmonary consolidation with air bronchograms",
    "brain":   "focal hypointense intracranial lesion with surrounding edema",
    "heart":   "regional wall motion abnormality",
    "kidney":  "cortical thinning with loss of corticomedullary differentiation",
    "liver":   "heterogeneous hepatic parenchyma with focal hypoattenuating lesion",
    "bone":    "cortical discontinuity with periosteal reaction",
    "skin":    "irregular asymmetric pigmented lesion with indistinct borders",
    "eye":     "retinal vascular tortuosity with disc pallor",
    "abdomen": "mesenteric fat stranding with adjacent soft tissue thickening",
}


def get_modality_prompt(organ: str) -> str:
    organ_lower = organ.lower().strip()
    if organ_lower in MODALITY_PROMPTS:
        return MODALITY_PROMPTS[organ_lower]
    for key in MODALITY_PROMPTS:
        if key in organ_lower or organ_lower in key:
            return MODALITY_PROMPTS[key]
    return f"grayscale medical diagnostic scan of {organ}, DICOM style, high contrast"


def _sanitize_profile(organ, condition, age, sex, symptoms):
    organ = str(organ or "chest").strip().lower()
    if not organ or organ == "unspecified":
        organ = "chest"

    condition = str(condition or "").strip()
    if not condition or condition.lower() in {"unspecified condition", "unknown"}:
        condition = ORGAN_FALLBACK_CONDITION.get(organ, "focal medical abnormality")

    try:
        age = int(age)
    except Exception:
        age = 45
    if age <= 0 or age > 110:
        age = 45

    sex = str(sex or "Unknown").strip().upper()
    if sex not in {"M", "F"}:
        sex = "Unknown"

    if not isinstance(symptoms, list):
        symptoms = []
    symptoms = [str(s).strip().lower() for s in symptoms if str(s).strip()]
    return organ, condition, age, sex, symptoms[:4]


def build_prompt(organ, condition, age, sex, symptoms):
    modality   = get_modality_prompt(organ)
    symptom_str = ", ".join(symptoms) if symptoms else "clinically significant findings"
    sex_str     = "male" if sex == "M" else "female" if sex == "F" else "adult"
    return (
        f"{modality}, "
        f"showing {condition}, "
        f"{age}-year-old {sex_str} patient, "
        f"clinical presentation: {symptom_str}, "
        "anatomically accurate pathology, professional radiology quality, "
        "monochrome, single image, no annotations, no labels, no text overlay, no watermark"
    )


NEGATIVE_PROMPT = (
    "color, RGB, saturated, colorful, hue, tint, "
    "cartoon, illustration, painting, drawing, anime, sketch, "
    "watermark, text, label, annotation, arrows, logo, banner, "
    "low quality, blurry, distorted, pixelated, overexposed, underexposed, "
    "non-medical, fantasy, portrait, face, clipart, collage, "
    "multiple panels, split image, side by side"
)


def add_watermark(image: Image.Image) -> Image.Image:
    width, height = image.size
    band_top = max(height - 24, 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, band_top), (width, height)], fill=(0, 0, 0, 180))
    draw.text((6, band_top + 4), SYNTHETIC_WATERMARK, fill=(255, 255, 255))
    return image


def _enhance_output(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageEnhance.Sharpness(image).enhance(1.2)
    # Convert to grayscale and back to RGB to ensure monochrome look
    image = image.convert("L").convert("RGB")
    return image


def run_generator(ner_result: dict, seed: int = 42) -> str:
    organ      = ner_result.get("organ", "chest")
    conditions = ner_result.get("conditions", [])
    condition  = conditions[0]["label"] if conditions else "unspecified condition"
    age        = ner_result.get("age", 45)
    sex        = ner_result.get("sex", "Unknown")
    symptoms   = ner_result.get("symptoms", [])

    organ, condition, age, sex, symptoms = _sanitize_profile(organ, condition, age, sex, symptoms)

    prompt       = build_prompt(organ, condition, age, sex, symptoms)
    model_source = resolve_model_source(SD_MODEL_ID)
    model_dtype  = torch.float16 if DEVICE == "cuda" else torch.float32

    print(f"  [Generator] Prompt: {prompt[:140]}...")
    print(f"  [Generator] Loading from: {model_source}")
    print_vram_usage("Before Generator")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_source,
            torch_dtype=model_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(DEVICE)
        if DEVICE == "cuda":
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()
    except Exception as e:
        raise RuntimeError(f"Stable Diffusion failed to load: {e}")

    print("  [Generator] Generating image...")

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=SD_STEPS,
            guidance_scale=SD_GUIDANCE_SCALE,
            width=SD_WIDTH,
            height=SD_HEIGHT,
            num_images_per_prompt=1,
            generator=torch.Generator(DEVICE if DEVICE == "cuda" else "cpu").manual_seed(seed),
        )

    image = result.images[0]
    image = _enhance_output(image)
    image = add_watermark(image)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()

    print(f"  [Generator] Image generated ({len(b64)} bytes base64)")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [Generator] Pipeline unloaded. VRAM cleared.")
    print_vram_usage("After Generator cleanup")

    return b64
