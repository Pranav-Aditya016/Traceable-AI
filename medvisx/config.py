"""
MedVis-X Configuration
All hyperparameters, paths, and model settings.
Everything runs 100% locally — no cloud APIs needed.
"""
import os
import torch

# ── Hardware ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(BASE_DIR)  # c:\Pranav Aditya\Traceable AI
OCR_MODEL_PATH  = os.path.join(PROJECT_ROOT, "my_trained_models", "checkpoints", "crnn-best.ckpt")
LORA_WEIGHTS    = os.path.join(PROJECT_ROOT, "MedVisX_Trained_LoRA_Weights", "final_weights")
SD_BASE_MODEL   = "runwayml/stable-diffusion-v1-5"

# ── OCR ───────────────────────────────────────────────────────────────────
OCR_IMG_H       = 32
OCR_IMG_W       = 128
OCR_HIDDEN_SIZE = 64
OCR_CHARSET     = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,/-"
OCR_BLANK_IDX   = 0

# ── Training (OCR) ────────────────────────────────────────────────────────
OCR_EPOCHS      = 50
OCR_BATCH_SIZE  = 32
OCR_LR          = 1e-3
OCR_AUGMENT_FACTOR = 5   # 5x augmentation

# ── Training (LoRA) ───────────────────────────────────────────────────────
LORA_RANK       = 8
LORA_ALPHA      = 8
LORA_TRAIN_STEPS = 5000
LORA_LR         = 2e-4
LORA_BATCH_SIZE = 4
LORA_IMG_SIZE   = 512
IMAGES_PER_DATASET = 50

# ── Generation ────────────────────────────────────────────────────────────
# SDXL is primary; SD v1.5 is automatic fallback
SD_BASE_MODEL      = "stabilityai/stable-diffusion-xl-base-1.0"
SD_INFERENCE_STEPS = 25
SD_GUIDANCE_SCALE  = 9.0
SD_SEED            = None   # None = random seed per request (varied output)

# ── Local LLM (Ollama) ───────────────────────────────────────────────────
# Ollama runs locally — no API key needed
# Install Ollama from https://ollama.com then run: ollama pull llava
OLLAMA_BASE_URL      = "http://localhost:11434"
OLLAMA_MODEL         = "llava"          # multimodal model for image+text
OLLAMA_MAX_TOKENS    = 2048
OLLAMA_TIMEOUT       = 300              # seconds (increased for two-model approach)

# ── Hypothesis Scoring ────────────────────────────────────────────────────
TOP_K_HYPOTHESES = 3
