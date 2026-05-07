"""
TraceMD Pipeline Debug Test
Runs Stage 1 (TrOCR) and Stage 2 (MedGemma NER) sequentially
with CUDA_LAUNCH_BLOCKING=1 to find exact failure point.
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import gc
import traceback
from PIL import Image
import numpy as np

print("=" * 60)
print("TraceMD Pipeline Debug Test")
print("=" * 60)
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {vram:.1f} GB")
print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
print()

# Create a test image (handwritten-like)
test_img = Image.fromarray(
    np.random.randint(180, 255, (400, 600, 3), dtype=np.uint8)
)

# ── STAGE 1: TrOCR ──
print("=" * 60)
print("[TEST] Stage 1: TrOCR")
print("=" * 60)
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    print("  Loading processor...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    print("  Loading model (fp32)...")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    print("  Moving to CUDA...")
    model = model.to("cuda")
    model.eval()
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("  Running inference...")
    pixel_values = processor(images=test_img, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=50)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"  Output: '{text[:100]}'")

    print("  Cleaning up TrOCR...")
    del model, processor, pixel_values, ids
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("  ✓ Stage 1 PASSED")
    ocr_text = text if text.strip() else "Patient John age 45 male headache fever cough"
except Exception as e:
    print(f"  ✗ Stage 1 FAILED: {e}")
    traceback.print_exc()
    ocr_text = "Patient John age 45 male headache fever cough"
    # Try to clean up
    gc.collect()
    torch.cuda.empty_cache()

print()

# ── STAGE 2: MedGemma NER ──
print("=" * 60)
print("[TEST] Stage 2: MedGemma NER")
print("=" * 60)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Using OCR text: '{ocr_text[:80]}...'")
    print(f"  VRAM before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
    print("  Loading model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it", torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    prompt = f"""You are a clinical AI assistant. Extract medical entities from this text.
Return ONLY valid JSON: {{"organ": "string", "symptoms": ["list"], "age": 0, "sex": "M"}}

Text: {ocr_text}

JSON:"""

    print("  Tokenizing (direct, no apply_chat_template)...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    print(f"  Input shape: {inputs['input_ids'].shape}")

    print("  Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=256,
            temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    print(f"  Output: '{raw[:200]}'")

    print("  Cleaning up MedGemma...")
    del model, tokenizer, inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("  ✓ Stage 2 PASSED")
except Exception as e:
    print(f"  ✗ Stage 2 FAILED: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("[TEST] Complete")
print("=" * 60)
