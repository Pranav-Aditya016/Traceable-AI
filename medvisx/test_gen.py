"""Quick test: SD + LoRA image generation only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LORA_WEIGHTS
print(f"LoRA weights path: {LORA_WEIGHTS}")
print(f"Exists: {os.path.exists(LORA_WEIGHTS)}")
print(f"Files: {os.listdir(LORA_WEIGHTS)}")

print("\nLoading generator...")
from pipeline.stage4_generation.lora_generator import LoRAGenerator
gen = LoRAGenerator.get_instance()

print("\nGenerating test image...")
prompt = "high resolution chest X-ray showing bilateral lower lobe consolidation consistent with pneumonia, medical imaging, grayscale"
img = gen.generate(prompt)
out_path = os.path.join(os.path.dirname(__file__), "test_generated.png")
img.save(out_path)
print(f"\nSaved: {out_path} ({img.size})")
print("SUCCESS")
