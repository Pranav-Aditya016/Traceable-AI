"""
SDXL Medical Image Generator
=============================
Uses Stable Diffusion XL for high-quality medical image synthesis.
Optimised for 8GB VRAM (RTX 4060) with CPU offloading.
Falls back to SD v1.5 if SDXL is unavailable.
Random seeds ensure varied output for every request.
"""
import os
import sys
import random
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEVICE, DTYPE


class MedicalImageGenerator:
    """Singleton SDXL generator with automatic fallback to SD v1.5."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.pipe = None
        self.model_name = None
        self._load_model()

    def _load_model(self):
        """Try SDXL first (if cached), fall back to SD v1.5 otherwise."""

        # ── Check if SDXL is already downloaded ──────────────────────
        sdxl_cached = self._is_model_cached("stabilityai/stable-diffusion-xl-base-1.0")

        if sdxl_cached:
            try:
                from diffusers import StableDiffusionXLPipeline

                print("[Generator] Loading Stable Diffusion XL (cached)...")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )

                # CPU offloading: peak VRAM ~3-4 GB (fits easily in 8 GB)
                self.pipe.enable_model_cpu_offload()
                try:
                    self.pipe.enable_vae_slicing()
                except Exception:
                    pass

                self.model_name = "SDXL"
                self._steps = 15
                self._guidance = 7.5
                self._height = 768
                self._width = 768
                print("[Generator] SDXL loaded with CPU offloading ✓")
                return

            except Exception as e:
                print(f"[Generator] SDXL load failed: {e}")
        else:
            print("[Generator] SDXL not cached yet — using SD v1.5 with enhanced prompts")
            print("[Generator] Run: python -c \"from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', variant='fp16')\" to pre-download SDXL")

        # ── Fallback: SD v1.5 (always available) ─────────────────────
        try:
            from diffusers import StableDiffusionPipeline

            print("[Generator] Loading Stable Diffusion v1.5 with enhanced prompts...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )
            if DEVICE == "cuda":
                self.pipe = self.pipe.to(DEVICE)
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_vae_slicing()
                except Exception:
                    pass
            else:
                self.pipe = self.pipe.to(DEVICE)

            self.model_name = "SD_v1.5"
            self._steps = 40
            self._guidance = 7.5
            self._height = 512
            self._width = 512
            print("[Generator] SD v1.5 loaded ✓")

        except Exception as e:
            print(f"[Generator] CRITICAL: No generation model available: {e}")
            raise RuntimeError("Cannot load any image generation model") from e

    @staticmethod
    def _is_model_cached(model_id: str) -> bool:
        """Check if a HuggingFace model is already downloaded."""
        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if not os.path.isdir(cache_dir):
            return False
        # HF cache uses models--org--name format
        expected = "models--" + model_id.replace("/", "--")
        model_dir = os.path.join(cache_dir, expected)
        if not os.path.isdir(model_dir):
            return False
        # Check that snapshots exist (model actually downloaded)
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            return True
        return False

    def generate(self, prompt: str, negative_prompt: str = None) -> Image.Image:
        """
        Generate a medical image from a clinical text prompt.
        Uses a RANDOM seed each time to ensure varied output.
        """
        # Random seed — never fixed, guarantees different output each call
        seed = random.randint(0, 2**32 - 1)
        print(f"[Generator] Model={self.model_name} | Seed={seed}")
        print(f"[Generator] Prompt: {prompt[:150]}...")

        # Generator device: "cpu" for CPU-offloaded SDXL, DEVICE for on-GPU SD v1.5
        gen_device = "cpu" if self.model_name == "SDXL" else DEVICE
        generator = torch.Generator(gen_device).manual_seed(seed)

        if negative_prompt is None:
            negative_prompt = (
                "photograph of a person, human body, skin, nude, nsfw, "
                "face, portrait, selfie, anatomy photo, "
                "blurry, low quality, distorted, text, watermark, "
                "cartoon, illustration, drawing, painting, anime, "
                "3d render, colorful, bright colors, "
                "non-medical, natural photo, landscape, deformed"
            )

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance,
            generator=generator,
            height=self._height,
            width=self._width,
        )

        img = result.images[0]
        print(f"[Generator] Done — {img.size[0]}x{img.size[1]}")
        return img

    def unload(self):
        """Free GPU memory by moving model to CPU."""
        if hasattr(self, "pipe") and self.pipe is not None:
            try:
                self.pipe = self.pipe.to("cpu")
            except Exception:
                pass
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print("[Generator] Unloaded from GPU")


# Backward-compatible alias
LoRAGenerator = MedicalImageGenerator
