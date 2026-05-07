"""
LoRA Fine-Tuning for Stable Diffusion v1.5
Trains on medical image datasets using LoRA adapters.

Usage: python training/train_lora.py
"""
import os
import sys
import glob
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from tqdm import tqdm

from config import (
    SD_BASE_MODEL, LORA_RANK, LORA_ALPHA, LORA_TRAIN_STEPS,
    LORA_LR, LORA_BATCH_SIZE, LORA_IMG_SIZE, IMAGES_PER_DATASET,
)

# ── Dataset Paths — update to your local paths ───────────────────────────
# These can be Kaggle dataset paths or local extracted paths
DATASET_PATHS = {
    "Brain_MRI":            "extracted_data/CT SCANS/Computed Tomography (CT) of the Brain",
    "Chest_CT":             "extracted_data/CT SCANS/Chest CT-Scan images Dataset",
    "Breast_Cancer":        "extracted_data/CT SCANS/Breast Cancer",
    "Dental_OPG":           "extracted_data/CT SCANS/Dental OPG Xray Dataset",
    "DRR_Bones":            "extracted_data/CT SCANS/Digitally Reconstructed Radiographs (DRR) - Bones",
    "Lung_Cancer":          "extracted_data/CT SCANS/The IQ_OTH_NCCD lung cancer dataset",
    "Kidney":               "extracted_data/KIDNEY/data",
}

PROMPT_TEMPLATES = {
    "Brain_MRI":            "brain MRI scan showing {class_name}, medical imaging",
    "Chest_CT":             "chest CT scan showing {class_name}, axial view",
    "Breast_Cancer":        "breast cancer CT scan showing {class_name}, medical imaging",
    "Dental_OPG":           "dental OPG xray showing {class_name}, panoramic radiograph",
    "DRR_Bones":            "digitally reconstructed radiograph of {class_name}, bone imaging",
    "Lung_Cancer":          "lung cancer CT scan showing {class_name}, axial view",
    "Kidney":               "kidney CT scan showing {class_name}, medical imaging",
}


class MedicalFewShotDataset(Dataset):
    def __init__(self, image_paths, prompts):
        self.image_paths = image_paths
        self.prompts = prompts
        self.transform = A.Compose([
            A.Resize(LORA_IMG_SIZE, LORA_IMG_SIZE,
                     interpolation=cv2.INTER_LANCZOS4),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.Normalize([0.5] * 3, [0.5] * 3, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((LORA_IMG_SIZE, LORA_IMG_SIZE, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return {
            "pixel_values": self.transform(image=img)["image"],
            "text": self.prompts[idx],
        }


def collect_samples():
    """Collect image paths and prompts from all configured datasets."""
    all_paths, all_prompts = [], []
    for ds_name, ds_path in DATASET_PATHS.items():
        # Try absolute and relative paths
        search_path = ds_path if os.path.isabs(ds_path) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "MP01", ds_path
        )

        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images += glob.glob(
                str(Path(search_path) / "**" / ext), recursive=True
            )
        if not images:
            print(f"  WARNING: No images found for {ds_name} at {search_path}")
            continue
        sampled = random.sample(images, min(IMAGES_PER_DATASET, len(images)))
        for p in sampled:
            cls = Path(p).parent.name.replace("_", " ").lower()
            all_paths.append(p)
            all_prompts.append(
                PROMPT_TEMPLATES.get(ds_name, "medical image of {class_name}")
                .format(class_name=cls)
            )
        print(f"  {ds_name}: {len(sampled)} images collected")
    return all_paths, all_prompts


def train_lora():
    """Run LoRA fine-tuning on collected medical images."""
    random.seed(42)
    torch.manual_seed(42)
    accelerator = Accelerator(mixed_precision="fp16")

    print("=" * 50)
    print("  MedVis-X LoRA Fine-Tuning")
    print("=" * 50)

    print("\nCollecting training samples...")
    paths, prompts = collect_samples()
    if not paths:
        print("ERROR: No training images found. Update DATASET_PATHS.")
        return

    print(f"\nTotal training samples: {len(paths)}")
    dataset = MedicalFewShotDataset(paths, prompts)
    loader = DataLoader(
        dataset, batch_size=LORA_BATCH_SIZE, shuffle=True, num_workers=2
    )

    # Load pipeline components
    print("\nLoading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_BASE_MODEL, torch_dtype=torch.float32
    )
    unet = pipe.unet
    vae = pipe.vae.to(accelerator.device)
    text_encoder = pipe.text_encoder.to(accelerator.device)
    tokenizer = pipe.tokenizer
    noise_sched = DDPMScheduler.from_pretrained(
        SD_BASE_MODEL, subfolder="scheduler"
    )

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Apply LoRA to UNet cross-attention
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=LORA_LR, weight_decay=0.01
    )
    from diffusers.optimization import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 50, LORA_TRAIN_STEPS
    )

    unet, optimizer, loader, scheduler = accelerator.prepare(
        unet, optimizer, loader, scheduler
    )

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "lora_weights"
    )

    global_step = 0
    print(f"\nStarting training for {LORA_TRAIN_STEPS} steps...")
    while global_step < LORA_TRAIN_STEPS:
        unet.train()
        for batch in tqdm(
            loader, desc=f"Step {global_step}/{LORA_TRAIN_STEPS}"
        ):
            with accelerator.accumulate(unet):
                # Encode images
                pixel_values = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(pixel_values.float()).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                )
                noisy_latents = noise_sched.add_noise(
                    latents, noise, timesteps
                )

                # Encode text
                tokens = tokenizer(
                    batch["text"], padding="max_length",
                    max_length=77, truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                encoder_hidden = text_encoder(tokens)[0]

                # Predict noise
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden
                ).sample
                loss = torch.nn.functional.mse_loss(
                    noise_pred.float(), noise.float()
                )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")
                ckpt_dir = os.path.join(
                    output_dir, f"checkpoint-{global_step}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir)
                print(f"  Checkpoint saved: {ckpt_dir}")

            if global_step >= LORA_TRAIN_STEPS:
                break

    # Save final weights
    final_dir = os.path.join(output_dir, "final_weights")
    os.makedirs(final_dir, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(final_dir)
    print(f"\nLoRA training complete. Final weights saved to {final_dir}")


if __name__ == "__main__":
    train_lora()
