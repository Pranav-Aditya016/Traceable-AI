"""
Stage 5 — Grad-CAM Explainability.
DenseNet121 applied to the ORIGINAL uploaded image (not the generated one).
This produces a saliency map highlighting which regions of the actual
clinical document/scan the model finds most diagnostically salient.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from config import DEVICE, print_vram_usage


def _image_to_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_gradcam(original_image: Image.Image) -> dict:
    """
    Compute Grad-CAM saliency on the original uploaded image.
    Returns base64-encoded saliency overlay and binary mask overlay.
    """
    print("  [GradCAM] Loading DenseNet121...")
    print_vram_usage("Before GradCAM")

    image = original_image.convert("RGB").resize((512, 512))

    model = densenet121(weights=DenseNet121_Weights.DEFAULT).eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    print("  [GradCAM] Computing activation map on original image...")

    # Context manager ensures torchcam 0.4.x hooks register/deregister correctly
    with GradCAM(model, target_layer="features.denseblock4") as cam_extractor:
        model.zero_grad()
        output    = model(tensor)
        class_idx = output.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_idx, output)

    cam_array = activation_map[0].squeeze(0).cpu().numpy()
    cam_norm  = (cam_array - cam_array.min()) / (cam_array.max() - cam_array.min() + 1e-8)
    cam_pil   = Image.fromarray((cam_norm * 255).astype(np.uint8)).resize((512, 512))

    # Saliency — jet heatmap blended over original image
    saliency = overlay_mask(image, cam_pil, alpha=0.55, colormap="jet")

    # Binary mask — regions above threshold highlighted in red
    mask_array   = (cam_norm > 0.45).astype(np.uint8) * 255
    mask_rgb     = np.stack([mask_array, np.zeros_like(mask_array), np.zeros_like(mask_array)], axis=-1).astype(np.uint8)
    mask_pil     = Image.fromarray(mask_rgb).resize((512, 512))
    mask_blended = Image.blend(image, mask_pil.convert("RGB"), alpha=0.40)

    print(f"  [GradCAM] Maps computed. Top class index: {class_idx}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [GradCAM] Model unloaded. VRAM cleared.")
    print_vram_usage("After GradCAM cleanup")

    return {
        "saliency_b64": _image_to_b64(saliency),
        "mask_b64":     _image_to_b64(mask_blended),
    }
