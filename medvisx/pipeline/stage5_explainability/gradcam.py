"""
Grad-CAM on generated medical images.
Uses a pretrained DenseNet121 (CheXNet-style) as the classification backbone
to produce spatially relevant activation maps.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEVICE


class GradCAMExplainer:
    def __init__(self):
        print("[Grad-CAM] Loading DenseNet121 backbone...")
        # Use DenseNet121 pretrained on ImageNet as proxy backbone
        # (In production you'd use a CheXNet-pretrained checkpoint)
        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model = backbone.to(DEVICE).eval()

        # Target the last conv layer in DenseNet
        target_layers = [self.model.features.denseblock4.denselayer16.conv2]
        self.cam = GradCAM(model=self.model, target_layers=target_layers)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        print("[Grad-CAM] Ready")

    def generate_heatmap(self, pil_image: Image.Image,
                          target_class: int = None) -> Image.Image:
        """
        Generate Grad-CAM heatmap overlay on the generated medical image.
        Returns a PIL Image with the heatmap blended onto the original.
        """
        # Prepare input tensor
        img_rgb = np.array(pil_image.convert("RGB").resize((224, 224)))
        img_float = img_rgb.astype(np.float32) / 255.0
        tensor = self.preprocess(pil_image).unsqueeze(0).to(DEVICE)

        # Compute Grad-CAM
        grayscale_cam = self.cam(input_tensor=tensor,
                                  targets=None)  # None = highest scoring class
        grayscale_cam = grayscale_cam[0]  # (H, W)

        # Blend onto original image
        visualization = show_cam_on_image(
            img_float, grayscale_cam,
            use_rgb=True, colormap=cv2.COLORMAP_JET
        )
        return Image.fromarray(visualization)

    def unload(self):
        """Free GPU memory."""
        self.model = self.model.to("cpu")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("[Grad-CAM] Unloaded from GPU")
