"""
Image utility functions for MedVis-X pipeline.
"""
import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL Image."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_keep_aspect(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image while maintaining aspect ratio with padding."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Pad to target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def create_side_by_side(img1: Image.Image, img2: Image.Image,
                         gap: int = 10) -> Image.Image:
    """Create a side-by-side comparison of two PIL images."""
    w1, h1 = img1.size
    w2, h2 = img2.size
    max_h = max(h1, h2)
    canvas = Image.new("RGB", (w1 + gap + w2, max_h), (255, 255, 255))
    canvas.paste(img1, (0, (max_h - h1) // 2))
    canvas.paste(img2, (w1 + gap, (max_h - h2) // 2))
    return canvas
