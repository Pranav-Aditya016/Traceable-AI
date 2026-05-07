"""
TraceMD Model Downloader
Run ONCE before starting: python download_models.py
Downloads all required models to ./models/ directory.

NOTE: bitsandbytes is NOT available on Windows.
All models use fp16 (half-precision) instead of 4-bit quantization.
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CATALOG = [
    {
        "id": "microsoft/trocr-large-handwritten",
        "folder": "trocr-large-handwritten",
        "gated": False,
    },
    {
        "id": "google/paligemma2-3b-pt-896",
        "folder": "paligemma2-3b-pt-896",
        "gated": True,
    },
    {
        "id": "google/medgemma-4b-it",
        "folder": "medgemma-4b-it",
        "gated": True,
    },
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "folder": "stable-diffusion-v1-5",
        "gated": False,
    },
]


def _has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def check_and_download(model_spec: dict) -> bool:
    model_id = model_spec["id"]
    save_path = MODELS_DIR / model_spec["folder"]

    if _has_files(save_path):
        print(f"  ✓ Already available: {model_id} -> {save_path}")
        return True

    print(f"  ↓ Downloading: {model_id} ...")
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        kwargs = {
            "repo_id": model_id,
            "local_dir": str(save_path),
        }
        if token:
            kwargs["token"] = token

        snapshot_download(**kwargs)

        print(f"  ✓ Downloaded: {model_id} -> {save_path}")
        return True

    except Exception as e:
        print(f"  ✗ FAILED to download {model_id}: {e}")
        if model_spec.get("gated", False):
            print(f"    → This is a gated model. Accept terms on Hugging Face first.")
        print(f"    → Make sure you are logged in: huggingface-cli login")
        print(f"    → Or set token env var: HF_TOKEN / HUGGINGFACE_HUB_TOKEN")
        return False


def check_densenet() -> bool:
    try:
        import torchvision

        torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.DEFAULT
        )
        print("  ✓ DenseNet121 weights available")
        return True
    except Exception as e:
        print(f"  ✗ FAILED to prepare DenseNet121 weights: {e}")
        return False


if __name__ == "__main__":
    print("\n=== TraceMD Model Downloader ===\n")
    print(f"Model directory: {MODELS_DIR}\n")

    failures = []

    for model_spec in MODEL_CATALOG:
        ok = check_and_download(model_spec)
        if not ok:
            failures.append(model_spec["id"])

    if not check_densenet():
        failures.append("densenet121")

    if failures:
        print("\n=== Completed with failures ===")
        for model_id in failures:
            print(f"  - {model_id}")
        print("\nFix access/auth issues, then rerun download_models.py")
        sys.exit(1)

    print("\n=== All models downloaded successfully ===")
    print("You can now start the server: uvicorn backend.main:app --reload\n")
