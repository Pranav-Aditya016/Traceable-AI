"""
TraceMD Model Verifier
Checks all models are present and loadable.
Run before starting the server to catch issues early.
"""

import sys
import torch
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR.parent / "models"
errors = []


def _count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def _find_existing_model_dir(candidate_names: list):
    for name in candidate_names:
        path = MODELS_DIR / name
        if path.exists():
            file_count = _count_files(path)
            if file_count > 0:
                return path, file_count

            snapshots_root = path / "snapshots"
            if snapshots_root.exists():
                for snapshot_dir in snapshots_root.iterdir():
                    if snapshot_dir.is_dir():
                        snapshot_files = _count_files(snapshot_dir)
                        if snapshot_files > 0:
                            return snapshot_dir, snapshot_files

            hub_root = path / "hub"
            if hub_root.exists():
                for model_dir in hub_root.glob("models--*"):
                    snapshots = model_dir / "snapshots"
                    if snapshots.exists():
                        for snapshot_dir in snapshots.iterdir():
                            if snapshot_dir.is_dir():
                                snapshot_files = _count_files(snapshot_dir)
                                if snapshot_files > 0:
                                    return snapshot_dir, snapshot_files

    return None, 0

print("\n=== TraceMD Model Verification ===\n")
print(f"Model directory: {MODELS_DIR}\n")

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {total_vram:.1f} GB")
    if total_vram < 7.5:
        print("  ⚠ WARNING: Less than 8GB VRAM. Pipeline may fail on large models.")
else:
    print("  ⚠ WARNING: CUDA not available. CPU fallback may be very slow.")

print()

# Check each model directory
checks = [
    (
        "TrOCR Large Handwritten",
        [
            "trocr-large-handwritten",
            "microsoft--trocr-large-handwritten",
            "models--microsoft--trocr-large-handwritten",
        ],
    ),
    (
        "PaliGemma 2 3B PT 896",
        [
            "paligemma2-3b-pt-896",
            "google--paligemma2-3b-pt-896",
            "models--google--paligemma2-3b-pt-896",
            "paligemma2-3b-mix-896",
            "google--paligemma2-3b-mix-896",
            "models--google--paligemma2-3b-mix-896",
        ],
    ),
    (
        "MedGemma 4B IT",
        [
            "medgemma-4b-it",
            "google--medgemma-4b-it",
            "models--google--medgemma-4b-it",
        ],
    ),
    (
        "Stable Diffusion v1.5",
        [
            "stable-diffusion-v1-5",
            "runwayml--stable-diffusion-v1-5",
            "models--runwayml--stable-diffusion-v1-5",
        ],
    ),
]

for name, candidate_names in checks:
    path, file_count = _find_existing_model_dir(candidate_names)
    if path is not None:
        print(f"  ✓ {name} — {file_count} files found ({path})")
    else:
        print(f"  ✗ MISSING: {name} — run: python download_models.py")
        errors.append(name)

# Check DenseNet via torchvision
try:
    import torchvision
    torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
    print(f"  ✓ DenseNet121 — torchvision weights available")
except Exception as e:
    print(f"  ✗ DenseNet121 — {e}")
    errors.append("DenseNet121")

print()

if errors:
    print(f"✗ {len(errors)} model(s) missing: {', '.join(errors)}")
    print("  Run: python download_models.py")
    sys.exit(1)
else:
    print("✓ All models verified. Safe to start server.\n")
