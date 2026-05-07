"""
TraceMD — Configuration
All model paths, constants, and VRAM management utilities.
"""

import os
import torch
from pathlib import Path

# ─── Paths ───
BACKEND_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = BACKEND_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR = BACKEND_DIR.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ─── Model IDs ───
# These must match what's actually downloaded in HuggingFace cache
TROCR_MODEL_ID     = "microsoft/trocr-large-handwritten"
PALIGEMMA_MODEL_ID = "google/paligemma2-3b-pt-896"
MEDGEMMA_MODEL_ID  = "google/medgemma-4b-it"
SD_MODEL_ID        = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# ─── Device ───
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Generation settings ───
SD_STEPS          = 36
SD_GUIDANCE_SCALE = 7.5
SD_WIDTH          = 512
SD_HEIGHT         = 512

# ─── Disclaimer ───
RESEARCH_DISCLAIMER = "Research prototype. Not for clinical diagnostic use."
SYNTHETIC_WATERMARK = "SYNTHETIC — NOT A REAL PATIENT IMAGE"


def model_cache(model_id: str) -> str:
    """Return local cache path for a model."""
    return str(MODELS_DIR / model_id.replace("/", "--"))


def _latest_snapshot_path(base_path: Path):
    """Return latest HF snapshot path under base_path, if present."""
    snapshot_dirs = []

    direct_snapshots = base_path / "snapshots"
    if direct_snapshots.exists():
        snapshot_dirs.extend([p for p in direct_snapshots.iterdir() if p.is_dir()])

    hub_dir = base_path / "hub"
    if hub_dir.exists():
        for model_dir in hub_dir.glob("models--*"):
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                snapshot_dirs.extend([p for p in snapshots.iterdir() if p.is_dir()])

    if not snapshot_dirs:
        return None

    snapshot_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshot_dirs[0]


def resolve_model_source(model_id: str) -> str:
    """
    Resolve the best source for a model.
    Priority:
    1) Local folders under TraceMD/models/
    2) HF cache-style folders under TraceMD/models/
    3) Remote model ID fallback
    """
    model_slug = model_id.split("/", 1)[-1]
    hf_key = model_id.replace("/", "--")

    candidates = [
        MODELS_DIR / model_slug,
        MODELS_DIR / hf_key,
        MODELS_DIR / f"models--{hf_key}",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        snapshot = _latest_snapshot_path(candidate)
        if snapshot is not None:
            return str(snapshot)

        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)

    return model_id


def require_cuda(stage_name: str = ""):
    """Abort with clear message if CUDA is unavailable."""
    if not torch.cuda.is_available():
        msg = f"CUDA not available. GPU required for {stage_name}."
        print(f"  ✗ CRITICAL: {msg}")
        raise RuntimeError(msg)


def clear_vram(*objects):
    """Delete objects and clear CUDA cache."""
    for obj in objects:
        if obj is not None:
            try:
                del obj
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def print_vram_usage(label: str = ""):
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"  [VRAM] {label}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
