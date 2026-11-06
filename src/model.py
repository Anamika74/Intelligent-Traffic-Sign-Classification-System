"""Model loading utilities.

Keeps everything about *building and loading* the ResNet18 in one place so the
Streamlit app, the Grad-CAM code and the tests all share the same logic.
This file has no Streamlit imports, so it can be reused from a training script.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent

# One entry per experiment from the README. `classes_file` is optional:
# if it does not exist the app falls back to "Class N" labels.
_ALL_MODELS = {
    "Native Indian v2 (96.5% acc, 93.7% macro-F1)": {
        "path": ROOT / "models" / "native_indian_v2.pth",
        "num_classes": 85,
        "classes_file": ROOT / "classes_indian.json",
    },
    "Native Indian v1, original (91.2% acc, 74.6% macro-F1)": {
        "path": ROOT / "models" / "native_indian_baseline_e4.pth",
        "num_classes": 85,
        "classes_file": ROOT / "classes_indian.json",
    },
    "German GTSRB model (95.0% on German signs)": {
        "path": ROOT / "models" / "gtsrb_baseline_resnet18.pth",
        "num_classes": 43,
        "classes_file": ROOT / "classes_gtsrb.json",
    },
}
# Only offer models whose weight files are present, so a deployed app never shows a broken option.
# (If no weights are found at all, keep the full list so the error message names the missing file.)
MODEL_REGISTRY = {name: cfg for name, cfg in _ALL_MODELS.items() if cfg["path"].exists()} or _ALL_MODELS
DEFAULT_MODEL = next(iter(MODEL_REGISTRY))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _extract_state_dict(checkpoint):
    """Accept a plain state_dict or a dict that wraps one, and strip 'module.'."""
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def load_model(weights_path, num_classes: int, device=None) -> nn.Module:
    device = device or get_device()
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Put the .pth file in the 'models/' "
            "folder (file names are case-sensitive on Linux)."
        )
    model = build_resnet18(num_classes)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    return model.to(device).eval()


def load_class_names(classes_file, num_classes: int) -> list:
    """Return a list of length `num_classes` indexed by class id.

    Accepts either the dict from `ImageFolder.class_to_idx` ({name: idx})
    or a plain JSON list of names. Falls back to "Class N" if the file is missing.
    """
    fallback = [f"Class {i}" for i in range(num_classes)]
    path = Path(classes_file) if classes_file else None
    if not path or not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        names = fallback[:]
        for name, idx in data.items():
            if 0 <= int(idx) < num_classes:
                names[int(idx)] = str(name)
        return names
    if isinstance(data, list) and len(data) == num_classes:
        return [str(n) for n in data]
    return fallback


def pretty_name(raw: str) -> str:
    """'SPEED_LIMIT_30' -> 'Speed limit 30'."""
    return raw.replace("_", " ").strip().capitalize()