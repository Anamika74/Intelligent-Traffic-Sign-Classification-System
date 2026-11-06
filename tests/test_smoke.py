"""Smoke tests: run with `pytest`. They use random weights, so no .pth files are needed."""
import numpy as np
import torch
from PIL import Image

from src.model import build_resnet18, load_class_names
from src.predict import run_inference


def _fake_image():
    return Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))


def test_inference_returns_valid_output():
    model = build_resnet18(num_classes=85).eval()
    names = [f"Class {i}" for i in range(85)]
    pred = run_inference(model, _fake_image(), names, torch.device("cpu"))
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_k) == 3
    assert pred.overlay.shape == (224, 224, 3)
    assert pred.overlay.dtype == np.uint8


def test_top_probabilities_are_sorted():
    model = build_resnet18(num_classes=43).eval()
    names = [f"Class {i}" for i in range(43)]
    pred = run_inference(model, _fake_image(), names, torch.device("cpu"))
    probs = [p for _, p in pred.top_k]
    assert probs == sorted(probs, reverse=True)


def test_missing_class_file_falls_back():
    assert load_class_names("does_not_exist.json", 5) == [f"Class {i}" for i in range(5)]