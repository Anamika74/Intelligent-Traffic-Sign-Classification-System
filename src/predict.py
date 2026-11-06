"""Inference: preprocessing, top-k probabilities, Grad-CAM and timing in one call."""
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.grad_cam import forward_with_cam, overlay_cam

# Same normalisation as ImageNet-style training in your notebook.
# If you trained with different mean/std, change them here.
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class Prediction:
    label: str                 # predicted class name
    confidence: float          # softmax probability of the top class, 0-1
    top_k: list                # [(class name, probability), ...] best first
    overlay: np.ndarray        # RGB image with the Grad-CAM heatmap
    latency_ms: float          # forward + Grad-CAM time


def run_inference(model, pil_img: Image.Image, class_names: list, device, k: int = 3) -> Prediction:
    x = _PREPROCESS(pil_img.convert("RGB")).unsqueeze(0).to(device)

    start = time.perf_counter()
    logits, cam = forward_with_cam(model, x)          # one pass: logits + heatmap
    latency_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=1)[0]
    k = min(k, probs.numel())
    top_probs, top_idx = probs.topk(k)
    top_k = [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]

    return Prediction(
        label=top_k[0][0],
        confidence=top_k[0][1],
        top_k=top_k,
        overlay=overlay_cam(pil_img, cam),
        latency_ms=latency_ms,
    )