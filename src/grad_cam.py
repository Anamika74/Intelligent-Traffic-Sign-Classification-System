"""
================================================================================
MODEL INTERPRETABILITY & EXPLAINABLE AI (XAI)

OVERVIEW:
This module implements Gradient-weighted Class Activation Mapping (Grad-CAM) for a custom-headed ResNet18 convolutional neural network. It functions as a diagnostic layer to look inside the network's final convolutional layer block (layer4[-1]) to visually verify the network's attention mechanisms.

Grad-CAM acts as an X-ray for our AI's brain. It looks at the absolute last 
convolutional layer of our loaded ResNet18 model to figure out exactly which 
pixels the AI focused on before making its final traffic sign prediction.

================================================================================
"""

import threading

import cv2
import numpy as np
import torch
from PIL import Image

# The model object is shared by all Streamlit sessions (st.cache_resource), so
# only one Grad-CAM pass may attach hooks to it at a time.
_LOCK = threading.Lock()


def forward_with_cam(model, x, class_idx=None, target_layer=None):
    """Forward + backward pass on a single image tensor of shape (1, 3, H, W).

    Returns:
        logits  : tensor (1, num_classes), detached
        cam     : numpy array (h, w) in [0, 1] for `class_idx` (default: predicted class)
    """
    layer = target_layer if target_layer is not None else model.layer4[-1]
    store = {}

    def forward_hook(_module, _inputs, output):
        store["activations"] = output
        # A tensor hook captures the gradient w.r.t. this output. It is more
        # reliable than register_full_backward_hook with ResNet's in-place ReLU.
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    with _LOCK:
        handle = layer.register_forward_hook(forward_hook)
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                logits = model(x)
                target = int(logits.argmax(dim=1).item()) if class_idx is None else class_idx
                logits[0, target].backward()
        finally:
            handle.remove()

    activations = store["activations"].detach()[0]   # (C, h, w)
    gradients = store["gradients"].detach()[0]       # (C, h, w)
    weights = gradients.mean(dim=(1, 2))             # one weight per channel
    cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    return logits.detach(), cam.cpu().numpy()


def overlay_cam(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.4, size: int = 224):
    """Blend the heatmap over the image. Returns an RGB uint8 array."""
    base = np.array(pil_img.convert("RGB").resize((size, size)))
    heat = cv2.resize(cam, (size, size))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)  # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)