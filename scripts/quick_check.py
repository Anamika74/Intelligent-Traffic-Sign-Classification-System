r"""Sanity check: how accurate is a checkpoint on the ITSD test folder when it goes
through the app's exact preprocessing?

Run from the project root:
    py scripts\quick_check.py
    py scripts\quick_check.py --data "Datasets\ITSD\test" --limit 500

What the result means:
    ~85-92%  -> the app pipeline is correct; bad predictions on other images are
                about the images (not cropped, different style, not a road sign).
    far lower -> preprocessing or class order differs from training. Compare the
                transforms in your training notebook with src/predict.py.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import DEFAULT_MODEL, MODEL_REGISTRY, get_device, load_class_names, load_model  # noqa: E402
from src.predict import _PREPROCESS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"Datasets\ITSD\test", help="folder with one sub-folder per class")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_REGISTRY))
    ap.add_argument("--limit", type=int, default=0, help="use a random subset of N images (0 = all)")
    args = ap.parse_args()

    device = get_device()
    cfg = MODEL_REGISTRY[args.model]
    dataset = datasets.ImageFolder(args.data, transform=_PREPROCESS)

    expected = load_class_names(cfg["classes_file"], cfg["num_classes"])
    if dataset.classes != expected:
        print("WARNING: class folders in the data differ from the class names file.")
        print(f"  data has {len(dataset.classes)} classes, names file has {len(expected)}.")
        print("  Predictions may be mislabeled. Tell me about this before going further.\n")

    if args.limit and args.limit < len(dataset):
        dataset = Subset(dataset, random.sample(range(len(dataset)), args.limit))
        base_classes = dataset.dataset.classes
    else:
        base_classes = dataset.classes

    model = load_model(cfg["path"], cfg["num_classes"], device)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    seen, right = Counter(), Counter()
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            for label, pred in zip(labels.tolist(), preds.tolist()):
                seen[label] += 1
                right[label] += int(label == pred)

    total, correct = sum(seen.values()), sum(right.values())
    print(f"Model: {args.model}")
    print(f"Images checked: {total}")
    print(f"Accuracy: {correct / total:.2%}\n")

    per_class = sorted(((right[c] / seen[c], c) for c in seen))
    print("10 weakest classes:")
    for acc, c in per_class[:10]:
        print(f"  {acc:6.1%}  {base_classes[c]}  ({seen[c]} images)")


if __name__ == "__main__":
    main()