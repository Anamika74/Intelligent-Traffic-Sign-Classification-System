r"""Full evaluation of one checkpoint on a test folder (one sub-folder per class).

Run from the project root:
    py scripts\evaluate_model.py
    py scripts\evaluate_model.py --data "Datasets\ITSD\test"

Writes into results/<model-name>/ :
    metrics.json                       accuracy, macro-F1, weighted-F1
    per_class.csv                      precision, recall, F1 and image count per class
    top_confusions.csv                 most common mistakes (true -> predicted)
    confusion_matrix.png               picture of all the mistakes
    confidence_vs_accuracy.png / .csv  what each confidence threshold costs and buys
    worst_errors.png                   the most confident wrong predictions
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import (confusion_matrix, coverage_table, per_class_metrics,  # noqa: E402
                         summarize, top_confusions)
from src.model import DEFAULT_MODEL, MODEL_REGISTRY, get_device, load_model  # noqa: E402
from src.predict import _PREPROCESS  # noqa: E402
from src.report import plot_confusion_matrix, plot_coverage_curve, plot_worst_errors  # noqa: E402


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"Datasets\ITSD\test")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_REGISTRY))
    ap.add_argument("--out", default="results")
    ap.add_argument("--worst", type=int, default=12, help="how many wrong images to show in worst_errors.png")
    args = ap.parse_args()

    device = get_device()
    cfg = MODEL_REGISTRY[args.model]
    dataset = datasets.ImageFolder(args.data, transform=_PREPROCESS)
    class_names = dataset.classes
    n = len(class_names)
    if n != cfg["num_classes"]:
        sys.exit(f"The data has {n} classes but this model expects {cfg['num_classes']}. "
                 "Pick the right --model or --data.")

    model = load_model(cfg["path"], cfg["num_classes"], device)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true, y_pred, conf = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            top_p, top_i = probs.max(dim=1)
            y_true += labels.tolist()
            y_pred += top_i.tolist()
            conf += top_p.tolist()

    y_true, y_pred, conf = np.array(y_true), np.array(y_pred), np.array(conf)
    correct = y_true == y_pred

    slug = re.sub(r"[^a-z0-9]+", "_", args.model.lower()).strip("_")
    out = Path(args.out) / slug
    out.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, n)
    summary = summarize(cm)
    summary["model"] = args.model
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    precision, recall, f1, support = per_class_metrics(cm)
    rows = sorted(zip(class_names, support.tolist(), precision.round(3), recall.round(3), f1.round(3)),
                  key=lambda r: r[4])
    write_csv(out / "per_class.csv", ["class", "images", "precision", "recall", "f1"], rows)

    confusions = top_confusions(cm, class_names, k=20)
    write_csv(out / "top_confusions.csv", ["true_class", "predicted_as", "count"], confusions)

    plot_confusion_matrix(cm, out / "confusion_matrix.png")

    table = coverage_table(conf, correct, thresholds=np.arange(0.3, 0.96, 0.05))
    write_csv(out / "confidence_vs_accuracy.csv", ["threshold", "images_kept", "accuracy_on_kept"],
              [(round(t, 2), round(c, 4), round(a, 4)) for t, c, a in table])
    plot_coverage_curve(table, out / "confidence_vs_accuracy.png")

    wrong = np.where(~correct)[0]
    wrong = wrong[np.argsort(conf[wrong])[::-1]][: args.worst]
    plot_worst_errors(
        [(dataset.samples[i][0], class_names[y_true[i]], class_names[y_pred[i]], conf[i]) for i in wrong],
        out / "worst_errors.png",
    )

    print(f"Model:        {args.model}")
    print(f"Images:       {summary['n_images']}  ({summary['n_classes']} classes)")
    print(f"Accuracy:     {summary['accuracy']:.2%}")
    print(f"Macro-F1:     {summary['macro_f1']:.2%}   <- counts every class equally")
    print(f"Weighted-F1:  {summary['weighted_f1']:.2%}\n")
    print("Most common mistakes (true -> predicted):")
    for true_name, pred_name, count in confusions[:10]:
        print(f"  {count:3d}x  {true_name}  ->  {pred_name}")
    print(f"\nFiles saved in: {out}")


if __name__ == "__main__":
    main()