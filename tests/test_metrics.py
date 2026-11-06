"""Tests for the numpy metrics and the report plots (no model or dataset needed)."""
import numpy as np

from src.metrics import confusion_matrix, coverage_table, per_class_metrics, summarize, top_confusions
from src.report import plot_confusion_matrix, plot_coverage_curve


def _example():
    # 3 classes. true: A A A B B C   predicted: A A B B A C
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 0, 2]
    return confusion_matrix(y_true, y_pred, 3)


def test_confusion_matrix_counts():
    cm = _example()
    assert cm.tolist() == [[2, 1, 0], [1, 1, 0], [0, 0, 1]]


def test_summary_values():
    s = summarize(_example())
    assert abs(s["accuracy"] - 4 / 6) < 1e-9
    # F1 per class: A = 2/3, B = 1/2, C = 1.0  ->  macro = (2/3 + 1/2 + 1) / 3
    assert abs(s["macro_f1"] - (2 / 3 + 1 / 2 + 1) / 3) < 1e-9


def test_per_class_handles_empty_class():
    cm = confusion_matrix([0, 1], [0, 0], 3)          # class 2 never appears
    precision, recall, f1, support = per_class_metrics(cm)
    assert support[2] == 0 and f1[2] == 0               # no division-by-zero crash
    assert summarize(cm)["n_classes"] == 2


def test_top_confusions_order():
    rows = top_confusions(_example(), ["A", "B", "C"], k=5)
    assert rows[0][2] == 1 and len(rows) == 2           # A->B and B->A, once each


def test_coverage_table():
    conf = [0.9, 0.8, 0.4, 0.3]
    correct = [True, True, False, False]
    table = coverage_table(conf, correct, [0.5, 0.95])
    assert table[0] == (0.5, 0.5, 1.0)                  # keep 2 of 4, both right
    assert np.isnan(table[1][2])                        # nothing kept at 0.95


def test_plots_are_written(tmp_path):
    plot_confusion_matrix(_example(), tmp_path / "cm.png")
    plot_coverage_curve(coverage_table([0.9, 0.5], [True, False], [0.3, 0.6]), tmp_path / "c.png")
    assert (tmp_path / "cm.png").exists() and (tmp_path / "c.png").exists()