"""
ModelEvaluator — Classification Metrics for Quality Tier Models
===============================================================

Port of building-stock-analysis/T3.1/Code/src/utils/models_utils.py

Provides the same evaluation suite used in the EPC A→G classification pipeline:
  - Confusion matrix with class labels
  - Per-class Precision / Recall / F1  (mirrors 'micro_scores' in T3.1)
  - Balanced accuracy + macro F1 + weighted F1  (mirrors 'macro_scores' in T3.1)

Usage
-----
from src.models.model_evaluator import ModelEvaluator

ev = ModelEvaluator(labels=["G","F","E","D","C","B","A"])
res = ev.get_scores(y_true, y_pred, verbose=True)
# res["confusion_matrix"]  — pd.DataFrame with labelled rows/cols
# res["micro_scores"]      — per-class Precision / Recall / F1
# res["macro_scores"]      — Balanced Accuracy / F1 Macro / F1 Weighted

Pure Python fallback is provided when sklearn is not installed.
"""

import numpy as np
from typing import Optional, List, Dict, Union

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy metric helpers (used when sklearn unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray,
                             n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


def _precision_recall_f1_numpy(cm: np.ndarray) -> tuple:
    """Per-class precision, recall, F1 from confusion matrix (macro)."""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall    = np.zeros(n)
    f1        = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp + 1e-10)
        recall[i]    = tp / (tp + fn + 1e-10)
        f1[i]        = (2 * precision[i] * recall[i] /
                        (precision[i] + recall[i] + 1e-10))
    return precision, recall, f1


def _balanced_accuracy_numpy(cm: np.ndarray) -> float:
    """Balanced accuracy = mean per-class recall."""
    recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
    return float(recalls.mean())


# ─────────────────────────────────────────────────────────────────────────────
# ModelEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluation metrics for quality tier classification.

    Mirrors models_utils.py from building-stock-analysis:
      get_confusion_matrix()  → labelled confusion matrix
      get_scores()            → per-class + macro metrics

    Parameters
    ----------
    labels : list of str, optional
        Class labels in order (e.g. ["G","F","E","D","C","B","A"]).
        If None, inferred from data.
    """

    DEFAULT_LABELS = ["G", "F", "E", "D", "C", "B", "A"]

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        self.labels = labels if labels is not None else self.DEFAULT_LABELS

    def _to_int(self, y: Union[np.ndarray, list]) -> np.ndarray:
        """Convert string tier labels to int indices if needed."""
        arr = np.asarray(y)
        if arr.dtype.kind in ("U", "S", "O"):
            label_map = {lb: i for i, lb in enumerate(self.labels)}
            return np.array([label_map.get(str(v), 3) for v in arr], dtype=int)
        return arr.astype(int)

    # ── Confusion Matrix ──────────────────────────────────────────────────────

    def get_confusion_matrix(self,
                              y_true,
                              y_pred,
                              normalize: Optional[str] = None):
        """
        Build labelled confusion matrix.

        Parameters
        ----------
        normalize : None | 'true' | 'pred' | 'all'
            Mirrors sklearn confusion_matrix normalize parameter.

        Returns
        -------
        pd.DataFrame (if pandas available) or np.ndarray
        """
        yt = self._to_int(y_true)
        yp = self._to_int(y_pred)
        n  = len(self.labels)

        try:
            from sklearn.metrics import confusion_matrix as _skl_cm
            cm = _skl_cm(yt, yp, labels=list(range(n)), normalize=normalize)
        except ImportError:
            cm = _confusion_matrix_numpy(yt, yp, n).astype(float)
            if normalize == "true":
                row_sums = cm.sum(axis=1, keepdims=True)
                cm = cm / (row_sums + 1e-10)
            elif normalize == "pred":
                col_sums = cm.sum(axis=0, keepdims=True)
                cm = cm / (col_sums + 1e-10)
            elif normalize == "all":
                cm = cm / (cm.sum() + 1e-10)

        if _PANDAS:
            df = pd.DataFrame(cm, columns=self.labels, index=self.labels)
            df.index.name = "True"
            return df
        return cm

    # ── Per-class + Macro Scores ──────────────────────────────────────────────

    def get_scores(self,
                   y_true,
                   y_pred,
                   sample_weight=None,
                   verbose: bool = False) -> Dict:
        """
        Compute full evaluation suite.

        Mirrors T3.1/Code/src/utils/models_utils.py :: get_scores()

        Returns
        -------
        dict with keys:
            confusion_matrix : labelled confusion matrix (pd.DataFrame or ndarray)
            micro_scores     : per-class Precision / Recall / F1
            macro_scores     : Balanced Accuracy / F1 Macro / F1 Weighted
        """
        yt = self._to_int(y_true)
        yp = self._to_int(y_pred)
        n  = len(self.labels)

        conf_mat = self.get_confusion_matrix(yt, yp)

        # ── Per-class scores (mirrors 'micro_scores' in T3.1) ────────────────
        try:
            from sklearn.metrics import (precision_score, recall_score,
                                          f1_score, balanced_accuracy_score)
            precision = precision_score(yt, yp, average=None, labels=list(range(n)),
                                        zero_division=0)
            recall    = recall_score(   yt, yp, average=None, labels=list(range(n)),
                                        zero_division=0)
            f1_cls    = f1_score(       yt, yp, average=None, labels=list(range(n)),
                                        zero_division=0)
            bal_acc   = balanced_accuracy_score(yt, yp, sample_weight=sample_weight)
            f1_macro  = f1_score(yt, yp, average="macro",    zero_division=0)
            f1_wtd    = f1_score(yt, yp, average="weighted",
                                  sample_weight=sample_weight, zero_division=0)
        except ImportError:
            # Pure-numpy fallback
            cm_np     = _confusion_matrix_numpy(yt, yp, n)
            precision, recall, f1_cls = _precision_recall_f1_numpy(cm_np)
            bal_acc   = _balanced_accuracy_numpy(cm_np)
            f1_macro  = float(f1_cls.mean())
            f1_wtd    = float(np.average(f1_cls,
                                          weights=cm_np.sum(axis=1) + 1e-10))

        if _PANDAS:
            micro_scores = pd.DataFrame({
                "Precision": precision,
                "Recall":    recall,
                "F1 Score":  f1_cls,
            }, index=self.labels).T

            macro_scores = pd.DataFrame({
                "Score": [bal_acc, f1_macro, f1_wtd]
            }, index=["Accuracy Balanced", "F1 Score Macro", "F1 Score Weighted"])
        else:
            micro_scores = {
                "labels":     self.labels,
                "Precision":  precision.tolist(),
                "Recall":     recall.tolist(),
                "F1 Score":   f1_cls.tolist(),
            }
            macro_scores = {
                "Accuracy Balanced": bal_acc,
                "F1 Score Macro":    f1_macro,
                "F1 Score Weighted": f1_wtd,
            }

        result = {
            "confusion_matrix": conf_mat,
            "micro_scores":     micro_scores,
            "macro_scores":     macro_scores,
        }

        if verbose:
            print("Confusion Matrix")
            if _PANDAS:
                print(conf_mat.round(3))
            else:
                print(conf_mat)
            print("\nPer-class scores")
            if _PANDAS:
                print(micro_scores.round(3))
            else:
                for k, v in micro_scores.items():
                    print(f"  {k}: {v}")
            print("\nMacro scores")
            if _PANDAS:
                print(macro_scores.round(3))
            else:
                for k, v in macro_scores.items():
                    print(f"  {k}: {v:.4f}")

        return result

    # ── Convenience: evaluate UniverseClassifier predictions ─────────────────

    def evaluate_universe(self,
                          y_true_tiers: List[str],
                          y_pred_tiers: List[str],
                          verbose: bool = True) -> Dict:
        """
        Evaluate UniverseClassifier output directly with tier labels.

        Parameters
        ----------
        y_true_tiers : list of str  (e.g. ["A","B","D","G", ...])
        y_pred_tiers : list of str
        """
        return self.get_scores(y_true_tiers, y_pred_tiers, verbose=verbose)
