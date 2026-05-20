"""Метрики бинарной классификации.

accuracy, precision, recall, F1, confusion matrix — реализованы
вручную через NumPy. ``roc_auc`` — через ``sklearn.metrics`` (по ТЗ).

Соглашения
----------
* класс 1 — «положительный»;
* :func:`confusion_matrix` возвращает матрицу ``[[TN, FP], [FN, TP]]``
  (строки — истинный класс, столбцы — предсказанный).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score as _sk_roc_auc_score


def _as_int_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Матрица ошибок ``[[TN, FP], [FN, TP]]``."""
    yt, yp = _as_int_arrays(y_true, y_pred)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля правильных предсказаний."""
    yt, yp = _as_int_arrays(y_true, y_pred)
    return float(np.mean(yt == yp))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``TP / (TP + FP)``. На пустом знаменателе — 0."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = int(cm[1, 1]), int(cm[0, 1])
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``TP / (TP + FN)``. На пустом знаменателе — 0."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = int(cm[1, 1]), int(cm[1, 0])
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``2·P·R / (P + R)``."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC ROC через :func:`sklearn.metrics.roc_auc_score`."""
    return float(_sk_roc_auc_score(y_true, y_score))
