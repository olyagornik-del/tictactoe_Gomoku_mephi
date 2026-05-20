"""Графики для лабораторной.

Использует matplotlib в headless-режиме (``Agg``) — графики
сохраняются в файлы, окна не открываются. Все функции принимают
``save_path`` и пишут PNG туда.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend — для CI и скриптов без дисплея
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from agents.perceptron import Perceptron  # noqa: E402
from lab.data_prep import ScalerParams  # noqa: E402


def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# --- loss curves --------------------------------------------------------


def plot_loss_curves(
    history: Mapping[str, Sequence[float]],
    title: str,
    save_path: str,
) -> None:
    """Кривые потерь train/val на одной фигуре."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["loss_train"], label="train", linewidth=2)
    ax.plot(history["loss_val"], label="val", linewidth=2, linestyle="--")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("BCE-потеря")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_multi_loss_curves(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
    title: str,
    save_path: str,
    which: str = "loss_train",
) -> None:
    """Несколько кривых на одной фигуре (для сеточных экспериментов).

    :param histories: словарь ``label → history``.
    :param which: ``"loss_train"`` или ``"loss_val"``.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in histories.items():
        ax.plot(hist[which], label=str(label), linewidth=1.8)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("BCE-потеря" + (" (train)" if which == "loss_train" else " (val)"))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# --- decision boundary --------------------------------------------------


def plot_decision_boundary(
    model: Perceptron,
    X: np.ndarray,
    y: np.ndarray,
    scaler_params: ScalerParams,
    save_path: str,
    title: str = "Граница решения",
) -> None:
    """Точки и линия ``W·x = 0`` для 2D-перцептрона.

    Данные ``X`` ожидаются **стандартизованными** (т.е. в том же
    пространстве, на котором обучалась модель). ``scaler_params``
    используется для подписи осей в исходных единицах.

    Уравнение прямой в стандартизованных координатах
    (с учётом ``b ≡ 0``):

    .. math::

        W_0 x_0 + W_1 x_1 = 0 \\;\\Longrightarrow\\;
        x_1 = -\\frac{W_0}{W_1}\\, x_0.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary поддерживает только 2D-данные")

    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, marker, color in [(0, "o", "tab:blue"), (1, "s", "tab:orange")]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], marker=marker, alpha=0.7,
                   edgecolors="k", linewidths=0.4, c=color,
                   label=f"класс {cls}")

    # Линия W·x + b = 0 в стандартизованных координатах.
    w0, w1 = float(model.W[0]), float(model.W[1])
    b = float(model.b)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)
    if abs(w1) > 1e-9:
        ys = -(w0 * xs + b) / w1
        ax.plot(xs, ys, "k-", linewidth=2, label="W·x + b = 0")
    else:
        # Вырожденный случай: вертикальная граница.
        x_line = -b / w0 if abs(w0) > 1e-9 else 0.0
        ax.axvline(x_line, color="k", linewidth=2, label="W·x + b = 0")

    ax.set_xlabel(f"Z-score(feat 0); mean={scaler_params['mean'][0]:.2f}, "
                  f"std={scaler_params['std'][0]:.2f}")
    ax.set_ylabel(f"Z-score(feat 1); mean={scaler_params['mean'][1]:.2f}, "
                  f"std={scaler_params['std'][1]:.2f}")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# --- ROC ----------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    title: str = "ROC-кривая",
) -> None:
    """ROC-кривая с площадью под ней."""
    _ensure_dir(save_path)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(np.trapezoid(tpr, fpr))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="случайный")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# --- misclassified ------------------------------------------------------


def plot_misclassified(
    model: Perceptron,
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
    title: str = "Ошибки классификации",
) -> None:
    """Точки, окрашенные по корректности предсказания.

    Зелёные — правильно классифицированные, красные — ошибки.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_misclassified поддерживает только 2D-данные")
    _ensure_dir(save_path)
    pred = model.predict(X)
    correct = pred == y
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[correct, 0], X[correct, 1], c="tab:green",
               marker="o", alpha=0.65, edgecolors="k", linewidths=0.3,
               label=f"верно ({int(correct.sum())})")
    ax.scatter(X[~correct, 0], X[~correct, 1], c="tab:red",
               marker="X", s=70, alpha=0.85, edgecolors="k", linewidths=0.5,
               label=f"ошибка ({int((~correct).sum())})")
    ax.set_xlabel("признак 0 (стандартизован)")
    ax.set_ylabel("признак 1 (стандартизован)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
