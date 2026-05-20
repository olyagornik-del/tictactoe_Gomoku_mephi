"""Подготовка синтетических данных для лабораторной.

Использует ``sklearn.make_classification`` и ``train_test_split``;
стандартизация выполнена вручную (параметры считаются *только* на
``train`` и переиспользуются на ``test`` — иначе утечка).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class ScalerParams(TypedDict):
    """Параметры Z-score стандартизации, полученные на ``train``."""

    mean: np.ndarray
    std: np.ndarray


def load_classification_data(
    test_size: float = 0.3,
    split_random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ScalerParams]:
    """Сгенерировать и подготовить датасет для классификации.

    * ``make_classification(n_samples=500, n_features=2, n_redundant=0,
      n_informative=2, random_state=42, n_clusters_per_class=1)``;
    * стратифицированный split 70/30 по умолчанию;
    * Z-score: ``mean``, ``std`` считаются только по ``X_train`` и
      применяются к ``X_test``.

    :return: ``(X_train, X_test, y_train, y_test, scaler_params)``.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=split_random_state
    )
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # Защита от деления на ноль на патологическом признаке.
    std = np.where(std == 0.0, 1.0, std)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    scaler: ScalerParams = {"mean": mean, "std": std}
    return X_train_std, X_test_std, y_train, y_test, scaler


def unstandardize(X_std: np.ndarray, scaler: ScalerParams) -> np.ndarray:
    """Обратное преобразование Z-score (для подписей осей и т.п.)."""
    return X_std * scaler["std"] + scaler["mean"]
