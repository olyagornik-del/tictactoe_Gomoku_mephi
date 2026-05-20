"""Эксперименты лабораторной: lr, batch_size, init.

Каждая функция обучает несколько перцептронов, собирает истории
потерь, строит общий график loss-кривых и печатает таблицу метрик.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from agents.perceptron import InitMethod, Perceptron
from lab.metrics import accuracy
from lab.visualization import plot_multi_loss_curves

# Стандартные значения для воспроизводимости.
DEFAULT_EPOCHS: int = 100
DEFAULT_SEED: int = 0


# --- утилиты ------------------------------------------------------------


def _train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    lr: float = 0.1,
    batch_size: int = 32,
    init_method: InitMethod = "small_random",
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> tuple[Perceptron, dict[str, list[float]]]:
    """Обучить один перцептрон с заданными гиперпараметрами."""
    n_features = X_train.shape[1]
    model = Perceptron(n_features, init_method=init_method, random_state=seed)
    history = model.fit(
        X_train, y_train, X_test, y_test,
        epochs=epochs, lr=lr, batch_size=batch_size, random_state=seed,
    )
    return model, history


def _print_table(title: str, header: tuple[str, ...],
                 rows: Iterable[tuple[object, ...]]) -> None:
    """Печать ASCII-таблицы фиксированной ширины колонок."""
    rows_list = list(rows)
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows_list), default=0))
              for i, h in enumerate(header)]
    sep = "  ".join("-" * w for w in widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(f"=== {title} ===")
    print(fmt.format(*header))
    print(sep)
    for r in rows_list:
        print(fmt.format(*r))


# --- эксперимент 1: learning rate --------------------------------------


LRS: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0)


def experiment_learning_rate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "lab/figures/exp_learning_rate.png",
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> dict[float, dict[str, list[float]]]:
    """Перебор скоростей обучения. Все остальные гиперпараметры дефолтны."""
    histories: dict[float, dict[str, list[float]]] = {}
    rows: list[tuple[object, ...]] = []
    for lr in LRS:
        model, hist = _train_one(
            X_train, y_train, X_test, y_test,
            lr=lr, epochs=epochs, seed=seed,
        )
        histories[lr] = hist
        tr_acc = accuracy(y_train, model.predict(X_train))
        te_acc = accuracy(y_test, model.predict(X_test))
        rows.append((f"{lr:g}", f"{tr_acc:.3f}", f"{te_acc:.3f}",
                     f"{hist['loss_train'][-1]:.4f}"))

    _print_table(
        "Эксперимент: learning rate",
        ("lr", "train_acc", "test_acc", "final_loss"),
        rows,
    )
    plot_multi_loss_curves(
        {f"lr={lr:g}": h for lr, h in histories.items()},
        title="Кривые потерь по learning rate",
        save_path=save_path,
    )
    return histories


# --- эксперимент 2: batch size -----------------------------------------


BATCH_SIZES: tuple[int, ...] = (1, 16, 32, 64, 256)


def experiment_batch_size(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "lab/figures/exp_batch_size.png",
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> dict[int, dict[str, list[float]]]:
    """Перебор размеров мини-батча. ``lr = 0.1``."""
    histories: dict[int, dict[str, list[float]]] = {}
    rows: list[tuple[object, ...]] = []
    for bs in BATCH_SIZES:
        # batch_size = 1 — чистый стохастический SGD; sklearn-style.
        model, hist = _train_one(
            X_train, y_train, X_test, y_test,
            lr=0.1, batch_size=bs, epochs=epochs, seed=seed,
        )
        histories[bs] = hist
        tr_acc = accuracy(y_train, model.predict(X_train))
        te_acc = accuracy(y_test, model.predict(X_test))
        rows.append((str(bs), f"{tr_acc:.3f}", f"{te_acc:.3f}",
                     f"{hist['loss_train'][-1]:.4f}"))

    _print_table(
        "Эксперимент: batch size",
        ("batch_size", "train_acc", "test_acc", "final_loss"),
        rows,
    )
    plot_multi_loss_curves(
        {f"bs={bs}": h for bs, h in histories.items()},
        title="Кривые потерь по batch size",
        save_path=save_path,
    )
    return histories


# --- эксперимент 3: initialization -------------------------------------


INIT_METHODS: tuple[InitMethod, ...] = ("zeros", "small_random", "large_random")


def experiment_initialization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "lab/figures/exp_initialization.png",
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, list[float]]]:
    """Перебор инициализаций весов.

    Ожидаемая «лекционная» картина: ``large_random`` должен плохо
    учиться, потому что огромные ``|z|`` насыщают sigmoid
    (``σ(z) ≈ 0`` или ``≈ 1``), а её производная
    ``σ'(z) = σ(z)(1-σ(z)) ≈ 0`` — градиент исчезает.

    На самом деле для пары **BCE + sigmoid** градиент по весам имеет
    замечательный вид без множителя ``σ'``:

    .. math::

        \\nabla_W L = \\frac{1}{N}\\, X^{\\top}(\\hat{y} - y).

    Множитель ``σ'(z)`` сокращается с матчинговым ``1/(σ(z)(1-σ(z)))``
    от производной BCE. Поэтому даже при насыщенной сигмоиде модуль
    градиента ограничен снизу величиной ошибки и масштабом ``X`` —
    обучение не «зависает». «Vanishing gradient» проявляется в паре
    **MSE + sigmoid** или в глубоких сетях, но не здесь.

    Эмпирически все три инита сходятся к одному и тому же оптимуму на
    наших данных (см. таблицу) — что согласуется с этой математикой.
    """
    histories: dict[str, dict[str, list[float]]] = {}
    rows: list[tuple[object, ...]] = []
    for init in INIT_METHODS:
        model, hist = _train_one(
            X_train, y_train, X_test, y_test,
            lr=0.1, batch_size=32, init_method=init, epochs=epochs, seed=seed,
        )
        histories[init] = hist
        tr_acc = accuracy(y_train, model.predict(X_train))
        te_acc = accuracy(y_test, model.predict(X_test))
        rows.append((init, f"{tr_acc:.3f}", f"{te_acc:.3f}",
                     f"{hist['loss_train'][-1]:.4f}"))

    _print_table(
        "Эксперимент: initialization",
        ("init", "train_acc", "test_acc", "final_loss"),
        rows,
    )
    plot_multi_loss_curves(
        histories,
        title="Кривые потерь по инициализации",
        save_path=save_path,
    )
    return histories
