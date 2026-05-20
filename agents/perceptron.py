"""Перцептрон-инференс для Гомоку.

Архитектура: один линейный слой плюс ``tanh`` → скаляр в ``[-1, 1]``.

Кодирование входа: окно 9×9 вокруг последнего хода, three one-hot
каналами «пусто / свой / чужой» относительно фиксированной точки
зрения → 243 признака. Если ходов ещё не было, окно центрируется на
``(0, 0)``.

Семантика выхода и инференса
----------------------------
Сэмпл регистрируется *после* того, как игрок ``P`` сделал ход.
Метка — оценка получившейся позиции с точки зрения ``P``
(нормализованная). Тогда на инференсе агент пробует каждый кандидат,
ставит свой камень, кодирует окно «от себя» и выбирает ход с
**максимумом** — что в точности соответствует ТЗ.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from agents.base import Agent
from game.board import Board, Coord, Symbol

#: Радиус окна (в полуходах от центра). 9×9 → radius = 4.
WINDOW_RADIUS: int = 4
#: Размер одной стороны окна.
WINDOW_SIDE: int = 2 * WINDOW_RADIUS + 1  # 9
#: Размер вектора признаков: 9 * 9 * 3 = 243.
FEATURE_DIM: int = WINDOW_SIDE * WINDOW_SIDE * 3
#: Делитель для приведения эвристической оценки в ``(-1, 1)``.
SCORE_NORM: float = 10_000.0
#: Путь к весам по умолчанию.
DEFAULT_MODEL_PATH: str = os.path.join("models", "perceptron.npz")


# ---------- кодирование --------------------------------------------------

def encode(board: Board, perspective: Symbol) -> np.ndarray:
    """One-hot кодирование окна 9×9 вокруг последнего хода.

    :param board: текущее состояние доски.
    :param perspective: «своя» сторона; «чужая» — противоположная.
    :return: float-вектор формы ``(243,)`` со значениями ``{0., 1.}``.
    """
    cx, cy = board.last_move if board.last_move is not None else (0, 0)
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    cells = board.cells
    idx = 0
    for dx in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
        for dy in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
            s = cells.get((cx + dx, cy + dy))
            if s is None:
                feats[idx] = 1.0
            elif s == perspective:
                feats[idx + 1] = 1.0
            else:
                feats[idx + 2] = 1.0
            idx += 3
    return feats


def normalize_score(score: float) -> float:
    """``tanh(score / SCORE_NORM)`` — метка в ``[-1, 1]``."""
    return float(np.tanh(score / SCORE_NORM))


# ---------- модель -------------------------------------------------------

class Perceptron:
    """Однослойный перцептрон ``y = tanh(W·x + b)``.

    Реализация на NumPy: SGD по mini-batch + MSE-потеря. Хранит веса
    как массивы ``np.float32``.
    """

    def __init__(self, W: np.ndarray | None = None, b: float = 0.0,
                 seed: int | None = None) -> None:
        if W is None:
            rng = np.random.default_rng(seed)
            W = (rng.standard_normal(FEATURE_DIM) * 0.01).astype(np.float32)
        self.W: np.ndarray = W.astype(np.float32, copy=False)
        self.b: float = float(b)

    # инференс ------------------------------------------------------------

    def predict(self, x: np.ndarray) -> float:
        """Прогон одного вектора признаков."""
        return float(np.tanh(self.W @ x + self.b))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Прогон батча ``(N, 243)`` → ``(N,)``."""
        return np.tanh(X @ self.W + self.b)

    # сериализация -------------------------------------------------------

    def save(self, path: str) -> None:
        """Сохранить веса в ``.npz``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, W=self.W, b=np.float32(self.b))

    @classmethod
    def load(cls, path: str) -> "Perceptron":
        """Загрузить веса из ``.npz``."""
        with np.load(path) as data:
            return cls(W=data["W"], b=float(data["b"]))

    # обучение -----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 64,
        seed: int | None = None,
        verbose: bool = False,
    ) -> list[float]:
        """SGD по mini-batch с MSE-потерей.

        Возвращает список средних потерь по эпохам — для отладки.
        """
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        rng = np.random.default_rng(seed)
        n = len(X)
        losses: list[float] = []
        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb = X[idx]
                yb = y[idx]
                z = xb @ self.W + self.b
                pred = np.tanh(z)
                err = pred - yb
                # dL/dz = 2/N * err * (1 - tanh²)
                dz = (2.0 / len(idx)) * err * (1.0 - pred * pred)
                grad_W = xb.T @ dz
                grad_b = float(dz.sum())
                self.W -= lr * grad_W.astype(np.float32)
                self.b -= lr * grad_b
                epoch_loss += float((err * err).mean()) * len(idx)
            epoch_loss /= n
            losses.append(epoch_loss)
            if verbose:
                print(f"  эпоха {epoch + 1}/{epochs}: MSE={epoch_loss:.4f}")
        return losses


# ---------- агент --------------------------------------------------------

#: Сколько партий собирает auto-train, если файла весов нет. Меньше,
#: чем спецификация (2000) — реалистичный fallback, чтобы не вешать
#: GUI на часы. Для полного обучения запускайте training-скрипт явно.
AUTO_TRAIN_GAMES: int = 200


class PerceptronAgent(Agent):
    """Агент, выбирающий ход по максимуму выхода перцептрона."""

    name = "Перцептрон"
    metric_name = "прогонов"

    def __init__(
        self,
        symbol: Symbol,
        model_path: str = DEFAULT_MODEL_PATH,
        model: Perceptron | None = None,
    ) -> None:
        super().__init__(symbol)
        if model is not None:
            self.model = model
        elif os.path.exists(model_path):
            self.model = Perceptron.load(model_path)
        else:
            # Ленивый импорт — иначе круговая зависимость с training/.
            from training.train_perceptron import train

            print(
                f"[perceptron] Веса {model_path!r} не найдены — "
                f"запускаю авто-обучение ({AUTO_TRAIN_GAMES} партий). "
                "Для полного обучения запустите training/train_perceptron.py."
            )
            self.model = train(num_games=AUTO_TRAIN_GAMES, out_path=model_path)

    def choose_move(self, board: Board) -> Coord:
        moves = board.search_window()
        self.last_nodes = len(moves)
        if len(moves) == 1:
            return moves[0]

        best_move = moves[0]
        best_value = -float("inf")
        for m in moves:
            board.place(*m, self.symbol)
            value = self.model.predict(encode(board, self.symbol))
            board.undo()
            if value > best_value:
                best_value = value
                best_move = m
        return best_move


__all__: Iterable[str] = (
    "FEATURE_DIM",
    "WINDOW_SIDE",
    "WINDOW_RADIUS",
    "SCORE_NORM",
    "DEFAULT_MODEL_PATH",
    "AUTO_TRAIN_GAMES",
    "Perceptron",
    "PerceptronAgent",
    "encode",
    "normalize_score",
)
