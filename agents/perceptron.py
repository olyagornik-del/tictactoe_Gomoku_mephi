"""Однослойный перцептрон-классификатор (sigmoid + BCE).

Реализация только на NumPy. Один и тот же класс обслуживает:

* лабораторную работу (синтетика из ``sklearn.make_classification``,
  эксперименты по lr / batch / init);
* агент Гомоку (см. :class:`PerceptronAgent` в фазе 4в).

Математика
----------
Модель — линейный классификатор с **фиксированным нулевым смещением**:

.. math::

    \\hat{y} = \\sigma(W \\cdot x), \\qquad b \\equiv 0

Потеря — средняя бинарная кросс-энтропия (BCE):

.. math::

    L = -\\frac{1}{N} \\sum_i \\big[
        y_i \\log \\hat{y}_i + (1 - y_i) \\log (1 - \\hat{y}_i)
    \\big].

Градиент по весам (благодаря свойству :math:`\\sigma'(z) = \\hat{y}(1-\\hat{y})`
он сокращается с производной BCE):

.. math::

    \\nabla_W L = \\frac{1}{N}\\, X^{\\top} (\\hat{y} - y).
"""

from __future__ import annotations

import os
from typing import Callable, Literal

import numpy as np

from agents.base import Agent
from game.board import Board, Coord, Symbol

InitMethod = Literal["zeros", "small_random", "large_random"]

#: Радиус окна вокруг хода (9×9 → радиус 4 в каждую сторону).
WINDOW_RADIUS: int = 4
#: Размер вектора признаков: 9 × 9 × 3 = 243.
FEATURE_DIM: int = (2 * WINDOW_RADIUS + 1) ** 2 * 3
#: Путь к весам по умолчанию для PerceptronAgent.
DEFAULT_MODEL_PATH: str = os.path.join("models", "perceptron.npz")


class Perceptron:
    """Однослойный перцептрон с сигмоидой и BCE-потерей.

    :param n_features: размер входного вектора.
    :param init_method: способ инициализации весов:

        * ``"zeros"`` — нули;
        * ``"small_random"`` — :math:`\\mathcal{N}(0, 0.01^2)` (по умолчанию);
        * ``"large_random"`` — :math:`\\mathcal{N}(0, 10^2)`.

    :param random_state: сид для воспроизводимости.

    Атрибуты:

    * :attr:`W` — массив весов формы ``(n_features,)``.
    * :attr:`b` — смещение, всегда ``0.0`` (не обучается, см. ТЗ).
    """

    def __init__(
        self,
        n_features: int,
        init_method: InitMethod = "small_random",
        random_state: int | None = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError("n_features должно быть положительным")
        self.n_features: int = int(n_features)
        self.init_method: InitMethod = init_method
        rng = np.random.default_rng(random_state)
        if init_method == "zeros":
            self.W: np.ndarray = np.zeros(n_features, dtype=np.float64)
        elif init_method == "small_random":
            self.W = rng.normal(0.0, 0.01, size=n_features)
        elif init_method == "large_random":
            self.W = rng.normal(0.0, 10.0, size=n_features)
        else:
            raise ValueError(f"неизвестный init_method: {init_method!r}")
        #: Смещение всегда равно нулю и не обновляется при обучении.
        self.b: float = 0.0

    # --- активация ------------------------------------------------------

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Численно устойчивая логистическая функция.

        Формулы:

        * для :math:`z \\ge 0`: :math:`\\sigma(z) = 1 / (1 + e^{-z})`;
        * для :math:`z < 0`:    :math:`\\sigma(z) = e^{z} / (1 + e^{z})`.

        Такое ветвление гарантирует, что ``exp`` всегда получает
        неположительный аргумент → переполнения нет даже на ``z=±1000``.
        """
        z = np.asarray(z, dtype=np.float64)
        out = np.empty_like(z)
        pos = z >= 0
        # для неотрицательных z
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        # для отрицательных z
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    # --- предсказания ---------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Вернуть вероятности класса 1 для матрицы ``X`` ``(N, n_features)``.

        Поддерживает и одиночный вектор формы ``(n_features,)`` —
        результатом будет скаляр-массив формы ``()``.
        """
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.W + self.b
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Классификация с порогом ``threshold`` (по умолчанию 0.5)."""
        return (self.forward(X) >= threshold).astype(np.int64)

    # --- потеря ---------------------------------------------------------

    @staticmethod
    def compute_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eps: float = 1e-15,
    ) -> float:
        """Средняя бинарная кросс-энтропия.

        ``y_pred`` обрезается в ``[eps, 1-eps]`` — это защищает
        :math:`\\log` от ``-inf`` на насыщенных предсказаниях
        (особенно при ``init='large_random'``).
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)
        return float(
            -np.mean(
                y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
            )
        )

    # --- обучение -------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        lr: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> dict[str, list[float]]:
        """Mini-batch SGD по BCE.

        Веса обновляются как
        :math:`W \\leftarrow W - \\eta \\cdot \\frac{1}{B} X_b^{\\top}(\\hat{y} - y_b)`
        для каждого мини-батча размера :math:`B`.

        В конце каждой эпохи измеряются полные потери на train и val.

        :return: ``{"loss_train": [...], "loss_val": [...]}`` длины
            ``epochs`` — для построения кривых обучения.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64).ravel()
        if X_train.shape[1] != self.n_features:
            raise ValueError(
                f"X_train имеет {X_train.shape[1]} признаков, "
                f"ожидалось {self.n_features}"
            )

        n = len(X_train)
        rng = np.random.default_rng(random_state)
        history: dict[str, list[float]] = {"loss_train": [], "loss_val": []}

        for _ in range(epochs):
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb = X_train[idx]
                yb = y_train[idx]
                yhat = self.forward(xb)
                # ∇L/∇W = (1/B) Xᵀ (ŷ - y) ; b не обновляется.
                grad_W = xb.T @ (yhat - yb) / len(idx)
                self.W -= lr * grad_W
            history["loss_train"].append(
                self.compute_loss(y_train, self.forward(X_train))
            )
            history["loss_val"].append(
                self.compute_loss(y_val, self.forward(X_val))
            )
        return history

    # --- сериализация --------------------------------------------------

    def save_weights(self, path: str) -> None:
        """Сохранить веса в ``.npz``.

        В архиве два массива: ``W`` (форма ``(n_features,)``) и ``b``
        (скаляр). Папка-родитель создаётся при необходимости.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez(path, W=self.W, b=np.float64(self.b))

    @classmethod
    def load_weights(cls, path: str) -> "Perceptron":
        """Загрузить веса из ``.npz`` и вернуть готовый ``Perceptron``.

        Размер ``n_features`` определяется по форме ``W``. Метод
        инициализации заполняется как ``"zeros"`` (после чего веса
        перезаписываются загруженными), потому что фактическая
        инициализация уже зашита в сохранённый ``W``.
        """
        with np.load(path) as data:
            W = np.asarray(data["W"], dtype=np.float64)
            b = float(data["b"])
        model = cls(n_features=len(W), init_method="zeros")
        model.W = W
        # b всегда 0 в этом классе, но честно копируем то, что записано.
        model.b = b
        return model


# ============================================================================
# Интеграция с Гомоку
# ============================================================================


def board_to_features(state: Board, move: Coord) -> np.ndarray:
    """Превратить позицию Гомоку в 243-мерный one-hot вектор.

    Окно ``9×9`` центрируется на сделанном ходе ``move``. Для каждой
    из 81 клетки записывается 3 канала ``(пусто, свой, чужой)``:

    * **«свой»**  — символ игрока, поставившего камень в ``move``;
    * **«чужой»** — противоположный символ;
    * **«пусто»** — клетка свободна.

    Точно один канал на клетку равен 1, остальные — 0, поэтому
    сумма всех элементов вектора всегда ``81``.

    :raises ValueError: если ``move`` не занят на доске — функция
        ожидает уже сделанный ход (центр окна должен быть камнем).
    """
    perspective = state.get(*move)
    if perspective is None:
        raise ValueError(
            f"клетка {move} пуста — board_to_features ожидает сделанный ход"
        )
    own: Symbol = perspective
    cx, cy = move
    cells = state.cells
    feats = np.zeros(FEATURE_DIM, dtype=np.float64)
    idx = 0
    for dx in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
        for dy in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
            s = cells.get((cx + dx, cy + dy))
            if s is None:
                feats[idx] = 1.0
            elif s == own:
                feats[idx + 1] = 1.0
            else:
                feats[idx + 2] = 1.0
            idx += 3
    return feats


#: Тип функции извлечения признаков.
FeatureFn = Callable[[Board, Coord], np.ndarray]


class PerceptronAgent(Agent):
    """Агент, выбирающий ход по максимуму выхода перцептрона.

    Стратегия: для каждого хода-кандидата в окне поиска поставить свой
    камень, посчитать признаки через ``feature_fn``, прогнать
    ``perceptron.forward(...)``, выбрать ход с наибольшим выходом.
    Откатить пробный ход и вернуть результат.

    Все три зависимости передаются явно — никаких ``None`` по
    умолчанию. Для удобства запуска из main/GUI есть
    :meth:`from_disk`, который грузит веса (или обучает их при
    отсутствии файла) и собирает агента.
    """

    name = "Перцептрон"
    metric_name = "прогонов"

    def __init__(
        self,
        symbol: Symbol,
        perceptron: "Perceptron",
        feature_fn: FeatureFn,
    ) -> None:
        super().__init__(symbol)
        self._perceptron = perceptron
        self._feature_fn = feature_fn

    @classmethod
    def from_disk(
        cls,
        symbol: Symbol,
        path: str = DEFAULT_MODEL_PATH,
        feature_fn: FeatureFn = board_to_features,
    ) -> "PerceptronAgent":
        """Сконструировать агента, загрузив веса с диска.

        Если файла ``path`` нет — запускается обучение через
        :func:`training.train_perceptron.train` с дефолтными
        параметрами ТЗ и веса сохраняются по этому пути.
        """
        if os.path.exists(path):
            perceptron = Perceptron.load_weights(path)
        else:
            # Ленивый импорт — иначе круговая зависимость
            # (training/ зависит от Perceptron из этого модуля).
            from training.train_perceptron import train

            print(
                f"[perceptron] {path!r} не найден — запускаю обучение. "
                "Это займёт время; для быстрого fallback см. флаг --games "
                "в training/train_perceptron.py."
            )
            perceptron = train(out_path=path)
        return cls(symbol, perceptron, feature_fn)

    def choose_move(self, board: Board) -> Coord:
        moves = board.search_window()
        self.last_nodes = 0
        if len(moves) == 1:
            return moves[0]

        best_move: Coord = moves[0]
        best_value = -np.inf
        for m in moves:
            board.place(*m, self.symbol)
            x = self._feature_fn(board, m)
            value = float(self._perceptron.forward(x))
            board.undo()
            self.last_nodes += 1
            if value > best_value:
                best_value = value
                best_move = m
        return best_move
