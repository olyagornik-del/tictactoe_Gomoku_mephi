"""Генератор обучающих пар для перцептрона.

Партии Alpha-Beta vs Alpha-Beta с подмешиванием случайного хода
(``epsilon``-шум) — для разнообразия позиций. На каждом полу-ходе
записывается пара ``(board_to_features(state, move), label)``:

* ``state`` — доска уже после хода;
* ``move`` — координаты только что сделанного хода;
* ``label = 1``, если ``heuristic.evaluate(state, mover) > 0``
  (позиция после хода выгодна сходившему), иначе ``0``.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np

from agents.alphabeta import AlphaBetaAgent
from agents.heuristic import evaluate
from agents.perceptron import FEATURE_DIM, FeatureFn, board_to_features
from game.board import O, X, Board, Symbol
from game.rules import winner


def _move_with_noise(
    agent: AlphaBetaAgent,
    board: Board,
    rng: random.Random,
    epsilon: float,
) -> tuple[int, int]:
    """Ход агента с вероятностью ``epsilon`` заменяется на случайный."""
    if rng.random() < epsilon:
        return rng.choice(board.search_window())
    return agent.choose_move(board)


def generate_dataset(
    num_games: int = 2000,
    epsilon: float = 0.1,
    depth: int = 2,
    max_moves: int = 60,
    seed: int | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Сгенерировать датасет AB-vs-AB.

    :param num_games: число партий (по ТЗ — 2000).
    :param epsilon: вероятность случайного хода вместо AB.
    :param depth: глубина AB-учителя. ``2`` — разумный компромисс
        качество/скорость для генерации.
    :param max_moves: «потолок» длины партии; защищает от зацикливания
        при сильном ε-шуме.
    :return: ``(X, y)``: ``X`` формы ``(N, 243)``, ``y`` ∈ {0, 1}.
    """
    rng = random.Random(seed)
    ab_x = AlphaBetaAgent(X, depth=depth)
    ab_o = AlphaBetaAgent(O, depth=depth)

    xs: list[np.ndarray] = []
    ys: list[int] = []

    for game_idx in range(num_games):
        board = Board()
        turn = X
        agent = ab_x
        for _ in range(max_moves):
            move = _move_with_noise(agent, board, rng, epsilon)
            board.place(*move, turn)
            # Сэмпл: позиция после хода turn, фичи и метка от лица turn.
            xs.append(board_to_features(board, move))
            ys.append(1 if evaluate(board, turn) > 0 else 0)
            if winner(board) is not None:
                break
            turn = O if turn == X else X
            agent = ab_o if turn == O else ab_x
        if verbose and (game_idx + 1) % max(1, num_games // 10) == 0:
            print(f"  партий: {game_idx + 1}/{num_games}, сэмплов: {len(xs)}")

    X_arr = (
        np.stack(xs).astype(np.float64)
        if xs else np.zeros((0, FEATURE_DIM), np.float64)
    )
    y_arr = np.asarray(ys, dtype=np.float64)
    return X_arr, y_arr


def generate_imitation_dataset(
    num_games: int,
    feature_fn: FeatureFn,
    n_features: int,
    neg_per_pos: int = 4,
    epsilon: float = 0.1,
    depth: int = 2,
    max_moves: int = 60,
    seed: int | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Имитационный датасет: «какой ход выбрал бы Alpha-Beta».

    На каждой позиции AB-vs-AB-партии:

    * ход, выбранный Alpha-Beta → положительный пример (метка 1);
    * ``neg_per_pos`` случайных других ходов из окна → отрицательные (0).

    Признаки считаются ``feature_fn(board, move, player)`` без фактической
    постановки камня. Партия продвигается с ε-шумом (для разнообразия
    позиций), но метку всегда даёт лучший ход AB.

    :return: ``(X, y)`` формы ``(N, n_features)`` и ``(N,)`` ∈ {0, 1}.
    """
    rng = random.Random(seed)
    ab_x = AlphaBetaAgent(X, depth=depth)
    ab_o = AlphaBetaAgent(O, depth=depth)

    xs: list[np.ndarray] = []
    ys: list[int] = []

    for game_idx in range(num_games):
        board = Board()
        turn: Symbol = X
        agent = ab_x
        for _ in range(max_moves):
            window = board.search_window()
            best = agent.choose_move(board)
            # положительный пример — ход AB
            xs.append(feature_fn(board, best, turn))
            ys.append(1)
            # отрицательные — случайные другие ходы
            others = [m for m in window if m != best]
            rng.shuffle(others)
            for m in others[:neg_per_pos]:
                xs.append(feature_fn(board, m, turn))
                ys.append(0)
            # продвигаем партию (с ε-шумом)
            played = rng.choice(window) if rng.random() < epsilon else best
            board.place(*played, turn)
            if winner(board) is not None:
                break
            turn = O if turn == X else X
            agent = ab_o if turn == O else ab_x
        if verbose and (game_idx + 1) % max(1, num_games // 10) == 0:
            print(f"  партий: {game_idx + 1}/{num_games}, сэмплов: {len(xs)}")

    X_arr = (
        np.stack(xs).astype(np.float64)
        if xs else np.zeros((0, n_features), np.float64)
    )
    y_arr = np.asarray(ys, dtype=np.float64)
    return X_arr, y_arr
