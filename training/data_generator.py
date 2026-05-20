"""Генератор обучающих пар для перцептрона.

Игра Alpha-Beta vs Alpha-Beta с подмешиванием случайного хода
вероятностью ``epsilon`` — для разнообразия позиций. На каждом
полу-ходе записывается пара ``(окно 9×9, нормализованная оценка)``
с точки зрения только что сходившего игрока.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np

from agents.alphabeta import AlphaBetaAgent
from agents.heuristic import evaluate
from agents.perceptron import FEATURE_DIM, encode, normalize_score
from game.board import O, X, Board
from game.rules import winner


def _epsilon_move(
    agent: AlphaBetaAgent,
    board: Board,
    rng: random.Random,
    epsilon: float,
) -> tuple[int, int]:
    """Выбор хода: с вероятностью ``epsilon`` — случайный из окна."""
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
    """Сгенерировать тренировочный датасет AB-vs-AB.

    :return: пара ``(X, y)`` — ``X`` формы ``(N, 243)``, ``y`` —
        ``(N,)``. ``N`` зависит от длины партий.
    """
    rng = random.Random(seed)
    ab_x = AlphaBetaAgent(X, depth=depth)
    ab_o = AlphaBetaAgent(O, depth=depth)

    xs: list[np.ndarray] = []
    ys: list[float] = []

    for game_idx in range(num_games):
        board = Board()
        turn = X
        agent = ab_x
        for _ in range(max_moves):
            move = _epsilon_move(agent, board, rng, epsilon)
            board.place(*move, turn)
            # Сэмпл: позиция после хода turn, оценка от лица turn.
            xs.append(encode(board, turn))
            ys.append(normalize_score(evaluate(board, turn)))
            if winner(board) is not None:
                break
            turn = O if turn == X else X
            agent = ab_o if turn == O else ab_x
        if verbose and (game_idx + 1) % max(1, num_games // 10) == 0:
            print(f"  партий: {game_idx + 1}/{num_games}, сэмплов: {len(xs)}")

    X_arr = np.stack(xs).astype(np.float32) if xs else np.zeros((0, FEATURE_DIM), np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    return X_arr, y_arr
