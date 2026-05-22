"""Проигрывание одной партии двух агентов с поход овым логом.

Используется и в перф-бенчмарке, и в турнире. Не зависит от GUI.
"""

from __future__ import annotations

from time import perf_counter
from typing import TypedDict

from agents.base import Agent
from game.board import O, X, Board
from game.rules import winner


class MoveLog(TypedDict):
    """Запись об одном ходе."""

    player: str          # 'X' | 'O'
    time_ms: float       # время choose_move в миллисекундах
    metric: int          # last_nodes агента (узлы / симуляции / прогоны)
    metric_name: str     # подпись метрики алгоритма


class GameResult(TypedDict):
    """Итог партии."""

    winner: str          # 'X' | 'O' | 'draw'
    num_moves: int
    moves: list[MoveLog]


def play_one_game(
    agent_x: Agent,
    agent_o: Agent,
    max_moves: int = 80,
) -> GameResult:
    """Сыграть одну партию ``agent_x`` (X, первым) против ``agent_o`` (O).

    Время каждого хода измеряется ``time.perf_counter()`` вокруг
    ``choose_move``. По достижении ``max_moves`` без победителя —
    результат ``'draw'``.
    """
    board = Board()
    players: dict[str, Agent] = {X: agent_x, O: agent_o}
    turn = X
    moves: list[MoveLog] = []

    for _ in range(max_moves):
        agent = players[turn]
        t0 = perf_counter()
        move = agent.choose_move(board)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        board.place(*move, turn)
        moves.append(
            MoveLog(
                player=turn,
                time_ms=elapsed_ms,
                metric=int(agent.last_nodes),
                metric_name=agent.metric_name,
            )
        )
        win = winner(board)
        if win is not None:
            return GameResult(winner=win, num_moves=len(moves), moves=moves)
        turn = O if turn == X else X

    return GameResult(winner="draw", num_moves=len(moves), moves=moves)
