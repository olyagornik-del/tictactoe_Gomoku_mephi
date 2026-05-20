"""Minimax с ограничением глубины (без отсечений).

На листе дерева (исчерпана глубина) применяется общая эвристика.
Терминальная победа оценивается с поправкой на остаточную глубину,
чтобы более быстрые победы (и более медленные поражения) ценились выше.
"""

from __future__ import annotations

import math

from agents.base import Agent, nearest_first
from agents.heuristic import TERMINAL_WIN, evaluate
from game.board import Board, Coord, opponent
from game.rules import is_win_at


class MinimaxAgent(Agent):
    """Полный перебор до фиксированной глубины ``depth`` (в полуходах)."""

    name = "Минимакс"
    metric_name = "узлов"

    def __init__(self, symbol: str, depth: int = 3) -> None:
        super().__init__(symbol)
        self.depth = depth

    def choose_move(self, board: Board) -> Coord:
        self.last_nodes = 0
        moves = board.search_window()
        if len(moves) == 1:
            return moves[0]

        best_move: Coord = moves[0]
        best_value = -math.inf
        for move in nearest_first(board, moves):
            board.place(*move, self.symbol)
            self.last_nodes += 1
            value = self._value(board, self.depth - 1, maximizing=False)
            board.undo()
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def _value(self, board: Board, depth: int, maximizing: bool) -> float:
        last = board.last_move
        if last is not None and is_win_at(board, last):
            winner_is_me = board.get(*last) == self.symbol
            score = TERMINAL_WIN + depth
            return score if winner_is_me else -score
        if depth == 0:
            return evaluate(board, self.symbol)

        moves = board.search_window()
        if maximizing:
            value = -math.inf
            for move in moves:
                board.place(*move, self.symbol)
                self.last_nodes += 1
                value = max(value, self._value(board, depth - 1, False))
                board.undo()
            return value
        opp = opponent(self.symbol)
        value = math.inf
        for move in moves:
            board.place(*move, opp)
            self.last_nodes += 1
            value = min(value, self._value(board, depth - 1, True))
            board.undo()
        return value
