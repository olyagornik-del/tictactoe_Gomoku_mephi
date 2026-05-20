"""Абстрактный агент и случайный игрок.

Все ИИ-алгоритмы — отдельные классы, наследующие :class:`Agent`.
Агент знает, каким символом он ходит; смена алгоритма в GUI означает
создание нового агента с тем же символом без сброса доски.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from game.board import Board, Coord, Symbol


def nearest_first(board: Board, moves: list[Coord]) -> list[Coord]:
    """Упорядочить ходы по близости к последнему ходу (центр — раньше).

    Не меняет результат полного перебора, но улучшает отсечения
    Alpha-Beta: релевантные ходы рассматриваются первыми.
    """
    last = board.last_move
    if last is None:
        return moves
    lx, ly = last
    return sorted(moves, key=lambda m: max(abs(m[0] - lx), abs(m[1] - ly)))


class Agent(ABC):
    """Базовый полиморфный агент.

    :param symbol: символ, которым ходит агент (``'X'`` или ``'O'``).

    Атрибут :attr:`last_nodes` после каждого хода содержит метрику
    трудозатрат — число просмотренных узлов (Minimax/AB), симуляций
    (MCTS) или прогонов сети (перцептрон). Профайлер (фаза 5) читает
    это поле. :attr:`metric_name` — подпись метрики для GUI.
    """

    #: Человекочитаемое имя алгоритма (для переключателя в GUI).
    name: str = "Агент"
    #: Подпись единицы трудозатрат для боковой панели.
    metric_name: str = "узлов"

    def __init__(self, symbol: Symbol) -> None:
        self.symbol: Symbol = symbol
        self.last_nodes: int = 0

    @abstractmethod
    def choose_move(self, board: Board) -> Coord:
        """Выбрать ход ``(x, y)`` для текущего состояния доски.

        Реализация не обязана модифицировать ``board`` необратимо:
        фактический ход на доске делает игровой цикл.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol={self.symbol!r})"


class RandomAgent(Agent):
    """Случайный игрок: равномерный выбор из окна поиска.

    Служит слабым эталоном для headless-проверок («ИИ должен
    обыгрывать Random»). Принимает необязательный ``seed`` для
    воспроизводимости тестов.
    """

    name = "Случайный"
    metric_name = "ходов"

    def __init__(self, symbol: Symbol, seed: int | None = None) -> None:
        super().__init__(symbol)
        self._rng = random.Random(seed)

    def choose_move(self, board: Board) -> Coord:
        moves = board.search_window()
        self.last_nodes = len(moves)
        return self._rng.choice(moves)
