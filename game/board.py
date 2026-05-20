"""Бесконечная доска для Гомоку.

Доска хранится как ``dict[(x, y) -> 'X' | 'O']`` и растёт по мере ходов.
Координаты целочисленные и неограниченные (могут быть отрицательными).
"""

from __future__ import annotations
from typing import Iterator

# Типы и константы 

Symbol = str  # 'X' или 'O'
Coord = tuple[int, int]

X: Symbol = "X"
O: Symbol = "O"

# Радиус (в метрике Чебышёва) вокруг сыгранных клеток, в котором
# рассматриваются ходы-кандидаты. Ограничивает ветвление для Minimax/AB.
SEARCH_RADIUS: int = 2

# Ход по умолчанию на пустой доске.
FIRST_MOVE: Coord = (0, 0)


def opponent(symbol: Symbol) -> Symbol:
    """Вернуть символ противника для ``symbol``."""
    return O if symbol == X else X


class Board:
    """Условно-бесконечная доска Гомоку.

    Внутреннее представление — словарь занятых клеток. Пустые клетки
    в словаре отсутствуют. Дополнительно ведётся история ходов, чтобы
    знать последний ход (нужно для проверки победы и для окна 9x9
    перцептрона) и поддерживать дешёвый откат хода (``undo``).
    """

    def __init__(self) -> None:
        self.cells: dict[Coord, Symbol] = {}
        self.history: list[Coord] = []

    #   доступ к клеткам                 

    def get(self, x: int, y: int) -> Symbol | None:
        """Вернуть символ в клетке ``(x, y)`` или ``None``, если пусто."""
        return self.cells.get((x, y))

    def is_empty(self, x: int, y: int) -> bool:
        """Проверить, свободна ли клетка ``(x, y)``."""
        return (x, y) not in self.cells

    def __contains__(self, coord: Coord) -> bool:
        return coord in self.cells

    def __len__(self) -> int:
        """Количество сыгранных клеток."""
        return len(self.cells)

    def __iter__(self) -> Iterator[tuple[Coord, Symbol]]:
        return iter(self.cells.items())

    #   ходы                     

    def place(self, x: int, y: int, symbol: Symbol) -> None:
        """Поставить ``symbol`` в клетку ``(x, y)``.

        :raises ValueError: если клетка уже занята.
        """
        if (x, y) in self.cells:
            raise ValueError(f"Клетка {(x, y)} уже занята: {self.cells[(x, y)]}")
        self.cells[(x, y)] = symbol
        self.history.append((x, y))

    def undo(self) -> Coord:
        """Откатить последний ход и вернуть его координату.

        :raises IndexError: если история ходов пуста.
        """
        coord = self.history.pop()
        del self.cells[coord]
        return coord

    @property
    def last_move(self) -> Coord | None:
        """Координата последнего сделанного хода (или ``None``)."""
        return self.history[-1] if self.history else None

    #   генерация ходов и геометрия             

    def search_window(self, radius: int = SEARCH_RADIUS) -> list[Coord]:
        """Список пустых клеток-кандидатов для следующего хода.

        Это все свободные клетки в радиусе ``radius`` (метрика Чебышёва)
        от любой уже сыгранной клетки. На пустой доске возвращается
        ``[FIRST_MOVE]``. Результат отсортирован для детерминизма
        (важно для воспроизводимости агентов и тестов).
        """
        if not self.cells:
            return [FIRST_MOVE]

        candidates: set[Coord] = set()
        for (cx, cy) in self.cells:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    coord = (cx + dx, cy + dy)
                    if coord not in self.cells:
                        candidates.add(coord)
        return sorted(candidates)

    def window_around(
        self, cx: int, cy: int, radius: int = SEARCH_RADIUS
    ) -> list[Coord]:
        """Пустые клетки в радиусе ``radius`` (Чебышёв) от ``(cx, cy)``.

        Локальная и дешёвая (``O(radius²)``) альтернатива
        :meth:`search_window`: не зависит от числа сыгранных клеток.
        Используется MCTS в rollout/expansion для локальности игры.
        Результат отсортирован для детерминизма.
        """
        result: list[Coord] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                coord = (cx + dx, cy + dy)
                if coord not in self.cells:
                    result.append(coord)
        return result

    def bounding_box(self) -> tuple[int, int, int, int] | None:
        """Габаритный прямоугольник сыгранных клеток.

        :return: ``(min_x, min_y, max_x, max_y)`` либо ``None`` для
            пустой доски. Используется камерой GUI для центрирования.
        """
        if not self.cells:
            return None
        xs = [x for (x, _) in self.cells]
        ys = [y for (_, y) in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    #   копирование

    def copy(self) -> "Board":
        """Поверхностная копия доски, независимая по клеткам и истории.

        Символы — неизменяемые строки, поэтому копировать сами значения
        не нужно; копируются только контейнеры.
        """
        clone = Board()
        clone.cells = dict(self.cells)
        clone.history = list(self.history)
        return clone

    def __repr__(self) -> str:
        return f"Board(moves={len(self.cells)}, last={self.last_move})"
