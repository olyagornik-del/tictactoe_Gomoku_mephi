"""Эвристическая оценка позиции (общая для Minimax и Alpha-Beta).

Оценка = (сумма весов паттернов своих камней) − (сумма весов
паттернов оппонента), просматриваемая по 4 направлениям. Веса
вынесены в константы — их легко тюнить.
"""

from __future__ import annotations

from game.board import Board, Symbol, opponent
from game.rules import DIRECTIONS, WIN_LENGTH

# Веса паттернов ----------------------------------------------------------

FIVE: int = 1_000_000          # пять подряд (победа)
OPEN_FOUR: int = 100_000       # xxxx с двумя свободными концами
CLOSED_FOUR: int = 10_000      # четвёрка с одним свободным концом
OPEN_THREE: int = 5_000        # открытая тройка
CLOSED_THREE: int = 500        # закрытая тройка
OPEN_TWO: int = 100            # открытая двойка
CLOSED_TWO: int = 10           # закрытая двойка

#: Терминальная ценность выигранной позиции для поиска. Заведомо больше
#: любой суммы эвристических весов, чтобы реальная победа всегда
#: доминировала над «красивой» оценкой.
TERMINAL_WIN: int = 10_000_000


def _pattern_weight(length: int, open_ends: int) -> int:
    """Вес одной непрерывной линии длины ``length``.

    ``open_ends`` — сколько из двух концов линии свободны (пустая
    клетка). Линия с нулём свободных концов и длиной < ``WIN_LENGTH``
    бесполезна (её нельзя достроить до пяти) — вес 0.
    """
    if length >= WIN_LENGTH:
        return FIVE
    if length == 4:
        return (0, CLOSED_FOUR, OPEN_FOUR)[open_ends]
    if length == 3:
        return (0, CLOSED_THREE, OPEN_THREE)[open_ends]
    if length == 2:
        return (0, CLOSED_TWO, OPEN_TWO)[open_ends]
    return 0  # одиночный камень не оценивается


def _score_for(board: Board, symbol: Symbol) -> int:
    """Суммарный вес всех линий символа ``symbol``.

    Каждая максимальная линия учитывается один раз: обрабатывается
    только её «начальная» клетка (предыдущая по направлению — не того
    же символа).
    """
    score = 0
    cells = board.cells
    for (x, y), s in cells.items():
        if s != symbol:
            continue
        for dx, dy in DIRECTIONS:
            # Пропускаем, если это не начало линии.
            if cells.get((x - dx, y - dy)) == symbol:
                continue
            length = 1
            cx, cy = x + dx, y + dy
            while cells.get((cx, cy)) == symbol:
                length += 1
                cx += dx
                cy += dy
            before = cells.get((x - dx, y - dy))  # None → свободный конец
            after = cells.get((cx, cy))
            open_ends = (before is None) + (after is None)
            score += _pattern_weight(length, open_ends)
    return score


def evaluate(board: Board, symbol: Symbol) -> int:
    """Оценка позиции с точки зрения игрока ``symbol``.

    Положительное значение — преимущество ``symbol``, отрицательное —
    преимущество оппонента.
    """
    return _score_for(board, symbol) - _score_for(board, opponent(symbol))
