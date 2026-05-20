"""Правила Гомоку: условие победы и завершение партии.

Победа — пять (или более) одинаковых символов подряд по горизонтали,
вертикали или любой из двух диагоналей. Для скорости проверка ведётся
только относительно последнего сделанного хода: только он мог создать
новую выигрышную линию.
"""

from __future__ import annotations

from game.board import Board, Coord, Symbol

#: Длина линии, необходимая для победы.
WIN_LENGTH: int = 5

#: Четыре направления линий (вторая половина получается отражением):
#: горизонталь, вертикаль, главная и побочная диагонали.
DIRECTIONS: tuple[Coord, ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


def _count_in_direction(
    board: Board, x: int, y: int, dx: int, dy: int, symbol: Symbol
) -> int:
    """Сколько подряд клеток с ``symbol`` начиная от ``(x+dx, y+dy)``.

    Сама клетка ``(x, y)`` не учитывается — она прибавляется вызывающим.
    """
    count = 0
    cx, cy = x + dx, y + dy
    while board.get(cx, cy) == symbol:
        count += 1
        cx += dx
        cy += dy
    return count


def is_win_at(board: Board, coord: Coord) -> bool:
    """Создаёт ли камень в ``coord`` линию длиной ``WIN_LENGTH`` и более.

    Клетка ``coord`` должна быть занята. Проверяются обе стороны каждого
    из четырёх направлений.
    """
    x, y = coord
    symbol = board.get(x, y)
    if symbol is None:
        return False
    for dx, dy in DIRECTIONS:
        total = (
            1
            + _count_in_direction(board, x, y, dx, dy, symbol)
            + _count_in_direction(board, x, y, -dx, -dy, symbol)
        )
        if total >= WIN_LENGTH:
            return True
    return False


def winner(board: Board) -> Symbol | None:
    """Вернуть символ победителя или ``None``.

    Проверяется только последний ход — единственный, который мог
    завершить партию.
    """
    last = board.last_move
    if last is None:
        return None
    if is_win_at(board, last):
        return board.get(*last)
    return None


def is_terminal(board: Board) -> bool:
    """Завершена ли партия (есть победитель).

    На бесконечной доске ничьей по заполнению не бывает, поэтому
    единственное условие конца — чья-то победа.
    """
    return winner(board) is not None
