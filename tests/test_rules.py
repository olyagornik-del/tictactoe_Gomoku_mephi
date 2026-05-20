"""Тесты правил победы Гомоку (game/rules.py)."""

from game.board import O, X, Board
from game.rules import WIN_LENGTH, is_terminal, is_win_at, winner


def _play_line(start: tuple[int, int], step: tuple[int, int], n: int,
               symbol: str) -> Board:
    """Поставить ``n`` камней ``symbol`` подряд по направлению ``step``."""
    b = Board()
    sx, sy = start
    dx, dy = step
    for i in range(n):
        b.place(sx + i * dx, sy + i * dy, symbol)
    return b


def test_no_winner_on_empty_board() -> None:
    b = Board()
    assert winner(b) is None
    assert not is_terminal(b)


def test_horizontal_win() -> None:
    b = _play_line((0, 0), (1, 0), WIN_LENGTH, X)
    assert winner(b) == X
    assert is_terminal(b)


def test_vertical_win() -> None:
    b = _play_line((2, -3), (0, 1), WIN_LENGTH, O)
    assert winner(b) == O
    assert is_terminal(b)


def test_main_diagonal_win() -> None:
    b = _play_line((0, 0), (1, 1), WIN_LENGTH, X)
    assert winner(b) == X


def test_anti_diagonal_win() -> None:
    b = _play_line((0, 0), (1, -1), WIN_LENGTH, O)
    assert winner(b) == O


def test_four_in_a_row_is_not_a_win() -> None:
    b = _play_line((0, 0), (1, 0), WIN_LENGTH - 1, X)
    assert winner(b) is None
    assert not is_terminal(b)


def test_overline_counts_as_win() -> None:
    # Шесть подряд — всё ещё победа (свободное правило Гомоку).
    b = _play_line((0, 0), (1, 0), WIN_LENGTH + 1, X)
    assert winner(b) == X


def test_win_detected_when_last_move_in_middle() -> None:
    # Линия из 5, но последним ставится центральный камень — победа
    # должна определяться счётом в обе стороны.
    b = Board()
    b.place(0, 0, X)
    b.place(1, 0, X)
    b.place(3, 0, X)
    b.place(4, 0, X)
    assert winner(b) is None
    b.place(2, 0, X)  # замыкающий ход посередине
    assert winner(b) == X


def test_mixed_symbols_not_a_win() -> None:
    b = Board()
    for i in range(WIN_LENGTH):
        b.place(i, 0, X if i % 2 == 0 else O)
    assert winner(b) is None


def test_is_win_at_helper() -> None:
    b = _play_line((0, 0), (1, 0), WIN_LENGTH, X)
    assert is_win_at(b, (0, 0))
    assert is_win_at(b, (4, 0))
    b2 = Board()
    b2.place(0, 0, X)
    assert not is_win_at(b2, (0, 0))
    assert not is_win_at(b2, (9, 9))  # пустая клетка
