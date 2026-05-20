"""Тесты бесконечной доски (game/board.py)."""

import pytest

from game.board import (
    FIRST_MOVE,
    O,
    X,
    Board,
    opponent,
)


def test_empty_board_state() -> None:
    b = Board()
    assert len(b) == 0
    assert b.last_move is None
    assert b.bounding_box() is None
    assert b.search_window() == [FIRST_MOVE]
    assert b.is_empty(0, 0)
    assert b.get(0, 0) is None


def test_place_and_access() -> None:
    b = Board()
    b.place(1, 2, X)
    assert b.get(1, 2) == X
    assert not b.is_empty(1, 2)
    assert (1, 2) in b
    assert len(b) == 1
    assert b.last_move == (1, 2)


def test_place_on_occupied_raises() -> None:
    b = Board()
    b.place(0, 0, X)
    with pytest.raises(ValueError):
        b.place(0, 0, O)


def test_undo_restores_state() -> None:
    b = Board()
    b.place(0, 0, X)
    b.place(3, 1, O)
    coord = b.undo()
    assert coord == (3, 1)
    assert b.is_empty(3, 1)
    assert b.last_move == (0, 0)
    assert len(b) == 1


def test_undo_on_empty_raises() -> None:
    b = Board()
    with pytest.raises(IndexError):
        b.undo()


def test_search_window_around_single_cell() -> None:
    b = Board()
    b.place(0, 0, X)
    window = b.search_window(radius=1)
    # 3x3 окрестность минус занятая центральная клетка = 8 клеток.
    assert len(window) == 8
    assert (0, 0) not in window
    assert (1, 1) in window
    assert (-1, -1) in window
    assert window == sorted(window)


def test_search_window_excludes_occupied_includes_union() -> None:
    b = Board()
    b.place(0, 0, X)
    b.place(5, 5, O)
    window = set(b.search_window(radius=2))
    # Занятые клетки не предлагаются.
    assert (0, 0) not in window
    assert (5, 5) not in window
    # Кандидаты вокруг обеих сыгранных клеток присутствуют.
    assert (2, 2) in window  # рядом с (0,0)
    assert (3, 3) in window  # рядом с (5,5)
    # Далёкая клетка не входит.
    assert (10, 10) not in window


def test_bounding_box_with_negatives() -> None:
    b = Board()
    b.place(-3, 2, X)
    b.place(4, -1, O)
    b.place(0, 0, X)
    assert b.bounding_box() == (-3, -1, 4, 2)


def test_copy_is_independent() -> None:
    b = Board()
    b.place(0, 0, X)
    clone = b.copy()
    clone.place(1, 1, O)
    assert len(b) == 1
    assert len(clone) == 2
    assert b.is_empty(1, 1)
    # Откат в копии не трогает оригинал.
    clone.undo()
    clone.undo()
    assert len(clone) == 0
    assert b.get(0, 0) == X


def test_opponent() -> None:
    assert opponent(X) == O
    assert opponent(O) == X
