"""Тесты эвристической оценки (agents/heuristic.py)."""

from agents.heuristic import (
    CLOSED_TWO,
    FIVE,
    OPEN_THREE,
    OPEN_TWO,
    evaluate,
)
from game.board import O, X, Board


def test_empty_board_is_zero() -> None:
    assert evaluate(Board(), X) == 0


def test_open_two() -> None:
    b = Board()
    b.place(0, 0, X)
    b.place(1, 0, X)
    assert evaluate(b, X) == OPEN_TWO
    # Симметрия: с точки зрения O это зеркальная оценка.
    assert evaluate(b, O) == -OPEN_TWO


def test_closed_two() -> None:
    b = Board()
    b.place(-1, 0, O)  # блокирует левый конец
    b.place(0, 0, X)
    b.place(1, 0, X)
    # Линия X длиной 2 с одним свободным концом.
    assert evaluate(b, X) == CLOSED_TWO


def test_open_three() -> None:
    b = Board()
    for i in range(3):
        b.place(i, 0, X)
    assert evaluate(b, X) == OPEN_THREE


def test_five_in_a_row() -> None:
    b = Board()
    for i in range(5):
        b.place(i, 0, X)
    assert evaluate(b, X) == FIVE


def test_symmetry_on_mixed_board() -> None:
    b = Board()
    b.place(0, 0, X)
    b.place(1, 0, X)
    b.place(0, 1, O)
    b.place(3, 3, O)
    b.place(4, 3, O)
    assert evaluate(b, X) == -evaluate(b, O)
