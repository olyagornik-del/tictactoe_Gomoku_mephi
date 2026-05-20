"""Headless-проверки MCTS (agents/mcts.py).

Тактические тесты мгновенны (срабатывает предохранитель до симуляций).
Полные партии используют небольшое число симуляций ради скорости;
агенты сидированы, поэтому исходы детерминированы.

О конкурентности: чистый MCTS с лёгким playout в Гомоку слабее
эвристического минимакса (ожидаемый и поучительный результат). Здесь
он уверенно обыгрывает Random и AlphaBeta малой глубины; против AB
глубины 2 проигрывает, но затягивает партию.
"""

from __future__ import annotations

from agents.alphabeta import AlphaBetaAgent
from agents.base import RandomAgent
from agents.mcts import MCTSAgent
from game.board import O, X, Board
from game.rules import is_win_at
from tests.test_agents import run_match


def test_mcts_completes_open_four() -> None:
    """Предохранитель MCTS обязан забрать немедленную победу."""
    b = Board()
    for i in range(4):
        b.place(i, 0, X)
    move = MCTSAgent(X, simulations=50, seed=0).choose_move(b)
    b.place(*move, X)
    assert is_win_at(b, move)


def test_mcts_blocks_opponent_four() -> None:
    """Предохранитель MCTS обязан закрыть единственный конец четвёрки."""
    b = Board()
    b.place(-1, 0, X)
    for i in range(4):
        b.place(i, 0, O)
    move = MCTSAgent(X, simulations=50, seed=0).choose_move(b)
    assert move == (4, 0)


def test_mcts_beats_random_as_first_player() -> None:
    assert run_match(MCTSAgent(X, simulations=200, seed=1),
                     RandomAgent(O, seed=1)) == X


def test_mcts_beats_random_as_second_player() -> None:
    assert run_match(RandomAgent(X, seed=1),
                     MCTSAgent(O, simulations=200, seed=1)) == O


def test_mcts_beats_shallow_alphabeta() -> None:
    """Конкурентность: MCTS обыгрывает поисковый AB глубины 1."""
    assert run_match(MCTSAgent(X, simulations=300, seed=1),
                     AlphaBetaAgent(O, depth=1)) == X
