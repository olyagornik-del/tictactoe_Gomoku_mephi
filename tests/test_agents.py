"""Headless-проверки агентов: само-игра и базовая компетентность.

Функция :func:`run_match` переиспользуется в тестах последующих фаз
(MCTS, перцептрон).
"""

from __future__ import annotations

from agents.alphabeta import AlphaBetaAgent
from agents.base import Agent, RandomAgent
from game.board import O, X, Board
from game.rules import is_win_at, winner


def run_match(agent_x: Agent, agent_o: Agent, max_moves: int = 80) -> str | None:
    """Сыграть одну партию. Вернуть символ победителя или ``None``.

    ``agent_x`` ходит первым (символ ``X``), ``agent_o`` — вторым.
    """
    board = Board()
    players = {X: agent_x, O: agent_o}
    turn = X
    for _ in range(max_moves):
        move = players[turn].choose_move(board)
        board.place(*move, turn)
        if winner(board) is not None:
            return turn
        turn = O if turn == X else X
    return None


def test_alphabeta_completes_open_four() -> None:
    """AB обязан замкнуть пятёрку, когда у него открытая четвёрка."""
    b = Board()
    for i in range(4):
        b.place(i, 0, X)  # XXXX_ , ходит X
    agent = AlphaBetaAgent(X, depth=2)
    move = agent.choose_move(b)
    b.place(*move, X)
    assert is_win_at(b, move)


def test_alphabeta_blocks_opponent_four() -> None:
    """AB обязан закрыть единственный конец четвёрки оппонента.

    Левый конец уже перекрыт камнем X, поэтому O выигрывает только
    ходом ``(4, 0)`` — его и должен заблокировать AB. (Открытую
    четвёрку заблокировать в принципе нельзя — это победная угроза.)
    """
    b = Board()
    b.place(-1, 0, X)  # перекрыт левый конец
    for i in range(4):
        b.place(i, 0, O)  # X OOOO _
    agent = AlphaBetaAgent(X, depth=2)
    move = agent.choose_move(b)
    assert move == (4, 0)


def test_alphabeta_beats_random_as_first_player() -> None:
    for seed in (1, 2, 3):
        winner_sym = run_match(
            AlphaBetaAgent(X, depth=2), RandomAgent(O, seed=seed)
        )
        assert winner_sym == X, f"AB не выиграл (seed={seed}): {winner_sym}"


def test_alphabeta_beats_random_as_second_player() -> None:
    wins = 0
    for seed in (1, 2, 3):
        winner_sym = run_match(
            RandomAgent(X, seed=seed), AlphaBetaAgent(O, depth=2)
        )
        if winner_sym == O:
            wins += 1
    assert wins >= 2, f"AB вторым выиграл лишь {wins}/3"
