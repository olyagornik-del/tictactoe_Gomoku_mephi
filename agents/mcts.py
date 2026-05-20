"""Monte Carlo Tree Search с UCB1 для бесконечного Гомоку.

Особенности под бесконечную доску (референсы написаны под 3x3/Реверси):

* **Ограниченное ветвление** — кандидаты берутся из окна поиска,
  упорядочиваются от центра и обрезаются до :data:`MAX_CHILDREN`.
  Без этого фактор ветвления неограничен.
* **Управляемый rollout** — равномерно случайная доигровка на
  бесконечной доске почти никогда не даёт пять в ряд, поэтому
  симуляции бесполезны. Политика playout сначала берёт немедленный
  выигрыш, затем блокирует немедленную угрозу, иначе ходит случайно.
* **Ограничение глубины** rollout (:data:`ROLLOUT_DEPTH`): если за
  лимит победитель не определён — результат считается ничейным (0.5).
"""

from __future__ import annotations

import math
import random

from agents.base import Agent, nearest_first
from game.board import Board, Coord, Symbol, opponent
from game.rules import DIRECTIONS, is_win_at

#: Константа исследования в UCB1 (sqrt(2) — классический выбор).
C_UCB1: float = math.sqrt(2)
#: Максимум плиев в одной доигровке до объявления ничьей.
ROLLOUT_DEPTH: int = 40
#: Сколько ходов-кандидатов (ближайших к центру) рассматривать в узле.
MAX_CHILDREN: int = 16


class _Node:
    """Узел дерева поиска.

    ``player_just_moved`` — символ игрока, чей ход привёл в этот узел;
    относительно него считается награда (это игрок, который выбирает
    данный узел в родителе, поэтому UCB1-эксплуатация корректна).
    """

    __slots__ = (
        "move",
        "player_just_moved",
        "parent",
        "children",
        "untried",
        "visits",
        "wins",
    )

    def __init__(
        self,
        move: Coord | None,
        player_just_moved: Symbol,
        parent: "_Node | None",
    ) -> None:
        self.move = move
        self.player_just_moved = player_just_moved
        self.parent = parent
        self.children: list[_Node] = []
        self.untried: list[Coord] = []
        self.visits: int = 0
        self.wins: float = 0.0


def _candidates(board: Board) -> list[Coord]:
    """Локальные ходы-кандидаты вокруг последнего хода.

    Берём пустые клетки в радиусе 2 от последнего хода (на пустой
    доске — от ``(0, 0)``), сортируем от центра, обрезаем до
    :data:`MAX_CHILDREN`. Локальность естественна для Гомоку и делает
    rollout дешёвым (не зависит от числа камней на доске).
    """
    cx, cy = board.last_move if board.last_move is not None else (0, 0)
    window = board.window_around(cx, cy)
    if not window:  # вся локальная окрестность занята — берём глобальное окно
        window = board.search_window()
    return nearest_first(board, window)[:MAX_CHILDREN]


def _winning_move(board: Board, moves: list[Coord], symbol: Symbol) -> Coord | None:
    """Первый ход из ``moves``, дающий ``symbol`` немедленную победу."""
    for m in moves:
        board.place(*m, symbol)
        win = is_win_at(board, m)
        board.undo()
        if win:
            return m
    return None


def _best_line(board: Board, x: int, y: int, symbol: Symbol) -> int:
    """Длина самой длинной линии ``symbol`` через ``(x, y)``.

    Клетка считается своей; смотрим в обе стороны по 4 направлениям.
    Дешёвая (``O(1)``) мера «срочности» хода для политики playout.
    """
    cells = board.cells
    best = 1
    for dx, dy in DIRECTIONS:
        length = 1
        for sign in (1, -1):
            cx, cy = x + sign * dx, y + sign * dy
            while cells.get((cx, cy)) == symbol:
                length += 1
                cx += sign * dx
                cy += sign * dy
        best = max(best, length)
    return best


class MCTSAgent(Agent):
    """MCTS с UCB1 и конфигурируемым числом симуляций."""

    name = "MCTS"
    metric_name = "симуляций"

    def __init__(
        self,
        symbol: Symbol,
        simulations: int = 800,
        exploration: float = C_UCB1,
        seed: int | None = None,
    ) -> None:
        super().__init__(symbol)
        self.simulations = simulations
        self.c = exploration
        self._rng = random.Random(seed)

    # --- публичный интерфейс --------------------------------------------

    def choose_move(self, board: Board) -> Coord:
        self.last_nodes = 0
        root_moves = board.search_window()
        if len(root_moves) == 1:
            return root_moves[0]

        # Тактический предохранитель (один проход по всей доске):
        # забрать немедленную победу, иначе закрыть немедленную угрозу.
        opp = opponent(self.symbol)
        win = _winning_move(board, root_moves, self.symbol)
        if win is not None:
            return win
        block = _winning_move(board, root_moves, opp)
        if block is not None:
            return block

        root = _Node(None, opp, None)
        root.untried = _candidates(board)

        for _ in range(self.simulations):
            self.last_nodes += 1
            self._run_simulation(board, root)

        # Робастный выбор: больше всего посещений (при равенстве — winrate).
        best = max(
            root.children,
            key=lambda c: (c.visits, c.wins / c.visits if c.visits else 0.0),
        )
        return best.move  # type: ignore[return-value]

    # --- одна симуляция -------------------------------------------------

    def _run_simulation(self, board: Board, root: _Node) -> None:
        sim = board.copy()
        node = root
        player = self.symbol  # кто ходит в корне

        # Selection: спускаемся по полностью раскрытым узлам.
        while not node.untried and node.children:
            node = self._select(node)
            sim.place(*node.move, player)  # type: ignore[misc]
            player = opponent(player)

        winner = sim.get(*sim.last_move) if (
            sim.last_move and is_win_at(sim, sim.last_move)
        ) else None

        # Expansion: раскрываем один новый ход.
        if winner is None and node.untried:
            move = node.untried.pop(0)
            sim.place(*move, player)
            child = _Node(move, player, node)
            node.children.append(child)
            node = child
            if is_win_at(sim, move):
                winner = player
            else:
                child.untried = _candidates(sim)
            player = opponent(player)

        # Rollout (если позиция ещё не терминальная).
        if winner is None:
            winner = self._rollout(sim, player)

        # Backpropagation.
        while node is not None:
            node.visits += 1
            if winner is None:
                node.wins += 0.5
            elif node.player_just_moved == winner:
                node.wins += 1.0
            node = node.parent

    # --- составляющие ---------------------------------------------------

    def _select(self, node: _Node) -> _Node:
        """Выбрать ребёнка с максимальным UCB1."""
        log_n = math.log(node.visits) if node.visits > 0 else 0.0
        best_node = node.children[0]
        best_val = -math.inf
        for child in node.children:
            if child.visits == 0:
                value = math.inf
            else:
                exploit = child.wins / child.visits
                explore = self.c * math.sqrt(log_n / child.visits)
                value = exploit + explore
            if value > best_val:
                best_val = value
                best_node = child
        return best_node

    def _rollout(self, sim: Board, player: Symbol) -> Symbol | None:
        """Доигровка управляемой политикой. Возвращает победителя/``None``."""
        for _ in range(ROLLOUT_DEPTH):
            move = self._rollout_policy(sim, player)
            sim.place(*move, player)
            if is_win_at(sim, move):
                return player
            player = opponent(player)
        return None

    def _rollout_policy(self, sim: Board, player: Symbol) -> Coord:
        """Информированная политика доигровки.

        Приоритет: немедленный выигрыш → блок немедленной угрозы →
        жадно по длине линий (продлить свою / перекрыть чужую самую
        длинную линию). Среди равных лучших — случайный выбор. Это
        обычный «тяжёлый» playout (не общая эвристика Minimax/AB).
        """
        cand = _candidates(sim)
        win = _winning_move(sim, cand, player)
        if win is not None:
            return win
        opp = opponent(player)
        block = _winning_move(sim, cand, opp)
        if block is not None:
            return block

        best_score = -1
        best: list[Coord] = []
        for m in cand:
            score = max(
                _best_line(sim, m[0], m[1], player),
                _best_line(sim, m[0], m[1], opp),
            )
            if score > best_score:
                best_score = score
                best = [m]
            elif score == best_score:
                best.append(m)
        return self._rng.choice(best)
