"""Эксперимент B: сила игры — круговой турнир четырёх агентов.

Все 6 пар, по 30 партий на пару (15 — первый агент за X, 15 — за O).
Параметры по умолчанию: minimax depth=3, alphabeta depth=3,
mcts simulations=1000, perceptron — как есть.

Результат: ``results/tournament.csv``. Помимо столбцов из ТЗ добавлен
``first_color`` (X|O) — без него нельзя симметризовать результаты по
цвету на этапе анализа.
"""

from __future__ import annotations

import csv
import os
from itertools import combinations

from agents.alphabeta import AlphaBetaAgent
from agents.base import Agent
from agents.mcts import MCTSAgent
from agents.minimax import MinimaxAgent
from agents.perceptron import (
    DEFAULT_MODEL_PATH,
    Perceptron,
    PerceptronAgent,
    board_to_features,
)
from experiments.runner import play_one_game
from game.board import O, X

RESULTS_DIR = "results"
TOURNAMENT_CSV = os.path.join(RESULTS_DIR, "tournament.csv")
GAMES_PER_COLOR = 15      # 15 + 15 = 30 на пару
MAX_MOVES = 80

# Параметры по умолчанию и человекочитаемая подпись.
PARAM_LABEL = {
    "minimax": "depth=3",
    "alphabeta": "depth=3",
    "mcts": "sims=1000",
    "perceptron": "-",
}


def _make_agent(
    algorithm: str,
    symbol: str,
    seed: int,
    perceptron_model: Perceptron | None,
) -> Agent:
    """Свежий агент. MCTS сидируется ``seed`` (различие между партиями)."""
    if algorithm == "minimax":
        return MinimaxAgent(symbol, depth=3)
    if algorithm == "alphabeta":
        return AlphaBetaAgent(symbol, depth=3)
    if algorithm == "mcts":
        return MCTSAgent(symbol, simulations=1000, seed=seed)
    if algorithm == "perceptron":
        assert perceptron_model is not None
        return PerceptronAgent(symbol, perceptron_model, board_to_features)
    raise ValueError(algorithm)


def _aggregate_times(moves: list[dict], first_color: str) -> tuple[float, float, float, float]:
    """Суммарное и среднее время на ход для первого и второго агента."""
    second_color = O if first_color == X else X
    t_first = sum(m["time_ms"] for m in moves if m["player"] == first_color)
    t_second = sum(m["time_ms"] for m in moves if m["player"] == second_color)
    n_first = sum(1 for m in moves if m["player"] == first_color)
    n_second = sum(1 for m in moves if m["player"] == second_color)
    t_first_s = t_first / 1000.0
    t_second_s = t_second / 1000.0
    avg_first = t_first_s / n_first if n_first else 0.0
    avg_second = t_second_s / n_second if n_second else 0.0
    return t_first_s, t_second_s, avg_first, avg_second


def run() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    algorithms = ["minimax", "alphabeta", "mcts", "perceptron"]
    perceptron_model: Perceptron | None = None
    if os.path.exists(DEFAULT_MODEL_PATH):
        perceptron_model = Perceptron.load_weights(DEFAULT_MODEL_PATH)
    else:
        print(f"[tournament] {DEFAULT_MODEL_PATH} не найден — "
              "пары с перцептроном пропускаются.")
        algorithms.remove("perceptron")

    rows: list[dict] = []
    game_id = 0

    for algo_a, algo_b in combinations(algorithms, 2):
        for color_round in (X, O):  # первый агент (A) играет сначала X, потом O
            for _ in range(GAMES_PER_COLOR):
                game_id += 1
                if color_round == X:
                    first_color = X
                    agent_x = _make_agent(algo_a, X, game_id, perceptron_model)
                    agent_o = _make_agent(algo_b, O, game_id, perceptron_model)
                else:
                    first_color = O
                    agent_x = _make_agent(algo_b, X, game_id, perceptron_model)
                    agent_o = _make_agent(algo_a, O, game_id, perceptron_model)

                result = play_one_game(agent_x, agent_o, max_moves=MAX_MOVES)
                t_f, t_s, avg_f, avg_s = _aggregate_times(
                    result["moves"], first_color
                )
                rows.append({
                    "game_id": game_id,
                    "algo_first": algo_a,
                    "algo_second": algo_b,
                    "param_first": PARAM_LABEL[algo_a],
                    "param_second": PARAM_LABEL[algo_b],
                    "first_color": first_color,
                    "winner": result["winner"],
                    "num_moves": result["num_moves"],
                    "time_first_total_sec": round(t_f, 4),
                    "time_second_total_sec": round(t_s, 4),
                    "avg_time_per_move_first": round(avg_f, 4),
                    "avg_time_per_move_second": round(avg_s, 4),
                })
        print(f"[tournament] {algo_a} vs {algo_b}: 30 партий сыграно")

    with open(TOURNAMENT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game_id", "algo_first", "algo_second", "param_first",
            "param_second", "first_color", "winner", "num_moves",
            "time_first_total_sec", "time_second_total_sec",
            "avg_time_per_move_first", "avg_time_per_move_second",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"готово: {TOURNAMENT_CSV}, {len(rows)} строк")


if __name__ == "__main__":
    run()
