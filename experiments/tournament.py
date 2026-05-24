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
    PIXEL_MODEL_PATH,
    Perceptron,
    PerceptronAgent,
    move_features,
    pixel_features,
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
    "perceptron": "eng",        # инженерные признаки, без предохранителя
    "perceptron_pixel": "pix",  # пиксельные признаки + предохранитель
}


def _make_agent(
    algorithm: str,
    symbol: str,
    seed: int,
    perceptron_models: dict[str, Perceptron],
) -> Agent:
    """Свежий агент. MCTS сидируется ``seed`` (различие между партиями)."""
    if algorithm == "minimax":
        return MinimaxAgent(symbol, depth=3)
    if algorithm == "alphabeta":
        return AlphaBetaAgent(symbol, depth=3)
    if algorithm == "mcts":
        return MCTSAgent(symbol, simulations=1000, seed=seed)
    if algorithm == "perceptron":
        # Инженерные признаки → защищается сама, предохранитель не нужен.
        return PerceptronAgent(
            symbol, perceptron_models["perceptron"], move_features,
            tactical=False,
        )
    if algorithm == "perceptron_pixel":
        # Пиксельные признаки → нужен тактический предохранитель.
        return PerceptronAgent(
            symbol, perceptron_models["perceptron_pixel"], pixel_features,
            tactical=True,
        )
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


FIELDNAMES = [
    "game_id", "algo_first", "algo_second", "param_first",
    "param_second", "first_color", "winner", "num_moves",
    "time_first_total_sec", "time_second_total_sec",
    "avg_time_per_move_first", "avg_time_per_move_second",
]


def _completed_game_ids(path: str) -> set[int]:
    """Множество уже сыгранных game_id из существующего CSV (для resume)."""
    done: set[int] = set()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    done.add(int(row["game_id"]))
                except (KeyError, ValueError):
                    pass
    return done


def run(resume: bool = True) -> None:
    """Круговой турнир с инкрементальной записью и возобновлением.

    Каждая партия дописывается в CSV сразу (flush), поэтому прерывание
    процесса не теряет прогресс. При повторном запуске уже сыгранные
    ``game_id`` пропускаются.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    algorithms = ["minimax", "alphabeta", "mcts"]
    perceptron_models: dict[str, Perceptron] = {}
    if os.path.exists(DEFAULT_MODEL_PATH):
        perceptron_models["perceptron"] = Perceptron.load_weights(DEFAULT_MODEL_PATH)
        algorithms.append("perceptron")
    else:
        print(f"[tournament] {DEFAULT_MODEL_PATH} не найден — "
              "инженерный перцептрон пропускается.")
    if os.path.exists(PIXEL_MODEL_PATH):
        perceptron_models["perceptron_pixel"] = Perceptron.load_weights(PIXEL_MODEL_PATH)
        algorithms.append("perceptron_pixel")
    else:
        print(f"[tournament] {PIXEL_MODEL_PATH} не найден — "
              "пиксельный перцептрон пропускается.")

    completed = _completed_game_ids(TOURNAMENT_CSV) if resume else set()
    has_data = os.path.exists(TOURNAMENT_CSV) and os.path.getsize(TOURNAMENT_CSV) > 0
    append = resume and has_data
    if completed:
        print(f"[tournament] возобновление: уже сыграно {len(completed)} партий")

    f = open(TOURNAMENT_CSV, "a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not append:
        writer.writeheader()
        f.flush()

    written = len(completed)
    game_id = 0
    try:
        for algo_a, algo_b in combinations(algorithms, 2):
            for color_round in (X, O):  # A играет сначала X, потом O
                for _ in range(GAMES_PER_COLOR):
                    game_id += 1
                    if game_id in completed:
                        continue
                    if color_round == X:
                        first_color = X
                        agent_x = _make_agent(algo_a, X, game_id, perceptron_models)
                        agent_o = _make_agent(algo_b, O, game_id, perceptron_models)
                    else:
                        first_color = O
                        agent_x = _make_agent(algo_b, X, game_id, perceptron_models)
                        agent_o = _make_agent(algo_a, O, game_id, perceptron_models)

                    result = play_one_game(agent_x, agent_o, max_moves=MAX_MOVES)
                    t_f, t_s, avg_f, avg_s = _aggregate_times(
                        result["moves"], first_color
                    )
                    writer.writerow({
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
                    f.flush()  # прогресс переживает прерывание
                    written += 1
            print(f"[tournament] {algo_a} vs {algo_b}: пара готова "
                  f"(всего строк: {written})")
    finally:
        f.close()

    print(f"готово: {TOURNAMENT_CSV}, {written} строк")


if __name__ == "__main__":
    run()
