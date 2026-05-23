"""Эксперимент A: скорость агентов на фиксированных позициях.

Для каждой комбинации (позиция, алгоритм, параметр) делается 5 замеров
времени хода (агент играет за O). Минимаксу глубокий перебор может не
успеть — действует таймаут 60 секунд на один ход (при срыве — NaN).

Результат: ``results/perf.csv`` со столбцами
``position_id, algorithm, param_name, param_value, run, time_sec,
metric_value``.
"""

from __future__ import annotations

import csv
import math
import os
import signal
from time import perf_counter

from agents.alphabeta import AlphaBetaAgent
from agents.mcts import MCTSAgent
from agents.minimax import MinimaxAgent
from agents.perceptron import DEFAULT_MODEL_PATH, PerceptronAgent
from game.board import O, X, Board

RESULTS_DIR = "results"
PERF_CSV = os.path.join(RESULTS_DIR, "perf.csv")
REPEATS = 5
TIMEOUT_SEC = 60

# --- фиксированные позиции (списки ходов из стартовой; после применения
#     ход всегда за O — длина списков нечётная) ----------------------------
POSITIONS: dict[str, list[tuple[int, int]]] = {
    # Дебют: X в центре, O делает первый ответ (окно ~24 клетки).
    "pos1_opening": [(0, 0)],
    # Миттельшпиль: плотный кластор вокруг центра без готовых угроз.
    "pos2_midgame": [
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 2),
        (2, 1), (-1, -1), (0, 2), (2, 0),
    ],
    # Тактика: у X открытая тройка (0,0)-(1,0)-(2,0), O обязан блокировать.
    "pos3_tactical": [
        (0, 0), (0, 5), (1, 0), (1, 5), (2, 0),
    ],
}


def _build_board(moves: list[tuple[int, int]]) -> Board:
    """Воспроизвести позицию: ходы чередуются X, O, X, O, …"""
    board = Board()
    turn = X
    for mv in moves:
        board.place(*mv, turn)
        turn = O if turn == X else X
    return board


# --- таймаут на один ход (SIGALRM, главный поток) -----------------------

class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ANN001
    raise _Timeout()


def _timed_move(agent, board: Board, timeout_sec: int) -> tuple[float, float]:
    """Сделать ход с таймаутом. Вернуть ``(time_sec, metric)`` или NaN."""
    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_sec)
    try:
        t0 = perf_counter()
        agent.choose_move(board)
        elapsed = perf_counter() - t0
        return elapsed, float(agent.last_nodes)
    except _Timeout:
        return math.nan, math.nan
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# --- конфигурации агентов ----------------------------------------------

def _agent_configs() -> list[tuple[str, str, list[int]]]:
    """``(algorithm, param_name, [param_values])``."""
    configs = [
        ("minimax", "depth", [1, 2, 3, 4]),
        ("alphabeta", "depth", [1, 2, 3, 4, 5, 6]),
        ("mcts", "simulations", [200, 500, 1000, 2000]),
    ]
    if os.path.exists(DEFAULT_MODEL_PATH):
        configs.append(("perceptron", "none", [0]))  # один замер, без параметра
    else:
        print(
            f"[perf] {DEFAULT_MODEL_PATH} не найден — перцептрон пропущен."
        )
    return configs


def _make_agent(algorithm: str, param_value: int):
    """Свежий агент за O для заданного алгоритма/параметра."""
    if algorithm == "minimax":
        return MinimaxAgent(O, depth=param_value)
    if algorithm == "alphabeta":
        return AlphaBetaAgent(O, depth=param_value)
    if algorithm == "mcts":
        # сид фиксируем для воспроизводимости времени
        return MCTSAgent(O, simulations=param_value, seed=42)
    if algorithm == "perceptron":
        return PerceptronAgent.from_disk(O)
    raise ValueError(algorithm)


def run() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []

    for pos_id, moves in POSITIONS.items():
        for algorithm, param_name, values in _agent_configs():
            for param_value in values:
                # Если конфиг детерминированно сорвался по таймауту на
                # первом повторе — остальные повторы тоже NaN (без прогона).
                timed_out = False
                for run_idx in range(1, REPEATS + 1):
                    if timed_out:
                        time_sec, metric = math.nan, math.nan
                    else:
                        board = _build_board(moves)
                        agent = _make_agent(algorithm, param_value)
                        time_sec, metric = _timed_move(agent, board, TIMEOUT_SEC)
                        if math.isnan(time_sec):
                            timed_out = True
                    rows.append({
                        "position_id": pos_id,
                        "algorithm": algorithm,
                        "param_name": param_name,
                        "param_value": param_value,
                        "run": run_idx,
                        "time_sec": time_sec,
                        "metric_value": metric,
                    })
                status = "NaN" if math.isnan(rows[-1]["time_sec"]) else \
                    f"{rows[-1]['time_sec']:.3f}s"
                print(f"[perf] {pos_id} {algorithm} {param_name}={param_value}"
                      f" → {status}")

    with open(PERF_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "position_id", "algorithm", "param_name", "param_value",
            "run", "time_sec", "metric_value",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"готово: {PERF_CSV}, {len(rows)} строк")


if __name__ == "__main__":
    run()
