"""Рис. 7. Сила игры vs стоимость хода (по результатам турнира).

Положение пяти участников турнира в координатах
«среднее время одного хода (с, лог-шкала)» — «сила игры».

Сила игры = доля очков, набранных агентом во всех 120 сыгранных им
партиях (победа = 1, ничья = 0.5, поражение = 0). Время — среднее по
тем же 120 партиям. Вход — ``results/tournament.csv``, выход —
``results/fig_strength_vs_cost.png``.

Запуск: ``python -m experiments.fig_strength_vs_cost``

Зависимостей сверх проекта нет: чтение через ``csv``, расчёты — ``numpy``
(pandas намеренно не используется, как и в остальном анализе).
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = "results"
TOURNAMENT_CSV = os.path.join(RESULTS_DIR, "tournament.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "fig_strength_vs_cost.png")

ALGOS = ["minimax", "alphabeta", "mcts", "perceptron", "perceptron_pixel"]
LABELS = {
    "minimax": "Минимакс",
    "alphabeta": "Альфа-бета",
    "mcts": "MCTS",
    "perceptron": "Перц-инж",
    "perceptron_pixel": "Перц-пикс",
}
COLORS = {
    "minimax": "#1f77b4",
    "alphabeta": "#2ca02c",
    "mcts": "#d62728",
    "perceptron": "#9467bd",
    "perceptron_pixel": "#ff7f0e",
}
# Подписи на графике, где нужен перенос строки (только для аннотаций;
# в таблице/печати используется обычное имя из LABELS). \n = новая строка.
ANNOT_LABEL = {
    "perceptron_pixel": "Перц-\nпикс",
}
# Смещения подписей (dx, dy в пунктах) — чтобы не наезжали друг на друга.
LABEL_OFFSET = {
    "minimax": (8, 8),
    "alphabeta": (8, 8),
    "mcts": (8, -4),
    "perceptron": (6, 8),       # «Перц-инж» (фиолетовая) — вверх-вправо
    "perceptron_pixel": (-18, -25),  # «Перц-пикс» (оранжевая) — влево от точки
}


def compute_strength_and_time(
    rows: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """Доля очков и среднее время на ход для каждого агента (120 партий)."""
    scores: dict[str, float] = {}
    times: dict[str, float] = {}
    for a in ALGOS:
        s_total = 0.0
        time_vals: list[float] = []
        for r in rows:
            if r["algo_first"] == a:
                time_vals.append(float(r["avg_time_per_move_first"]))
                if r["winner"] == "draw":
                    s_total += 0.5
                elif r["winner"] == r["first_color"]:
                    s_total += 1.0
            if r["algo_second"] == a:
                time_vals.append(float(r["avg_time_per_move_second"]))
                if r["winner"] == "draw":
                    s_total += 0.5
                elif r["winner"] != r["first_color"]:
                    s_total += 1.0
        n_games = len(time_vals)
        if n_games == 0:
            continue
        scores[a] = s_total / n_games
        times[a] = float(np.mean(time_vals))
    return scores, times


def main() -> None:
    if not os.path.exists(TOURNAMENT_CSV):
        raise SystemExit(f"нет {TOURNAMENT_CSV} — сначала запусти турнир")
    with open(TOURNAMENT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scores, times = compute_strength_and_time(rows)
    present = [a for a in ALGOS if a in scores]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for a in present:
        ax.scatter(times[a], scores[a], s=140, zorder=3,
                   color=COLORS.get(a), edgecolors="black", linewidths=0.6)
        ax.annotate(
            ANNOT_LABEL.get(a, LABELS.get(a, a)), (times[a], scores[a]),
            xytext=LABEL_OFFSET.get(a, (8, 8)),
            textcoords="offset points", fontsize=11,
            ha="center" if a in ANNOT_LABEL else "left",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Среднее время на ход, с (лог-шкала)")
    ax.set_ylabel("Доля очков в турнире (1 = победа, 0.5 = ничья)")
    ax.set_title("Сила игры vs стоимость хода (по результатам турнира)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    # Печать таблицы для статьи.
    print(f"{'агент':<18}{'доля очков':>12}{'время/ход, с':>16}")
    for a in present:
        print(f"{LABELS.get(a, a):<18}{scores[a]:>12.3f}{times[a]:>16.4f}")
    print(f"готово: {OUT_PNG}")


if __name__ == "__main__":
    main()
