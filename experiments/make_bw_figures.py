"""Чёрно-белые версии всех графиков для печатной статьи.

Каждый график сохраняется отдельным файлом ``results/bw_*.png``
(префикс ``bw_`` = black-and-white). Серии различаются не цветом, а
начертанием: тип линии (сплошная / штрих / пунктир / штрих-пунктир),
форма маркера, штриховка баров, оттенки серого в heatmap — чтобы график
читался в ч/б печати.

Входные данные те же, что у ``analyze.py``: ``results/perf.csv`` и
``results/tournament.csv``.

Запуск: ``python -m experiments.make_bw_figures``
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.fig_strength_vs_cost import (  # noqa: E402
    ANNOT_LABEL,
    LABEL_OFFSET,
    compute_strength_and_time,
)

RESULTS_DIR = "results"
PERF_CSV = os.path.join(RESULTS_DIR, "perf.csv")
TOURNAMENT_CSV = os.path.join(RESULTS_DIR, "tournament.csv")

ALGO_LABEL = {
    "minimax": "Минимакс",
    "alphabeta": "Альфа-бета",
    "mcts": "MCTS",
    "perceptron": "Перц-инж",
    "perceptron_pixel": "Перц-пикс",
}

#: Стили линий для серий: (linestyle, marker). Всё чёрное.
LINE_STYLES: list[tuple[str, str]] = [
    ("-", "o"),    # сплошная, кружок
    ("--", "s"),   # штриховая, квадрат
    (":", "^"),    # пунктир, треугольник
    ("-.", "D"),   # штрих-пунктир, ромб
    ("-", "v"),    # сплошная, перевёрнутый треугольник
    ("--", "x"),   # штриховая, крест
]

#: Штриховки для bar chart (по агенту).
HATCHES = ["", "///", "...", "xxx", "\\\\\\"]

#: Маркеры для scatter (по агенту): (marker, заливка чёрная?).
SCATTER_MARKERS = {
    "minimax": ("o", True),
    "alphabeta": ("s", False),
    "mcts": ("^", True),
    "perceptron": ("D", False),
    "perceptron_pixel": ("v", True),
}


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"готово: {path}")


# ========================================================================
# Перф-графики (линии)
# ========================================================================


def _grouped_perf() -> dict:
    rows = _load_csv(PERF_CSV)
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"time": [], "metric": []}
    )
    for r in rows:
        key = (r["position_id"], r["algorithm"],
               r["param_name"], int(r["param_value"]))
        grouped[key]["time"].append(_to_float(r["time_sec"]))
        grouped[key]["metric"].append(_to_float(r["metric_value"]))
    return grouped


def bw_perf_time(grouped: dict) -> None:
    """time vs param на алгоритм; позиции различаются типом линии."""
    algos = sorted({k[1] for k in grouped if k[1] != "perceptron"})
    positions = sorted({k[0] for k in grouped})
    for algo in algos:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, pos in enumerate(positions):
            pts = sorted(
                (pval, float(np.nanmedian(v["time"])))
                for (p, a, _pn, pval), v in grouped.items()
                if a == algo and p == pos
            )
            if not pts:
                continue
            ls, marker = LINE_STYLES[i % len(LINE_STYLES)]
            ax.plot([p for p, _ in pts], [t for _, t in pts],
                    linestyle=ls, marker=marker, color="black",
                    markerfacecolor="white", markersize=7, label=pos)
        ax.set_yscale("log")
        ax.set_xlabel("параметр (глубина / симуляции)")
        ax.set_ylabel("время хода, с (log)")
        ax.set_title(f"Скорость: {ALGO_LABEL.get(algo, algo)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _save(fig, f"bw_perf_time_{algo}.png")


def bw_branching(grouped: dict) -> None:
    """nodes vs depth: каждая серия (алгоритм × позиция) — свой стиль."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    series_idx = 0
    for algo in ("minimax", "alphabeta"):
        positions = sorted({k[0] for k in grouped if k[1] == algo})
        for pos in positions:
            pts = sorted(
                (pval, float(np.nanmedian(v["metric"])))
                for (p, a, _pn, pval), v in grouped.items()
                if a == algo and p == pos
            )
            if not pts:
                continue
            ls, marker = LINE_STYLES[series_idx % len(LINE_STYLES)]
            series_idx += 1
            ax.plot([d for d, _ in pts], [n for _, n in pts],
                    linestyle=ls, marker=marker, color="black",
                    markerfacecolor="white", markersize=6,
                    label=f"{ALGO_LABEL.get(algo, algo)} / {pos}")
    ax.set_yscale("log")
    ax.set_xlabel("глубина")
    ax.set_ylabel("узлов (log)")
    ax.set_title("Узлы vs глубина (Минимакс / Альфа-бета)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "bw_perf_branching.png")


# ========================================================================
# Турнирные графики
# ========================================================================


def _tournament_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    algos = sorted(
        {r["algo_first"] for r in rows} | {r["algo_second"] for r in rows}
    )
    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        a, b = r["algo_first"], r["algo_second"]
        fc, w = r["first_color"], r["winner"]
        if w == "draw":
            wins[(a, b)] += 0.5
            wins[(b, a)] += 0.5
        elif w == fc:
            wins[(a, b)] += 1.0
        else:
            wins[(b, a)] += 1.0
        games[(a, b)] += 1
        games[(b, a)] += 1
    n = len(algos)
    matrix = np.full((n, n), np.nan)
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if i != j and games[(a, b)]:
                matrix[i, j] = wins[(a, b)] / games[(a, b)]
    return matrix, algos


def bw_winrate_heatmap(rows: list[dict]) -> None:
    """Heatmap в оттенках серого; текст инвертируется на тёмных клетках."""
    matrix, algos = _tournament_matrix(rows)
    labels = [ALGO_LABEL.get(a, a) for a in algos]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(algos)))
    ax.set_yticks(range(len(algos)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Win rate: строка против столбца (ч/б)")
    for i in range(len(algos)):
        for j in range(len(algos)):
            if not np.isnan(matrix[i, j]):
                # На тёмном фоне (высокий winrate) пишем белым.
                color = "white" if matrix[i, j] > 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center",
                        va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "bw_winrate_heatmap.png")


def bw_avg_time_bar(rows: list[dict]) -> None:
    """Bar chart: белые бары с разной штриховкой."""
    algos = sorted(
        {r["algo_first"] for r in rows} | {r["algo_second"] for r in rows}
    )
    avg_times: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        avg_times[r["algo_first"]].append(
            _to_float(r["avg_time_per_move_first"]))
        avg_times[r["algo_second"]].append(
            _to_float(r["avg_time_per_move_second"]))
    labels = [ALGO_LABEL.get(a, a) for a in algos]
    means = [float(np.nanmean(avg_times[a])) if avg_times[a] else 0.0
             for a in algos]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, means, color="white", edgecolor="black",
                  linewidth=1.2)
    for bar, hatch in zip(bars, HATCHES * 2):
        bar.set_hatch(hatch)
    ax.set_ylabel("среднее время на ход, с")
    ax.set_title("Среднее время на ход по алгоритмам (ч/б)")
    positive = [m for m in means if m > 0]
    if positive and max(means) / max(min(positive), 1e-9) > 50:
        ax.set_yscale("log")
        ax.set_ylabel("среднее время на ход, с (log)")
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "bw_avg_time_bar.png")


def bw_strength_vs_cost(rows: list[dict]) -> None:
    """Scatter «сила vs стоимость»: агенты различаются формой маркера."""
    scores, times = compute_strength_and_time(rows)
    present = [a for a in scores]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for a in present:
        marker, filled = SCATTER_MARKERS.get(a, ("o", True))
        ax.scatter(times[a], scores[a], s=150, zorder=3, marker=marker,
                   facecolors="black" if filled else "white",
                   edgecolors="black", linewidths=1.4)
        # Подписи у точек — те же смещения, что в цветной версии.
        ax.annotate(
            ANNOT_LABEL.get(a, ALGO_LABEL.get(a, a)), (times[a], scores[a]),
            xytext=LABEL_OFFSET.get(a, (8, 8)),
            textcoords="offset points", fontsize=11,
            ha="center" if a in ANNOT_LABEL else "left",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Среднее время на ход, с (лог-шкала)")
    ax.set_ylabel("Доля очков в турнире (1 = победа, 0.5 = ничья)")
    ax.set_title("Сила игры vs стоимость хода (ч/б)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _save(fig, "bw_fig_strength_vs_cost.png")


# ========================================================================


def run() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(PERF_CSV):
        grouped = _grouped_perf()
        bw_perf_time(grouped)
        bw_branching(grouped)
    else:
        print(f"[bw] нет {PERF_CSV} — пропускаю перф-графики")
    if os.path.exists(TOURNAMENT_CSV):
        rows = _load_csv(TOURNAMENT_CSV)
        bw_winrate_heatmap(rows)
        bw_avg_time_bar(rows)
        bw_strength_vs_cost(rows)
    else:
        print(f"[bw] нет {TOURNAMENT_CSV} — пропускаю турнирные графики")


if __name__ == "__main__":
    run()
