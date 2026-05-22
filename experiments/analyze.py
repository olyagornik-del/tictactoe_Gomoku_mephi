"""Анализ результатов экспериментов A и B: графики и сводные таблицы.

Читает ``results/perf.csv`` и ``results/tournament.csv``, строит PNG
(150 dpi, без TeX) и печатает/сохраняет сводные таблицы.

Запуск: ``python -m experiments.analyze`` (после perf_benchmark и
tournament).
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless, без дисплея
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = "results"
PERF_CSV = os.path.join(RESULTS_DIR, "perf.csv")
TOURNAMENT_CSV = os.path.join(RESULTS_DIR, "tournament.csv")
PERF_SUMMARY_CSV = os.path.join(RESULTS_DIR, "perf_summary.csv")

ALGO_LABEL = {
    "minimax": "Минимакс",
    "alphabeta": "Альфа-бета",
    "mcts": "MCTS",
    "perceptron": "Перцептрон",
}


# --- загрузка -----------------------------------------------------------

def _load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


# ========================================================================
# Эксперимент A: производительность
# ========================================================================

def analyze_perf() -> None:
    if not os.path.exists(PERF_CSV):
        print(f"[analyze] нет {PERF_CSV} — пропускаю эксперимент A")
        return
    rows = _load_csv(PERF_CSV)

    # Группируем по (position, algorithm, param_value).
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"time": [], "metric": []}
    )
    for r in rows:
        key = (r["position_id"], r["algorithm"],
               r["param_name"], int(r["param_value"]))
        grouped[key]["time"].append(_to_float(r["time_sec"]))
        grouped[key]["metric"].append(_to_float(r["metric_value"]))

    # --- сводная таблица медиан ---
    summary: list[dict] = []
    for (pos, algo, pname, pval), vals in sorted(grouped.items()):
        med_t = float(np.nanmedian(vals["time"]))
        med_m = float(np.nanmedian(vals["metric"]))
        summary.append({
            "position_id": pos, "algorithm": algo,
            "param_name": pname, "param_value": pval,
            "median_time_sec": round(med_t, 5),
            "median_metric": round(med_m, 1),
        })

    with open(PERF_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "position_id", "algorithm", "param_name", "param_value",
            "median_time_sec", "median_metric",
        ])
        w.writeheader()
        w.writerows(summary)

    print("\n=== Медианы по (позиция, алгоритм, параметр) ===")
    print(f"{'позиция':<14}{'алгоритм':<12}{'параметр':<14}"
          f"{'медиана t, с':>14}{'медиана метрики':>18}")
    for s in summary:
        pv = f"{s['param_name']}={s['param_value']}"
        tt = "NaN" if np.isnan(s["median_time_sec"]) else f"{s['median_time_sec']:.4f}"
        mm = "NaN" if np.isnan(s["median_metric"]) else f"{s['median_metric']:.0f}"
        print(f"{s['position_id']:<14}{s['algorithm']:<12}{pv:<14}{tt:>14}{mm:>18}")

    _plot_perf_time(grouped)
    _plot_branching(grouped)
    print(f"готово: {PERF_SUMMARY_CSV}, {len(summary)} строк")


def _plot_perf_time(grouped: dict) -> None:
    """time vs param для каждого алгоритма (лог-шкала Y), линия на позицию."""
    algos = sorted({k[1] for k in grouped if k[1] != "perceptron"})
    positions = sorted({k[0] for k in grouped})
    for algo in algos:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for pos in positions:
            pts = sorted(
                (pval, float(np.nanmedian(v["time"])))
                for (p, a, _pn, pval), v in grouped.items()
                if a == algo and p == pos
            )
            if not pts:
                continue
            xs = [p for p, _ in pts]
            ys = [t for _, t in pts]
            ax.plot(xs, ys, marker="o", label=pos)
        ax.set_yscale("log")
        ax.set_xlabel("параметр (глубина / симуляции)")
        ax.set_ylabel("время хода, с (log)")
        ax.set_title(f"Скорость: {ALGO_LABEL.get(algo, algo)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, f"perf_time_{algo}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _plot_branching(grouped: dict) -> None:
    """nodes vs depth + эффективный коэффициент ветвления для minimax/AB."""
    print("\n=== Эффективный коэффициент ветвления nodes(d)/nodes(d-1) ===")
    fig, ax = plt.subplots(figsize=(7, 4.5))
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
            depths = [d for d, _ in pts]
            nodes = [n for _, n in pts]
            ax.plot(depths, nodes, marker="o",
                    label=f"{ALGO_LABEL.get(algo, algo)} / {pos}")
            # коэффициент ветвления
            ratios = []
            for i in range(1, len(nodes)):
                if nodes[i - 1] and not np.isnan(nodes[i]) and not np.isnan(nodes[i - 1]):
                    ratios.append(nodes[i] / nodes[i - 1])
            ratio_str = ", ".join(f"{r:.1f}" for r in ratios) if ratios else "—"
            print(f"  {algo:<10} {pos:<14} b≈ [{ratio_str}]")
    ax.set_yscale("log")
    ax.set_xlabel("глубина")
    ax.set_ylabel("узлов (log)")
    ax.set_title("Узлы vs глубина (Минимакс / Альфа-бета)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "perf_branching.png"), dpi=150)
    plt.close(fig)


# ========================================================================
# Эксперимент B: турнир
# ========================================================================

def analyze_tournament() -> None:
    if not os.path.exists(TOURNAMENT_CSV):
        print(f"[analyze] нет {TOURNAMENT_CSV} — пропускаю эксперимент B")
        return
    rows = _load_csv(TOURNAMENT_CSV)

    algos = sorted(
        {r["algo_first"] for r in rows} | {r["algo_second"] for r in rows}
    )

    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], int] = defaultdict(int)
    avg_times: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        a, b = r["algo_first"], r["algo_second"]
        fc, w = r["first_color"], r["winner"]
        if w == "draw":
            wins[(a, b)] += 0.5
            wins[(b, a)] += 0.5
        elif w == fc:        # выиграл первый агент
            wins[(a, b)] += 1.0
        else:                # выиграл второй
            wins[(b, a)] += 1.0
        games[(a, b)] += 1
        games[(b, a)] += 1
        avg_times[a].append(_to_float(r["avg_time_per_move_first"]))
        avg_times[b].append(_to_float(r["avg_time_per_move_second"]))

    # --- матрица win rate ---
    n = len(algos)
    matrix = np.full((n, n), np.nan)
    print("\n=== Win rate (строка против столбца, draws=0.5) ===")
    header = "          " + "".join(f"{ALGO_LABEL.get(a, a)[:8]:>10}" for a in algos)
    print(header)
    for i, a in enumerate(algos):
        line = f"{ALGO_LABEL.get(a, a)[:9]:<10}"
        for j, b in enumerate(algos):
            if i == j or games[(a, b)] == 0:
                line += f"{'—':>10}"
                continue
            wr = wins[(a, b)] / games[(a, b)]
            matrix[i, j] = wr
            line += f"{wr:>10.2f}"
        print(line)

    _plot_winrate_heatmap(matrix, algos)
    _plot_avg_time_bar(avg_times, algos)
    print(f"готово: матрица {n}x{n}, графики в {RESULTS_DIR}/")


def _plot_winrate_heatmap(matrix: np.ndarray, algos: list[str]) -> None:
    labels = [ALGO_LABEL.get(a, a) for a in algos]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(algos)))
    ax.set_yticks(range(len(algos)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Win rate: строка против столбца")
    for i in range(len(algos)):
        for j in range(len(algos)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "winrate_heatmap.png"), dpi=150)
    plt.close(fig)


def _plot_avg_time_bar(avg_times: dict[str, list[float]], algos: list[str]) -> None:
    labels = [ALGO_LABEL.get(a, a) for a in algos]
    means = [float(np.nanmean(avg_times[a])) if avg_times[a] else 0.0
             for a in algos]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, means, color="#4a7fc8")
    ax.set_ylabel("среднее время на ход, с")
    ax.set_title("Среднее время на ход по алгоритмам")
    if any(m > 0 for m in means) and max(means) / max(min(m for m in means if m > 0), 1e-9) > 50:
        ax.set_yscale("log")
        ax.set_ylabel("среднее время на ход, с (log)")
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "avg_time_bar.png"), dpi=150)
    plt.close(fig)


def run() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    analyze_perf()
    analyze_tournament()


if __name__ == "__main__":
    run()
