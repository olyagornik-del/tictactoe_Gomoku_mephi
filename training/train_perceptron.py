"""Скрипт обучения перцептрона на партиях AB-vs-AB.

Запуск:

    python -m training.train_perceptron [--games N] [--epochs N] ...

Без аргументов — дефолты из ТЗ (2000 партий, 50 эпох, lr=0.1,
batch_size=32, init='small_random', train/val split 80/20).

В конце каждого запуска дописывает строку в журнал
``training_log.md`` — удобно сравнивать эксперименты.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from agents.perceptron import (
    DEFAULT_MODEL_PATH,
    FEATURE_DIM,
    InitMethod,
    Perceptron,
)
from training.data_generator import generate_dataset

#: Файл-журнал экспериментов по умолчанию.
DEFAULT_LOG_PATH: str = "training_log.md"


@dataclass
class TrainResult:
    """Всё, что узнали за один запуск ``train()`` — для журнала."""

    model: Perceptron
    history: dict[str, list[float]]
    num_samples: int
    pos_ratio: float
    train_acc: float
    val_acc: float
    gen_time_s: float
    fit_time_s: float
    out_path: str
    hyperparams: dict[str, Any] = field(default_factory=dict)


def _generate_or_load(
    num_games: int,
    epsilon: float,
    depth: int,
    max_moves: int,
    seed: int | None,
    cache_path: str | None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Сгенерировать датасет AB-vs-AB или загрузить его из кэша.

    Если ``cache_path`` указан и файл существует — берёт данные оттуда
    (``gen_time = 0``). Если файла нет — генерит и сохраняет.
    Если ``cache_path`` ``None`` — просто генерит без кэширования.
    """
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"[train] Загружаю датасет из кэша {cache_path}")
        with np.load(cache_path) as data:
            X = data["X"]
            y = data["y"]
        if verbose:
            print(
                f"[train] Сэмплов в кэше: {len(X)}. "
                f"(параметр --games игнорируется)"
            )
        return X, y, 0.0

    if verbose:
        print(
            f"[train] Генерация {num_games} партий "
            f"(AB depth={depth}, epsilon={epsilon})…"
        )
    t0 = time.time()
    X, y = generate_dataset(
        num_games=num_games, epsilon=epsilon, depth=depth,
        max_moves=max_moves, seed=seed, verbose=verbose,
    )
    gen_time = time.time() - t0

    if cache_path:
        parent = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez(cache_path, X=X, y=y)
        if verbose:
            print(f"[train] Кэшировал датасет в {cache_path}")

    return X, y, gen_time


def train(
    num_games: int = 2000,
    epochs: int = 50,
    lr: float = 0.1,
    epsilon: float = 0.1,
    depth: int = 2,
    batch_size: int = 32,
    init_method: InitMethod = "small_random",
    max_moves: int = 60,
    val_size: float = 0.2,
    seed: int | None = 42,
    out_path: str = DEFAULT_MODEL_PATH,
    cache_path: str | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Полный цикл: датасет → 80/20 split → SGD → save_weights.

    Возвращает :class:`TrainResult` со всеми метриками — для журнала.
    """
    X, y, gen_time = _generate_or_load(
        num_games=num_games, epsilon=epsilon, depth=depth,
        max_moves=max_moves, seed=seed,
        cache_path=cache_path, verbose=verbose,
    )
    pos = float(y.mean())
    if verbose:
        print(
            f"[train] Сэмплов: {len(X)}; "
            f"доля класса 1: {pos:.2%}; {gen_time:.1f}s"
        )

    if len(X) < 4:
        raise RuntimeError("слишком мало сэмплов — увеличь num_games")

    stratify = y if (0.05 < pos < 0.95) else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=stratify
    )

    model = Perceptron(
        n_features=FEATURE_DIM, init_method=init_method, random_state=seed,
    )
    t1 = time.time()
    history = model.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=epochs, lr=lr, batch_size=batch_size, random_state=seed,
    )
    fit_time = time.time() - t1
    train_acc = float((model.predict(X_tr) == y_tr).mean())
    val_acc = float((model.predict(X_val) == y_val).mean())
    if verbose:
        print(
            f"[train] Обучение: {fit_time:.1f}s; "
            f"loss train {history['loss_train'][0]:.4f} → "
            f"{history['loss_train'][-1]:.4f}; "
            f"val {history['loss_val'][0]:.4f} → "
            f"{history['loss_val'][-1]:.4f}"
        )
        print(f"[train] accuracy: train={train_acc:.3f}, val={val_acc:.3f}")

    model.save_weights(out_path)
    if verbose:
        print(f"[train] Сохранено: {out_path}")

    return TrainResult(
        model=model, history=history,
        num_samples=len(X), pos_ratio=pos,
        train_acc=train_acc, val_acc=val_acc,
        gen_time_s=gen_time, fit_time_s=fit_time,
        out_path=out_path,
        hyperparams={
            "num_games": num_games, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "init_method": init_method,
            "epsilon": epsilon, "depth": depth, "seed": seed,
        },
    )


# ============================================================================
# Быстрая оценка модели в Гомоку
# ============================================================================


def quick_eval_vs_random(
    model_path: str,
    n_per_color: int = 2,
    seed_start: int = 1,
    max_moves: int = 80,
) -> tuple[int, int]:
    """Сыграть ``2 * n_per_color`` партий vs ``RandomAgent``.

    Половина партий перцептрон — X (первым), половина — O (вторым).
    Возвращает ``(побед, всего)``. Никакого pytest — обычный код.
    """
    # Локальные импорты — чтобы не тянуть тяжёлое в API train().
    from agents.base import RandomAgent
    from agents.perceptron import PerceptronAgent
    from game.board import O, X, Board
    from game.rules import winner

    def _play(ax, ao) -> str | None:
        b = Board()
        players = {X: ax, O: ao}
        turn = X
        for _ in range(max_moves):
            b.place(*players[turn].choose_move(b), turn)
            if winner(b) is not None:
                return turn
            turn = O if turn == X else X
        return None

    wins = 0
    total = 0
    for s in range(seed_start, seed_start + n_per_color):
        # перцептрон X
        if _play(PerceptronAgent.from_disk(X, path=model_path),
                 RandomAgent(O, seed=s)) == X:
            wins += 1
        total += 1
        # перцептрон O
        if _play(RandomAgent(X, seed=s),
                 PerceptronAgent.from_disk(O, path=model_path)) == O:
            wins += 1
        total += 1
    return wins, total


# ============================================================================
# Журнал экспериментов
# ============================================================================


#: Заголовки и ширины колонок (фиксированная ширина → выравнивание
#: даже в plain-text-просмотре, без markdown preview).
_COLUMNS: tuple[tuple[str, int], ...] = (
    ("Дата",         16),
    ("Игр",           5),
    ("Эпох",          4),
    ("LR",            5),
    ("Batch",         5),
    ("Init",         12),
    ("Seed",          4),
    ("Сэмплов",       7),
    ("Кл.1",          5),
    ("Loss train",   13),  # start→end
    ("Loss val",     13),  # start→end
    ("Acc tr/val",   13),
    ("vs Rand",       8),
    ("Время",         6),
    ("Файл весов",   26),
)

_LOG_INTRO = (
    "# Журнал экспериментов обучения перцептрона\n"
    "\n"
    "Каждая строка — один запуск `python -m training.train_perceptron`.\n"
    "Метрики: `train` — на данных, которые модель видела при SGD; "
    "`val` — на отложенных 20%, модель их не видела. `vs Rand` — "
    "быстрая проверка игры перцептрон-vs-Random (победы / всего партий).\n"
    "\n"
)


def _format_row(cells: list[str]) -> str:
    """Markdown-строка с колонками фиксированной ширины."""
    parts = ["|"]
    for cell, (_, width) in zip(cells, _COLUMNS):
        parts.append(f" {cell:<{width}} |")
    parts.append("\n")
    return "".join(parts)


def _make_header() -> str:
    head = _format_row([name for name, _ in _COLUMNS])
    sep = "|" + "|".join("-" * (width + 2) for _, width in _COLUMNS) + "|\n"
    return head + sep


def log_run(
    log_path: str,
    result: TrainResult,
    winrate: tuple[int, int],
) -> None:
    """Дописать строку с результатами в markdown-журнал.

    Если файла нет — создаст с заголовком таблицы. Если есть —
    просто аппендит строку с фиксированной шириной колонок.
    """
    h = result.hyperparams
    lt = result.history["loss_train"]
    lv = result.history["loss_val"]
    cells = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        str(h["num_games"]),
        str(h["epochs"]),
        f"{h['lr']:g}",
        str(h["batch_size"]),
        str(h["init_method"]),
        str(h["seed"]),
        str(result.num_samples),
        f"{result.pos_ratio:.0%}",
        f"{lt[0]:.3f}→{lt[-1]:.3f}",
        f"{lv[0]:.3f}→{lv[-1]:.3f}",
        f"{result.train_acc:.1%}/{result.val_acc:.1%}",
        f"{winrate[0]}/{winrate[1]}",
        f"{result.gen_time_s + result.fit_time_s:.0f}s",
        f"`{result.out_path}`",
    ]
    row = _format_row(cells)

    new_file = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(_LOG_INTRO)
            f.write(_make_header())
        f.write(row)


# ============================================================================
# CLI
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Обучение перцептрона для Гомоку.")
    p.add_argument("--games", type=int, default=2000, dest="num_games")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--depth", type=int, default=2, help="Глубина AB-учителя.")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--init", type=str, default="small_random",
                   choices=["zeros", "small_random", "large_random"],
                   dest="init_method")
    p.add_argument("--max-moves", type=int, default=60, dest="max_moves")
    p.add_argument("--val-size", type=float, default=0.2, dest="val_size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=DEFAULT_MODEL_PATH,
                   dest="out_path")
    p.add_argument("--cache-data", type=str, default=None,
                   dest="cache_path",
                   help="Путь к .npz для кэша датасета. Если файл "
                        "существует — данные загружаются (--games игнорируется); "
                        "если нет — генерируются и сохраняются.")
    p.add_argument("--log-path", type=str, default=DEFAULT_LOG_PATH,
                   dest="log_path",
                   help="Markdown-журнал экспериментов (append-only).")
    p.add_argument("--eval-games", type=int, default=4, dest="eval_games",
                   help="Партий vs Random для быстрой оценки (0 — пропустить).")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    np.set_printoptions(precision=4, suppress=True)
    result = train(
        num_games=args.num_games, epochs=args.epochs, lr=args.lr,
        epsilon=args.epsilon, depth=args.depth, batch_size=args.batch_size,
        init_method=args.init_method, max_moves=args.max_moves,
        val_size=args.val_size, seed=args.seed,
        out_path=args.out_path, cache_path=args.cache_path,
        verbose=not args.quiet,
    )

    if args.eval_games > 0:
        if not args.quiet:
            print(f"[train] Быстрая оценка: {args.eval_games} партий vs Random…")
        # n_per_color = половина (нечётное округлится вниз).
        wins, total = quick_eval_vs_random(
            model_path=args.out_path,
            n_per_color=max(1, args.eval_games // 2),
        )
    else:
        wins, total = 0, 0
    if not args.quiet:
        print(f"[train] vs Random: {wins}/{total}")

    log_run(args.log_path, result, (wins, total))
    if not args.quiet:
        print(f"[train] Записано в журнал: {args.log_path}")


if __name__ == "__main__":
    main()
