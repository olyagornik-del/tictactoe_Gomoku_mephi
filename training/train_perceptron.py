"""Скрипт обучения перцептрона: данные → обучение → сохранение весов.

Запуск:

    python -m training.train_perceptron [--games N] [--epochs N] ...

Без аргументов — значения по умолчанию из ТЗ (2000 партий, 50 эпох,
lr=0.01, epsilon=0.1). Удобно переопределять для быстрых прогонов.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agents.perceptron import DEFAULT_MODEL_PATH, Perceptron
from training.data_generator import generate_dataset


def train(
    num_games: int = 2000,
    epochs: int = 50,
    lr: float = 0.01,
    epsilon: float = 0.1,
    depth: int = 2,
    batch_size: int = 64,
    max_moves: int = 60,
    seed: int | None = 42,
    out_path: str = DEFAULT_MODEL_PATH,
    verbose: bool = True,
) -> Perceptron:
    """Полный цикл: датасет → SGD → сохранение. Возвращает модель."""
    if verbose:
        print(
            f"[train] Генерация {num_games} партий "
            f"(AB depth={depth}, epsilon={epsilon})..."
        )
    t0 = time.time()
    X, y = generate_dataset(
        num_games=num_games,
        epsilon=epsilon,
        depth=depth,
        max_moves=max_moves,
        seed=seed,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[train] Сэмплов: {len(X)}, "
            f"диапазон меток: [{y.min():.3f}, {y.max():.3f}], "
            f"{time.time() - t0:.1f}s"
        )

    model = Perceptron(seed=seed)
    t1 = time.time()
    losses = model.fit(
        X, y, epochs=epochs, lr=lr, batch_size=batch_size,
        seed=seed, verbose=verbose,
    )
    if verbose:
        print(
            f"[train] Обучение: {time.time() - t1:.1f}s; "
            f"MSE start={losses[0]:.4f} → end={losses[-1]:.4f}"
        )

    model.save(out_path)
    if verbose:
        print(f"[train] Сохранено: {out_path}")
    return model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Обучение перцептрона для Гомоку.")
    p.add_argument("--games", type=int, default=2000, dest="num_games")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--depth", type=int, default=2, help="Глубина AB-учителя.")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--max-moves", type=int, default=60, dest="max_moves")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=DEFAULT_MODEL_PATH, dest="out_path")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    np.set_printoptions(precision=4, suppress=True)
    train(
        num_games=args.num_games,
        epochs=args.epochs,
        lr=args.lr,
        epsilon=args.epsilon,
        depth=args.depth,
        batch_size=args.batch_size,
        max_moves=args.max_moves,
        seed=args.seed,
        out_path=args.out_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
