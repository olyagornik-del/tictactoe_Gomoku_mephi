"""Скрипт обучения перцептрона на партиях AB-vs-AB.

Запуск:

    python -m training.train_perceptron [--games N] [--epochs N] ...

Без аргументов — дефолты из ТЗ (2000 партий, 50 эпох, lr=0.1,
batch_size=32, init='small_random', train/val split 80/20).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.model_selection import train_test_split

from agents.perceptron import DEFAULT_MODEL_PATH, FEATURE_DIM, Perceptron
from training.data_generator import generate_dataset


def train(
    num_games: int = 2000,
    epochs: int = 50,
    lr: float = 0.1,
    epsilon: float = 0.1,
    depth: int = 2,
    batch_size: int = 32,
    max_moves: int = 60,
    val_size: float = 0.2,
    seed: int | None = 42,
    out_path: str = DEFAULT_MODEL_PATH,
    verbose: bool = True,
) -> Perceptron:
    """Полный цикл: датасет → 80/20 split → SGD → save_weights.

    :return: обученная модель (она же сохранена в ``out_path``).
    """
    if verbose:
        print(
            f"[train] Генерация {num_games} партий "
            f"(AB depth={depth}, epsilon={epsilon})…"
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
    pos = float(y.mean())
    if verbose:
        print(
            f"[train] Сэмплов: {len(X)}; "
            f"доля класса 1: {pos:.2%}; {time.time() - t0:.1f}s"
        )

    if len(X) < 4:
        raise RuntimeError("слишком мало сэмплов — увеличь num_games")

    # Train/val split 80/20 (стратифицированный, если есть оба класса).
    stratify = y if (0.05 < pos < 0.95) else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=stratify
    )

    model = Perceptron(
        n_features=FEATURE_DIM, init_method="small_random", random_state=seed
    )
    t1 = time.time()
    history = model.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=epochs, lr=lr, batch_size=batch_size, random_state=seed,
    )
    if verbose:
        print(
            f"[train] Обучение: {time.time() - t1:.1f}s; "
            f"loss train {history['loss_train'][0]:.4f} → "
            f"{history['loss_train'][-1]:.4f}; "
            f"val {history['loss_val'][0]:.4f} → "
            f"{history['loss_val'][-1]:.4f}"
        )
        train_acc = (model.predict(X_tr) == y_tr).mean()
        val_acc = (model.predict(X_val) == y_val).mean()
        print(f"[train] accuracy: train={train_acc:.3f}, val={val_acc:.3f}")

    model.save_weights(out_path)
    if verbose:
        print(f"[train] Сохранено: {out_path}")
    return model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Обучение перцептрона для Гомоку.")
    p.add_argument("--games", type=int, default=2000, dest="num_games")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--depth", type=int, default=2, help="Глубина AB-учителя.")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--max-moves", type=int, default=60, dest="max_moves")
    p.add_argument("--val-size", type=float, default=0.2, dest="val_size")
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
        val_size=args.val_size,
        seed=args.seed,
        out_path=args.out_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
