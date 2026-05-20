"""Тесты перцептрона: кодирование, модель, обучение и игра vs Random.

Интеграционный тест обучает миниатюрный модель в ``tmp_path`` —
он медленнее юнит-тестов (несколько секунд), но не сохраняет
артефактов в репозитории.
"""

from __future__ import annotations

import os

import numpy as np

from agents.base import RandomAgent
from agents.perceptron import (
    FEATURE_DIM,
    Perceptron,
    PerceptronAgent,
    encode,
    normalize_score,
)
from game.board import O, X, Board
from tests.test_agents import run_match
from training.train_perceptron import train

# --- кодирование ---------------------------------------------------------


def test_encode_shape_and_one_hot() -> None:
    b = Board()
    b.place(0, 0, X)
    v = encode(b, X)
    assert v.shape == (FEATURE_DIM,)
    assert v.dtype == np.float32
    # На каждую из 81 клетки приходится ровно одна единица.
    assert v.sum() == 81.0


def test_encode_empty_board_all_empty_channel() -> None:
    v = encode(Board(), X)
    # Канал «пусто» = индексы 0, 3, 6, ... (каждая 3-я компонента).
    assert v[0::3].sum() == 81.0
    assert v[1::3].sum() == 0.0
    assert v[2::3].sum() == 0.0


def test_encode_perspective_swaps_channels() -> None:
    b = Board()
    b.place(0, 0, X)  # «свой» для X, «чужой» для O
    vx = encode(b, X)
    vo = encode(b, O)
    # Центральная клетка окна — это (0,0): индекс idx_центр = 3*(40)
    # (dx=0,dy=0 → 41-я клетка при iter), но безопаснее сравнить
    # суммарную статистику: число «своих» и «чужих» поменяться местами.
    assert vx[1::3].sum() == vo[2::3].sum()  # свои X == чужие O
    assert vx[2::3].sum() == vo[1::3].sum()


def test_normalize_score_bounds() -> None:
    # При очень больших по модулю значениях tanh насыщается ровно в ±1 —
    # это и есть желаемое поведение нормализации.
    assert normalize_score(-1_000_000) == -1.0
    assert normalize_score(1_000_000) == 1.0
    assert normalize_score(0) == 0.0
    # Промежуточные значения — строго внутри (-1, 1).
    mid = normalize_score(5_000)
    assert 0.0 < mid < 1.0


# --- модель --------------------------------------------------------------


def test_predict_range() -> None:
    p = Perceptron(seed=0)
    x = encode(Board(), X)
    v = p.predict(x)
    assert -1.0 <= v <= 1.0


def test_save_load_roundtrip(tmp_path) -> None:
    p = Perceptron(seed=0)
    path = tmp_path / "weights.npz"
    p.save(str(path))
    p2 = Perceptron.load(str(path))
    assert np.allclose(p.W, p2.W)
    assert p.b == p2.b
    x = encode(Board(), X)
    assert p.predict(x) == p2.predict(x)


def test_fit_reduces_loss() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, FEATURE_DIM)).astype(np.float32)
    true_W = rng.standard_normal(FEATURE_DIM).astype(np.float32) * 0.1
    y_train = np.tanh(X_train @ true_W).astype(np.float32)
    p = Perceptron(seed=1)
    losses = p.fit(X_train, y_train, epochs=30, lr=0.05, batch_size=32, seed=1)
    assert losses[-1] < losses[0] * 0.5  # минимум двукратное улучшение


# --- интеграция: перцептрон обыгрывает Random ---------------------------


def test_perceptron_beats_random(tmp_path) -> None:
    """Обучаем мини-модель и проверяем игру против Random.

    Параметры намеренно маленькие — для скорости тестов; даже так
    перцептрон должен уверенно обыгрывать случайного игрока.
    """
    model_path = tmp_path / "tiny.npz"
    train(
        num_games=20,
        epochs=50,
        lr=0.01,
        depth=1,
        max_moves=40,
        seed=0,
        out_path=str(model_path),
        verbose=False,
    )
    assert os.path.exists(model_path)

    wins = 0
    for seed in (1, 2, 3):
        w = run_match(
            PerceptronAgent(X, model_path=str(model_path)),
            RandomAgent(O, seed=seed),
            max_moves=60,
        )
        if w == X:
            wins += 1
    assert wins >= 2, f"перцептрон выиграл лишь {wins}/3"
