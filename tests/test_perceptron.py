"""Юнит-тесты класса :class:`agents.perceptron.Perceptron`.

Покрывают численную устойчивость, инициализацию, инвариант ``b ≡ 0``,
форму истории обучения и качество на линейно разделимой синтетике
``sklearn.make_classification``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from agents.perceptron import (
    FEATURE_DIM,
    Perceptron,
    PerceptronAgent,
    board_to_features,
)
from game.board import O, X, Board


# --- инициализация ------------------------------------------------------


def test_init_zeros_weights() -> None:
    p = Perceptron(n_features=5, init_method="zeros")
    assert p.W.shape == (5,)
    assert np.all(p.W == 0.0)
    assert p.b == 0.0


def test_init_small_random_seeded_is_reproducible() -> None:
    p1 = Perceptron(10, init_method="small_random", random_state=42)
    p2 = Perceptron(10, init_method="small_random", random_state=42)
    assert np.array_equal(p1.W, p2.W)
    # «small» — почти все веса по модулю меньше 0.1.
    assert np.std(p1.W) < 0.1


def test_init_large_random_has_large_std() -> None:
    p = Perceptron(1000, init_method="large_random", random_state=0)
    # Ожидаемый std ≈ 10; берём широкий доверительный интервал.
    assert 5.0 < np.std(p.W) < 20.0


def test_init_invalid_raises() -> None:
    with pytest.raises(ValueError):
        Perceptron(5, init_method="banana")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Perceptron(0)


# --- активация и форма --------------------------------------------------


def test_sigmoid_stable_on_extreme_inputs() -> None:
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    y = Perceptron.sigmoid(z)
    assert not np.any(np.isnan(y))
    assert np.all((y >= 0.0) & (y <= 1.0))
    # Насыщение к 0 / 1 без переполнений.
    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert y[2] == pytest.approx(0.5, abs=1e-12)
    assert y[-1] == pytest.approx(1.0, abs=1e-12)


def test_forward_shape_and_range() -> None:
    p = Perceptron(4, init_method="small_random", random_state=0)
    X = np.random.default_rng(0).standard_normal((7, 4))
    y = p.forward(X)
    assert y.shape == (7,)
    assert np.all((y > 0.0) & (y < 1.0))


def test_predict_returns_binary_only() -> None:
    p = Perceptron(3, init_method="small_random", random_state=0)
    X = np.random.default_rng(0).standard_normal((20, 3))
    pred = p.predict(X)
    assert pred.shape == (20,)
    assert set(np.unique(pred).tolist()).issubset({0, 1})


# --- потеря -------------------------------------------------------------


def test_compute_loss_handles_saturated_predictions() -> None:
    """BCE не должна давать NaN даже при ``ŷ ∈ {0, 1}`` — за это
    отвечает обрезка на ``eps``."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.0, 1.0, 1.0, 0.0])
    loss = Perceptron.compute_loss(y_true, y_pred)
    assert np.isfinite(loss)
    assert loss > 0


def test_compute_loss_zero_on_perfect_predictions() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    # На идеальных вероятностях 0/1 BCE ≈ 0 (с поправкой на eps очень мала).
    assert Perceptron.compute_loss(y, y) < 1e-10


# --- обучение -----------------------------------------------------------


def test_fit_history_lengths() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.25, random_state=0)
    p = Perceptron(4, random_state=0)
    hist = p.fit(Xt, yt, Xv, yv, epochs=15, lr=0.1, batch_size=16, random_state=0)
    assert set(hist.keys()) == {"loss_train", "loss_val"}
    assert len(hist["loss_train"]) == 15
    assert len(hist["loss_val"]) == 15
    # Базовый sanity: train-потеря в конце меньше начальной.
    assert hist["loss_train"][-1] < hist["loss_train"][0]


def test_bias_stays_zero_after_training() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    y = (X[:, 0] > 0).astype(int)
    p = Perceptron(3, random_state=0)
    p.fit(X, y, X, y, epochs=20, lr=0.5, batch_size=16, random_state=0)
    assert p.b == 0.0


def test_accuracy_above_90_on_linearly_separable() -> None:
    """На по-настоящему линейно разделимых данных accuracy > 0.9.

    Два гауссовых блоба смещены на ±2.5σ вдоль фиксированного
    направления — после Z-score центр данных в нуле и оптимальная
    граница ``W·x = 0`` проходит через начало координат (это важно
    при фиксированном ``b = 0``).
    """
    rng = np.random.default_rng(0)
    n = 500
    direction = np.array([1.0, 2.0]) / np.linalg.norm([1.0, 2.0])
    mu = 2.5
    X_pos = rng.standard_normal((n // 2, 2)) + mu * direction
    X_neg = rng.standard_normal((n // 2, 2)) - mu * direction
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    model = Perceptron(n_features=2, init_method="small_random", random_state=0)
    model.fit(
        X_tr, y_tr, X_te, y_te,
        epochs=100, lr=0.5, batch_size=32, random_state=0,
    )
    acc = (model.predict(X_te) == y_te).mean()
    assert acc > 0.9, f"accuracy={acc:.3f} ниже 0.9"


def test_matches_no_bias_baseline_on_make_classification() -> None:
    """Sanity на стандартной лабораторной синтетике.

    ``make_classification`` (class_sep=1.0 по умолчанию) — *не* линейно
    разделимая выборка: оптимум при ``b=0`` равен accuracy sklearn'овской
    LogisticRegression с ``fit_intercept=False`` (~0.88). Наша
    реализация должна выйти на этот плато, не уступив больше 5 п.п.
    """
    X, y = make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_informative=2,
        random_state=42, n_clusters_per_class=1,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    model = Perceptron(2, init_method="small_random", random_state=0)
    model.fit(X_tr, y_tr, X_te, y_te,
              epochs=100, lr=0.1, batch_size=32, random_state=0)
    acc = (model.predict(X_te) == y_te).mean()
    # Эмпирический потолок без биаса на этих данных ≈ 0.887.
    assert acc > 0.83, f"accuracy={acc:.3f} существенно ниже плато"


# --- интеграция с Гомоку: board_to_features ----------------------------


def test_board_to_features_shape_and_one_hot() -> None:
    b = Board()
    b.place(0, 0, X)
    v = board_to_features(b, (0, 0))
    assert v.shape == (FEATURE_DIM,)
    assert v.dtype == np.float64
    # Ровно одна единица на клетку → сумма == 81.
    assert v.sum() == 81.0


def test_board_to_features_empty_centre_raises() -> None:
    """Центр окна должен быть занят — это сделанный ход."""
    b = Board()
    b.place(0, 0, X)  # есть какие-то камни, но (5,5) пуст
    with pytest.raises(ValueError):
        board_to_features(b, (5, 5))


def test_board_to_features_perspective_symmetry() -> None:
    """Перспектива зависит от того, чей ход — не от букв X/O.

    Возьмём две изоморфные позиции: одна как «X сходил на (0,0),
    O стоял на (1,1)», другая — наоборот. Признаки должны быть
    **идентичны**: каналы «свой/чужой» определяются ходящим.
    """
    bx = Board(); bx.place(0, 0, X); bx.place(1, 1, O)
    bo = Board(); bo.place(0, 0, O); bo.place(1, 1, X)
    vx = board_to_features(bx, (0, 0))  # ракурс X (только что сходил)
    vo = board_to_features(bo, (0, 0))  # ракурс O (только что сходил)
    assert np.array_equal(vx, vo)


def test_board_to_features_channels_correctly_assigned() -> None:
    """Проверяем индексы каналов на конкретной позиции.

    Положим X в центр (свой ход), O рядом, оставим пустые клетки.
    Сумма по каналу «свой» в окне 9×9 = число X-камней (включая центр).
    """
    b = Board()
    b.place(0, 0, X)        # центр — свой
    b.place(1, 0, X)        # ещё один свой рядом
    b.place(-1, 0, O)       # чужой рядом
    v = board_to_features(b, (0, 0))
    assert v[0::3].sum() == 81 - 3  # 78 пустых клеток в окне
    assert v[1::3].sum() == 2       # два X-камня (включая центр)
    assert v[2::3].sum() == 1       # один O-камень


# --- save/load weights -------------------------------------------------


def test_save_load_weights_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    p = Perceptron(n_features=243, random_state=0)
    p.W = rng.standard_normal(243)  # подменяем веса на узнаваемые
    p.save_weights(str(tmp_path / "w.npz"))
    p2 = Perceptron.load_weights(str(tmp_path / "w.npz"))
    assert p2.n_features == 243
    assert np.allclose(p.W, p2.W)
    assert p2.b == 0.0
    # Предсказания совпадают.
    x = rng.standard_normal(243)
    assert float(p.forward(x)) == float(p2.forward(x))


# --- PerceptronAgent ---------------------------------------------------


def _untrained_agent(symbol):
    """Маленький детерминированный агент без обучения — для юнит-тестов
    логики выбора хода (а не качества игры)."""
    p = Perceptron(n_features=FEATURE_DIM, init_method="small_random",
                   random_state=0)
    return PerceptronAgent(symbol, p, board_to_features)


def test_perceptron_agent_returns_legal_move_on_empty_board() -> None:
    agent = _untrained_agent(X)
    move = agent.choose_move(Board())
    assert move == (0, 0)  # на пустой доске окно поиска = [(0,0)]


def test_perceptron_agent_returns_move_inside_search_window() -> None:
    b = Board()
    b.place(0, 0, X)
    b.place(1, 1, O)
    agent = _untrained_agent(X)
    move = agent.choose_move(b)
    # Ход должен быть свободной клеткой из окна поиска.
    assert move in set(b.search_window())
    assert b.is_empty(*move)


def test_perceptron_agent_counts_forward_passes() -> None:
    b = Board()
    b.place(0, 0, X)
    b.place(1, 1, O)
    agent = _untrained_agent(X)
    moves = b.search_window()
    agent.choose_move(b)
    # Один forward на каждый кандидат (плюс инвариант, что окно > 1).
    assert agent.last_nodes == len(moves) > 1


# --- интеграция: PerceptronAgent vs Random -----------------------------


def test_perceptron_beats_random_after_training(tmp_path) -> None:
    """Обучаемся на маленьком датасете и обыгрываем Random ≥ 60%.

    25 партий depth=2 → ~600 сэмплов → 50 эпох SGD. Полная партия в
    тесте проигрывается до победы или ``max_moves=60``.
    """
    from agents.base import RandomAgent
    from tests.test_agents import run_match
    from training.train_perceptron import train

    model_path = tmp_path / "tiny.npz"
    train(
        num_games=25,
        epochs=50,
        lr=0.1,
        batch_size=32,
        depth=2,
        max_moves=30,
        val_size=0.2,
        seed=0,
        out_path=str(model_path),
        verbose=False,
    )

    wins = 0
    games = 0
    for seed in (1, 2, 3):
        # PerceptronAgent первым (X)
        agent_x = PerceptronAgent.from_disk(X, path=str(model_path))
        if run_match(agent_x, RandomAgent(O, seed=seed), max_moves=60) == X:
            wins += 1
        games += 1
        # PerceptronAgent вторым (O)
        agent_o = PerceptronAgent.from_disk(O, path=str(model_path))
        if run_match(RandomAgent(X, seed=seed), agent_o, max_moves=60) == O:
            wins += 1
        games += 1
    winrate = wins / games
    assert winrate > 0.60, f"перцептрон winrate={winrate:.0%} ниже 60%"
