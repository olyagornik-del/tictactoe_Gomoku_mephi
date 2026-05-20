"""Точка входа лабораторной работы.

Запуск:

    python -m lab.run_lab

Делает по порядку:

1. baseline (``lr=0.1, epochs=100, batch_size=32, init='small_random'``):
   loss-кривые, граница решения, accuracy на train/test;
2. эксперимент по learning rate;
3. эксперимент по batch size;
4. эксперимент по инициализации;
5. ROC-кривая, precision / recall / F1 / confusion matrix для baseline.

Все графики кладутся в ``lab/figures/``.
"""

from __future__ import annotations

import os

from agents.perceptron import Perceptron
from lab.data_prep import load_classification_data
from lab.experiments import (
    experiment_batch_size,
    experiment_initialization,
    experiment_learning_rate,
)
from lab.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
    roc_auc,
)
from lab.visualization import (
    plot_decision_boundary,
    plot_loss_curves,
    plot_misclassified,
    plot_roc_curve,
)

FIGURES_DIR = "lab/figures"


def _path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def main(seed: int = 0) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- данные ---------------------------------------------------------
    X_train, X_test, y_train, y_test, scaler = load_classification_data()
    print(f"[lab] train: {X_train.shape}, test: {X_test.shape}")

    # --- baseline -------------------------------------------------------
    print("\n[lab] baseline: lr=0.1, epochs=100, batch_size=32, init=small_random")
    model = Perceptron(n_features=2, init_method="small_random", random_state=seed)
    history = model.fit(
        X_train, y_train, X_test, y_test,
        epochs=100, lr=0.1, batch_size=32, random_state=seed,
    )

    tr_acc = accuracy(y_train, model.predict(X_train))
    te_acc = accuracy(y_test, model.predict(X_test))
    print(f"[lab] baseline accuracy: train={tr_acc:.3f}, test={te_acc:.3f}")
    print(f"[lab] финальная loss: train={history['loss_train'][-1]:.4f}, "
          f"val={history['loss_val'][-1]:.4f}")
    print(f"[lab] веса: W={model.W}, b={model.b}")

    plot_loss_curves(history, "Baseline: BCE-потеря",
                     _path("baseline_loss.png"))
    plot_decision_boundary(model, X_train, y_train, scaler,
                           _path("baseline_decision_train.png"),
                           title="Baseline: граница решения (train)")
    plot_decision_boundary(model, X_test, y_test, scaler,
                           _path("baseline_decision_test.png"),
                           title="Baseline: граница решения (test)")
    plot_misclassified(model, X_test, y_test,
                       _path("baseline_misclassified.png"),
                       title="Baseline: ошибки на test")

    # --- эксперименты ---------------------------------------------------
    experiment_learning_rate(X_train, y_train, X_test, y_test,
                             save_path=_path("exp_learning_rate.png"), seed=seed)
    experiment_batch_size(X_train, y_train, X_test, y_test,
                          save_path=_path("exp_batch_size.png"), seed=seed)
    experiment_initialization(X_train, y_train, X_test, y_test,
                              save_path=_path("exp_initialization.png"), seed=seed)

    # --- ROC + PRF1 для baseline ---------------------------------------
    y_score = model.forward(X_test)
    auc = roc_auc(y_test, y_score)
    pr = precision(y_test, model.predict(X_test))
    rc = recall(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))

    print()
    print("=== Метрики baseline на test ===")
    print(f"  accuracy : {te_acc:.3f}")
    print(f"  precision: {pr:.3f}")
    print(f"  recall   : {rc:.3f}")
    print(f"  F1       : {f1:.3f}")
    print(f"  ROC AUC  : {auc:.3f}")
    print(f"  confusion matrix [[TN FP][FN TP]] =\n{cm}")

    plot_roc_curve(y_test, y_score,
                   _path("baseline_roc.png"),
                   title=f"Baseline: ROC (AUC={auc:.3f})")

    print(f"\n[lab] Готово. Графики: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
