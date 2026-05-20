"""Профилирование одного хода агента.

Контекстный менеджер :class:`MoveProfiler` оборачивает вызов
``agent.choose_move(board)``. По выходу из ``with``-блока он содержит:

* ``elapsed_ms`` — затраченное время в миллисекундах;
* ``nodes`` — счётчик трудозатрат, который сам агент пишет в
  ``last_nodes`` (узлы для Minimax/AB, симуляции для MCTS,
  forward-проходы для перцептрона);
* ``agent_name`` / ``metric_name`` — для подписи в боковой панели GUI.

Запись метрик ходов агрегируется в :class:`metrics.stats.SessionStats`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import TracebackType

from agents.base import Agent


@dataclass(frozen=True)
class MoveRecord:
    """Снимок метрик одного хода."""

    agent_name: str
    elapsed_ms: float
    nodes: int
    metric_name: str


class MoveProfiler:
    """Контекстный менеджер для измерения времени и узлов хода.

    Пример::

        with MoveProfiler(agent) as prof:
            move = agent.choose_move(board)
        stats.record(prof.record)

    Все поля заполняются на ``__exit__``; внутри ``with`` они равны
    «нулевым» значениям. Используется монотонный ``perf_counter``.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._t0: float = 0.0
        self.elapsed_ms: float = 0.0
        self.nodes: int = 0
        self.agent_name: str = agent.name
        self.metric_name: str = agent.metric_name

    def __enter__(self) -> "MoveProfiler":
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self.nodes = self._agent.last_nodes

    @property
    def record(self) -> MoveRecord:
        """Иммутабельный снимок для передачи в агрегатор статистики."""
        return MoveRecord(
            agent_name=self.agent_name,
            elapsed_ms=self.elapsed_ms,
            nodes=self.nodes,
            metric_name=self.metric_name,
        )
