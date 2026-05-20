"""Агрегация метрик по агентам за сессию.

GUI рисует таблицу: для каждого алгоритма (Минимакс / Альфа-бета /
MCTS / Перцептрон, плюс Случайный для бенчмарка) показываются:
среднее, минимум, максимум, последний ход (мс), число ходов и
суммарные «узлы/симуляции/прогоны».

«Сбросить статистику» в боковой панели вызывает :meth:`SessionStats.reset`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from metrics.profiler import MoveRecord


@dataclass
class AgentStats:
    """Накопитель метрик по одному агенту."""

    moves: int = 0
    total_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0
    total_nodes: int = 0
    last_nodes: int = 0
    metric_name: str = ""

    @property
    def avg_ms(self) -> float:
        """Среднее время хода в миллисекундах (0 для пустой статистики)."""
        return self.total_ms / self.moves if self.moves else 0.0

    @property
    def avg_nodes(self) -> float:
        """Среднее число «узлов» (или симуляций/прогонов) на ход."""
        return self.total_nodes / self.moves if self.moves else 0.0

    def update(self, record: MoveRecord) -> None:
        """Добавить один сэмпл из :class:`metrics.profiler.MoveRecord`."""
        ms = record.elapsed_ms
        if self.moves == 0:
            self.min_ms = ms
            self.max_ms = ms
        else:
            if ms < self.min_ms:
                self.min_ms = ms
            if ms > self.max_ms:
                self.max_ms = ms
        self.moves += 1
        self.total_ms += ms
        self.last_ms = ms
        self.total_nodes += record.nodes
        self.last_nodes = record.nodes
        self.metric_name = record.metric_name


@dataclass
class SessionStats:
    """Все агенты сессии. Ключ словаря — :attr:`Agent.name`."""

    by_agent: dict[str, AgentStats] = field(default_factory=dict)

    def record(self, record: MoveRecord) -> None:
        """Зафиксировать ход в строке нужного агента."""
        stats = self.by_agent.get(record.agent_name)
        if stats is None:
            stats = AgentStats()
            self.by_agent[record.agent_name] = stats
        stats.update(record)

    def reset(self) -> None:
        """Очистить всю статистику (для кнопки «Сбросить статистику»)."""
        self.by_agent.clear()

    def rows(self) -> list[dict[str, object]]:
        """Готовые строки для табличного виджета боковой панели.

        Каждый словарь содержит ключи: ``agent``, ``moves``, ``avg_ms``,
        ``last_ms``, ``min_ms``, ``max_ms``, ``avg_nodes``,
        ``last_nodes``, ``total_nodes``, ``metric_name``.
        """
        return [
            {
                "agent": name,
                "moves": s.moves,
                "avg_ms": s.avg_ms,
                "last_ms": s.last_ms,
                "min_ms": s.min_ms,
                "max_ms": s.max_ms,
                "avg_nodes": s.avg_nodes,
                "last_nodes": s.last_nodes,
                "total_nodes": s.total_nodes,
                "metric_name": s.metric_name,
            }
            for name, s in self.by_agent.items()
        ]

    def __getitem__(self, agent_name: str) -> AgentStats:
        return self.by_agent[agent_name]

    def __contains__(self, agent_name: str) -> bool:
        return agent_name in self.by_agent

    def __len__(self) -> int:
        return len(self.by_agent)

    def __iter__(self) -> Iterable[str]:
        return iter(self.by_agent)
