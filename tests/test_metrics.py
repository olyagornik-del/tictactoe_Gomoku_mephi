"""Тесты профилирования и агрегации метрик."""

from __future__ import annotations

import time

from agents.base import RandomAgent
from game.board import O, X, Board
from metrics.profiler import MoveProfiler, MoveRecord
from metrics.stats import AgentStats, SessionStats


# --- MoveProfiler -------------------------------------------------------


def test_profiler_measures_elapsed_time() -> None:
    agent = RandomAgent(X, seed=0)
    board = Board()
    board.place(0, 0, X)  # окно > 1 для разнообразия
    with MoveProfiler(agent) as prof:
        # вставим небольшую паузу для надёжного измерения
        time.sleep(0.005)
        agent.choose_move(board)
    assert prof.elapsed_ms >= 5.0
    assert prof.agent_name == agent.name
    assert prof.metric_name == agent.metric_name


def test_profiler_captures_agent_last_nodes() -> None:
    agent = RandomAgent(X, seed=0)
    board = Board()
    board.place(0, 0, X)
    expected_window = len(board.search_window())
    with MoveProfiler(agent) as prof:
        agent.choose_move(board)
    # RandomAgent пишет в last_nodes длину окна.
    assert prof.nodes == expected_window > 1


def test_profiler_record_property() -> None:
    agent = RandomAgent(X, seed=0)
    with MoveProfiler(agent) as prof:
        agent.choose_move(Board())
    rec = prof.record
    assert isinstance(rec, MoveRecord)
    assert rec.agent_name == "Случайный"
    assert rec.metric_name == "ходов"
    assert rec.elapsed_ms >= 0.0


# --- AgentStats ---------------------------------------------------------


def _rec(name: str, ms: float, nodes: int = 0,
         metric_name: str = "узлов") -> MoveRecord:
    return MoveRecord(agent_name=name, elapsed_ms=ms,
                      nodes=nodes, metric_name=metric_name)


def test_agent_stats_avg_min_max_last() -> None:
    s = AgentStats()
    for ms in (10.0, 30.0, 20.0):
        s.update(_rec("X", ms, nodes=100))
    assert s.moves == 3
    assert s.min_ms == 10.0
    assert s.max_ms == 30.0
    assert s.last_ms == 20.0
    assert s.avg_ms == 20.0
    assert s.total_nodes == 300
    assert s.avg_nodes == 100.0
    assert s.last_nodes == 100


def test_agent_stats_empty_avg_is_zero_not_nan() -> None:
    s = AgentStats()
    assert s.avg_ms == 0.0
    assert s.avg_nodes == 0.0


# --- SessionStats -------------------------------------------------------


def test_session_stats_multi_agent() -> None:
    sess = SessionStats()
    sess.record(_rec("Минимакс", 12.0, 80))
    sess.record(_rec("Минимакс", 18.0, 120))
    sess.record(_rec("MCTS", 50.0, 1000, metric_name="симуляций"))
    assert len(sess) == 2
    mm = sess["Минимакс"]
    assert mm.moves == 2
    assert mm.avg_ms == 15.0
    assert mm.total_nodes == 200
    mc = sess["MCTS"]
    assert mc.moves == 1
    assert mc.metric_name == "симуляций"
    assert mc.last_nodes == 1000


def test_session_stats_rows_shape() -> None:
    sess = SessionStats()
    sess.record(_rec("AB", 10.0, 50))
    rows = sess.rows()
    assert len(rows) == 1
    row = rows[0]
    expected = {"agent", "moves", "avg_ms", "last_ms", "min_ms", "max_ms",
                "avg_nodes", "last_nodes", "total_nodes", "metric_name"}
    assert set(row.keys()) == expected
    assert row["agent"] == "AB"
    assert row["moves"] == 1


def test_session_stats_reset() -> None:
    sess = SessionStats()
    sess.record(_rec("AB", 10.0, 50))
    sess.record(_rec("MCTS", 20.0, 100))
    assert len(sess) == 2
    sess.reset()
    assert len(sess) == 0
    assert sess.rows() == []
    # После сброса новые записи накапливаются с нуля.
    sess.record(_rec("AB", 5.0, 25))
    assert sess["AB"].moves == 1
    assert sess["AB"].avg_ms == 5.0
