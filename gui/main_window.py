"""Главное окно pygame: цикл, события, оркестрация всех частей.

Окно 1280×800: слева доска (≈900×800), справа сайдбар (≈380×800).
Алгоритм ИИ переключается на лету — состояние партии не сбрасывается.
"""

from __future__ import annotations

import os
import sys
import time

import pygame

from agents.alphabeta import AlphaBetaAgent
from agents.base import Agent
from agents.mcts import MCTSAgent
from agents.minimax import MinimaxAgent
from agents.perceptron import (
    DEFAULT_MODEL_PATH,
    ENGINEERED_DIM,
    Perceptron,
    PerceptronAgent,
    move_features,
    pixel_features,
)
from game.board import O, X, Board
from game.rules import is_terminal, winner
from gui.board_view import BoardView
from gui.sidebar import ALGORITHM_LABEL_BY_KEY, Sidebar
from metrics.profiler import MoveProfiler
from metrics.stats import SessionStats

WINDOW_SIZE = (1280, 800)
BOARD_W = 900  # ширина области доски
FPS = 60


class GameWindow:
    """Главный класс приложения — владеет всеми остальными."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Гомоку: ИИ-арена")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()

        # Шрифты — Arial есть и на macOS, и в pygame fallback.
        self.font = pygame.font.SysFont("arial", 15)
        self.font_bold = pygame.font.SysFont("arial", 16, bold=True)
        self.font_small = pygame.font.SysFont("arial", 13)
        self.font_thinking = pygame.font.SysFont("arial", 22, bold=True)
        self.font_banner = pygame.font.SysFont("arial", 32, bold=True)

        # Модель и состояние.
        self.board = Board()
        # Человек ходит X (первым); ИИ ходит O. В ИИ-vs-ИИ оба автоматические.
        self.human_symbol: str = X
        self.ai_symbol: str = O
        self.current_player: str = X
        self.ai_vs_ai: bool = False
        self.game_over: bool = False

        # Метрики.
        self.session_stats = SessionStats()

        # Предзагрузка перцептрона (если веса есть).
        self._perceptron_model: Perceptron | None = None
        if os.path.exists(DEFAULT_MODEL_PATH):
            try:
                self._perceptron_model = Perceptron.load_weights(DEFAULT_MODEL_PATH)
            except Exception as exc:  # pragma: no cover
                print(f"[gui] Не удалось загрузить перцептрон: {exc}")

        # Доска и сайдбар.
        self.board_view = BoardView(
            rect=pygame.Rect(0, 0, BOARD_W, WINDOW_SIZE[1]),
            board=self.board,
        )
        self.sidebar = Sidebar(
            rect=pygame.Rect(BOARD_W, 0, WINDOW_SIZE[0] - BOARD_W, WINDOW_SIZE[1]),
            session_stats=self.session_stats,
            on_algorithm_change=self._on_algorithm_change,
            on_depth_change=lambda _v: None,   # пересоздание агента при ходе
            on_simulations_change=lambda _v: None,
            on_new_game=self._on_new_game,
            on_ai_vs_ai=self._on_toggle_ai_vs_ai,
            on_reset_stats=self._on_reset_stats,
        )
        if self._perceptron_model is None:
            self.sidebar.radio.disabled_keys.add("perceptron")

        self._update_legend()

        # Текущий ИИ-агент (создаётся по требованию).
        self._ai_cache: dict[tuple[str, str], Agent] = {}

        self.running = True

    # --- agent factory --------------------------------------------------

    def _make_agent(self, symbol: str) -> Agent:
        """Свежий агент по выбранному в сайдбаре алгоритму.

        Кэширует по (algo_key, symbol) — для воспроизводимости MCTS-сидов
        внутри сессии, но пересоздаёт при смене параметров (ниже).
        """
        algo = self.sidebar.current_algorithm
        if algo in ("minimax", "alphabeta"):
            depth = self.sidebar.current_depth
            key = (algo, symbol, depth)
        elif algo == "mcts":
            sims = self.sidebar.current_simulations
            key = (algo, symbol, sims)
        else:
            key = (algo, symbol)

        if key in self._ai_cache:
            return self._ai_cache[key]

        if algo == "minimax":
            agent = MinimaxAgent(symbol, depth=self.sidebar.current_depth)
        elif algo == "alphabeta":
            agent = AlphaBetaAgent(symbol, depth=self.sidebar.current_depth)
        elif algo == "mcts":
            agent = MCTSAgent(
                symbol, simulations=self.sidebar.current_simulations, seed=42,
            )
        elif algo == "perceptron":
            assert self._perceptron_model is not None
            # Признаки и предохранитель — по типу загруженной модели:
            # инженерная (12 весов) защищается сама → guard off;
            # пиксельная (243) — нет → guard on.
            engineered = self._perceptron_model.n_features == ENGINEERED_DIM
            feature_fn = move_features if engineered else pixel_features
            agent = PerceptronAgent(
                symbol, self._perceptron_model, feature_fn,
                tactical=not engineered,
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown algorithm: {algo}")
        self._ai_cache[key] = agent
        return agent

    # --- sidebar callbacks ---------------------------------------------

    def _on_algorithm_change(self, _new_key: str) -> None:
        # Кэш ходов не сбрасываем — кэшируется по ключу. Просто на
        # следующем ходу будет использован агент с новым алгоритмом.
        pass

    def _on_new_game(self) -> None:
        self.board = Board()
        self.board_view.board = self.board
        self.board_view.focus_on_bbox()
        self.board_view.banner = None
        self.current_player = X
        self.game_over = False

    def _on_toggle_ai_vs_ai(self) -> None:
        self.ai_vs_ai = not self.ai_vs_ai
        self.sidebar.set_ai_vs_ai_label(self.ai_vs_ai)
        self._update_legend()
        # Если включили посреди партии — сбросим её, чтобы оба играли с нуля.
        if self.ai_vs_ai:
            self._on_new_game()

    def _update_legend(self) -> None:
        if self.ai_vs_ai:
            self.board_view.legend = [(X, "ИИ"), (O, "ИИ")]
        else:
            self.board_view.legend = [(X, "вы"), (O, "ИИ")]

    def _on_reset_stats(self) -> None:
        self.session_stats.reset()

    # --- ход --------------------------------------------------------

    def _is_ai_turn(self) -> bool:
        if self.game_over:
            return False
        if self.ai_vs_ai:
            return True
        return self.current_player == self.ai_symbol

    def _do_ai_move(self) -> None:
        """Синхронный ход ИИ с индикатором «ИИ думает…»."""
        # Не даём играть Перцептроном, если веса не загружены.
        if self.sidebar.current_algorithm == "perceptron" and self._perceptron_model is None:
            return
        self.board_view.thinking = True
        self.draw()
        pygame.display.flip()

        agent = self._make_agent(self.current_player)
        try:
            with MoveProfiler(agent) as prof:
                move = agent.choose_move(self.board)
            self.session_stats.record(prof.record)
            self._place(move)
        finally:
            self.board_view.thinking = False

    def _place(self, move) -> None:
        self.board.place(*move, self.current_player)
        if is_terminal(self.board):
            win = winner(self.board)
            if win == X:
                self.board_view.banner = "Победили крестики (X)!"
            elif win == O:
                self.board_view.banner = "Победили нолики (O)!"
            else:
                self.board_view.banner = "Ничья"
            self.game_over = True
        else:
            self.current_player = O if self.current_player == X else X

    def _handle_human_click(self, pos: tuple[int, int]) -> None:
        if self.game_over or self.ai_vs_ai:
            return
        if self.current_player != self.human_symbol:
            return
        cell = self.board_view.click_to_cell(pos)
        if cell is None or not self.board.is_empty(*cell):
            return
        # Чтобы клик «зашёл» в окно поиска, в радиус 2 от ВСЕХ камней
        # уже попадают все разумные ходы. На пустой доске любая клетка ок.
        self._place(cell)

    # --- цикл -----------------------------------------------------------

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_human_click(event.pos)
                self.board_view.handle_event(event)
                self.sidebar.handle_event(event)

            if self._is_ai_turn():
                self._do_ai_move()

            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))
        self.board_view.draw(self.screen, self.font_thinking, self.font_banner)
        self.sidebar.draw(self.screen, self.font, self.font_bold, self.font_small)


def main() -> None:
    GameWindow().run()


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
