"""Виджет доски: отрисовка сетки/камней, камера, ввод мыши.

Камера задана центром в мировых координатах ``(cx, cy)`` и шагом
сетки ``cell_size`` (пиксели на одну клетку).

* **Зум** — колесо мыши умножает/делит ``cell_size``.
* **Пан** — драг правой кнопкой мыши.
* **Клик левой** в области доски возвращает координату клетки.
"""

from __future__ import annotations

import pygame

from game.board import Board, Coord, O, X

# Цвета — традиционная гомоку-доска.
COLOR_BG = (240, 207, 150)        # дерево
COLOR_GRID = (50, 35, 20)         # тёмные линии
COLOR_GRID_ORIGIN = (200, 60, 60) # подсветка осей через (0,0)
COLOR_LAST_MOVE = (220, 60, 60)   # обводка последнего хода
COLOR_X = (15, 15, 15)            # чёрный камень
COLOR_O = (245, 245, 245)         # белый камень
COLOR_O_BORDER = (15, 15, 15)
COLOR_THINKING_BG = (255, 255, 240)
COLOR_THINKING_FG = (60, 60, 60)
COLOR_BANNER_WIN_X = (40, 40, 40)
COLOR_BANNER_WIN_O = (220, 220, 220)
COLOR_BANNER_FG = (255, 255, 255)
COLOR_BANNER_FG_O = (15, 15, 15)

MIN_CELL = 14
MAX_CELL = 70


class BoardView:
    """Отрисовка доски и обработка ввода в её прямоугольнике.

    :param rect: pygame.Rect — область экрана под доску.
    :param board: модель доски.
    """

    def __init__(self, rect: pygame.Rect, board: Board) -> None:
        self.rect = rect
        self.board = board
        self.cell_size: float = 40.0
        # Центр камеры в мировых координатах (float для плавного пана).
        self.cx: float = 0.0
        self.cy: float = 0.0
        self._panning: bool = False
        self._pan_last: tuple[int, int] = (0, 0)
        self.thinking: bool = False
        self.banner: str | None = None  # текст победы / ничьей

    # --- камера ---------------------------------------------------------

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        sx = self.rect.centerx + (wx - self.cx) * self.cell_size
        sy = self.rect.centery + (wy - self.cy) * self.cell_size
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        wx = self.cx + (sx - self.rect.centerx) / self.cell_size
        wy = self.cy + (sy - self.rect.centery) / self.cell_size
        return wx, wy

    def click_to_cell(self, pos: tuple[int, int]) -> Coord | None:
        """Преобразовать позицию клика в координату клетки или ``None``."""
        if not self.rect.collidepoint(pos):
            return None
        wx, wy = self.screen_to_world(*pos)
        return (round(wx), round(wy))

    def focus_on_bbox(self) -> None:
        """Центрировать камеру на bbox сыгранных клеток."""
        bbox = self.board.bounding_box()
        if bbox is None:
            self.cx = self.cy = 0.0
            return
        min_x, min_y, max_x, max_y = bbox
        self.cx = (min_x + max_x) / 2.0
        self.cy = (min_y + max_y) / 2.0
        # Если игра разрослась — отъезжаем, чтоб всё помещалось.
        span = max(max_x - min_x, max_y - min_y) + 4
        if span > 0:
            needed_cell = min(
                self.rect.width / max(span, 1),
                self.rect.height / max(span, 1),
            )
            self.cell_size = max(MIN_CELL, min(MAX_CELL, needed_cell))

    # --- обработка ввода -----------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # колесо вверх → зум in
                self._zoom_at(event.pos, factor=1.15)
            elif event.button == 5:  # колесо вниз → зум out
                self._zoom_at(event.pos, factor=1 / 1.15)
            elif event.button == 3 and self.rect.collidepoint(event.pos):
                self._panning = True
                self._pan_last = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                self._panning = False
        elif event.type == pygame.MOUSEMOTION:
            if self._panning:
                dx = event.pos[0] - self._pan_last[0]
                dy = event.pos[1] - self._pan_last[1]
                self.cx -= dx / self.cell_size
                self.cy -= dy / self.cell_size
                self._pan_last = event.pos
        elif event.type == pygame.MOUSEWHEEL:
            # На macOS pygame чаще шлёт MOUSEWHEEL вместо BUTTONDOWN 4/5.
            mouse_pos = pygame.mouse.get_pos()
            if event.y > 0:
                self._zoom_at(mouse_pos, factor=1.15 ** event.y)
            elif event.y < 0:
                self._zoom_at(mouse_pos, factor=(1 / 1.15) ** (-event.y))

    def _zoom_at(self, mouse_pos: tuple[int, int], factor: float) -> None:
        """Зум с сохранением точки под курсором (если курсор над доской)."""
        if not self.rect.collidepoint(mouse_pos):
            # просто меняем зум вокруг центра камеры
            self.cell_size = max(MIN_CELL, min(MAX_CELL, self.cell_size * factor))
            return
        before = self.screen_to_world(*mouse_pos)
        self.cell_size = max(MIN_CELL, min(MAX_CELL, self.cell_size * factor))
        after = self.screen_to_world(*mouse_pos)
        self.cx += before[0] - after[0]
        self.cy += before[1] - after[1]

    # --- отрисовка ------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        font_thinking: pygame.font.Font,
        font_banner: pygame.font.Font,
    ) -> None:
        surface.set_clip(self.rect)
        surface.fill(COLOR_BG, self.rect)
        self._draw_grid(surface)
        self._draw_stones(surface)
        self._draw_last_move(surface)
        surface.set_clip(None)
        if self.thinking:
            self._draw_thinking(surface, font_thinking)
        if self.banner:
            self._draw_banner(surface, font_banner)

    def _draw_grid(self, surface: pygame.Surface) -> None:
        # Сколько клеток помещается в полу-ширину/высоту.
        half_w = self.rect.width / 2 / self.cell_size
        half_h = self.rect.height / 2 / self.cell_size
        x_min = int(self.cx - half_w) - 1
        x_max = int(self.cx + half_w) + 1
        y_min = int(self.cy - half_h) - 1
        y_max = int(self.cy + half_h) + 1
        for x in range(x_min, x_max + 1):
            sx, _ = self.world_to_screen(x, 0)
            color = COLOR_GRID_ORIGIN if x == 0 else COLOR_GRID
            width = 2 if x == 0 else 1
            pygame.draw.line(
                surface, color, (sx, self.rect.top), (sx, self.rect.bottom), width
            )
        for y in range(y_min, y_max + 1):
            _, sy = self.world_to_screen(0, y)
            color = COLOR_GRID_ORIGIN if y == 0 else COLOR_GRID
            width = 2 if y == 0 else 1
            pygame.draw.line(
                surface, color, (self.rect.left, sy), (self.rect.right, sy), width
            )

    def _draw_stones(self, surface: pygame.Surface) -> None:
        radius = self.cell_size * 0.42
        border = max(2, int(self.cell_size * 0.06))
        for (wx, wy), sym in self.board.cells.items():
            sx, sy = self.world_to_screen(wx, wy)
            if not self.rect.collidepoint(sx, sy):
                continue
            if sym == X:
                pygame.draw.circle(surface, COLOR_X, (int(sx), int(sy)), int(radius))
            elif sym == O:
                pygame.draw.circle(surface, COLOR_O, (int(sx), int(sy)), int(radius))
                pygame.draw.circle(
                    surface, COLOR_O_BORDER, (int(sx), int(sy)), int(radius), border
                )

    def _draw_last_move(self, surface: pygame.Surface) -> None:
        last = self.board.last_move
        if last is None:
            return
        sx, sy = self.world_to_screen(*last)
        if not self.rect.collidepoint(sx, sy):
            return
        radius = self.cell_size * 0.48
        pygame.draw.circle(
            surface, COLOR_LAST_MOVE, (int(sx), int(sy)), int(radius), 3
        )

    def _draw_thinking(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        text = font.render("ИИ думает…", True, COLOR_THINKING_FG)
        pad = 12
        box = text.get_rect()
        box.inflate_ip(pad * 2, pad)
        box.centerx = self.rect.centerx
        box.top = self.rect.top + 16
        pygame.draw.rect(surface, COLOR_THINKING_BG, box, border_radius=10)
        pygame.draw.rect(surface, COLOR_THINKING_FG, box, width=1, border_radius=10)
        text_rect = text.get_rect(center=box.center)
        surface.blit(text, text_rect)

    def _draw_banner(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        assert self.banner is not None
        # Цвет фона зависит от победителя для лучшей видимости.
        win_x = "X" in self.banner
        bg = COLOR_BANNER_WIN_X if win_x else COLOR_BANNER_WIN_O
        fg = COLOR_BANNER_FG if win_x else COLOR_BANNER_FG_O
        text = font.render(self.banner, True, fg)
        pad = 16
        box = text.get_rect()
        box.inflate_ip(pad * 2, pad)
        box.center = self.rect.center
        pygame.draw.rect(surface, bg, box, border_radius=12)
        text_rect = text.get_rect(center=box.center)
        surface.blit(text, text_rect)
