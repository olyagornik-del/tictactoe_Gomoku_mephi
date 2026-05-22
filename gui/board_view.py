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

# Цвета — крестики-нолики на разлинованном поле.
COLOR_BG = (252, 250, 242)        # светлый фон поля
COLOR_GRID = (180, 180, 180)      # линии сетки
COLOR_GRID_ORIGIN = (228, 236, 248)  # бледная подсветка клетки (0,0)
COLOR_LAST_MOVE = (90, 170, 90)   # обводка последнего хода (зелёная)
COLOR_X = (40, 90, 200)           # синий крестик
COLOR_O = (210, 60, 60)           # красный нолик
COLOR_THINKING_BG = (255, 255, 240)
COLOR_THINKING_FG = (60, 60, 60)
COLOR_BANNER_WIN_X = (40, 90, 200)
COLOR_BANNER_WIN_O = (210, 60, 60)
COLOR_BANNER_FG = (255, 255, 255)
COLOR_BANNER_FG_O = (255, 255, 255)
COLOR_LEGEND_BG = (255, 255, 255)
COLOR_LEGEND_FG = (50, 50, 50)
COLOR_LEGEND_BORDER = (210, 210, 210)

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
        #: Легенда «кто чем играет»: список (символ, подпись).
        self.legend: list[tuple[str, str]] = [(X, "крестики"), (O, "нолики")]

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
        self._draw_legend(surface, font_banner)
        if self.thinking:
            self._draw_thinking(surface, font_thinking)
        if self.banner:
            self._draw_banner(surface, font_banner)

    def _draw_legend(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Плашка «кто чем играет» в левом верхнем углу доски."""
        if not self.legend:
            return
        # font тут крупный (banner); для подписей используем мельче.
        small = pygame.font.SysFont("arial", 15)
        line_h = 24
        pad = 10
        box_w = 168
        box_h = pad * 2 + line_h * len(self.legend)
        box = pygame.Rect(self.rect.left + 12, self.rect.top + 12, box_w, box_h)
        pygame.draw.rect(surface, COLOR_LEGEND_BG, box, border_radius=8)
        pygame.draw.rect(
            surface, COLOR_LEGEND_BORDER, box, width=1, border_radius=8
        )
        for i, (sym, label) in enumerate(self.legend):
            cy = box.top + pad + line_h * i + line_h // 2
            mark_cx = box.left + pad + 12
            self._draw_mark(surface, sym, mark_cx, cy, 26)
            text = small.render(f"{sym} — {label}", True, COLOR_LEGEND_FG)
            surface.blit(text, (mark_cx + 22, cy - text.get_height() // 2))

    def _draw_grid(self, surface: pygame.Surface) -> None:
        # Сколько клеток помещается в полу-ширину/высоту.
        half_w = self.rect.width / 2 / self.cell_size
        half_h = self.rect.height / 2 / self.cell_size
        x_min = int(self.cx - half_w) - 1
        x_max = int(self.cx + half_w) + 1
        y_min = int(self.cy - half_h) - 1
        y_max = int(self.cy + half_h) + 1

        # Подсветка клетки (0,0), чтобы было видно начало координат.
        ox, oy = self.world_to_screen(0, 0)
        half = self.cell_size / 2
        origin_rect = pygame.Rect(
            int(ox - half), int(oy - half),
            int(self.cell_size), int(self.cell_size),
        )
        if self.rect.colliderect(origin_rect):
            pygame.draw.rect(surface, COLOR_GRID_ORIGIN, origin_rect)

        # Линии — на ГРАНИЦАХ клеток (полуцелые мировые координаты),
        # чтобы фигуры (в целых координатах) попадали внутрь клеток.
        for x in range(x_min, x_max + 1):
            sx, _ = self.world_to_screen(x - 0.5, 0)
            pygame.draw.line(
                surface, COLOR_GRID, (sx, self.rect.top), (sx, self.rect.bottom), 1
            )
        for y in range(y_min, y_max + 1):
            _, sy = self.world_to_screen(0, y - 0.5)
            pygame.draw.line(
                surface, COLOR_GRID, (self.rect.left, sy), (self.rect.right, sy), 1
            )

    def _draw_stones(self, surface: pygame.Surface) -> None:
        for (wx, wy), sym in self.board.cells.items():
            sx, sy = self.world_to_screen(wx, wy)
            if not self.rect.collidepoint(sx, sy):
                continue
            self._draw_mark(surface, sym, sx, sy, self.cell_size)

    @staticmethod
    def _draw_mark(
        surface: pygame.Surface, sym: str, sx: float, sy: float, cell: float
    ) -> None:
        """Нарисовать ✕ или ◯ в точке ``(sx, sy)`` под размер клетки ``cell``."""
        half = cell * 0.30
        line_w = max(2, int(cell * 0.11))
        if sym == X:
            pygame.draw.line(
                surface, COLOR_X,
                (sx - half, sy - half), (sx + half, sy + half), line_w,
            )
            pygame.draw.line(
                surface, COLOR_X,
                (sx - half, sy + half), (sx + half, sy - half), line_w,
            )
        elif sym == O:
            pygame.draw.circle(
                surface, COLOR_O, (int(sx), int(sy)), int(cell * 0.33), line_w,
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
