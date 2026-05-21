"""Боковая панель: виджеты + раскладка.

Виджеты написаны вручную (Button, Slider, RadioGroup, Label) — pygame
не имеет встроенных. Все надписи на русском (ТЗ).
"""

from __future__ import annotations

from typing import Callable

import pygame

from metrics.stats import SessionStats

# --- цвета ---------------------------------------------------------------
COLOR_BG = (245, 245, 245)
COLOR_TEXT = (30, 30, 30)
COLOR_MUTED = (110, 110, 110)
COLOR_HEADING = (40, 40, 40)
COLOR_BORDER = (200, 200, 200)
COLOR_TABLE_HEAD = (220, 220, 220)
COLOR_TABLE_ROW = (255, 255, 255)
COLOR_TABLE_ROW_ALT = (240, 240, 240)

COLOR_BUTTON = (235, 235, 235)
COLOR_BUTTON_HOVER = (215, 215, 215)
COLOR_BUTTON_BORDER = (170, 170, 170)
COLOR_BUTTON_DISABLED = (240, 240, 240)
COLOR_BUTTON_TEXT_DISABLED = (170, 170, 170)

COLOR_RADIO_ACTIVE_BG = (60, 110, 200)
COLOR_RADIO_ACTIVE_FG = (255, 255, 255)
COLOR_RADIO_INACTIVE_BG = (235, 235, 235)
COLOR_RADIO_INACTIVE_FG = (40, 40, 40)
COLOR_RADIO_BORDER = (170, 170, 170)

COLOR_SLIDER_TRACK = (210, 210, 210)
COLOR_SLIDER_FILL = (60, 110, 200)
COLOR_SLIDER_THUMB = (40, 80, 170)


# ============================================================================
# Виджеты
# ============================================================================


class Button:
    """Кнопка с подписью и callback."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        on_click: Callable[[], None],
    ) -> None:
        self.rect = rect
        self.label = label
        self.on_click = on_click
        self._hover = False
        self.enabled = True

    def handle_event(self, event: pygame.event.Event) -> None:
        if not self.enabled:
            return
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.enabled:
            bg = COLOR_BUTTON_DISABLED
            fg = COLOR_BUTTON_TEXT_DISABLED
        else:
            bg = COLOR_BUTTON_HOVER if self._hover else COLOR_BUTTON
            fg = COLOR_TEXT
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(
            surface, COLOR_BUTTON_BORDER, self.rect, width=1, border_radius=6
        )
        text = font.render(self.label, True, fg)
        surface.blit(text, text.get_rect(center=self.rect.center))


class RadioGroup:
    """Группа взаимоисключающих опций (вертикальная)."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        options: list[tuple[str, str]],
        default_key: str,
        on_change: Callable[[str], None],
        row_height: int = 34,
        gap: int = 6,
    ) -> None:
        """``options`` — список ``(key, label)``; key — внутренний идентификатор."""
        self.options = options
        self.selected: str = default_key
        self.on_change = on_change
        self.row_rects: dict[str, pygame.Rect] = {}
        self.disabled_keys: set[str] = set()
        for i, (key, _) in enumerate(options):
            self.row_rects[key] = pygame.Rect(
                x, y + i * (row_height + gap), width, row_height
            )

    @property
    def bottom(self) -> int:
        return max(r.bottom for r in self.row_rects.values())

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, rect in self.row_rects.items():
                if key in self.disabled_keys:
                    continue
                if rect.collidepoint(event.pos) and self.selected != key:
                    self.selected = key
                    self.on_change(key)
                    return

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        for key, label in self.options:
            rect = self.row_rects[key]
            disabled = key in self.disabled_keys
            active = key == self.selected
            if disabled:
                bg = COLOR_BUTTON_DISABLED
                fg = COLOR_BUTTON_TEXT_DISABLED
            elif active:
                bg = COLOR_RADIO_ACTIVE_BG
                fg = COLOR_RADIO_ACTIVE_FG
            else:
                bg = COLOR_RADIO_INACTIVE_BG
                fg = COLOR_RADIO_INACTIVE_FG
            pygame.draw.rect(surface, bg, rect, border_radius=6)
            pygame.draw.rect(
                surface, COLOR_RADIO_BORDER, rect, width=1, border_radius=6
            )
            text = font.render(label, True, fg)
            surface.blit(text, text.get_rect(midleft=(rect.left + 14, rect.centery)))


class Slider:
    """Горизонтальный ползунок целочисленных значений."""

    def __init__(
        self,
        rect: pygame.Rect,
        min_value: int,
        max_value: int,
        value: int,
        label_fmt: str,
        on_change: Callable[[int], None] | None = None,
    ) -> None:
        self.rect = rect
        self.min = min_value
        self.max = max_value
        self.value = value
        self.label_fmt = label_fmt
        self.on_change = on_change
        self._dragging = False
        self.visible = True
        self.enabled = True

    def _set_from_x(self, x: int) -> None:
        fx = (x - self.rect.left) / max(1, self.rect.width)
        fx = max(0.0, min(1.0, fx))
        new_value = int(round(self.min + fx * (self.max - self.min)))
        if new_value != self.value:
            self.value = new_value
            if self.on_change:
                self.on_change(new_value)

    def handle_event(self, event: pygame.event.Event) -> None:
        if not self.visible or not self.enabled:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._dragging = True
                self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            self._set_from_x(event.pos[0])

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.visible:
            return
        # label
        label = font.render(
            self.label_fmt.format(value=self.value), True, COLOR_TEXT
        )
        surface.blit(label, (self.rect.left, self.rect.top - 22))
        # track
        track_y = self.rect.centery
        pygame.draw.line(
            surface, COLOR_SLIDER_TRACK,
            (self.rect.left, track_y), (self.rect.right, track_y), 4,
        )
        frac = (self.value - self.min) / max(1, (self.max - self.min))
        fill_x = self.rect.left + int(frac * self.rect.width)
        pygame.draw.line(
            surface, COLOR_SLIDER_FILL,
            (self.rect.left, track_y), (fill_x, track_y), 4,
        )
        pygame.draw.circle(surface, COLOR_SLIDER_THUMB, (fill_x, track_y), 9)


# ============================================================================
# Sidebar — раскладка
# ============================================================================


# Ключи алгоритмов и их подписи (используются и для статистики, и для UI).
ALGORITHM_OPTIONS: list[tuple[str, str]] = [
    ("minimax",    "Минимакс"),
    ("alphabeta",  "Альфа-бета"),
    ("mcts",       "MCTS"),
    ("perceptron", "Перцептрон"),
]
ALGORITHM_LABEL_BY_KEY = {k: lbl for k, lbl in ALGORITHM_OPTIONS}


class Sidebar:
    """Правая боковая панель: переключатель, ползунок, таблица, кнопки.

    Изменения значений происходят через callback'и, переданные в
    конструкторе — их зовёт :class:`gui.main_window.GameWindow`.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        session_stats: SessionStats,
        on_algorithm_change: Callable[[str], None],
        on_depth_change: Callable[[int], None],
        on_simulations_change: Callable[[int], None],
        on_new_game: Callable[[], None],
        on_ai_vs_ai: Callable[[], None],
        on_reset_stats: Callable[[], None],
    ) -> None:
        self.rect = rect
        self.session_stats = session_stats
        self.on_algorithm_change_cb = on_algorithm_change
        self.on_depth_change = on_depth_change
        self.on_simulations_change = on_simulations_change

        pad = 16
        x = rect.left + pad
        width = rect.width - 2 * pad
        y = rect.top + pad

        # Заголовок «Алгоритм ИИ:» — рисуется в draw(), здесь только отступ.
        self._title_algo_y = y
        y += 28

        self.radio = RadioGroup(
            x=x, y=y, width=width, options=ALGORITHM_OPTIONS,
            default_key="alphabeta",
            on_change=self._on_radio_change,
        )
        y = self.radio.bottom + 22

        # Заголовок «Настройки:»
        self._title_settings_y = y
        y += 28

        # Два ползунка живут в одном слоте (видим только релевантный).
        slider_rect = pygame.Rect(x, y + 22, width, 24)
        self.depth_slider = Slider(
            rect=slider_rect, min_value=1, max_value=6, value=3,
            label_fmt="Глубина: {value}",
            on_change=self.on_depth_change,
        )
        self.sims_slider = Slider(
            rect=slider_rect, min_value=100, max_value=5000, value=800,
            label_fmt="Симуляций: {value}",
            on_change=self.on_simulations_change,
        )
        self._perceptron_note_y = slider_rect.top
        # По умолчанию активен AB → показываем depth.
        self._set_visible_slider("alphabeta")
        y = slider_rect.bottom + 28

        # Таблица метрик.
        self._table_top = y
        self._table_height = 4 * 24 + 28
        y += self._table_height + 16

        # Нижние кнопки.
        btn_h = 36
        self.btn_new_game = Button(
            pygame.Rect(x, y, width, btn_h), "Новая игра", on_new_game,
        )
        y += btn_h + 8
        self.btn_ai_vs_ai = Button(
            pygame.Rect(x, y, width, btn_h), "ИИ vs ИИ", on_ai_vs_ai,
        )
        y += btn_h + 8
        self.btn_reset = Button(
            pygame.Rect(x, y, width, btn_h), "Сбросить статистику",
            on_reset_stats,
        )

    # --- callback-обёртка для radio ------------------------------------

    def _on_radio_change(self, key: str) -> None:
        self._set_visible_slider(key)
        self.on_algorithm_change_cb(key)

    def _set_visible_slider(self, algo_key: str) -> None:
        is_minimax_ab = algo_key in ("minimax", "alphabeta")
        is_mcts = algo_key == "mcts"
        self.depth_slider.visible = is_minimax_ab
        self.sims_slider.visible = is_mcts

    # --- внешний API для GameWindow ------------------------------------

    @property
    def current_algorithm(self) -> str:
        return self.radio.selected

    @property
    def current_depth(self) -> int:
        return self.depth_slider.value

    @property
    def current_simulations(self) -> int:
        return self.sims_slider.value

    def set_ai_vs_ai_label(self, active: bool) -> None:
        self.btn_ai_vs_ai.label = "Остановить ИИ vs ИИ" if active else "ИИ vs ИИ"

    # --- события и отрисовка -------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        self.radio.handle_event(event)
        self.depth_slider.handle_event(event)
        self.sims_slider.handle_event(event)
        self.btn_new_game.handle_event(event)
        self.btn_ai_vs_ai.handle_event(event)
        self.btn_reset.handle_event(event)

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        font_bold: pygame.font.Font,
        font_small: pygame.font.Font,
    ) -> None:
        # фон
        surface.fill(COLOR_BG, self.rect)
        pygame.draw.line(
            surface, COLOR_BORDER,
            (self.rect.left, self.rect.top),
            (self.rect.left, self.rect.bottom), 1,
        )

        pad = 16
        x = self.rect.left + pad

        # заголовок «Алгоритм ИИ:»
        title = font_bold.render("Алгоритм ИИ:", True, COLOR_HEADING)
        surface.blit(title, (x, self._title_algo_y))
        self.radio.draw(surface, font)

        # заголовок «Настройки:»
        title2 = font_bold.render("Настройки:", True, COLOR_HEADING)
        surface.blit(title2, (x, self._title_settings_y))
        if not self.depth_slider.visible and not self.sims_slider.visible:
            note = font.render(
                "Перцептрон: без параметров", True, COLOR_MUTED,
            )
            surface.blit(note, (x, self._perceptron_note_y))
        self.depth_slider.draw(surface, font)
        self.sims_slider.draw(surface, font)

        # таблица
        self._draw_table(surface, font, font_bold, font_small)

        # кнопки
        self.btn_new_game.draw(surface, font)
        self.btn_ai_vs_ai.draw(surface, font)
        self.btn_reset.draw(surface, font)

    def _draw_table(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        font_bold: pygame.font.Font,
        font_small: pygame.font.Font,
    ) -> None:
        pad = 16
        x = self.rect.left + pad
        width = self.rect.width - 2 * pad

        title = font_bold.render("Статистика по ходам:", True, COLOR_HEADING)
        surface.blit(title, (x, self._table_top))

        head_y = self._table_top + 28
        row_h = 24
        col_widths = (0.42, 0.20, 0.20, 0.18)
        col_xs = []
        cur = x
        for frac in col_widths:
            col_xs.append(cur)
            cur += int(width * frac)

        headers = ("Алгоритм", "ср. мс", "посл. мс", "ходов")
        pygame.draw.rect(
            surface, COLOR_TABLE_HEAD,
            pygame.Rect(x, head_y, width, row_h),
        )
        for i, h in enumerate(headers):
            text = font_small.render(h, True, COLOR_HEADING)
            surface.blit(text, (col_xs[i] + 4, head_y + 4))

        # 4 строки — всегда показываем все 4 алгоритма
        for i, (key, label) in enumerate(ALGORITHM_OPTIONS):
            row_y = head_y + row_h * (i + 1)
            bg = COLOR_TABLE_ROW if i % 2 == 0 else COLOR_TABLE_ROW_ALT
            pygame.draw.rect(surface, bg, pygame.Rect(x, row_y, width, row_h))
            stats = self.session_stats.by_agent.get(label)
            if stats is None or stats.moves == 0:
                cells = (label, "—", "—", "0")
            else:
                cells = (
                    label,
                    f"{stats.avg_ms:.0f}",
                    f"{stats.last_ms:.0f}",
                    str(stats.moves),
                )
            for j, c in enumerate(cells):
                txt = font_small.render(c, True, COLOR_TEXT)
                surface.blit(txt, (col_xs[j] + 4, row_y + 4))

        pygame.draw.rect(
            surface, COLOR_BORDER,
            pygame.Rect(x, head_y, width, row_h * 5), width=1,
        )
