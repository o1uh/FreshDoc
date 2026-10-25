# -*- coding: utf-8 -*-
"""FD-33: общая muted-гамма для PR-кривых и калибровочных графиков (группа D).

Палитра (без смены содержания графиков — только цвета/шрифты):
  фон #FAF9F7, оси #FFFFFF, текст #2B2B2B, приглушённый #6B7280,
  кривые ТОЛЬКО из #C17C60 / #7D9B8A / #6B7C9B / #A8A29E
  (+ тёмный #2B2B2B для бейзлайна/идеальной диагонали при нужде),
  сетка #E5E0D8 светлая пунктирная. DPI=200, легенда/оси >= 10.

Фиксированный маппинг: один и тот же слот/модель — один и тот же цвет
во всех сетках (e1, fd31, external, e4_e5_e6).
"""

PAPER = "#FAF9F7"
AXES_BG = "#FFFFFF"
INK = "#2B2B2B"
MUTED = "#6B7280"
TERRA = "#C17C60"
SAGE = "#7D9B8A"
SLATE = "#6B7C9B"
STONE = "#A8A29E"
GRID = "#E5E0D8"

DPI = 200
LEGEND_FONTSIZE = 10
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 12
TICK_FONTSIZE = 10

# Точный маппинг модель/слот -> цвет (порядок и состав стабильны).
MODEL_COLORS = {
    "M0": STONE,       # слабый бейзлайн — серый
    "M1": SLATE,       # бустинг — slate-blue
    "M2": TERRA,       # основная модель (slot4_full / seed42) — терракота
    "M3": SAGE,        # LLM-судья — шалфей
    "M3max": INK,      # max-effort вариант — тёмный
    # Сиды FD-31 (slot4_full, тот же слот): seed42 == M2, остальные — из палитры.
    "seed42": TERRA,
    "seed43": SAGE,
    "seed44": INK,
    "M2_seed42": TERRA,
    "M2_seed43": SAGE,
    "M2_seed44": INK,
    "M2_s42": TERRA,
    "M2_s43": SAGE,
    "M2_s44": INK,
    "slot4_full": TERRA,
}

# Запасной цикл строго внутри палитры (для неизвестных имён).
_FALLBACK_CYCLE = (TERRA, SAGE, SLATE, STONE)


def color_for_model(name: str) -> str:
    """Фиксированный цвет модели/слота. Неизвестные имена — детерминированно
    из палитры (без ярких цветов tab10)."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    low = str(name).lower()
    if low in ("m0",) or low.startswith(("m0_", "m0-", "m0 ")):
        return STONE
    if low in ("m1",) or low.startswith(("m1_", "m1-", "m1 ")):
        return SLATE
    if "seed44" in low or low.endswith(("s44", "_44", "-44")):
        return INK
    if "seed43" in low or low.endswith(("s43", "_43", "-43")):
        return SAGE
    if "seed42" in low or low.endswith(("s42", "_42", "-42")):
        return TERRA
    if "m3max" in low or "m3_max" in low:
        return INK
    if low.startswith("m3"):
        return SAGE
    if low.startswith("m2") or low.startswith("slot4"):
        return TERRA
    if low.startswith("m1"):
        return SLATE
    if low.startswith("m0"):
        return STONE
    idx = abs(hash(low)) % len(_FALLBACK_CYCLE)
    return _FALLBACK_CYCLE[idx]


def apply_muted_style(plt) -> None:
    """Единые rcParams muted-гаммы (вызывать после matplotlib.use/import)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.facecolor": PAPER,
        "figure.dpi": DPI,
        "savefig.facecolor": PAPER,
        "savefig.dpi": DPI,
        "axes.facecolor": AXES_BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "grid.color": GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
    })
