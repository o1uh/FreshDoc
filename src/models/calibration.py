"""FD-25: температурная калибровка вероятностей M2 (E5).

Инвариант: температура T подбирается ТОЛЬКО на реальных строках val
(минимизация лог-потерь), применяется ко всем сплитам монотонно:
p_cal = sigmoid(logit(p) / T).

PR-AUC от монотонного масштабирования не меняется; меняются Brier и ECE.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

EPS = 1e-6
T_BOUNDS = (0.05, 10.0)


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    """Логит с клиппингом (защита от p=0/1, как у M0)."""
    pc = _clip(p)
    return np.log(pc / (1.0 - pc))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Численно устойчивая сигмоида через scipy.special.expit.

    Наивная формула 1/(1+exp(-z)) при z < -709 даёт RuntimeWarning
    (overflow в exp); expit обрабатывает края без предупреждений.
    """
    return np.asarray(expit(np.asarray(z, dtype=float)), dtype=float)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    """Применить температурное масштабирование: sigma(logit(p)/T)."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T должна быть > 0, получено {T}")
    return sigmoid(logit(p) / float(T))


def nll_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Средний лог-лосс (NLL) с клиппингом."""
    y = np.asarray(y_true, dtype=int)
    pc = _clip(p_pred)
    if y.shape != pc.shape:
        raise ValueError(f"Размерности не совпадают: y {y.shape} vs p {pc.shape}")
    if y.size == 0:
        return float("nan")
    # ручная формула, чтобы не зависеть от версий sklearn (labels)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1.0 - pc)))


def fit_temperature(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    bounds: tuple[float, float] = T_BOUNDS,
) -> tuple[float, dict]:
    """Подобрать T минимизацией NLL на val (bounded, Brent).

    Возвращает (T, info) где info содержит nll_before/nll_after.
    """
    y = np.asarray(y_true, dtype=int)
    p = _clip(np.asarray(p_pred, dtype=float))
    if len(np.unique(y)) < 2:
        raise ValueError("fit_temperature требует оба класса в y_true (иначе T неопределена)")
    nll_before = float(nll_score(y, p))
    # T=1 должно давать тот же NLL (проверка консистентности клиппинга)
    def _obj(logT: float) -> float:
        T = float(np.exp(logT))
        pc = _clip(apply_temperature(p, T))
        return float(nll_score(y, pc))
    # оптимизация в лог-пространстве для устойчивости на границах
    lo, hi = float(np.log(bounds[0])), float(np.log(bounds[1]))
    res = minimize_scalar(_obj, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-5})
    T_best = float(np.exp(res.x))
    nll_after = float(nll_score(y, apply_temperature(p, T_best)))
    info = {
        "nll_before": nll_before,
        "nll_after": nll_after,
        "success": bool(res.success),
        "n_val": int(len(y)),
    }
    return T_best, info


def reliability_table(
    y_true: np.ndarray,
    p_score: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Данные для диаграммы надёжности: по 10 равномерным корзинам."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(p_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            m = (s >= lo) & (s <= hi)
        else:
            m = (s >= lo) & (s < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": i, "lo": lo, "hi": hi, "n": 0,
                         "acc": float("nan"), "conf": float("nan")})
        else:
            rows.append({"bin": i, "lo": lo, "hi": hi, "n": n,
                         "acc": float(y[m].mean()), "conf": float(s[m].mean())})
    return rows


def thresholds_table(
    y_true: np.ndarray,
    p_score: np.ndarray,
    thresholds: np.ndarray | list[float] | None = None,
) -> list[dict]:
    """Таблица порог -> precision/recall/F1 (E6)."""
    from src.models.metrics import binary_at_threshold

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(p_score, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    rows: list[dict] = []
    for thr in thresholds:
        r = binary_at_threshold(y, s, float(thr))
        rows.append(r)
    return rows


def pick_working_threshold(
    y_true: np.ndarray,
    p_score: np.ndarray,
    min_precision: float = 0.8,
) -> tuple[float, dict]:
    """Рабочий порог: precision >= min_precision, максимум F1.

    Тай-брейк: больший recall, затем меньший порог (более чувствительный).
    Если ни один порог не даёт precision >= min_precision, возвращается
    порог с максимальной precision (с документированием в caller).

    Возвращаемый порог — СЕРЕДИНА интервала между выбранным точным скором
    и предыдущим уникальным скором ((s_k + s_{k-1}) / 2), а не сам скор.
    Причина: предикат s >= threshold на точном скоре s_k хрупок — расхождение
    в младших битах мантиссы при переносе между платформами/парсерами
    (YAML/CSV) превращает граничный TP в FN. Середина пустого интервала
    даёт ту же классификацию на выборке подбора и запас ~половины зазора.
    """
    from src.models.metrics import binary_at_threshold

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(p_score, dtype=float)
    uniq = sorted(float(v) for v in np.unique(s))
    cands = sorted(set([0.0, 1.0] + uniq))
    scored: list[dict] = []
    for thr in cands:
        r = binary_at_threshold(y, s, float(thr))
        scored.append(r)

    def _to_midpoint(thr_exact: float) -> tuple[float, dict]:
        below = [v for v in uniq if v < float(thr_exact)]
        if not below:
            return float(thr_exact), dict(binary_at_threshold(y, s, float(thr_exact)))
        thr_mid = float((float(thr_exact) + max(below)) / 2.0)
        return thr_mid, dict(binary_at_threshold(y, s, thr_mid))

    feasible = [r for r in scored if r["precision"] >= min_precision]
    if feasible:
        # max F1, затем max recall, затем min threshold
        feasible.sort(key=lambda r: (-r["f1"], -r["recall"], r["threshold"]))
        best = feasible[0]
        return _to_midpoint(float(best["threshold"]))
    # fallback: максимальная precision
    scored.sort(key=lambda r: (-r["precision"], -r["f1"], r["threshold"]))
    best = scored[0]
    return _to_midpoint(float(best["threshold"]))
