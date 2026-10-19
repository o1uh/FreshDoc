from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Сплиты оценки (по отдельности, без усреднения).
EVAL_SPLITS = ("val", "test_real", "test_synth", "gold")

#: Группы внутри сплита.
EVAL_GROUPS = ("all", "sig_changed", "sig_same")

#: Типы дрейфа, исключаемые из оценки type_pred (неизвестный тип).
UNKNOWN_DRIFT = "unknown"

#: Отсутствие дрейфа — не класс головы (R10: 8 классов без none).
NONE_DRIFT = "none"


def group_mask(sig_type: np.ndarray, group: str) -> np.ndarray:
    """Булева маска группы по sig_type (M0 видит sig_changed, слеп на sig_same)."""
    s = np.asarray(sig_type, dtype=str)
    if group == "all":
        return np.ones(len(s), dtype=bool)
    if group == "sig_changed":
        return s != "none"
    if group == "sig_same":
        return s == "none"
    raise ValueError(f"Неизвестная группа: {group}")


def pick_threshold_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Порог с максимальным F1. Кандидаты — уникальные скоры + 0.0/1.0.

    Тай-брейк: из равных F1 выбирается меньший порог (более чувствительный).
    Вырожденный случай (один класс в y_true) — 0.5.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = sorted(set([0.0, 1.0] + [float(v) for v in np.unique(y_score)]))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        y_pred = (y_score >= thr).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1 + 1e-12:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def binary_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict:
    """Precision/recall/F1 на фиксированном пороге."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def pr_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC (average precision). nan при одном классе."""
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score, dtype=float)))


def roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC. nan при одном классе."""
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
    except ValueError:
        return float("nan")


def brier_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(brier_score_loss(np.asarray(y_true, dtype=int),
                                  np.asarray(y_score, dtype=float)))


def ece_score(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error, 10 равномерных корзин (R11)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (y_score >= lo) & (y_score <= hi)
        else:
            m = (y_score >= lo) & (y_score < hi)
        n_bin = int(m.sum())
        if n_bin == 0:
            continue
        acc = float(y_true[m].mean())
        conf = float(y_score[m].mean())
        ece += abs(acc - conf) * n_bin / len(y_true)
    return float(ece)


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target: float = 0.8
) -> float:
    """Максимальный recall среди порогов с precision >= target (R11, E6)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # precision_recall_curve возвращает precision/recall длиной n_thr+1;
    # берём точки, где precision >= target, максимум recall.
    feasible = recall[precision >= target]
    if len(feasible) == 0:
        return 0.0
    return float(feasible.max())


def bootstrap_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Бутстрэп-интервал PR-AUC (2.5/97.5 перцентили). Скип одно-классовых ресэмплов."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return (float("nan"), float("nan"))
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            vals.append(float(average_precision_score(y_true[idx], y_score[idx])))
        except ValueError:
            continue
    if len(vals) < 100:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Все бинарные метрики R11 для одного среза (сплит x группа)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = int(len(y_true))
    n_pos = int(y_true.sum()) if n else 0
    prevalence = float(y_true.mean()) if n else float("nan")
    pr_auc = pr_auc_safe(y_true, y_score)
    lo, hi = bootstrap_pr_auc(y_true, y_score, n_boot=n_boot, seed=seed)
    out = {
        "n": n,
        "n_pos": n_pos,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lo": lo,
        "pr_auc_hi": hi,
        "roc_auc": roc_auc_safe(y_true, y_score),
        "brier": brier_safe(y_true, y_score),
        "ece": ece_score(y_true, y_score),
        "recall_at_p80": recall_at_precision(y_true, y_score, target=0.8),
    }
    out.update(binary_at_threshold(y_true, y_score, threshold))
    return out


def evaluate_drift(
    drift_true: np.ndarray, drift_pred: np.ndarray,
    y_stale: np.ndarray | None = None,
) -> dict:
    """Macro-F1 по drift_type только на y_stale=1 с известным типом (R11).

    Подвыборка: строки с истинным рассинхроном (``y_stale == 1``,
    если передано) и известным типом (``drift_type`` не ``unknown``
    и не ``none``). Классы ``none``/``unknown`` из усреднения исключаются:
    предсказанные ``none``/``unknown`` засчитываются как обычная ошибка
    (штраф recall класса), но отдельного класса не образуют.

    ``y_stale`` опционален для обратной совместимости: без него
    фильтруется только по известному типу.
    """
    dt = np.asarray(drift_true, dtype=str)
    dp = np.asarray(drift_pred, dtype=str)
    m = (dt != UNKNOWN_DRIFT) & (dt != NONE_DRIFT)
    if y_stale is not None:
        ys = np.asarray(y_stale, dtype=int)
        if len(ys) != len(dt):
            raise ValueError(f"y_stale len {len(ys)} != drift len {len(dt)}")
        m = m & (ys == 1)
    dt, dp = dt[m], dp[m]
    n = int(len(dt))
    if n == 0:
        return {"n_drift": 0, "drift_acc": float("nan"),
                "drift_macro_f1": float("nan"), "drift_per_class_f1": {}}
    labels = sorted((set(dt.tolist()) | set(dp.tolist())) - {UNKNOWN_DRIFT, NONE_DRIFT})
    if not labels:
        return {"n_drift": n, "drift_acc": float((dt == dp).mean()),
                "drift_macro_f1": float("nan"), "drift_per_class_f1": {}}
    macro = float(f1_score(dt, dp, labels=labels, average="macro", zero_division=0))
    per = f1_score(dt, dp, labels=labels, average=None, zero_division=0)
    return {
        "n_drift": n,
        "drift_acc": float((dt == dp).mean()),
        "drift_macro_f1": macro,
        "drift_per_class_f1": {lab: float(v) for lab, v in zip(labels, per)},
    }


def pr_curve_points(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict:
    """Точки PR-кривой для графика (precision, recall, thresholds)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"precision": [], "recall": [], "thresholds": []}
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return {
        "precision": [float(v) for v in precision],
        "recall": [float(v) for v in recall],
        "thresholds": [float(v) for v in thresholds],
    }
