"""FD-20: бейзлайны M0 (правило) и M1 (градиентный бустинг), признаки R10.

Инвариант данных: признаки строятся ТОЛЬКО из содержимого кода и
документации (``sig_old``, ``diff_code``, ``doc_old``, ``code_old`` /
``code_new`` — нормализованный код без docstring, ``sig_type`` —
пересчитанный тип по сигнатурам). Колонки ``project``, ``source``,
``label_*``, ``drift_type``, ``_leak_*`` и ``_aux_*`` в признаки НЕ входят
и в этом модуле даже не читаются (см. ``ALLOWED_FEATURE_COLS``).

Признаки R10:
- изменение: diff_added/diff_removed, n_code_lines, n_params_old,
  n_sig_changes, one-hot sig_type (7 категорий);
- документация: doc_len_chars/lines, has_args/returns/raises,
  n_params_mentioned, has_default_word;
- связь: link_removed_mentioned, link_default_mentioned.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.data.drift import extract_signature
from src.data.normalizer import parse_function_ast
from src.data.docsections import (
    _RE_ARGS,
    _RE_RAISES,
    _RE_RETURNS,
    doc_section_flags,
)

#: Колонки датасета, которые разрешено читать при построении признаков.
#: Всё остальное (project, source, label_*, drift_type, split, _leak_*, _aux_*)
#: запрещено и здесь не используется.
ALLOWED_FEATURE_COLS = frozenset(
    {"sig_old", "diff_code", "doc_old", "code_old", "code_new", "sig_type"}
)

#: Категории sig_type (фиксированный порядок one-hot; behavior_changed и
#: unknown в sig_type не встречаются, но обрабатываются как все-нули).
SIG_TYPE_CATS = [
    "none",
    "param_renamed",
    "param_removed",
    "param_added",
    "default_changed",
    "return_changed",
    "exception_changed",
]

FEATURE_COLS = [
    "diff_added",
    "diff_removed",
    "n_code_lines",
    "n_params_old",
    "n_sig_changes",
    "sig_none",
    "sig_param_renamed",
    "sig_param_removed",
    "sig_param_added",
    "sig_default_changed",
    "sig_return_changed",
    "sig_exception_changed",
    "doc_len_chars",
    "doc_len_lines",
    "has_args",
    "has_returns",
    "has_raises",
    "n_params_mentioned",
    "has_default_word",
    "link_removed_mentioned",
    "link_default_mentioned",
]

# Маркеры разделов docstring — единый источник: src/data/docsections.py
# (Google-стиль, RST, NumPy-подчёркивания). Импортированы выше как
# _RE_ARGS / _RE_RETURNS / _RE_RAISES для совместимости.


def predict_m0(df: pd.DataFrame) -> pd.Series:
    """M0 (R10): p_stale = 1, если sig_type != none, иначе 0."""
    sig = df["sig_type"].astype(str)
    return pd.Series(np.where(sig != "none", 1.0, 0.0), index=df.index, dtype=float)


def count_diff_lines(diff_code: Any) -> tuple[int, int]:
    """Число добавленных/удаленных строк unified diff (без заголовков)."""
    added, removed = 0, 0
    if diff_code is None:
        return 0, 0
    text = diff_code if isinstance(diff_code, str) else str(diff_code)
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _base_name(param: str) -> str:
    return param.lstrip("*")


def _parse_sigs(
    code_old: Any, code_new: Any, sig_old: str = ""
) -> tuple[dict | None, dict | None]:
    """Разбор старой и новой сигнатур независимо друг от друга.

    Если ``code_old`` не разбирается, пробуем восстановить старую сигнатуру из
    ``sig_old`` (строка ``def ...:`` + заглушка тела): признаки старой сигнатуры
    (``n_params_old``, ``n_params_mentioned``) сохраняются, даже когда новая
    версия битая. Возвращает (sold, snew), где каждая сторона — None при неудаче.
    """

    def _parse_old() -> Any:
        try:
            fn = (parse_function_ast(code_old)
                  if isinstance(code_old, str) else None)
            if fn is None and isinstance(sig_old, str) and sig_old.strip():
                fn = parse_function_ast(sig_old.strip() + "\n    pass")
            return fn
        except Exception:
            return None

    def _parse_new() -> Any:
        try:
            return (parse_function_ast(code_new)
                    if isinstance(code_new, str) else None)
        except Exception:
            return None

    def _sig(fn: Any) -> dict | None:
        if fn is None:
            return None
        try:
            return extract_signature(fn)
        except Exception:
            return None

    return _sig(_parse_old()), _sig(_parse_new())


def _word_in_doc(name: str, doc: str) -> bool:
    if not name or not doc:
        return False
    return re.search(r"\b" + re.escape(name) + r"\b", doc) is not None


def _doc_section_flags(doc: str) -> tuple[int, int, int]:
    return doc_section_flags(doc if isinstance(doc, str) else "")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки R10 для всех строк. На выходе — FEATURE_COLS, без NaN."""
    unknown_cols = set(df.columns) & {"project", "source", "drift_type",
                                      "label_stale", "label_source", "split",
                                      "weight", "pair_id"}
    _ = unknown_cols  # запрещено читать перечисленное — проверка тестами.
    # Колонки заранее в списки: без упаковки строк в pd.Series на каждой итерации.
    n = len(df)
    diffs = (df["diff_code"].tolist() if "diff_code" in df.columns else [""] * n)
    docs = (df["doc_old"].tolist() if "doc_old" in df.columns else [""] * n)
    olds = (df["code_old"].tolist() if "code_old" in df.columns else [""] * n)
    news = (df["code_new"].tolist() if "code_new" in df.columns else [""] * n)
    sigs = (df["sig_old"].tolist() if "sig_old" in df.columns else [""] * n)
    sig_types = (df["sig_type"].tolist() if "sig_type" in df.columns else ["none"] * n)
    rows: list[dict[str, float]] = []
    for i in range(n):
        diff_code = diffs[i]
        doc = docs[i]
        if doc is None or (not isinstance(doc, str) and pd.isna(doc)):
            doc = ""
        doc = doc if isinstance(doc, str) else str(doc)
        code_old, code_new = olds[i], news[i]
        sig_old = sigs[i] if isinstance(sigs[i], str) else ""
        sig_type = str(sig_types[i])

        added, removed = count_diff_lines(diff_code)

        if isinstance(code_old, str) and code_old.strip():
            n_code_lines = len(code_old.splitlines())
        else:
            n_code_lines = 0

        sold, snew = _parse_sigs(code_old, code_new, sig_old)
        old_names: list[str] = [p[0] for p in sold["params"]] if sold else []
        new_names: list[str] = [p[0] for p in snew["params"]] if snew else []
        n_params_old = len(old_names)
        removed_names: list[str] = []
        changed_defaults: list[str] = []
        defaults_old: dict[str, str] = {}
        defaults_new: dict[str, str] = {}
        if sold is not None and snew is not None:
            removed_names = [nm for nm in old_names if nm not in new_names]
            added_names = [nm for nm in new_names if nm not in old_names]
            defaults_old = dict(sold["params"])
            defaults_new = dict(snew["params"])
            changed_defaults = [
                nm for nm in old_names
                if nm in defaults_new and defaults_old[nm] != defaults_new[nm]
            ]
            n_sig_changes = len(removed_names) + len(added_names)
            n_sig_changes += len(changed_defaults)
            if sold["returns"] != snew["returns"]:
                n_sig_changes += 1
            n_sig_changes += len(set(sold["raises"]) ^ set(snew["raises"]))
        else:
            # Частичный разбор: одна сторона известна, вторая нет.
            # Резерв по sig_type: известно, менялась ли сигнатура.
            n_sig_changes = 0 if sig_type == "none" else 1

        feat: dict[str, float] = {
            "diff_added": float(added),
            "diff_removed": float(removed),
            "n_code_lines": float(n_code_lines),
            "n_params_old": float(n_params_old),
            "n_sig_changes": float(n_sig_changes),
        }
        for cat in SIG_TYPE_CATS:
            feat["sig_" + cat] = 1.0 if sig_type == cat else 0.0

        feat["doc_len_chars"] = float(len(doc))
        feat["doc_len_lines"] = float(doc.count("\n") + 1) if doc else 0.0
        ha, hr, hx = _doc_section_flags(doc)
        feat["has_args"] = float(ha)
        feat["has_returns"] = float(hr)
        feat["has_raises"] = float(hx)

        mention_names = [
            _base_name(n) for n in old_names
            if _base_name(n) not in ("self", "cls", "")
        ]
        feat["n_params_mentioned"] = float(
            sum(1 for n in mention_names if _word_in_doc(n, doc))
        )
        low = doc.lower()
        has_default_word = int(
            "default" in low or "по умолчанию" in low
        )
        feat["has_default_word"] = float(has_default_word)

        feat["link_removed_mentioned"] = float(
            1 if any(_word_in_doc(_base_name(n), doc) for n in removed_names) else 0
        )
        link_default = 0
        for n in changed_defaults:
            base = _base_name(n)
            if not _word_in_doc(base, doc):
                continue
            old_d, new_d = defaults_old.get(n, ""), defaults_new.get(n, "")
            substr_hit = any(
                d and (d in doc or d.strip("'\"") in doc)
                for d in (old_d, new_d)
                if d.strip("'\"")
            )
            if has_default_word or substr_hit:
                link_default = 1
                break
        feat["link_default_mentioned"] = float(link_default)

        rows.append(feat)

    out = pd.DataFrame(rows, index=df.index)
    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0
    out = out[FEATURE_COLS].astype(float).fillna(0.0)
    return out


def train_m1(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
):
    """HistGradientBoostingClassifier для M1 (веса строк — sample_weight)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        random_state=seed,
    )
    clf.fit(X[FEATURE_COLS], np.asarray(y, dtype=int), sample_weight=sample_weight)
    return clf


def predict_proba_m1(clf, X: pd.DataFrame) -> np.ndarray:
    """Вероятности положительного класса M1."""
    return np.asarray(clf.predict_proba(X[FEATURE_COLS])[:, 1], dtype=float)
