"""FD-39 фаза 1: pairing Go-функций между коммитами (чистые функции + тесты).

Кандидат: символ есть в обоих коммитах и изменился code или doc.
code_old/code_new — текст объявления БЕЗ godoc; sig_old — заголовок,
doc_old/doc_new — godoc. diff — сырой unified diff code.
"""

from __future__ import annotations

import difflib
import random
from typing import Any


def pair_maps(
    old: dict[tuple[str, str], dict[str, str]],
    new: dict[tuple[str, str], dict[str, str]],
    commit_old: str,
    commit_new: str,
) -> list[dict[str, Any]]:
    """Сопоставить карты {(file, symbol): {sig, doc, code}} двух коммитов.

    Дифф строится ТОЛЬКО по коду без godoc (R1): правка комментария видна
    судье лишь через doc_old/doc_new, но не через diff_code. Иначе судья
    видел бы сам факт правки доки в диффе (утечка прокси doc_changed).
    """
    cands: list[dict[str, Any]] = []
    for key in sorted(set(old) & set(new)):
        o, n = old[key], new[key]
        if (o.get("code") or "") == (n.get("code") or "") \
                and (o.get("doc") or "") == (n.get("doc") or ""):
            continue
        file, symbol = key
        diff = "\n".join(
            difflib.unified_diff(
                (o.get("code") or "").splitlines(),
                (n.get("code") or "").splitlines(),
                fromfile="a/go",
                tofile="b/go",
                lineterm="",
            )
        )
        cands.append({
            "commit_old": commit_old,
            "commit_new": commit_new,
            "file": file,
            "symbol": symbol,
            "sig_old": o.get("sig") or "",
            "code_old": o.get("code") or "",
            "code_new": n.get("code") or "",
            "diff_code": diff,
            "doc_old": o.get("doc") or "",
            "doc_new": n.get("doc") or "",
            "doc_changed": (o.get("doc") or "") != (n.get("doc") or ""),
            "code_changed": (o.get("code") or "") != (n.get("code") or ""),
            "sig_changed": (o.get("sig") or "") != (n.get("sig") or ""),
        })
    return cands


def with_doc(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Только кандидаты с непустым doc_old (иначе разметчику нечего оценивать)."""
    return [c for c in cands if (c.get("doc_old") or "").strip()]


def judgeable(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Пилот-пул: непустые doc_old И diff (пустой дифф = тривиальный C)."""
    return [c for c in with_doc(cands) if (c.get("diff_code") or "").strip()]


def select_pilot(
    cands: list[dict[str, Any]], n: int = 30, seed: int = 42
) -> list[dict[str, Any]]:
    """Стратифицированный пилот: половина с правкой дока, половина без.

    Best-effort: если меньшинства не хватает, добираем из остатка
    (пилот полный всегда, страта — по возможности).
    """
    rng = random.Random(seed)
    with_doc = [c for c in cands if c["doc_changed"]]
    without = [c for c in cands if not c["doc_changed"]]
    rng.shuffle(with_doc)
    rng.shuffle(without)
    half = n // 2
    picked = with_doc[:half] + without[:half]
    rest = with_doc[half:] + without[half:]
    rng.shuffle(rest)
    picked += rest[: max(0, n - len(picked))]
    rng.shuffle(picked)
    return picked


def stability(old_keys: set, new_keys: set) -> dict[str, float]:
    """Доля символов, переживших коммит (стабильность имён)."""
    if not old_keys:
        return {"kept": 0.0, "n_old": 0, "n_new": len(new_keys)}
    kept = len(set(old_keys) & set(new_keys))
    return {"kept": kept / len(old_keys), "n_old": len(old_keys), "n_new": len(new_keys)}
