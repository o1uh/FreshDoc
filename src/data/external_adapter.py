"""FD-39: адаптер внешних наборов (Java CUP/HebCup, JIT Panthaplackel, Python-2, Go-silver).

Принципы (скриншот фазы 0):
- сырой текст без нормализации R2 (Java/Go нельзя разобрать Python-AST);
- drift_type="unknown" для всех внешних пар, sig_type="unknown";
- без фильтров R1 (длина/docstring не режутся);
- метки только из коробки источника (label из поля набора);
- вес 0.0 — внешние пары не для обучения, только тест;
- split — отдельный тег источника (ext_hebcup, ext_jit_<тип>, ext_python2, ext_go_silver).

Пары годятся для инференса M2 (predict_m2.py читает sig_old/diff_code/doc_old)
и для оценки scripts/evaluate.py (нужны pair_id/label_stale/split/sig_type/drift_type/source).
Группы sig_changed/sig_same для external игнорировать: sig_type всегда unknown
(попадает в sig_changed по маске != none) — смотреть только группу all.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any

PAIR_COLUMNS = [
    "pair_id", "source", "project", "commit_sha", "symbol", "function_hash",
    "sig_old", "code_old", "code_new", "diff_code", "doc_old",
    "label_stale", "drift_type", "sig_type", "label_source", "weight",
    "n_src_lines", "split",
    "_leak_doc_new", "_leak_diff_docstring", "_leak_commit_message",
    "_aux_src_old",
]

_METHOD_NAME_RE = re.compile(r"(\w+)\s*\(")


def raw_unified_diff(code_old: str, code_new: str) -> str:
    """Сырой unified diff без нормализации (для Java/Go/Python-2)."""
    return "\n".join(
        difflib.unified_diff(
            (code_old or "").splitlines(),
            (code_new or "").splitlines(),
            fromfile="a/code.py",
            tofile="b/code.py",
            lineterm="",
        )
    )


def java_sig_heuristic(src_method: str) -> str:
    """Первая строка с '(' (обычно сигнатура метода), иначе первая непустая."""
    for line in (src_method or "").splitlines():
        if "(" in line:
            return line.strip()[:300]
    for line in (src_method or "").splitlines():
        if line.strip():
            return line.strip()[:300]
    return ""


def extract_method_name(sig_line: str) -> str:
    """Последнее слово перед '(' — имя метода (эвристика для symbol)."""
    found = _METHOD_NAME_RE.findall(sig_line or "")
    return found[-1] if found else "unknown"


def make_pair_id(source: str, project: str, symbol: str, commit: str, code_old: str, code_new: str = "") -> str:
    key = f"{source}:{project}:{symbol}:{commit}:{code_old}:{code_new}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def make_function_hash(code_old: str) -> str:
    return hashlib.sha256((code_old or "").encode("utf-8")).hexdigest()[:16]


def generic_pair(
    *,
    source: str,
    project: str,
    commit_sha: str,
    symbol: str,
    code_old: str,
    code_new: str,
    doc_old: str,
    label_stale: int,
    label_source: str,
    split: str,
    doc_new: str = "",
) -> dict[str, Any]:
    """Собрать строку схемы R1 из сырых текстов (без фильтров R1)."""
    code_old = code_old or ""
    code_new = code_new or ""
    doc_old = doc_old or ""
    sig_old = java_sig_heuristic(code_old)
    diff_code = raw_unified_diff(code_old, code_new)
    n_src_lines = len(code_old.splitlines()) + len(doc_old.splitlines())
    return {
        "pair_id": make_pair_id(source, project, symbol, commit_sha, code_old, code_new),
        "source": source,
        "project": str(project),
        "commit_sha": str(commit_sha),
        "symbol": str(symbol),
        "function_hash": make_function_hash(code_old),
        "sig_old": sig_old,
        "code_old": code_old,
        "code_new": code_new,
        "diff_code": diff_code,
        "doc_old": doc_old,
        "label_stale": int(label_stale),
        "drift_type": "unknown",
        "sig_type": "unknown",
        "label_source": label_source,
        "weight": 0.0,
        "n_src_lines": int(n_src_lines),
        "split": split,
        "_leak_doc_new": doc_new or "",
        "_leak_diff_docstring": "",
        "_leak_commit_message": "",
        "_aux_src_old": code_old,
    }


def hebcup_record_to_pair(rec: dict[str, Any]) -> dict[str, Any]:
    """Запись HebCup test/train/valid_clean.jsonl -> пара (все label_stale=1).

    HebCup = очищенный CUP: каждая строка — co-change (код и javadoc правили
    вместе), поэтому все строки положительные. Отрицательных в коробке нет —
    PR-AUC на чистом HebCup не определён; HebCup служит проверкой recall/F1
    на Java-позитивах, а двухклассовая оценка — по JIT.
    """
    code_old = rec.get("src_method") or ""
    code_new = rec.get("dst_method") or ""
    doc_old = rec.get("src_javadoc") or rec.get("src_desc") or ""
    doc_new = rec.get("dst_javadoc") or rec.get("dst_desc") or ""
    sig = java_sig_heuristic(code_old)
    name = extract_method_name(sig)
    symbol = f"{name}#{rec.get('sample_id', '')}"
    return generic_pair(
        source="ext_hebcup",
        project=rec.get("full_name") or "unknown",
        commit_sha=rec.get("commit_id") or "",
        symbol=symbol,
        code_old=code_old,
        code_new=code_new,
        doc_old=doc_old,
        label_stale=1,
        label_source="hebcup_java_coupled",
        split="ext_hebcup",
        doc_new=doc_new,
    )


def jit_record_to_pair(rec: dict[str, Any], split: str = "ext_jit") -> dict[str, Any]:
    """Запись Panthaplackel JIT (Return/Param/Summary train/valid/test.json).

    Метка из коробки: label 1 = коммент стал inconsistent после изменения.
    id вида 'Owner_repo-PR-Num-Type-N' -> project = часть до '-<num>-<type>-<n>'.
    """
    label = int(str(rec.get("label", 0)))
    rid = str(rec.get("id", ""))
    # id вида 'Owner_repo-PR-Num-Type-N': отрезаем последние 3 сегмента
    # ('-<num>-<type>-<n>'), остаток — project (в owner могут быть дефисы).
    project = rid.rsplit("-", 3)[0] if rid.count("-") >= 3 else (rid or "unknown")
    ctype = rec.get("comment_type") or "Unknown"
    symbol = f"{ctype}:{rid}"
    return generic_pair(
        source="ext_jit",
        project=project,
        commit_sha="",
        symbol=symbol,
        code_old=rec.get("old_code_raw") or "",
        code_new=rec.get("new_code_raw") or "",
        doc_old=rec.get("old_comment_raw") or "",
        label_stale=label,
        label_source=f"jit_{str(ctype).lower()}",
        split=split,
        doc_new=rec.get("new_comment_raw") or "",
    )


_COCC_SEP_RE = re.compile(r"^\s*-{5,}\s*$")
_COCC_TRAIL_RE = re.compile(r"={5,}\s*$")


def split_cocc_blocks(text: str) -> tuple[str, str]:
    """Разбить N.py на (old, new) по строке-разделителю из дефисов.

    Формат CoCC python_ccset_raw: старый блок, строка из >=5 дефисов
    (отступы/число дефисов произвольны), новый блок, хвост из '='.
    """
    lines = (text or "").splitlines()
    sep = next((i for i, ln in enumerate(lines) if _COCC_SEP_RE.match(ln)), None)
    if sep is None:
        return (text or "").strip(), ""
    old = "\n".join(lines[:sep]).strip()
    new = "\n".join(lines[sep + 1:]).strip()
    new = _COCC_TRAIL_RE.sub("", new).strip()
    return old, new


def _cocc_comment(block: str) -> str:
    """Строки комментария ('#' после отступа) блока."""
    return "\n".join(
        ln for ln in block.splitlines() if ln.lstrip().startswith("#"))


def _cocc_code(block: str) -> str:
    """Строки кода блока (без комментариев)."""
    return "\n".join(
        ln for ln in block.splitlines() if not ln.lstrip().startswith("#")).strip()


def cocc_code_has_docstring_block(code_old: str, code_new: str) -> bool:
    """Есть ли docstring-блок (тройные кавычки) в коде CoCC-пары.

    Такие блоки НЕ вырезаем из code_old/code_new (сырой текст без
    нормализации), только считаем для манифеста сборки (см.
    scripts/build_external.py: inputs["cocc_docstring_blocks"]).
    """
    return '"""' in (code_old or "") or '"""' in (code_new or "")


def cocc_record_to_pair(name: str, text: str) -> dict[str, Any] | None:
    """Файл CoCC python_ccset_raw (N.py) -> пара.

    Метка из коробки по правилу статьи (раздел 2): комментарии до/после
    различаются -> 1 (outdated), совпадают -> 0. Возвращает None, если нет
    ни комментария, ни кода (мусорный файл).
    """
    old_block, new_block = split_cocc_blocks(text)
    doc_old = _cocc_comment(old_block)
    doc_new = _cocc_comment(new_block)
    code_old, code_new = _cocc_code(old_block), _cocc_code(new_block)
    if not doc_old and not code_old:
        return None
    label = int(doc_old.strip() != doc_new.strip())
    return generic_pair(
        source="ext_python2",
        project="cocc_python",
        commit_sha="",
        symbol=f"cocc#{name}",
        code_old=code_old or old_block,
        code_new=code_new or new_block,
        doc_old=doc_old,
        label_stale=label,
        label_source="cocc_python_block",
        split="ext_python2",
        doc_new=doc_new,
    )


def go_record_to_pair(row: dict[str, Any], proxy_label: int) -> dict[str, Any]:
    """Строка Go silver-набора (make_go_silver.py) -> пара.

    Метка — silver-прокси из sidecar (doc_changed: код+док правили вместе → 1,
    иначе 0; label_source=go_silver_proxy). После полной разметки двумя судьями
    метки заменяются консенсусом (label_source=go_silver_consensus) —
    см. scripts/apply_go_consensus.py (план).
    """
    return generic_pair(
        source="ext_go_silver",
        project=str(row.get("repo") or "go"),
        commit_sha=str(row.get("commit_new") or ""),
        symbol=str(row.get("symbol") or "unknown"),
        code_old=row.get("code_old") or "",
        code_new=row.get("code_new") or "",
        doc_old=row.get("doc_old") or "",
        label_stale=int(proxy_label),
        label_source="go_silver_proxy",
        split="ext_go_silver",
        doc_new="",
    )
