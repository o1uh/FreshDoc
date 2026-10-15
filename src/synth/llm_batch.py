from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.normalizer import (
    extract_docstring_and_clean_node,
    extract_sig_line,
    parse_function_ast,
)
from src.synth.llm_validators import parse_batch_response, validate_p1, validate_p2, validate_p3

PILOT_N = 12
PILOT_SEED = 42
BATCH_SIZE = 6

P1_FILES = ("pilot_p1_batch1.prompt.txt", "pilot_p1_batch2.prompt.txt")
P2_FILES = ("pilot_p2_batch1.prompt.txt", "pilot_p2_batch2.prompt.txt")
MANIFEST_NAME = "pilot_manifest.csv"

_SINGLE_FUNC_RE = re.compile(r"<function>\s*\{function_source\}\s*</function>")


def sample_pilot_functions(df_pool: pd.DataFrame, n: int = PILOT_N, seed: int = PILOT_SEED) -> pd.DataFrame:
    if len(df_pool) < n:
        raise ValueError(f"В пуле {len(df_pool)} строк, нужно {n}")
    rng = random.Random(seed)
    by_project: dict[str, list[int]] = {}
    for idx, row in df_pool.iterrows():
        by_project.setdefault(str(row["project"]), []).append(idx)
    for idxs in by_project.values():
        rng.shuffle(idxs)
    projects = sorted(by_project)
    rng.shuffle(projects)
    picked: list[Any] = []
    while len(picked) < n:
        progressed = False
        for project in projects:
            if len(picked) >= n:
                break
            if by_project[project]:
                picked.append(by_project[project].pop(0))
                progressed = True
        if not progressed:
            break
    sample = df_pool.loc[picked].copy().reset_index(drop=True)
    sample["pilot_id"] = [f"P{i + 1:04d}" for i in range(len(sample))]
    return sample


def render_batch_task(single_template: str, count: int) -> str:
    if "{function_source}" not in single_template:
        raise ValueError("В одиночном промпте нет {function_source}")
    func_note = (
        f"The {count} functions are given below, each wrapped in "
        '<function id="..."> tags. Do the task for EACH function separately.'
    )
    text, subs = _SINGLE_FUNC_RE.subn(func_note, single_template)
    if subs != 1:
        raise ValueError("Блок <function>{function_source}</function> не найден ровно один раз")
    needle_single = "Output ONLY one valid JSON object."
    if needle_single not in text:
        raise ValueError("В одиночном промпте нет строки про один JSON object")
    text = text.replace(needle_single, "Output ONLY one valid JSON array.")
    needle_ret = "Return JSON with these keys:"
    if needle_ret not in text:
        raise ValueError("В одиночном промпте нет строки Return JSON with these keys:")
    text = text.replace(
        needle_ret,
        "Return ONE JSON array with one object per function (same order), inside a single "
        'code block, nothing else. Each object starts with {"id": "<same id>"} followed by these keys:',
    )
    text += '\nIf you cannot satisfy the constraints for a function, return {"id": "<id>", "skip": true} for it.'
    return text


def render_batch_file(task: str, rows: pd.DataFrame, prompts_dir: Path) -> str:
    single_name = {"p1": "p1_behavior.txt", "p2": "p2_neutral.txt", "p3": "p3_paraphrase.txt"}[task]
    single = (prompts_dir / single_name).read_text(encoding="utf-8")
    header_tpl = (prompts_dir / "chat_batch.txt").read_text(encoding="utf-8")
    header = header_tpl.format(COUNT=len(rows))
    task_text = render_batch_task(single, len(rows))
    blocks = "\n\n".join(f'<function id="{r["pilot_id"]}">\n{r["src"]}\n</function>' for _, r in rows.iterrows())
    return f"{header}\n\n{task_text}\n\n{blocks}\n"


def _symbol_of(src: str) -> str:
    fn = parse_function_ast(src)
    return fn.name if fn is not None else ""


def export_pilot_batches(
    df_pool: pd.DataFrame,
    out_dir: Path,
    prompts_dir: Path,
    seed: int = PILOT_SEED,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = sample_pilot_functions(df_pool, PILOT_N, seed)
    manifest = pd.DataFrame(
        {
            "pilot_id": sample["pilot_id"],
            "function_hash": sample["function_hash"],
            "project": sample["project"],
            "symbol": [_symbol_of(s) for s in sample["src"]],
        }
    )
    manifest_path = out_dir / MANIFEST_NAME
    manifest.to_csv(manifest_path, index=False)
    halves = [sample.iloc[:BATCH_SIZE], sample.iloc[BATCH_SIZE : 2 * BATCH_SIZE]]
    written: list[str] = []
    for fname, task in ((P1_FILES, "p1"), (P2_FILES, "p2")):
        for path_name, half in zip(fname, halves):
            content = render_batch_file(task, half, prompts_dir)
            (out_dir / path_name).write_text(content, encoding="utf-8")
            written.append(path_name)
    projects = sorted(sample["project"].unique().tolist())
    return {
        "manifest": str(manifest_path),
        "prompts": written,
        "n_functions": len(sample),
        "n_projects": len(projects),
        "projects": projects,
    }


def _pair_id(function_hash: str, drift_type: str, diff: str) -> str:
    return hashlib.sha1(f"{function_hash}:{drift_type}:{diff}".encode("utf-8")).hexdigest()[:16]


def build_synth_row(
    task: str,
    pilot_id: str,
    function_hash: str,
    project: str,
    symbol: str,
    src_old: str,
    parts: dict[str, str],
) -> dict[str, Any]:
    # R1: в выходе только 17 колонок схемы (без pilot_id/m_id); трассировка
    # к запросу — через pair_id + function_hash + манифест + responses/.
    # pilot_id принят для совместимости вызовов, в parquet не пишется.
    drift = "behavior_changed" if task == "p1" else "none"
    return {
        "pair_id": _pair_id(function_hash, drift, parts["diff"]),
        "source": "synth_llm",
        "project": project,
        "commit_sha": "",
        "symbol": symbol,
        "function_hash": function_hash,
        "sig_old": extract_sig_line(parts["code_old"]),
        "code_old": parts["code_old"],
        "code_new": parts["code_new"],
        "diff_code": parts["diff"],
        "doc_old": parts["doc_old"],
        "label_stale": 1 if task == "p1" else 0,
        "drift_type": drift,
        "sig_type": "none",
        "label_source": "llm_p1" if task == "p1" else "llm_p2",
        "weight": 1.0 if task == "p1" else 0.5,
        "n_src_lines": len(src_old.strip().splitlines()),
    }


def _task_of_filename(name: str) -> str | None:
    if "_p1_" in name:
        return "p1"
    if "_p2_" in name:
        return "p2"
    if "_p3_" in name or name.startswith("p3_"):
        return "p3"
    return None


def build_paraphrase_row(function_hash: str, doc_orig: str, doc_para: str, style: str) -> dict[str, Any]:
    return {
        "function_hash": function_hash,
        "doc_orig": doc_orig,
        "doc_para": doc_para,
        "style": style,
    }


def export_p3_batches(
    df_pool: pd.DataFrame,
    out_dir: Path,
    prompts_dir: Path,
    n: int = 60,
    batch_size: int = BATCH_SIZE,
    seed: int = PILOT_SEED,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = sample_pilot_functions(df_pool, n, seed).copy()
    sample["pilot_id"] = [f"Q{i + 1:04d}" for i in range(len(sample))]
    manifest = pd.DataFrame(
        {
            "pilot_id": sample["pilot_id"],
            "function_hash": sample["function_hash"],
            "project": sample["project"],
            "symbol": [_symbol_of(s) for s in sample["src"]],
        }
    )
    manifest_path = out_dir / "p3_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    written: list[str] = []
    for k in range(0, len(sample), batch_size):
        chunk = sample.iloc[k : k + batch_size]
        name = f"p3_batch{k // batch_size + 1}.prompt.txt"
        (out_dir / name).write_text(render_batch_file("p3", chunk, prompts_dir), encoding="utf-8")
        written.append(name)
    return {"manifest": str(manifest_path), "prompts": written, "n_functions": len(sample)}


def _doc_of_src(src: str) -> str:
    fn = parse_function_ast(src)
    if fn is None:
        return ""
    doc, _ = extract_docstring_and_clean_node(fn)
    return (doc or "").strip()


def import_p3_responses(pool_path: Path, p3_dir: Path, out_parquet: Path) -> dict[str, Any]:
    df_pool = pd.read_parquet(pool_path)
    pool_by_hash = {str(r["function_hash"]): str(r["src"]) for _, r in df_pool.iterrows()}
    manifest_path = p3_dir / "p3_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str) if manifest_path.exists() else None
    manifest_by_id = (
        {str(r["pilot_id"]): str(r["function_hash"]) for _, r in manifest.iterrows()} if manifest is not None else {}
    )
    stats: dict[str, Any] = {
        "files_total": 0,
        "files_missing_response": 0,
        "objects_total": 0,
        "accepted": 0,
        "skipped": 0,
        "reasons": {},
    }
    rows: list[dict[str, Any]] = []
    prompt_files = sorted(p3_dir.glob("*.prompt.txt"))
    stats["files_total"] = len(prompt_files)
    for prompt_path in prompt_files:
        response_path = prompt_path.with_suffix("").with_suffix(".response.txt")
        if not response_path.exists():
            stats["files_missing_response"] += 1
            continue
        objects, error = parse_batch_response(response_path.read_text(encoding="utf-8"))
        if error is not None:
            stats["reasons"]["bad_json"] = stats["reasons"].get("bad_json", 0) + 1
            continue
        for obj in objects:
            stats["objects_total"] += 1
            func_hash = manifest_by_id.get(str(obj.get("id")))
            if func_hash is None or func_hash not in pool_by_hash:
                stats["reasons"]["unknown_id"] = stats["reasons"].get("unknown_id", 0) + 1
                continue
            doc_orig = _doc_of_src(pool_by_hash[func_hash])
            ok, reason, parts = validate_p3(doc_orig, obj)
            if not ok:
                if reason == "skipped":
                    stats["skipped"] += 1
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                continue
            assert parts is not None
            rows.append(build_paraphrase_row(func_hash, doc_orig, parts["doc_para"], parts["style"]))
            stats["accepted"] += 1
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["function_hash", "doc_orig", "doc_para", "style"]).to_parquet(out_parquet, index=False)
    return stats


def import_pilot_responses(
    pool_path: Path,
    pilot_dir: Path,
    report_path: Path,
    out_parquet: Path,
) -> dict[str, Any]:
    from src.data.normalizer import parse_function_ast as _parse

    df_pool = pd.read_parquet(pool_path)
    pool_by_hash = {str(r["function_hash"]): (str(r["src"]), str(r["project"])) for _, r in df_pool.iterrows()}
    manifest_path = pilot_dir / MANIFEST_NAME
    manifest = pd.read_csv(manifest_path, dtype=str) if manifest_path.exists() else None
    manifest_by_id = (
        {str(r["pilot_id"]): str(r["function_hash"]) for _, r in manifest.iterrows()} if manifest is not None else {}
    )
    stats: dict[str, Any] = {
        "files_total": 0,
        "files_missing_response": 0,
        "files_unparseable": 0,
        "objects_total": 0,
        "accepted_p1": 0,
        "accepted_p2": 0,
        "skipped": 0,
        "unknown_id": 0,
        "reasons": {},
        "per_file": [],
    }
    accepted_rows: list[dict[str, Any]] = []
    prompt_files = sorted(pilot_dir.glob("*.prompt.txt"))
    stats["files_total"] = len(prompt_files)
    for prompt_path in prompt_files:
        task = _task_of_filename(prompt_path.name)
        response_path = prompt_path.with_suffix("").with_suffix(".response.txt")
        file_rec: dict[str, Any] = {"file": prompt_path.name, "task": task, "accepted": 0, "rejected": 0}
        if task is None:
            file_rec["error"] = "unknown_task_in_filename"
            stats["per_file"].append(file_rec)
            continue
        if not response_path.exists():
            stats["files_missing_response"] += 1
            file_rec["error"] = "missing_response"
            stats["per_file"].append(file_rec)
            continue
        objects, error = parse_batch_response(response_path.read_text(encoding="utf-8"))
        if error is not None:
            stats["files_unparseable"] += 1
            file_rec["error"] = error
            stats["per_file"].append(file_rec)
            continue
        validator = validate_p1 if task == "p1" else validate_p2
        for obj in objects:
            stats["objects_total"] += 1
            pid = obj.get("id")
            func_hash = manifest_by_id.get(str(pid))
            if func_hash is None or func_hash not in pool_by_hash:
                stats["unknown_id"] += 1
                stats["reasons"]["unknown_id"] = stats["reasons"].get("unknown_id", 0) + 1
                file_rec["rejected"] += 1
                continue
            src_old, project = pool_by_hash[func_hash]
            ok, reason, parts = validator(src_old, obj)
            if not ok:
                if reason == "skipped":
                    stats["skipped"] += 1
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                file_rec["rejected"] += 1
                continue
            assert parts is not None
            fn = _parse(src_old)
            symbol = fn.name if fn is not None else ""
            accepted_rows.append(build_synth_row(task, str(pid), func_hash, project, symbol, src_old, parts))
            stats[f"accepted_{task}"] += 1
            file_rec["accepted"] += 1
        stats["per_file"].append(file_rec)
    df_out = pd.DataFrame(accepted_rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if len(df_out):
        df_out.to_parquet(out_parquet, index=False)
    else:
        pd.DataFrame(
            columns=[
                "pair_id", "source", "project", "commit_sha", "symbol", "function_hash",
                "sig_old", "code_old", "code_new", "diff_code", "doc_old", "label_stale",
                "drift_type", "sig_type", "label_source", "weight", "n_src_lines",
            ]
        ).to_parquet(out_parquet, index=False)
    stats["accepted_total"] = stats["accepted_p1"] + stats["accepted_p2"]
    denom = stats["objects_total"]
    stats["accept_rate"] = round(stats["accepted_total"] / denom, 3) if denom else 0.0
    _write_pilot_report(report_path, stats)
    return stats


def _write_pilot_report(report_path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# FD-10: пробный прогон LLM-синтетики (P1/P2)",
        "",
        "- Генератор: Muse Spark 1.3 (effort xhigh). Дата экспорта: 12.10.2026.",
        f"- Файлов промптов: {stats['files_total']}; без ответов: {stats['files_missing_response']}; "
        f"не разобрано: {stats['files_unparseable']}.",
        f"- Объектов всего: {stats['objects_total']}; принято P1: {stats['accepted_p1']}; "
        f"принято P2: {stats['accepted_p2']}; пропущено skip: {stats['skipped']}.",
        f"- Доля принятых: {stats['accept_rate']} (критерий FD-10: >= 0.40).",
        "",
        "## По файлам",
        "",
        "| file | task | accepted | rejected | error |",
        "|---|---|---|---|---|",
    ]
    for rec in stats["per_file"]:
        lines.append(
            f"| {rec.get('file')} | {rec.get('task')} | {rec.get('accepted', 0)} | "
            f"{rec.get('rejected', 0)} | {rec.get('error', '')} |"
        )
    lines += ["", "## Причины отказов", "", "| reason | n |", "|---|---|"]
    for reason, count in sorted(stats["reasons"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Решение по few-shot",
        "",
        (
            "Доля принятых < 0.40: добавить пример prompts/p1_fewshot_clamp.txt в P1 и повторить пилот."
            if stats["accept_rate"] < 0.40
            else "Доля принятых >= 0.40: few-shot не требуется."
        ),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
