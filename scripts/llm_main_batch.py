from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.normalizer import parse_function_ast
from src.synth.deepseek_client import (
    AVG_COMP_TOKENS,
    AVG_IN_TOKENS,
    BASE_URL_DEFAULT,
    COMP_TOKENS_RANGE,
    MODEL,
    MODEL_VERSION,
    PEAK_PLUS7,
    PEAK_UTC,
    DeepSeekParams,
    build_messages,
    call_deepseek,
    cost_table,
    cost_usd,
    load_api_key,
)
from src.synth.llm_batch import build_paraphrase_row, build_synth_row
from src.synth.llm_validators import parse_batch_response, validate_p1, validate_p2, validate_p3

TASKS = ("p1", "p2")
MANIFEST_NAME = "main_manifest.csv"
REQUESTS_DIR = "requests"
RESPONSES_DIR = "responses"
CACHE_DIR = "cache"
USAGE_NAME = "usage.csv"
CONFIG_DEFAULT = Path("config/llm_main.yaml")


def load_config(path: Path | str = CONFIG_DEFAULT) -> DeepSeekParams:
    """Параметры DeepSeek из config/llm_main.yaml (FD-11 fix: скрипт больше
    не игнорирует конфиг). При отсутствии файла — дефолт dataclass."""
    cfg_path = Path(path)
    # Резолв относительно корня репозитория, если вызван из другой cwd.
    if not cfg_path.is_absolute():
        cand = PROJECT_ROOT / cfg_path
        if cand.exists():
            cfg_path = cand
    if not cfg_path.exists():
        return DeepSeekParams()
    import yaml

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return DeepSeekParams(
        model=str(cfg.get("model", MODEL)),
        thinking=str(cfg.get("thinking", "disabled")),
        temperature=float(cfg.get("temperature", 0.7)),
        top_p=float(cfg.get("top_p", 0.95)),
        max_tokens=int(cfg.get("max_tokens", 1200)),
    )


def _symbol_of(src: str) -> str:
    fn = parse_function_ast(src)
    return fn.name if fn is not None else ""


def sample_main_functions(
    df_pool: pd.DataFrame,
    n: int,
    seed: int,
    exclude_hashes: set[str] | None = None,
    id_prefix: str = "M",
) -> pd.DataFrame:
    """Round-robin по проектам (как пилот FD-10), детерминированно по seed."""
    df = df_pool
    if exclude_hashes:
        df = df[~df["function_hash"].astype(str).isin(exclude_hashes)].copy()
    if len(df) < n:
        raise ValueError(f"В пуле {len(df)} строк после исключений, нужно {n}")
    rng = random.Random(seed)
    by_project: dict[str, list[Any]] = {}
    for idx, row in df.iterrows():
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
    sample = df.loc[picked].copy().reset_index(drop=True)
    sample["m_id"] = [f"{id_prefix}{i + 1:04d}" for i in range(len(sample))]
    return sample


def load_pilot_hashes(pool_dir: Path) -> set[str]:
    manifest = pool_dir / "pilot_manifest.csv"
    if not manifest.exists():
        # пилотный каталог по умолчанию
        alt = Path("data/llm_pilot/pilot_manifest.csv")
        manifest = alt if alt.exists() else manifest
    if not manifest.exists():
        return set()
    df = pd.read_csv(manifest, dtype=str)
    return set(df["function_hash"].astype(str).tolist())


def cmd_export(args: argparse.Namespace) -> None:
    df_pool = pd.read_parquet(args.pool)
    exclude = load_pilot_hashes(Path(args.dir)) if not args.include_pilot else set()
    sample = sample_main_functions(df_pool, args.n, args.seed, exclude)
    out_dir = Path(args.dir)
    (out_dir / REQUESTS_DIR).mkdir(parents=True, exist_ok=True)
    prompts_dir = Path(args.prompts)
    manifest = pd.DataFrame({
        "m_id": sample["m_id"],
        "function_hash": sample["function_hash"],
        "project": sample["project"],
        "symbol": [_symbol_of(s) for s in sample["src"]],
    })
    manifest.to_csv(out_dir / MANIFEST_NAME, index=False)
    src_by_hash = {str(r["function_hash"]): str(r["src"]) for _, r in sample.iterrows()}
    params = load_config(getattr(args, "config", CONFIG_DEFAULT))
    for _, row in manifest.iterrows():
        src = src_by_hash[str(row["function_hash"])]
        for task in TASKS:
            messages = build_messages(task, src, prompts_dir)
            req = {
                "m_id": str(row["m_id"]),
                "task": task,
                "function_hash": str(row["function_hash"]),
                "model": params.model,
                "model_version": MODEL_VERSION,
                "params": params.__dict__,
                "messages": messages,
            }
            (out_dir / REQUESTS_DIR / f"{row['m_id']}_{task}.json").write_text(
                json.dumps(req, ensure_ascii=False), encoding="utf-8")
    n_req = len(manifest) * 2
    print(f"Выборка: {len(manifest)} функций из {manifest['project'].nunique()} проектов "
          f"(seed={args.seed}, исключено пилотных: {len(exclude)})")
    print(f"Запросов P1+P2: {n_req}; каталог: {out_dir}")
    print(f"Смета (оценка): off-peak ${cost_usd(n_req, AVG_IN_TOKENS, AVG_COMP_TOKENS, False):.3f} / "
          f"peak ${cost_usd(n_req, AVG_IN_TOKENS, AVG_COMP_TOKENS, True):.3f}")


def _run_one(req_path: Path, out_dir: Path, api_key: str, base_url: str,
             params: DeepSeekParams, sleep: float, force: bool) -> dict[str, Any]:
    req = json.loads(req_path.read_text(encoding="utf-8"))
    resp_path = out_dir / RESPONSES_DIR / f"{req_path.stem}.response.json"
    if resp_path.exists() and not force:
        rec = json.loads(resp_path.read_text(encoding="utf-8"))
        return {"m_id": req["m_id"], "task": req["task"], **rec.get("usage", {}),
                "latency_s": rec.get("latency_s", 0), "attempts": rec.get("attempts", 0),
                "cached": True, "status": "cached"}
    res = call_deepseek(req["messages"], api_key, base_url, params,
                        cache_dir=out_dir / CACHE_DIR, max_retries=3)
    payload = {
        "m_id": req["m_id"], "task": req["task"], "function_hash": req["function_hash"],
        "model": res["model"], "model_version": MODEL_VERSION,
        "content": res["content"], "finish_reason": res.get("finish_reason"),
        "usage": res["usage"], "latency_s": res["latency_s"], "attempts": res["attempts"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / RESPONSES_DIR).mkdir(parents=True, exist_ok=True)
    resp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if sleep:
        time.sleep(sleep)
    return {"m_id": req["m_id"], "task": req["task"], **res["usage"],
            "latency_s": res["latency_s"], "attempts": res["attempts"],
            "cached": res["cached"], "status": "ok"}


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    req_files = sorted((out_dir / REQUESTS_DIR).glob("M*_p*.json"))
    if args.limit:
        # первые limit функций => по 2 запроса
        keep_ids = sorted({p.name.split("_")[0] for p in req_files})[:args.limit]
        req_files = [p for p in req_files if p.name.split("_")[0] in keep_ids]
    if not req_files:
        raise RuntimeError(f"В {out_dir / REQUESTS_DIR} нет запросов: сначала export")
    api_key = load_api_key()
    params = load_config(getattr(args, "config", CONFIG_DEFAULT))
    (out_dir / RESPONSES_DIR).mkdir(parents=True, exist_ok=True)
    (out_dir / CACHE_DIR).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def job(path: Path) -> None:
        try:
            rec = _run_one(path, out_dir, api_key, args.base_url, params, args.sleep, args.force)
        except Exception as exc:  # noqa: BLE001 — один запрос не роняет партию
            rec = {"m_id": path.name.split("_")[0], "task": path.stem.split("_")[-1],
                   "status": f"error: {exc}"}
        with lock:
            rows.append(rec)
            done = len(rows)
        print(f"[{done}/{len(req_files)}] {path.name}: {rec.get('status')}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(job, req_files))
    ok_rows = [r for r in rows if r.get("status") in ("ok", "cached")]
    usage_path = out_dir / USAGE_NAME
    with open(usage_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["m_id", "task", "prompt_tokens",
                                               "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                                               "completion_tokens", "reasoning_tokens", "total_tokens",
                                               "latency_s", "attempts", "cached", "status"])
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (str(x.get("m_id")), str(x.get("task")))):
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    errors = [r for r in rows if str(r.get("status")).startswith("error")]
    print(f"Готово: ok/cached {len(ok_rows)}/{len(req_files)}, ошибок: {len(errors)}; usage: {usage_path}")
    if errors:
        for r in errors[:10]:
            print(f"  {r['m_id']} {r['task']}: {r['status']}")
    comp = [r.get("completion_tokens", 0) for r in ok_rows if isinstance(r.get("completion_tokens"), int)]
    if comp:
        print(f"Completion-токенов: среднее {sum(comp) / len(comp):.0f}, max {max(comp)} "
              f"(проверьте смету: закладывалось {AVG_COMP_TOKENS}, диапазон {COMP_TOKENS_RANGE})")


def cmd_import(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    df_pool = pd.read_parquet(args.pool)
    pool_src = {str(r["function_hash"]): str(r["src"]) for _, r in df_pool.iterrows()}
    pool_proj = {str(r["function_hash"]): str(r["project"]) for _, r in df_pool.iterrows()}
    manifest = pd.read_csv(out_dir / MANIFEST_NAME, dtype=str)
    if getattr(args, "limit", None):
        manifest = manifest.head(args.limit)
    stats: dict[str, Any] = {"objects_total": 0, "accepted_p1": 0, "accepted_p2": 0,
                             "skipped": 0, "unknown_id": 0, "reasons": {}, "per_task": {},
                             "slots": len(manifest) * 2}
    accepted_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        mid, fhash = str(row["m_id"]), str(row["function_hash"])
        src_old = pool_src.get(fhash)
        if src_old is None:
            stats["unknown_id"] += 2
            continue
        for task in TASKS:
            resp_path = out_dir / RESPONSES_DIR / f"{mid}_{task}.response.json"
            if not resp_path.exists():
                stats["reasons"]["missing_response"] = stats["reasons"].get("missing_response", 0) + 1
                continue
            payload = json.loads(resp_path.read_text(encoding="utf-8"))
            objects, error = parse_batch_response(payload.get("content", ""))
            if error is not None or len(objects) != 1:
                reason = error or f"bad_count:{len(objects)}"
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                continue
            stats["objects_total"] += 1
            validator = validate_p1 if task == "p1" else validate_p2
            ok, reason, parts = validator(src_old, objects[0])
            if not ok:
                if reason == "skipped":
                    stats["skipped"] += 1
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                continue
            assert parts is not None
            accepted_rows.append(build_synth_row(task, mid, fhash, pool_proj[fhash],
                                                str(row.get("symbol", "")), src_old, parts))
            stats[f"accepted_{task}"] += 1
    df_out = pd.DataFrame(accepted_rows)
    out_parquet = Path(args.out)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if len(df_out):
        df_out.to_parquet(out_parquet, index=False)
    else:
        # R1: 17 колонок схемы без pilot_id (исправлено с устаревшей pilot_id).
        pd.DataFrame(columns=["pair_id", "source", "project", "commit_sha", "symbol", "function_hash",
                              "sig_old", "code_old", "code_new", "diff_code", "doc_old", "label_stale",
                              "drift_type", "sig_type", "label_source", "weight", "n_src_lines",
                              ]).to_parquet(out_parquet, index=False)
    stats["accepted_total"] = stats["accepted_p1"] + stats["accepted_p2"]
    denom = stats["objects_total"]
    stats["accept_rate"] = round(stats["accepted_total"] / denom, 3) if denom else 0.0
    pre_parse = sum(v for k, v in stats["reasons"].items()
                    if k.startswith(("bad_json", "bad_count", "missing_response", "unknown_id")))
    valid_fails = sum(stats["reasons"].values()) - pre_parse
    stats["check_line"] = (
        f"слотов {stats['slots']} = разобрано {stats['objects_total']} + до-разбора {pre_parse}; "
        f"разобрано = принято {stats['accepted_total']} + отказы валидации {valid_fails}")
    _write_main_report(Path(args.report), stats, out_dir, args)
    print(f"Принято P1/P2: {stats['accepted_p1']}/{stats['accepted_p2']}, доля: {stats['accept_rate']}")
    print(f"Причины отказов: {stats['reasons']}")
    print(f"Отчёт: {args.report}; таблица: {args.out} ({len(df_out)} строк)")


def _usage_totals(out_dir: Path) -> dict[str, int]:
    usage_path = out_dir / USAGE_NAME
    totals = {"prompt_tokens": 0, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
              "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "n": 0}
    if not usage_path.exists():
        return totals
    with open(usage_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("status")) not in ("ok", "cached"):
                continue
            totals["n"] += 1
            for k in list(totals):
                if k == "n":
                    continue
                try:
                    totals[k] += int(float(row.get(k) or 0))
                except (TypeError, ValueError):
                    pass
    return totals


def _write_main_report(report_path: Path, stats: dict[str, Any], out_dir: Path,
                       args: argparse.Namespace) -> None:
    totals = _usage_totals(out_dir)
    p = load_config(getattr(args, "config", CONFIG_DEFAULT))
    n_req = totals.get("n") or stats["objects_total"]
    in_tok = totals.get("prompt_tokens", 0)
    comp_tok = totals.get("completion_tokens", 0)
    actual_off = (in_tok * 0.15 + comp_tok * 0.60) / 1_000_000
    actual_peak = (in_tok * 0.30 + comp_tok * 1.20) / 1_000_000
    lines = [
        "# FD-11: основная партия LLM-синтетики (P1/P2, DeepSeek API)",
        "",
        f"- Генератор: {MODEL} ({MODEL_VERSION}), thinking={p.thinking}, "
        f"temperature={p.temperature}, top_p={p.top_p}, max_tokens={p.max_tokens}. "
        f"Дата прогона: см. fetched_at в responses/.",
        "- Судья M3 (FD-22, Claude Haiku 5.5) для данных не используется.",
        f"- Каталог партии: {out_dir}; выборка: {getattr(args, 'limit', None) or getattr(args, 'n', None) or 'все'} "
        f"(seed — в имени каталога/документации).",
        f"- Объектов всего: {stats['objects_total']}; принято P1: {stats['accepted_p1']}; "
        f"принято P2: {stats['accepted_p2']}; пропущено skip: {stats['skipped']}.",
        f"- Доля принятых: {stats['accept_rate']} (ожидание FD-11: порядка 200+200 при минимуме 300 функций).",
        f"- Контроль арифметики: {stats.get('check_line', '—')}",
        "",
        "## Причины отказов",
        "",
        "| reason | n |",
        "|---|---|",
    ]
    for reason, count in sorted(stats["reasons"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Фактические токены и стоимость",
        "",
        f"- Запросов с usage: {totals['n']}; вход: {in_tok}; completion: {comp_tok} "
        f"(в т.ч. reasoning: {totals['reasoning_tokens']}).",
        f"- Факт: off-peak ${actual_off:.3f} / peak ${actual_peak:.3f}.",
        f"- Пик ({PEAK_UTC}; +7: {PEAK_PLUS7}). Вне пика — вдвое дешевле.",
        "",
        "## Аудит (вручную, FD-11 шаг 4)",
        "",
        "- Команда: `python scripts/llm_main_batch.py audit-sample --in-path <parquet> --out reports/llm_audit_sample.csv`",
        "- Проверить 25 P1 (стало ли неверным хотя бы одно утверждение docstring) и 25 P2 "
        "(сохранено ли поведение). Если в P2 > 20% меняют поведение — вес 0.3 или исключить P2.",
        "- Итог — в `reports/audit_v1.md`.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def cmd_export_p3(args: argparse.Namespace) -> None:
    """Запросы P3 (парафраз docstring) для функций существующего манифеста."""
    from src.data.normalizer import extract_docstring_and_clean_node, parse_function_ast

    out_dir = Path(args.dir)
    df_pool = pd.read_parquet(args.pool)
    pool_src = {str(r["function_hash"]): str(r["src"]) for _, r in df_pool.iterrows()}
    manifest = pd.read_csv(out_dir / MANIFEST_NAME, dtype=str)
    (out_dir / REQUESTS_DIR).mkdir(parents=True, exist_ok=True)
    prompts_dir = Path(args.prompts)
    params = load_config(getattr(args, "config", CONFIG_DEFAULT))
    n = 0
    for _, row in manifest.iterrows():
        src = pool_src.get(str(row["function_hash"]))
        if src is None:
            continue
        messages = build_messages("p3", src, prompts_dir)
        req = {
            "m_id": str(row["m_id"]),
            "task": "p3",
            "function_hash": str(row["function_hash"]),
            "model": params.model,
            "model_version": MODEL_VERSION,
            "params": params.__dict__,
            "messages": messages,
        }
        (out_dir / REQUESTS_DIR / f"{row['m_id']}_p3.json").write_text(
            json.dumps(req, ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"P3-запросов: {n}; каталог: {out_dir} (прогон — тем же run)")


def cmd_import_p3(args: argparse.Namespace) -> None:
    from src.data.normalizer import extract_docstring_and_clean_node, parse_function_ast

    def doc_of(src: str) -> str:
        fn = parse_function_ast(src)
        if fn is None:
            return ""
        doc, _ = extract_docstring_and_clean_node(fn)
        return (doc or "").strip()

    out_dir = Path(args.dir)
    df_pool = pd.read_parquet(args.pool)
    pool_src = {str(r["function_hash"]): str(r["src"]) for _, r in df_pool.iterrows()}
    manifest = pd.read_csv(out_dir / MANIFEST_NAME, dtype=str)
    stats: dict[str, Any] = {"objects_total": 0, "accepted": 0, "skipped": 0, "reasons": {}}
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        fhash = str(row["function_hash"])
        src_old = pool_src.get(fhash)
        if src_old is None:
            stats["reasons"]["unknown_id"] = stats["reasons"].get("unknown_id", 0) + 1
            continue
        resp_path = out_dir / RESPONSES_DIR / f"{row['m_id']}_p3.response.json"
        if not resp_path.exists():
            stats["reasons"]["missing_response"] = stats["reasons"].get("missing_response", 0) + 1
            continue
        payload = json.loads(resp_path.read_text(encoding="utf-8"))
        objects, error = parse_batch_response(payload.get("content", ""))
        if error is not None or len(objects) != 1:
            reason = error or f"bad_count:{len(objects)}"
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue
        stats["objects_total"] += 1
        ok, reason, parts = validate_p3(doc_of(src_old), objects[0])
        if not ok:
            if reason == "skipped":
                stats["skipped"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue
        assert parts is not None
        rows.append(build_paraphrase_row(fhash, doc_of(src_old), parts["doc_para"], parts["style"]))
        stats["accepted"] += 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["function_hash", "doc_orig", "doc_para", "style"]).to_parquet(out_path, index=False)
    denom = stats["objects_total"]
    rate = round(stats["accepted"] / denom, 3) if denom else 0.0
    print(f"P3 принято: {stats['accepted']}/{stats['objects_total']}, доля: {rate}")
    print(f"Причины отказов: {stats['reasons']}")
    print(f"Таблица: {out_path} ({len(rows)} строк)")


def cmd_audit_sample(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.in_path)
    rng = random.Random(args.seed)
    parts = []
    for source, n in (("llm_p1", 25), ("llm_p2", 25)):
        sub = df[df["label_source"] == source]
        if len(sub) < n:
            raise ValueError(f"В {args.in_path} только {len(sub)} строк {source}, нужно {n}")
        idx = rng.sample(range(len(sub)), n)
        parts.append(sub.iloc[idx])
    sample = pd.concat(parts, ignore_index=True)
    cols = ["pair_id", "label_source", "project", "symbol", "sig_old", "diff_code", "doc_old"]
    sample = sample[cols].copy()
    sample["verdict"] = ""
    sample["comment"] = ""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"Аудит-выборка: {len(sample)} строк (25 P1 + 25 P2) -> {out}")


def cmd_estimate(args: argparse.Namespace) -> None:
    p = load_config(getattr(args, "config", CONFIG_DEFAULT))
    print(f"Модель: {MODEL} ({MODEL_VERSION}); thinking={p.thinking}, "
          f"temperature={p.temperature}, top_p={p.top_p}, max_tokens={p.max_tokens}")
    print(f"Пик: {PEAK_UTC}; +7: {PEAK_PLUS7}. Off-peak вдвое дешевле (включая выходные).")
    print(f"Допущения (факт 13.10): вход {AVG_IN_TOKENS} ток./запрос, completion {AVG_COMP_TOKENS} "
          f"(диапазон {COMP_TOKENS_RANGE[0]}-{COMP_TOKENS_RANGE[1]}).")
    print("")
    print("| партия | запросов | off-peak, $ | peak, $ |")
    print("|---|---|---|---|")
    for row in cost_table([12, 600, 1400]):
        print(f"| {row['n_requests']} | {row['n_requests']} | {row['usd_offpeak']:.3f} | {row['usd_peak']:.3f} |")
    print("")
    print("Пересчёт от факта: запросы * (вход*цена_вх + completion*цена_вых)/1M по usage.csv.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FD-11: основная партия через DeepSeek API.")
    parser.add_argument("--config", default=str(CONFIG_DEFAULT),
                        help="путь к config/llm_main.yaml (по умолчанию %(default)s)")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("estimate", help="смета без сети")
    p.set_defaults(func=cmd_estimate)
    p = sub.add_parser("export", help="выборка функций и файлы запросов (без ключа)")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--tag", default="main")
    p.add_argument("--dir", default=None)
    p.add_argument("--pool", default="data/interim/functions_pool_v1.parquet")
    p.add_argument("--prompts", default="prompts")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--include-pilot", action="store_true")
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("run", help="вызовы API (нужен DEEPSEEK_API_KEY)")
    p.add_argument("--dir", required=True)
    p.add_argument("--base-url", default=BASE_URL_DEFAULT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("import", help="валидация ответов → parquet + отчёт (без ключа)")
    p.add_argument("--dir", required=True)
    p.add_argument("--pool", default="data/interim/functions_pool_v1.parquet")
    p.add_argument("--report", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="первые N функций манифеста (отчёт без missing_response)")
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("export-p3", help="запросы P3 для функций манифеста (без ключа)")
    p.add_argument("--dir", required=True)
    p.add_argument("--pool", default="data/interim/functions_pool_v1.parquet")
    p.add_argument("--prompts", default="prompts")
    p.set_defaults(func=cmd_export_p3)
    p = sub.add_parser("import-p3", help="валидация P3-ответов → parquet (без ключа)")
    p.add_argument("--dir", required=True)
    p.add_argument("--pool", default="data/interim/functions_pool_v1.parquet")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_import_p3)
    p = sub.add_parser("audit-sample", help="выборка 25+25 для ручного аудита")
    p.add_argument("--in-path", dest="in_path", required=True)
    p.add_argument("--out", default="reports/llm_audit_sample.csv")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_audit_sample)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "export" and args.dir is None:
        args.dir = f"data/llm_{args.tag}{args.n}"
    if args.mode == "import":
        name = Path(args.dir.rstrip("/")).name
        if name.startswith("llm_"):
            name = name[len("llm_"):]
        if args.report is None:
            args.report = f"reports/llm_{name}.md"
        if args.out is None:
            args.out = f"data/interim/synth_llm_{name}.parquet"
    args.func(args)


if __name__ == "__main__":
    main()
