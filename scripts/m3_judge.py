"""FD-22: нулевой LLM-судья M3 (R10) на Claude Haiku 5.5 через Go Inference API.

Выбор судьи: Gemini 3.8 Flash рассматривался и отклонён — он размечал все 200
строк gold-кандидатов (reports/gold_report.md), судить по ним же — петля
самооценки; Muse Spark отклонён по той же причине (60 unlabeled + eval-аудит +
пилот). Haiku в проекте ранее не использовался (ноль пересечений), от
генератора синтетики DeepSeek отличается (R7 соблюдено).

    python scripts/m3_judge.py estimate
    python scripts/m3_judge.py export  # выборка 500 (идемпотентно)
    python scripts/m3_judge.py run --dir data/m3_judge --limit 6   # smoke
    python scripts/m3_judge.py run --dir data/m3_judge              # полный прогон
    python scripts/m3_judge.py import-api --dir data/m3_judge --out reports/preds/m3.csv

Выборка (фиксирована): 300 строк test_real (seed 22) + весь gold (200) = 500;
отклонение от FD-22 (250+250) — в gold всего 200 строк (FD-18).

run: одиночные запросы (точный R10, один кейс на запрос), дисковый кэш
(data/m3_judge/cache, ключ — хэш модели+параметров+сообщений) — повторный
запуск ключ не тратит. Ответы: data/m3_judge/responses/m3_<m3id>.response.json
+ data/m3_judge/usage.csv. Ключ OPENCODE_API_KEY — только для run.

Оценка: python scripts/evaluate.py --pred M0=reports/preds/m0.csv
--pred M1=reports/preds/m1.csv --pred M3=reports/preds/m3.csv
(M3 покрывает только test_real+gold; пустые срезы val/test_synth дают nan —
протокол R11 без усреднений).
"""

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

from src.models.m3_judge import (
    JUDGE_MODEL,
    parse_batch_response,
    validate_judge_object,
)

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUT_DIR_DEFAULT = PROJECT_ROOT / "data" / "m3_judge"
MANIFEST_NAME = "m3_manifest.csv"
SELECT_JSON = PROJECT_ROOT / "reports" / "tables" / "m3_select.json"
RAW_NAME = "m3_raw.jsonl"
RESPONSES_DIR = "responses"
CACHE_DIR = "cache"
USAGE_NAME = "usage.csv"
DEFAULT_SEED = 22
N_TEST_REAL = 300


def _load_cfg(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        path = PROJECT_ROOT / "config" / "m3_judge.yaml"
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cand = PROJECT_ROOT / cfg_path
        if cand.exists():
            cfg_path = cand
    if not cfg_path.exists():
        return {}
    import yaml

    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _api_params(cfg: dict[str, Any]):
    from src.synth.opencode_client import OpenCodeParams

    api = cfg.get("api", {}) if isinstance(cfg.get("api", {}), dict) else {}
    raw_temp = api.get("temperature", None)
    return OpenCodeParams(
        model=str(api.get("model", cfg.get("model", JUDGE_MODEL))),
        temperature=None if raw_temp is None else float(raw_temp),
        max_tokens=int(api.get("max_tokens", 1000)),
    )


def cmd_export(args: argparse.Namespace) -> None:
    cfg = _load_cfg(getattr(args, "config", None))
    seed = int(getattr(args, "seed", None) or cfg.get("sampling", {}).get("seed", DEFAULT_SEED))
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise SystemExit(f"Датасет отсутствует: {ds_path}")
    df = pd.read_parquet(ds_path)
    need = {"pair_id", "split", "diff_code", "doc_old"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"В датасете нет колонок: {sorted(missing)}")

    test_real = df[df["split"] == "test_real"].copy()
    gold = df[df["split"] == "gold"].copy()
    if len(gold) != 200:
        print(f"ВНИМАНИЕ: строк gold {len(gold)}, ожидалось 200 (FD-18)")
    if len(test_real) < N_TEST_REAL:
        raise SystemExit(f"В test_real {len(test_real)} строк, нужно {N_TEST_REAL}")
    rng = random.Random(seed)
    idx = list(test_real.index)
    rng.shuffle(idx)
    sel_test = test_real.loc[idx[:N_TEST_REAL]].copy()
    sel = pd.concat([sel_test, gold], ignore_index=True)
    order = list(range(len(sel)))
    rng = random.Random(seed + 1)
    rng.shuffle(order)
    sel = sel.iloc[order].reset_index(drop=True)
    sel.insert(0, "m3_id", [f"j{i + 1:05d}" for i in range(len(sel))])
    assert len(sel) == N_TEST_REAL + len(gold), len(sel)
    assert sel["pair_id"].is_unique, "дубли pair_id в отборе M3"

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = sel[["m3_id", "pair_id", "split"]].copy()
    manifest.to_csv(out_dir / MANIFEST_NAME, index=False)

    api = cfg.get("api", {}) if isinstance(cfg.get("api", {}), dict) else {}
    select = {
        "task": "FD-22",
        "model": str(api.get("model", cfg.get("model", JUDGE_MODEL))),
        "temperature": api.get("temperature", None),
        "max_tokens": int(api.get("max_tokens", 1000)),
        "seed": seed,
        "n_total": len(sel),
        "n_test_real": int((sel["split"] == "test_real").sum()),
        "n_gold": int((sel["split"] == "gold").sum()),
        "note": ("Отклонение от FD-22 (250 test_real + 250 gold): в gold всего 200 строк "
                 "(FD-18 зафиксирован), взяты весь gold (200) + 300 test_real = 500."),
        "m3_id_to_pair_id": dict(zip(manifest["m3_id"].astype(str), manifest["pair_id"].astype(str))),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    SELECT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SELECT_JSON.write_text(json.dumps(select, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Отбор M3: {len(sel)} строк (test_real {select['n_test_real']} + gold {select['n_gold']}, seed={seed})")
    print(f"Манифест: {out_dir / MANIFEST_NAME}")
    print(f"Список зафиксирован: {SELECT_JSON}")


def _write_outputs(
    preds: dict[str, dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
    report_text: str,
) -> None:
    out_pred = Path(args.out)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    df_pred = pd.DataFrame(
        [{"pair_id": v["pair_id"], "p_stale": v["p_stale"], "type_pred": v["type_pred"]}
         for v in preds.values()]
    )
    if len(df_pred):
        df_pred.to_csv(out_pred, index=False)
    else:
        pd.DataFrame(columns=["pair_id", "p_stale", "type_pred"]).to_csv(out_pred, index=False)

    raw_path = out_dir / RAW_NAME
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Принято: {stats['accepted']}/{stats['objects_total']} объектов")
    print(f"Причины: {stats['reasons']}")
    print(f"Предсказания: {out_pred} ({len(df_pred)} строк)")
    print(f"Сырые ответы: {raw_path}")
    print(f"Отчёт: {report_path}")


def _report_text(stats: dict[str, Any], out_dir: Path, args: argparse.Namespace,
                 usage_totals: dict[str, int] | None, effort: str | None = None,
                 max_tokens: str | None = None) -> str:
    cfg = _load_cfg(getattr(args, "config", None))
    api = cfg.get("api", {}) if isinstance(cfg.get("api", {}), dict) else {}
    model = str(api.get("model", cfg.get("model", JUDGE_MODEL)))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    effort_str = f", effort={effort}" if effort else " (без thinking)"
    mt = max_tokens or str(api.get("max_tokens", 1000))
    mode_lines = [
        f"- Модель-судья: {model} (temperature не отправляется — дефолт шлюза, "
        f"max_tokens={mt}{effort_str}). Дата импорта: {now}.",
        "- Прогон: одиночные запросы (точный R10, один кейс на запрос), дисковый кэш — повтор ключ не тратит.",
        "- p_stale_from: confidence (лог-вероятности Messages API не отдаёт, R10 fallback).",
    ]
    if usage_totals is not None:
        usd = (usage_totals.get("prompt_tokens", 0) * 0.10
               + usage_totals.get("completion_tokens", 0) * 0.50) / 1_000_000
        mode_lines.append(
            f"- Фактические токены: вход {usage_totals.get('prompt_tokens', 0)}, "
            f"выход {usage_totals.get('completion_tokens', 0)} "
            f"(запросов с usage: {usage_totals.get('n', 0)}); факт ~${usd:.3f} "
            f"(тир ≤100K: $0.10/$0.50 за 1M).")
    else:
        mode_lines.append("- Фактические токены: usage.csv отсутствует.")
    lines = [
        "# FD-22: нулевой LLM-судья M3 (Claude Haiku 5.5, через API)",
        "",
        *mode_lines,
        "- Генератор синтетики: deepseek-flash — с судьёй не совпадает (R7 соблюдено).",
        f"- Каталог партии: {out_dir}.",
        f"- Отбор: 300 test_real + весь gold (200) = 500 (отклонение от 250+250: "
        f"в gold всего 200, список — `reports/tables/m3_select.json`).",
        f"- Файлов: {stats.get('files_total', stats.get('requests_total', 0))}; "
        f"без ответов: {stats.get('files_missing_response', 0)}.",
        f"- Объектов всего: {stats['objects_total']}; принято: {stats['accepted']}; "
        f"пропущено skip: {stats['skipped']}.",
        f"- Доля принятых: "
        f"{round(stats['accepted'] / stats['objects_total'], 3) if stats['objects_total'] else 0.0}.",
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
        "## Оценка",
        "",
        "- `python scripts/evaluate.py --pred M0=reports/preds/m0.csv "
        "--pred M1=reports/preds/m1.csv --pred M3=reports/preds/m3.csv`",
        "- Порог для F1 выбирается на реальных строках val и применяется ко всем сплитам; "
        "M3 покрывает только test_real+gold, пустые срезы val/test_synth дают nan (протокол R11, без усреднений).",
        "",
    ]
    return "\n".join(lines)


def cmd_estimate(args: argparse.Namespace) -> None:
    from src.synth.opencode_client import (
        AVG_COMP_TOKENS,
        AVG_IN_TOKENS,
        MODEL,
        MODEL_LABEL,
        cost_table,
    )

    cfg = _load_cfg(getattr(args, "config", None))
    api = cfg.get("api", {}) if isinstance(cfg.get("api", {}), dict) else {}
    print(f"Судья: {MODEL_LABEL} ({api.get('model', MODEL)}), "
          f"temperature={'не отправляется (дефолт шлюза)' if api.get('temperature') is None else api.get('temperature')}, "
          f"max_tokens={api.get('max_tokens', 1000)}, thinking выкл (R10).")
    print(f"Допущения: вход ~{AVG_IN_TOKENS} ток./запрос, выход ~{AVG_COMP_TOKENS} "
          f"(пересчитать после smoke по usage.csv).")
    print("Прайс, USD за 1M (тир <=100K): вход 0.10, выход 0.50.")
    print("")
    print("| запросов | USD |")
    print("|---|---|")
    for row in cost_table([6, 500]):
        print(f"| {row['n_requests']} | {row['usd']:.3f} |")
    print("")
    print("Кэш: повторный run ключ не тратит; --force — перезапросить.")


def _run_one(m3_id: str, pair_id: str, diff_code: str, doc_old: str, out_dir: Path,
             api_key: str, base_url: str, params: Any, sleep: float, force: bool,
             session_id: str, timeout: int = 120) -> dict[str, Any]:
    from src.synth.opencode_client import build_messages, call_opencode

    resp_dir = out_dir / RESPONSES_DIR
    resp_path = resp_dir / f"m3_{m3_id}.response.json"
    if resp_path.exists() and not force:
        rec = json.loads(resp_path.read_text(encoding="utf-8"))
        return {"m3_id": m3_id, **rec.get("usage", {}),
                "latency_s": rec.get("latency_s", 0), "attempts": rec.get("attempts", 0),
                "cached": True, "status": "cached"}
    messages = build_messages(diff_code, doc_old, PROJECT_ROOT / "prompts")
    res = call_opencode(messages, api_key, base_url, params,
                        cache_dir=out_dir / CACHE_DIR, max_retries=3,
                        session_id=session_id, timeout=timeout)
    payload = {
        "m3_id": m3_id, "pair_id": pair_id,
        "model": res["model"],
        "params": {"model": params.model, "temperature": params.temperature,
                   "max_tokens": params.max_tokens, "effort": params.effort},
        "content": res["content"], "finish_reason": res.get("finish_reason"),
        "usage": res["usage"], "latency_s": res["latency_s"], "attempts": res["attempts"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    resp_dir.mkdir(parents=True, exist_ok=True)
    resp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if sleep:
        time.sleep(sleep)
    return {"m3_id": m3_id, **res["usage"],
            "latency_s": res["latency_s"], "attempts": res["attempts"],
            "cached": res["cached"], "status": "ok"}


def cmd_run(args: argparse.Namespace) -> None:
    from src.synth.opencode_client import BASE_URL_DEFAULT, OpenCodeParams, load_api_key

    cfg = _load_cfg(getattr(args, "config", None))
    api = cfg.get("api", {}) if isinstance(cfg.get("api", {}), dict) else {}
    out_dir = Path(args.dir)
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"Нет манифеста {manifest_path}: сначала export")
    manifest = pd.read_csv(manifest_path, dtype=str)
    if getattr(args, "limit", None):
        manifest = manifest.head(args.limit)
    ds = pd.read_parquet(args.dataset)
    by_pair = {str(r["pair_id"]): (str(r["diff_code"] or ""), str(r["doc_old"] or ""))
               for _, r in ds.iterrows()}
    missing = [p for p in manifest["pair_id"].astype(str) if p not in by_pair]
    if missing:
        raise SystemExit(f"В датасете нет {len(missing)} pair_id из манифеста")

    api_key = load_api_key()
    params = _api_params(cfg)
    if getattr(args, "effort", None) is not None:
        params = OpenCodeParams(model=params.model, temperature=params.temperature,
                                max_tokens=args.max_tokens or params.max_tokens,
                                effort=args.effort)
    elif getattr(args, "max_tokens", None) is not None:
        params = OpenCodeParams(model=params.model, temperature=params.temperature,
                                max_tokens=args.max_tokens, effort=params.effort)
    base_url = str(api.get("base_url", BASE_URL_DEFAULT))
    (out_dir / RESPONSES_DIR).mkdir(parents=True, exist_ok=True)
    (out_dir / CACHE_DIR).mkdir(parents=True, exist_ok=True)
    import uuid

    session_id = uuid.uuid4().hex  # один ID на партию: стабильная сессия для роутинга/кэша Go
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def job(rec: tuple[str, str]) -> None:
        m3_id, pair_id = rec
        diff_code, doc_old = by_pair[pair_id]
        try:
            out = _run_one(m3_id, pair_id, diff_code, doc_old, out_dir,
                           api_key, base_url, params, args.sleep, args.force,
                           session_id, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001 — один запрос не роняет партию
            out = {"m3_id": m3_id, "status": f"error: {exc}"}
        with lock:
            rows.append(out)
            done = len(rows)
        print(f"[{done}/{len(manifest)}] {m3_id}: {out.get('status')}")

    pairs = list(zip(manifest["m3_id"].astype(str), manifest["pair_id"].astype(str)))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(job, pairs))

    ok_rows = [r for r in rows if r.get("status") in ("ok", "cached")]
    usage_path = out_dir / USAGE_NAME
    with open(usage_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["m3_id", "prompt_tokens",
                                               "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                                               "completion_tokens", "reasoning_tokens", "total_tokens",
                                               "latency_s", "attempts", "cached", "status"])
        writer.writeheader()
        for r in sorted(rows, key=lambda x: str(x.get("m3_id"))):
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    errors = [r for r in rows if str(r.get("status")).startswith("error")]
    comp = [r["completion_tokens"] for r in ok_rows if isinstance(r.get("completion_tokens"), int)]
    print(f"Готово: ok/cached {len(ok_rows)}/{len(manifest)}, ошибок: {len(errors)}; usage: {usage_path}")
    if comp:
        print(f"Completion-токенов: среднее {sum(comp) / len(comp):.0f}, max {max(comp)} "
              f"(лимит {params.max_tokens}; при упоре в лимит — ответы режутся)")
    for r in errors[:10]:
        print(f"  {r['m3_id']}: {r['status']}")


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


def cmd_import_api(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"Нет манифеста {manifest_path}: сначала export")
    manifest = pd.read_csv(manifest_path, dtype=str)
    by_id = {str(r["m3_id"]): str(r["pair_id"]) for _, r in manifest.iterrows()}
    resp_files = sorted((out_dir / RESPONSES_DIR).glob("m3_*.response.json"))
    if not resp_files:
        raise SystemExit(f"В {out_dir / RESPONSES_DIR} нет ответов: сначала run")

    stats: dict[str, Any] = {"requests_total": len(by_id),
                             "files_total": len(resp_files),
                             "files_missing_response": len(by_id) - len(resp_files),
                             "objects_total": 0, "accepted": 0, "skipped": 0,
                             "reasons": {}, "duplicate_ids": 0}
    if stats["files_missing_response"] > 0:
        stats["reasons"]["missing_response"] = stats["files_missing_response"]
    preds: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    for resp_path in resp_files:
        try:
            payload = json.loads(resp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            stats["reasons"]["bad_file"] = stats["reasons"].get("bad_file", 0) + 1
            raw_rows.append({"file": resp_path.name, "error": f"bad_file: {exc}"})
            continue
        mid = str(payload.get("m3_id") or "")
        objects, error = parse_batch_response(payload.get("content", ""))
        if error is not None or len(objects) != 1:
            reason = error.split(':')[0] if error else f"bad_count:{len(objects)}"
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            raw_rows.append({"file": resp_path.name, "m3_id": mid, "error": error or reason})
            continue
        obj = objects[0]
        if not isinstance(obj.get("id"), str):
            obj = dict(obj, id=mid)  # одиночный ответ без id — id из имени файла
        stats["objects_total"] += 1
        ok, reason, parts = validate_judge_object(obj)
        if not ok:
            if reason == "skipped":
                stats["skipped"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            raw_rows.append({"file": resp_path.name, "m3_id": mid,
                              "error": reason, "raw": obj})
            continue
        assert parts is not None
        if mid not in by_id:
            stats["reasons"]["unknown_id"] = stats["reasons"].get("unknown_id", 0) + 1
            continue
        if mid in preds:
            stats["duplicate_ids"] += 1
            continue
        pair_id = by_id[mid]
        preds[mid] = {"pair_id": pair_id, "p_stale": parts["p_stale"],
                      "type_pred": parts["type_pred"]}
        stats["accepted"] += 1
        raw_rows.append({"file": resp_path.name, "m3_id": mid, "pair_id": pair_id,
                         "p_stale": parts["p_stale"], "type_pred": parts["type_pred"],
                         "stale": parts["stale"], "confidence": parts["confidence"],
                         "reason": parts["reason"],
                         "usage": payload.get("usage", {}),
                         "latency_s": payload.get("latency_s", 0)})

    totals = _usage_totals(out_dir)
    effort = None
    mt_set: set[int] = set()
    for resp_path in sorted((out_dir / RESPONSES_DIR).glob("m3_*.response.json")):
        try:
            params = json.loads(resp_path.read_text(encoding="utf-8")).get("params", {})
            if effort is None:
                effort = params.get("effort")
            if isinstance(params.get("max_tokens"), int):
                mt_set.add(int(params["max_tokens"]))
        except (json.JSONDecodeError, OSError, AttributeError):
            continue
    mt_label = str(sorted(mt_set)[0]) if len(mt_set) == 1 else "mixed:" + "/".join(map(str, sorted(mt_set)))
    report = _report_text(stats, out_dir, args, usage_totals=totals, effort=effort, max_tokens=mt_label)
    _write_outputs(preds, raw_rows, stats, out_dir, args, report)
    if stats["accepted"] < len(by_id):
        print(f"WARNING: принято {stats['accepted']} из {len(by_id)} - "
              f"повторите run (кэш сохранит готовое), иначе evaluate.py даст nan.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FD-22: M3 судья (Claude Haiku 5.5, API).")
    parser.add_argument("--config", default="config/m3_judge.yaml")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("export", help="отбор 500 строк + манифест (без сети)")
    p.add_argument("--dataset", default=str(IN_DATASET))
    p.add_argument("--dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("estimate", help="смета API-прогона (без сети и ключа)")
    p.set_defaults(func=cmd_estimate)
    p = sub.add_parser("run", help="вызовы API (нужен OPENCODE_API_KEY)")
    p.add_argument("--dataset", default=str(IN_DATASET))
    p.add_argument("--dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout", type=int, default=120,
                   help="таймаут HTTP-запроса, с (max effort отвечает медленно)")
    p.add_argument("--effort", type=str, default=None,
                   help="effort шлюза (напр. max): thinking adaptive; новый кэш, полный реран")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=None,
                   help="переопределить лимит токенов из конфига")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("import-api", help="разбор responses/*.response.json -> preds + отчёт (без сети)")
    p.add_argument("--dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--out", default="reports/preds/m3.csv")
    p.add_argument("--report", default="reports/m3_judge.md")
    p.set_defaults(func=cmd_import_api)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
