"""FD-26: сведение двух независимых LLM-разметок ошибок M2 и выделение споров.

    python scripts/combine_fd26.py --spark reports/fd26_spark.csv --gemini reports/fd26_gemini.csv
    python scripts/combine_fd26.py --check  # только проверка артефактов отбора (без LLM-файлов)
    python scripts/combine_fd26.py --export-blind  # строгий blind-файл споров (без таргетов)
    python scripts/combine_fd26.py --spark ... --gemini ... --human reports/fd26_disputes_labeled.csv

LLM-файлы: err_id,label,stale_statement,why (label в A/B/C/D, шкала R9).

Проверки: покрытие всех err_id из манифеста, без дублей; label валиден;
для A stale_statement непуст и входит в doc_old (нормализованно: CRLF->LF,
strip кавычек/пробелов); для не-A stale_statement пуст.

Выходы:
- reports/fd26_combined.csv — err_id, spark, gemini, consensus, human, final,
  cause (+ dataset_stale, m2_pred, error_type, sig_type, drift_type, флаги усечения);
- reports/fd26_disputes.csv — только споры (consensus == dispute) + тексты full;
- reports/fd26_disputes_blind.csv (--export-blind) — строго слепой файл споров:
  err_id,project,symbol,sig_old,diff_code,doc_old,sig_type,n_src_lines +
  пустые label/stale_statement/why; без split/source/меток/скоров;
- каппа Коэна (4-класс и бинарно A/B→1, C→0; D исключается; явные labels).

Причина cause (эксклюзивно, порядок правил):
  needs_context (финал D) > noise (финал == M2 и != датасета) >
  truncation (флаги усечения) > fn_miss (FN) >
  sig_overtrigger (FP + sig_type != none) > neutral_refactor (остальные FP).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.fd26util import contains_quote, key_of

IN_FULL = PROJECT_ROOT / "reports" / "fd26_errors_full.csv"
IN_ANNO = PROJECT_ROOT / "reports" / "fd26_for_annotation.csv"
IN_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "fd26_manifest.json"
IN_SHARDS = PROJECT_ROOT / "reports" / "fd26_shards"
OUT_COMBINED = PROJECT_ROOT / "reports" / "fd26_combined.csv"
OUT_DISPUTES = PROJECT_ROOT / "reports" / "fd26_disputes.csv"
OUT_BLIND = PROJECT_ROOT / "reports" / "fd26_disputes_blind.csv"

VALID = ("A", "B", "C", "D")
# Причины: 6 автоматических (порядок правил в assign_cause) + complex_docstring
# только для ручного override (автоправила её не ставят; см. отчёт FD-26).
AUTO_CAUSES = ("needs_context", "noise", "truncation", "fn_miss",
               "sig_overtrigger", "neutral_refactor")
ALLOWED_CAUSES = AUTO_CAUSES + ("complex_docstring",)


def _is_set(value) -> bool:
    """Флаг усечения как строка CSV ("True") или нативный bool."""
    return str(value) == "True" or value is True


def assign_cause(final: str, ds: int, pred: int, error_type: str,
                 sig_type: str, truncated, doc_truncated) -> str:
    """Эксклюзивная причина (порядок правил; нарушение контракта — ValueError).

    1. needs_context (финал D); 2. noise (финал == M2 и != датасета);
    3. truncation (флаги усечения); 4. fn_miss (FN);
    5. sig_overtrigger (FP + sig_type != none); 6. neutral_refactor (остальные FP).
    """
    if final not in ("A", "B", "C", "D"):
        raise ValueError(f"final вне A/B/C/D: {final!r}")
    if error_type not in ("FP", "FN"):
        raise ValueError(f"error_type вне FP/FN: {error_type!r}")
    if final == "D":
        return "needs_context"
    if sig_type is None or (isinstance(sig_type, float) and pd.isna(sig_type)) \
            or str(sig_type).strip() == "":
        raise ValueError(f"пустой sig_type при final={final}")
    bmap = {"A": 1, "B": 1, "C": 0}
    if bmap[final] == pred and bmap[final] != ds:
        return "noise"
    if _is_set(truncated) or _is_set(doc_truncated):
        return "truncation"
    if error_type == "FN":
        return "fn_miss"
    if sig_type != "none":
        return "sig_overtrigger"
    return "neutral_refactor"


BLIND_COLS = ["err_id", "project", "symbol", "sig_old", "diff_code", "doc_old",
              "sig_type", "n_src_lines"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def check_artifacts(full_p: Path, anno_p: Path, manifest_p: Path, shards_dir: Path) -> int:
    manifest = json.loads(Path(manifest_p).read_text(encoding="utf-8"))
    full = read_csv(Path(full_p))
    anno = read_csv(Path(anno_p))
    assert len(full) == manifest["n_total"] == len(anno), \
        f"счётчики: full {len(full)}, anno {len(anno)}, manifest {manifest['n_total']}"
    assert full["err_id"].is_unique and anno["err_id"].is_unique
    assert set(full["err_id"]) == set(anno["err_id"]) == set(manifest["err_id_to_pair_id"])
    # Слепота anno: нет сплита/источника, меток, скоров и флагов усечения.
    forbidden = {"split", "source", "label_stale", "m2_p", "m2_pred",
                 "error_type", "truncated", "doc_truncated"}
    leak = forbidden & set(anno.columns)
    assert not leak, f"в anno утечка колонок: {sorted(leak)}"
    # Ключи sha1 совпадают.
    keys = manifest["keys"]
    bad_keys = [r["err_id"] for _, r in full.iterrows()
                if keys.get(str(r["err_id"])) != key_of(r["diff_code"], r["doc_old"])]
    assert not bad_keys, f"ключи sha1 не сошлись: {bad_keys[:5]}"
    # Шарды покрывают anno без дублей.
    shards = sorted(Path(shards_dir).glob("fd26_shard_*.csv"))
    assert shards, f"нет шардов в {shards_dir}"
    seen: set[str] = set()
    total = 0
    for p in shards:
        df = read_csv(p)
        assert set(df["err_id"]) & seen == set(), f"дубль err_id в {p.name}"
        seen |= set(df["err_id"].astype(str))
        total += len(df)
    assert seen == set(anno["err_id"].astype(str)), "шарды не покрывают anno"
    print(f"check OK: {len(full)} ошибок "
          f"(gold {manifest['n_gold']}, test_real {manifest['n_test_real']}), "
          f"шардов {len(shards)} ({total} строк), ключи sha1 сошлись.")
    print(f"Усечение: A {manifest['n_truncated_A']}, doc {manifest['n_truncated_doc']}.")
    return 0


def load_llm(path: Path, tag: str) -> pd.DataFrame:
    df = read_csv(path)
    need = {"err_id", "label", "stale_statement", "why"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{tag}: нет колонок {sorted(missing)} в {path}")
    if df["err_id"].duplicated().any():
        dup = df[df["err_id"].duplicated()]["err_id"].tolist()[:5]
        raise SystemExit(f"{tag}: дубли err_id: {dup}")
    bad = sorted(set(df["label"]) - set(VALID))
    if bad:
        raise SystemExit(f"{tag}: невалидные label: {bad}")
    return df


def validate_quotes(full: pd.DataFrame, df: pd.DataFrame, tag: str) -> None:
    doc_by_err = {str(r["err_id"]): str(r["doc_old"]) for _, r in full.iterrows()}
    bad = []
    for _, r in df.iterrows():
        doc = doc_by_err[str(r["err_id"])]
        if r["label"] == "A":
            if not contains_quote(doc, r["stale_statement"]):
                bad.append(str(r["err_id"]))
        elif str(r["stale_statement"]).strip():
            bad.append(str(r["err_id"]))
    if bad:
        raise SystemExit(f"{tag}: stale_statement не цитата doc_old ({len(bad)}): {bad[:5]}")


def cmd_export_blind(args: argparse.Namespace) -> None:
    disputes = read_csv(Path(args.out_disputes))
    full = read_csv(Path(args.full))
    err_ids = disputes["err_id"].astype(str).tolist()
    blind = full[full["err_id"].astype(str).isin(err_ids)][BLIND_COLS].copy()
    blind = blind.sort_values("err_id").reset_index(drop=True)
    blind["label"] = ""
    blind["stale_statement"] = ""
    blind["why"] = ""
    out = Path(args.out_blind)
    out.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(out, index=False)
    print(f"blind: {out} ({len(blind)} строк, без split/source/меток/скоров)")


def main() -> None:
    ap = argparse.ArgumentParser(description="FD-26: сведение LLM-разметок.")
    ap.add_argument("--spark", default=str(PROJECT_ROOT / "reports" / "fd26_spark.csv"))
    ap.add_argument("--gemini", default=str(PROJECT_ROOT / "reports" / "fd26_gemini.csv"))
    ap.add_argument("--full", default=str(IN_FULL))
    ap.add_argument("--manifest", default=str(IN_MANIFEST))
    ap.add_argument("--anno", default=str(IN_ANNO))
    ap.add_argument("--shards_dir", default=str(IN_SHARDS))
    ap.add_argument("--out_combined", default=str(OUT_COMBINED))
    ap.add_argument("--out_disputes", default=str(OUT_DISPUTES))
    ap.add_argument("--out_blind", default=str(OUT_BLIND))
    ap.add_argument("--human", default=None,
                    help="файл ручного арбитража споров (err_id,label[,cause]) — "
                         "метки принимаются ТОЛЬКО для строк со статусом dispute; "
                         "непустая колонка cause — ручной override причины")
    ap.add_argument("--check", action="store_true",
                    help="только проверка артефактов отбора (LLM-файлы не нужны)")
    ap.add_argument("--export-blind", action="store_true",
                    help="только генерация строгого blind-файла споров из disputes")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check_artifacts(Path(args.full), Path(args.anno),
                                         Path(args.manifest), Path(args.shards_dir)))
    if args.export_blind:
        cmd_export_blind(args)
        return

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    full = read_csv(Path(args.full))
    spark = load_llm(Path(args.spark), "spark")
    gemini = load_llm(Path(args.gemini), "gemini")

    expected = set(manifest["err_id_to_pair_id"])
    for tag, df in (("spark", spark), ("gemini", gemini)):
        missing = expected - set(df["err_id"])
        extra = set(df["err_id"]) - expected
        if missing or extra:
            raise SystemExit(f"{tag}: missing {len(missing)} (напр. {sorted(missing)[:3]}), "
                             f"extra {len(extra)} (напр. {sorted(extra)[:3]})")

    validate_quotes(full, spark, "spark")
    validate_quotes(full, gemini, "gemini")

    human_map: dict[str, str] = {}
    human_cause: dict[str, str] = {}
    if args.human is not None:
        hum = load_llm(Path(args.human), "human")
        # Арбитраж — только споры: метки вне dispute-строк отклоняются.
        _s = dict(zip(spark["err_id"].astype(str), spark["label"].astype(str)))
        _g = dict(zip(gemini["err_id"].astype(str), gemini["label"].astype(str)))
        dispute_ids = {eid for eid in _s if _s[eid] != _g.get(eid)}
        outside = sorted(set(hum["err_id"].astype(str)) - dispute_ids)
        if outside:
            raise SystemExit(f"human: метки вне споров ({len(outside)}): {outside[:5]} "
                             f"(арбитраж только для consensus==dispute)")
        missing = sorted(dispute_ids - set(hum["err_id"].astype(str)))
        if missing:
            raise SystemExit(f"human: не покрыты споры ({len(missing)}): {missing[:5]}")
        human_map = dict(zip(hum["err_id"].astype(str), hum["label"].astype(str)))
        if "cause" in hum.columns:
            for _, hr in hum.iterrows():
                ov = str(hr["cause"]).strip()
                if ov:
                    if ov not in ALLOWED_CAUSES:
                        raise SystemExit(f"human: недопустимый cause {ov!r} "
                                         f"у {hr['err_id']} (разрешены {ALLOWED_CAUSES})")
                    human_cause[str(hr["err_id"])] = ov

    s_map = dict(zip(spark["err_id"].astype(str), spark["label"].astype(str)))
    g_map = dict(zip(gemini["err_id"].astype(str), gemini["label"].astype(str)))
    rows = []
    for _, r in full.iterrows():
        eid = str(r["err_id"])
        s, g = s_map[eid], g_map[eid]
        consensus = s if s == g else "dispute"
        hum = human_map.get(eid, "")
        fin = hum if hum else ("" if consensus == "dispute" else consensus)
        cause = ""
        if fin:
            if eid in human_cause:
                cause = human_cause[eid]
            else:
                cause = assign_cause(fin, int(r["label_stale"]), int(r["m2_pred"]),
                                     str(r["error_type"]), str(r["sig_type"]),
                                     r["truncated"], r["doc_truncated"])
        rows.append({"err_id": eid, "pair_id": str(r["pair_id"]), "split": str(r["split"]),
                     "error_type": str(r["error_type"]), "dataset_stale": str(r["label_stale"]),
                     "m2_pred": str(r["m2_pred"]), "sig_type": str(r["sig_type"]),
                     "drift_type": str(r["drift_type"]), "truncated": str(r["truncated"]),
                     "doc_truncated": str(r["doc_truncated"]),
                     "spark": s, "gemini": g, "consensus": consensus,
                     "human": hum, "final": fin, "cause": cause})
    combined = pd.DataFrame(rows)
    disputes = combined[combined["consensus"] == "dispute"].copy()
    # Для разбора приложить тексты из full (внутренний файл, не для разметчиков).
    text_cols = ["project", "symbol", "sig_old", "diff_code", "doc_old"]
    full_text = full[["err_id"] + text_cols].copy()
    disputes = disputes.merge(full_text, on="err_id", how="left")

    Path(args.out_combined).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_combined, index=False)
    disputes.to_csv(args.out_disputes, index=False)

    # Каппа с явными классами (стабильная размерность матрицы).
    try:
        from sklearn.metrics import cohen_kappa_score
        s4 = combined["spark"].tolist()
        g4 = combined["gemini"].tolist()
        k4 = cohen_kappa_score(s4, g4, labels=list(VALID))
        bmap = {"A": 1, "B": 1, "C": 0}
        pairs = [(bmap[s], bmap[g]) for s, g in zip(s4, g4)
                 if s in bmap and g in bmap]
        kb = (cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs],
                                labels=[0, 1]) if pairs else float("nan"))
        print(f"spark-vs-gemini: n={len(combined)}, совпало {(combined['spark'] == combined['gemini']).sum()}, "
              f"kappa4={k4:.3f}, kappa_bin={kb:.3f} (D исключены: {len(s4) - len(pairs)})")
    except ImportError:
        print("sklearn нет — каппа не посчитана.")
    print(f"combined: {args.out_combined} ({len(combined)} строк)")
    print(f"disputes: {args.out_disputes} ({len(disputes)} строк) — вручную разбирает человек.")


if __name__ == "__main__":
    raise SystemExit(main())
