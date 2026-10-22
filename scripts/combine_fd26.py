"""FD-26: сведение двух независимых LLM-разметок ошибок M2 и выделение споров.

    python scripts/combine_fd26.py --spark reports/fd26_spark.csv --gemini reports/fd26_gemini.csv
    python scripts/combine_fd26.py --check  # только проверка артефактов отбора (без LLM-файлов)

LLM-файлы (придут после прогона Spark 1.3 XHigh и Gemini 3.8 Flash High):
  err_id,label,stale_statement,why   (label в A/B/C/D, шкала R9 из prompts/fd26_annotate.txt)

Проверки (как в split_eval5146.py):
- покрытие всех err_id из reports/tables/fd26_manifest.json, без дублей;
- label валиден; для A stale_statement непуст и точная подстрока doc_old;
- для не-A stale_statement пуст.

Выходы:
- reports/fd26_combined.csv — err_id, spark, gemini, consensus (совпало / dispute),
  dataset_label (A/B/C/D-эквивалент? здесь бинарно stale 0/1), m2_pred, error_type;
- reports/fd26_disputes.csv — только споры (spark != gemini) + поля full для ручного разбора;
- каппа Коэна spark-vs-gemini (4-класс и бинарно A/B→1, C→0; D исключается).

До прогона LLM скрипт запускается с --check: сверяет full/anno/shards/manifest
(счётчики, уникальность, слепота anno, ключи sha1) и ничего не требует.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

IN_FULL = PROJECT_ROOT / "reports" / "fd26_errors_full.csv"
IN_ANNO = PROJECT_ROOT / "reports" / "fd26_for_annotation.csv"
IN_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "fd26_manifest.json"
IN_SHARDS = PROJECT_ROOT / "reports" / "fd26_shards"
OUT_COMBINED = PROJECT_ROOT / "reports" / "fd26_combined.csv"
OUT_DISPUTES = PROJECT_ROOT / "reports" / "fd26_disputes.csv"

VALID = {"A", "B", "C", "D"}


def key_of(diff_code: str, doc_old: str) -> str:
    return hashlib.sha1((str(diff_code or "") + str(doc_old or "")).encode("utf-8")).hexdigest()


def check_artifacts() -> int:
    manifest = json.loads(IN_MANIFEST.read_text(encoding="utf-8"))
    full = pd.read_csv(IN_FULL)
    anno = pd.read_csv(IN_ANNO)
    assert len(full) == manifest["n_total"] == len(anno), \
        f"счётчики: full {len(full)}, anno {len(anno)}, manifest {manifest['n_total']}"
    assert full["err_id"].is_unique and anno["err_id"].is_unique
    assert set(full["err_id"]) == set(anno["err_id"]) == set(manifest["err_id_to_pair_id"])
    # Слепота anno: нет меток, скоров и флагов усечения.
    forbidden = {"label_stale", "m2_p", "m2_pred", "error_type", "truncated", "doc_truncated"}
    leak = forbidden & set(anno.columns)
    assert not leak, f"в anno утечка колонок: {sorted(leak)}"
    # Ключи sha1 совпадают.
    keys = manifest["keys"]
    bad_keys = [r["err_id"] for _, r in full.iterrows()
                if keys.get(str(r["err_id"])) != key_of(r["diff_code"], r["doc_old"])]
    assert not bad_keys, f"ключи sha1 не сошлись: {bad_keys[:5]}"
    # Шарды покрывают anno без дублей.
    shards = sorted(IN_SHARDS.glob("fd26_shard_*.csv"))
    assert shards, f"нет шардов в {IN_SHARDS}"
    seen: set[str] = set()
    total = 0
    for p in shards:
        df = pd.read_csv(p)
        assert set(df["err_id"]) & seen == set(), f"дубль err_id в {p.name}"
        seen |= set(df["err_id"].astype(str))
        total += len(df)
    assert seen == set(anno["err_id"].astype(str)), "шарды не покрывают anno"
    # Разбивка по сплитам/типам из manifest.
    print(f"check OK: {len(full)} ошибок "
          f"(gold {manifest['n_gold']}, test_real {manifest['n_test_real']}), "
          f"шардов {len(shards)} ({total} строк), ключи sha1 сошлись.")
    print(f"Усечение: A {manifest['n_truncated_A']}, doc {manifest['n_truncated_doc']}.")
    return 0


def load_llm(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    need = {"err_id", "label", "stale_statement", "why"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{tag}: нет колонок {sorted(missing)} в {path}")
    if df["err_id"].duplicated().any():
        dup = df[df["err_id"].duplicated()]["err_id"].tolist()[:5]
        raise SystemExit(f"{tag}: дубли err_id: {dup}")
    bad = sorted(set(df["label"]) - VALID)
    if bad:
        raise SystemExit(f"{tag}: невалидные label: {bad}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="FD-26: сведение LLM-разметок.")
    ap.add_argument("--spark", default="reports/fd26_spark.csv")
    ap.add_argument("--gemini", default="reports/fd26_gemini.csv")
    ap.add_argument("--full", default=str(IN_FULL))
    ap.add_argument("--manifest", default=str(IN_MANIFEST))
    ap.add_argument("--out_combined", default=str(OUT_COMBINED))
    ap.add_argument("--out_disputes", default=str(OUT_DISPUTES))
    ap.add_argument("--check", action="store_true",
                    help="только проверка артефактов отбора (LLM-файлы не нужны)")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check_artifacts())

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    full = pd.read_csv(args.full, dtype=str).fillna("")
    spark = load_llm(Path(args.spark), "spark")
    gemini = load_llm(Path(args.gemini), "gemini")

    expected = set(manifest["err_id_to_pair_id"])
    for tag, df in (("spark", spark), ("gemini", gemini)):
        missing = expected - set(df["err_id"])
        extra = set(df["err_id"]) - expected
        if missing or extra:
            raise SystemExit(f"{tag}: missing {len(missing)} (напр. {sorted(missing)[:3]}), "
                             f"extra {len(extra)} (напр. {sorted(extra)[:3]})")

    doc_by_err = {str(r["err_id"]): str(r["doc_old"] or "") for _, r in full.iterrows()}
    for tag, df in (("spark", spark), ("gemini", gemini)):
        bad = []
        for _, r in df.iterrows():
            doc = doc_by_err[str(r["err_id"])]
            s = str(r["stale_statement"] or "")
            if r["label"] == "A":
                if not s or s not in doc:
                    bad.append(str(r["err_id"]))
            elif s:
                bad.append(str(r["err_id"]))
        if bad:
            raise SystemExit(f"{tag}: stale_statement не цитата doc_old ({len(bad)}): {bad[:5]}")

    s_map = dict(zip(spark["err_id"].astype(str), spark["label"].astype(str)))
    g_map = dict(zip(gemini["err_id"].astype(str), gemini["label"].astype(str)))
    rows = []
    for _, r in full.iterrows():
        eid = str(r["err_id"])
        s, g = s_map[eid], g_map[eid]
        consensus = s if s == g else "dispute"
        rows.append({"err_id": eid, "pair_id": str(r["pair_id"]), "split": str(r["split"]),
                     "error_type": str(r["error_type"]), "dataset_stale": str(r["label_stale"]),
                     "m2_pred": str(r["m2_pred"]), "sig_type": str(r["sig_type"]),
                     "drift_type": str(r["drift_type"]), "truncated": str(r["truncated"]),
                     "doc_truncated": str(r["doc_truncated"]),
                     "spark": s, "gemini": g, "consensus": consensus})
    combined = pd.DataFrame(rows)
    disputes = combined[combined["consensus"] == "dispute"].copy()
    # Для ручного разбора приложить тексты из full.
    text_cols = ["project", "symbol", "sig_old", "diff_code", "doc_old"]
    full_text = full[["err_id"] + text_cols].copy()
    disputes = disputes.merge(full_text, on="err_id", how="left")

    Path(args.out_combined).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_combined, index=False)
    disputes.to_csv(args.out_disputes, index=False)

    # Каппа spark-vs-gemini.
    try:
        from sklearn.metrics import cohen_kappa_score
        s4 = combined["spark"].tolist()
        g4 = combined["gemini"].tolist()
        k4 = cohen_kappa_score(s4, g4)
        bmap = {"A": 1, "B": 1, "C": 0, "D": -1}
        pairs = [(bmap[s], bmap[g]) for s, g in zip(s4, g4) if s != "D" and g != "D"]
        if pairs:
            kb = cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs])
        else:
            kb = float("nan")
        print(f"spark-vs-gemini: n={len(combined)}, совпало {(combined['spark'] == combined['gemini']).sum()}, "
              f"kappa4={k4:.3f}, kappa_bin={kb:.3f} (D исключены: {len(s4) - len(pairs)})")
    except ImportError:
        print("sklearn нет — каппа не посчитана.")
    print(f"combined: {args.out_combined} ({len(combined)} строк)")
    print(f"disputes: {args.out_disputes} ({len(disputes)} строк) — вручную разбирает человек.")


if __name__ == "__main__":
    raise SystemExit(main())
