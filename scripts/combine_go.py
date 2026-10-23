"""FD-39 фаза 2: сведение двух разметок полного Go-набора, согласие и арбитраж.

    python scripts/combine_go.py --spark reports/go_full381_spark.csv --gemini reports/go_full381_gemini.csv
    python scripts/combine_go.py --check  # только проверка артефактов (без LLM-файлов)
    python scripts/combine_go.py --export-blind  # строгий blind-файл споров (без таргетов)
    python scripts/combine_go.py --spark ... --gemini ... --human reports/go_full381_human.csv

Входы LLM: id,label,stale_statement,why (label в A/B/C/D, шкала R9).
Проверки: покрытие 381/381 без дублей; label валидны; для A stale_statement —
точная подстрока doc_old, для не-A пуст.
Выходы: reports/go_full381_combined.csv (id, spark, gemini, consensus, human, final),
reports/go_full381_disputes.csv (споры + тексты для ручного арбитража),
reports/go_full381_disputes_blind.csv (--export-blind: строго слепой файл споров:
id,file,symbol,sig_old,diff_code,doc_old + пустые label/stale_statement/why;
без прокси/меток/скоров),
каппа Коэна (4-класс и бинарная A/B→1, C→0; строки с D из бинарной исключаются
со счётчиком и WARNING).
Порог согласия >= 0.7 — критерий приёмки, иначе ненулевой exit-код.
Доля D у каждого судьи должна быть < 10%, иначе WARNING.
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

from src.fd26util import contains_quote

PILOT = PROJECT_ROOT / "data" / "processed" / "go_full381.csv"
BATCHES = PROJECT_ROOT / "reports" / "go_batches_full"
OUT_COMBINED = PROJECT_ROOT / "reports" / "go_full381_combined.csv"
OUT_DISPUTES = PROJECT_ROOT / "reports" / "go_full381_disputes.csv"
OUT_BLIND = PROJECT_ROOT / "reports" / "go_full381_disputes_blind.csv"

VALID = ("A", "B", "C", "D")
AGREE_GATE = 0.7
D_WARN_FRAC = 0.10


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_llm(path: Path, tag: str) -> pd.DataFrame:
    df = read_csv(path)
    need = {"id", "label", "stale_statement", "why"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{tag}: нет колонок {sorted(missing)} в {path}")
    if df["id"].duplicated().any():
        raise SystemExit(f"{tag}: дубли id")
    bad = sorted(set(df["label"]) - set(VALID))
    if bad:
        raise SystemExit(f"{tag}: невалидные label: {bad}")
    return df


def check_artifacts(pilot_p: Path = PILOT, shards_dir: Path = BATCHES) -> int:
    pilot = read_csv(pilot_p)
    assert pilot["id"].is_unique, "дубли id в наборе"
    assert (pilot["label"].str.strip() == "").all(), "набор не слепой: label заполнены"
    assert (pilot["doc_old"].str.strip() != "").all(), "пустые doc_old в наборе"
    assert (pilot["diff_code"].str.strip() != "").all(), "пустые diff в наборе"
    batches = sorted(shards_dir.glob("go_batch*.prompt.txt"))
    assert batches, f"нет батчей в {shards_dir}"
    n = sum(1 for p in batches for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.startswith("=== CASE id="))
    assert n == len(pilot), f"батчи покрывают {n} кейсов, ждали {len(pilot)}"
    print(f"check OK: набор {len(pilot)}/{len(pilot)} слепой, батчей {len(batches)}, кейсов в батчах {n}")
    return 0


BLIND_COLS = ["id", "file", "symbol", "sig_old", "diff_code", "doc_old"]


def load_human(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    need = {"id", "label"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"human: нет колонок {sorted(missing)} в {path}")
    if df["id"].duplicated().any():
        raise SystemExit("human: дубли id")
    bad = sorted(set(df["label"].astype(str)) - set(VALID))
    if bad:
        raise SystemExit(f"human: невалидные label: {bad}")
    return df[["id", "label"]].copy()


def cmd_export_blind(args: argparse.Namespace) -> None:
    disputes = read_csv(Path(args.out_disputes))
    missing = [c for c in BLIND_COLS if c not in disputes.columns]
    if missing:
        raise SystemExit(f"export-blind: в disputes нет колонок {missing} "
                         f"(сначала сведение без --export-blind)")
    blind = disputes[BLIND_COLS].copy()
    blind = blind.sort_values("id").reset_index(drop=True)
    blind["label"] = ""
    blind["stale_statement"] = ""
    blind["why"] = ""
    out = Path(args.out_blind)
    out.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(out, index=False)
    print(f"blind: {out} ({len(blind)} строк, без прокси/меток/скоров)")


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-39: сведение разметок полного Go-набора.")
    ap.add_argument("--spark", default=str(PROJECT_ROOT / "reports" / "go_full381_spark.csv"))
    ap.add_argument("--gemini", default=str(PROJECT_ROOT / "reports" / "go_full381_gemini.csv"))
    ap.add_argument("--input", "--pilot", dest="input", default=str(PILOT),
                    help="входной CSV набора (слепой). --pilot — deprecated алиас.")
    ap.add_argument("--shards_dir", default=str(BATCHES))
    ap.add_argument("--out_combined", default=str(OUT_COMBINED))
    ap.add_argument("--out_disputes", default=str(OUT_DISPUTES))
    ap.add_argument("--out_blind", default=str(OUT_BLIND))
    ap.add_argument("--human", default=None,
                    help="файл ручного арбитража споров (id,label) — "
                         "метки принимаются ТОЛЬКО для строк со статусом dispute; "
                         "final = human для споров, consensus для согласия")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--export-blind", dest="export_blind", action="store_true",
                    help="только генерация строгого blind-файла споров из disputes")
    args = ap.parse_args()
    if args.check:
        return check_artifacts(Path(args.input), Path(args.shards_dir))
    if args.export_blind:
        cmd_export_blind(args)
        return 0

    pilot = read_csv(Path(args.input))
    expected = set(pilot["id"].astype(str))
    spark = load_llm(Path(args.spark), "spark")
    gemini = load_llm(Path(args.gemini), "gemini")
    for tag, df in (("spark", spark), ("gemini", gemini)):
        missing = expected - set(df["id"])
        extra = set(df["id"]) - expected
        if missing or extra:
            raise SystemExit(f"{tag}: missing {len(missing)}, extra {len(extra)}")

    doc_by_id = {str(r["id"]): str(r["doc_old"]) for _, r in pilot.iterrows()}
    for tag, df in (("spark", spark), ("gemini", gemini)):
        bad = []
        for _, r in df.iterrows():
            doc = doc_by_id[str(r["id"])]
            if r["label"] == "A":
                if not contains_quote(doc, r["stale_statement"]):
                    bad.append(str(r["id"]))
            elif str(r["stale_statement"]).strip():
                bad.append(str(r["id"]))
        if bad:
            raise SystemExit(f"{tag}: stale_statement не цитата doc_old: {bad[:5]}")

    s_map = dict(zip(spark["id"].astype(str), spark["label"].astype(str)))
    g_map = dict(zip(gemini["id"].astype(str), gemini["label"].astype(str)))
    human_map: dict[str, str] = {}
    if args.human is not None:
        hum = load_human(Path(args.human))
        # Арбитраж — только споры: метки вне dispute-строк отклоняются.
        dispute_ids = {eid for eid in s_map if s_map[eid] != g_map.get(eid)}
        outside = sorted(set(hum["id"].astype(str)) - dispute_ids)
        if outside:
            raise SystemExit(f"human: метки вне споров ({len(outside)}): {outside[:5]} "
                             f"(арбитраж только для consensus==dispute)")
        human_map = dict(zip(hum["id"].astype(str), hum["label"].astype(str)))
    text_cols = ["file", "symbol", "sig_old", "diff_code", "doc_old"]
    rows = []
    for _, r in pilot.iterrows():
        eid = str(r["id"])
        s, g = s_map[eid], g_map[eid]
        consensus = s if s == g else "dispute"
        hum = human_map.get(eid, "")
        fin = hum if hum else ("" if consensus == "dispute" else consensus)
        rows.append({"id": eid, "spark": s, "gemini": g,
                     "consensus": consensus, "human": hum, "final": fin,
                     **{c: str(r[c]) for c in text_cols}})
    combined = pd.DataFrame(rows)
    disputes = combined[combined["consensus"] == "dispute"].copy()
    Path(args.out_combined).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_combined, index=False)
    disputes.to_csv(args.out_disputes, index=False)

    agree = float((combined["spark"] == combined["gemini"]).mean())
    n_d_spark = int((combined["spark"] == "D").sum())
    n_d_gemini = int((combined["gemini"] == "D").sum())
    for tag, n_d in (("spark", n_d_spark), ("gemini", n_d_gemini)):
        frac = n_d / max(len(combined), 1)
        if frac >= D_WARN_FRAC:
            print(f"WARNING: доля D у {tag} {frac:.3f} ({n_d}/{len(combined)}) "
                  f">= {D_WARN_FRAC} — проверить инструкции/батчи")
        else:
            print(f"доля D у {tag}: {frac:.3f} ({n_d}/{len(combined)})")
    try:
        from sklearn.metrics import cohen_kappa_score
        k4 = cohen_kappa_score(combined["spark"].tolist(),
                              combined["gemini"].tolist(), labels=list(VALID))
        bmap = {"A": 1, "B": 1, "C": 0}
        n_drop_d = int(((~combined["spark"].isin(bmap)) |
                        (~combined["gemini"].isin(bmap))).sum())
        if n_drop_d:
            print(f"WARNING: kappa_bin исключает {n_drop_d} строк с D "
                  f"(бинарная шкала A/B→1, C→0; D — не консенсус)")
        pairs = [(bmap[s], bmap[g]) for s, g in
                 zip(combined["spark"], combined["gemini"]) if s in bmap and g in bmap]
        kb = (cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs], labels=[0, 1])
              if pairs else float("nan"))
        print(f"spark-vs-gemini: n={len(combined)}, согласие {agree:.3f}, "
              f"kappa4={k4:.3f}, kappa_bin={kb:.3f}")
    except ImportError:
        k4, kb = float("nan"), float("nan")
        print(f"spark-vs-gemini: n={len(combined)}, согласие {agree:.3f} (sklearn нет)")

    gate = "PASS" if agree >= AGREE_GATE else "FAIL"
    print(f"disputes: {args.out_disputes} ({len(disputes)} строк) — вручную разбирает человек")
    if args.human is not None:
        n_fin = int((combined["final"] != "").sum())
        print(f"human: {args.human} ({len(human_map)} меток только для споров), "
              f"final заполнен: {n_fin}/{len(combined)}")
    print(f"согласие >= {AGREE_GATE} — критерий приёмки: {gate}")
    if gate == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
