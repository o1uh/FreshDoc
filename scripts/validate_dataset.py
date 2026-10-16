"""FD-13: локальная команда проверки dataset_v1 (R8).

    python scripts/validate_dataset.py [--in PATH] [--pools PATH] [--out PATH]

Проверки — в src/data/validate.py. Код выхода 0, если все зелёные, иначе 1.
Человекочитаемый отчёт — в stdout, машиночитаемый — в JSON (--out).
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

from src.data.validate import all_ok, run_all_checks

IN_DATA = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
IN_POOLS = PROJECT_ROOT / "data" / "interim" / "project_pools.parquet"
OUT_REPORT = PROJECT_ROOT / "reports" / "dataset_v1.checks.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-13: проверки качества dataset_v1 (R8).")
    ap.add_argument("--in", dest="inp", default=str(IN_DATA),
                    help="проверяемый parquet датасета")
    ap.add_argument("--pools", default=str(IN_POOLS),
                    help="project_pools.parquet для проверки пула синтетики")
    ap.add_argument("--no-pools", action="store_true",
                    help="пропустить проверку пула синтетики")
    ap.add_argument("--out", default=str(OUT_REPORT),
                    help="куда записать JSON-отчёт")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"Вход отсутствует: {inp}")
        return 2
    df = pd.read_parquet(inp)
    print(f"Строк: {len(df)}, колонок: {len(df.columns)}")

    pool_of = None
    if not args.no_pools:
        pools_path = Path(args.pools)
        if pools_path.exists():
            df_pools = pd.read_parquet(pools_path)
            pool_of = dict(zip(df_pools["project"].astype(str),
                               df_pools["pool"].astype(str)))
            print(f"Пулов: {len(pool_of)}")
        else:
            print(f"Пулы отсутствуют ({pools_path}) — проверка пула пропущена")

    results = run_all_checks(df, pool_of)
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"[{mark}] {r.name}: {r.detail}")

    ok = all_ok(results)
    report = {
        "dataset": inp.name,
        "rows": int(len(df)),
        "by_split": df["split"].value_counts().to_dict() if "split" in df.columns else {},
        "ok": ok,
        "checks": [{"name": r.name, "ok": r.ok, "detail": r.detail, "n_fail": r.n_fail}
                   for r in results],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {out_path}")
    print("ВСЕ ПРОВЕРКИ ЗЕЛЁНЫЕ" if ok else "ЕСТЬ КРАСНЫЕ ПРОВЕРКИ")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
