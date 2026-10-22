"""FD-26: 95% бутстрэп-CI долей причин (1000 повторов, сид 42) по сплитам.

    python scripts/fd26_cause_ci.py [--combined ...] [--out ...] [--n_boot 1000] [--seed 42]

Читает колонку cause из combined, ресемплирует строки каждого сплита
с возвратом, считает перцентильные интервалы [2.5, 97.5] для доли каждой
причины. Набор категорий — единый (объединение по сплитам), отсутствующим
ставится count 0. Выход — JSON.
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

IN_COMBINED = PROJECT_ROOT / "reports" / "fd26_combined.csv"
OUT_JSON = PROJECT_ROOT / "reports" / "tables" / "fd26_cause_ci.json"


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description="FD-26: бутстрэп-CI долей причин.")
    ap.add_argument("--combined", default=str(IN_COMBINED))
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.combined, dtype=str, keep_default_na=False)
    if (df["cause"] == "").any():
        bad = df[df["cause"] == ""]["err_id"].tolist()[:5]
        raise SystemExit(f"пустые cause (нужен --human при сведении): {bad}")
    cats = sorted(set(df["cause"]))
    rng = np.random.default_rng(args.seed)
    out: dict = {"n_boot": args.n_boot, "seed": args.seed,
                 "causes": cats, "splits": {}}
    for split, gr in df.groupby("split"):
        arr = gr["cause"].to_numpy()
        n = len(arr)
        idx = rng.integers(0, n, size=(args.n_boot, n))
        counts: dict[str, list[float]] = {c: [] for c in cats}
        for rep in idx:
            sample = arr[rep]
            for c in cats:
                counts[c].append(float((sample == c).mean()))
        out["splits"][str(split)] = {
            "n": n,
            "causes": {c: {"share": round(float((arr == c).mean()), 4),
                           "ci_lo": round(float(np.percentile(counts[c], 2.5)), 4),
                           "ci_hi": round(float(np.percentile(counts[c], 97.5)), 4),
                           "count": int((arr == c).sum())}
                       for c in cats},
        }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for split, info in out["splits"].items():
        print(f"{split} (n={info['n']}):")
        for c, v in sorted(info["causes"].items()):
            print(f"  {c}: {v['count']} ({v['share']:.3f} [{v['ci_lo']:.3f},{v['ci_hi']:.3f}])")
    print(f"отчёт: {out_path}")


if __name__ == "__main__":
    main()
