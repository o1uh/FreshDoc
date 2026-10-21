from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import REAL_SOURCES
from src.models.metrics import (
    EVAL_GROUPS,
    EVAL_SPLITS,
    evaluate_binary,
    evaluate_drift,
    group_mask,
    pick_threshold_f1,
    pr_curve_points,
)

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUT_METRICS = PROJECT_ROOT / "reports" / "metrics.json"
OUT_TABLE = PROJECT_ROOT / "reports" / "tables" / "e1.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "e1"

TRUTH_COLS = ["pair_id", "label_stale", "split", "sig_type", "drift_type", "source"]


def _parse_pred_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"--pred ожидает NAME=PATH, получено: {arg}")
    name, path = arg.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Пустое имя модели в --pred: {arg}")
    return name, Path(path)


def _load_pred(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Предсказания отсутствуют: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    need = {"pair_id", "p_stale"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: нет колонок {sorted(missing)}")
    if df["pair_id"].duplicated().any():
        n = int(df["pair_id"].duplicated().sum())
        raise ValueError(f"{path}: дубли pair_id: {n}")
    bad = pd.to_numeric(df["p_stale"], errors="coerce")
    if bad.isna().any():
        raise ValueError(f"{path}: p_stale вне [0,1] или не число: {int(bad.isna().sum())}")
    if ((bad < 0.0) | (bad > 1.0)).any():
        raise ValueError(f"{path}: p_stale вне [0,1]")
    df = df.copy()
    df["p_stale"] = bad.astype(float)
    if "type_pred" in df.columns:
        df["type_pred"] = df["type_pred"].astype(object).fillna("unknown").astype(str)
    return df


def _val_real_mask(truth: pd.DataFrame) -> np.ndarray:
    """Реальные строки val для подбора порога (R11, FD-19 шаг 2)."""
    m = truth["split"].eq("val").to_numpy()
    if "source" in truth.columns:
        m = m & truth["source"].isin(list(REAL_SOURCES)).to_numpy()
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-19: единая оценка моделей (R11).")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--pred", action="append", default=[],
                    help="NAME=PATH (CSV/parquet: pair_id,p_stale[,type_pred]). Повторить для каждой модели.")
    ap.add_argument("--out_metrics", default=str(OUT_METRICS))
    ap.add_argument("--out_table", default=str(OUT_TABLE))
    ap.add_argument("--figdir", default=str(FIG_DIR))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.pred:
        print("Нужно хотя бы одно --pred NAME=PATH")
        return 2
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"Датасет отсутствует: {ds_path}")
        return 2
    truth = pd.read_parquet(ds_path)
    missing = [c for c in TRUTH_COLS if c not in truth.columns]
    if missing:
        print(f"В датасете нет колонок: {missing}")
        return 2

    models: dict[str, pd.DataFrame] = {}
    for raw in args.pred:
        name, path = _parse_pred_arg(raw)
        if name in models:
            print(f"Дублирующееся имя модели: {name}")
            return 2
        models[name] = _load_pred(path)
    print(f"Датасет: {ds_path} ({len(truth)} строк)")
    print(f"Модели: {sorted(models)}")

    metrics: dict = {
        "dataset": str(ds_path),
        "dataset_rows": int(len(truth)),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "threshold_from": "val real rows (max F1), per model",
        "splits_separate_no_averaging": True,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    flat_rows: list[dict] = []
    # PR-точки для графиков: curves[(split, group)][model] = points
    curves: dict[tuple[str, str], dict[str, dict]] = {}

    for name, pred in models.items():
        merged = truth.merge(pred, on="pair_id", how="inner")
        n_missing = int(truth["pair_id"].nunique() - merged["pair_id"].nunique())
        extra = sorted(set(pred["pair_id"].astype(str)) - set(truth["pair_id"].astype(str)))
        print(f"[{name}] пересечение: {len(merged)}/{len(truth)} строк, "
              f"нет в предсказаниях: {n_missing}, лишних: {len(extra)}")
        has_type = "type_pred" in pred.columns

        # Порог — только на реальных строках val.
        vm = _val_real_mask(merged)
        yv = merged.loc[vm, "label_stale"].to_numpy(dtype=int)
        sv = merged.loc[vm, "p_stale"].to_numpy(dtype=float)
        if len(yv) == 0:
            print(f"[{name}] ВНИМАНИЕ: строк val в предсказаниях нет — порог 0.5 по умолчанию")
            thr = 0.5
        else:
            thr = pick_threshold_f1(yv, sv)
        print(f"[{name}] порог по val (n={len(yv)}, pos={int(yv.sum()) if len(yv) else 0}): {thr:.4f}")

        model_entry: dict = {"threshold_val": float(thr),
                             "n_overlap": int(len(merged)),
                             "n_missing": n_missing,
                             "has_type_pred": bool(has_type),
                             "splits": {}}
        for split in EVAL_SPLITS:
            sub = merged.loc[merged["split"] == split]
            model_entry["splits"][split] = {}
            for group in EVAL_GROUPS:
                m = group_mask(sub["sig_type"].to_numpy(), group)
                g = sub.loc[m]
                yt = g["label_stale"].to_numpy(dtype=int)
                ys = g["p_stale"].to_numpy(dtype=float)
                ev = evaluate_binary(yt, ys, threshold=thr,
                                     n_boot=args.n_boot, seed=args.seed)
                is_subset = 0 < len(sub) < len(truth.loc[truth["split"] == split])
                split_label = f"{split}_subset" if is_subset else split
                ev["split"], ev["group"], ev["model"] = split_label, group, name
                ev["threshold"] = float(thr)
                if has_type and len(g):
                    d = evaluate_drift(g["drift_type"].to_numpy(),
                                       g["type_pred"].fillna("unknown").to_numpy(),
                                       g["label_stale"].to_numpy(dtype=int))
                    ev.update(d)
                model_entry["splits"][split][group] = ev
                flat_rows.append(ev)
                key = (split, group)
                curves.setdefault(key, {})[name] = pr_curve_points(yt, ys)
                print(f"  [{name}] {split}/{group}: n={ev['n']} pos={ev['n_pos']} "
                      f"prev={ev['prevalence']:.3f} PR-AUC={ev['pr_auc']:.3f} "
                      f"[{ev['pr_auc_lo']:.3f},{ev['pr_auc_hi']:.3f}] F1={ev['f1']:.3f}")
        metrics["models"][name] = model_entry

    # JSON-отчёт.
    out_m = Path(args.out_metrics)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Метрики: {out_m}")

    # Плоская таблица e1.csv (по строке на model x split x group, без усреднений).
    flat = pd.DataFrame(flat_rows)
    col_order = ["model", "split", "group", "n", "n_pos", "prevalence",
                 "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc",
                 "f1", "precision", "recall", "recall_at_p80",
                 "brier", "ece", "threshold",
                 "n_drift", "drift_acc", "drift_macro_f1"]
    for c in col_order:
        if c not in flat.columns:
            flat[c] = np.nan
    flat = flat[col_order].sort_values(["model", "split", "group"]).reset_index(drop=True)
    out_t = Path(args.out_table)
    out_t.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(out_t, index=False)
    print(f"Таблица: {out_t} ({len(flat)} строк)")

    # PR-кривые: все модели на одном графике для каждого сплита и группы.
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    for (split, group), per_model in sorted(curves.items()):
        plt.figure()
        n_lines = 0
        for name in sorted(per_model):
            pts = per_model[name]
            if not pts["recall"]:
                continue
            plt.step(pts["recall"], pts["precision"], where="post", label=name)
            n_lines += 1
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR: {split}/{group}")
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        if n_lines == 0:
            plt.text(0.5, 0.5, "single class: no PR curve",
                     ha="center", va="center", transform=plt.gca().transAxes)
        else:
            plt.legend()
        fig = figdir / f"pr_{split}_{group}.png"
        plt.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"График: {fig}")

    print("Готово: сплиты посчитаны по отдельности, усреднений нет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
