from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.evaluate import _load_pred, _parse_pred_arg
from src.models.metrics import group_mask, evaluate_binary, evaluate_drift

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
IN_M3_SELECT = PROJECT_ROOT / "reports" / "tables" / "m3_select.json"
IN_METRICS = PROJECT_ROOT / "reports" / "metrics.json"
OUT_TABLE = PROJECT_ROOT / "reports" / "tables" / "e1_m3subset.csv"

DEFAULT_PRED_PATHS: dict[str, Path] = {
    "M0": PROJECT_ROOT / "reports" / "preds" / "m0.csv",
    "M1": PROJECT_ROOT / "reports" / "preds" / "m1.csv",
    "M2": PROJECT_ROOT / "reports" / "preds" / "m2.csv",
    "M3": PROJECT_ROOT / "reports" / "preds" / "m3.csv",
    "M3max": PROJECT_ROOT / "reports" / "preds" / "m3_max.csv",
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "M0": 1.0,
    "M1": 0.4978702999652788,
    "M2": 0.9897632598876952,
    "M3": 0.5,
    "M3max": 0.5,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка моделей на M3-подмножестве (300 строк test_real).")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--select", default=str(IN_M3_SELECT))
    ap.add_argument("--metrics_json", default=str(IN_METRICS),
                    help="Путь к reports/metrics.json для чтения точных порогов val.")
    ap.add_argument("--pred", action="append", default=[],
                    help="NAME=PATH для переопределения моделей (по умолчанию M0, M1, M2, M3, M3max).")
    ap.add_argument("--out", default=str(OUT_TABLE))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    ds = pd.read_parquet(args.dataset)
    with open(args.select, "r", encoding="utf-8") as f:
        select_data = json.load(f)
    all_selected_ids = set(select_data["m3_id_to_pair_id"].values())
    sub_ds = ds[(ds["split"] == "test_real") & (ds["pair_id"].isin(all_selected_ids))].copy()
    print(f"M3 подмножество test_real: {len(sub_ds)} строк (положительных: {int(sub_ds['label_stale'].sum())})")

    # Чтение точных порогов из metrics.json с дефолтным фолбэком (R11)
    thresholds = dict(DEFAULT_THRESHOLDS)
    metrics_path = Path(args.metrics_json)
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            for m_name, m_info in metrics_data.get("models", {}).items():
                if "threshold_val" in m_info and m_info["threshold_val"] is not None:
                    thresholds[m_name] = float(m_info["threshold_val"])
        except Exception as exc:
            print(f"Предупреждение: не удалось прочитать {metrics_path}: {exc}")

    # Модели для оценки: из CLI --pred или стандартный набор
    if args.pred:
        models_to_eval: list[tuple[str, Path, float]] = []
        for raw in args.pred:
            name, path = _parse_pred_arg(raw)
            thr = thresholds.get(name, 0.5)
            models_to_eval.append((name, path, thr))
    else:
        models_to_eval = [
            (name, path, thresholds.get(name, 0.5))
            for name, path in DEFAULT_PRED_PATHS.items()
        ]

    flat_rows: list[dict] = []
    for name, pred_path, thr in models_to_eval:
        if not pred_path.exists():
            print(f"[{name}] Файл предсказаний не найден: {pred_path}, пропуск")
            continue
        pred = _load_pred(pred_path)
        merged = sub_ds.merge(pred, on="pair_id", how="inner")
        has_type = "type_pred" in pred.columns

        for group in ["all", "sig_changed", "sig_same"]:
            m = group_mask(merged["sig_type"].to_numpy(), group)
            g = merged.loc[m]
            yt = g["label_stale"].to_numpy(dtype=int)
            ys = g["p_stale"].to_numpy(dtype=float)
            ev = evaluate_binary(yt, ys, threshold=thr, n_boot=args.n_boot, seed=args.seed)
            ev["model"] = name
            ev["split"] = "test_real_subset300"
            ev["group"] = group
            ev["threshold"] = float(thr)
            if has_type and len(g):
                d = evaluate_drift(
                    g["drift_type"].to_numpy(),
                    g["type_pred"].fillna("unknown").to_numpy(),
                    g["label_stale"].to_numpy(dtype=int),
                )
                ev.update(d)
            flat_rows.append(ev)
            print(
                f"  [{name}] subset/{group}: n={ev['n']} pos={ev['n_pos']} "
                f"PR-AUC={ev['pr_auc']:.3f} [{ev['pr_auc_lo']:.3f},{ev['pr_auc_hi']:.3f}] F1={ev['f1']:.3f}"
            )

    out_df = pd.DataFrame(flat_rows)
    col_order = [
        "model", "split", "group", "n", "n_pos", "prevalence",
        "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc",
        "f1", "precision", "recall", "recall_at_p80",
        "brier", "ece", "threshold",
        "n_drift", "drift_acc", "drift_macro_f1"
    ]
    for c in col_order:
        if c not in out_df.columns:
            out_df[c] = float("nan")
    out_df = out_df[col_order].sort_values(["model", "group"]).reset_index(drop=True)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)
    print(f"Сохранено: {out_p} ({len(out_df)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
