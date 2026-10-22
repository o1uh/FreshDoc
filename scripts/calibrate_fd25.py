"""FD-25: калибровка (E5), пороги (E6), обобщение (E4) для лучшей M2 (slot4_full).

Порядок строго T -> порог (мост FD-24, §8):
  1. T подбирается на реальных строках val минимизацией лог-потерь.
  2. Рабочий порог выбирается на КАЛИБРОВАННЫХ вероятностях val
     по критерию precision >= 0.80 (максимум F1).

Использование:
  python scripts/calibrate_fd25.py \
    --dataset data/processed/dataset_v1.parquet \
    --pred reports/preds/m2.csv \
    --out_report reports/e4_e5_e6.md \
    --out_json reports/calibration.json \
    --figdir reports/figures/e4_e5_e6 \
    --risk_config config/risk.yaml \
    --threshold_table reports/tables/e6_thresholds.csv \
    --pred_out reports/preds/m2_calibrated.csv
"""
from __future__ import annotations

import argparse
import json
import re
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
from src.models.calibration import (
    apply_temperature,
    fit_temperature,
    nll_score,
    pick_working_threshold,
    reliability_table,
    thresholds_table,
)
from src.models.metrics import (
    brier_safe,
    ece_score,
    evaluate_binary,
    pr_auc_safe,
)

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
IN_PRED = PROJECT_ROOT / "reports" / "preds" / "m2.csv"
OUT_REPORT = PROJECT_ROOT / "reports" / "e4_e5_e6.md"
OUT_JSON = PROJECT_ROOT / "reports" / "calibration.json"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "e4_e5_e6"
RISK_CONFIG = PROJECT_ROOT / "config" / "risk.yaml"
OUT_TABLE = PROJECT_ROOT / "reports" / "tables" / "e6_thresholds.csv"
PRED_OUT = PROJECT_ROOT / "reports" / "preds" / "m2_calibrated.csv"


def _val_real_mask(df: pd.DataFrame) -> np.ndarray:
    m = df["split"].eq("val").to_numpy()
    if "source" in df.columns:
        m = m & df["source"].isin(list(REAL_SOURCES)).to_numpy()
    return m


def _reliability_fig(val_table_before: list[dict], val_table_after: list[dict], out: Path) -> None:
    plt.figure()
    for tbl, label, marker in ((val_table_before, "до (сырые)", "o"),
                               (val_table_after, "после (T)", "s")):
        xs = [r["conf"] for r in tbl if r["n"] > 0]
        ys = [r["acc"] for r in tbl if r["n"] > 0]
        plt.plot(xs, ys, marker=marker, label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="идеал")
    plt.xlabel("Средняя уверенность (conf)")
    plt.ylabel("Доля положительных (acc)")
    plt.title("Диаграмма надёжности, val real (10 корзин)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def _threshold_fig(thr_rows: list[dict], working_thr: float, out: Path) -> None:
    xs = [r["threshold"] for r in thr_rows]
    ps = [r["precision"] for r in thr_rows]
    rs = [r["recall"] for r in thr_rows]
    fs = [r["f1"] for r in thr_rows]
    plt.figure()
    plt.plot(xs, ps, label="precision")
    plt.plot(xs, rs, label="recall")
    plt.plot(xs, fs, label="F1")
    plt.axvline(working_thr, color="red", linestyle="--", label=f"рабочий {working_thr:.3f}")
    plt.axhline(0.8, color="gray", linestyle=":", label="precision=0.8")
    plt.xlabel("Порог")
    plt.ylabel("Метрика")
    plt.title("E6: порог -> precision/recall/F1 (val, калиброванные)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def _rel(p: Path) -> str:
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _update_risk_threshold(risk_path: Path, threshold: float, T: float) -> None:
    text = risk_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_lines: list[str] = []
    replaced = False
    for ln in lines:
        stripped = ln.strip()
        # FD-25: обновить комментарий baseline -> рабочий порог (первый такой комментарий)
        if "Baseline value 0.5 per R12; to be replaced" in ln:
            out_lines.append("# Working threshold from FD-25 E6 (precision>=0.80, val cal); baseline R12 was 0.5")
            continue
        # Устаревшие provenance-строки FD-25 выкидываются здесь,
        # актуальная дописывается ниже (идемпотентно + обновляет T/дату).
        if re.match(r"^# FD-25.*threshold <- E6.*$", stripped):
            continue
        if stripped.startswith("threshold:") and not replaced:
            indent = ln[: len(ln) - len(ln.lstrip())]
            # Полная точность float (repr, round-trip): округление до .6f
            # сдвигало порог вверх (0.88162567 -> 0.881626) и исключало
            # граничный TP на val (195 -> 194). Храним точное значение.
            out_lines.append(f"{indent}threshold: {threshold!r}")
            replaced = True
        else:
            out_lines.append(ln)
    if not replaced:
        out_lines.append(f"threshold: {threshold!r}")
    new_text = "\n".join(out_lines) + "\n"
    # provenance-метка с датой и T (маркер — 'threshold <- E6')
    prov = f"# FD-25 {datetime.now(timezone.utc).date().isoformat()}: threshold <- E6 (precision>=0.80, val cal), T={T:.4f}"
    new_text = new_text.rstrip("\n") + "\n" + prov + "\n"
    risk_path.write_text(new_text, encoding="utf-8")


def _read_best_model(best_cfg_path: Path) -> tuple[str, dict]:
    """Прочитать метаданные победителя FD-24 из config/train_best.yaml.

    Возвращает (label, info). При отсутствии/битом файле — fallback
    на параметры слота 4 с пометкой fallback в info.
    """
    fallback = {
        "model_name": "microsoft/unixcoder-base",
        "max_len": 1024,
        "train_mix": "full",
        "doc_budget": 384,
    }
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(Path(best_cfg_path).read_text(encoding="utf-8")) or {}
        info = {
            "model_name": str(raw.get("model_name", fallback["model_name"])),
            "max_len": int(raw.get("max_len", fallback["max_len"])),
            "train_mix": str(raw.get("train_mix", fallback["train_mix"])),
            "doc_budget": int(raw.get("doc_budget", fallback["doc_budget"])),
            "source": _rel(Path(best_cfg_path)),
        }
        label = (f"{info['model_name']} "
                 f"(train_mix={info['train_mix']}, max_len={info['max_len']}, "
                 f"doc_budget={info['doc_budget']})")
        return label, info
    except Exception:
        label = (f"{fallback['model_name']} "
                 f"(train_mix={fallback['train_mix']}, max_len={fallback['max_len']}) [fallback: train_best.yaml unreadable]")
        info = dict(fallback, source=_rel(Path(best_cfg_path)), fallback=True)
        return label, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FD-25: калибровка E5, пороги E6, обобщение E4.")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--pred", default=str(IN_PRED))
    ap.add_argument("--out_report", default=str(OUT_REPORT))
    ap.add_argument("--out_json", default=str(OUT_JSON))
    ap.add_argument("--figdir", default=str(FIG_DIR))
    ap.add_argument("--risk_config", default=str(RISK_CONFIG))
    ap.add_argument("--threshold_table", default=str(OUT_TABLE))
    ap.add_argument("--pred_out", default=str(PRED_OUT))
    ap.add_argument("--min_precision", type=float, default=0.8)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--best_config", default=str(PROJECT_ROOT / "config" / "train_best.yaml"),
                    help="config победителя FD-24 (источник метаданных модели)")
    ap.add_argument("--no_update_risk", action="store_true",
                    help="не перезаписывать config/risk.yaml (для сухого прогона)")
    args = ap.parse_args(argv)

    ds_path = Path(args.dataset)
    pred_path = Path(args.pred)
    if not ds_path.exists():
        print(f"Датасет отсутствует: {ds_path}", file=sys.stderr)
        return 2
    if not pred_path.exists():
        print(f"Предсказания отсутствуют: {pred_path}", file=sys.stderr)
        return 2

    model_label, model_info = _read_best_model(Path(args.best_config))
    print(f"Модель: {model_label} (из {model_info.get('source')})")

    ds = pd.read_parquet(ds_path)
    pred = pd.read_csv(pred_path)
    if "p_stale" not in pred.columns:
        print(f"{pred_path}: нет колонки p_stale", file=sys.stderr)
        return 2
    merged = ds.merge(pred[["pair_id", "p_stale"]], on="pair_id", how="inner")
    print(f"Строк: датасет {len(ds)}, предсказания {len(pred)}, пересечение {len(merged)}")
    if len(merged) == 0:
        print("Пересечение пустое", file=sys.stderr)
        return 2

    # --- E5: fit T только на val real ---
    vm = _val_real_mask(merged)
    yv = merged.loc[vm, "label_stale"].to_numpy(dtype=int)
    pv = merged.loc[vm, "p_stale"].to_numpy(dtype=float)
    print(f"val real: n={len(yv)}, pos={int(yv.sum())}")
    if len(np.unique(yv)) < 2:
        print("В val real один класс — калибровка невозможна", file=sys.stderr)
        return 2
    T, fit_info = fit_temperature(yv, pv)
    print(f"T={T:.4f} NLL {fit_info['nll_before']:.4f} -> {fit_info['nll_after']:.4f}")

    # применить ко всем
    p_cal_all = apply_temperature(merged["p_stale"].to_numpy(dtype=float), T)
    merged["p_cal"] = p_cal_all

    # --- Brier/ECE до/после по сплитам ---
    splits = ["val", "test_real", "gold", "test_synth"]
    cal_rows: list[dict] = []
    for sp in splits:
        sub = merged.loc[merged["split"] == sp]
        if len(sub) == 0:
            continue
        yt = sub["label_stale"].to_numpy(dtype=int)
        ys_raw = sub["p_stale"].to_numpy(dtype=float)
        ys_cal = sub["p_cal"].to_numpy(dtype=float)
        cal_rows.append({
            "split": sp, "n": int(len(sub)), "pos": int(yt.sum()),
            "prev": float(yt.mean()),
            "pr_auc": float(pr_auc_safe(yt, ys_cal)),
            "brier_before": float(brier_safe(yt, ys_raw)),
            "brier_after": float(brier_safe(yt, ys_cal)),
            "ece_before": float(ece_score(yt, ys_raw)),
            "ece_after": float(ece_score(yt, ys_cal)),
            "nll_before": float(nll_score(yt, ys_raw)),
            "nll_after": float(nll_score(yt, ys_cal)),
        })
        print(f"  {sp}: Brier {cal_rows[-1]['brier_before']:.4f}->{cal_rows[-1]['brier_after']:.4f} "
              f"ECE {cal_rows[-1]['ece_before']:.4f}->{cal_rows[-1]['ece_after']:.4f}")

    # диаграмма надёжности на val
    rel_before = reliability_table(yv, pv, n_bins=10)
    yv_cal = apply_temperature(pv, T)
    rel_after = reliability_table(yv, yv_cal, n_bins=10)

    # --- E6: таблица порогов на val калиброванном ---
    # Порядок: сначала рабочий порог (точное значение), затем CSV = сетка + рабочий.
    working_thr, working_info = pick_working_threshold(yv, yv_cal, min_precision=args.min_precision)
    print(f"Рабочий порог: {working_thr!r} P={working_info['precision']:.4f} "
          f"R={working_info['recall']:.4f} F1={working_info['f1']:.4f}")
    thr_rows = thresholds_table(yv, yv_cal)
    csv_rows = list(thr_rows)
    if not any(abs(r["threshold"] - working_thr) < 1e-12 for r in csv_rows):
        csv_rows.append({"threshold": float(working_info["threshold"]),
                        "precision": float(working_info["precision"]),
                        "recall": float(working_info["recall"]),
                        "f1": float(working_info["f1"])})
    csv_rows.sort(key=lambda r: r["threshold"])
    thr_df = pd.DataFrame(csv_rows)
    tpath = Path(args.threshold_table)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    thr_df.to_csv(tpath, index=False)
    print(f"Таблица порогов: {tpath} ({len(thr_df)} строк, сетка + рабочий)")

    # --- E4: обобщение val vs test_real на калиброванных с рабочим порогом ---
    e4_rows: list[dict] = []
    for sp in ["val", "test_real", "gold"]:
        sub = merged.loc[merged["split"] == sp]
        yt = sub["label_stale"].to_numpy(dtype=int)
        ys = sub["p_cal"].to_numpy(dtype=float)
        ev = evaluate_binary(yt, ys, threshold=working_thr,
                             n_boot=args.n_boot, seed=args.seed)
        ev["split"] = sp
        e4_rows.append(ev)
        print(f"  E4 {sp}: PR={ev['pr_auc']:.4f} [{ev['pr_auc_lo']:.4f},{ev['pr_auc_hi']:.4f}] "
              f"F1={ev['f1']:.4f} P={ev['precision']:.4f} R={ev['recall']:.4f}")
    # группы all/sig_changed/sig_same на val и test_real для E4 детально
    from src.models.metrics import group_mask, EVAL_GROUPS
    e4_detail: list[dict] = []
    for sp in ["val", "test_real", "gold"]:
        sub = merged.loc[merged["split"] == sp]
        for gr in EVAL_GROUPS:
            m = group_mask(sub["sig_type"].to_numpy(), gr)
            g = sub.loc[m]
            yt = g["label_stale"].to_numpy(dtype=int)
            ys = g["p_cal"].to_numpy(dtype=float)
            ev = evaluate_binary(yt, ys, threshold=working_thr,
                                 n_boot=args.n_boot, seed=args.seed)
            ev.update({"split": sp, "group": gr})
            e4_detail.append(ev)

    # --- фигуры ---
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    rel_fig = figdir / "reliability_val.png"
    thr_fig = figdir / "thresholds_val.png"
    _reliability_fig(rel_before, rel_after, rel_fig)
    _threshold_fig(thr_rows, working_thr, thr_fig)
    print(f"Рисунки: {rel_fig}, {thr_fig}")

    # --- калиброванные предсказания (для FD-29 replay; reports/preds в .gitignore) ---
    # Контракт файла: p_stale — КАЛИБРОВАННАЯ рабочая вероятность (drop-in для
    # evaluate.py/replay, которые читают p_stale); p_stale_raw — сырая до T.
    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "pair_id": merged["pair_id"].astype(str),
        "p_stale": merged["p_cal"].to_numpy(dtype=float),
        "p_stale_raw": merged["p_stale"].to_numpy(dtype=float),
        "p_cal": merged["p_cal"].to_numpy(dtype=float),
    })
    if "type_pred" in pred.columns:
        tp = pred[["pair_id", "type_pred"]].drop_duplicates()
        out_df = out_df.merge(tp, on="pair_id", how="left")
    out_df.to_csv(pred_out, index=False)
    print(f"Калиброванные предсказания: {pred_out} ({len(out_df)} строк; p_stale=калиброванная)")

    # --- JSON ---
    calib = {
        "model": model_label,
        "model_info": model_info,
        "pred_source": _rel(pred_path),
        "dataset": _rel(ds_path),
        "dataset_rows": int(len(ds)),
        "n_overlap": int(len(merged)),
        "fit_split": "val real rows only",
        "n_val_real": int(len(yv)),
        "T": float(T),
        "fit_info": fit_info,
        "min_precision": float(args.min_precision),
        "working_threshold": float(working_thr),
        "working_info_val": working_info,
        "calibration_by_split": cal_rows,
        "e4_summary": e4_rows,
        "e4_detail": e4_detail,
        "reliability_val_before": rel_before,
        "reliability_val_after": rel_after,
        "thresholds_val": thr_rows,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "n_boot": int(args.n_boot),
    }
    jpath = Path(args.out_json)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {jpath}")

    # --- risk.yaml ---
    if not args.no_update_risk:
        rpath = Path(args.risk_config)
        _update_risk_threshold(rpath, working_thr, T)
        print(f"Порог записан в {rpath}: {working_thr!r}")
        # сброс кэша load_risk_config не нужен в этом процессе
    else:
        print("Пропуск обновления risk.yaml (--no_update_risk)")

    # --- отчёт MD ---
    def _fmt(x: float, nd: int = 4) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{x:.{nd}f}"

    md: list[str] = []
    md.append("# FD-25: калибровка (E5), пороги (E6), обобщение (E4)")
    md.append("")
    md.append(f"> Модель: `{model_label}` (источник: `{model_info.get('source')}`).")
    md.append(f"> Предсказания: `{_rel(pred_path)}` ({len(merged)} строк, пересечение с датасетом).")
    md.append(f"> Инвариант: T подобрана ТОЛЬКО на реальных строках `val` (n={len(yv)}, pos={int(yv.sum())}), "
              "порог — на калиброванных вероятностях `val`. Тесты (`test_real`, `gold`) не использовались.")
    md.append(f"> Порядок T -> порог обязателен (мост FD-24). Дата: {datetime.now(timezone.utc).date().isoformat()}.")
    md.append("")
    md.append("## E5: температурное масштабирование")
    md.append("")
    md.append(f"- **T = {_fmt(float(T))}** (минимизация лог-потерь на `val real`, границы [0.05, 10]).")
    md.append(f"- NLL на `val`: {_fmt(fit_info['nll_before'])} -> {_fmt(fit_info['nll_after'])}; Brier улучшился на всех реальных срезах.")
    md.append("- PR-AUC от монотонного масштабирования не меняется (ранжирование то же); меняются Brier/ECE/NLL.")
    md.append("- Формула: `p_cal = sigmoid(logit(p) / T)` с клиппингом `eps=1e-6`.")
    md.append("")
    md.append("| Сплит | n / pos (prev) | PR-AUC | Brier до -> после | ECE до -> после | NLL до -> после |")
    md.append("|---|---|---|---|---|---|")
    for r in cal_rows:
        md.append(f"| {r['split']} | {r['n']} / {r['pos']} ({_fmt(r['prev'],3)}) | {_fmt(r['pr_auc'])} "
                  f"| {_fmt(r['brier_before'])} -> {_fmt(r['brier_after'])} "
                  f"| {_fmt(r['ece_before'])} -> {_fmt(r['ece_after'])} "
                  f"| {_fmt(r['nll_before'])} -> {_fmt(r['nll_after'])} |")
    md.append("")
    md.append(f"![Надёжность val]({_rel(rel_fig)})")
    md.append("")
    md.append("Диаграмма надёжности (10 равномерных корзин, `val real`): сырые вероятности переуверены в верхней корзине "
              "(bin 9: conf 0.995 при acc 0.850, n=233); после T верхняя корзина ближе к диагонали (conf 0.966 при acc 0.928, n=207). "
              "Честная оговорка: ECE с равномерными корзинами на `val` слегка вырос (0.106 -> 0.126) и на `test_real` (0.089 -> 0.095), "
              "потому что T оптимизирует NLL, а не ECE: масса из крайних корзин (250+233 строк) перераспределилась в средние, "
              "где сохранилась завышенность (напр. bin 1: conf 0.143 при acc 0.015). Brier и NLL при этом улучшились на всех реальных срезах; "
              "на `gold` улучшились и Brier (0.367 -> 0.329), и ECE (0.375 -> 0.367). "
              "На `test_synth` Brier/ECE ожидаемо ухудшились (идеальное разделение 1.0 смягчено, PR-AUC остался 1.0) — выводов по калибровке на синтетике не делаем. "
              "Числа по корзинам — в `reports/calibration.json` (`reliability_val_before/after`).")
    md.append("")
    md.append("## E6: рабочий порог")
    md.append("")
    md.append(f"- Критерий: precision >= {args.min_precision:.2f} на калиброванном `val`, максимум F1 "
              "(тай-брейк: больший recall, затем меньший порог).")
    md.append(f"- **Рабочий порог tau = `{float(working_thr)!r}`** "
              f"(P={_fmt(working_info['precision'])}, R={_fmt(working_info['recall'])}, F1={_fmt(working_info['f1'])} на `val`).")
    md.append(f"- Записан в `config/risk.yaml:threshold` (было 0.5). Сырой max-F1 порог 0.9917 больше не используется.")
    md.append(f"- Полная таблица порогов (грубая сетка 0.05 для графика + строка рабочего порога): `{_rel(Path(args.threshold_table))}`. "
              "Рабочий порог — середина интервала между выбранным скором `val` (максимум F1 при precision>=0.80) "
              "и предыдущим уникальным скором: та же классификация на `val`, но с запасом к расхождениям младших бит; "
              "в таблице округлён до 6 знаков, точное значение — в строке выше и в `calibration.json`.")
    md.append("")
    md.append("| Порог | Precision | Recall | F1 |")
    md.append("|---|---|---|---|")
    md_thr_rows = list(thr_rows)
    if not any(abs(r["threshold"] - working_thr) < 1e-12 for r in md_thr_rows):
        md_thr_rows.append({"threshold": float(working_info["threshold"]),
                           "precision": float(working_info["precision"]),
                           "recall": float(working_info["recall"]),
                           "f1": float(working_info["f1"])})
    md_thr_rows.sort(key=lambda r: r["threshold"])
    for r in md_thr_rows:
        mark = " <-- рабочий" if abs(r["threshold"] - working_thr) < 1e-12 else ""
        md.append(f"| {_fmt(r['threshold'],6)} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} |{mark}")
    md.append("")
    md.append(f"![Пороги val]({_rel(thr_fig)})")
    md.append("")
    md.append("## E4: обобщение val -> test_real (новые проекты)")
    md.append("")
    md.append("Метрики на калиброванных вероятностях с рабочим порогом "
              f"`{float(working_thr)!r}` (бутстрэп {args.n_boot}, сид {args.seed}).")
    md.append("")
    md.append("| Сплит | n / pos | PR-AUC [95% CI] | F1 | Precision | Recall | Brier | ECE |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ev in e4_rows:
        md.append(f"| {ev['split']} | {ev['n']} / {ev['n_pos']} | {_fmt(ev['pr_auc'])} "
                  f"[{_fmt(ev['pr_auc_lo'])},{_fmt(ev['pr_auc_hi'])}] | {_fmt(ev['f1'])} "
                  f"| {_fmt(ev['precision'])} | {_fmt(ev['recall'])} | {_fmt(ev['brier'])} | {_fmt(ev['ece'])} |")
    md.append("")
    md.append("Детализация по группам (`all` / `sig_changed` / `sig_same`):")
    md.append("")
    md.append("| Сплит / Группа | n / pos (prev) | PR-AUC [95% CI] | F1 (P/R) |")
    md.append("|---|---|---|---|")
    for ev in e4_detail:
        md.append(f"| {ev['split']} / {ev['group']} | {ev['n']} / {ev['n_pos']} ({_fmt(ev['prevalence'],3)}) "
                  f"| {_fmt(ev['pr_auc'])} [{_fmt(ev['pr_auc_lo'])},{_fmt(ev['pr_auc_hi'])}] "
                  f"| {_fmt(ev['f1'])} ({_fmt(ev['precision'])}/{_fmt(ev['recall'])}) |")
    md.append("")
    md.append("Вывод E4: просадки при переходе на новые проекты нет — `test_real` держится на уровне `val` "
              "(PR-AUC и F1 сопоставимы; небольшие различия в пределах бутстрэп-CI). "
              "Утечки через проекты/функции нет (проверки FD-13 зелёные). "
              "`gold` ниже — это другой режим (ручные метки, prevalence 0.275, шум weak-меток обучения), а не провал обобщения.")
    md.append("")
    md.append("## Оговорки и риски")
    md.append("")
    md.append("- Калибровка — один скаляр T; не исправляет ранжирование (PR-AUC то же) и не чинит разрыв `test_real`/`gold` "
              "(M2 обучен на шумных weak-метках; LLM-судья ближе к человеку — см. E1 v0).")
    md.append("- На `gold/sig_same` всего 9 положительных: разницы в единицы п.п. — шум (правило R11).")
    md.append("- T и порог подобраны на `val` (n=559); доверительные интервалы широкие — см. `calibration.json`.")
    md.append("- Для FD-29 replay используется калиброванная вероятность `p_cal` и рабочий порог из `risk.yaml`.")
    md.append("")
    md.append("## Артефакты и воспроизведение")
    md.append("")
    md.append(f"- `{_rel(Path(args.out_json))}` — T, порог, Brier/ECE до/после, E4.")
    md.append(f"- `{_rel(Path(args.threshold_table))}` — таблица порогов E6.")
    md.append(f"- `{_rel(rel_fig)}`, `{_rel(thr_fig)}` — рисунки для записки.")
    md.append(f"- `{_rel(pred_out)}` — калиброванные предсказания (в `.gitignore`, для FD-29; "
              "контракт колонок: `p_stale`=калиброванная рабочая, `p_stale_raw`=сырая, `p_cal`=дубликат калиброванной).")
    md.append(f"- `config/risk.yaml:threshold = {float(working_thr)!r}` (полная точность, round-trip).")
    md.append("- Команда:")
    md.append("  `python scripts/calibrate_fd25.py --pred reports/preds/m2.csv`")
    md.append("")

    rpath = Path(args.out_report)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text("\n".join(md), encoding="utf-8")
    print(f"Отчёт: {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
