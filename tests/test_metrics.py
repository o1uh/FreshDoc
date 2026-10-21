from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.metrics import (
    EVAL_GROUPS,
    EVAL_SPLITS,
    binary_at_threshold,
    bootstrap_pr_auc,
    brier_safe,
    ece_score,
    evaluate_binary,
    evaluate_drift,
    group_mask,
    pick_threshold_f1,
    pr_auc_safe,
    pr_curve_points,
    recall_at_precision,
    roc_auc_safe,
)
from scripts.evaluate import _load_pred
from scripts.evaluate import main as evaluate_main


def test_binary_at_threshold_empty_slice():
    """Пустой срез (модель без строк сплита, как M3 из FD-22) — nan, не ValueError."""
    out = binary_at_threshold(np.array([]), np.array([]), 0.5)
    assert out["threshold"] == 0.5
    assert np.isnan(out["precision"]) and np.isnan(out["recall"]) and np.isnan(out["f1"])
    ev = evaluate_binary(np.array([]), np.array([]), threshold=0.5, n_boot=50, seed=0)
    assert ev["n"] == 0 and np.isnan(ev["f1"]) and np.isnan(ev["pr_auc"])


def test_evaluate_drift_unknown_pred_not_a_class():
    """Предсказанный unknown — обычная ошибка, а не отдельный класс в macro-F1.

    Протокол R11: только y_stale=1 с известным типом; none/unknown
    в метрику не входят (истинные none-строки исключаются из n_drift).
    """
    dt = np.array(["param_added", "param_added", "none", "none"])
    dp = np.array(["param_added", "unknown", "none", "none"])
    out = evaluate_drift(dt, dp)
    # истинные none-строки исключены: n=2, верно 1
    assert out["n_drift"] == 2
    assert out["drift_acc"] == 0.5
    assert "unknown" not in out["drift_per_class_f1"]
    assert "none" not in out["drift_per_class_f1"]
    assert abs(out["drift_macro_f1"] - 2 / 3) < 1e-9


def test_load_pred_categorical_type_pred():
    """parquet с Categorical type_pred без категории unknown — fillna не падает."""
    import tempfile
    from pathlib import Path
    df = pd.DataFrame({
        "pair_id": ["a", "b"],
        "p_stale": [0.2, 0.8],
        "type_pred": pd.Categorical(["none", None], categories=["none", "param_added"]),
    })
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pred.parquet"
        df.to_parquet(p, index=False)
        out = _load_pred(p)
    assert out["type_pred"].tolist() == ["none", "unknown"]


def _cli_truth() -> pd.DataFrame:
    rows = []
    for split, source in (("val", "codocbench"), ("test_real", "codocbench"),
                          ("test_synth", "synth_ast"), ("gold", "gold")):
        drift = "param_added" if source != "synth_ast" else "return_changed"
        rows.append({"pair_id": f"{split}_c1", "label_stale": 1, "split": split,
                     "sig_type": "param_added", "drift_type": drift, "source": source})
        rows.append({"pair_id": f"{split}_c2", "label_stale": 1, "split": split,
                     "sig_type": "param_added", "drift_type": drift, "source": source})
        rows.append({"pair_id": f"{split}_s1", "label_stale": 0, "split": split,
                     "sig_type": "none", "drift_type": "none", "source": source})
        rows.append({"pair_id": f"{split}_s2", "label_stale": 1 if split == "gold" else 0,
                     "split": split, "sig_type": "none",
                     "drift_type": "unknown" if split == "gold" else "none",
                     "source": source})
    return pd.DataFrame(rows)


def _run_cli(monkeypatch, truth_path, pred_arg, out_metrics, out_table, figdir):
    import sys
    monkeypatch.setattr(sys, "argv", ["evaluate.py",
                                      "--dataset", str(truth_path),
                                      "--pred", pred_arg,
                                      "--out_metrics", str(out_metrics),
                                      "--out_table", str(out_table),
                                      "--figdir", str(figdir),
                                      "--n_boot", "50", "--seed", "42"])
    return evaluate_main()


def test_cli_end_to_end_full(tmp_path, monkeypatch):
    import json
    truth_path = tmp_path / "truth.parquet"
    _cli_truth().to_parquet(truth_path, index=False)
    pred = _cli_truth()[["pair_id"]].copy()
    pred["p_stale"] = [0.9, 0.8, 0.1, 0.7] * 4
    pred["type_pred"] = "param_added"
    pred_path = tmp_path / "pred.csv"
    pred.to_csv(pred_path, index=False)
    out_m, out_t, figdir = tmp_path / "m.json", tmp_path / "e1.csv", tmp_path / "fig"
    assert _run_cli(monkeypatch, truth_path, f"m0={pred_path}", out_m, out_t, figdir) == 0
    rep = json.loads(out_m.read_text(encoding="utf-8"))
    assert set(rep["models"]["m0"]["splits"]) == {"val", "test_real", "test_synth", "gold"}
    assert "avg" not in out_m.read_text(encoding="utf-8")
    flat = pd.read_csv(out_t)
    assert len(flat) == 12  # 1 модель x 4 сплита x 3 группы, без усреднений
    assert len(list(figdir.glob("pr_*.png"))) == 12


def test_cli_end_to_end_subset_no_val(tmp_path, monkeypatch):
    """M3-подобный сабсет (только test_real+gold): пустые срезы — nan, код 0."""
    import json
    import math
    truth_path = tmp_path / "truth.parquet"
    _cli_truth().to_parquet(truth_path, index=False)
    sub = _cli_truth().loc[_cli_truth()["split"].isin(["test_real", "gold"]), ["pair_id"]].copy()
    sub["p_stale"] = 0.6
    pred_path = tmp_path / "pred_sub.csv"
    sub.to_csv(pred_path, index=False)
    out_m, out_t, figdir = tmp_path / "m.json", tmp_path / "e1.csv", tmp_path / "fig"
    assert _run_cli(monkeypatch, truth_path, f"m3={pred_path}", out_m, out_t, figdir) == 0
    rep = json.loads(out_m.read_text(encoding="utf-8"))
    assert rep["models"]["m3"]["threshold_val"] == 0.5
    val_all = rep["models"]["m3"]["splits"]["val"]["all"]
    assert val_all["n"] == 0 and math.isnan(val_all["f1"])
    ts = rep["models"]["m3"]["splits"]["test_synth"]["all"]
    assert ts["n"] == 0 and math.isnan(ts["pr_auc"])


def test_eval_splits_and_groups_lists():
    assert tuple(EVAL_SPLITS) == ("val", "test_real", "test_synth", "gold")
    assert tuple(EVAL_GROUPS) == ("all", "sig_changed", "sig_same")


def test_group_mask():
    sig = np.array(["none", "param_added", "none", "return_changed"])
    assert group_mask(sig, "all").tolist() == [True, True, True, True]
    assert group_mask(sig, "sig_changed").tolist() == [False, True, False, True]
    assert group_mask(sig, "sig_same").tolist() == [True, False, True, False]


def test_pick_threshold_separable():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    thr = pick_threshold_f1(y, s)
    assert 0.2 < thr <= 0.8
    assert binary_at_threshold(y, s, thr)["f1"] == 1.0


def test_pick_threshold_single_class_defaults():
    assert pick_threshold_f1(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])) == 0.5
    assert pick_threshold_f1(np.array([1, 1]), np.array([0.1, 0.9])) == 0.5


def test_perfect_predictions_metrics():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.0, 0.1, 0.9, 1.0])
    assert pr_auc_safe(y, s) == 1.0
    assert roc_auc_safe(y, s) == 1.0
    ev = evaluate_binary(y, s, threshold=0.5, n_boot=200, seed=0)
    assert ev["f1"] == 1.0 and ev["precision"] == 1.0 and ev["recall"] == 1.0
    assert ev["brier"] < 0.01
    assert ev["ece"] < 0.1
    assert ev["recall_at_p80"] == 1.0
    assert ev["prevalence"] == 0.5 and ev["n"] == 4 and ev["n_pos"] == 2
    # бутстрэп на идеальном разделении — интервал у единицы
    assert ev["pr_auc_lo"] >= 0.9 or np.isnan(ev["pr_auc_lo"])


def test_random_predictions_low_pr_auc():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    s = rng.random(400)
    pr = pr_auc_safe(y, s)
    assert 0.3 < pr < 0.7  # около prevalence ~0.5
    ev = evaluate_binary(y, s, threshold=0.5, n_boot=200, seed=1)
    assert ev["f1"] < 0.7
    assert ev["pr_auc_lo"] <= pr <= ev["pr_auc_hi"]


def test_single_class_returns_nan_not_crash():
    y = np.array([0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isnan(pr_auc_safe(y, s))
    assert np.isnan(roc_auc_safe(y, s))
    ev = evaluate_binary(y, s, threshold=0.5, n_boot=100, seed=0)
    assert np.isnan(ev["pr_auc"]) and ev["recall_at_p80"] == 0.0
    assert np.isnan(bootstrap_pr_auc(y, s)[0])


def test_brier_and_ece_known_values():
    y = np.array([0, 1])
    s = np.array([0.0, 1.0])
    assert brier_safe(y, s) == 0.0
    assert ece_score(y, s) == 0.0
    # полная неуверенность 0.5 при балансе: brier 0.25
    assert abs(brier_safe(y, np.array([0.5, 0.5])) - 0.25) < 1e-12


def test_recall_at_precision_none_feasible():
    # инвертированные скоры: высокой точности не достичь
    y = np.array([0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert recall_at_precision(y, s, target=0.99) == 0.0


def test_bootstrap_deterministic():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 10)
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6] * 10)
    lo1, hi1 = bootstrap_pr_auc(y, s, n_boot=200, seed=42)
    lo2, hi2 = bootstrap_pr_auc(y, s, n_boot=200, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= pr_auc_safe(y, s) <= hi1


def test_evaluate_drift_known_only():
    dt = np.array(["param_added", "unknown", "none", "param_added"])
    dp = np.array(["param_added", "none", "none", "none"])
    out = evaluate_drift(dt, dp)
    # unknown и none (истинные) исключены: n=2 (только param_added), верно 1
    assert out["n_drift"] == 2
    assert abs(out["drift_acc"] - 1 / 2) < 1e-12
    assert 0.0 <= out["drift_macro_f1"] <= 1.0
    assert "param_added" in out["drift_per_class_f1"]
    assert "none" not in out["drift_per_class_f1"]
    assert "unknown" not in out["drift_per_class_f1"]


def test_evaluate_drift_y_stale_filter():
    """Фильтр y_stale=1: синхронные (0) и unknown исключаются (R11)."""
    dt = np.array(["param_added", "param_added", "none", "unknown", "return_changed"])
    dp = np.array(["param_added", "param_added", "none", "param_added", "none"])
    y = np.array([1, 0, 0, 1, 1])
    out = evaluate_drift(dt, dp, y)
    # остаются idx0 (param_added верный) и idx4 (return_changed -> none, ошибка)
    assert out["n_drift"] == 2
    assert abs(out["drift_acc"] - 0.5) < 1e-12
    assert set(out["drift_per_class_f1"]) == {"param_added", "return_changed"}
    assert "none" not in out["drift_per_class_f1"]
    assert "unknown" not in out["drift_per_class_f1"]
    # без y_stale тот же набор дал бы 3 строки (idx0, idx1, idx4)
    out_no_y = evaluate_drift(dt, dp)
    assert out_no_y["n_drift"] == 3


def test_evaluate_drift_empty_known():
    out = evaluate_drift(np.array(["unknown"]), np.array(["none"]))
    assert out["n_drift"] == 0 and np.isnan(out["drift_macro_f1"])


def test_pr_curve_points_single_class_empty():
    pts = pr_curve_points(np.array([0, 0]), np.array([0.1, 0.2]))
    assert pts["precision"] == [] and pts["recall"] == []


def test_no_averaging_across_splits_contract():
    # контракт: evaluate_binary вызывается отдельно на каждый срез;
    # здесь проверяем, что метрики срезов различаются и не смешиваются.
    df = pd.DataFrame({
        "split": ["val"] * 4 + ["gold"] * 4,
        "sig_type": ["none", "none", "param_added", "param_added"] * 2,
        "label_stale": [0, 0, 1, 1, 0, 1, 0, 1],
        "p_stale": [0.1, 0.2, 0.8, 0.9, 0.9, 0.1, 0.8, 0.2],
    })
    by_split = {}
    for split in ("val", "gold"):
        by_split[split] = evaluate_binary(
            df.loc[df["split"] == split, "label_stale"].to_numpy(),
            df.loc[df["split"] == split, "p_stale"].to_numpy(),
            threshold=0.5, n_boot=100, seed=0)
    assert by_split["val"]["f1"] == 1.0
    assert by_split["gold"]["f1"] < 1.0  # разные срезы — разные цифры


def test_evaluate_m3subset_runs(tmp_path):
    import pytest
    from scripts.evaluate_m3subset import (
        main as subset_main,
        IN_DATASET,
        IN_M3_SELECT,
        DEFAULT_PRED_PATHS,
    )

    if not IN_DATASET.exists() or not IN_M3_SELECT.exists():
        pytest.skip("Отсутствуют входные файлы датасета/выборки (data/ в .gitignore по R15)")
    if not all(p.exists() for p in DEFAULT_PRED_PATHS.values()):
        pytest.skip("Отсутствуют файлы предсказаний (reports/preds/ в .gitignore по R15)")

    out_csv = tmp_path / 'e1_m3subset.csv'
    ret = subset_main(['--out', str(out_csv), '--n_boot', '10'])
    assert ret == 0
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 15
    assert set(df['model']) == {'M0', 'M1', 'M2', 'M3', 'M3max'}


def test_evaluate_m3subset_mock_cli(tmp_path):
    """Тест CLI на синтетических данных: всегда исполняется в CI без внешних файлов."""
    import json
    from scripts.evaluate_m3subset import main as subset_main

    ds = pd.DataFrame({
        "pair_id": ["p1", "p2", "p3", "p4"],
        "label_stale": [1, 1, 0, 0],
        "split": ["test_real", "test_real", "test_real", "val"],
        "sig_type": ["param_added", "none", "none", "none"],
        "drift_type": ["param_added", "none", "none", "none"],
        "source": ["codocbench", "codocbench", "codocbench", "codocbench"],
    })
    ds_path = tmp_path / "ds.parquet"
    ds.to_parquet(ds_path, index=False)

    sel_path = tmp_path / "select.json"
    sel_path.write_text(json.dumps({
        "m3_id_to_pair_id": {"0": "p1", "1": "p2", "2": "p3"}
    }), encoding="utf-8")

    pred = pd.DataFrame({
        "pair_id": ["p1", "p2", "p3"],
        "p_stale": [0.9, 0.7, 0.2],
        "type_pred": ["param_added", "none", "none"]
    })
    pred_path = tmp_path / "pred_test.csv"
    pred.to_csv(pred_path, index=False)

    out_csv = tmp_path / "out_subset.csv"
    ret = subset_main([
        "--dataset", str(ds_path),
        "--select", str(sel_path),
        "--pred", f"TestM={pred_path}",
        "--out", str(out_csv),
        "--n_boot", "10",
    ])
    assert ret == 0
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 3
    assert df["model"].iloc[0] == "TestM"

