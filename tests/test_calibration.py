"""FD-25: тесты температурной калибровки и выбора порога (E5/E6)."""
from __future__ import annotations

import numpy as np

from src.models.calibration import (
    apply_temperature,
    fit_temperature,
    logit,
    nll_score,
    pick_working_threshold,
    reliability_table,
    sigmoid,
    thresholds_table,
)


def test_logit_sigmoid_roundtrip():
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert np.allclose(sigmoid(logit(p)), p, atol=1e-9)


def test_apply_temperature_identity_and_softening():
    p = np.array([0.1, 0.9, 0.99])
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-9)
    # T>1 смягчает экстремумы к 0.5
    soft = apply_temperature(p, 2.0)
    assert soft[1] < 0.9 and soft[1] > 0.5
    assert soft[0] > 0.1 and soft[0] < 0.5
    # T<1 заостряет
    sharp = apply_temperature(np.array([0.6]), 0.5)[0]
    assert sharp > 0.6


def test_apply_temperature_clips_0_1():
    # M0 даёт точные 0/1 — не должно быть inf/nan
    out = apply_temperature(np.array([0.0, 1.0]), 1.5)
    assert np.all(np.isfinite(out)) and 0.0 < out[0] < 1.0 and 0.0 < out[1] < 1.0


def test_fit_temperature_reduces_nll_on_overconfident():
    rng = np.random.default_rng(0)
    # переуверенные скоры: правильные, но слишком экстремальные
    y = np.array([0] * 50 + [1] * 50)
    p = np.where(y == 1, 0.99, 0.01)
    p = np.clip(p + rng.normal(0, 0.005, len(p)), 1e-6, 1 - 1e-6)
    before = nll_score(y, p)
    T, info = fit_temperature(y, p)
    after = nll_score(y, apply_temperature(p, T))
    assert after < before
    assert 0.05 <= T <= 10.0
    assert info["nll_after"] < info["nll_before"]


def test_fit_temperature_single_class_raises():
    import pytest
    with pytest.raises(ValueError):
        fit_temperature(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_pick_working_threshold_respects_precision():
    # синтетика: порог 0.5 даёт P~0.67, выше — P>=0.8
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95])
    thr, info = pick_working_threshold(y, s, min_precision=0.8)
    assert info["precision"] >= 0.8
    assert 0.0 <= thr <= 1.0


def test_pick_working_threshold_fallback_when_unfeasible():
    # Достижимый max precision = 0.5, требуемый 0.9 -> срабатывание fallback
    y = np.array([0, 0, 1, 0])
    s = np.array([0.2, 0.4, 0.6, 0.8])
    thr, info = pick_working_threshold(y, s, min_precision=0.9)
    assert 0.0 <= thr <= 1.0
    assert info["precision"] <= 0.9


def test_sigmoid_extreme_no_overflow_warning():
    """Края z=±1000: 0/1 без RuntimeWarning (expit вместо наивного exp)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sigmoid(np.array([-1000.0, -800.0, 0.0, 800.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[-1] == 1.0
    assert abs(out[2] - 0.5) < 1e-12


def test_thresholds_table_monotone_counts():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.2, 0.4, 0.6, 0.8])
    rows = thresholds_table(y, s, thresholds=[0.3, 0.5, 0.7])
    assert len(rows) == 3
    assert rows[0]["threshold"] == 0.3


def test_reliability_table_bins_sum():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.4, 0.6])
    tbl = reliability_table(y, s, n_bins=5)
    assert len(tbl) == 5
    assert sum(r["n"] for r in tbl) == 4


def _make_synth_frames(tmp_path):
    """Синтетические ds/pred для сквозных прогонов calibrate_fd25 (val/test_real/gold)."""
    import pandas as pd

    rng = np.random.default_rng(1)
    # маленький датасет: val/test_real/gold
    rows = []
    pid = 0
    for split, n, prev in (("val", 60, 0.4), ("test_real", 80, 0.5), ("gold", 40, 0.3)):
        for _ in range(n):
            y = 1 if rng.random() < prev else 0
            # переуверенные скоры вокруг истины
            p = 0.95 if y == 1 else 0.05
            p = float(np.clip(p + rng.normal(0, 0.1), 1e-4, 1 - 1e-4))
            rows.append({"pair_id": f"p{pid}", "label_stale": y, "split": split,
                         "sig_type": "none", "drift_type": "none",
                         "source": "codocbench", "project": "a/b"})
            pid += 1
    ds = pd.DataFrame(rows)
    ds_path = tmp_path / "ds.parquet"
    ds.to_parquet(ds_path, index=False)
    pred = ds[["pair_id"]].copy()
    # восстановить p по строкам
    pred["p_stale"] = [r for r in ds["label_stale"].map({1: 0.9, 0: 0.1})]
    pred_path = tmp_path / "pred.csv"
    pred.to_csv(pred_path, index=False)
    return ds_path, pred_path


def test_calibrate_script_dry_run(tmp_path):
    """Сквозной прогон скрипта на синтетике (с обновлением risk.yaml)."""
    from scripts.calibrate_fd25 import main as cal_main

    ds_path, pred_path = _make_synth_frames(tmp_path)
    out_rep = tmp_path / "e4_e5_e6.md"
    out_json = tmp_path / "cal.json"
    figdir = tmp_path / "fig"
    ttable = tmp_path / "thr.csv"
    pout = tmp_path / "pcal.csv"
    rc = tmp_path / "risk.yaml"
    rc.write_text("threshold: 0.5\n", encoding="utf-8")
    ret = cal_main(["--dataset", str(ds_path), "--pred", str(pred_path),
                    "--out_report", str(out_rep), "--out_json", str(out_json),
                    "--figdir", str(figdir), "--risk_config", str(rc),
                    "--threshold_table", str(ttable), "--pred_out", str(pout),
                    "--n_boot", "50"])
    assert ret == 0
    assert out_rep.exists() and out_json.exists() and ttable.exists() and pout.exists()
    assert (figdir / "reliability_val.png").exists()
    import json
    import pandas as pd
    import yaml
    cal = json.loads(out_json.read_text(encoding="utf-8"))
    assert 0.05 <= cal["T"] <= 10.0
    # threshold записан с полной точностью (round-trip, без сдвига границы)
    rc_raw = yaml.safe_load(rc.read_text(encoding="utf-8"))
    assert float(rc_raw["threshold"]) == float(cal["working_threshold"])
    # CSV содержит рабочий порог (контракт md <-> csv)
    thr_csv = pd.read_csv(ttable)
    assert any(abs(float(v) - float(cal["working_threshold"])) < 1e-12
               for v in thr_csv["threshold"].tolist())
    # контракт файла предсказаний: p_stale — калиброванная рабочая,
    # p_stale_raw — сырая; downstream читает p_stale и не откатится
    pcal = pd.read_csv(pout)
    assert {"pair_id", "p_stale", "p_stale_raw", "p_cal"} <= set(pcal.columns)
    assert np.allclose(pcal["p_stale"].to_numpy(float),
                       pcal["p_cal"].to_numpy(float), atol=0, rtol=0)


def test_calibrate_script_no_update_risk(tmp_path):
    """Ветка --no_update_risk: risk.yaml не трогается, остальное пишется."""
    from scripts.calibrate_fd25 import main as cal_main

    ds_path, pred_path = _make_synth_frames(tmp_path)
    out_rep = tmp_path / "e4_e5_e6.md"
    out_json = tmp_path / "cal.json"
    figdir = tmp_path / "fig"
    ttable = tmp_path / "thr.csv"
    pout = tmp_path / "pcal.csv"
    rc = tmp_path / "risk.yaml"
    before = "threshold: 0.5\n"
    rc.write_text(before, encoding="utf-8")
    ret = cal_main(["--dataset", str(ds_path), "--pred", str(pred_path),
                    "--out_report", str(out_rep), "--out_json", str(out_json),
                    "--figdir", str(figdir), "--risk_config", str(rc),
                    "--threshold_table", str(ttable), "--pred_out", str(pout),
                    "--n_boot", "50", "--no_update_risk"])
    assert ret == 0
    assert out_rep.exists() and out_json.exists()
    assert rc.read_text(encoding="utf-8") == before


def test_update_risk_threshold_replaces_provenance(tmp_path):
    """Идемпотентность + обновление T: повторный прогон заменяет строку, а не плодит."""
    from scripts.calibrate_fd25 import _update_risk_threshold

    rc = tmp_path / "risk.yaml"
    rc.write_text("threshold: 0.5\n", encoding="utf-8")
    _update_risk_threshold(rc, 0.8, 1.0)
    first = rc.read_text(encoding="utf-8")
    assert first.count("threshold <- E6") == 1
    assert "T=1.0000" in first
    assert "threshold: 0.8" in first
    _update_risk_threshold(rc, 0.8816256707073543, 2.3847)
    second = rc.read_text(encoding="utf-8")
    assert second.count("threshold <- E6") == 1
    assert "T=2.3847" in second
    assert "T=1.0000" not in second
    assert "threshold: 0.8816256707073543" in second
    # третий прогон с теми же значениями — текст стабилен кроме даты
    _update_risk_threshold(rc, 0.8816256707073543, 2.3847)
    third = rc.read_text(encoding="utf-8")
    assert third.count("threshold <- E6") == 1


def test_read_best_model_from_yaml(tmp_path):
    """Метаданные модели читаются из train_best.yaml, а не хардкод."""
    from scripts.calibrate_fd25 import _read_best_model

    cfg = tmp_path / "train_best.yaml"
    cfg.write_text("model_name: test-org/test-enc\nmax_len: 2048\ntrain_mix: full\ndoc_budget: 123\n",
                   encoding="utf-8")
    label, info = _read_best_model(cfg)
    assert "test-org/test-enc" in label
    assert "2048" in label
    assert info["model_name"] == "test-org/test-enc"
    assert info["max_len"] == 2048
    # битый файл — fallback без падения
    bad = tmp_path / "bad.yaml"
    bad.write_text(":\n: bad", encoding="utf-8")
    label2, info2 = _read_best_model(bad)
    assert "fallback" in label2.lower() or "fallback" in str(info2)


def test_nll_score_empty_returns_nan_without_warning():
    """Пустой вход: nan без RuntimeWarning (контракт как у brier_safe/ece_score)."""
    import math
    import warnings
    from src.models.calibration import nll_score

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(nll_score(np.array([]), np.array([])))


def test_nll_score_length_mismatch_raises():
    """Несовпадающие длины: явный ValueError вместо неявного бродкастинга."""
    import pytest
    from src.models.calibration import nll_score

    with pytest.raises(ValueError, match="не совпадают"):
        nll_score(np.array([0, 1, 0]), np.array([0.2, 0.8]))


def test_pick_working_threshold_midpoint_between_scores():
    """Порог — середина пустого интервала: не совпадает ни с одним скором,
    классификация на выборке подбора та же, что на точном скоре."""
    from src.models.calibration import pick_working_threshold
    from src.models.metrics import binary_at_threshold

    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95])
    thr, info = pick_working_threshold(y, s, min_precision=0.8)
    uniq = sorted(float(v) for v in np.unique(s))
    assert not any(abs(thr - v) < 1e-18 for v in uniq)
    below = [v for v in uniq if v < thr]
    above = [v for v in uniq if v >= thr]
    assert below and above  # строго внутри диапазона скоров
    # та же классификация, что на ближайшем точном скоре сверху
    exact = min(above)
    r_mid = binary_at_threshold(y, s, thr)
    r_exact = binary_at_threshold(y, s, exact)
    assert (r_mid["precision"], r_mid["recall"], r_mid["f1"]) == \
           (r_exact["precision"], r_exact["recall"], r_exact["f1"])


def test_markdown_contains_working_marker(tmp_path):
    """Таблица порогов в md содержит строку рабочего порога с маркером."""
    from scripts.calibrate_fd25 import main as cal_main

    ds_path, pred_path = _make_synth_frames(tmp_path)
    out_rep = tmp_path / "e4_e5_e6.md"
    out_json = tmp_path / "cal.json"
    figdir = tmp_path / "fig"
    ttable = tmp_path / "thr.csv"
    pout = tmp_path / "pcal.csv"
    rc = tmp_path / "risk.yaml"
    rc.write_text("threshold: 0.5\n", encoding="utf-8")
    assert cal_main(["--dataset", str(ds_path), "--pred", str(pred_path),
                     "--out_report", str(out_rep), "--out_json", str(out_json),
                     "--figdir", str(figdir), "--risk_config", str(rc),
                     "--threshold_table", str(ttable), "--pred_out", str(pout),
                     "--n_boot", "50"]) == 0
    md = out_rep.read_text(encoding="utf-8")
    assert "<-- рабочий" in md
