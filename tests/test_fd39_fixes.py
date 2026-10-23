"""FD-39: тесты фиксов M3-full + Go-full381 + миноров (фикстуры, без реальных данных)."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_external  # noqa: E402
import combine_go  # noqa: E402
import download_cocc  # noqa: E402
import make_go_batches  # noqa: E402
import make_go_silver  # noqa: E402
import mine_go  # noqa: E402
from evaluate import resolve_splits  # noqa: E402
from m3_judge import cmd_export_external  # noqa: E402
from src.data.external_adapter import (  # noqa: E402
    cocc_code_has_docstring_block,
    jit_record_to_pair,
)


# ---------- build_external: D-консенсус, хэши ----------

def _go_csv(path: Path, rows: list[dict]) -> Path:
    cols = ["id", "diff_code", "repo", "symbol", "commit_new",
            "code_old", "code_new", "doc_old"]
    pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows]).to_csv(path, index=False)
    return path


def _go_sidecar(path: Path, proxies: dict) -> Path:
    path.write_text(json.dumps(
        [{"id": k, "doc_changed_proxy": bool(v), "sig_changed": False}
         for k, v in proxies.items()], ensure_ascii=False), encoding="utf-8")
    return path


def _go_row(i: str, proxy_note: str = "d") -> dict:
    return {"id": i, "diff_code": f"-a\n+b // {i}", "repo": "gin", "symbol": "F",
            "commit_new": "c2", "code_old": f"old {i}", "code_new": f"new {i}",
            "doc_old": proxy_note}


def _run_build(tmp_path: Path, extra: list[str]) -> dict:
    out = tmp_path / "ext.parquet"
    man = tmp_path / "ext.manifest.json"
    argv = ["build_external.py", "--no_hebcup", "--no_jit", "--no_cocc",
            "--out", str(out), "--manifest", str(man), *extra]
    import sys as _sys
    old = _sys.argv
    _sys.argv = argv
    try:
        assert build_external.main() == 0
    finally:
        _sys.argv = old
    assert out.exists()
    return json.loads(man.read_text(encoding="utf-8"))


def test_build_drops_D_rows_with_counter(tmp_path):
    csv = _go_csv(tmp_path / "go.csv", [_go_row("go000"), _go_row("go001")])
    _go_sidecar(tmp_path / "go_sidecar.json", {"go000": True, "go001": True})
    cons = tmp_path / "cons.csv"
    cons.write_text("id,label\ngo000,A\ngo001,D\n", encoding="utf-8")
    man = _run_build(tmp_path, ["--go_csv", str(csv),
                                "--go_sidecar", str(tmp_path / "go_sidecar.json"),
                                "--go_consensus", str(cons)])
    df = pd.read_parquet(tmp_path / "ext.parquet")
    assert len(df) == 1  # D-строка исключена, а не откачена к прокси
    assert man["inputs"]["go_rows"] == "1"
    assert man["inputs"]["go_skipped_D"] == "1"
    assert man["inputs"]["go_consensus"] == "1"


def test_build_consensus_abc_mapping(tmp_path):
    csv = _go_csv(tmp_path / "go.csv",
                  [_go_row("go000"), _go_row("go001"), _go_row("go002")])
    _go_sidecar(tmp_path / "go_sidecar.json",
                {"go000": False, "go001": False, "go002": False})
    cons = tmp_path / "cons.csv"
    cons.write_text("id,label\ngo000,A\ngo001,B\ngo002,C\n", encoding="utf-8")
    _run_build(tmp_path, ["--go_csv", str(csv),
                          "--go_sidecar", str(tmp_path / "go_sidecar.json"),
                          "--go_consensus", str(cons)])
    df = pd.read_parquet(tmp_path / "ext.parquet")
    assert len(df) == 3
    assert sorted(df["label_stale"].tolist()) == [0, 1, 1]
    assert set(df["label_source"].tolist()) == {"go_silver_consensus"}


def test_build_no_consensus_falls_back_to_proxy(tmp_path):
    csv = _go_csv(tmp_path / "go.csv", [_go_row("go000"), _go_row("go001")])
    _go_sidecar(tmp_path / "go_sidecar.json", {"go000": True, "go001": False})
    man = _run_build(tmp_path, ["--go_csv", str(csv),
                                "--go_sidecar", str(tmp_path / "go_sidecar.json")])
    df = pd.read_parquet(tmp_path / "ext.parquet")
    assert sorted(df["label_stale"].tolist()) == [0, 1]
    assert man["inputs"]["go_skipped_D"] == "0"


def test_build_skips_empty_diff(tmp_path):
    rows = [_go_row("go000"), dict(_go_row("go001"), diff_code="  ")]
    csv = _go_csv(tmp_path / "go.csv", rows)
    _go_sidecar(tmp_path / "go_sidecar.json", {"go000": True, "go001": True})
    man = _run_build(tmp_path, ["--go_csv", str(csv),
                                "--go_sidecar", str(tmp_path / "go_sidecar.json")])
    assert man["inputs"]["go_rows"] == "1"


def test_build_go_input_hashes(tmp_path):
    csv = _go_csv(tmp_path / "go.csv", [_go_row("go000")])
    side = _go_sidecar(tmp_path / "go_sidecar.json", {"go000": True})
    cons = tmp_path / "cons.csv"
    cons.write_text("id,label\ngo000,C\n", encoding="utf-8")
    man = _run_build(tmp_path, ["--go_csv", str(csv), "--go_sidecar", str(side),
                                "--go_consensus", str(cons)])
    for k in ("go_csv_sha256", "go_sidecar_sha256", "go_consensus_sha256"):
        assert len(man["inputs"][k]) == 64


def test_build_jit_input_hash(tmp_path):
    jit = tmp_path / "jit" / "Param"
    jit.mkdir(parents=True)
    rec = {"id": "O_r-1-Param-1", "label": "1", "comment_type": "Param",
           "old_comment_raw": "@p x", "new_comment_raw": "@p y",
           "old_code_raw": "void f() {}", "new_code_raw": "void f() {}"}
    (jit / "test.json").write_text(json.dumps([rec]), encoding="utf-8")
    out = tmp_path / "ext.parquet"
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["build_external.py", "--no_hebcup", "--no_cocc",
                 "--jit_dir", str(tmp_path / "jit"), "--jit_types", "Param",
                 "--out", str(out)]
    try:
        assert build_external.main() == 0
    finally:
        _sys.argv = old
    man = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert man["inputs"]["jit_rows"] == "1"
    assert len(man["inputs"]["jit_sha256"]) == 64


def test_build_cocc_hash_and_docstring_counter(tmp_path):
    cocc = tmp_path / "cocc"
    cocc.mkdir()
    (cocc / "0.py").write_text("# c\nx = 1\n-----\n# c2\nx = 2\n", encoding="utf-8")
    (cocc / "1.py").write_text('"""mod doc"""\n# c\ny = 1\n-----\n# c\ny = 2\n',
                               encoding="utf-8")
    out = tmp_path / "ext.parquet"
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["build_external.py", "--no_hebcup", "--no_jit",
                 "--cocc_dir", str(cocc), "--out", str(out)]
    try:
        assert build_external.main() == 0
    finally:
        _sys.argv = old
    man = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert man["inputs"]["cocc_rows"] == "2"
    assert len(man["inputs"]["cocc_sha256"]) == 64
    assert man["inputs"]["cocc_docstring_blocks"] == "1"


# ---------- m3_judge export-external: чистый parquet ----------

def _ext_row(pid, split, leak=True):
    row = {"pair_id": pid, "split": split, "sig_old": "s",
           "diff_code": "-a\n+b", "doc_old": "d", "label_stale": 1}
    if leak:
        row.update({"_leak_doc_new": "secret", "_aux_src_old": "srctext"})
    return row


def _run_export(tmp_path: Path, df: pd.DataFrame, **kw) -> Path:
    ds = tmp_path / "ext.parquet"
    df.to_parquet(ds, index=False)
    out_dir = tmp_path / "m3ext"
    args = argparse.Namespace(dataset=str(ds), dir=str(out_dir), seed=7,
                              n_hebcup=kw.get("n_hebcup", 10**9),
                              n_jit=kw.get("n_jit", 10**9),
                              n_python2=kw.get("n_python2", 10**9),
                              n_go=kw.get("n_go", 10**9),
                              select_json=str(tmp_path / "sel.json"),
                              out_parquet=str(tmp_path / "m3_ext.parquet"))
    cmd_export_external(args)
    return tmp_path / "m3_ext.parquet"


def test_export_writes_clean_parquet_without_leaks(tmp_path):
    df = pd.DataFrame(
        [_ext_row("h1", "ext_hebcup"), _ext_row("j1", "ext_jit"),
         _ext_row("p1", "ext_python2"), _ext_row("g1", "ext_go_silver")])
    pq = _run_export(tmp_path, df)
    sel = pd.read_parquet(pq)
    assert list(sel.columns) == ["m3_id", "pair_id", "split", "sig_old",
                                 "diff_code", "doc_old"]
    assert not any(c.startswith(("_leak_", "_aux_")) for c in sel.columns)
    assert len(sel) == 4 and sel["pair_id"].is_unique


def test_export_drops_empty_diff_doc(tmp_path):
    df = pd.DataFrame(
        [_ext_row("h1", "ext_hebcup"),
         dict(_ext_row("h2", "ext_hebcup"), doc_old="  "),
         dict(_ext_row("j1", "ext_jit"), diff_code="")])
    pq = _run_export(tmp_path, df)
    sel = pd.read_parquet(pq)
    assert sel["pair_id"].tolist() == ["h1"] or set(sel["pair_id"]) == {"h1"}
    assert len(sel) == 1


def test_export_quotas_respected(tmp_path):
    df = pd.DataFrame(
        [_ext_row(f"h{i}", "ext_hebcup") for i in range(5)]
        + [_ext_row(f"j{i}", "ext_jit") for i in range(6)])
    pq = _run_export(tmp_path, df, n_hebcup=2, n_jit=3)
    sel = pd.read_parquet(pq)
    assert (sel["split"] == "ext_hebcup").sum() == 2
    assert (sel["split"] == "ext_jit").sum() == 3


def test_export_select_json_reports_actual(tmp_path):
    df = pd.DataFrame([_ext_row(f"p{i}", "ext_python2") for i in range(4)])
    _run_export(tmp_path, df)
    sel = json.loads((tmp_path / "sel.json").read_text(encoding="utf-8"))
    assert sel["actual"] == {"ext_python2": 4}
    assert sel["n_total"] == 4


# ---------- make_go_batches: full381 ----------

def _blind_df(n=25, **cols):
    base = [{"id": f"go{i:03d}", "file": "a.go", "symbol": "F",
             "repo": "gin", "commit_old": "c1", "commit_new": "c2",
             "sig_old": "func F() int", "diff_code": "-a\n+b",
             "doc_old": "Does things well.", "code_old": "old",
             "code_new": "new", "label": "", "comment": ""} for i in range(n)]
    for row in base:
        row.update(cols)
    return pd.DataFrame(base)


def test_batches_full_header_and_counts(tmp_path, monkeypatch):
    pilot = tmp_path / "full.csv"
    _blind_df(25).to_csv(pilot, index=False)
    out = tmp_path / "b"
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(pilot),
                                      "--out_dir", str(out), "--per_batch", "12"])
    assert make_go_batches.main() == 0
    prompts = sorted(out.glob("go_batch*.prompt.txt"))
    assert len(prompts) == 3
    text = prompts[0].read_text(encoding="utf-8")
    assert "full381" in text and "gin + cobra" in text
    assert "пилот" not in text.lower() and "30 строк" not in text
    n = sum(1 for p in prompts for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.startswith("=== CASE id="))
    assert n == 25


def test_batches_default_per_batch_12(tmp_path, monkeypatch):
    pilot = tmp_path / "full.csv"
    _blind_df(13).to_csv(pilot, index=False)
    out = tmp_path / "b"
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(pilot),
                                      "--out_dir", str(out)])
    assert make_go_batches.main() == 0
    assert len(sorted(out.glob("go_batch*.prompt.txt"))) == 2


def test_batches_shards_dir_alias(tmp_path, monkeypatch):
    pilot = tmp_path / "full.csv"
    _blind_df(3).to_csv(pilot, index=False)
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(pilot),
                                      "--shards_dir", str(out), "--per_batch", "2"])
    assert make_go_batches.main() == 0
    assert len(sorted(out.glob("go_batch*.prompt.txt"))) == 2


@pytest.mark.parametrize("col", ["doc_new", "sig_changed", "label_source", "_leak_doc_new"])
def test_batches_rejects_label_leak_columns(tmp_path, monkeypatch, col):
    df = _blind_df(2)
    df[col] = "x"
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(bad),
                                      "--out_dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        make_go_batches.main()


def test_batches_allows_canonical_code_new(tmp_path, monkeypatch):
    pilot = tmp_path / "full.csv"
    _blind_df(2).to_csv(pilot, index=False)  # code_old/code_new — канон silver-схемы
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(pilot),
                                      "--out_dir", str(tmp_path / "o")])
    assert make_go_batches.main() == 0


def test_batches_rejects_empty_diff_and_dup_ids(tmp_path, monkeypatch):
    df = _blind_df(2)
    df.loc[0, "diff_code"] = "  "
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(bad),
                                      "--out_dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        make_go_batches.main()
    df2 = _blind_df(2)
    df2.loc[1, "id"] = "go000"
    bad2 = tmp_path / "bad2.csv"
    df2.to_csv(bad2, index=False)
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(bad2),
                                      "--out_dir", str(tmp_path / "o2")])
    with pytest.raises(SystemExit):
        make_go_batches.main()


# ---------- combine_go: гейт, D ----------

def _llm_df(rows):
    return pd.DataFrame([{"id": i, "label": lab, "stale_statement": st, "why": "w"}
                         for i, lab, st in rows])


def _run_combine(tmp_path, monkeypatch, spark_rows, gemini_rows, n=4, capsys=None):
    pilot = tmp_path / "pilot.csv"
    base = [{"id": f"go{i:03d}", "file": "a.go", "symbol": "F",
             "sig_old": "func F() int", "diff_code": "-a\n+b",
             "doc_old": "Does things well."} for i in range(n)]
    pd.DataFrame(base).to_csv(pilot, index=False)
    spark = tmp_path / "s.csv"
    gemini = tmp_path / "g.csv"
    _llm_df(spark_rows).to_csv(spark, index=False)
    _llm_df(gemini_rows).to_csv(gemini, index=False)
    out_c, out_d = tmp_path / "c.csv", tmp_path / "d.csv"
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(spark),
                                      "--gemini", str(gemini), "--pilot", str(pilot),
                                      "--out_combined", str(out_c),
                                      "--out_disputes", str(out_d)])
    return combine_go.main()


def test_combine_fail_returns_nonzero(monkeypatch):
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        tmp = Path(td)
        rc = _run_combine(tmp, monkeypatch,
                          [("go000", "C", ""), ("go001", "C", ""),
                           ("go002", "C", ""), ("go003", "C", "")],
                          [("go000", "B", ""), ("go001", "B", ""),
                           ("go002", "B", ""), ("go003", "B", "")])
        assert rc == 1


def test_combine_pass_returns_zero(monkeypatch):
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        tmp = Path(td)
        rc = _run_combine(tmp, monkeypatch,
                          [("go000", "C", ""), ("go001", "C", ""),
                           ("go002", "C", ""), ("go003", "C", "")],
                          [("go000", "C", ""), ("go001", "C", ""),
                           ("go002", "C", ""), ("go003", "C", "")])
        assert rc == 0


def test_combine_d_drops_from_kappa_bin_with_warning(tmp_path, monkeypatch, capsys):
    rc = _run_combine(tmp_path, monkeypatch,
                      [("go000", "D", ""), ("go001", "C", ""),
                       ("go002", "C", ""), ("go003", "C", "")],
                      [("go000", "D", ""), ("go001", "C", ""),
                       ("go002", "C", ""), ("go003", "B", "")])
    assert rc == 0  # согласие 3/4
    out = capsys.readouterr().out
    assert "исключает 1 строк" in out
    assert "доля D" in out and "WARNING" in out


def test_combine_rejects_stale_statement_for_non_a(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.csv"
    pd.DataFrame([{"id": "go000", "file": "a.go", "symbol": "F",
                   "sig_old": "s", "diff_code": "-a\n+b",
                   "doc_old": "Does things well."}]).to_csv(pilot, index=False)
    spark = tmp_path / "s.csv"
    gem = tmp_path / "g.csv"
    _llm_df([("go000", "C", "лишняя цитата")]).to_csv(spark, index=False)
    _llm_df([("go000", "C", "")]).to_csv(gem, index=False)
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(spark),
                                      "--gemini", str(gem), "--pilot", str(pilot),
                                      "--out_combined", str(tmp_path / "c.csv"),
                                      "--out_disputes", str(tmp_path / "d.csv")])
    with pytest.raises(SystemExit):
        combine_go.main()


def test_combine_rejects_missing_ids(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.csv"
    pd.DataFrame([{"id": "go000", "file": "a.go", "symbol": "F",
                   "sig_old": "s", "diff_code": "-a\n+b",
                   "doc_old": "d"},
                  {"id": "go001", "file": "a.go", "symbol": "G",
                   "sig_old": "s", "diff_code": "-a\n+b",
                   "doc_old": "d"}]).to_csv(pilot, index=False)
    spark = tmp_path / "s.csv"
    gem = tmp_path / "g.csv"
    _llm_df([("go000", "C", "")]).to_csv(spark, index=False)
    _llm_df([("go000", "C", ""), ("go001", "C", "")]).to_csv(gem, index=False)
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(spark),
                                      "--gemini", str(gem), "--pilot", str(pilot),
                                      "--out_combined", str(tmp_path / "c.csv"),
                                      "--out_disputes", str(tmp_path / "d.csv")])
    with pytest.raises(SystemExit):
        combine_go.main()


# ---------- evaluate: threshold_from, present-only ----------

def _truth_parquet(path: Path, splits=("ext_x",), n=20):
    rows = [{"pair_id": f"p{i}", "label_stale": i % 2, "split": splits[0],
             "sig_type": "unknown", "drift_type": "unknown", "source": "x"}
            for i in range(n)]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _pred_csv(path: Path, n=20):
    pd.DataFrame([{"pair_id": f"p{i}", "p_stale": 0.1 + 0.8 * (i % 2)}
                  for i in range(n)]).to_csv(path, index=False)


def _run_evaluate(tmp_path: Path, fixed: bool):
    truth = tmp_path / "truth.parquet"
    _truth_parquet(truth)
    pred = tmp_path / "pred.csv"
    _pred_csv(pred)
    out_m = tmp_path / "m.json"
    out_t = tmp_path / "t.csv"
    argv = ["evaluate.py", "--dataset", str(truth), "--pred", f"M2={pred}",
            "--n_boot", "10", "--out_metrics", str(out_m),
            "--out_table", str(out_t), "--figdir", str(tmp_path / "fig")]
    if fixed:
        argv += ["--threshold_fixed", "0.5"]
    import sys as _sys
    from evaluate import main as ev_main
    old = _sys.argv
    _sys.argv = argv
    try:
        assert ev_main() == 0
    finally:
        _sys.argv = old
    return json.loads(out_m.read_text(encoding="utf-8"))


def test_evaluate_threshold_from_fixed(tmp_path):
    m = _run_evaluate(tmp_path, fixed=True)
    assert m["threshold_from"] == "fixed"
    assert m["models"]["M2"]["threshold_val"] == 0.5
    assert list(m["models"]["M2"]["splits"].keys()) == ["ext_x"]


def test_evaluate_threshold_from_default_and_present_only(tmp_path):
    m = _run_evaluate(tmp_path, fixed=False)
    assert m["threshold_from"] == "val real rows (max F1), per model"
    assert list(m["models"]["M2"]["splits"].keys()) == ["ext_x"]


def test_evaluate_drop_empty_filters_and_counts(tmp_path):
    truth = tmp_path / "truth.parquet"
    rows = [{"pair_id": f"p{i}", "label_stale": i % 2, "split": "ext_x",
             "sig_type": "unknown", "drift_type": "unknown", "source": "x",
             "diff_code": ("-a\n+b" if i else "  "),
             "doc_old": ("doc" if i else "")} for i in range(6)]
    pd.DataFrame(rows).to_parquet(truth, index=False)
    pred = tmp_path / "pred.csv"
    pd.DataFrame([{"pair_id": f"p{i}", "p_stale": 0.1 + 0.8 * (i % 2)}
                  for i in range(6)]).to_csv(pred, index=False)
    out_m, out_t = tmp_path / "m.json", tmp_path / "t.csv"
    argv = ["evaluate.py", "--dataset", str(truth), "--pred", f"M2={pred}",
            "--n_boot", "10", "--threshold_fixed", "0.5", "--drop_empty",
            "--out_metrics", str(out_m), "--out_table", str(out_t),
            "--figdir", str(tmp_path / "fig")]
    import sys as _sys
    from evaluate import main as ev_main
    old = _sys.argv
    _sys.argv = argv
    try:
        assert ev_main() == 0
    finally:
        _sys.argv = old
    m = json.loads(out_m.read_text(encoding="utf-8"))
    assert m["drop_empty"] is True and m["dropped_empty"] == 1
    assert m["dataset_rows"] == 5 and m["dataset_rows_total"] == 6
    assert m["models"]["M2"]["splits"]["ext_x"]["all"]["n"] == 5


def test_resolve_splits_present_only_and_excludes_train():
    assert resolve_splits({"train", "ext_a", "ext_b"}, "") == ["ext_a", "ext_b"]
    assert resolve_splits({"val", "ext_jit", "train"}, "") == ["val", "ext_jit"]
    assert resolve_splits(set(), "") == []
    assert resolve_splits({"val"}, "ext_x,val") == ["ext_x", "val"]


# ---------- external_adapter: jit owner, cocc helper ----------

def test_jit_project_keeps_dashed_owner():
    rec = {"id": "my-owner_repo-1692-Param-1", "label": "1",
           "comment_type": "Param", "old_comment_raw": "a",
           "new_comment_raw": "b", "old_code_raw": "c", "new_code_raw": "d"}
    assert jit_record_to_pair(rec)["project"] == "my-owner_repo"


def test_jit_project_short_id_fallback():
    rec = {"id": "abc", "label": "0", "comment_type": "Return",
           "old_comment_raw": "a", "new_comment_raw": "a",
           "old_code_raw": "c", "new_code_raw": "c"}
    assert jit_record_to_pair(rec)["project"] == "abc"


def test_cocc_docstring_helper():
    assert cocc_code_has_docstring_block('"""doc"""\nx=1', "y=2")
    assert cocc_code_has_docstring_block("x=1", 'y="""q"""')
    assert not cocc_code_has_docstring_block("x=1", "y=2")
    assert not cocc_code_has_docstring_block("", "")


# ---------- download_cocc: guard пустых строк ----------

def test_load_ids_skips_blank_and_bad_lines(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("fid1 0.py\n\n   \nfid2 12.py\nlonelyline\n", encoding="utf-8")
    pairs = download_cocc.load_ids(ids)
    assert pairs == [("fid1", "0.py"), ("fid2", "12.py")]


def test_load_ids_all_blank(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("\n   \n", encoding="utf-8")
    assert download_cocc.load_ids(ids) == []


# ---------- mine_go: shlex, команды ----------

def test_gominer_cmd_go_run_simple():
    assert mine_go.gominer_cmd("go run ./scripts/gominer", "/tmp/x") == [
        "go", "run", "./scripts/gominer", "-dir", "/tmp/x"]


def test_gominer_cmd_quoted_path_with_spaces():
    cmd = mine_go.gominer_cmd('go run "C:/my dir/gominer"', "/tmp/x")
    assert cmd == ["go", "run", "C:/my dir/gominer", "-dir", "/tmp/x"]


def test_gominer_cmd_plain_binary():
    assert mine_go.gominer_cmd("/tmp/gominer", "/tmp/x") == [
        "/tmp/gominer", "-dir", "/tmp/x"]


# ---------- make_go_silver: _matchkey с repo ----------

def _pool_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _silver_cand(repo="gin"):
    return {"doc_changed": True, "sig_changed": False, "file": "a.go",
            "symbol": "F", "commit_old": "c1", "commit_new": "c2",
            "sig_old": "s", "diff_code": "-a\n+b", "doc_old": "d",
            "repo": repo, "code_old": "o", "code_new": "n"}


def test_silver_matchkey_respects_repo_column(tmp_path, monkeypatch):
    pool = _pool_jsonl(tmp_path / "c.jsonl",
                       [_silver_cand("gin"), _silver_cand("cobra")])
    must = tmp_path / "must.csv"
    pd.DataFrame([{"id": "x", "file": "a.go", "symbol": "F",
                   "commit_old": "c1", "commit_new": "c2", "repo": "cobra",
                   "sig_old": "s", "diff_code": "-a\n+b", "doc_old": "d",
                   "label": "", "comment": ""}]).to_csv(must, index=False)
    out = tmp_path / "silver.csv"
    monkeypatch.setattr(sys, "argv", ["make_go_silver.py", "--pool", f"{pool}:gin",
                                      "--pool", f"{pool}:cobra",
                                      "--out", str(out), "--n", "4", "--seed", "1",
                                      "--must_include", str(must)])
    assert make_go_silver.main() == 0
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert (df["repo"] == "cobra").any()


def test_silver_must_without_repo_warns_and_picks(tmp_path, monkeypatch, capsys):
    pool = _pool_jsonl(tmp_path / "c.jsonl",
                       [_silver_cand("gin"), _silver_cand("cobra")])
    must = tmp_path / "must.csv"
    pd.DataFrame([{"id": "x", "file": "a.go", "symbol": "F",
                   "commit_old": "c1", "commit_new": "c2",
                   "sig_old": "s", "diff_code": "-a\n+b", "doc_old": "d",
                   "label": "", "comment": ""}]).to_csv(must, index=False)
    out = tmp_path / "silver.csv"
    monkeypatch.setattr(sys, "argv", ["make_go_silver.py", "--pool", f"{pool}:gin",
                                      "--pool", f"{pool}:cobra",
                                      "--out", str(out), "--n", "4", "--seed", "1",
                                      "--must_include", str(must)])
    assert make_go_silver.main() == 0
    assert "кандидатов пула" in capsys.readouterr().out


def test_silver_must_missing_repo_skipped(tmp_path, monkeypatch, capsys):
    pool = _pool_jsonl(tmp_path / "c.jsonl", [_silver_cand("gin")])
    must = tmp_path / "must.csv"
    pd.DataFrame([{"id": "x", "file": "a.go", "symbol": "F",
                   "commit_old": "c1", "commit_new": "c2", "repo": "cobra",
                   "sig_old": "s", "diff_code": "-a\n+b", "doc_old": "d",
                   "label": "", "comment": ""}]).to_csv(must, index=False)
    out = tmp_path / "silver.csv"
    monkeypatch.setattr(sys, "argv", ["make_go_silver.py", "--pool", f"{pool}:gin",
                                      "--out", str(out), "--n", "2", "--seed", "1",
                                      "--must_include", str(must)])
    assert make_go_silver.main() == 0
    assert "нет в пулах" in capsys.readouterr().out


# ---------- gominer: нет src в stdout, go 1.21 ----------

def test_gominer_stdout_has_no_src_field():
    src = (Path(__file__).resolve().parent.parent / "scripts" / "gominer" / "main.go"
           ).read_text(encoding="utf-8")
    assert '`json:"src"`' not in src
    assert "Src " not in src


def test_gominer_go_mod_pinned():
    mod = (Path(__file__).resolve().parent.parent / "scripts" / "gominer" / "go.mod"
           ).read_text(encoding="utf-8")
    assert "go 1.21" in mod
