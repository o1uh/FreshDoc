"""FD-39 фаза 1: тесты pairing Go-карт (фикстуры, без git/go)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_go_silver  # noqa: E402
from src.mining.go_pair import pair_maps, select_pilot, stability  # noqa: E402


def _m(sig="func F(a int) int", doc="Doc.", code="func F(a int) int {\nreturn a\n}"):
    return {"sig": sig, "doc": doc, "code": code}


def test_pair_changed_and_flags():
    old = {("f.go", "F"): _m()}
    new = {("f.go", "F"): _m(doc="Doc changed.")}
    c = pair_maps(old, new, "c1", "c2")
    assert len(c) == 1 and c[0]["doc_changed"] and not c[0]["code_changed"]
    assert c[0]["doc_old"] == "Doc." and c[0]["commit_new"] == "c2"
    assert "//" not in c[0]["diff_code"]  # дифф без godoc


def test_pair_unchanged_skipped_and_sig_flag():
    old = {("f.go", "F"): _m(), ("g.go", "G"): _m()}
    new = {
        ("f.go", "F"): _m(),
        ("g.go", "G"): _m(sig="func G(a string) int",
                          code="func G(a string) int {\nreturn 1\n}"),
    }
    c = pair_maps(old, new, "c1", "c2")
    assert len(c) == 1 and c[0]["symbol"] == "G"
    assert c[0]["sig_changed"] and c[0]["code_changed"]


def test_select_pilot_stratified():
    cands = (
        [{"doc_changed": True, "i": i} for i in range(20)]
        + [{"doc_changed": False, "i": i} for i in range(20)]
    )
    p = select_pilot(cands, n=30, seed=1)
    assert len(p) == 30
    assert sum(1 for x in p if x["doc_changed"]) == 15
    # детерминированность
    assert p == select_pilot(cands, n=30, seed=1)


def test_select_pilot_best_effort_when_minority_short():
    cands = ([{"doc_changed": True, "i": i} for i in range(4)]
             + [{"doc_changed": False, "i": i} for i in range(40)])
    p = select_pilot(cands, n=30, seed=1)
    assert len(p) == 30
    assert sum(1 for x in p if x["doc_changed"]) == 4  # всё меньшинство + добор


def test_stability():
    s = stability({1, 2, 3}, {2, 3, 4})
    assert abs(s["kept"] - 2 / 3) < 1e-9
    assert stability(set(), {1})["kept"] == 0.0


def test_with_doc_filters_empty():
    from src.mining.go_pair import judgeable, with_doc
    cands = [{"doc_old": "x", "diff_code": "-a\n+b"}, {"doc_old": "  ", "diff_code": "d"},
             {"doc_old": "", "diff_code": "d"}, {}]
    assert with_doc(cands) == [{"doc_old": "x", "diff_code": "-a\n+b"}]
    assert judgeable([{"doc_old": "x", "diff_code": "  "},
                      {"doc_old": "x", "diff_code": "-a\n+b"}]) == [
        {"doc_old": "x", "diff_code": "-a\n+b"}]


def _silver_cands():
    return ([{"doc_changed": True, "sig_changed": False, "file": "a.go",
              "symbol": "F", "commit_old": "c1", "commit_new": "c2",
              "sig_old": "s", "diff_code": "-a\n+b", "doc_old": "d"}
             for _ in range(5)]
            + [{"doc_changed": False, "sig_changed": True, "file": "b.go",
                "symbol": "G", "commit_old": "c1", "commit_new": "c2",
                "sig_old": "s", "diff_code": "-x\n+y", "doc_old": "e"}
               for _ in range(10)])


def test_make_go_silver_end_to_end(tmp_path, monkeypatch):
    import json as _json
    pool = tmp_path / "cands.jsonl"
    pool.write_text("\n".join(_json.dumps(c) for c in _silver_cands()), encoding="utf-8")
    out = tmp_path / "silver.csv"
    monkeypatch.setattr(sys, "argv", ["make_go_silver.py", "--pool", f"{pool}:gin",
                                      "--out", str(out), "--n", "12", "--seed", "1"])
    assert make_go_silver.main() == 0
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(df) == 12 and df["id"].is_unique
    assert set(df.columns) >= {"id", "repo", "label", "comment", "code_old", "code_new"}
    assert (df["label"].str.strip() == "").all()
    side = _json.loads((tmp_path / "silver_sidecar.json").read_text(encoding="utf-8"))
    assert len(side) == 12 and sum(1 for s in side if s["doc_changed_proxy"]) == 5


def test_make_go_silver_must_include(tmp_path, monkeypatch):
    import json as _json
    pool = tmp_path / "cands.jsonl"
    pool.write_text("\n".join(_json.dumps(c) for c in _silver_cands()), encoding="utf-8")
    must = tmp_path / "must.csv"
    rows = _silver_cands()
    pd.DataFrame([{"id": "x", "file": "a.go", "symbol": "F",
                   "commit_old": "c1", "commit_new": "c2",
                   "sig_old": "s", "diff_code": "-a\n+b", "doc_old": "d",
                   "label": "", "comment": ""}]).to_csv(must, index=False)
    _ = rows
    out = tmp_path / "silver.csv"
    monkeypatch.setattr(sys, "argv", ["make_go_silver.py", "--pool", f"{pool}:gin",
                                      "--out", str(out), "--n", "8", "--seed", "1",
                                      "--must_include", str(must)])
    assert make_go_silver.main() == 0
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(df) == 8
    assert ((df["file"] == "a.go") & (df["symbol"] == "F")).any()
