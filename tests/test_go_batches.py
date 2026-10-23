"""FD-39 фаза 1: тесты батчей и сведения Go-пилота (фикстуры, без реальных данных)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import combine_go  # noqa: E402
import make_go_batches  # noqa: E402


def _pilot_df(n=12):
    return pd.DataFrame([{
        "id": f"go{i:03d}", "file": "a.go", "symbol": "F",
        "commit_old": "c1", "commit_new": "c2",
        "sig_old": "func F() int", "diff_code": "-a\n+b",
        "doc_old": "Does things well.", "label": "", "comment": "",
    } for i in range(n)])


def test_make_batches_cover_and_blind(tmp_path, monkeypatch):
    pilot = tmp_path / "input.csv"
    _pilot_df(12).to_csv(pilot, index=False)
    out = tmp_path / "b"
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(pilot),
                                      "--out_dir", str(out), "--per_batch", "6"])
    assert make_go_batches.main() == 0
    prompts = sorted(out.glob("go_batch*.prompt.txt"))
    assert len(prompts) == 2
    text = prompts[0].read_text(encoding="utf-8")
    assert "=== CASE id=go000 ===" in text and "DOC_OLD" in text
    assert "doc_changed_proxy" not in text and "p_stale" not in text


def test_make_batches_rejects_leak_and_empty_doc(tmp_path, monkeypatch):
    df = _pilot_df(2)
    df["proxy"] = "x"
    bad = tmp_path / "p.csv"
    df.to_csv(bad, index=False)
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(bad),
                                      "--out_dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        make_go_batches.main()
    df2 = _pilot_df(2)
    df2.loc[0, "doc_old"] = "  "
    bad2 = tmp_path / "p2.csv"
    df2.to_csv(bad2, index=False)
    monkeypatch.setattr(sys, "argv", ["make_go_batches.py", "--pilot", str(bad2),
                                      "--out_dir", str(tmp_path / "o2")])
    with pytest.raises(SystemExit):
        make_go_batches.main()


def _llm_df(rows):
    return pd.DataFrame([{"id": i, "label": lab, "stale_statement": st, "why": "w"}
                         for i, lab, st in rows])


def test_combine_human_only_disputes_and_final(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.csv"
    _pilot_df(4).to_csv(pilot, index=False)
    spark = tmp_path / "s.csv"
    gemini = tmp_path / "g.csv"
    _llm_df([("go000", "A", "Does things well."), ("go001", "C", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(spark, index=False)
    _llm_df([("go000", "A", "Does things well."), ("go001", "B", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(gemini, index=False)
    human = tmp_path / "h.csv"
    pd.DataFrame([{"id": "go001", "label": "B"}]).to_csv(human, index=False)
    out_c, out_d = tmp_path / "c.csv", tmp_path / "d.csv"
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(spark),
                                      "--gemini", str(gemini), "--input", str(pilot),
                                      "--human", str(human),
                                      "--out_combined", str(out_c),
                                      "--out_disputes", str(out_d)])
    assert combine_go.main() == 0
    comb = pd.read_csv(out_c, dtype=str, keep_default_na=False)
    assert set(("human", "final")) <= set(comb.columns)
    row = comb.loc[comb["id"] == "go001"].iloc[0]
    assert row["consensus"] == "dispute" and row["human"] == "B" and row["final"] == "B"
    agree = comb.loc[comb["id"] == "go002"].iloc[0]
    assert agree["consensus"] == "C" and agree["final"] == "C"


def test_combine_human_rejects_agreement_labels(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.csv"
    _pilot_df(4).to_csv(pilot, index=False)
    spark = tmp_path / "s.csv"
    gemini = tmp_path / "g.csv"
    _llm_df([("go000", "A", "Does things well."), ("go001", "C", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(spark, index=False)
    _llm_df([("go000", "A", "Does things well."), ("go001", "B", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(gemini, index=False)
    human = tmp_path / "h.csv"
    pd.DataFrame([{"id": "go002", "label": "B"}]).to_csv(human, index=False)
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(spark),
                                      "--gemini", str(gemini), "--input", str(pilot),
                                      "--human", str(human),
                                      "--out_combined", str(tmp_path / "c.csv"),
                                      "--out_disputes", str(tmp_path / "d.csv")])
    with pytest.raises(SystemExit):
        combine_go.main()


def test_combine_export_blind(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.csv"
    _pilot_df(4).to_csv(pilot, index=False)
    spark = tmp_path / "s.csv"
    gemini = tmp_path / "g.csv"
    _llm_df([("go000", "A", "Does things well."), ("go001", "C", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(spark, index=False)
    _llm_df([("go000", "A", "Does things well."), ("go001", "B", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(gemini, index=False)
    out_c, out_d = tmp_path / "c.csv", tmp_path / "d.csv"
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(spark),
                                      "--gemini", str(gemini), "--input", str(pilot),
                                      "--out_combined", str(out_c),
                                      "--out_disputes", str(out_d)])
    assert combine_go.main() == 0
    blind = tmp_path / "blind.csv"
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--export-blind",
                                      "--out_disputes", str(out_d),
                                      "--out_blind", str(blind)])
    assert combine_go.main() == 0
    df = pd.read_csv(blind, dtype=str, keep_default_na=False)
    assert df["id"].tolist() == ["go001"]
    assert (df["label"] == "").all() and (df["stale_statement"] == "").all()
    leak = ({"spark", "gemini", "consensus", "human", "final", "proxy",
             "label_stale", "p_stale"} & set(df.columns))
    assert not leak, f"в blind утечка колонок: {sorted(leak)}"


def test_combine_agreement_and_quotes(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.csv"
    _pilot_df(4).to_csv(pilot, index=False)
    spark = tmp_path / "s.csv"
    gemini = tmp_path / "g.csv"
    _llm_df([("go000", "A", "Does things well."), ("go001", "C", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(spark, index=False)
    _llm_df([("go000", "A", "Does things well."), ("go001", "B", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(gemini, index=False)
    out_c, out_d = tmp_path / "c.csv", tmp_path / "d.csv"
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(spark),
                                      "--gemini", str(gemini), "--pilot", str(pilot),
                                      "--out_combined", str(out_c),
                                      "--out_disputes", str(out_d)])
    assert combine_go.main() == 0
    comb = pd.read_csv(out_c, dtype=str)
    assert len(comb) == 4
    assert comb.loc[comb["id"] == "go001", "consensus"].iloc[0] == "dispute"
    disp = pd.read_csv(out_d, dtype=str)
    assert disp["id"].tolist() == ["go001"]
    # битая цитата отклоняется
    bad = tmp_path / "bad.csv"
    _llm_df([("go000", "A", "не из доки"), ("go001", "C", ""),
             ("go002", "C", ""), ("go003", "B", "")]).to_csv(bad, index=False)
    monkeypatch.setattr(sys, "argv", ["combine_go.py", "--spark", str(bad),
                                      "--gemini", str(gemini), "--pilot", str(pilot),
                                      "--out_combined", str(out_c),
                                      "--out_disputes", str(out_d)])
    with pytest.raises(SystemExit):
        combine_go.main()
