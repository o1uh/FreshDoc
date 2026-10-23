"""FD-39: тесты адаптера внешних наборов (фикстуры, без реальных данных)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate import resolve_splits  # noqa: E402
from src.data.external_adapter import (  # noqa: E402
    PAIR_COLUMNS,
    cocc_record_to_pair,
    extract_method_name,
    generic_pair,
    go_record_to_pair,
    hebcup_record_to_pair,
    java_sig_heuristic,
    jit_record_to_pair,
    raw_unified_diff,
    split_cocc_blocks,
)


def _hebcup_rec():
    return {
        "sample_id": 1,
        "full_name": "owner/repo",
        "commit_id": "abc123",
        "src_method": "public Interface getInterface() {\n        return theInterface;\n    }",
        "dst_method": "public Interface getInterface() {\n        return this.onosInterface;\n    }",
        "src_javadoc": "Get the PIM Interface.",
        "dst_javadoc": "Return the ONOS Interface.",
        "src_desc": "Get the PIM Interface.",
        "dst_desc": "Return the ONOS Interface.",
    }


def _jit_rec(label="1"):
    return {
        "id": "Owner_repo-1692-Param-1",
        "label": label,
        "comment_type": "Param",
        "old_comment_raw": "@param file file to upload",
        "new_comment_raw": "@param requiredFile file to upload",
        "old_code_raw": "public void upload(File file) {\n}",
        "new_code_raw": "public void upload(File requiredFile) {\n}",
    }


def test_hebcup_positive_unknown_no_filters():
    row = hebcup_record_to_pair(_hebcup_rec())
    assert row["label_stale"] == 1
    assert row["drift_type"] == "unknown" and row["sig_type"] == "unknown"
    assert row["weight"] == 0.0
    assert row["split"] == "ext_hebcup"
    assert set(PAIR_COLUMNS) <= set(row.keys())
    assert "theInterface" in row["diff_code"] and row["doc_old"].startswith("Get the PIM")
    assert row["symbol"].startswith("getInterface#")


def test_jit_labels_from_box():
    pos = jit_record_to_pair(_jit_rec("1"))
    neg = jit_record_to_pair(_jit_rec("0"))
    assert (pos["label_stale"], neg["label_stale"]) == (1, 0)
    assert pos["drift_type"] == "unknown" and pos["weight"] == 0.0
    # id 'Owner_repo-1692-Param-1': отрезаются 3 хвостовых сегмента
    assert pos["project"] == "Owner_repo"


def test_pair_id_deterministic_and_diff():
    a = generic_pair(
        source="ext_x", project="p", commit_sha="c", symbol="s",
        code_old="a\nb\n", code_new="a\nc\n", doc_old="d",
        label_stale=1, label_source="ls", split="ext_x",
    )
    b = generic_pair(
        source="ext_x", project="p", commit_sha="c", symbol="s",
        code_old="a\nb\n", code_new="a\nc\n", doc_old="d",
        label_stale=1, label_source="ls", split="ext_x",
    )
    assert a["pair_id"] == b["pair_id"]
    assert "-b" in a["diff_code"] and "+c" in a["diff_code"]
    assert raw_unified_diff("same\n", "same\n") == ""


def test_sig_heuristics():
    assert java_sig_heuristic("") == ""
    assert "upload" in java_sig_heuristic("public void upload(File f) {\n}")
    assert extract_method_name("public Interface getInterface() {") == "getInterface"


def test_resolve_splits_standard_and_external():
    assert resolve_splits({"train", "val", "test_real", "test_synth", "gold"}, "") == [
        "val", "test_real", "test_synth", "gold",
    ]
    # FD-39 fix: только присутствующие в данных сплиты (без пустых EVAL_SPLITS)
    assert resolve_splits({"ext_hebcup", "ext_jit"}, "") == [
        "ext_hebcup", "ext_jit",
    ]
    assert resolve_splits({"val", "ext_go_silver"}, "") == [
        "val", "ext_go_silver",
    ]
    assert resolve_splits({"val", "ext_jit"}, "ext_jit") == ["ext_jit"]


def _cocc_changed():
    return ("0.py",
            "# Test alteration\nwith editor:\n    op(1)\n"
            "--------------------------------\n"
            "# Test alteration v2\nwith editor:\n    op(2)\n"
            "======================\n")


def _cocc_same():
    return ("1.py",
            "# Same note\nx = 1\n"
            "------------\n"
            "# Same note\nx = 2\n")


def test_cocc_label_from_comment_equality():
    name, text = _cocc_changed()
    row = cocc_record_to_pair(name, text)
    assert row is not None and row["label_stale"] == 1
    assert row["source"] == "ext_python2" and row["split"] == "ext_python2"
    assert row["drift_type"] == "unknown" and row["weight"] == 0.0
    assert "Test alteration" in row["doc_old"]
    assert row["doc_old"] != row["_leak_doc_new"]
    row2 = cocc_record_to_pair(*_cocc_same())
    assert row2 is not None and row2["label_stale"] == 0


def test_cocc_split_blocks_indented_sep():
    old, new = split_cocc_blocks(
        "# c\ncode()\n    ----------------\n# c2\ncode2()\n    ==============\n")
    assert old == "# c\ncode()"
    assert new == "# c2\ncode2()"
    assert split_cocc_blocks("no separator here") == ("no separator here", "")
    assert cocc_record_to_pair("x.py", "") is None


def test_go_record_uses_proxy_and_tags():
    row = {"repo": "gin", "symbol": "Ctx.ShouldBind", "commit_old": "a", "commit_new": "b",
           "sig_old": "func (c *Context) ShouldBind(obj any) error",
           "code_old": "func (c *Context) ShouldBind(obj any) error {\nreturn nil\n}",
           "code_new": "func (c *Context) ShouldBind(obj any) error {\nreturn err\n}",
           "doc_old": "ShouldBind checks the method."}
    pos = go_record_to_pair(row, 1)
    assert pos["source"] == "ext_go_silver" and pos["split"] == "ext_go_silver"
    assert pos["label_stale"] == 1 and pos["label_source"] == "go_silver_proxy"
    assert pos["weight"] == 0.0 and pos["drift_type"] == "unknown"
    assert "ShouldBind" in pos["sig_old"] and "return err" in pos["diff_code"]
    neg = go_record_to_pair(row, 0)
    assert neg["label_stale"] == 0
    assert neg["pair_id"] == pos["pair_id"]  # метка не входит в identity
