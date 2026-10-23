"""Tests for FD-05 git mining (R3, R2.7-8)."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from src.mining.miner import (
    changed_py_files,
    compare_pair,
    cosmetic_equal,
    iter_functions,
    list_commits,
    mine_project,
    show_blob,
)

OLD = '''
class Store:
    """Store things."""

    def put(self, key, value="x"):
        """Put value by key.

        Args:
            key: the key to store under.
            value: the value. Defaults to x.
        """
        self._data[key] = value
        return True

def helper(items):
    """Sum items.

    Returns the total of all items given.
    """
    total = 0
    for i in items:
        total += i
    return total
'''


def _fn(tree_src, qualname):
    tree = ast.parse(tree_src)
    return iter_functions(tree)[qualname][1]


def test_iter_functions_qualnames():
    fns = iter_functions(ast.parse(OLD))
    assert set(fns) == {"Store.put", "helper"}
    assert fns["Store.put"][0] == "put"  # short name
    assert fns["helper"][0] == "helper"


def test_show_blob_strips_utf8_bom(monkeypatch):
    """Point 5: Windows BOM files must parse, not be skipped via SyntaxError."""
    import src.mining.miner as miner

    src = "def f(x):\n    '''Docstring here.'''\n    return x\n"
    monkeypatch.setattr(miner, "_git", lambda *a, **k: b"\xef\xbb\xbf" + src.encode())
    blob = show_blob("/fake/repo", "sha", "m.py", before=False)
    assert blob is not None
    assert not blob.startswith("\ufeff")
    ast.parse(blob)  # must not raise


def test_cosmetic_rename_local():
    new = OLD.replace("total = 0", "total_sum = 0").replace("total +=", "total_sum +=")
    new = new.replace("return total", "return total_sum")
    assert cosmetic_equal(_fn(OLD, "helper"), _fn(new, "helper")) is True


def test_cosmetic_body_change_is_not_cosmetic():
    new = OLD.replace("total += i", "total -= i")
    assert cosmetic_equal(_fn(OLD, "helper"), _fn(new, "helper")) is False


def test_compare_pair_signature_changed():
    new = OLD.replace('def put(self, key, value="x")', 'def put(self, key, val="x")')
    cmp = compare_pair(_fn(OLD, "Store.put"), _fn(new, "Store.put"))
    assert cmp is not None and cmp["category"] == "signature_changed"
    assert cmp["drift_type"] == "param_renamed"


def test_compare_pair_cosmetic():
    new = OLD.replace("self._data[key] = value", "self._data[key]=value")
    # whitespace-only change vanishes after normalization -> unchanged
    assert compare_pair(_fn(OLD, "Store.put"), _fn(new, "Store.put")) is None
    renamed = OLD.replace("total = 0", "acc = 0").replace("total +=", "acc +=")
    renamed = renamed.replace("for i in items:\n        acc += i", "for i in items:\n        acc += i")
    renamed = renamed.replace("return total", "return acc")
    cmp = compare_pair(_fn(OLD, "helper"), _fn(renamed, "helper"))
    assert cmp is not None and cmp["category"] == "cosmetic"
    assert cmp["drift_type"] == "none"


def test_compare_pair_body_changed():
    new = OLD.replace("total += i", "total += 2 * i")
    cmp = compare_pair(_fn(OLD, "helper"), _fn(new, "helper"))
    assert cmp is not None and cmp["category"] == "body_changed"


def test_compare_pair_doc_changed_discarded():
    new = OLD.replace("Sum items.", "Sum items quickly.")
    assert compare_pair(_fn(OLD, "helper"), _fn(new, "helper")) is None


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "init")
    (repo / "a.py").write_text(OLD, encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "add a")
    return repo


def test_list_and_diff_plumbing(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text(OLD.replace("total += i", "total += 2 * i"), encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "tweak")
    commits = list_commits(repo, 10)
    assert len(commits) == 3  # init + add + tweak
    sha = commits[0][0]
    assert changed_py_files(repo, sha) == ["a.py"]


def test_mine_project_end_to_end(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text(OLD.replace("total += i", "total += 2 * i"), encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "tweak")
    stats = mine_project(repo, "demo", max_commits=10, time_limit_s=300)
    assert stats["commits_processed"] == 3
    assert stats["commit_errors"] == 0
    assert stats["body_changed"] == 1
    assert stats["candidates"] == 1


def test_mine_project_doc_change_yields_nothing(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text(OLD.replace("Sum items.", "Sum items fast."), encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "docs")
    stats = mine_project(repo, "demo", max_commits=10, time_limit_s=300)
    assert stats["candidates"] == 0
