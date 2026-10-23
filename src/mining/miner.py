"""Git history mining for FD-05 (R3).

Uses git CLI via subprocess (not PyDriller): lists commits, diffs
``.py`` files before/after each commit, compares functions present in
both versions and emits candidates with ``doc_changed == False`` only
(rows with changed docstrings are discarded per R3 role table).

Conventions (fixed):
  - ``symbol`` = qualified name (``Class.method``, nested via dots);
  - ``name`` = short ``ast.FunctionDef.name``;
  - only functions present in BOTH file versions are compared
    (added/removed functions are skipped);
  - ``n_src_lines`` and ``src_old`` mirror FD-04 adapter formulas.
"""

from __future__ import annotations

import ast
import copy
import subprocess
import time
from pathlib import Path

from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import (
    build_aux_source_code,
    compute_code_diff,
    extract_docstring_and_clean_node,
    extract_sig_line,
    normalize_code_from_node,
)

MAX_PY_PER_COMMIT = 50
MAX_FILE_BYTES = 300 * 1024
GIT_TIMEOUT = 120


def _git(repo_path, *args, timeout=GIT_TIMEOUT):
    cmd = ["git", "-c", "core.quotepath=off", "-C", str(repo_path), *args]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout or b""


def list_commits(repo_path, max_commits):
    out = _git(repo_path, "log", "--no-merges", "-n", str(max_commits), "--format=%H|%cI")
    commits = []
    for line in out.decode("utf-8", errors="replace").splitlines():
        if "|" in line:
            sha, date = line.split("|", 1)
            commits.append((sha.strip(), date.strip()))
    return commits


def changed_py_files(repo_path, sha):
    """Return modified ``.py`` paths, or None if the commit is skipped (>50)."""
    out = _git(repo_path, "diff-tree", "--no-commit-id", "--name-status", "-r", sha)
    paths = []
    for line in out.decode("utf-8", errors="replace").splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts
        if status == "M" and path.endswith(".py"):
            paths.append(path)
    if len(paths) > MAX_PY_PER_COMMIT:
        return None
    return paths


def show_blob(repo_path, sha, path, before):
    """File content as text, or None when missing/too big (root commits)."""
    rev = f"{sha}^:{path}" if before else f"{sha}:{path}"
    try:
        blob = _git(repo_path, "show", rev)
    except subprocess.CalledProcessError:
        return None
    if len(blob) > MAX_FILE_BYTES:
        return None
    # utf-8-sig strips a BOM when present and behaves like utf-8 otherwise;
    # a kept BOM char (\ufeff) makes ast.parse fail and drops the file.
    return blob.decode("utf-8-sig", errors="replace")


def iter_functions(tree):
    """Map qualname -> (short name, node); classes and defs nest via dots."""
    found = {}

    def visit(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, stack + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(stack + [child.name])
                found[qualname] = (child.name, child)
                visit(child, stack + [child.name])

    visit(tree, [])
    return found


def _local_binding_names(fn_node):
    """Names bound inside the function, order of first appearance.

    R2.7 locals: assignment/for/with/comprehension targets,
    walrus and ``del`` targets, ``except ... as`` names.
    Parameters, call targets and decorators are NOT locals.
    """
    names = []
    seen = set()

    def add(name):
        if name not in seen:
            seen.add(name)
            names.append(name)

    for child in ast.walk(fn_node):
        if isinstance(child, ast.Name) and isinstance(
            child.ctx, (ast.Store, ast.Del)
        ):
            add(child.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            add(child.name)
    return names


def cosmetic_equal(fn_old, fn_new):
    """R2.7: equal after locals -> v0, v1, ... in order of appearance.

    Only names BOUND inside the function are renamed (in all their
    occurrences). Free names -- call targets, decorators, attributes,
    parameters -- keep their identifiers, so renaming them is a
    behavior change and is NOT cosmetic.
    """

    def canonical(node):
        node = copy.deepcopy(node)
        mapping = {n: f"v{i}" for i, n in enumerate(_local_binding_names(node))}
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in mapping:
                child.id = mapping[child.id]
            elif isinstance(child, ast.ExceptHandler) and child.name in mapping:
                child.name = mapping[child.name]
        return ast.dump(node)

    return canonical(fn_old) == canonical(fn_new)


def compare_pair(fn_old, fn_new):
    """Normalize, categorize (R2.8), discard when docstring changed.

    Returns code/doc fields or None when skipped
    (unchanged code or ``doc_changed == True``).
    """
    doc_old = ast.get_docstring(fn_old, clean=True) or ""
    doc_new = ast.get_docstring(fn_new, clean=True) or ""
    if doc_old != doc_new:
        return None  # R3: doc_changed rows are covered by CoDocBench.
    _, old_no_doc = extract_docstring_and_clean_node(fn_old)
    _, new_no_doc = extract_docstring_and_clean_node(fn_new)
    code_old = normalize_code_from_node(old_no_doc)
    code_new = normalize_code_from_node(new_no_doc)
    if code_old == code_new:
        return None
    drift_type = derive_drift_type(extract_signature(fn_old), extract_signature(fn_new))
    if drift_type != "none":
        category = "signature_changed"
    elif cosmetic_equal(old_no_doc, new_no_doc):
        category = "cosmetic"
    else:
        category = "body_changed"
    return {
        "category": category,
        "drift_type": drift_type,
        "doc_old": doc_old.strip(),
        "code_old": code_old,
        "code_new": code_new,
        "sig_old": extract_sig_line(code_old),
        "diff_code": compute_code_diff(code_old, code_new),
        "src_old": build_aux_source_code(old_no_doc, doc_old),
        "n_src_lines": len(code_old.splitlines()) + len(doc_old.splitlines()),
    }


def mine_project(repo_path, project, max_commits, time_limit_s=1200, out_path=None,
                 progress_every=50):
    """Mine one repo; optionally save rows to ``out_path`` parquet.

    Every commit is wrapped in try/except: failures are counted, never raised.
    Stops early when ``time_limit_s`` is exceeded (``truncated=True``).
    """
    import pandas as pd

    stats = {
        "project": project,
        "commits_listed": 0,
        "commits_processed": 0,
        "commits_skipped_many_files": 0,
        "commit_errors": 0,
        "candidates": 0,
        "skipped_no_doc": 0,
        "cosmetic": 0,
        "body_changed": 0,
        "signature_changed": 0,
        "elapsed_s": 0.0,
        "truncated": False,
        "error_sample": [],
    }
    rows = []
    start = time.monotonic()
    try:
        commits = list_commits(repo_path, max_commits)
    except Exception as exc:  # noqa: BLE001 - repo-level failure goes to stats
        stats["error_sample"].append(f"list_commits: {exc!r}"[:300])
        stats["commit_errors"] += 1
        return stats
    stats["commits_listed"] = len(commits)

    for sha, date in commits:
        if time.monotonic() - start > time_limit_s:
            stats["truncated"] = True
            break
        try:
            paths = changed_py_files(repo_path, sha)
            if paths is None:
                stats["commits_skipped_many_files"] += 1
                continue
            for path in paths:
                old_text = show_blob(repo_path, sha, path, before=True)
                new_text = show_blob(repo_path, sha, path, before=False)
                if old_text is None or new_text is None:
                    continue
                try:
                    old_tree = ast.parse(old_text)
                    new_tree = ast.parse(new_text)
                except SyntaxError:
                    continue
                old_fns = iter_functions(old_tree)
                new_fns = iter_functions(new_tree)
                for qualname, (name, fn_old) in old_fns.items():
                    if qualname not in new_fns:
                        continue  # added/removed functions are out of scope
                    _, fn_new = new_fns[qualname]
                    if not isinstance(fn_new, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    cmp = compare_pair(fn_old, fn_new)
                    if cmp is None:
                        continue
                    if not cmp["doc_old"]:
                        # FD-06 requires docstring >= 40 chars; doc-less rows
                        # would all be dropped there, so skip them here.
                        stats["skipped_no_doc"] += 1
                        continue
                    rows.append({
                        "project": project,
                        "commit_sha": sha,
                        "date": date,
                        "file": path,
                        "symbol": qualname,
                        "name": name,
                        **cmp,
                        "doc_changed": False,
                    })
                    stats[cmp["category"]] += 1
            stats["commits_processed"] += 1
        except Exception as exc:  # noqa: BLE001 - per-commit isolation (R3)
            stats["commit_errors"] += 1
            if len(stats["error_sample"]) < 5:
                stats["error_sample"].append(f"{sha[:8]}: {exc!r}"[:300])
        if stats["commits_processed"] % progress_every == 0:
            print(f"  [{project}] {stats['commits_processed']}/{len(commits)} commits, "
                  f"{len(rows)} candidates", flush=True)

    stats["candidates"] = len(rows)
    stats["elapsed_s"] = round(time.monotonic() - start, 1)
    if out_path is not None and rows:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
    return stats
