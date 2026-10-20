from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "node_modules",
}


def _extract_call_names(node: ast.Call) -> str | None:
    """Extract symbol name from an ast.Call node (ast.Name or ast.Attribute)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def extract_leaf_symbol(symbol: str) -> str:
    """Extract leaf identifier from qualified name, e.g. 'Client.fetch' -> 'fetch'."""
    return symbol.rsplit(".", 1)[-1].strip()


def parse_fan_in_from_source(source_code: str) -> Counter[str]:
    """Extract call count for all symbols called in source code.

    Approximation per R12: identical names across different modules are merged.
    """
    counts: Counter[str] = Counter()
    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError):
        return counts

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _extract_call_names(node)
            if name:
                counts[name] += 1
    return counts


def analyze_repo_fan_in(
    repo_path: Path | str,
    target_symbols: Iterable[str] | None = None,
) -> dict[str, int]:
    """Scan all Python files in a repository or directory and calculate fan-in counts.

    Per R12:
      fan_in is the approximate number of calls to the function by name across
      the repository. Same-named symbols across modules are merged.

    Supports qualified symbol names in target_symbols (e.g. 'Client.fetch').
    """
    base = Path(repo_path)
    if not base.exists():
        return {}

    target_list = list(target_symbols) if target_symbols is not None else None
    targets_leaf_map: dict[str, str] = {}
    leaf_set: set[str] = set()

    if target_list is not None:
        for sym in target_list:
            leaf = extract_leaf_symbol(sym)
            targets_leaf_map[sym] = leaf
            if leaf:
                leaf_set.add(leaf)

    total_counts: Counter[str] = Counter()

    if base.is_file():
        file_paths = [base] if base.suffix == ".py" else []
    else:
        file_paths = []
        for p in base.rglob("*.py"):
            # Skip ignored directories
            if any(part in IGNORED_DIRS for part in p.parts):
                continue
            file_paths.append(p)

    for p in file_paths:
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            file_counts = parse_fan_in_from_source(content)
            if target_list is not None:
                for leaf in leaf_set:
                    if leaf in file_counts:
                        total_counts[leaf] += file_counts[leaf]
            else:
                total_counts.update(file_counts)
        except Exception:
            continue

    if target_list is not None:
        return {sym: total_counts.get(targets_leaf_map[sym], 0) for sym in target_list}
    return dict(total_counts)


def count_symbol_calls(repo_path: Path | str, symbol: str) -> int:
    """Return total call count of a symbol across repository.

    Supports qualified names (e.g. 'Client.fetch' -> counts calls to 'fetch').
    """
    if not symbol:
        return 0
    result = analyze_repo_fan_in(repo_path, target_symbols=[symbol])
    return result.get(symbol, 0)
