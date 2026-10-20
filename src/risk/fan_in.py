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
    """
    base = Path(repo_path)
    if not base.exists():
        return {}

    targets = set(target_symbols) if target_symbols is not None else None
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
            if targets is not None:
                for sym in targets:
                    if sym in file_counts:
                        total_counts[sym] += file_counts[sym]
            else:
                total_counts.update(file_counts)
        except Exception:
            continue

    if targets is not None:
        return {sym: total_counts.get(sym, 0) for sym in targets}
    return dict(total_counts)


def count_symbol_calls(repo_path: Path | str, symbol: str) -> int:
    """Return total call count of a symbol across repository."""
    result = analyze_repo_fan_in(repo_path, target_symbols=[symbol])
    return result.get(symbol, 0)
