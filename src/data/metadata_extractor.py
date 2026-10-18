from __future__ import annotations

import ast
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def dedent_code(code_str: str) -> str:
    """Dedent and strip code string for reliable parsing."""
    return textwrap.dedent(code_str).strip()


def parse_function_ast(code_str: str) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Parse source string and return the first function definition node.

    Returns ``None`` if parsing fails or no function is found.
    """
    if not code_str or not code_str.strip():
        return None
    try:
        tree = ast.parse(dedent_code(code_str))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
    except (SyntaxError, ValueError):
        return None
    return None


def func_length(code_str: str) -> int:
    """Return the number of non‑empty lines of the function body.

    The function is first parsed to obtain its AST node, then the original
    source is split into lines. Empty lines are ignored to match typical
    length statistics used in the project.
    """
    node = parse_function_ast(code_str)
    if node is None:
        return 0
    # Extract source lines for the node via ``ast.get_source_segment`` if available
    # Fallback to manual slicing based on ``node.lineno`` and ``node.end_lineno``.
    try:
        source = ast.get_source_segment(dedent_code(code_str), node)
    except Exception:
        source = None
    if source is None:
        lines = dedent_code(code_str).splitlines()
    else:
        lines = source.splitlines()
    return sum(1 for line in lines if line.strip())


def param_count(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count total parameters (positional, var‑args, kw‑only and kw‑var‑args)."""
    args = fn_node.args
    total = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    if args.vararg:
        total += 1
    if args.kwarg:
        total += 1
    return total


def has_section(docstring: str, section: str) -> bool:
    """Check whether a docstring contains a given section header.

    Section detection is case‑insensitive and tolerant to leading whitespace.
    Example sections: ``"Args"``, ``"Returns"``, ``"Raises"``.
    """
    if not docstring:
        return False
    pattern = rf"^\s*{re.escape(section)}\s*:"
    return any(re.search(pattern, line, flags=re.IGNORECASE) for line in docstring.splitlines())


def extract_metadata(code_str: str, docstring: str) -> Dict[str, Any]:
    """Collect all metadata required for FD‑15.

    Returns a dictionary with keys:
    - ``len``: function length (lines)
    - ``params``: total number of parameters
    - ``has_args`` / ``has_returns`` / ``has_raises``: bool flags for docsections
    """
    node = parse_function_ast(code_str)
    if node is None:
        return {
            "len": 0,
            "params": 0,
            "has_args": False,
            "has_returns": False,
            "has_raises": False,
        }
    return {
        "len": func_length(code_str),
        "params": param_count(node),
        "has_args": has_section(docstring, "Args"),
        "has_returns": has_section(docstring, "Returns"),
        "has_raises": has_section(docstring, "Raises"),
    }


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach metadata columns required for FD‑15.

    Extracts features from ``code_old`` and ``doc_old``:
    - ``func_len``: number of non-empty lines in function
    - ``param_cnt``: total parameter count
    - ``has_args`` / ``has_returns`` / ``has_raises``: docstring sections
    """
    func_len = []
    param_cnt = []
    has_args = []
    has_returns = []
    has_raises = []

    for _, row in df.iterrows():
        code = str(row.get("code_old", ""))
        doc = str(row.get("doc_old", ""))
        meta = extract_metadata(code, doc)
        func_len.append(meta.get("len", 0))
        param_cnt.append(meta.get("params", 0))
        has_args.append(meta.get("has_args", False))
        has_returns.append(meta.get("has_returns", False))
        has_raises.append(meta.get("has_raises", False))

    df = df.copy()
    df["func_len"] = func_len
    df["param_cnt"] = param_cnt
    df["has_args"] = has_args
    df["has_returns"] = has_returns
    df["has_raises"] = has_raises
    return df
