from __future__ import annotations

import ast
import copy
import difflib
import textwrap
from typing import Optional, Tuple


def parse_function_ast(code_str: str) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    if not code_str or not code_str.strip():
        return None
    try:
        dedented = textwrap.dedent(code_str).strip()
        tree = ast.parse(dedented)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
    except (SyntaxError, ValueError):
        return None
    return None


def extract_docstring_and_clean_node(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Tuple[Optional[str], ast.FunctionDef | ast.AsyncFunctionDef]:
    cleaned_node = copy.deepcopy(fn_node)
    docstring = ast.get_docstring(fn_node, clean=True)

    if cleaned_node.body and isinstance(cleaned_node.body[0], ast.Expr):
        val_node = cleaned_node.body[0].value
        if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
            cleaned_node.body.pop(0)

    if not cleaned_node.body:
        cleaned_node.body.append(ast.Pass())

    return docstring, cleaned_node


def normalize_code_from_node(fn_node_no_doc: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ast.unparse(fn_node_no_doc).strip()


def extract_sig_line(normalized_code: str) -> str:
    for line in normalized_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            return stripped
    return normalized_code.splitlines()[0] if normalized_code else ""


def build_aux_source_code(
    fn_node_no_doc: ast.FunctionDef | ast.AsyncFunctionDef,
    docstring: str,
) -> str:
    node_with_doc = copy.deepcopy(fn_node_no_doc)
    if docstring and docstring.strip():
        doc_node = ast.Expr(value=ast.Constant(value=docstring.strip()))
        node_with_doc.body.insert(0, doc_node)
    return ast.unparse(node_with_doc).strip()


def compute_code_diff(code_old: str, code_new: str) -> str:
    old_lines = [line + "\n" for line in code_old.splitlines()]
    new_lines = [line + "\n" for line in code_new.splitlines()]
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="a/code.py",
            tofile="b/code.py",
            lineterm="",
        )
    )
    return "\n".join(diff)