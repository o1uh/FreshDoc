from __future__ import annotations

import ast
from typing import Any, Dict, List, Set, Tuple


def _get_raises(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
    raises: Set[str] = set()
    stack = list(fn_node.body)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(curr, ast.Raise):
            if curr.exc is None:
                raises.add("re-raise")
            elif isinstance(curr.exc, ast.Name):
                raises.add(curr.exc.id)
            elif isinstance(curr.exc, ast.Call):
                if isinstance(curr.exc.func, ast.Name):
                    raises.add(curr.exc.func.id)
                elif isinstance(curr.exc.func, ast.Attribute):
                    raises.add(ast.unparse(curr.exc.func))
                else:
                    raises.add(ast.unparse(curr.exc.func))
            else:
                raises.add(ast.unparse(curr.exc))
        for child in ast.iter_child_nodes(curr):
            stack.append(child)
    return sorted(list(raises))


def extract_signature(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, Any]:
    args_obj = fn_node.args
    pos_args = args_obj.posonlyargs + args_obj.args
    n_pos = len(pos_args)
    n_defaults = len(args_obj.defaults)

    params: List[Tuple[str, str]] = []
    for i, arg in enumerate(pos_args):
        default_idx = i - (n_pos - n_defaults)
        if default_idx >= 0:
            default_val = ast.unparse(args_obj.defaults[default_idx]).strip()
        else:
            default_val = ""
        params.append((arg.arg, default_val))

    if args_obj.vararg:
        params.append((f"*{args_obj.vararg.arg}", ""))

    for arg, default in zip(args_obj.kwonlyargs, args_obj.kw_defaults):
        default_val = ast.unparse(default).strip() if default is not None else ""
        params.append((arg.arg, default_val))

    if args_obj.kwarg:
        params.append((f"**{args_obj.kwarg.arg}", ""))

    returns = ast.unparse(fn_node.returns).strip() if fn_node.returns is not None else ""
    raises = _get_raises(fn_node)

    return {
        "params": params,
        "returns": returns,
        "raises": raises,
    }


def derive_drift_type(old: Dict[str, Any], new: Dict[str, Any]) -> str:
    no, nn = [p[0] for p in old["params"]], [p[0] for p in new["params"]]
    removed = [n for n in no if n not in nn]
    added = [n for n in nn if n not in no]

    found: List[str] = []
    if removed and added and len(removed) == len(added):
        found.append("param_renamed")
    else:
        if removed:
            found.append("param_removed")
        if added:
            found.append("param_added")

    do, dn = dict(old["params"]), dict(new["params"])
    if any(do[n] != dn[n] for n in no if n in dn):
        found.append("default_changed")

    if old["returns"] != new["returns"]:
        found.append("return_changed")

    if set(old["raises"]) != set(new["raises"]):
        found.append("exception_changed")

    order = [
        "param_renamed",
        "param_removed",
        "param_added",
        "default_changed",
        "return_changed",
        "exception_changed",
    ]
    matched = [t for t in order if t in found]
    return matched[0] if matched else "none"