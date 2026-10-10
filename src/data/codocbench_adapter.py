from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import (
    build_aux_source_code,
    compute_code_diff,
    extract_docstring_and_clean_node,
    extract_sig_line,
    normalize_code_from_node,
    parse_function_ast,
)


def resolve_json_path(data: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    curr = data
    for part in parts:
        if isinstance(curr, dict):
            curr = curr.get(part)
        elif isinstance(curr, (list, tuple)):
            try:
                idx = int(part)
                curr = curr[idx] if 0 <= idx < len(curr) else None
            except ValueError:
                return None
        else:
            return None
        if curr is None:
            return None
    return curr


def make_pair_id(source: str, project: str, symbol: str, commit_sha: str, code_old_norm: str) -> str:
    key = f"{source}:{project}:{symbol}:{commit_sha}:{code_old_norm}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def make_function_hash(code_old_norm: str) -> str:
    return hashlib.sha256(code_old_norm.encode("utf-8")).hexdigest()[:16]


class CoDocBenchAdapter:
    def __init__(self, mapping_config_path: str | Path):
        with open(mapping_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def process_record(
        self,
        raw_record: Dict[str, Any],
        split_pool: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        project = resolve_json_path(raw_record, self.config["project"])
        symbol = resolve_json_path(raw_record, self.config["symbol"]) or "unknown"
        old_code_raw = resolve_json_path(raw_record, self.config["old_code"])
        new_code_raw = resolve_json_path(raw_record, self.config["new_code"])
        commit_sha = resolve_json_path(raw_record, self.config["commit_sha"]) or ""
        old_docstring_raw = resolve_json_path(raw_record, self.config.get("old_docstring", ""))

        if not project or not old_code_raw or not new_code_raw:
            return None, "нет_обязательных_полей"

        fn_old = parse_function_ast(old_code_raw)
        fn_new = parse_function_ast(new_code_raw)
        if fn_old is None or fn_new is None:
            return None, "не_разобралось"

        ast_doc_old, fn_old_no_doc = extract_docstring_and_clean_node(fn_old)
        _, fn_new_no_doc = extract_docstring_and_clean_node(fn_new)

        doc_old = (ast_doc_old or old_docstring_raw or "").strip()

        code_old = normalize_code_from_node(fn_old_no_doc)
        code_new = normalize_code_from_node(fn_new_no_doc)

        if code_old == code_new:
            return None, "код_не_изменился"

        aux_src_old = build_aux_source_code(fn_old_no_doc, doc_old)
        n_src_lines = len(code_old.splitlines()) + len(doc_old.splitlines())

        # Единые фильтры (R1)
        if fn_old.name.startswith("_"):
            return None, "не_прошёл_фильтры"
        if len(doc_old) < 40:
            return None, "не_прошёл_фильтры"
        if not (4 <= n_src_lines <= 300):
            return None, "не_прошёл_фильтры"

        sig_old_obj = extract_signature(fn_old)
        sig_new_obj = extract_signature(fn_new)
        computed_drift = derive_drift_type(sig_old_obj, sig_new_obj)

        sig_type = computed_drift
        drift_type = "unknown" if computed_drift == "none" else computed_drift

        sig_old = extract_sig_line(code_old)
        diff_code = compute_code_diff(code_old, code_new)

        leak_cfg = self.config.get("leak_fields", {})
        doc_new = resolve_json_path(raw_record, leak_cfg.get("doc_new", "")) or ""
        diff_docstring = resolve_json_path(raw_record, leak_cfg.get("diff_docstring", "")) or ""
        commit_message = resolve_json_path(raw_record, leak_cfg.get("commit_message", "")) or ""

        pair_row = {
            "pair_id": make_pair_id("codocbench", project, symbol, commit_sha, code_old),
            "source": "codocbench",
            "project": str(project),
            "commit_sha": str(commit_sha),
            "symbol": str(symbol),
            "function_hash": make_function_hash(code_old),
            "sig_old": sig_old,
            "code_old": code_old,
            "code_new": code_new,
            "diff_code": diff_code,
            "doc_old": doc_old,
            "label_stale": 1,
            "drift_type": drift_type,
            "sig_type": sig_type,
            "label_source": "real_coupled",
            "weight": 1.0,
            "n_src_lines": n_src_lines,
            "split_pool": split_pool,
            "_leak_doc_new": str(doc_new),
            "_leak_diff_docstring": str(diff_docstring),
            "_leak_commit_message": str(commit_message),
            "_aux_src_old": aux_src_old,
        }
        return pair_row, "успех"

    def process_file(
        self,
        jsonl_path: str | Path,
        split_pool: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        rows: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {
            "всего": 0,
            "успех": 0,
            "нет_обязательных_полей": 0,
            "не_разобралось": 0,
            "код_не_изменился": 0,
            "не_прошёл_фильтры": 0,
        }
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats["всего"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["не_разобралось"] += 1
                    continue

                row, reason = self.process_record(record, split_pool)
                if row is not None:
                    rows.append(row)
                    stats["успех"] += 1
                else:
                    stats[reason] += 1

        return rows, stats