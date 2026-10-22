"""FD-29: History replay and lead time metrics computation (R12).

Iterates over first-parent git commits of a target project, tracks
function code/docstring evolution, scores code drifts using M0 or M2,
accumulates risk, records docstring updates, calculates lead time
before updates, and honest noise estimation (open flags > 90 days).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Protocol

import numpy as np

from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import (
    compute_code_diff,
    extract_docstring_and_clean_node,
    extract_sig_line,
    normalize_code_from_node,
)
from src.mining.miner import iter_functions, show_blob
from src.models.calibration import apply_temperature
from src.registry import storage
from src.risk.calculator import (
    accumulate_next_risk,
    compute_impact,
    compute_urgency,
    is_public_symbol,
)
from src.risk.config import RiskConfig, load_risk_config


# ---------------------------------------------------------------------------
# Caching subsystem
# ---------------------------------------------------------------------------


def make_cache_key(diff_code: str, doc_old: str) -> str:
    """Generate SHA1 key for replay drift cache per R12 step 5."""
    raw = f"{diff_code}::{doc_old}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class ReplayCache:
    """JSON-backed persistent cache for replay predictions."""

    def __init__(self, cache_path: Path | str | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self._data: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.cache_path and self.cache_path.exists():
            self.load()

    def load(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            self._data = {}

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get(self, scorer_name: str, diff_code: str, doc_old: str) -> dict[str, Any] | None:
        key = f"{scorer_name}:{make_cache_key(diff_code, doc_old)}"
        val = self._data.get(key)
        if val is not None:
            self.hits += 1
            return val
        self.misses += 1
        return None

    def set(
        self,
        scorer_name: str,
        diff_code: str,
        doc_old: str,
        p_stale: float,
        drift_type: str,
    ) -> None:
        key = f"{scorer_name}:{make_cache_key(diff_code, doc_old)}"
        self._data[key] = {
            "p_stale": float(p_stale),
            "drift_type": str(drift_type),
        }


# ---------------------------------------------------------------------------
# Scorer implementations (M0 & M2)
# ---------------------------------------------------------------------------


class DriftScorer(Protocol):
    name: str

    def score(
        self,
        sig_old: str,
        diff_code: str,
        doc_old: str,
        fn_old: ast.AST,
        fn_new: ast.AST,
    ) -> tuple[float, str]: ...


class M0Scorer:
    """M0 Rule baseline (R10): p_stale = 1.0 if signature changed, else 0.0."""

    name = "m0"

    def score(
        self,
        sig_old: str,
        diff_code: str,
        doc_old: str,
        fn_old: ast.AST,
        fn_new: ast.AST,
    ) -> tuple[float, str]:
        sig_old_obj = extract_signature(fn_old)  # type: ignore[arg-type]
        sig_new_obj = extract_signature(fn_new)  # type: ignore[arg-type]
        drift_type = derive_drift_type(sig_old_obj, sig_new_obj)
        p_stale = 1.0 if drift_type != "none" else 0.0
        return p_stale, drift_type


class M2Scorer:
    """M2 Neural dual-head encoder with temperature calibration (FD-25)."""

    name = "m2"

    def __init__(
        self,
        checkpoint_dir: Path | str,
        temperature: float = 2.384710179084571,
        threshold: float = 0.8742609896291749,
        device: str | None = None,
        max_len: int = 1024,
        doc_budget: int = 384,
    ) -> None:
        # Lazy imports of heavy deep learning packages (FD-29 Defect 3)
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from safetensors.torch import load_file
        from src.models.m2_data import DOC_BUDGET_TOKENS, DRIFT_CLASSES
        from src.models.m2_model import DualHeadCodeClassifier

        self.ckpt_path = Path(checkpoint_dir)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.max_len = max_len
        self.doc_budget = doc_budget or DOC_BUDGET_TOKENS
        self.drift_classes = DRIFT_CLASSES

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.ckpt_path))
        cfg = AutoConfig.from_pretrained(str(self.ckpt_path))
        backbone = AutoModel.from_config(cfg)
        self.model = DualHeadCodeClassifier(
            backbone,
            hidden_size=backbone.config.hidden_size,
            num_drift_classes=len(self.drift_classes),
        )
        state = load_file(self.ckpt_path / "model.safetensors")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def score(
        self,
        sig_old: str,
        diff_code: str,
        doc_old: str,
        fn_old: ast.AST,
        fn_new: ast.AST,
    ) -> tuple[float, str]:
        import torch
        from src.models.m2_data import encode_pair

        enc = encode_pair(
            self.tokenizer,
            sig_old=sig_old,
            diff_code=diff_code,
            doc_old=doc_old,
            max_len=self.max_len,
            doc_budget=self.doc_budget,
        )
        inp = torch.tensor([enc["input_ids"]], dtype=torch.long, device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(inp, attention_mask=mask)
            z_stale = out["logits_stale"]
            z_drift = out["logits_drift"]
            raw_p = torch.sigmoid(z_stale).squeeze().item()
            cal_p = float(apply_temperature(np.array([raw_p]), T=self.temperature)[0])
            drift_id = int(torch.argmax(z_drift, dim=-1).item())
            drift_name = self.drift_classes[drift_id] if cal_p > self.threshold else "none"

        return cal_p, drift_name


# ---------------------------------------------------------------------------
# Replay state tracking & execution
# ---------------------------------------------------------------------------


@dataclass
class SectionTrackingState:
    section_id: str
    symbol: str
    doc_path: str
    current_risk: float = 0.0
    urgency: float = 0.0
    impact: float = 0.0
    verified_commit: str | None = None
    verified_dt: datetime | None = None
    first_flagged_dt: datetime | None = None
    first_flagged_commit: str | None = None
    lead_times_days: list[float] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReplayMetrics:
    repo: str
    scorer: str
    total_commits: int
    total_fixes: int
    preceded_fixes: int
    preceded_fraction: float
    median_lead_time_days: float
    mean_lead_time_days: float
    max_lead_time_days: float
    open_flags: int
    open_flags_older_90d: int
    noise_fraction: float
    timeline_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    section_histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _git_cmd(repo_path: Path | str, *args: str) -> str:
    cmd = ["git", "-c", "core.quotepath=off", "-C", str(repo_path), *args]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout.decode("utf-8", errors="replace")


def get_first_parent_commits(repo_path: Path | str, max_commits: int = 300) -> list[tuple[str, str]]:
    """Return list of (sha, iso_date) following the first-parent lineage (newest first)."""
    raw = _git_cmd(
        repo_path,
        "log",
        "--first-parent",
        "-n",
        str(max_commits),
        "--format=%H|%cI",
        "HEAD",
    )
    commits = []
    for line in raw.splitlines():
        if "|" in line:
            sha, dt = line.split("|", 1)
            commits.append((sha.strip(), dt.strip()))
    return commits


def get_modified_py_files(repo_path: Path | str, sha: str) -> list[str] | None:
    """Return modified .py files between first parent and sha, or None if skipped (>50)."""
    try:
        raw = _git_cmd(
            repo_path,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            f"{sha}^1",
            sha,
        )
    except subprocess.CalledProcessError:
        return None

    paths = []
    for line in raw.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0] == "M" and parts[1].endswith(".py"):
            paths.append(parts[1])
    if len(paths) > 50:
        return None
    return paths


def replay_project_history(
    repo_path: Path | str,
    repo_name: str,
    branch: str = "main",
    max_commits: int = 300,
    scorer: DriftScorer | None = None,
    cache: ReplayCache | None = None,
    db_path: Path | str | None = None,
    risk_config: RiskConfig | None = None,
    sample_symbols: Iterable[str] | None = None,
) -> tuple[ReplayMetrics, list[dict[str, Any]]]:
    """Execute replay of git history according to R12."""
    cfg = risk_config or load_risk_config()
    threshold = float(cfg.threshold)
    active_scorer = scorer or M0Scorer()
    scorer_name = active_scorer.name

    commits_desc = get_first_parent_commits(repo_path, max_commits)
    if not commits_desc:
        raise ValueError(f"No commits retrieved for {repo_path}")

    # Chronological traversal: oldest to newest (R12 step 1)
    commits = list(reversed(commits_desc))
    last_commit_dt = datetime.fromisoformat(commits[-1][1])

    states: dict[str, SectionTrackingState] = {}
    records: list[dict[str, Any]] = []
    all_lead_times: list[float] = []
    total_fixes = 0
    preceded_fixes = 0

    tracked_samples = set(sample_symbols) if sample_symbols else set()
    sample_timelines: dict[str, list[dict[str, Any]]] = {sym: [] for sym in tracked_samples}

    for sha, dt_str in commits:
        commit_dt = datetime.fromisoformat(dt_str)
        py_files = get_modified_py_files(repo_path, sha)
        if not py_files:
            continue

        for file_path in py_files:
            blob_old = show_blob(repo_path, sha, file_path, before=True)
            blob_new = show_blob(repo_path, sha, file_path, before=False)
            if not blob_old or not blob_new:
                continue

            try:
                tree_old = ast.parse(blob_old)
                tree_new = ast.parse(blob_new)
            except SyntaxError:
                continue

            fns_old = iter_functions(tree_old)
            fns_new = iter_functions(tree_new)
            common = sorted(set(fns_old.keys()) & set(fns_new.keys()))

            for qualname in common:
                short_name, fn_old = fns_old[qualname]
                _, fn_new = fns_new[qualname]

                # Invariant 3: function name without '_' (public protocol)
                if not is_public_symbol(short_name) or short_name.startswith("_"):
                    continue

                doc_old = (ast.get_docstring(fn_old, clean=True) or "").strip()
                doc_new = (ast.get_docstring(fn_new, clean=True) or "").strip()

                # Invariant 3: docstring >= 40 chars
                if max(len(doc_old), len(doc_new)) < 40:
                    continue

                # Invariant 3: 4..300 lines of code
                _, old_clean = extract_docstring_and_clean_node(fn_old)
                code_old = normalize_code_from_node(old_clean)
                n_src_lines = len(code_old.splitlines()) + len(doc_old.splitlines())
                if not (4 <= n_src_lines <= 300):
                    continue

                _, new_clean = extract_docstring_and_clean_node(fn_new)
                code_new = normalize_code_from_node(new_clean)

                section_id = f"{file_path}::{qualname}#docstring"
                if section_id not in states:
                    states[section_id] = SectionTrackingState(
                        section_id=section_id,
                        symbol=qualname,
                        doc_path=file_path,
                    )
                st = states[section_id]

                # Event Case A: docstring updated (R12 step 3)
                if doc_old != doc_new:
                    total_fixes += 1
                    was_flagged = st.first_flagged_dt is not None
                    if was_flagged:
                        preceded_fixes += 1
                        lead_days = (commit_dt - st.first_flagged_dt).total_seconds() / 86400.0  # type: ignore[operator]
                        st.lead_times_days.append(lead_days)
                        all_lead_times.append(lead_days)

                    st.current_risk = 0.0
                    st.urgency = 0.0
                    st.verified_commit = sha
                    st.verified_dt = commit_dt
                    st.first_flagged_dt = None
                    st.first_flagged_commit = None
                    event = "doc_updated"

                    rec = {
                        "repo": repo_name,
                        "branch": branch,
                        "commit_sha": sha,
                        "doc_path": file_path,
                        "section_id": section_id,
                        "symbol": qualname,
                        "p_stale": 0.0,
                        "drift_type": None,
                        "risk": 0.0,
                        "urgency": 0.0,
                        "impact": st.impact,
                        "evidence": {"doc_old": doc_old[:100], "doc_new": doc_new[:100]},
                        "verified_commit": sha,
                        "event": event,
                        "computed_at": dt_str,
                    }
                    records.append(rec)
                    st.history.append({
                        "commit_sha": sha,
                        "date": dt_str,
                        "risk": 0.0,
                        "event": event,
                    })

                    if qualname in tracked_samples or short_name in tracked_samples:
                        sample_timelines.setdefault(qualname, []).append({
                            "commit_sha": sha,
                            "date": dt_str,
                            "risk": 0.0,
                            "event": event,
                        })

                # Event Case B: code changed, docstring untouched (R12 step 4)
                elif code_old != code_new:
                    diff_code = compute_code_diff(code_old, code_new)
                    sig_old_line = extract_sig_line(code_old)

                    # Check cache first (R12 step 5)
                    cached = cache.get(scorer_name, diff_code, doc_old) if cache else None
                    if cached is not None:
                        p_stale = float(cached["p_stale"])
                        drift_type = str(cached["drift_type"])
                    else:
                        p_stale, drift_type = active_scorer.score(
                            sig_old=sig_old_line,
                            diff_code=diff_code,
                            doc_old=doc_old,
                            fn_old=fn_old,
                            fn_new=fn_new,
                        )
                        if cache:
                            cache.set(scorer_name, diff_code, doc_old, p_stale, drift_type)

                    st.current_risk = accumulate_next_risk(st.current_risk, p_stale)
                    st.impact = compute_impact(
                        fan_in=1.0,
                        is_public=True,
                        fan_in_cap=cfg.fan_in_cap,
                        private_factor=cfg.private_factor,
                    )
                    days_verified = (
                        (commit_dt - st.verified_dt).total_seconds() / 86400.0
                        if st.verified_dt
                        else 0.0
                    )
                    st.urgency = compute_urgency(
                        risk=st.current_risk,
                        doc_type="docstring",
                        days_since_verified=days_verified,
                        config=cfg,
                    )

                    if st.current_risk >= threshold:
                        if st.first_flagged_dt is None:
                            st.first_flagged_dt = commit_dt
                            st.first_flagged_commit = sha
                            event = "risk_flagged"
                        else:
                            event = "risk_accumulated"
                    else:
                        event = "code_changed"

                    rec = {
                        "repo": repo_name,
                        "branch": branch,
                        "commit_sha": sha,
                        "doc_path": file_path,
                        "section_id": section_id,
                        "symbol": qualname,
                        "p_stale": p_stale,
                        "drift_type": drift_type if drift_type != "none" else None,
                        "risk": round(st.current_risk, 4),
                        "urgency": round(st.urgency, 4),
                        "impact": round(st.impact, 4),
                        "evidence": {"diff": diff_code[:200]},
                        "verified_commit": st.verified_commit,
                        "event": event,
                        "computed_at": dt_str,
                    }
                    records.append(rec)
                    st.history.append({
                        "commit_sha": sha,
                        "date": dt_str,
                        "risk": round(st.current_risk, 4),
                        "event": event,
                    })

                    if qualname in tracked_samples or short_name in tracked_samples:
                        sample_timelines.setdefault(qualname, []).append({
                            "commit_sha": sha,
                            "date": dt_str,
                            "risk": round(st.current_risk, 4),
                            "event": event,
                        })

    # Noise analysis (R12 step 6): open flags older than 90 days
    open_flags = 0
    open_flags_90d = 0
    for st in states.values():
        if st.first_flagged_dt is not None and st.current_risk >= threshold:
            open_flags += 1
            age_days = (last_commit_dt - st.first_flagged_dt).total_seconds() / 86400.0
            if age_days > 90.0:
                open_flags_90d += 1

    preceded_frac = (preceded_fixes / total_fixes) if total_fixes > 0 else 0.0
    med_lead = float(np.median(all_lead_times)) if all_lead_times else 0.0
    mean_lead = float(np.mean(all_lead_times)) if all_lead_times else 0.0
    max_lead = float(np.max(all_lead_times)) if all_lead_times else 0.0
    noise_frac = (open_flags_90d / open_flags) if open_flags > 0 else 0.0

    # Persist into registry SQLite if db_path is provided
    if db_path:
        storage.init_db(db_path)
        with storage.get_connection(db_path) as conn:
            conn.cursor().execute(
                "DELETE FROM doc_state WHERE repo = ? AND branch = ?",
                (repo_name, branch),
            )
            conn.commit()
        storage.insert_batch(db_path, records)

    section_histories = {st.symbol: st.history for st in states.values() if st.history}

    metrics = ReplayMetrics(
        repo=repo_name,
        scorer=scorer_name,
        total_commits=len(commits),
        total_fixes=total_fixes,
        preceded_fixes=preceded_fixes,
        preceded_fraction=round(preceded_frac, 4),
        median_lead_time_days=round(med_lead, 2),
        mean_lead_time_days=round(mean_lead, 2),
        max_lead_time_days=round(max_lead, 2),
        open_flags=open_flags,
        open_flags_older_90d=open_flags_90d,
        noise_fraction=round(noise_frac, 4),
        timeline_samples=sample_timelines,
        section_histories=section_histories,
    )

    return metrics, records
