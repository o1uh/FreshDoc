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
import os
from pathlib import Path
import subprocess
import tempfile
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
)
from src.risk.config import RiskConfig, load_risk_config
from src.risk.fan_in import analyze_repo_fan_in

# Evidence sizes for the registry (FD-30 section card needs the cause).
# Previous limits (100/200 chars) truncated the diff cause; R12 does not cap it.
EVIDENCE_DIFF_LIMIT = 2000
EVIDENCE_DOC_LIMIT = 500


# ---------------------------------------------------------------------------
# Caching subsystem
# ---------------------------------------------------------------------------


# Cache payload version. Bump when the key schema changes: old entries are
# never hit (harmless orphans) instead of silently returning wrong scores.
# v1: sha1(diff + "::" + doc). v2: sha1(sig + "::" + diff + "::" + doc).
CACHE_KEY_VERSION = "v2"


def make_cache_key(sig_old: str, diff_code: str, doc_old: str) -> str:
    """Generate SHA1 key for replay drift cache per R12 step 5.

    Full cache key is ``{scorer_name}:{sha1}`` (see ReplayCache):
    the sha1 covers ``sig_old + "::" + diff_code + "::" + doc_old`` because
    the M2 input is sig + diff + doc — same diff+doc under different
    signatures must not collide. The scorer prefix isolates M0 vs M2
    predictions for the same input.
    """
    raw = f"{CACHE_KEY_VERSION}|{sig_old}::{diff_code}::{doc_old}".encode("utf-8")
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
            # Never silently drop hours of M2 inference: quarantine the
            # corrupted file for inspection instead of resetting memory.
            try:
                backup = self.cache_path.with_suffix(
                    self.cache_path.suffix + ".corrupted.bak"
                )
                os.replace(self.cache_path, backup)
            except OSError:
                pass
            self._data = {}

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a killed process (Ctrl+C, OOM, timeout) must never
        # leave a truncated JSON behind. Write to a temp file in the same
        # directory, then atomically replace the target.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.cache_path.parent),
            prefix=self.cache_path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self.cache_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def get(
        self,
        scorer_name: str,
        diff_code: str,
        doc_old: str,
        sig_old: str = "",
    ) -> dict[str, Any] | None:
        key = f"{scorer_name}:{make_cache_key(sig_old, diff_code, doc_old)}"
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
        sig_old: str = "",
    ) -> None:
        key = f"{scorer_name}:{make_cache_key(sig_old, diff_code, doc_old)}"
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


def gate_drift(
    drift_classes: list[str] | tuple[str, ...],
    drift_id: int,
    cal_p: float,
    threshold: float,
) -> str:
    """Threshold gate for the M2 drift head (R10: no `none` class in the head).

    Pure helper (no torch) so the gating logic is unit-testable: the argmax
    class is reported only when the calibrated stale probability exceeds the
    working threshold, otherwise `none`.
    """
    if cal_p > threshold and 0 <= drift_id < len(drift_classes):
        return drift_classes[drift_id]
    return "none"


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
            drift_name = gate_drift(self.drift_classes, drift_id, cal_p, self.threshold)

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
    first_seen_dt: datetime | None = None
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
    # S2 breakdown: same-commit code+doc fixes vs delayed doc-only fixes.
    # Headline R12 counters above are unchanged; this split only explains them.
    atomic_fixes: int = 0
    delayed_fixes: int = 0
    # DLC survival: observed flag->fix lead times and right-censored ages of
    # still-open flags at the window end (pooled across projects for KM).
    fix_times: list[float] = field(default_factory=list)
    censored_times: list[float] = field(default_factory=list)
    timeline_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    section_histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _git_cmd(repo_path: Path | str, *args: str) -> str:
    cmd = ["git", "-c", "core.quotepath=off", "-C", str(repo_path), *args]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout.decode("utf-8", errors="replace")


def get_first_parent_commits(repo_path: Path | str, max_commits: int = 300) -> list[tuple[str, str]]:
    """Return list of (sha, iso_date) following the first-parent lineage (newest first).

    An empty repo (no commits yet) makes `git log` exit non-zero: return []
    and let the caller raise a clear ValueError instead of leaking git errors.
    """
    try:
        raw = _git_cmd(
            repo_path,
            "log",
            "--first-parent",
            "-n",
            str(max_commits),
            "--format=%H|%cI",
            "HEAD",
        )
    except subprocess.CalledProcessError:
        return []
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


def _load_fan_in_map(repo_path: Path | str) -> dict[str, int]:
    """Approximate call counts by name across the repo checkout (R12, FD-28).

    Same-named symbols across modules are merged (see src.risk.fan_in).
    Computed once on the current checkout (HEAD) as an approximation for the
    whole replay window; per-commit checkouts would be prohibitively slow.
    Returns {} when the repo cannot be scanned (e.g. mocked paths in tests).
    """
    try:
        return analyze_repo_fan_in(repo_path)
    except Exception:
        return {}


def _nested_qualnames(tree: ast.AST) -> set[str]:
    """Qualnames of functions nested inside other functions (closures).

    The registry tracks module-level API docs (module functions and class
    methods); local helpers defined inside a function body only add noise
    and get a distorted fan_in (their leaf name matches unrelated calls).
    Class nesting is fine — only function ancestors count. Contained here
    (not in miner.iter_functions) so the FD-05 mining pipeline is untouched.
    """
    nested: set[str] = set()

    def visit(node: ast.AST, stack: list[str], in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, stack + [child.name], in_func)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(stack + [child.name])
                if in_func:
                    nested.add(qualname)
                visit(child, stack + [child.name], True)

    visit(tree, [], False)
    return nested


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

    # R12 impact approximation (FD-28): call counts by name on the HEAD
    # checkout; same names across modules are merged. One scan per replay.
    fan_in_map = _load_fan_in_map(repo_path)

    # Chronological traversal: oldest to newest (R12 step 1)
    commits = list(reversed(commits_desc))
    last_commit_dt = datetime.fromisoformat(commits[-1][1])

    states: dict[str, SectionTrackingState] = {}
    records: list[dict[str, Any]] = []
    all_lead_times: list[float] = []
    total_fixes = 0
    preceded_fixes = 0
    atomic_fixes = 0
    delayed_fixes = 0

    tracked_samples = set(sample_symbols) if sample_symbols else set()
    sample_timelines: dict[str, list[dict[str, Any]]] = {sym: [] for sym in tracked_samples}

    for sha, dt_str in commits:
        commit_dt = datetime.fromisoformat(dt_str)
        py_files = get_modified_py_files(repo_path, sha)
        if not py_files:
            continue

        for file_path in py_files:
            # show_blob enforces R3 guards: root commits (no parent) and
            # files > 300 KB return None and are skipped below.
            blob_old = show_blob(repo_path, sha, file_path, before=True)
            blob_new = show_blob(repo_path, sha, file_path, before=False)
            if not blob_old or not blob_new:
                continue

            try:
                tree_old = ast.parse(blob_old)
                tree_new = ast.parse(blob_new)
            except (SyntaxError, ValueError, RecursionError):
                # ValueError: null bytes in fake-.py binaries; RecursionError:
                # pathological nesting. A scanner must skip the file, not die.
                continue

            fns_old = iter_functions(tree_old)
            fns_new = iter_functions(tree_new)
            nested = _nested_qualnames(tree_old) | _nested_qualnames(tree_new)
            common = sorted(set(fns_old.keys()) & set(fns_new.keys()))

            for qualname in common:
                # Closures (helpers inside function bodies) are not API docs.
                if qualname in nested:
                    continue
                short_name, fn_old = fns_old[qualname]
                _, fn_new = fns_new[qualname]

                # R1 invariant: every part of the qualified name must be
                # public — this excludes `_private` functions, methods of
                # private classes (`_Worker.run`), and dunders such as
                # `__init__` (workflow.md R1 explicitly includes __init__).
                if any(part.startswith("_") for part in qualname.split(".")):
                    continue

                doc_old = (ast.get_docstring(fn_old, clean=True) or "").strip()
                doc_new = (ast.get_docstring(fn_new, clean=True) or "").strip()

                # R1 invariant: docstring >= 40 chars
                if max(len(doc_old), len(doc_new)) < 40:
                    continue

                # R1 invariant: 4..300 source lines. Formula mirrors FD-05
                # mining (miner.compare_pair): normalized code lines (comments
                # and blank lines stripped by ast.unparse) + docstring lines.
                # Approximation documented: raw file lines may differ slightly.
                _, old_clean = extract_docstring_and_clean_node(fn_old)
                code_old = normalize_code_from_node(old_clean)
                n_src_lines = len(code_old.splitlines()) + len(doc_old.splitlines())
                if not (4 <= n_src_lines <= 300):
                    continue

                _, new_clean = extract_docstring_and_clean_node(fn_new)
                code_new = normalize_code_from_node(new_clean)

                section_id = f"{file_path}::{qualname}#docstring"
                if section_id not in states:
                    # Impact is known from the start (HEAD fan_in map), so
                    # doc_updated records carry a real impact even when the
                    # function never changed code inside the window.
                    fan_in_init = float(fan_in_map.get(short_name, 0))
                    states[section_id] = SectionTrackingState(
                        section_id=section_id,
                        symbol=qualname,
                        doc_path=file_path,
                        impact=compute_impact(
                            fan_in=fan_in_init,
                            is_public=True,
                            fan_in_cap=cfg.fan_in_cap,
                            private_factor=cfg.private_factor,
                        ),
                        first_seen_dt=commit_dt,
                    )
                st = states[section_id]
                if st.first_seen_dt is None:
                    st.first_seen_dt = commit_dt

                # Days since last confirmation; falls back to the section age
                # (first sighting in the window) when nothing was ever
                # verified — otherwise long-abandoned docs never gain urgency.
                if st.verified_dt is not None:
                    age_days = (commit_dt - st.verified_dt).total_seconds() / 86400.0
                elif st.first_seen_dt is not None:
                    age_days = (commit_dt - st.first_seen_dt).total_seconds() / 86400.0
                else:
                    age_days = 0.0
                age_days = max(0.0, age_days)

                # Event Case A: docstring updated (R12 step 3)
                if doc_old != doc_new:
                    if not doc_new.strip() or len(doc_new.strip()) < 40:
                        # Docs deleted or degraded below the R1 validity floor:
                        # not a fix. A stub ("TODO: fix") must not reset risk
                        # or count as drift removal — it is the maximally
                        # stale state, same as deletion.
                        event = "doc_removed" if not doc_new.strip() else "doc_degraded"
                        st.current_risk = accumulate_next_risk(st.current_risk, 1.0)
                        st.urgency = compute_urgency(
                            risk=st.current_risk,
                            doc_type="docstring",
                            days_since_verified=age_days,
                            config=cfg,
                        )
                        if st.current_risk >= threshold and st.first_flagged_dt is None:
                            st.first_flagged_dt = commit_dt
                            st.first_flagged_commit = sha
                        rec = {
                            "repo": repo_name,
                            "branch": branch,
                            "commit_sha": sha,
                            "doc_path": file_path,
                            "section_id": section_id,
                            "symbol": qualname,
                            "p_stale": 1.0,
                            "drift_type": None,
                            "risk": round(st.current_risk, 4),
                            "urgency": round(st.urgency, 4),
                            "impact": round(st.impact, 4),
                            "evidence": {"doc_old": doc_old[:EVIDENCE_DOC_LIMIT]},
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
                    else:
                        total_fixes += 1
                        # S2: same-commit code+doc sync vs delayed doc-only fix.
                        if code_old != code_new:
                            atomic_fixes += 1
                        else:
                            delayed_fixes += 1
                        was_flagged = st.first_flagged_dt is not None
                        if was_flagged:
                            preceded_fixes += 1
                            # Git dates are not monotonic (rebases, bad clocks):
                            # never report a negative lead time.
                            lead_days = max(
                                0.0,
                                (commit_dt - st.first_flagged_dt).total_seconds() / 86400.0,  # type: ignore[operator]
                            )
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
                            "evidence": {"doc_old": doc_old[:EVIDENCE_DOC_LIMIT], "doc_new": doc_new[:EVIDENCE_DOC_LIMIT]},
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
                    cached = cache.get(scorer_name, diff_code, doc_old, sig_old_line) if cache else None
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
                            cache.set(scorer_name, diff_code, doc_old, p_stale, drift_type, sig_old_line)

                    st.current_risk = accumulate_next_risk(st.current_risk, p_stale)
                    # R12 impact: approximate call count by short name on the
                    # HEAD checkout (merged across modules); 0 when never called.
                    fan_in = float(fan_in_map.get(short_name, 0))
                    st.impact = compute_impact(
                        fan_in=fan_in,
                        is_public=True,
                        fan_in_cap=cfg.fan_in_cap,
                        private_factor=cfg.private_factor,
                    )
                    st.urgency = compute_urgency(
                        risk=st.current_risk,
                        doc_type="docstring",
                        days_since_verified=age_days,
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
                        "evidence": {"diff": diff_code[:EVIDENCE_DIFF_LIMIT]},
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

    # Noise analysis (R12 step 6): open flags older than 90 days.
    # All open-flag ages are also kept for DLC survival (right-censored).
    open_flags = 0
    open_flags_90d = 0
    censored_times: list[float] = []
    for st in states.values():
        if st.first_flagged_dt is not None and st.current_risk >= threshold:
            open_flags += 1
            age_days = (last_commit_dt - st.first_flagged_dt).total_seconds() / 86400.0
            censored_times.append(max(0.0, age_days))
            if age_days > 90.0:
                open_flags_90d += 1

    preceded_frac = (preceded_fixes / total_fixes) if total_fixes > 0 else 0.0
    med_lead = float(np.median(all_lead_times)) if all_lead_times else 0.0
    mean_lead = float(np.mean(all_lead_times)) if all_lead_times else 0.0
    max_lead = float(np.max(all_lead_times)) if all_lead_times else 0.0
    noise_frac = (open_flags_90d / open_flags) if open_flags > 0 else 0.0

    # Persist into registry SQLite if db_path is provided.
    # Single atomic transaction: a failed insert must never leave the
    # branch wiped (DELETE and INSERT share one connection/transaction).
    if db_path:
        storage.init_db(db_path)
        storage.replace_branch_records(db_path, repo_name, branch, records)

    # Keyed by section_id (file::qualname#docstring): short qualnames such
    # as `setup` or `parse` repeat across modules and would overwrite each
    # other when keyed by symbol.
    section_histories = {st.section_id: st.history for st in states.values() if st.history}

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
        atomic_fixes=atomic_fixes,
        delayed_fixes=delayed_fixes,
        fix_times=list(all_lead_times),
        censored_times=censored_times,
        timeline_samples=sample_timelines,
        section_histories=section_histories,
    )

    return metrics, records
