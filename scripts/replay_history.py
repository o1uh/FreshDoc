"""FD-29 CLI: Replay git history on 2 test projects, evaluate M0 & M2,

compute lead times and noise, generate risk timelines and report.

Usage:
    python scripts/replay_history.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.replay import (
    M0Scorer,
    M2Scorer,
    ReplayCache,
    ReplayMetrics,
    replay_project_history,
)
from src.registry.storage import init_db
from src.risk.config import RiskConfig, load_risk_config

CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "replay_predictions.json"
REGISTRY_DB_PATH = PROJECT_ROOT / "data" / "registry.db"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "replay.md"
REPOS_DIR = PROJECT_ROOT / "data" / "repos"

PROJECT_URLS = {
    "tornado": "https://github.com/tornadoweb/tornado.git",
    "ipython": "https://github.com/ipython/ipython.git",
    "pandas": "https://github.com/pandas-dev/pandas.git",
    "matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "textual": "https://github.com/Textualize/textual.git",
}

# Sensitivity marks (days) for the noise-cutoff table: answers "why 90".
CUTOFF_MARKS = (30.0, 60.0, 90.0, 180.0)


def ensure_repo(
    name: str,
    url: str,
    depth: int = 350,
    override_path: str | None = None,
    repos_dir: Path | None = None,
) -> Path:
    """Return a local clone of the project, cloning only when needed.

    `override_path` (CLI --tornado-path/--ipython-path) uses an existing
    local repository as-is — for offline/sandboxed runs without network.
    """
    if override_path:
        target = Path(override_path)
        if not (target / ".git").exists():
            raise FileNotFoundError(f"--{name}-path is not a git repo: {target}")
        print(f"Using existing {name} repo at {target}", flush=True)
        return target
    base = repos_dir or REPOS_DIR
    target = base / name
    if (target / ".git").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {name} from {url} (depth={depth})...", flush=True)
    cmd = ["git", "clone", "--depth", str(depth), url, str(target)]
    subprocess.run(cmd, check=True)
    return target


def plot_risk_timelines(
    timelines_m0: dict[str, list[dict]],
    timelines_m2: dict[str, list[dict]],
    symbols: list[str],
    threshold_m0: float,
    threshold_m2: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not symbols:
        print("Warning: no timeline symbols provided for plotting.")
        return

    n_plots = len(symbols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(11, 3.2 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    for ax, sym in zip(axes, symbols):
        # M0 points
        events_m0 = timelines_m0.get(sym, [])
        if events_m0:
            dates_m0 = [datetime.fromisoformat(e["date"]) for e in events_m0]
            risks_m0 = [e["risk"] for e in events_m0]
            ax.step(
                dates_m0,
                risks_m0,
                where="post",
                color="#e67e22",
                linestyle="--",
                label="M0 (Rule)",
                alpha=0.75,
            )
            ax.scatter(dates_m0, risks_m0, color="#e67e22", s=25, alpha=0.75)

        # M2 points
        events_m2 = timelines_m2.get(sym, [])
        if events_m2:
            dates_m2 = [datetime.fromisoformat(e["date"]) for e in events_m2]
            risks_m2 = [e["risk"] for e in events_m2]
            ax.step(
                dates_m2,
                risks_m2,
                where="post",
                color="#2980b9",
                linewidth=2,
                label="M2 (Neural Calibrated)",
            )
            ax.scatter(dates_m2, risks_m2, color="#2980b9", s=35)

            # Mark docstring fixes and removals
            for e in events_m2:
                if e["event"] == "doc_updated":
                    fix_dt = datetime.fromisoformat(e["date"])
                    ax.axvline(
                        fix_dt,
                        color="#27ae60",
                        linestyle=":",
                        alpha=0.7,
                        label="Doc Fix" if "Doc Fix" not in ax.get_legend_handles_labels()[1] else "",
                    )
                elif e["event"] in ("doc_removed", "doc_degraded"):
                    rm_dt = datetime.fromisoformat(e["date"])
                    ax.axvline(
                        rm_dt,
                        color="#c0392b",
                        linestyle=":",
                        alpha=0.7,
                        label="Doc Removed" if "Doc Removed" not in ax.get_legend_handles_labels()[1] else "",
                    )

        ax.axhline(
            threshold_m2,
            color="#c0392b",
            linestyle="-.",
            alpha=0.6,
            label=f"M2 threshold (tau={threshold_m2:.2f})",
        )
        ax.set_ylabel("Risk R")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Section: {sym} ({len(events_m2)} events)", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Risk timeline chart saved to {out_path} ({n_plots} functions)", flush=True)


def plot_survival_curves(
    pooled: dict[str, tuple[list, list]],
    out_path: Path,
) -> None:
    """Kaplan-Meier survival curves per scorer (DLC): X — days since flag,
    Y — share of alerts still open. Censored (still open) alerts only shrink
    the at-risk set; the curve drops solely at observed fixes."""
    from src.registry.survival import kaplan_meier

    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"m0": ("#e67e22", "--", "M0 (Rule)"), "m2": ("#2980b9", "-", "M2 (Neural)")}
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for key in ("m0", "m2"):
        durs, obs = pooled.get(key, ([], []))
        curve = kaplan_meier(list(durs), list(obs))
        if not curve:
            continue
        xs = [t for t, _ in curve]
        ys = [s for _, s in curve]
        color, style, label = colors[key]
        ax.step(xs, ys, where="post", color=color, linestyle=style,
                label=f"{label} (n_fix={sum(obs)}, n_open={len(obs) - sum(obs)})")
        ax.scatter(xs, ys, color=color, s=20)
    ax.set_xlabel("Days since flag")
    ax.set_ylabel("Share of alerts still open")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Warning survival (Kaplan-Meier, pooled farm)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Survival curve saved to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-29: Replay git history.")
    parser.add_argument("--max-commits", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ckpt", default="checkpoints/slot4_full/best")
    # Offline runs: point at existing local clones instead of `git clone`.
    # Generic override for any project: --repo-path name=/path/to/clone
    # (repeatable); keeps --tornado-path/--ipython-path as shortcuts.
    parser.add_argument("--tornado-path", default=None)
    parser.add_argument("--ipython-path", default=None)
    parser.add_argument("--repo-path", action="append", default=[])
    parser.add_argument("--repos-dir", default=None)
    parser.add_argument(
        "--projects",
        default=",".join(PROJECT_URLS),
        help="Comma-separated subset of PROJECT_URLS to replay (default: all).",
    )
    args = parser.parse_args()

    risk_cfg = load_risk_config()
    tau_m2 = float(risk_cfg.threshold)
    tau_m0 = 0.5

    cache = ReplayCache(CACHE_PATH)
    print(f"Loaded cache from {CACHE_PATH} (entries: {len(cache._data)})")

    init_db(REGISTRY_DB_PATH)

    # Initialize scorers
    m0_scorer = M0Scorer()
    print(f"Initializing M2Scorer from {args.ckpt} on {args.device}...")
    m2_scorer = M2Scorer(
        checkpoint_dir=PROJECT_ROOT / args.ckpt,
        threshold=tau_m2,
        device=args.device,
    )

    all_metrics: dict[str, dict[str, ReplayMetrics]] = {}
    path_overrides = {
        "tornado": args.tornado_path,
        "ipython": args.ipython_path,
    }
    for spec in args.repo_path:
        if "=" in spec:
            name, _, path = spec.partition("=")
            path_overrides[name.strip()] = path.strip()
    repos_dir = Path(args.repos_dir) if args.repos_dir else None
    wanted = [p.strip() for p in str(args.projects).split(",") if p.strip()]
    unknown = [p for p in wanted if p not in PROJECT_URLS]
    if unknown:
        raise SystemExit(f"Unknown projects: {unknown}. Known: {sorted(PROJECT_URLS)}")

    for proj_name in wanted:
        proj_url = PROJECT_URLS[proj_name]
        repo_dir = ensure_repo(
            proj_name,
            proj_url,
            depth=args.max_commits + 50,
            override_path=path_overrides.get(proj_name),
            repos_dir=repos_dir,
        )
        print(f"\n==========================================")
        print(f"Replaying {proj_name} (commits={args.max_commits})")
        print(f"==========================================")

        # 1. Run M0
        print(f"--- Running M0 Scorer (Rule, threshold={tau_m0}) ---")
        cfg_m0 = RiskConfig(threshold=tau_m0)
        metrics_m0, _ = replay_project_history(
            repo_path=repo_dir,
            repo_name=proj_name,
            branch="replay/m0",
            max_commits=args.max_commits,
            scorer=m0_scorer,
            cache=cache,
            db_path=REGISTRY_DB_PATH,
            risk_config=cfg_m0,
        )
        cache.save()
        print(
            f"M0 {proj_name}: fixes={metrics_m0.total_fixes}, "
            f"preceded={metrics_m0.preceded_fixes} ({metrics_m0.preceded_fraction * 100:.1f}%), "
            f"lead_time_med={metrics_m0.median_lead_time_days}d, "
            f"open_flags={metrics_m0.open_flags}, noise_>90d={metrics_m0.open_flags_older_90d}"
        )

        # 2. Run M2
        print(f"--- Running M2 Scorer (Neural, threshold={tau_m2:.3f}) ---")
        metrics_m2, _ = replay_project_history(
            repo_path=repo_dir,
            repo_name=proj_name,
            branch="replay/m2",
            max_commits=args.max_commits,
            scorer=m2_scorer,
            cache=cache,
            db_path=REGISTRY_DB_PATH,
            risk_config=risk_cfg,
        )
        cache.save()
        print(
            f"M2 {proj_name}: fixes={metrics_m2.total_fixes}, "
            f"preceded={metrics_m2.preceded_fixes} ({metrics_m2.preceded_fraction * 100:.1f}%), "
            f"lead_time_med={metrics_m2.median_lead_time_days}d, "
            f"open_flags={metrics_m2.open_flags}, noise_>90d={metrics_m2.open_flags_older_90d}"
        )

        all_metrics[proj_name] = {"m0": metrics_m0, "m2": metrics_m2}

    # Save cache
    cache.save()
    print(f"\nReplay cache saved to {CACHE_PATH} (hits: {cache.hits}, misses: {cache.misses})")

    # Per-project timelines (top-2 with fix preferred) + per-project figure,
    # so every farm project is represented regardless of farm size.
    # Keys are prefixed with the project to avoid collisions when two
    # projects contain the same qualified symbol name.
    from collections import Counter

    from src.registry.survival import kaplan_meier, survival_stats

    combined_histories_m0: dict[str, list[dict]] = {}
    combined_histories_m2: dict[str, list[dict]] = {}
    for proj_name, m_dict in all_metrics.items():
        for sym, h in m_dict["m0"].section_histories.items():
            combined_histories_m0[f"{proj_name}::{sym}"] = h
        for sym, h in m_dict["m2"].section_histories.items():
            combined_histories_m2[f"{proj_name}::{sym}"] = h

    figure_paths: dict[str, str] = {}
    for proj_name, m_dict in all_metrics.items():
        counts: Counter[str] = Counter()
        has_fix: dict[str, bool] = {}
        for sym, h in m_dict["m2"].section_histories.items():
            if len(h) >= 2:
                key = f"{proj_name}::{sym}"
                counts[key] += len(h)
                if any(e.get("event") == "doc_updated" for e in h):
                    has_fix[key] = True
        with_fix = sorted(
            [s for s in counts if has_fix.get(s)],
            key=lambda s: counts[s],
            reverse=True,
        )
        pick = with_fix[:2]
        if len(pick) < 2:
            rest = [s for s, _ in counts.most_common() if s not in pick]
            pick += rest[: 2 - len(pick)]
        proj_hist_m0 = {k: v for k, v in combined_histories_m0.items()
                        if k.startswith(proj_name + "::")}
        proj_hist_m2 = {k: v for k, v in combined_histories_m2.items()
                        if k.startswith(proj_name + "::")}
        out_png = FIGURES_DIR / f"replay_risk_timeline_{proj_name}.png"
        plot_risk_timelines(
            proj_hist_m0,
            proj_hist_m2,
            symbols=pick[:2],
            threshold_m0=tau_m0,
            threshold_m2=tau_m2,
            out_path=out_png,
        )
        figure_paths[proj_name] = f"figures/replay_risk_timeline_{proj_name}.png"
    print(f"\nPer-project timeline figures: {figure_paths}")

    # DLC survival: pool flag->fix times (observed) and open-flag ages
    # (censored) across farm projects, per scorer; KM curve figure.
    pooled: dict[str, tuple[list, list]] = {}
    for key in ("m0", "m2"):
        durs: list[float] = []
        obs: list[bool] = []
        for m_dict in all_metrics.values():
            m = m_dict[key]
            durs += list(m.fix_times) + list(m.censored_times)
            obs += [True] * len(m.fix_times) + [False] * len(m.censored_times)
        pooled[key] = (durs, obs)
    survival_png = FIGURES_DIR / "replay_survival.png"
    plot_survival_curves(pooled, out_path=survival_png)

    # Generate markdown report
    generate_report(all_metrics, tau_m0, tau_m2, cache, figure_paths)


def generate_report(
    metrics: dict[str, dict[str, ReplayMetrics]],
    tau_m0: float,
    tau_m2: float,
    cache: ReplayCache,
    figure_paths: dict[str, str] | None = None,
) -> None:
    from src.registry.survival import survival_stats

    figure_paths = figure_paths or {}
    lines = [
        "# FD-29: Проигрывание истории (Replay) и метрики опережения",
        "",
        "> Эксперимент выполнен в соответствии со справочником R12 и регламентом workflow.md.",
        f"> **Дата генерации:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"> **Пороги срабатывания:** M0: $\\tau = {tau_m0}$, M2 (FD-25 cal): $\\tau = {tau_m2:.4f}$",
        f"> **Проекты фермы ({len(metrics)}):** {', '.join(f'`{p}`' for p in metrics)}",
        "> **База реестра:** `data/registry.db`",
        "> **Кэш предсказаний:** `data/cache/replay_predictions.json`",
        "",
        "## 1. Сводная таблица метрик опережения",
        "",
        "| Проект | Модель | Коммиты | Исправлений docstring | Из них с опережением | Доля с опережением (%) | Медиана опережения (дни) | Среднее (дни) | Максимум (дни) | Открытых пометок | Пометок >90 дн (Шум) | Доля шума (%) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for proj, m_dict in metrics.items():
        for model_key in ["m0", "m2"]:
            m = m_dict[model_key]
            lines.append(
                f"| **{proj}** | {m.scorer.upper()} | {m.total_commits} | {m.total_fixes} | "
                f"{m.preceded_fixes} | **{m.preceded_fraction * 100:.1f}%** | "
                f"**{m.median_lead_time_days:.1f}** | {m.mean_lead_time_days:.1f} | "
                f"{m.max_lead_time_days:.1f} | {m.open_flags} | "
                f"**{m.open_flags_older_90d}** | {m.noise_fraction * 100:.1f}% |"
            )

    lines.extend([
        "",
        "## 2. Динамика риска по проектам",
        "",
        "На графиках показана траектория накопления риска при изменениях кода, момент превышения порога и сброс риска в ноль при подтверждении исправления docstring (по 2 самые насыщенные функции с doc-фиксом на проект):",
        "",
    ])
    for proj in metrics:
        fig = figure_paths.get(proj, "figures/replay_risk_timeline.png")
        lines.extend([f"### {proj}", "", f"![Динамика риска ({proj})]({fig})", ""])
    lines.extend([
        "## 3. Анализ шума и опережения",
        "",
        "1. **Эмпирические результаты по опережению:**",
    ])

    # Per-project lead summary, generated from the metrics above (not hardcoded).
    total_pre_m0 = sum(m_dict["m0"].preceded_fixes for m_dict in metrics.values())
    total_pre_m2 = sum(m_dict["m2"].preceded_fixes for m_dict in metrics.values())
    for proj, m_dict in metrics.items():
        m0 = m_dict["m0"]
        m2 = m_dict["m2"]
        lines.append(
            f"   - `{proj}` ({m0.total_commits} коммитов первого родителя): "
            f"M0 — {m0.preceded_fixes} опережений из {m0.total_fixes} исправлений "
            f"({m0.preceded_fraction * 100:.1f}%), медиана {m0.median_lead_time_days:.1f} дн.; "
            f"M2 — {m2.preceded_fixes} из {m2.total_fixes} "
            f"({m2.preceded_fraction * 100:.1f}%), медиана {m2.median_lead_time_days:.1f} дн."
        )
        lines.append(
            f"     * Разложение исправлений M2: atomic_sync (код+doc в одном коммите) — "
            f"{m2.atomic_fixes}, delayed_fix (doc-only после флага) — {m2.delayed_fixes}."
        )
    if total_pre_m2 == total_pre_m0:
        lines.append(
            f"   - Модель M2 не выявила дополнительных опережающих исправлений по сравнению "
            f"с бейзлайном M0 (всего {total_pre_m0}): зафиксированные случаи сопровождались "
            f"изменением сигнатуры, где правило M0 уже срабатывает."
        )
    elif total_pre_m2 > total_pre_m0:
        lines.append(
            f"   - Модель M2 выявила больше опережающих исправлений, чем M0 "
            f"({total_pre_m2} против {total_pre_m0})."
        )
    else:
        lines.append(
            f"   - Модель M2 выявила меньше опережающих исправлений, чем M0 "
            f"({total_pre_m2} против {total_pre_m0}); см. фильтрацию шума ниже."
        )

    lines.append(
        f"   - Практическое преимущество M2 — **фильтрация ложных тревог** благодаря "
        f"консервативному калиброванному порогу $\\tau = {tau_m2:.4f}$ (FD-25, против $\\tau = {tau_m0}$ у M0):"
    )
    for proj, m_dict in metrics.items():
        m0 = m_dict["m0"]
        m2 = m_dict["m2"]
        removed = m0.open_flags_older_90d - m2.open_flags_older_90d
        removed_pct = (removed / m0.open_flags_older_90d * 100) if m0.open_flags_older_90d else 0.0
        lines.append(
            f"     * В проекте `{proj}` число висячих пометок старше 90 дней: "
            f"M0 — {m0.open_flags_older_90d} ({m0.noise_fraction * 100:.1f}% шума), "
            f"M2 — {m2.open_flags_older_90d} ({m2.noise_fraction * 100:.1f}% шума) — "
            f"устранено {removed} ({removed_pct:.0f}% ложных тревог)."
        )
    # Lead-time comparison on projects where both models flagged at least once.
    for proj, m_dict in metrics.items():
        m0 = m_dict["m0"]
        m2 = m_dict["m2"]
        if m0.preceded_fixes > 0 and m2.preceded_fixes > 0:
            if m2.median_lead_time_days < m0.median_lead_time_days:
                lines.append(
                    f"     * Медиана опережения на `{proj}`: M2 — {m2.median_lead_time_days:.1f} дн. "
                    f"против {m0.median_lead_time_days:.1f} дн. у M0. Более позднее срабатывание M2 "
                    f"означает отсечение преждевременных срабатываний на поверхностных изменениях сигнатуры."
                )
            elif m2.median_lead_time_days == m0.median_lead_time_days:
                lines.append(
                    f"     * Медиана опережения на `{proj}` совпадает у M0 и M2 "
                    f"({m2.median_lead_time_days:.1f} дн.): обе модели сработали на одних и тех же "
                    f"изменениях сигнатуры."
                )
            else:
                lines.append(
                    f"     * Медиана опережения на `{proj}`: M2 — {m2.median_lead_time_days:.1f} дн. "
                    f"против {m0.median_lead_time_days:.1f} дн. у M0 (M2 сработала раньше)."
                )
    tot_m0_90 = sum(m_dict["m0"].open_flags_older_90d for m_dict in metrics.values())
    tot_m2_90 = sum(m_dict["m2"].open_flags_older_90d for m_dict in metrics.values())
    lines.append(
        f"   - Итого по ферме висячих пометок старше 90 дней: M0 — {tot_m0_90}, M2 — {tot_m2_90}."
    )

    # DLC survival (pooled farm): KM curve stats per scorer + cutoff sensitivity.
    pooled_stats: dict[str, dict] = {}
    pooled_open: dict[str, int] = {}
    pooled_cens: dict[str, list[float]] = {}
    for key in ("m0", "m2"):
        durs: list[float] = []
        obs: list[bool] = []
        for m_dict in metrics.values():
            m = m_dict[key]
            durs += list(m.fix_times) + list(m.censored_times)
            obs += [True] * len(m.fix_times) + [False] * len(m.censored_times)
        pooled_stats[key] = survival_stats(durs, obs, marks=CUTOFF_MARKS)
        pooled_open[key] = sum(m_dict[key].open_flags for m_dict in metrics.values())
        pooled_cens[key] = [c for m_dict in metrics.values() for c in m_dict[key].censored_times]

    lines.extend([
        "",
        "## 4. Выживаемость предупреждений (Kaplan-Meier, пул фермы)",
        "",
        "X — дни от пометки, Y — доля всё ещё открытых алертов. Открытые пометки — правая цензура (учитываются в at-risk, кривую роняют только состоявшиеся исправления).",
        "",
        "![Выживаемость предупреждений](figures/replay_survival.png)",
        "",
    ])
    for key in ("m0", "m2"):
        st = pooled_stats[key]
        med = f"{st['median_days']:.1f} дн." if st["median_days"] is not None else "не достигнута (цензура)"
        lines.append(
            f"   - **{key.upper()}**: событий {st['n_events']}, цензурировано {st['n_censored']}; "
            f"медиана дожития {med}."
        )
    n_ev = pooled_stats["m2"]["n_events"]
    if n_ev < 10:
        lines.append(
            f"   - Событий мало ({n_ev}), поэтому сильные заявления вида «95%» неуместны: "
            f"ниже — только эмпирика и честная оговорка."
        )
    lines.extend([
        "",
        "Доля исправленных к сроку (1 − S, по кривой):",
        "",
        "| Модель | ≤30 дн | ≤60 дн | ≤90 дн | ≤180 дн |",
        "|---|---|---|---|---|",
    ])
    for key in ("m0", "m2"):
        fb = pooled_stats[key]["fixed_by_mark"]
        lines.append(
            f"| {key.upper()} | {fb[30.0] * 100:.1f}% | {fb[60.0] * 100:.1f}% | "
            f"{fb[90.0] * 100:.1f}% | {fb[180.0] * 100:.1f}% |"
        )
    lines.extend([
        "",
        "Чувствительность доли шума к отсечке (открытые пометки старше T, пул фермы):",
        "",
        "| Модель | >30 дн | >60 дн | >90 дн | >180 дн |",
        "|---|---|---|---|---|",
    ])
    for key in ("m0", "m2"):
        cells = []
        for t in CUTOFF_MARKS:
            share = (sum(1 for c in pooled_cens[key] if c > t) / pooled_open[key]
                     if pooled_open[key] else 0.0)
            cells.append(f"{share * 100:.1f}%")
        lines.append(f"| {key.upper()} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## 5. Шум по проектам",
        "",
        "**Висячие пометки >90 дней:**",
    ])
    for proj, m_dict in metrics.items():
        m2 = m_dict["m2"]
        lines.append(
            f"   - `{proj}`: {m2.open_flags_older_90d} из {m2.open_flags} открытых пометок M2 "
            f"висят >90 дней ({m2.noise_fraction * 100:.1f}%)."
        )
    lines.extend([
        "   - В реальных open-source библиотеках устаревание второстепенных параметров и внутренних утилит может игнорироваться разработчиками месяцами или годами, пока не возникнет тикет или внешний pull request.",
        "   - Это доказывает необходимость механизма ручного подтверждения (`POST /verify`) и дашборда приоритетов (FD-30), где функции ранжируются не только по риску, но и по влиянию (`impact`) и срочности (`urgency`).",
        "",
        "**Методологические допущения (по R12):**",
        "   - **Правка docstring без связанного изменения кода** в том же коммите трактуется как исправление накопившегося ранее дрейфа (подтверждение актуальности). Удаление docstring (`doc_removed`) и деградация ниже 40 символов (`doc_degraded`) — не исправления: риск доводится до 1.0.",
        "   - История исследовалась строго по первому родителю (`--first-parent`), что отражает реальный ход интеграции веток в основную линию разработки.",
        "   - `impact` — приближение по R12/FD-28: число вызовов функции по имени на HEAD-чекауте проекта (одинаковые имена склеиваются); переименования файлов (статус R в git) не отслеживаются и теряются как исчезнувшие функции.",
        "   - Фильтры R1 (публичные имена без `_` во всех частях qualified name, docstring ≥ 40 символов, 4–300 строк) применяются к каждому событию; формула строк повторяет `miner.compare_pair` (нормализованный код + docstring).",
        "   - Возраст для `urgency` отсчитывается от последнего подтверждения, а при его отсутствии — от первого появления секции в окне (иначе заброшенные docs не набирают срочность).",
        "",
        "**Работа кэша:**",
        f"   - Обращений к кэшу за сессию: {cache.hits} попаданий, {cache.misses} промахов; всего записей в файле: {len(cache._data)}.",
        "   - Ключ кэша: `{scorer}:sha1(sig_old + \"::\" + diff_code + \"::\" + doc_old)` per R12 step 5 (префикс scorer изолирует M0/M2; сигнатура входит в ключ, т.к. M2 принимает sig+diff+doc). Повторный запуск полностью исключает повторные вызовы инференса модели. Запись атомарна (tmp + `os.replace`), повреждённый файл карантинируется в `*.corrupted.bak`.",
    ])
    if cache.misses == 0 and (cache.hits or 0) > 0:
        lines.append(
            "   - 0 промахов означает полностью кэшированный повторный прогон; холодный прогон даёт misses > 0 (число инференсов модели)."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
