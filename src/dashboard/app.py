from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
import pandas as pd
import requests
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.core import (
    DEFAULT_API_URL,
    DEFAULT_DB_PATH,
    execute_verify,
    get_available_branches,
    get_available_commits,
    get_available_projects,
    get_risk_level_info,
    load_freshness_data,
    load_section_history,
    resolve_db_path,
)


def _check_api_health(api_url: str) -> bool:
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False


def _extract_diff(evidence: Any) -> str | None:
    if not evidence:
        return None
    if isinstance(evidence, dict):
        if "diff" in evidence:
            return str(evidence["diff"])
        if "code_diff" in evidence:
            return str(evidence["code_diff"])
    elif isinstance(evidence, str):
        trimmed = evidence.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                parsed = json.loads(trimmed)
                if isinstance(parsed, dict):
                    return parsed.get("diff") or parsed.get("code_diff")
            except Exception:
                pass
        return evidence
    return None


def main() -> None:
    st.set_page_config(
        page_title="FreshDoc — Карта свежести документации",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #e9ecef;
        }
        .risk-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("📑 FreshDoc")
        st.caption("Реестр состояния и карта свежести документации (R12 / FD-30)")

        # Connection settings
        with st.expander("⚙️ Источник данных и API", expanded=False):
            db_input = st.text_input(
                "Путь к SQLite БД",
                value=str(resolve_db_path()),
                help="Путь к файлу doc_state SQLite",
            )
            api_input = st.text_input(
                "URL FastAPI",
                value=os.environ.get("FRESHDOC_API_URL", DEFAULT_API_URL),
                help="Адрес REST API FreshDoc для вызовов POST /verify",
            )

        db_path = Path(db_input)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path

        api_online = _check_api_health(api_input)
        if api_online:
            st.success("🟢 API активен", icon="✅")
        else:
            st.info("⚪ API офлайн (прямой доступ к БД)", icon="ℹ️")

        st.divider()

        # Repositories & branches
        if not db_path.exists():
            st.warning(f"База данных {db_path.name} не найдена.")
            repos = []
        else:
            repos = get_available_projects(db_path)

        if not repos:
            st.warning("В реестре пока нет записей.")
            if st.button("🌱 Посеять демонстрационные данные"):
                from scripts.seed_demo_registry import build_demo_repo, seed_demo_db

                demo_repo = build_demo_repo(PROJECT_ROOT / "data/demo_repo")
                seed_demo_db(db_path, demo_repo, branch="demo")
                st.success("Демо-данные успешно добавлены!")
                st.rerun()
            return

        selected_repo = st.selectbox(
            "Проект (Repository)",
            options=repos,
            index=0,
            help="Выбор репозитория для анализа",
        )

        branches = get_available_branches(db_path, selected_repo)
        if not branches:
            st.warning(f"Нет веток для репозитория {selected_repo}")
            return

        selected_branch = st.selectbox(
            "Ветка (Branch)",
            options=branches,
            index=0,
            help="Выбор ветки проекта (смена ветки меняет состояние разделов)",
        )

        # Commits
        commits = get_available_commits(db_path, selected_repo, selected_branch)
        commit_options = ["(Последний срез / Head)"] + [
            f"{c['commit_sha'][:8]} ({c.get('computed_at', '')[:10]})"
            for c in commits
        ]
        commit_choice = st.selectbox(
            "Срез по коммиту (опционально)", options=commit_options, index=0
        )

        selected_commit_sha = None
        if commit_choice != "(Последний срез / Head)":
            idx = commit_options.index(commit_choice) - 1
            selected_commit_sha = commits[idx]["commit_sha"]

        st.divider()

        # Filters
        min_risk_filter = st.slider(
            "Мин. порог риска (min_risk)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Показывать только разделы с риском выше порога",
        )

        search_query = st.text_input(
            "Поиск по символу / файлу",
            value="",
            placeholder="например, auth.py или fetch",
        )

        st.caption(f"Проект: **{selected_repo}** | Ветка: **{selected_branch}**")

    # -------------------------------------------------------------------------
    # Main View
    # -------------------------------------------------------------------------
    st.subheader(f"Карта свежести: `{selected_repo}` @ `{selected_branch}`")
    if selected_commit_sha:
        st.caption(f"Срез на коммите: `{selected_commit_sha}`")
    else:
        st.caption("Актуальный срез на вершине ветки")

    sections = load_freshness_data(
        db_path=db_path,
        repo=selected_repo,
        branch=selected_branch,
        commit=selected_commit_sha,
        min_risk=min_risk_filter if min_risk_filter > 0.0 else None,
        search=search_query if search_query else None,
    )

    # Key Metrics
    total_sections = len(sections)
    at_risk_sections = sum(1 for s in sections if (s.get("risk") or 0.0) >= 0.5)
    max_risk = max((s.get("risk") or 0.0 for s in sections), default=0.0)
    avg_urgency = (
        sum((s.get("urgency") or 0.0 for s in sections)) / total_sections
        if total_sections > 0
        else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего разделов", total_sections)
    with col2:
        st.metric("Требуют внимания (риск ≥ 0.5)", at_risk_sections)
    with col3:
        st.metric("Максимальный риск", f"{max_risk:.3f}")
    with col4:
        st.metric("Средняя срочность", f"{avg_urgency:.3f}")

    tab_map, tab_replay, tab_api = st.tabs(
        ["🗺️ Карта свежести и карточка", "📈 Replay динамика", "🔌 API и Демо"]
    )

    # -------------------------------------------------------------------------
    # TAB 1: Freshness Map & Section Card
    # -------------------------------------------------------------------------
    with tab_map:
        if not sections:
            st.info("Разделов, удовлетворяющих заданным критериям фильтрации, не найдено.")
            return

        table_data = []
        for s in sections:
            rsk = s.get("risk")
            urg = s.get("urgency")
            imp = s.get("impact")
            table_data.append(
                {
                    "Срочность": float(f"{urg:.3f}") if urg is not None else 0.0,
                    "Риск": float(f"{rsk:.3f}") if rsk is not None else 0.0,
                    "Влияние": float(f"{imp:.3f}") if imp is not None else 0.0,
                    "Символ": s.get("symbol") or "—",
                    "Тип дрейфа": s.get("drift_type") or "—",
                    "Событие": s.get("event") or "—",
                    "Файл": s.get("doc_path") or "—",
                    "Идентификатор раздела": s.get("section_id"),
                }
            )

        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            column_config={
                "Срочность": st.column_config.ProgressColumn(
                    "Срочность (Urgency)",
                    help="Срочность обновления по формуле R12",
                    format="%.2f",
                    min_value=0.0,
                    max_value=2.0,
                ),
                "Риск": st.column_config.ProgressColumn(
                    "Риск (Risk)",
                    help="Накопленный риск устаревания docstring",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Влияние": st.column_config.NumberColumn(
                    "Влияние (Impact)",
                    format="%.2f",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Section Detail & Action Card
        st.subheader("🔍 Карточка раздела и подтверждение")
        section_ids = [s["section_id"] for s in sections]
        selected_section_id = st.selectbox(
            "Выберите раздел для инспекции и подтверждения:",
            options=section_ids,
            index=0,
            help="Выберите раздел, чтобы изучить дифф кода и при необходимости сбросить риск",
        )

        chosen_item = next(
            (s for s in sections if s["section_id"] == selected_section_id), None
        )

        if chosen_item:
            card_risk = chosen_item.get("risk") or 0.0
            label, color = get_risk_level_info(card_risk)

            c_info, c_action = st.columns([3, 1])
            with c_info:
                st.markdown(
                    f"### `{chosen_item.get('symbol') or chosen_item['section_id']}`"
                )
                badge_html = (
                    f"<span class='risk-badge' style='background-color: {color};'>"
                    f"{label} ({card_risk:.3f})</span>"
                )
                st.markdown(badge_html, unsafe_allow_html=True)
                st.write(
                    f"**Файл:** `{chosen_item.get('doc_path')}` | "
                    f"**Тип дрейфа:** `{chosen_item.get('drift_type') or 'не определён'}` | "
                    f"**Событие:** `{chosen_item.get('event')}` | "
                    f"**Последний коммит:** `{chosen_item.get('commit_sha')}`"
                )
                if chosen_item.get("verified_commit"):
                    st.caption(
                        f"Ранее подтверждён на коммите: `{chosen_item.get('verified_commit')}`"
                    )

            with c_action:
                st.write("**Действие инженера / агента:**")
                verify_btn = st.button(
                    "✅ Подтвердить актуальность",
                    type="primary",
                    use_container_width=True,
                    help="Обнуляет накопленный риск раздела через POST /verify (критерий приёмки)",
                )

                if verify_btn:
                    with st.spinner("Фиксация подтверждения..."):
                        target_commit = (
                            chosen_item.get("commit_sha")
                            or selected_commit_sha
                            or "manual_verify"
                        )
                        result = execute_verify(
                            repo=selected_repo,
                            branch=selected_branch,
                            section_id=chosen_item["section_id"],
                            commit=target_commit,
                            api_url=api_input if api_online else None,
                            db_path=db_path,
                            verified_by="dashboard_user",
                        )
                        src_msg = "через API" if result["source"] == "api" else "в локальный реестр"
                        st.success(f"Раздел подтверждён ({src_msg})! Риск сброшен в 0.0.")
                        st.rerun()

            # Diff & History Tabs
            d_tab, h_tab = st.tabs(["📝 Причина дрейфа (Diff)", "📜 История изменений"])

            with d_tab:
                diff_text = _extract_diff(chosen_item.get("evidence"))
                if diff_text:
                    st.caption("Изменения кода, приведшие к дрейфу документации:")
                    st.code(diff_text, language="diff")
                else:
                    st.info("Дифф кода в записи отсутствует или документация не расходилась.")

                evidence_raw = chosen_item.get("evidence")
                if isinstance(evidence_raw, dict):
                    if "doc_old" in evidence_raw or "doc_new" in evidence_raw:
                        st.caption("Сравнение docstring:")
                        col_old, col_new = st.columns(2)
                        with col_old:
                            st.text_area(
                                "Предыдущий docstring",
                                value=evidence_raw.get("doc_old", ""),
                                height=150,
                                disabled=True,
                            )
                        with col_new:
                            st.text_area(
                                "Обновлённый docstring",
                                value=evidence_raw.get("doc_new", ""),
                                height=150,
                                disabled=True,
                            )

            with h_tab:
                hist_records = load_section_history(
                    db_path=db_path,
                    section_id=chosen_item["section_id"],
                    repo=selected_repo,
                    branch=selected_branch,
                )
                if hist_records:
                    h_df = pd.DataFrame(
                        [
                            {
                                "Дата": h.get("computed_at", "")[:19],
                                "Коммит": (h.get("commit_sha") or "")[:8],
                                "Событие": h.get("event"),
                                "Риск": float(f"{h.get('risk', 0.0):.3f}")
                                if h.get("risk") is not None
                                else 0.0,
                                "Срочность": float(f"{h.get('urgency', 0.0):.3f}")
                                if h.get("urgency") is not None
                                else 0.0,
                                "Подтверждён на": (h.get("verified_commit") or "")[:8],
                            }
                            for h in hist_records
                        ]
                    )
                    st.dataframe(h_df, use_container_width=True, hide_index=True)

                    # Plot risk progression
                    if len(hist_records) > 1:
                        st.caption("Динамика накопления и сброса риска:")
                        chart_data = pd.DataFrame(
                            {
                                "Шаг (seq)": [h.get("seq", i) for i, h in enumerate(hist_records)],
                                "Риск": [h.get("risk") or 0.0 for h in hist_records],
                            }
                        ).set_index("Шаг (seq)")
                        st.line_chart(chart_data)
                else:
                    st.info("История событий для данного раздела пуста.")

    # -------------------------------------------------------------------------
    # TAB 2: Replay Timeline
    # -------------------------------------------------------------------------
    with tab_replay:
        st.subheader("Динамика риска на истории коммитов (FD-29 / R12)")
        st.markdown(
            """
            График иллюстрирует накопление риска моделью M2 / M0 при изменениях кода,
            момент превышения порога, опережение и сброс риска при обновлении docstring.
            """
        )

        fig_path = PROJECT_ROOT / f"reports/figures/replay_risk_timeline_{selected_repo}.png"
        if fig_path.exists():
            st.image(str(fig_path), caption=f"Динамика риска во времени: {selected_repo}")
        else:
            st.info(f"График replay для '{selected_repo}' не найден в reports/figures/.")

        # Replay lead-time summary highlights
        replay_summary_md = PROJECT_ROOT / "reports/replay.md"
        if replay_summary_md.exists():
            with st.expander("📖 Справка из отчёта reports/replay.md"):
                st.markdown(replay_summary_md.read_text(encoding="utf-8")[:1500] + "\n...")

    # -------------------------------------------------------------------------
    # TAB 3: API & Agent Demo
    # -------------------------------------------------------------------------
    with tab_api:
        st.subheader("Взаимодействие с API «от лица агента» (R12)")
        st.markdown(
            """
            Агенты разработки и CI/CD могут запрашивать состояние реестра через REST API
            и отправлять подтверждения `POST /verify`.
            """
        )

        sample_curl = (
            f'# Запрос статуса от внешнего агента:\n'
            f'curl -X GET "{api_input}/status?repo={selected_repo}&branch={selected_branch}'
            f'&min_risk=0.5"\n\n'
            f'# Подтверждение актуальности:\n'
            f'curl -X POST "{api_input}/verify" \\\n'
            f'     -H "Content-Type: application/json" \\\n'
            f'     -d \'{{"repo": "{selected_repo}", "branch": "{selected_branch}", '
            f'"section_id": "...", "commit": "..."}}\'\n'
        )
        st.code(sample_curl, language="bash")

        if st.button("🚀 Выполнить тестовый запрос GET /status через API"):
            if not api_online:
                st.error("FastAPI сервер недоступен. Запустите: uvicorn src.api.app:app")
            else:
                try:
                    resp = requests.get(
                        f"{api_input}/status",
                        params={
                            "repo": selected_repo,
                            "branch": selected_branch,
                            "min_risk": 0.5,
                        },
                        timeout=3.0,
                    )
                    st.json(resp.json())
                except Exception as e:
                    st.error(f"Ошибка запроса: {e}")
                    st.json(resp.json())
                except Exception as e:
                    st.error(f"Ошибка запроса: {e}")


if __name__ == "__main__":
    main()
