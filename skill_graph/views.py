"""Compact result block and multi-resume comparison for the application."""
from html import escape

import plotly.graph_objects as go
import streamlit as st

from .network import render_network
from .presentation import coverage_chart, competency_chart, render_missing_recommendations
from .scoring import STATUS_NAMES
from ui_theme import show_plotly


def counts(result):
    categories = result["categories_analysis"]
    return {field: sum(len(c[field]) for c in categories.values()) for field in ("matches", "partials", "missing")}


def card(column, value, label):
    with column:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{escape(str(value))}</div>'
                    f'<div class="metric-label">{escape(label)}</div></div>', unsafe_allow_html=True)


def render_single(results, profile_name, show_title=True, resume_id=None):
    key = resume_id or "single"
    categories = results["categories_analysis"]
    totals = counts(results)
    support = results["graph_support"]
    if show_title:
        st.write("### Результаты анализа")
    row = st.columns(3)
    card(row[0], f"{results['total_percentage']:g}%", "Общее соответствие")
    card(row[1], f"{support['supported']} / {support['total']}", "Распознано требований")
    card(row[2], sum(c["percentage"] > 0 for c in categories.values()), "Категорий навыков")
    row = st.columns(3)
    card(row[0], totals["matches"], "Подтверждённых навыков")
    card(row[1], totals["missing"], "Неподтверждённых навыков")
    card(row[2], totals["partials"], "Навыков нужно уточнить")
    tabs = st.tabs(["Визуализация", "Навыки", "Рекомендации"])
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            show_plotly(coverage_chart(categories), use_container_width=True, key=f"pie_chart_{key}")
        with right:
            show_plotly(competency_chart(categories), use_container_width=True, key=f"radar_chart_{key}")
    with tabs[1]:
        groups = [("Подтверждены", "practical"), ("Нужно уточнить", "partial"), ("Не подтверждены", "missing")]
        for tab, (label, group) in zip(st.tabs([g[0] for g in groups]), groups):
            with tab:
                rows = []
                for item in results["graph_details"]:
                    status = item["status"]
                    bucket = "practical" if status == "practical" else "partial" if status in ("declared", "participation") else "missing"
                    if bucket != group:
                        continue
                    quotes = list(dict.fromkeys(e["quote"] for e in item["evidence"] if e["status"] == status))
                    rows.append({"Категория": item["category"], "Навык": item["skill"],
                                 "Основание": STATUS_NAMES[status], "Фрагмент резюме": "\n".join(quotes[:2])})
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("В этой группе нет навыков.")
        st.caption("Не найдено подтверждения в резюме — не значит, что у кандидата нет навыка.")
    with tabs[2]:
        render_missing_recommendations(results)
    st.write("### Граф навыков")
    render_network(results["graph_details"])


def comparison_rows(resumes):
    ranked = sorted(enumerate(resumes), key=lambda pair: (-pair[1]["results"]["total_percentage"], pair[0]))
    return [(index, {"Кандидат": item["file_name"], "Соответствие, %": item["results"]["total_percentage"],
                      "Подтверждено": counts(item["results"])["matches"],
                      "Нужно уточнить": counts(item["results"])["partials"],
                      "Не подтверждено": counts(item["results"])["missing"]}) for index, item in ranked]


def comparison_heatmap(resumes):
    # Match rows by category+skill, never by the order a candidate returned them.
    requirements = list(dict.fromkeys((r["category"], r["skill"]) for item in resumes for r in item["results"]["graph_details"]))
    labels = [f"{i + 1:02d} · {item['file_name']}" for i, item in enumerate(resumes)]
    maps = [{(r["category"], r["skill"]): r for r in item["results"]["graph_details"]} for item in resumes]
    z, hover = [], []
    for category, skill in requirements:
        values, texts = [], []
        for label, rows in zip(labels, maps):
            row = rows.get((category, skill))
            status = row["status"] if row else "unsupported"
            values.append(-1 if status == "unsupported" else 2 if status == "practical" else 1 if status in ("declared", "participation") else 0)
            texts.append(f"{escape(label)}<br>{escape(category)} · {escape(skill)}<br>{STATUS_NAMES[status]}")
        z.append(values); hover.append(texts)
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=[f"{category} · {skill}" for category, skill in requirements],
                             customdata=hover, zmin=-1, zmax=2, xgap=5, ygap=5,
                             colorscale=[[0, "#c9c2e8"], [1/6, "#c9c2e8"], [1/6, "#dbd8ce"],
                                         [.5, "#dbd8ce"], [.5, "#ffda6e"], [5/6, "#ffda6e"],
                                         [5/6, "#57c795"], [1, "#57c795"]],
                             showscale=False, hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(height=max(330, len(requirements) * 32 + 130), margin=dict(l=25, r=20, t=25, b=90),
                      xaxis=dict(side="top", tickangle=-15), yaxis=dict(autorange="reversed"))
    return fig


def render_comparison(resumes_data, profile_name):
    st.write("### Сравнение резюме")
    ranking = comparison_rows(resumes_data)
    with st.expander(f"Сравнить кандидатов · {len(resumes_data)} резюме", expanded=True):
        overview, skills = st.tabs(["Рейтинг", "Сравнение навыков"])
        with overview:
            st.dataframe([row for _, row in ranking], use_container_width=True, hide_index=True,
                         column_config={"Соответствие, %": st.column_config.ProgressColumn("Соответствие", min_value=0, max_value=100, format="%.1f%%")})
        with skills:
            show_plotly(comparison_heatmap(resumes_data), use_container_width=True, key="comparison_skill_map")
            st.caption("Зелёный — подтверждено · жёлтый — уточнить · серый — не подтверждено · сиреневый — механизм не распознал требование.")
    labels = {index: f"{index + 1:02d} · {resumes_data[index]['file_name']}" for index, _ in ranking}
    options = [labels[index] for index, _ in ranking]
    selected = st.selectbox("Результаты и граф кандидата", options, key="comparison_candidate")
    index = next(index for index, label in labels.items() if label == selected)
    item = resumes_data[index]
    render_single(item["results"], profile_name, resume_id=f"candidate_{index}")
