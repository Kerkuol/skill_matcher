"""Presence-only recommendations and existing chart adapters for graph-based screening."""
import streamlit as st
from analytics_charts import coverage_chart as _coverage, competency_chart, distribution_chart as _distribution

STATUS_COLORS = {"Подтверждены": "#6ece9d", "Нужно уточнить": "#ffda6e", "Не подтверждены": "#b5b2a8"}


def relabel(figure):
    names = {"Требуют развития": "Нужно уточнить", "Отсутствуют": "Не подтверждены"}
    for trace in figure.data:
        if trace.name in names:
            trace.name = names[trace.name]
        if trace.type == "pie":
            trace.labels = [names.get(label, label) for label in trace.labels]
        if trace.hovertemplate:
            for old, new in names.items():
                trace.hovertemplate = trace.hovertemplate.replace(old, new)
    return figure


def coverage_chart(categories):
    return relabel(_coverage(categories))


def distribution_chart(categories):
    return relabel(_distribution(categories))


def missing_requirements(results):
    return [r for r in results["graph_details"] if r["status"] != "practical"]


def render_missing_recommendations(results):
    st.write("#### Какие навыки не подтверждены в резюме")
    st.caption("Это требования выбранной матрицы, для которых не найдено описания личной практики. "
               "Они могут быть у кандидата: уточните опыт на интервью или запросите пример работы.")
    rows = missing_requirements(results)
    gaps = [r for r in rows if r["status"] != "unsupported"]
    unsupported = [r for r in rows if r["status"] == "unsupported"]
    if gaps:
        from .scoring import STATUS_NAMES
        st.dataframe([{"Категория": r["category"], "Навык": r["skill"],
                       "Что найдено": STATUS_NAMES[r["status"]],
                       "Что уточнить": "Попросить пример личного применения навыка"} for r in gaps],
                     use_container_width=True, hide_index=True)
    elif not unsupported:
        st.success("Для всех требований матрицы найдено описание практики.")
    if unsupported:
        st.write("**Эти требования механизм пока не умеет проверять**")
        st.dataframe([{"Категория": r["category"], "Навык": r["skill"]} for r in unsupported],
                     use_container_width=True, hide_index=True)
        st.caption("Они не считаются доказанным пробелом кандидата и требуют ручной проверки.")
