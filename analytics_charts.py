"""Presentation of existing assessment results; no scoring or inference."""
from html import escape
import textwrap

import plotly.graph_objects as go

STATUS_COLORS = {"Подтверждены": "#6ece9d", "Требуют развития": "#ffda6e", "Отсутствуют": "#b5b2a8"}
STATUS_FIELDS = {"Подтверждены": "matches", "Требуют развития": "partials", "Отсутствуют": "missing"}


def category_label(name, width=19):
    return "<br>".join(escape(line) for line in textwrap.wrap(name, width=width, break_long_words=False, break_on_hyphens=False))


def coverage_chart(categories):
    totals = [sum(len(data[field]) for data in categories.values()) for field in STATUS_FIELDS.values()]
    total = sum(totals)
    fig = go.Figure(go.Pie(
        labels=list(STATUS_FIELDS), values=totals, hole=.76, sort=False,
        direction="clockwise", rotation=25, textinfo="none",
        marker=dict(colors=list(STATUS_COLORS.values()), line=dict(color="#f8f5ed", width=4)),
        hovertemplate="<b>%{label}</b><br>%{value} навыков · %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Покрытие матрицы", height=390,
        margin=dict(l=24, r=24, t=70, b=65),
        annotations=[dict(x=.5, y=.54, text=f"<b>{total}</b>", showarrow=False, font=dict(size=44)),
                     dict(x=.5, y=.34, text="навыков в матрице", showarrow=False, font=dict(size=12, color="#77746b"))],
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.13, yanchor="top"),
    )
    if not total:
        fig.add_annotation(x=.5, y=.8, text="Нет навыков для отображения", showarrow=False)
    return fig


def competency_chart(categories):
    names = list(categories)
    scores = [categories[name]["percentage"] for name in names]
    # Repeat the first point so the outline closes as well as its fill.
    labels = [category_label(name, width=10) for name in names]
    fig = go.Figure(go.Scatterpolar(
        r=scores + scores[:1], theta=labels + labels[:1],
        customdata=names + names[:1], mode="lines+markers", fill="toself",
        fillcolor="rgba(255,218,110,.30)", line=dict(color="#c49b32", width=2.5),
        marker=dict(size=8, color="#ffda6e", line=dict(color="#fffdf7", width=2)),
        hovertemplate="<b>%{customdata}</b><br>Соответствие: %{r:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Профиль компетенций", showlegend=False, height=390,
        margin=dict(l=80, r=80, t=85, b=55),
        annotations=[dict(x=.5, y=-.18, xref="paper", yref="paper", showarrow=False,
                          text="Соответствие по категориям · 0–100%", font=dict(size=10, color="#77746b"))],
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], showticklabels=False,
                                   ticksuffix="%", tickfont=dict(size=10), tickangle=0, angle=0, showline=False),
                   angularaxis=dict(tickfont=dict(size=12), direction="clockwise", rotation=90)),
    )
    return fig


def distribution_chart(categories):
    names = list(categories)
    fig = go.Figure()
    for status, field in STATUS_FIELDS.items():
        counts = [len(categories[name][field]) for name in names]
        fig.add_bar(
            x=[category_label(name) for name in names], y=counts, name=status,
            customdata=names, marker=dict(color=STATUS_COLORS[status], line=dict(width=0)),
            text=[str(count) if count else "" for count in counts], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#34352e", size=12),
            hovertemplate=f"<b>%{{customdata}}</b><br>{status}: %{{y}}<extra></extra>",
        )
    fig.update_layout(
        title="Навыки по категориям", barmode="stack", bargap=.62, height=440,
        margin=dict(l=40, r=24, t=75, b=130),
        legend=dict(orientation="h", x=0, y=-.25, yanchor="top", title_text=""),
        xaxis=dict(title=None, showgrid=False, tickfont=dict(size=12), fixedrange=True),
        yaxis=dict(title="Навыки", dtick=1, rangemode="tozero", fixedrange=True),
    )
    return fig
