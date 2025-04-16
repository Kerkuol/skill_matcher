import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# Sample data for skills distribution
sample_data = {
    "Резюме": ["Resume 1", "Resume 2"],
    "Категория": ["Category 1", "Category 2", "Category 3"],
    "Отсутствует": [2, 1, 3],
    "Развитие": [1, 2, 1],
    "Подтверждено": [3, 2, 1]
}

df_category = pd.DataFrame(sample_data)

# Create skills distribution chart
fig_distribution = go.Figure()

for category in df_category["Категория"].unique():
    df_cat = df_category[df_category["Категория"] == category]
    
    fig_distribution.add_trace(go.Bar(
        x=df_cat["Резюме"],
        y=df_cat["Отсутствует"],
        name=f"{category} (Отсутствует)",
        marker_color="#2c5282",
        text=df_cat["Отсутствует"],
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>" +
                    "Отсутствует: %{y}<br>" +
                    f"Категория: {category}<extra></extra>",
        orientation='v'
    ))

fig_distribution.update_layout(
    barmode='stack',
    title="Распределение навыков по резюме",
    xaxis_title="Резюме",
    yaxis_title="Количество навыков",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font={'color': '#e2e8f0'},
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=500
)

fig_distribution.update_xaxes(showgrid=False, color='#e2e8f0')
fig_distribution.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', color='#e2e8f0')

st.plotly_chart(fig_distribution, use_container_width=True, key="skills_distribution_chart") 