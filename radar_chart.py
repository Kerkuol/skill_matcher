import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# Sample data for radar chart
sample_data = {
    "Кандидат": ["Candidate 1", "Candidate 2"],
    "Категория": ["Category 1", "Category 2", "Category 3"],
    "Процент подтверждения": [80, 60, 70],
    "Процент развития": [10, 20, 15],
    "Процент отсутствия": [10, 20, 15]
}

df_analysis = pd.DataFrame(sample_data)

# Create radar chart
fig_radar = go.Figure()

for candidate in df_analysis["Кандидат"].unique():
    df_candidate = df_analysis[df_analysis["Кандидат"] == candidate]
    
    fig_radar.add_trace(go.Scatterpolar(
        r=df_candidate["Процент подтверждения"],
        theta=df_candidate["Категория"],
        name=f"{candidate} (Подтверждено)",
        fill='toself',
        line=dict(width=2),
        hovertemplate="<b>%{theta}</b><br>" +
                    "Подтверждено: %{r}%<br>" +
                    f"Кандидат: {candidate}<extra></extra>"
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=df_candidate["Процент развития"],
        theta=df_candidate["Категория"],
        name=f"{candidate} (Развитие)",
        fill='toself',
        line=dict(width=2, dash='dash'),
        hovertemplate="<b>%{theta}</b><br>" +
                    "Требует развития: %{r}%<br>" +
                    f"Кандидат: {candidate}<extra></extra>"
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=df_candidate["Процент отсутствия"],
        theta=df_candidate["Категория"],
        name=f"{candidate} (Отсутствует)",
        fill='toself',
        line=dict(width=2, dash='dot'),
        hovertemplate="<b>%{theta}</b><br>" +
                    "Отсутствует: %{r}%<br>" +
                    f"Кандидат: {candidate}<extra></extra>"
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            color='#e2e8f0',
            gridcolor='rgba(128,128,128,0.2)'
        ),
        angularaxis=dict(
            color='#e2e8f0'
        ),
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    font={'color': '#e2e8f0'},
    height=500
)

st.plotly_chart(fig_radar, use_container_width=True, key="skills_radar_chart") 