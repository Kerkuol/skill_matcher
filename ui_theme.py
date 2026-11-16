"""Local, build-free presentation assets for Streamlit and Plotly."""
import base64
from pathlib import Path
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "styles"
PAPER = "#f8f5ed"
INK = "#000000"
CHART_COLORS = ["#ffda6e", "#000000", "#6ece9d", "#707070", "#c7b67f"]


@st.cache_resource
def theme_css(revision):
    faces = []
    for weight in (400, 500, 600):
        data = base64.b64encode((ASSETS / "fonts" / f"Inter-{weight}.ttf").read_bytes()).decode("ascii")
        faces.append(f"@font-face {{font-family: 'Matcher Inter'; font-style: normal; font-weight: {weight}; font-display: swap; src: url(data:font/ttf;base64,{data}) format('truetype');}}")
    return "\n".join(faces + [(ASSETS / name).read_text(encoding="utf-8-sig") for name in ("variables.css", "main.css")])


def apply_theme():
    # Invalidate the cached string when local design assets change.
    revision = tuple(path.stat().st_mtime_ns for path in
                     [ASSETS / "variables.css", ASSETS / "main.css"] +
                     [ASSETS / "fonts" / f"Inter-{weight}.ttf" for weight in (400, 500, 600)])
    st.markdown(f"<style>{theme_css(revision)}</style>", unsafe_allow_html=True)


def show_plotly(figure, **kwargs):
    """Only presentation: retain traces, data, ranges and chart interactions."""
    figure.update_layout(
        template="plotly_white", paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family="Matcher Inter, Inter, sans-serif", color=INK),
        title_font=dict(color=INK, size=18),
        colorway=CHART_COLORS,
        legend=dict(font=dict(color=INK), bgcolor=PAPER),
    )
    figure.update_xaxes(color=INK, gridcolor="rgba(0,0,0,0.10)", zerolinecolor="rgba(0,0,0,0.18)")
    figure.update_yaxes(color=INK, gridcolor="rgba(0,0,0,0.10)", zerolinecolor="rgba(0,0,0,0.18)")
    figure.update_polars(bgcolor=PAPER, radialaxis=dict(color=INK, gridcolor="rgba(0,0,0,0.15)"), angularaxis=dict(color=INK, gridcolor="rgba(0,0,0,0.15)"))
    st.plotly_chart(figure, theme=None, **kwargs)
