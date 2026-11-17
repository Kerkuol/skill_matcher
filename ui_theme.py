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
    return "\n".join(faces + [(ASSETS / name).read_text(encoding="utf-8-sig") for name in ("variables.css", "main.css", "glass.css")])


def apply_theme():
    # Invalidate the cached string when local design assets change.
    revision = tuple(path.stat().st_mtime_ns for path in
                     [ASSETS / name for name in ("variables.css", "main.css", "glass.css")] +
                     [ASSETS / "fonts" / f"Inter-{weight}.ttf" for weight in (400, 500, 600)])
    st.markdown(f"<style>{theme_css(revision)}</style>", unsafe_allow_html=True)


def show_plotly(figure, **kwargs):
    """Only presentation: retain traces, data, ranges and chart interactions."""
    figure.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Matcher Inter, Inter, sans-serif", color="#34352e", size=13),
        title=dict(font=dict(color="#24251f", size=19), x=.045, xanchor="left", y=.95),
        colorway=CHART_COLORS,
        legend=dict(font=dict(color="#68665e", size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#fffdf7", bordercolor="#e7e1d2", font=dict(color=INK, size=13)),
        hovermode="closest", transition=dict(duration=650, easing="cubic-in-out"),
    )
    figure.update_xaxes(color="#77746b", gridcolor="rgba(85,78,57,0.07)", zeroline=False, showline=False, automargin=True)
    figure.update_yaxes(color="#77746b", gridcolor="rgba(85,78,57,0.07)", zeroline=False, showline=False, automargin=True)
    figure.update_polars(bgcolor="rgba(255,255,255,.15)", radialaxis=dict(color="#8b877c", gridcolor="rgba(85,78,57,0.10)"), angularaxis=dict(color="#68665e", gridcolor="rgba(85,78,57,0.08)", linecolor="rgba(85,78,57,0.10)"))
    kwargs.setdefault("config", {"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    st.plotly_chart(figure, theme=None, **kwargs)
