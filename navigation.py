"""Screen navigation compatible with Streamlit 1.32 and Community Cloud."""
import streamlit as st
from matrix_storage import editor_rows

PAGES = {"home": "Главная", "analysis": "Анализ резюме", "settings": "Настройки"}


def current_page():
    requested = st.query_params.get("view", "home")
    return requested if requested in PAGES else "home"


def capture_matrix_draft():
    """Keep a committed UI draft when Streamlit removes off-screen widgets."""
    draft = st.session_state.get("matrix_draft")
    if not draft:
        return
    token = draft["token"]
    draft["name"] = st.session_state.get(f"matrix_name_{token}", draft["name"])
    draft["profile"]["description"] = st.session_state.get(
        f"matrix_description_{token}", draft["profile"].get("description", ""))
    grid = f"matrix_table_{token}_{draft['generation']}"
    if grid in st.session_state:
        rows = editor_rows(st.session_state.get("matrix_data", []), st.session_state[grid])
        st.session_state.matrix_data = rows
        draft["rows_snapshot"] = rows
        draft["generation"] += 1


def go_to(page):
    if page not in PAGES:
        page = "home"
    capture_matrix_draft()
    st.query_params["view"] = page
    st.session_state.app_navigation = PAGES[page]


def _navigation_changed():
    page = next(key for key, label in PAGES.items() if label == st.session_state.app_navigation)
    go_to(page)


def render_navigation():
    page = current_page()
    st.session_state.app_navigation = PAGES[page]
    with st.sidebar:
        st.markdown('<div class="workspace-brand"><span class="brand-mark">SM</span><div><b>Skill Matcher</b><small>Рабочее пространство HR</small></div></div>', unsafe_allow_html=True)
        st.radio("Раздел приложения", list(PAGES.values()), key="app_navigation",
                 on_change=_navigation_changed, label_visibility="collapsed")
    return page


def render_history():
    with st.sidebar:
        st.divider()
        st.header("История анализа")
        if st.button("Очистить историю"):
            st.session_state.history = []
        for item in st.session_state.get("history", [])[::-1]:
            st.write(f"**{item['file']}** — {item['profile']} — {item['score']}%")


def render_landing():
    st.markdown('<div class="hero-kicker">SKILL MATCHER / ДЛЯ HR</div>', unsafe_allow_html=True)
    st.title("Первичный отбор по вашей матрице")
    st.markdown('<p class="hero-description">Сопоставляйте резюме с требованиями вакансии. Соберите матрицу компетенций, загрузите кандидатов и изучите результаты в одном рабочем пространстве.</p>', unsafe_allow_html=True)
    actions = st.columns([1, 1, 1])
    with actions[0]:
        st.button("Перейти к анализу", type="primary", on_click=go_to, args=("analysis",), use_container_width=True)
    with actions[1]:
        st.button("Настроить матрицы", on_click=go_to, args=("settings",), use_container_width=True)
    st.markdown("""
        <div class="landing-steps">
          <article class="flow-card"><span class="section-index">01 / ТРЕБОВАНИЯ</span>
            <h3>Ваша матрица</h3><p>Выберите готовую матрицу или создайте свою: навыки, категории и требуемые уровни.</p></article>
          <article class="flow-card"><span class="section-index">02 / КАНДИДАТЫ</span>
            <h3>Одно резюме или несколько</h3><p>Загрузите PDF, DOCX или TXT на странице анализа. Матрицу можно переключать там же.</p></article>
          <article class="flow-card"><span class="section-index">03 / РЕЗУЛЬТАТ</span>
            <h3>Аналитика для решения</h3><p>Изучите соответствие требованиям, распределение навыков и сравнение кандидатов.</p></article>
        </div>
        <div class="landing-note"><span class="status-dot"></span>Первичный скрининг помогает подготовиться к оценке кандидата. Итоговое решение остаётся за HR.</div>
    """, unsafe_allow_html=True)


def remember_uploads(widget_key):
    st.session_state.workspace_files = [
        {"name": file.name, "content": file.getvalue()}
        for file in st.session_state.get(widget_key, [])
    ]
    st.session_state.workspace_results = {}


def clear_resumes():
    st.session_state.workspace_files = []
    st.session_state.workspace_results = {}
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1


def choose_active_matrix():
    st.session_state.active_matrix_id = st.session_state.analysis_matrix_picker
