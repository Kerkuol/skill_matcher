"""Screen navigation compatible with Streamlit 1.32 and Community Cloud."""
import streamlit as st
import sqlite3
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
    if draft.get("edit_mode") == "Карточки":
        from matrix_editor import capture_skill_details
        capture_skill_details()
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
    actions = st.columns([.5, 1, 1, .5])
    with actions[1]:
        st.button("Перейти к анализу", type="primary", on_click=go_to, args=("analysis",), use_container_width=True)
    with actions[2]:
        st.button("Настроить матрицы", on_click=go_to, args=("settings",), use_container_width=True)
    st.markdown("""
        <div class="landing-steps">
          <article class="flow-card"><span class="section-index">01 / ТРЕБОВАНИЯ</span>
            <h3>Ваша матрица</h3><p>Выберите готовую матрицу или создайте свою: навыки, категории и описания требований.</p></article>
          <article class="flow-card"><span class="section-index">02 / КАНДИДАТЫ</span>
            <h3>Одно резюме или несколько</h3><p>Загрузите PDF, DOCX или TXT на странице анализа. Матрицу можно переключать там же.</p></article>
          <article class="flow-card"><span class="section-index">03 / РЕЗУЛЬТАТ</span>
            <h3>Аналитика для решения</h3><p>Изучите соответствие требованиям, распределение навыков и сравнение кандидатов.</p></article>
        </div>
        <div class="landing-note"><span class="status-dot"></span>Первичный скрининг помогает подготовиться к оценке кандидата. Итоговое решение остаётся за HR.</div>
    """, unsafe_allow_html=True)

    from matrix_storage import load_matrices
    from profiles import profiles as templates
    try:
        matrix_count = len(load_matrices(templates))
    except (OSError, ValueError, sqlite3.Error):
        matrix_count = "—"
    files_count = len(st.session_state.get("workspace_files", []))
    analysis_count = len(st.session_state.get("history", []))
    st.markdown(f'''<section class="workspace-panel reveal"><div><span class="section-index">ВАШЕ РАБОЧЕЕ ПРОСТРАНСТВО</span><h2>Всё готово к следующему кандидату</h2><p>Требования хранятся в библиотеке. Резюме и результаты остаются под рукой, пока вы работаете в этой сессии.</p></div><div class="workspace-stats"><div><b>{matrix_count}</b><span>матриц в библиотеке</span></div><div><b>{files_count}</b><span>резюме в работе</span></div><div><b>{analysis_count}</b><span>записей в истории</span></div></div></section>''', unsafe_allow_html=True)
    if files_count:
        st.button("Продолжить работу с резюме", on_click=go_to, args=("analysis",), key="landing_resume")
    st.markdown('''<section class="product-section reveal"><div><span class="section-index">ОТ ТРЕБОВАНИЙ К ОБСУЖДЕНИЮ</span><h2>Одна структура.<br/>Разные вакансии.</h2><p>Сформулируйте, что важно для роли: название навыка и пояснение. Копируйте матрицы для похожих вакансий, сохраняя исходные требования.</p><p>В редакторе можно работать с отдельными компетенциями или перейти к массовым правкам в таблице.</p></div><div class="sample-matrix"><span class="section-index">ПРИМЕР МАТРИЦЫ / РАЗРАБОТЧИК</span><div><b>Python</b><span class="level-pill">Навык</span></div><div><b>SQL</b><span class="level-pill">Навык</span></div><div><b>Docker</b><span class="level-pill">Навык</span></div><small>Состав требований определяете вы.</small></div></section>
    <section class="product-section reveal"><div class="insight-preview"><span class="section-index">НА СТРАНИЦЕ АНАЛИЗА</span><h3>Общая картина и детали</h3><div class="insight-chips"><span>Соответствие</span><span>Навыки</span><span>Категории</span><span>Сравнение</span></div><p>От общей метрики к конкретным требованиям и отдельным кандидатам.</p></div><div><span class="section-index">БОЛЬШЕ КОНТЕКСТА</span><h2>Изучайте результаты,<br/>а не только процент</h2><p>Просматривайте распределение навыков, подтверждённые и недостающие компетенции. Загружайте несколько резюме для сравнительного анализа.</p><p>Переключайте активную матрицу на странице анализа, чтобы рассмотреть тот же набор кандидатов под другие требования.</p></div></section>''', unsafe_allow_html=True)
    st.subheader("Перед началом работы")
    with st.expander("Какие файлы можно загрузить?"):
        st.write("PDF с текстовым слоем, DOCX и TXT. Можно добавить одно резюме или несколько; максимальный размер каждого файла — 10 МБ. Для сканированных PDF сначала потребуется распознавание текста.")
    with st.expander("Нужно ли создавать матрицу с нуля?"):
        st.write("Нет. В библиотеке есть готовые профили. Создайте копию подходящей матрицы и измените компетенции и описания под вашу вакансию.")
    with st.expander("Что означает процент соответствия?"):
        st.write("Это результат сопоставления сведений в резюме с требованиями выбранной матрицы. Он помогает организовать первичный скрининг, но не заменяет интервью и проверку навыков.")
    st.markdown('<section class="landing-final reveal"><span class="section-index">СЛЕДУЮЩИЙ ШАГ</span><h2>Начните с вашей вакансии</h2><p>Подготовьте требования или сразу перейдите к анализу по готовой матрице.</p></section>', unsafe_allow_html=True)
    finish = st.columns(2)
    with finish[0]:
        st.button("Открыть анализ резюме", type="primary", on_click=go_to, args=("analysis",), use_container_width=True)
    with finish[1]:
        st.button("Открыть библиотеку матриц", on_click=go_to, args=("settings",), use_container_width=True)


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
