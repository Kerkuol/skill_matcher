from pathlib import Path
import os
import tempfile
PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_TEMP = Path(__file__).resolve().parent / "tmp"
RUNTIME_TEMP.mkdir(exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(RUNTIME_TEMP)
os.environ["MPLCONFIGDIR"] = str(RUNTIME_TEMP / "matplotlib")
tempfile.tempdir = str(RUNTIME_TEMP)
from skill_graph.scoring import assess_resume, matcher, ENGINE_REVISION
from skill_graph.views import render_single, render_comparison
import streamlit as st
from io import BytesIO
from resume_parser import parse_resume
from datetime import datetime
from profiles import profiles as default_profiles
from matrix_storage import load_matrices
import sqlite3
from ui_theme import apply_theme, show_plotly, CHART_COLORS
from matrix_editor import render_matrix_settings
from navigation import render_navigation, render_landing, render_history, go_to, remember_uploads, clear_resumes, choose_active_matrix
import hashlib
import re
import json

# Настройка страницы
st.set_page_config(
    page_title="AI Skill Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=str(PROJECT_ROOT / "cardano_ada_crypto_icon_264376.ico")
)

# Оформление из локальных ресурсов; без JavaScript и внешней сборки.
COLOR_SCHEME = CHART_COLORS
apply_theme()

# Инициализация состояния
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# История
if "history" not in st.session_state:
    st.session_state.history = []

# Инициализация состояния аналитики
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = {
        "total_resumes": 0,
        "average_score": 0,
        "category_scores": {},
        "skill_gaps": {},
        "timestamps": [],
        "scores_history": [],
        "skills_frequency": {},
        "experience_levels": {
            "junior": 0,
            "middle": 0,
            "senior": 0
        }
    }

# Инициализация локального графа
def init_models():
    """Compatibility initializer: the local graph, without ML models."""
    return matcher(), None



def normalize_skill_name(skill_name: str) -> str:
    """Нормализует название навыка для лучшего сопоставления."""
    # Приводим к нижнему регистру и убираем лишние пробелы
    normalized = skill_name.lower().strip()
    
    # Заменяем специальные символы и множественные пробелы
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Стандартизируем написание некоторых общих терминов
    replacements = {
        'javascript': 'js',
        'typescript': 'ts',
        'node.js': 'nodejs',
        'node js': 'nodejs',
        'react.js': 'react',
        'react js': 'react',
        'vue.js': 'vue',
        'vue js': 'vue',
        'postgresql': 'postgres',
        'microsoft sql server': 'mssql',
        'machine learning': 'ml',
        'artificial intelligence': 'ai',
        'amazon web services': 'aws',
        'google cloud platform': 'gcp',
        'microsoft azure': 'azure',
        'deep learning': 'dl',
        'natural language processing': 'nlp',
        'computer vision': 'cv'
    }
    
    for old, new in replacements.items():
        if normalized == old:
            return new
    
    return normalized

def process_resume_text(text):
    return assess_resume(text, profiles[selected_profile])


# Кэширование рекомендаций курсов

def determine_experience_level(*args):
    return "не оценивается"


def update_analytics(categories_analysis, total_percentage, extracted_skills):
    analytics = st.session_state.analytics_data
    analytics["total_resumes"] += 1
    analytics["scores_history"].append(total_percentage)
    analytics["average_score"] = sum(analytics["scores_history"]) / len(analytics["scores_history"])
    analytics["timestamps"].append(datetime.now())
    for category, data in categories_analysis.items():
        analytics["category_scores"].setdefault(category, []).append(data["percentage"])
        for skill in data["missing"]:
            analytics["skill_gaps"][skill["skill"]] = analytics["skill_gaps"].get(skill["skill"], 0) + 1
    for skill in extracted_skills:
        analytics["skills_frequency"][skill["skill"]] = analytics["skills_frequency"].get(skill["skill"], 0) + 1



def display_single_resume_analysis(results, profile_name, show_title=True, resume_id=None):
    render_single(results, profile_name, show_title, resume_id)


def display_comparative_analysis(resumes_data, profile_name):
    render_comparison(resumes_data, profile_name)


def display_welcome_screen():
    """Те же возможности и метрики в карточках дизайн-системы."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="section-index">01 / РЕЗУЛЬТАТЫ</div>
                <h3>Метрики</h3>
                <ul>
                    <li>Общий процент соответствия</li>
                    <li>Уровень кандидата</li>
                    <li>Распределение навыков</li>
                    <li>Профиль компетенций</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="section-index">02 / ИНСТРУМЕНТЫ</div>
                <h3>Возможности</h3>
                <ul>
                    <li>Анализ соответствия профилю</li>
                    <li>Оценка уровня навыков</li>
                    <li>Рекомендации по развитию</li>
                    <li>Сравнение кандидатов</li>
                    <li>Аналитика и метрики</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


def display_loader(message: str):
    """Отображает индикатор загрузки."""
    with st.container():
        st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; padding: 2rem;">
                <div style="color: #000000; font-weight: 500;">{message}</div>
            </div>
        """, unsafe_allow_html=True)

def display_progress(current: int, total: int, message: str):
    """Отображает прогресс обработки с процентами."""
    progress = current / total
    with st.container():
        st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div style="color: #000000; font-weight: 500; margin-bottom: 0.5rem;">{message}</div>
                <div style="color: #000000; font-weight: 500;">{int(progress * 100)}%</div>
            </div>
        """, unsafe_allow_html=True)
        st.progress(progress)


# Screens share the same Streamlit session, but only analysis runs the engine.
page = render_navigation()
if page == "home":
    render_landing()
elif page == "settings":
    render_matrix_settings()
else:
    st.markdown('<div class="hero-kicker">РАБОЧЕЕ ПРОСТРАНСТВО</div>', unsafe_allow_html=True)
    st.title("Анализ резюме")
    st.markdown('<p class="hero-description">Выберите требования вакансии и загрузите одно или несколько резюме. Результаты останутся здесь при переходе в другие разделы.</p>', unsafe_allow_html=True)
    try:
        matrix_records = load_matrices(default_profiles)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f"Не удалось прочитать локальные матрицы: {exc}")
        st.stop()
    profiles = {record["name"]: record["profile"] for record in matrix_records.values()}
    if st.session_state.get("active_matrix_id") not in matrix_records:
        previous = st.session_state.get("matrix_profile_choice")
        st.session_state.active_matrix_id = next(
            (key for key, record in matrix_records.items() if record["name"] == previous),
            next(iter(matrix_records)))
    st.session_state.analysis_matrix_picker = st.session_state.active_matrix_id
    _, selection, settings_link, _ = st.columns([.15, 3.2, 1.2, .15])
    with selection:
        selected_id = st.selectbox(
            "Активная матрица компетенций", list(matrix_records),
            format_func=lambda key: matrix_records[key]["name"],
            key="analysis_matrix_picker", on_change=choose_active_matrix,
        )
    with settings_link:
        st.markdown('<span class="matrix-actions-anchor" aria-hidden="true"></span>', unsafe_allow_html=True)
        st.button("Управление матрицами", on_click=go_to, args=("settings",), use_container_width=True)
    selected_record = matrix_records[selected_id]
    selected_profile = selected_record["name"]
    if selected_record["profile"].get("description"):
        st.caption(selected_record["profile"]["description"])

    upload_key = f"analysis_upload_{st.session_state.get('upload_generation', 0)}"
    uploaded_files = st.file_uploader(
        "Загрузите одно или несколько резюме", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, key=upload_key,
        on_change=remember_uploads, args=(upload_key,),
    )
    files = st.session_state.get("workspace_files", [])
    if files:
        st.caption("В работе: " + ", ".join(file["name"] for file in files))
        if not uploaded_files:
            st.caption("Ранее загруженные резюме сохранены. Новая загрузка заменит текущий набор файлов.")
        st.button("Очистить резюме", on_click=clear_resumes)
        # Cache by matrix revision, graph revision and file content.
        signature = hashlib.sha256()
        signature.update(ENGINE_REVISION.encode())
        signature.update(json.dumps(
            {"matrix": selected_id, "revision": selected_record["revision"]},
            sort_keys=True).encode("utf-8"))
        for file in files:
            signature.update(json.dumps(file["name"]).encode("utf-8"))
            signature.update(hashlib.sha256(file["content"]).digest())
        cache_key = signature.hexdigest()
        cache = st.session_state.setdefault("workspace_results", {})
        if cache_key not in cache:
            processed_resumes, errors = [], []
            processing_slot = st.empty()
            try:
                with processing_slot.container():
                    with st.status("Обработка резюме…", expanded=True) as processing:
                        st.markdown('<div class="processing-banner" role="status"><span class="processing-orbit" aria-hidden="true"></span><div><strong>Готовим аналитику</strong><p>Читаем резюме и сопоставляем навыки с выбранной матрицей.</p></div></div>', unsafe_allow_html=True)
                        stage = st.empty()
                        progress = st.progress(0, text="Подготовка графа")
                        stage.write("Подготавливаем граф понятий…")
                        ner_model, embedding_model = init_models()
                        for file_index, file in enumerate(files, 1):
                            stage.write(f"Читаем резюме {file_index} из {len(files)}: {file['name']}")
                            source = BytesIO(file["content"])
                            source.name = file["name"]
                            text = parse_resume(source)
                            if text.startswith("["):
                                errors.append(f"Ошибка при обработке {file['name']}: {text}")
                                progress.progress(file_index / len(files), text=f"Проверено файлов: {file_index} из {len(files)}")
                                continue
                            stage.write(f"Сопоставляем навыки с матрицей: {file['name']}")
                            results = process_resume_text(text)
                            if results:
                                processed_resumes.append({"file_name": file["name"], "results": results})
                                st.session_state.history.append({
                                    "file": file["name"], "profile": selected_profile,
                                    "score": results["total_percentage"],
                                })
                                update_analytics(results["categories_analysis"], results["total_percentage"], results["extracted_skills"])
                            progress.progress(file_index / len(files), text=f"Проверено файлов: {file_index} из {len(files)}")
                        stage.write("Обработка завершена. Результаты доступны ниже.")
                        processing.update(label=f"Обработано резюме: {len(processed_resumes)} из {len(files)}" if not errors else f"Обработка завершена: успешно {len(processed_resumes)}, с ошибками {len(errors)}", state="complete" if not errors else "error", expanded=bool(errors))
            finally:
                processing_slot.empty()
            cache[cache_key] = {"resumes": processed_resumes, "errors": errors}
        entry = cache[cache_key]
        for error in entry["errors"]:
            st.error(error)
        processed_resumes = entry["resumes"]
        if len(processed_resumes) == 1:
            display_single_resume_analysis(processed_resumes[0]["results"], selected_profile)
        elif len(processed_resumes) > 1:
            display_comparative_analysis(processed_resumes, selected_profile)
    else:
        st.markdown('''<div class="analysis-empty"><span class="section-index">ГОТОВО К ЗАГРУЗКЕ</span>
            <h3>Здесь появится аналитика кандидатов</h3>
            <p>Добавьте резюме в формате PDF, DOCX или TXT. Для каждого кандидата будут доступны метрики, навыки и рекомендации.</p></div>''', unsafe_allow_html=True)
    render_history()
