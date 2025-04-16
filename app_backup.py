import streamlit as st
import pandas as pd
from io import BytesIO
from resume_parser import parse_resume
from extract_skills import extract_skills
from compare_skills import compare_skills
from recommend_courses import recommend_courses
import plotly.graph_objects as go
import time
import requests
from datetime import datetime, timedelta
from typing import Dict
import logging
import plotly.express as px
import numpy as np

# Настройка страницы
st.set_page_config(
    page_title="AI Skill Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Определяем цветовую схему
COLOR_SCHEME = ['#1f77b4', '#2c5282', '#2b6cb0', '#3182ce']

# Стили для компонентов с поддержкой темной темы
st.markdown("""
    <style>
    /* Общие стили */
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
    .stDataFrame {
        font-size: 13px;
    }
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stPlotlyChart {
        padding: 0 !important;
    }
    
    /* Адаптивные стили для светлой/темной темы */
    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .metric-card {
        background-color: var(--card-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .metric-label {
        color: var(--text-color);
        font-size: 0.875rem;
    }
    
    h1, h2, h3 {
        color: var(--header-color);
        margin-bottom: 0.5rem !important;
    }
    
    .feature-card {
        background-color: var(--card-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }
    
    .feature-title {
        color: var(--header-color);
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: var(--text-color);
        font-size: 0.9rem;
    }
    
    /* Определение переменных для светлой темы */
    [data-testid="stAppViewContainer"].light {
        --background-color: white;
        --text-color: #4a5568;
        --header-color: #2c5282;
        --primary-color: #3182ce;
        --border-color: #e2e8f0;
        --card-background-color: white;
    }
    
    /* Определение переменных для темной темы */
    [data-testid="stAppViewContainer"].dark {
        --background-color: #1a1a1a;
        --text-color: #e2e8f0;
        --header-color: #63b3ed;
        --primary-color: #4299e1;
        --border-color: #2d3748;
        --card-background-color: #2d3748;
    }
    
    /* Стили для таблиц */
    .skills-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    
    .skills-table th {
        background-color: var(--card-background-color);
        padding: 0.5rem;
        text-align: left;
        color: var(--text-color);
        border-bottom: 2px solid var(--border-color);
    }
    
    .skills-table td {
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-color);
    }
    
    /* Стили для графиков */
    .js-plotly-plot {
        background-color: var(--background-color) !important;
    }
    
    .js-plotly-plot .plotly .main-svg {
        background-color: var(--background-color) !important;
    }

    /* Стили для информационных блоков */
    .stAlert {
        background-color: var(--card-background-color) !important;
        border-color: var(--border-color) !important;
    }

    .stAlert .markdown-text-container {
        color: var(--text-color) !important;
    }

    /* Стили для кнопок */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
    }

    .stButton > button:hover {
        background-color: var(--header-color) !important;
    }

    /* Стили для селектов и инпутов */
    .stSelectbox, .stTextInput {
        background-color: var(--card-background-color) !important;
        color: var(--text-color) !important;
    }

    .stSelectbox > div > div > div {
        color: var(--text-color) !important;
    }

    /* Стили для вкладок */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--card-background-color) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-color) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary-color) !important;
    }
    </style>
    
    <script>
        // Определяем текущую тему
        const theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        document.querySelector('[data-testid="stAppViewContainer"]').classList.add(theme);
    </script>
""", unsafe_allow_html=True)

# Инициализация состояния
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# История
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🕘 История анализа")
    if st.button("🧹 Очистить"):
        st.session_state.history = []
    for item in st.session_state.history[::-1]:
        st.write(f"**{item['file']}** — {item['profile']} — {item['score']}%")

# Кэширование инициализации моделей
@st.cache_resource
def init_models():
    """Инициализация и кэширование моделей."""
    from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    
    # Загружаем модели только один раз
    ner_model = pipeline("ner", model="dslim/bert-base-NER")
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    return ner_model, embedding_model

# Инициализируем модели при запуске
ner_model, embedding_model = init_models()

# Кэширование обработки резюме
@st.cache_data
def process_resume_text(text: str):
    """Кэширование результатов обработки текста резюме."""
    try:
        extracted_skills = extract_skills(text)
        if not extracted_skills:
            return {
                "total_percentage": 0,
                "categories_analysis": {},
                "extracted_skills": []
            }
        
        # Анализируем навыки по категориям
        categories_analysis = {}
        total_score = 0
        total_weight = 0
        
        # Создаем словарь для быстрого поиска навыков
        skill_dict = {skill["skill"].lower(): skill for skill in extracted_skills}
        
        for category_name, category_skills in profiles["AI Specialist"]["categories"].items():
            category_score = 0
            category_matches = []
            category_partials = []
            category_missing = []
            
            for skill_name, skill_data in category_skills.items():
                required_level = skill_data["level"]
                skill_found = False
                
                # Проверяем наличие навыка или его ключевых слов
                for keyword in [skill_name.lower()] + [k.lower() for k in skill_data["keywords"]]:
                    if keyword in skill_dict:
                        skill_found = True
                        extracted_skill = skill_dict[keyword]
                        candidate_level = 1 if extracted_skill["level"] == "beginner" else 2 if extracted_skill["level"] == "intermediate" else 3
                        
                        if candidate_level >= required_level:
                            category_matches.append({
                                "skill": skill_name,
                                "level": extracted_skill["level"],
                                "description": skill_data["description"]
                            })
                            category_score += 1
                        else:
                            category_partials.append({
                                "skill": skill_name,
                                "current_level": extracted_skill["level"],
                                "required_level": required_level,
                                "description": skill_data["description"]
                            })
                            category_score += 0.5
                        break
                
                if not skill_found:
                    category_missing.append({
                        "skill": skill_name,
                        "required_level": required_level,
                        "description": skill_data["description"]
                    })
            
            # Рассчитываем вес категории и общий скор
            category_weight = len(category_skills)
            total_weight += category_weight
            total_score += category_score
            
            categories_analysis[category_name] = {
                "score": category_score,
                "weight": category_weight,
                "percentage": round(category_score * 100 / category_weight, 2) if category_weight > 0 else 0,
                "matches": category_matches,
                "partials": category_partials,
                "missing": category_missing
            }
        
        # Общий процент соответствия
        total_percentage = round(total_score * 100 / total_weight, 2) if total_weight > 0 else 0
        
        return {
            "total_percentage": total_percentage,
            "categories_analysis": categories_analysis,
            "extracted_skills": extracted_skills
        }
    except Exception as e:
        logging.error(f"Ошибка при обработке резюме: {str(e)}")
        return {
            "total_percentage": 0,
            "categories_analysis": {},
            "extracted_skills": []
        }

# Кэширование рекомендаций курсов
@st.cache_data
def get_course_recommendations(skill: str, level: str):
    """Кэширование рекомендаций курсов."""
    return recommend_courses([skill], level)

# Профили на основе матрицы компетенций
profiles = {
    "AI Specialist": {
        "description": "Специалист по искусственному интеллекту, который разрабатывает и внедряет решения на основе ИИ",
        "categories": {
            "Фундаментальные знания": {
                "Mathematics": {
                    "level": 3,
                    "description": "Продвинутая математика, включая линейную алгебру, математический анализ, теорию вероятностей",
                    "keywords": ["linear algebra", "calculus", "probability theory", "statistics", "mathematical analysis"]
                },
                "Computer Science": {
                    "level": 2,
                    "description": "Алгоритмы, структуры данных, архитектура компьютеров",
                    "keywords": ["algorithms", "data structures", "computer architecture", "complexity theory"]
                }
            },
            "Программирование": {
                "Python": {
                    "level": 3,
                    "description": "Продвинутое владение Python, включая ООП, функциональное программирование",
                    "keywords": ["python", "object oriented programming", "functional programming", "python libraries"]
                },
                "Software Engineering": {
                    "level": 2,
                    "description": "Принципы разработки ПО, паттерны проектирования, CI/CD",
                    "keywords": ["software development", "design patterns", "git", "ci/cd", "testing"]
                }
            },
            "Машинное обучение": {
                "Machine Learning": {
                    "level": 3,
                    "description": "Глубокое понимание ML алгоритмов и их применения",
                    "keywords": ["machine learning", "supervised learning", "unsupervised learning", "reinforcement learning"]
                },
                "Deep Learning": {
                    "level": 3,
                    "description": "Нейронные сети, архитектуры глубокого обучения",
                    "keywords": ["deep learning", "neural networks", "cnn", "rnn", "transformers"]
                },
                "MLOps": {
                    "level": 2,
                    "description": "Развертывание и поддержка ML-моделей",
                    "keywords": ["mlops", "model deployment", "model monitoring", "ml pipeline"]
                }
            },
            "Специализированные области": {
                "Computer Vision": {
                    "level": 2,
                    "description": "Обработка и анализ изображений, видео",
                    "keywords": ["computer vision", "image processing", "object detection", "image segmentation"]
                },
                "NLP": {
                    "level": 2,
                    "description": "Обработка естественного языка",
                    "keywords": ["nlp", "natural language processing", "text analysis", "language models"]
                },
                "Reinforcement Learning": {
                    "level": 1,
                    "description": "Базовое понимание RL",
                    "keywords": ["reinforcement learning", "rl", "q-learning", "policy gradient"]
                }
            },
            "Инфраструктура": {
                "Big Data": {
                    "level": 2,
                    "description": "Работа с большими данными",
                    "keywords": ["big data", "hadoop", "spark", "distributed computing"]
                },
                "Cloud Platforms": {
                    "level": 2,
                    "description": "Облачные платформы и сервисы",
                    "keywords": ["aws", "azure", "gcp", "cloud computing"]
                },
                "Docker": {
                    "level": 2,
                    "description": "Контейнеризация и оркестрация",
                    "keywords": ["docker", "kubernetes", "containers", "orchestration"]
                }
            }
        }
    }
}

# Добавляем метаданные для анализа
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

# Выбор профиля и редактор матрицы компетенций
col1, col2 = st.columns([3, 1])
with col1:
    selected_profile = st.selectbox("📋 Выберите профиль для анализа", list(profiles.keys()))
    profile_data = profiles[selected_profile]

with col2:
    if st.button("✏️ Редактировать матрицу компетенций"):
        st.session_state.show_matrix_editor = True

# Редактор матрицы компетенций
if "show_matrix_editor" not in st.session_state:
    st.session_state.show_matrix_editor = False

if st.session_state.show_matrix_editor:
    st.markdown("### ✏️ Редактор матрицы компетенций")
    
    # Создаем копию текущего профиля для редактирования
    if "edited_profile" not in st.session_state:
        st.session_state.edited_profile = profiles[selected_profile].copy()
        
        # Преобразуем структуру профиля в табличный формат
        matrix_data = []
        for category, cat_data in st.session_state.edited_profile.get("categories", {}).items():
            for skill, skill_data in cat_data.items():
                matrix_data.append({
                    "Категория": category,
                    "Навык": skill,
                    "Уровень": skill_data["level"],
                    "Описание": skill_data.get("description", ""),
                    "Ключевые слова": ", ".join(skill_data.get("keywords", []))
                })
        st.session_state.matrix_data = matrix_data if matrix_data else [{"Категория": "", "Навык": "", "Уровень": 1, "Описание": "", "Ключевые слова": ""}]
    
    # Основная информация о профиле
    profile_description = st.text_area(
        "Описание профиля",
        value=st.session_state.edited_profile.get("description", ""),
        height=100
    )
    
    st.markdown("#### Матрица компетенций")
    
    # Кнопка добавления новой строки
    if st.button("➕ Добавить строку"):
        st.session_state.matrix_data.append({
            "Категория": "",
            "Навык": "",
            "Уровень": 1,
            "Описание": "",
            "Ключевые слова": ""
        })
    
    # Создаем редактируемую таблицу
    edited_df = pd.DataFrame(st.session_state.matrix_data)
    
    # Преобразуем числовые уровни в текстовые перед отображением
    edited_df["Уровень"] = edited_df["Уровень"].map({1: "Базовый", 2: "Средний", 3: "Продвинутый"})
    
    # Редактируемая таблица
    edited_data = st.data_editor(
        edited_df,
        column_config={
            "Категория": st.column_config.TextColumn(
                "Категория",
                help="Название категории навыков",
                required=True
            ),
            "Навык": st.column_config.TextColumn(
                "Навык",
                help="Название навыка",
                required=True
            ),
            "Уровень": st.column_config.SelectboxColumn(
                "Уровень",
                help="Требуемый уровень владения",
                options=["Базовый", "Средний", "Продвинутый"],
                required=True
            ),
            "Описание": st.column_config.TextColumn(
                "Описание",
                help="Описание навыка",
                width="large"
            ),
            "Ключевые слова": st.column_config.TextColumn(
                "Ключевые слова",
                help="Ключевые слова через запятую",
                width="large"
            )
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    
    # Кнопки управления
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Сохранить изменения"):
            # Преобразуем текстовые уровни обратно в числовые
            level_map = {"Базовый": 1, "Средний": 2, "Продвинутый": 3}
            
            # Преобразуем таблицу обратно в структуру профиля
            new_categories = {}
            for _, row in edited_data.iterrows():
                if row["Категория"] and row["Навык"]:  # Проверяем, что обязательные поля заполнены
                    category = row["Категория"]
                    if category not in new_categories:
                        new_categories[category] = {}
                    
                    new_categories[category][row["Навык"]] = {
                        "level": level_map[row["Уровень"]],
                        "description": row["Описание"],
                        "keywords": [k.strip() for k in row["Ключевые слова"].split(",") if k.strip()]
                    }
            
            # Обновляем профиль
            profiles[selected_profile] = {
                "description": profile_description,
                "categories": new_categories
            }
            st.session_state.show_matrix_editor = False
            del st.session_state.matrix_data
            del st.session_state.edited_profile
            st.success("✅ Матрица компетенций обновлена")
            st.rerun()
    
    with col2:
        if st.button("❌ Отменить"):
            st.session_state.show_matrix_editor = False
            if "matrix_data" in st.session_state:
                del st.session_state.matrix_data
            if "edited_profile" in st.session_state:
                del st.session_state.edited_profile
            st.rerun()

st.title("🧠 AI Skill Matcher — анализ соответствия резюме профилю")

uploaded_files = st.file_uploader(
    "📎 Загрузите одно или несколько резюме",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

def determine_experience_level(total_percentage: float, categories_analysis: dict) -> str:
    """Определяет уровень опыта на основе анализа навыков."""
    if total_percentage >= 80:
        return "senior"
    elif total_percentage >= 60:
        return "middle"
    else:
        return "junior"

def update_analytics(categories_analysis: dict, total_percentage: float, extracted_skills: list):
    """Обновляет аналитические данные."""
    analytics = st.session_state.analytics_data
    
    # Обновляем счетчик резюме
    if total_percentage > 0:  # Учитываем только успешно обработанные резюме
        analytics["total_resumes"] += 1
        analytics["timestamps"].append(datetime.now())
        analytics["scores_history"].append(total_percentage)
        
        # Обновляем средний скор
        analytics["average_score"] = sum(analytics["scores_history"]) / len(analytics["scores_history"])
        
        # Обновляем статистику по категориям
        for category, data in categories_analysis.items():
            if category not in analytics["category_scores"]:
                analytics["category_scores"][category] = []
            analytics["category_scores"][category].append(data["percentage"])
        
        # Обновляем пробелы в навыках
        for category, data in categories_analysis.items():
            for skill in data["missing"]:
                skill_name = skill["skill"]
                if skill_name not in analytics["skill_gaps"]:
                    analytics["skill_gaps"][skill_name] = 0
                analytics["skill_gaps"][skill_name] += 1
        
        # Обновляем частоту навыков
        for skill in extracted_skills:
            skill_name = skill["skill"]
            if skill_name not in analytics["skills_frequency"]:
                analytics["skills_frequency"][skill_name] = 0
            analytics["skills_frequency"][skill_name] += 1
        
        # Обновляем уровни опыта
        experience_level = determine_experience_level(total_percentage, categories_analysis)
        analytics["experience_levels"][experience_level] += 1

def display_single_resume_analysis(results: dict, profile_name: str, show_title: bool = True, resume_id: str = None):
    """
    Отображает детальный анализ одного резюме.
    
    Args:
        results: Результаты анализа резюме
        profile_name: Название профиля
        show_title: Показывать ли заголовок
        resume_id: Уникальный идентификатор резюме
    """
    total_percentage = results['total_percentage']
    categories_analysis = results['categories_analysis']
    
    if show_title:
        st.write("### Результаты анализа")
    
    # Основные метрики в две строки
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    # Первая строка метрик
    with row1_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_percentage}%</div>
                <div class="metric-label">Общее соответствие</div>
            </div>
        """, unsafe_allow_html=True)
    
    with row1_col2:
        level = determine_experience_level(total_percentage, categories_analysis)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{level.capitalize()}</div>
                <div class="metric-label">Уровень</div>
            </div>
        """, unsafe_allow_html=True)
    
    with row1_col3:
        categories_count = len([c for c in categories_analysis.values() if c['percentage'] > 0])
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{categories_count}</div>
                <div class="metric-label">Категорий навыков</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Вторая строка метрик
    with row2_col1:
        confirmed_skills = sum(len(data['matches']) for data in categories_analysis.values())
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confirmed_skills}</div>
                <div class="metric-label">Подтвержденных навыков</div>
            </div>
        """, unsafe_allow_html=True)
    
    with row2_col2:
        missing_skills = sum(len(data['missing']) for data in categories_analysis.values())
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{missing_skills}</div>
                <div class="metric-label">Отсутствующих навыков</div>
            </div>
        """, unsafe_allow_html=True)
    
    with row2_col3:
        partial_skills = sum(len(data['partials']) for data in categories_analysis.values())
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{partial_skills}</div>
                <div class="metric-label">Навыков для развития</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Вкладки для разных типов анализа
    analysis_tabs = st.tabs(["📊 Визуализация", "🎯 Навыки", "📈 Тренды", "💡 Рекомендации"])
    
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Круговая диаграмма
            categories = []
            scores = []
            for category, data in categories_analysis.items():
                if data['percentage'] > 0:
                    categories.append(category)
                    scores.append(data['percentage'])
            
            if categories:
                fig = go.Figure(data=[go.Pie(
                    labels=categories,
                    values=scores,
                    hole=.3,
                    marker=dict(colors=COLOR_SCHEME)
                )])
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300,
                    title="Распределение навыков"
                )
                st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{resume_id}")
        
        with col2:
            # Радарная диаграмма
            if categories:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    line=dict(color=COLOR_SCHEME[0])
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300,
                    title="Профиль компетенций"
                )
                st.plotly_chart(fig, use_container_width=True, key=f"radar_chart_{resume_id}")
        
        # Добавляем гистограмму распределения навыков
        skills_distribution = []
        for category, data in categories_analysis.items():
            skills_distribution.extend([
                {"category": category, "status": "Подтверждены", "count": len(data['matches'])},
                {"category": category, "status": "Требуют развития", "count": len(data['partials'])},
                {"category": category, "status": "Отсутствуют", "count": len(data['missing'])}
            ])
        
        df_distribution = pd.DataFrame(skills_distribution)
        if not df_distribution.empty:
            fig = px.bar(
                df_distribution,
                x="category",
                y="count",
                color="status",
                title="Распределение статусов навыков по категориям",
                color_discrete_sequence=COLOR_SCHEME,
                barmode="group"
            )
            fig.update_layout(
                xaxis_title="Категория",
                yaxis_title="Количество навыков",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"distribution_chart_{resume_id}")
    
    with analysis_tabs[1]:
        skill_tabs = st.tabs(["✓ Подтверждены", "⚠ Требуют развития", "× Отсутствуют"])
        
        for tab_idx, (tab, (status, icon)) in enumerate(zip(skill_tabs, [("Подтверждены", "✓"), ("Требуют развития", "⚠"), ("Отсутствуют", "×")])):
            with tab:
                skills_data = []
                for category, analysis in categories_analysis.items():
                    if status == "Подтверждены":
                        skills = analysis['matches']
                    elif status == "Требуют развития":
                        skills = analysis['partials']
                    else:
                        skills = analysis['missing']
                    
                    for skill in skills:
                        skills_data.append({
                            "Категория": category,
                            "Навык": skill['skill'],
                            "Уровень": skill.get('level', 'N/A'),
                            "Описание": skill['description']
                        })
                
                if skills_data:
                    df = pd.DataFrame(skills_data)
                    
                    # Добавляем фильтры с действительно уникальными ключами
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_key = f"categories_filter_{status}_{resume_id}_{tab_idx}"
                        selected_categories = st.multiselect(
                            "Фильтр по категориям",
                            options=sorted(df["Категория"].unique()),
                            default=sorted(df["Категория"].unique()),
                            key=filter_key
                        )
                    
                    filtered_df = df[df["Категория"].isin(selected_categories)]
                    st.table(filtered_df)
                else:
                    st.info(f"Нет навыков в категории '{status}'")
    
    with analysis_tabs[2]:
        st.write("#### Тренды и статистика")
        
        # Создаем DataFrame для анализа трендов
        trends_data = []
        for category, data in categories_analysis.items():
            total_skills = len(data['matches']) + len(data['partials']) + len(data['missing'])
            if total_skills > 0:
                confirmed_percent = len(data['matches']) * 100 / total_skills
                partial_percent = len(data['partials']) * 100 / total_skills
                missing_percent = len(data['missing']) * 100 / total_skills
                
                trends_data.append({
                    "Категория": category,
                    "Процент подтвержденных": confirmed_percent,
                    "Процент требующих развития": partial_percent,
                    "Процент отсутствующих": missing_percent,
                    "Всего навыков": total_skills
                })
        
        if trends_data:
            df_trends = pd.DataFrame(trends_data)
            
            # График трендов
            fig = go.Figure()
            for col in ["Процент подтвержденных", "Процент требующих развития", "Процент отсутствующих"]:
                fig.add_trace(go.Bar(
                    name=col,
                    x=df_trends["Категория"],
                    y=df_trends[col],
                    text=df_trends[col].round(1).astype(str) + "%",
                    textposition="auto",
                ))
            
            fig.update_layout(
                barmode='stack',
                title="Распределение статусов навыков по категориям (%)",
                xaxis_title="Категория",
                yaxis_title="Процент",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Таблица со статистикой
            st.write("#### Детальная статистика по категориям")
            st.table(df_trends.round(1))
    
    with analysis_tabs[3]:
        st.write("#### Рекомендации по развитию")
        
        # Анализируем сильные и слабые стороны
        strengths = []
        improvements = []
        
        for category, data in categories_analysis.items():
            category_percentage = data['percentage']
            
            if category_percentage >= 70:
                strengths.append({
                    "category": category,
                    "percentage": category_percentage,
                    "confirmed_skills": len(data['matches'])
                })
            elif category_percentage < 50:
                improvements.append({
                    "category": category,
                    "percentage": category_percentage,
                    "missing_skills": len(data['missing']),
                    "partial_skills": len(data['partials'])
                })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### 💪 Сильные стороны")
            if strengths:
                for strength in sorted(strengths, key=lambda x: x['percentage'], reverse=True):
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.1rem; font-weight: bold; color: #2c5282;">
                                {strength['category']}
                            </div>
                            <div style="color: #48bb78; font-size: 1.5rem; margin: 0.5rem 0;">
                                {strength['percentage']}%
                            </div>
                            <div class="metric-label">
                                Подтверждено навыков: {strength['confirmed_skills']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Нет категорий с высоким уровнем соответствия")
        
        with col2:
            st.write("##### 📈 Приоритеты развития")
            if improvements:
                for improvement in sorted(improvements, key=lambda x: x['percentage']):
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.1rem; font-weight: bold; color: #2c5282;">
                                {improvement['category']}
                            </div>
                            <div style="color: #e53e3e; font-size: 1.5rem; margin: 0.5rem 0;">
                                {improvement['percentage']}%
                            </div>
                            <div class="metric-label">
                                Требуется развить: {improvement['partial_skills']} навыков<br>
                                Отсутствует: {improvement['missing_skills']} навыков
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Нет категорий, требующих срочного развития")

def display_comparative_analysis(resumes_data: list, profile_name: str):
    """Отображает расширенный сравнительный анализ нескольких резюме."""
    st.write("### Сравнительный анализ")
    
    # Создаем DataFrame для сравнения
    comparison_data = []
    for resume in resumes_data:
        if "results" in resume:
            categories_analysis = resume["results"]["categories_analysis"]
            confirmed_skills = sum(len(data['matches']) for data in categories_analysis.values())
            missing_skills = sum(len(data['missing']) for data in categories_analysis.values())
            partial_skills = sum(len(data['partials']) for data in categories_analysis.values())
            
            comparison_data.append({
                "Кандидат": resume["file_name"],
                "Общий балл": resume["results"]["total_percentage"],
                "Уровень": determine_experience_level(
                    resume["results"]["total_percentage"], 
                    categories_analysis
                ).capitalize(),
                "Подтверждено навыков": confirmed_skills,
                "Требуют развития": partial_skills,
                "Отсутствуют": missing_skills
            })
    
    if comparison_data:
        # Сортируем по общему баллу
        comparison_data.sort(key=lambda x: x["Общий балл"], reverse=True)
        df = pd.DataFrame(comparison_data)
        
        # Вкладки для разных видов сравнения
        compare_tabs = st.tabs(["📊 Общий обзор", "📈 Детальное сравнение", "👤 Профили кандидатов"])
        
        with compare_tabs[0]:
            # Отображаем сводную таблицу
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.table(df)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # График сравнения общих баллов
            fig = go.Figure(data=[
                go.Bar(
                    x=[d["Кандидат"] for d in comparison_data],
                    y=[d["Общий балл"] for d in comparison_data],
                    marker_color=COLOR_SCHEME[0],
                    text=[f"{d['Общий балл']}%" for d in comparison_data],
                    textposition="auto",
                )
            ])
            fig.update_layout(
                title="Сравнение кандидатов по общему баллу",
                xaxis_title="",
                yaxis_title="Общий балл, %",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True, key="overall_comparison_chart")
        
        with compare_tabs[1]:
            # График сравнения по категориям
            categories_data = []
            for resume in resumes_data:
                if "results" in resume:
                    for category, analysis in resume["results"]["categories_analysis"].items():
                        categories_data.append({
                            "Кандидат": resume["file_name"],
                            "Категория": category,
                            "Процент": analysis["percentage"]
                        })
            
            if categories_data:
                df_categories = pd.DataFrame(categories_data)
                fig = px.bar(
                    df_categories,
                    x="Кандидат",
                    y="Процент",
                    color="Категория",
                    title="Сравнение по категориям навыков",
                    color_discrete_sequence=COLOR_SCHEME,
                    barmode="group"
                )
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Процент соответствия",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key="categories_comparison_chart")
            
            # Тепловая карта навыков
            skills_analysis_data = []
            for resume in resumes_data:
                if "results" in resume:
                    for category, analysis in resume["results"]["categories_analysis"].items():
                        confirmed = len(analysis['matches'])
                        partial = len(analysis['partials'])
                        missing = len(analysis['missing'])
                        total = confirmed + partial + missing
                        if total > 0:
                            skills_analysis_data.append({
                                "Кандидат": resume["file_name"],
                                "Категория": category,
                                "Подтверждено": confirmed,
                                "Требует развития": partial,
                                "Отсутствует": missing,
                                "Всего": total,
                                "Процент подтверждения": round(confirmed * 100 / total, 1),
                                "Процент развития": round(partial * 100 / total, 1),
                                "Процент отсутствия": round(missing * 100 / total, 1)
                            })
            
            if skills_analysis_data:
                df_analysis = pd.DataFrame(skills_analysis_data)
                
                # Создаем вкладки для разных видов визуализации
                viz_tabs = st.tabs(["📊 Распределение навыков", "📈 Сравнение категорий", "📋 Детальный анализ"])
                
                with viz_tabs[0]:
                    st.write("#### Распределение навыков по категориям")
                    
                    # Создаем столбчатую диаграмму с накоплением
                    fig_distribution = go.Figure()
                    
                    for candidate in df_analysis["Кандидат"].unique():
                        df_candidate = df_analysis[df_analysis["Кандидат"] == candidate]
                        
                        # Подтвержденные навыки
                        fig_distribution.add_trace(go.Bar(
                            name=f"{candidate} (Подтверждено)",
                            x=df_candidate["Категория"],
                            y=df_candidate["Подтверждено"],
                            marker_color="#63b3ed",
                            text=df_candidate["Подтверждено"],
                            textposition="inside",
                            hovertemplate="<b>%{x}</b><br>" +
                                        "Подтверждено: %{y}<br>" +
                                        f"Кандидат: {candidate}<extra></extra>"
                        ))
                        
                        # Навыки для развития
                        fig_distribution.add_trace(go.Bar(
                            name=f"{candidate} (Развитие)",
                            x=df_candidate["Категория"],
                            y=df_candidate["Требует развития"],
                            marker_color="#90cdf4",
                            text=df_candidate["Требует развития"],
                            textposition="inside",
                            hovertemplate="<b>%{x}</b><br>" +
                                        "Требует развития: %{y}<br>" +
                                        f"Кандидат: {candidate}<extra></extra>"
                        ))
                        
                        # Отсутствующие навыки
                        fig_distribution.add_trace(go.Bar(
                            name=f"{candidate} (Отсутствует)",
                            x=df_candidate["Категория"],
                            y=df_candidate["Отсутствует"],
                            marker_color="#2c5282",
                            text=df_candidate["Отсутствует"],
                            textposition="inside",
                            hovertemplate="<b>%{x}</b><br>" +
                                        "Отсутствует: %{y}<br>" +
                                        f"Кандидат: {candidate}<extra></extra>"
                        ))
                    
                    fig_distribution.update_layout(
                        barmode='group',
                        title="Распределение навыков по категориям",
                        xaxis_title="Категории",
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
                
                with viz_tabs[1]:
                    st.write("#### Сравнение категорий между кандидатами")
                    
                    # Создаем радарную диаграмму
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

                    # Добавляем тепловую карту
                    st.write("#### Тепловая карта соответствия навыков")
                    
                    # Создаем матрицу для тепловой карты
                    heatmap_data = df_analysis.pivot(
                        index="Кандидат",
                        columns="Категория",
                        values="Процент подтверждения"
                    )
                    
                    # Создаем тепловую карту
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='YlOrRd',
                        text=heatmap_data.values.round(1),
                        texttemplate="%{text}%",
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorbar=dict(
                            title="Процент подтверждения",
                            ticksuffix="%"
                        )
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Тепловая карта соответствия навыков по категориям",
                        xaxis_title="Категории",
                        yaxis_title="Кандидаты",
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e2e8f0'}
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True, key="skills_heatmap")
                
                with viz_tabs[2]:
                    st.write("#### Детальный анализ навыков")
                    
                    # Создаем интерактивную таблицу
                    detailed_df = df_analysis.copy()
                    detailed_df["Прогресс"] = detailed_df.apply(
                        lambda row: f"{row['Подтверждено']}/{row['Всего']} ({row['Процент подтверждения']}%)",
                        axis=1
                    )
                    
                    # Фильтры
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_candidates = st.multiselect(
                            "Выберите кандидатов",
                            options=detailed_df["Кандидат"].unique(),
                            default=detailed_df["Кандидат"].unique()
                        )
                    
                    with col2:
                        selected_categories = st.multiselect(
                            "Выберите категории",
                            options=detailed_df["Категория"].unique(),
                            default=detailed_df["Категория"].unique()
                        )
                    
                    # Фильтруем данные
                    filtered_df = detailed_df[
                        (detailed_df["Кандидат"].isin(selected_candidates)) &
                        (detailed_df["Категория"].isin(selected_categories))
                    ]
                    
                    # Отображаем таблицу с форматированием
                    st.dataframe(
                        filtered_df[[
                            "Кандидат", "Категория", "Прогресс",
                            "Подтверждено", "Требует развития", "Отсутствует"
                        ]].style.background_gradient(
                            subset=["Подтверждено", "Требует развития", "Отсутствует"],
                            cmap="YlOrRd"
                        ),
                        hide_index=True,
                        height=400
                    )
                    
                    # Добавляем статистику
                    st.write("#### Сводная статистика")
                    summary_stats = pd.DataFrame([{
                        "Метрика": "Среднее количество подтвержденных навыков",
                        "Значение": f"{filtered_df['Подтверждено'].mean():.1f}"
                    }, {
                        "Метрика": "Максимальное количество подтвержденных навыков",
                        "Значение": str(filtered_df['Подтверждено'].max())
                    }, {
                        "Метрика": "Средний процент подтверждения",
                        "Значение": f"{filtered_df['Процент подтверждения'].mean():.1f}%"
                    }])
                    
                    st.dataframe(summary_stats, hide_index=True)
        
        with compare_tabs[2]:
            # Детальные профили кандидатов
            for idx, resume in enumerate(resumes_data):
                if "results" in resume:
                    st.write(f"#### {resume['file_name']}")
                    # Передаем уникальный идентификатор для каждого резюме
                    display_single_resume_analysis(
                        resume["results"],
                        profile_name,
                        show_title=False,
                        resume_id=f"{idx}_{resume['file_name']}"
                    )
                    st.markdown("---")

def display_welcome_screen():
    """Отображает улучшенный стартовый экран приложения."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("AI Skill Matcher")
        
        with st.container():
            st.subheader("🎯 Интеллектуальный анализ резюме")
            st.write("""
                Загрузите одно или несколько резюме для анализа соответствия требуемому профилю.
                Система проведет детальный анализ навыков и предоставит рекомендации по развитию.
            """)
        
        with st.container():
            st.subheader("📊 Расширенная аналитика")
            st.write("""
                • Детальный анализ навыков по категориям
                • Визуализация результатов
                • Сравнительный анализ кандидатов
                • Тренды и статистика
            """)
    
    with col2:
        with st.container():
            st.subheader("✨ Возможности")
            st.write("""
                ✓ Анализ соответствия профилю
                ✓ Оценка уровня навыков
                ✓ Рекомендации по развитию
                ✓ Сравнение кандидатов
                ✓ Аналитика и метрики
            """)
        
        with st.container():
            st.subheader("📈 Метрики")
            st.write("""
                • Общий процент соответствия
                • Уровень кандидата
                • Распределение навыков
                • Профиль компетенций
            """)

def display_loader(message: str):
    """Отображает индикатор загрузки."""
    with st.container():
        st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; padding: 2rem;">
                <div style="color: #2c5282; font-weight: 500;">{message}</div>
            </div>
        """, unsafe_allow_html=True)

def display_progress(current: int, total: int, message: str):
    """Отображает прогресс обработки с процентами."""
    progress = current / total
    with st.container():
        st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div style="color: #2c5282; font-weight: 500; margin-bottom: 0.5rem;">{message}</div>
                <div style="color: #2c5282; font-weight: 500;">{int(progress * 100)}%</div>
            </div>
        """, unsafe_allow_html=True)
        st.progress(progress)

# Обновляем основной блок обработки файлов
if uploaded_files:
    total_files = len(uploaded_files)
    processed_resumes = []
    
    for i, f in enumerate(uploaded_files, 1):
        text = parse_resume(f)

        if text.startswith("["):
            st.error(f"Ошибка при обработке {f.name}: {text}")
            continue

        results = process_resume_text(text)
        
        if results:
            processed_resumes.append({
                "file_name": f.name,
                "results": results
            })
            
            # Добавляем в историю
        st.session_state.history.append({
            "file": f.name,
            "profile": selected_profile,
                "score": results["total_percentage"]
            })
    
    if len(processed_resumes) == 1:
        display_single_resume_analysis(processed_resumes[0]["results"], selected_profile)
    elif len(processed_resumes) > 1:
        display_comparative_analysis(processed_resumes, selected_profile)
else:
    display_welcome_screen() 