"""Existing matrix editor, isolated on the Settings screen."""
import sqlite3
from copy import deepcopy
from uuid import uuid4
import pandas as pd
import streamlit as st
from profiles import profiles as default_profiles
from matrix_storage import load_matrices, save_matrix, profile_rows, profile_from_rows, editor_rows
from navigation import go_to

def open_matrix_editor(mode, matrix_id, record):
    st.session_state.matrix_draft = {
        "id": matrix_id if mode == "edit" else None,
        "revision": record["revision"] if mode == "edit" else None,
        "name": record["name"] if mode == "edit" else record["name"] + " — копия" if mode == "copy" else "",
        "profile": deepcopy(record["profile"]) if mode != "create" else {"description": "", "categories": {}},
        "token": uuid4().hex, "generation": 0,
    }
    st.session_state.show_matrix_editor = True
    st.session_state.pop("matrix_editor_error", None)


def close_matrix_editor():
    st.session_state.show_matrix_editor = False
    st.session_state.pop("matrix_draft", None)
    st.session_state.pop("matrix_data", None)
    st.session_state.pop("matrix_editor_error", None)


def current_matrix_rows():
    draft = st.session_state.matrix_draft
    key = f"matrix_table_{draft['token']}_{draft['generation']}"
    return editor_rows(st.session_state.matrix_data, st.session_state.get(key, {}))


def append_matrix_row():
    st.session_state.matrix_data = current_matrix_rows() + [
        {"Категория": "", "Навык": "", "Уровень": "Базовый", "Описание": "", "Ключевые слова": ""}]
    st.session_state.matrix_draft["generation"] += 1
    st.session_state.matrix_draft["rows_snapshot"] = deepcopy(st.session_state.matrix_data)


def save_matrix_editor():
    draft = st.session_state.matrix_draft
    name = st.session_state[f"matrix_name_{draft['token']}"]
    description = st.session_state[f"matrix_description_{draft['token']}"]
    try:
        saved_profile = profile_from_rows(current_matrix_rows(), description)
        saved_id = save_matrix(name, saved_profile, draft["id"], draft["revision"])
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.session_state.matrix_editor_error = f"Не удалось сохранить матрицу: {exc}"
    else:
        close_matrix_editor()
        st.session_state.matrix_saved_id = saved_id
        st.session_state.matrix_notice = "Матрица компетенций сохранена"



def render_matrix_settings():
    st.markdown('<div class="hero-kicker">НАСТРОЙКИ</div>', unsafe_allow_html=True)
    st.title("Матрицы компетенций")
    st.markdown('<p class="hero-description">Создавайте и редактируйте требования к вакансиям. Активную матрицу для скрининга выбирайте на странице анализа.</p>', unsafe_allow_html=True)
    st.button("К анализу резюме", on_click=go_to, args=("analysis",))
    try:
        matrix_records = load_matrices(default_profiles)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f"Не удалось прочитать локальные матрицы: {exc}")
        st.stop()


    if "show_matrix_editor" not in st.session_state:
        st.session_state.show_matrix_editor = False
    if "matrix_saved_id" in st.session_state:
        st.session_state.settings_matrix_picker = st.session_state.pop("matrix_saved_id")
    if "matrix_notice" in st.session_state:
        st.success(st.session_state.pop("matrix_notice"))

    if st.session_state.get("settings_matrix_picker") not in matrix_records:
        st.session_state.settings_matrix_picker = next(iter(matrix_records))
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_id = st.selectbox(
            "Матрица для редактирования", list(matrix_records),
            format_func=lambda key: matrix_records[key]["name"],
            key="settings_matrix_picker", disabled=st.session_state.show_matrix_editor,
        )
        st.caption("Выбор здесь не меняет активную матрицу на странице анализа.")
    with col2:
        st.button("Редактировать матрицу компетенций", on_click=open_matrix_editor,
                  args=("edit", selected_id, matrix_records[selected_id]), disabled=st.session_state.show_matrix_editor)
        st.button("Создать матрицу", on_click=open_matrix_editor,
                  args=("create", selected_id, matrix_records[selected_id]), type="primary", disabled=st.session_state.show_matrix_editor)
        st.button("Копировать матрицу", on_click=open_matrix_editor,
                  args=("copy", selected_id, matrix_records[selected_id]), disabled=st.session_state.show_matrix_editor)

    # Редактор матрицы компетенций
    if "show_matrix_editor" not in st.session_state:
        st.session_state.show_matrix_editor = False

    if st.session_state.show_matrix_editor:
        st.markdown("### Редактор матрицы компетенций")

        draft = st.session_state.matrix_draft
        grid_key = f"matrix_table_{draft['token']}_{draft['generation']}"
        if grid_key not in st.session_state and "rows_snapshot" in draft:
            st.session_state.matrix_data = deepcopy(draft["rows_snapshot"])
            draft["generation"] += 1
        if "matrix_data" not in st.session_state:
            st.session_state.matrix_data = profile_rows(draft["profile"]) or [
                {"Категория": "", "Навык": "", "Уровень": "Базовый", "Описание": "", "Ключевые слова": ""}]
        profile_name = st.text_input("Название матрицы", value=draft["name"], key=f"matrix_name_{draft['token']}")

        # Основная информация о профиле
        profile_description = st.text_area(
            "Описание профиля",
            value=draft["profile"].get("description", ""),
            height=100, key=f"matrix_description_{draft['token']}"
        )

        st.markdown("#### Матрица компетенций")

        # Кнопка добавления новой строки
        st.button("Добавить строку", on_click=append_matrix_row)

        # Создаем редактируемую таблицу
        edited_df = pd.DataFrame(st.session_state.matrix_data)

        # Редактируемая таблица
        edited_data = st.data_editor(
            edited_df,
            key=f"matrix_table_{draft['token']}_{draft['generation']}",
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
        draft["name"] = profile_name
        draft["profile"]["description"] = profile_description
        draft["rows_snapshot"] = edited_data.to_dict("records")
        st.caption("Черновик сохранится при переходе между разделами в этой сессии. Чтобы применить требования, нажмите «Сохранить изменения».")
        if "matrix_editor_error" in st.session_state:
            st.error(st.session_state.matrix_editor_error)

        # Кнопки управления
        col1, col2 = st.columns(2)
        with col1:
            st.button("Сохранить изменения", on_click=save_matrix_editor, type="primary")

        with col2:
            st.button("Отменить", on_click=close_matrix_editor)
