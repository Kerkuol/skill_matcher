"""Existing matrix editor, isolated on the Settings screen."""
import sqlite3
from copy import deepcopy
from uuid import uuid4
from html import escape
import pandas as pd
import streamlit as st
from profiles import profiles as default_profiles
from matrix_storage import load_matrices, save_matrix, profile_rows, profile_from_rows, editor_rows
from navigation import go_to


def cell_text(row, field):
    value = row.get(field)
    return value if isinstance(value, str) else ""

def open_matrix_editor(mode, matrix_id, record):
    st.session_state.matrix_draft = {
        "id": matrix_id if mode == "edit" else None,
        "revision": record["revision"] if mode == "edit" else None,
        "name": record["name"] if mode == "edit" else record["name"] + " — копия" if mode == "copy" else "",
        "profile": deepcopy(record["profile"]) if mode != "create" else {"description": "", "categories": {}},
        "token": uuid4().hex, "generation": 0,
        "edit_mode": "Карточки", "detail_index": 0, "detail_generation": 0,
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
    return [dict(row, Уровень=row.get("Уровень") or "Базовый") for row in editor_rows(st.session_state.matrix_data, st.session_state.get(key, {}))]


def capture_skill_details():
    draft = st.session_state.get("matrix_draft")
    rows = st.session_state.get("matrix_data", [])
    if not draft or draft.get("edit_mode", "Таблица") != "Карточки" or not rows:
        return
    index = min(draft.get("detail_index", 0), len(rows) - 1)
    prefix = f"skill_{draft['token']}_{draft.get('detail_generation', 0)}_{index}"
    for field in ("Категория", "Навык", "Уровень", "Описание", "Ключевые слова"):
        key = f"{prefix}_{field}"
        if key in st.session_state:
            rows[index][field] = st.session_state[key]
    draft["rows_snapshot"] = deepcopy(rows)


def sync_rows():
    capture_skill_details()
    rows = current_matrix_rows()
    st.session_state.matrix_data = rows
    draft = st.session_state.matrix_draft
    draft["rows_snapshot"] = deepcopy(rows)
    draft["generation"] += 1


def change_editor_mode():
    sync_rows()
    draft = st.session_state.matrix_draft
    draft["edit_mode"] = st.session_state[f"matrix_mode_{draft['token']}"]
    draft["detail_generation"] = draft.get("detail_generation", 0) + 1


def select_skill():
    capture_skill_details()
    draft = st.session_state.matrix_draft
    label = st.session_state[f"skill_picker_{draft['token']}"]
    draft["detail_index"] = int(label.split(" · ", 1)[0]) - 1
    draft["detail_generation"] = draft.get("detail_generation", 0) + 1


def remove_skill():
    sync_rows()
    draft = st.session_state.matrix_draft
    rows = st.session_state.matrix_data
    if rows:
        rows.pop(min(draft.get("detail_index", 0), len(rows) - 1))
    draft["detail_index"] = max(0, min(draft.get("detail_index", 0), len(rows) - 1))
    draft["detail_generation"] = draft.get("detail_generation", 0) + 1
    draft["rows_snapshot"] = deepcopy(rows)


def append_matrix_row():
    capture_skill_details()
    st.session_state.matrix_data = current_matrix_rows() + [
        {"Категория": "", "Навык": "", "Уровень": "Базовый", "Описание": "", "Ключевые слова": ""}]
    st.session_state.matrix_draft["generation"] += 1
    st.session_state.matrix_draft["rows_snapshot"] = deepcopy(st.session_state.matrix_data)
    st.session_state.matrix_draft["detail_index"] = len(st.session_state.matrix_data) - 1
    st.session_state.matrix_draft["detail_generation"] = st.session_state.matrix_draft.get("detail_generation", 0) + 1


def save_matrix_editor():
    capture_skill_details()
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




def render_bulk_table(draft):
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
            "Уровень": None,
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
    return edited_data.to_dict("records")


def matrix_preview(rows):
    filled = [row for row in rows if cell_text(row, "Навык").strip()]
    if not filled:
        st.info("Добавьте первую компетенцию, чтобы увидеть матрицу.")
        return
    cells = []
    for row in filled:
        values = [escape(value) if isinstance(value, str) else "" for value in
                  (row.get("Навык"), row.get("Категория"), row.get("Уровень"), row.get("Описание"))]
        cells.append(f'<tr><td><strong>{values[0]}</strong></td><td>{values[1]}</td><td>{values[3]}</td></tr>')
    st.markdown('<div class="matrix-glass-table"><table><caption>Компетенции и требования</caption><thead><tr><th scope="col">Компетенция</th><th scope="col">Категория</th><th scope="col">Описание</th></tr></thead><tbody>' + "".join(cells) + '</tbody></table></div>', unsafe_allow_html=True)


def render_skill_cards(draft):
    rows = st.session_state.matrix_data
    if not rows:
        st.info("Матрица пока пустая. Добавьте компетенцию, чтобы начать.")
        return
    index = min(draft.get("detail_index", 0), len(rows) - 1)
    draft["detail_index"] = index
    labels = [f"{i + 1:02d} · {cell_text(row, 'Навык') or 'Новая компетенция'}" for i, row in enumerate(rows)]
    picker = f"skill_picker_{draft['token']}"
    st.session_state[picker] = labels[index]
    st.selectbox("Выберите компетенцию", labels, key=picker, on_change=select_skill)
    row = rows[index]
    prefix = f"skill_{draft['token']}_{draft.get('detail_generation', 0)}_{index}"
    st.markdown('<div class="editor-section-label">КАРТОЧКА КОМПЕТЕНЦИИ · ИЗМЕНЕНИЯ ПОПАДАЮТ В ЧЕРНОВИК</div>', unsafe_allow_html=True)
    st.text_input("Название навыка", value=cell_text(row, "Навык"), key=f"{prefix}_Навык", on_change=capture_skill_details, placeholder="Например, Python")
    st.text_input("Категория навыка", value=cell_text(row, "Категория"), key=f"{prefix}_Категория", on_change=capture_skill_details, placeholder="Например, Программирование")
    st.text_area("Что должен уметь кандидат", value=cell_text(row, "Описание"), key=f"{prefix}_Описание", on_change=capture_skill_details, height=100)
    st.text_input("Ключевые слова и синонимы", value=cell_text(row, "Ключевые слова"), key=f"{prefix}_Ключевые слова", on_change=capture_skill_details, help="Через запятую. Например: Python, Python3, pandas")
    st.button("Удалить компетенцию из черновика", on_click=remove_skill)
    st.caption("Все изменения применятся к сохранённой матрице только после нажатия «Сохранить изменения».")


def render_matrix_settings():
    st.markdown('<div class="hero-kicker">НАСТРОЙКИ / БИБЛИОТЕКА МАТРИЦ</div>', unsafe_allow_html=True)
    try:
        records = load_matrices(default_profiles)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f"Не удалось прочитать локальные матрицы: {exc}")
        st.stop()
    st.session_state.setdefault("show_matrix_editor", False)
    if "matrix_saved_id" in st.session_state:
        st.session_state.settings_matrix_picker = st.session_state.pop("matrix_saved_id")
    if "matrix_notice" in st.session_state:
        st.success(st.session_state.pop("matrix_notice"))
    if st.session_state.get("settings_matrix_picker") not in records:
        st.session_state.settings_matrix_picker = next(iter(records))
    editing = st.session_state.show_matrix_editor
    if editing:
        st.markdown('<h1 class="editor-page-title">Редактор матрицы</h1>', unsafe_allow_html=True)
        with st.expander("Исходная матрица"):
            selected = st.selectbox("Матрица для редактирования", list(records), format_func=lambda key: records[key]["name"], key="settings_matrix_picker", disabled=True)
    else:
        st.title("Требования под каждую роль")
        st.markdown('<p class="hero-description">Выберите роль, задайте компетенции и требуемые уровни. Готовые матрицы можно копировать и адаптировать под новую вакансию.</p>', unsafe_allow_html=True)
        selected = st.selectbox("Матрица для редактирования", list(records), format_func=lambda key: records[key]["name"], key="settings_matrix_picker")
    record = records[selected]
    if not editing:
        actions = st.columns(3)
        with actions[0]:
            st.button("Создать матрицу", on_click=open_matrix_editor, args=("create", selected, record), type="primary", use_container_width=True)
        with actions[1]:
            st.button("Редактировать матрицу компетенций", on_click=open_matrix_editor, args=("edit", selected, record), use_container_width=True)
        with actions[2]:
            st.button("Копировать матрицу", on_click=open_matrix_editor, args=("copy", selected, record), use_container_width=True)
        skills = profile_rows(record["profile"])
        categories = len(record["profile"]["categories"])
        st.markdown(f'<section class="matrix-summary reveal"><div><span class="section-index">ВЫБРАННАЯ МАТРИЦА</span><h3>{escape(record["name"])}</h3><p>{escape(record["profile"].get("description", ""))}</p></div><div class="matrix-counts"><span><b>{len(skills)}</b>компетенций</span><span><b>{categories}</b>категорий</span></div></section>', unsafe_allow_html=True)
        st.subheader("Обзор требований")
        matrix_preview(skills)
        st.caption("Активная матрица выбирается отдельно на странице анализа.")
        st.button("К анализу резюме", on_click=go_to, args=("analysis",))
        return
    draft = st.session_state.matrix_draft
    grid = f"matrix_table_{draft['token']}_{draft['generation']}"
    if grid not in st.session_state and "rows_snapshot" in draft:
        st.session_state.matrix_data = deepcopy(draft["rows_snapshot"])
        draft["generation"] += 1
    if "matrix_data" not in st.session_state:
        st.session_state.matrix_data = profile_rows(draft["profile"]) or [{"Категория": "", "Навык": "", "Уровень": "Базовый", "Описание": "", "Ключевые слова": ""}]
    st.caption("1. Назовите вакансию → 2. Заполните компетенции → 3. Сохраните матрицу")
    name = st.text_input("Название матрицы", value=draft["name"], key=f"matrix_name_{draft['token']}", placeholder="Например, Python-разработчик / Middle")
    description = st.text_area("Описание профиля", value=draft["profile"].get("description", ""), key=f"matrix_description_{draft['token']}", height=100)
    draft["name"], draft["profile"]["description"] = name, description
    st.radio("Способ редактирования", ["Карточки", "Таблица"], key=f"matrix_mode_{draft['token']}", index=0 if draft.get("edit_mode", "Карточки") == "Карточки" else 1, horizontal=True, on_change=change_editor_mode)
    st.button("Добавить компетенцию", on_click=append_matrix_row)
    if draft.get("edit_mode", "Карточки") == "Карточки":
        render_skill_cards(draft)
    else:
        st.caption("Режим массовых правок: строки можно добавлять и удалять прямо в таблице. Для длинных описаний удобнее карточки.")
        draft["rows_snapshot"] = render_bulk_table(draft)
    if "matrix_editor_error" in st.session_state:
        st.error(st.session_state.matrix_editor_error)
    with st.expander("Предпросмотр всей матрицы"):
        matrix_preview(current_matrix_rows())
    save, cancel = st.columns([2, 1])
    with save:
        st.button("Сохранить изменения", on_click=save_matrix_editor, type="primary", use_container_width=True)
    with cancel:
        st.button("Отменить", on_click=close_matrix_editor, use_container_width=True)
