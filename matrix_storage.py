"""Local matrix persistence, independent of the screening algorithm."""
import json
import sqlite3
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL

DATABASE = Path(__file__).resolve().parent / "data" / "matrices.sqlite3"
LEVELS = {"Базовый": 1, "Средний": 2, "Продвинутый": 3}


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def profile_from_rows(rows, description):
    categories = {}
    seen = set()
    for index, row in enumerate(rows, 1):
        category, skill = _text(row.get("Категория")), _text(row.get("Навык"))
        detail, keywords = _text(row.get("Описание")), _text(row.get("Ключевые слова"))
        if not any((category, skill, detail, keywords)):
            continue
        if not category or not skill:
            raise ValueError(f"Строка {index}: заполните категорию и навык.")
        key = (category.casefold(), skill.casefold())
        if key in seen:
            raise ValueError(f"Строка {index}: навык «{skill}» повторяется в категории «{category}».")
        seen.add(key)
        level = LEVELS.get(row.get("Уровень"))
        if level is None:
            raise ValueError(f"Строка {index}: выберите уровень навыка.")
        # Keep category spelling consistent when a user varies letter case.
        category = next((c for c in categories if c.casefold() == category.casefold()), category)
        categories.setdefault(category, {})[skill] = {
            "level": level, "description": detail,
            "keywords": list(dict.fromkeys(k.strip() for k in keywords.split(",") if k.strip())),
        }
    if not categories:
        raise ValueError("Добавьте хотя бы один навык.")
    return {"description": description, "categories": categories}


def profile_rows(profile):
    return [{"Категория": category, "Навык": skill, "Уровень": next(label for label, value in LEVELS.items() if value == spec["level"]), "Описание": spec.get("description", ""), "Ключевые слова": ", ".join(spec.get("keywords", []))}
            for category, skills in profile["categories"].items() for skill, spec in skills.items()]


def editor_rows(base_rows, changes):
    """Materialize the data_editor delta before a button callback changes rows."""
    rows = [dict(row) for row in base_rows]
    for index, values in changes.get("edited_rows", {}).items():
        rows[int(index)].update(values)
    deleted = set(changes.get("deleted_rows", []))
    rows = [row for index, row in enumerate(rows) if index not in deleted]
    rows.extend(dict(row) for row in changes.get("added_rows", []))
    return rows


def _connect():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE, timeout=10)
    connection.execute("CREATE TABLE IF NOT EXISTS matrices (id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE, profile TEXT NOT NULL, revision INTEGER NOT NULL)")
    return connection


def load_matrices(defaults):
    connection = _connect()
    try:
        with connection:
            # Seed only an empty database. Renamed templates must not reappear.
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT COUNT(*) FROM matrices").fetchone()[0] == 0:
                for name, profile in defaults.items():
                    connection.execute("INSERT INTO matrices VALUES (?, ?, ?, ?, 1)",
                                       (uuid5(NAMESPACE_URL, name).hex, name, name.casefold(), json.dumps(profile, ensure_ascii=False)))
            rows = connection.execute("SELECT id, name, profile, revision FROM matrices ORDER BY rowid").fetchall()
        return {row[0]: {"name": row[1], "profile": json.loads(row[2]), "revision": row[3]} for row in rows}
    finally:
        connection.close()


def save_matrix(name, profile, matrix_id=None, revision=None):
    name = _text(name)
    if not name:
        raise ValueError("Укажите название матрицы.")
    # Validate even when called outside the UI; preserve existing data format.
    profile = profile_from_rows(profile_rows(profile), profile.get("description", ""))
    connection = _connect()
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute("SELECT id FROM matrices WHERE name_key = ?", (name.casefold(),)).fetchone()
            if duplicate and duplicate[0] != matrix_id:
                raise ValueError("Матрица с таким названием уже существует.")
            payload = json.dumps(profile, ensure_ascii=False)
            if matrix_id is None:
                matrix_id = uuid4().hex
                connection.execute("INSERT INTO matrices VALUES (?, ?, ?, ?, 1)", (matrix_id, name, name.casefold(), payload))
            else:
                updated = connection.execute("UPDATE matrices SET name=?, name_key=?, profile=?, revision=revision+1 WHERE id=? AND revision=?",
                                             (name, name.casefold(), payload, matrix_id, revision))
                if updated.rowcount != 1:
                    raise ValueError("Матрица уже изменена в другом окне. Скопируйте введённые данные, отмените редактирование и откройте её заново.")
        return matrix_id
    finally:
        connection.close()
