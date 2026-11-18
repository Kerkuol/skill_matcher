"""Keep prototype persistence, logs and temporary files inside this folder."""
from io import BytesIO
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def initialize():
    temp = HERE / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temp)
    os.environ["MPLCONFIGDIR"] = str(HERE / "tmp/matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(HERE / "tmp/cache")
    tempfile.tempdir = str(temp)
    os.chdir(HERE)
    import matrix_storage
    target = HERE / "data/matrices.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        source = ROOT / "data/matrices.sqlite3"
        if source.exists():
            with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src:
                with sqlite3.connect(target) as dest:
                    src.backup(dest)
        matrix_storage.DATABASE = target
        from profiles import profiles
        records = matrix_storage.load_matrices(profiles)
        demo_profiles = {
            "Тест графа: инженер данных": ["Разработка ETL-процессов", "Автоматизация отчётности", "SQL", "Docker", "Контроль качества данных"],
            "Тест графа: подбор и адаптация": ["Подбор персонала", "Управление командой", "Обучение и наставничество"],
            "Тест графа: операционный руководитель": ["Управление командой", "Бюджетирование", "Переговоры с поставщиками", "Обучение и наставничество"],
        }
        for name, skills in demo_profiles.items():
            if not any(r["name"] == name for r in records.values()):
                profile = dict(description="Тестовая матрица отдельного графового прототипа.",
                               categories={"Компетенции": {s: dict(level=2, description="", keywords=[]) for s in skills}})
                matrix_storage.save_matrix(name, profile)
    matrix_storage.DATABASE = target
    # Remove the old demo-only caption, preserving user descriptions and data.
    with sqlite3.connect(target) as connection:
        for matrix_id, raw in connection.execute("SELECT id, profile FROM matrices").fetchall():
            profile = json.loads(raw)
            if profile.get("description") == "Тестовая матрица отдельного графового прототипа. Все уровни — средние.":
                profile["description"] = "Тестовая матрица отдельного графового прототипа."
                connection.execute("UPDATE matrices SET profile=?, revision=revision+1 WHERE id=?",
                                   (json.dumps(profile, ensure_ascii=False), matrix_id))


def parse_resume(source):
    """Same supported formats; no production logger or temporary DOCX file."""
    contents = source.getvalue()
    if not contents:
        return "[Файл пуст]"
    if len(contents) > 10 * 1024 * 1024:
        return "[Максимальный размер файла — 10 МБ]"
    suffix = Path(source.name).suffix.lower()
    try:
        if suffix == ".pdf":
            import fitz
            with fitz.open(stream=contents, filetype="pdf") as document:
                text = "\n".join(page.get_text() for page in document)
        elif suffix == ".docx":
            import docx2txt
            text = docx2txt.process(BytesIO(contents))
        elif suffix == ".txt":
            try:
                text = contents.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = contents.decode("cp1251")
        else:
            return "[Поддерживаются PDF, DOCX и TXT]"
        return text.strip() or "[Не удалось извлечь текст. Для PDF-скана нужен текстовый слой.]"
    except Exception as exc:
        return f"[Ошибка чтения: {exc}]"
