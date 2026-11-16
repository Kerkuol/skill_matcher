"""Exercise the actual editor block without starting unrelated ML models."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import matrix_storage as store
from profiles import profiles
from ui_test_support import prepare_selectboxes

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from matrix_editor import render_matrix_settings
render_matrix_settings()
"""



class MatrixEditorTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(store, "DATABASE", Path(self.directory.name) / "matrices.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.script = Path(self.directory.name) / "editor.py"
        self.script.write_text(BOOTSTRAP, encoding="utf-8")
        self.app = AppTest.from_file(str(self.script), default_timeout=30).run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error, [e.value for e in self.app.error])

    def click(self, label):
        prepare_selectboxes(self.app)
        next(b for b in self.app.button if b.label == label).click().run()
        self.assertFalse(self.app.exception)

    def table_key(self):
        draft = self.app.session_state["matrix_draft"]
        return f"matrix_table_{draft['token']}_{draft['generation']}"

    def test_create_add_row_save_and_fresh_session(self):
        self.click("Создать матрицу")
        self.assertTrue(self.app.selectbox[0].disabled)
        self.app.text_input[0].set_value("Backend")
        self.app.session_state[self.table_key()] = {"edited_rows": {0: {"Категория": "Код", "Навык": "Python", "Уровень": "Средний"}}, "added_rows": [], "deleted_rows": []}
        self.click("Добавить строку")
        self.assertEqual(self.app.session_state["matrix_data"][0]["Навык"], "Python")
        self.click("Сохранить изменения")
        self.assertFalse(self.app.error)
        self.assertEqual(store.load_matrices(profiles)[self.app.selectbox[0].value]["name"], "Backend")
        fresh = AppTest.from_file(str(self.script), default_timeout=30).run()
        self.assertIn("Backend", fresh.selectbox[0].options)

    def test_copy_cancel_and_rename(self):
        original = store.load_matrices(profiles)
        self.click("Копировать матрицу")
        self.assertTrue(self.app.text_input[0].value.endswith("— копия"))
        self.click("Отменить")
        self.assertEqual(store.load_matrices(profiles), original)
        self.click("Копировать матрицу")
        self.click("Сохранить изменения")
        self.assertEqual(len(store.load_matrices(profiles)), len(original) + 1)
        self.click("Редактировать матрицу компетенций")
        self.app.text_input[0].set_value("Переименованная матрица")
        self.click("Сохранить изменения")
        self.assertEqual(store.load_matrices(profiles)[self.app.selectbox[0].value]["name"], "Переименованная матрица")
        self.assertEqual(len(store.load_matrices(profiles)), len(original) + 1)

    def test_invalid_save_keeps_draft(self):
        self.click("Создать матрицу")
        self.app.text_input[0].set_value("Пустая")
        self.click("Сохранить изменения")
        self.assertTrue(self.app.error)
        self.assertEqual(self.app.text_input[0].value, "Пустая")
        self.assertTrue(self.app.session_state["show_matrix_editor"])

    def test_edit_skill_cancel_then_save(self):
        original = store.load_matrices(profiles)
        matrix_id = next(iter(original))
        self.click("Редактировать матрицу компетенций")
        changes = {"edited_rows": {0: {"Описание": "Изменённое описание", "Уровень": "Базовый"}}, "added_rows": [], "deleted_rows": []}
        self.app.session_state[self.table_key()] = changes
        self.click("Отменить")
        self.assertEqual(store.load_matrices(profiles), original)
        self.click("Редактировать матрицу компетенций")
        self.app.session_state[self.table_key()] = changes
        self.click("Сохранить изменения")
        saved = store.load_matrices(profiles)[matrix_id]
        first = store.profile_rows(saved["profile"])[0]
        self.assertEqual(first["Описание"], "Изменённое описание")
        self.assertEqual(first["Уровень"], "Базовый")
        self.assertEqual(saved["revision"], 2)
