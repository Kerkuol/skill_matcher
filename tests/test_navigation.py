"""Run the actual router without unrelated imports or ML downloads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import matrix_storage as store
from profiles import profiles
from ui_test_support import prepare_selectboxes

ROOT = Path(__file__).resolve().parents[1]
ROUTER = (ROOT / "app.py").read_text(encoding="utf-8").split(
    "# Screens share the same Streamlit session, but only analysis runs the engine.\n", 1)[1]
BOOTSTRAP = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import json, hashlib, sqlite3
from io import BytesIO
import streamlit as st
from profiles import profiles as default_profiles
from matrix_storage import load_matrices
from matrix_editor import render_matrix_settings
from navigation import render_navigation, render_landing, render_history, go_to, remember_uploads, clear_resumes, choose_active_matrix
st.session_state.setdefault("history", [])
'''


class NavigationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(store, "DATABASE", Path(self.directory.name) / "matrices.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.script = Path(self.directory.name) / "flow.py"
        self.script.write_text(BOOTSTRAP + ROUTER, encoding="utf-8")
        self.app = AppTest.from_file(str(self.script), default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def click(self, label):
        prepare_selectboxes(self.app)
        next(button for button in self.app.button if button.label == label).click().run()
        self.assertFalse(self.app.exception)

    def navigate(self, label):
        prepare_selectboxes(self.app)
        self.app.radio[0].set_value(label).run()
        self.assertFalse(self.app.exception)

    def test_landing_analysis_settings_are_separate(self):
        self.assertEqual(len(self.app.selectbox), 0)
        self.assertEqual(len(self.app.get("file_uploader")), 0)
        self.click("Перейти к анализу")
        self.assertEqual(self.app.query_params["view"], ["analysis"])
        self.assertEqual(self.app.selectbox[0].label, "Активная матрица компетенций")
        self.assertEqual(len(self.app.get("file_uploader")), 1)
        self.click("Управление матрицами")
        self.assertEqual(self.app.selectbox[0].label, "Матрица для редактирования")
        self.assertEqual(len(self.app.get("file_uploader")), 0)

    def test_settings_selection_does_not_change_active_matrix(self):
        self.click("Перейти к анализу")
        ids = list(store.load_matrices(profiles))
        self.app.selectbox[0].select_index(1).run()
        self.assertEqual(self.app.session_state["active_matrix_id"], ids[1])
        self.click("Управление матрицами")
        self.app.selectbox[0].select_index(2).run()
        self.click("К анализу резюме")
        self.assertEqual(self.app.selectbox[0].value, ids[1])

    def test_draft_survives_navigation(self):
        self.click("Настроить матрицы")
        self.click("Создать матрицу")
        self.app.text_input[0].set_value("Черновик вакансии")
        draft = self.app.session_state["matrix_draft"]
        key = f"matrix_table_{draft['token']}_{draft['generation']}"
        self.app.session_state[key] = {"edited_rows": {0: {"Категория": "Код", "Навык": "Python"}}, "added_rows": [], "deleted_rows": []}
        self.navigate("Главная")
        self.navigate("Настройки")
        self.assertEqual(self.app.text_input[0].value, "Черновик вакансии")
        self.assertEqual(self.app.session_state["matrix_data"][0]["Навык"], "Python")
        self.click("Отменить")
        self.assertEqual(len(store.load_matrices(profiles)), len(profiles))

    def test_direct_link_and_unknown_route(self):
        self.app.query_params["view"] = "settings"
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.selectbox[0].label, "Матрица для редактирования")
        self.app.query_params["view"] = "unknown"
        prepare_selectboxes(self.app)
        self.app.run()
        self.assertEqual(len(self.app.selectbox), 0)
        self.assertEqual(self.app.radio[0].value, "Главная")
