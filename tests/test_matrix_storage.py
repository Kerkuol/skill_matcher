import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matrix_storage as store
from profiles import profiles


class MatrixStorageTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(store, "DATABASE", Path(self.directory.name) / "matrices.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.records = store.load_matrices(profiles)
        self.key = next(iter(self.records))
        self.record = self.records[self.key]

    def test_original_profiles_roundtrip_unchanged(self):
        loaded = {r["name"]: r["profile"] for r in store.load_matrices(profiles).values()}
        self.assertEqual(loaded, profiles)

    def test_create_and_reload(self):
        key = store.save_matrix("Новая вакансия", self.record["profile"])
        self.assertEqual(store.load_matrices(profiles)[key]["name"], "Новая вакансия")

    def test_copy_is_independent(self):
        profile = copy.deepcopy(self.record["profile"])
        profile["description"] = "Копия"
        key = store.save_matrix("Копия", profile)
        loaded = store.load_matrices(profiles)
        self.assertEqual(loaded[self.key], self.record)
        self.assertEqual(loaded[key]["profile"]["description"], "Копия")

    def test_rename_preserves_id_and_does_not_reseed(self):
        store.save_matrix("Новое название", self.record["profile"], self.key, 1)
        loaded = store.load_matrices(profiles)
        self.assertEqual(len(loaded), len(profiles))
        self.assertEqual(loaded[self.key]["name"], "Новое название")

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            store.save_matrix(" " + self.record["name"].upper() + " ", self.record["profile"])

    def test_stale_editor_does_not_overwrite(self):
        store.save_matrix("Первое изменение", self.record["profile"], self.key, 1)
        with self.assertRaises(ValueError):
            store.save_matrix("Устаревшее изменение", self.record["profile"], self.key, 1)
        self.assertEqual(store.load_matrices(profiles)[self.key]["name"], "Первое изменение")

    def test_empty_and_incomplete_rows(self):
        with self.assertRaises(ValueError):
            store.profile_from_rows([], "")
        with self.assertRaises(ValueError):
            store.profile_from_rows([{"Навык": "Python"}], "")

    def test_row_validation_and_optional_fields(self):
        rows = [{"Категория": " Код ", "Навык": " Python ", "Уровень": "Средний", "Ключевые слова": "py, py, Python"}, {}]
        result = store.profile_from_rows(rows, "")
        self.assertEqual(result["categories"]["Код"]["Python"]["level"], 2)
        self.assertEqual(result["categories"]["Код"]["Python"]["keywords"], ["py", "Python"])
        rows.append({"Категория": "код", "Навык": "python", "Уровень": "Базовый"})
        with self.assertRaises(ValueError):
            store.profile_from_rows(rows, "")

    def test_write_error_does_not_update_profile(self):
        with patch.object(store, "_connect", side_effect=sqlite3.OperationalError("read only")):
            with self.assertRaises(sqlite3.Error):
                store.save_matrix("Ошибка", self.record["profile"], self.key, 1)
        self.assertEqual(store.load_matrices(profiles)[self.key], self.record)


if __name__ == "__main__":
    unittest.main()
