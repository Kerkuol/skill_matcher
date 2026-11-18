"""Exercise the deployed entry point with local graph inference, no stubs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
import matrix_storage as store
from profiles import profiles
from skill_graph.scoring import assess_resume
from skill_graph.network import graph_data
from ui_test_support import prepare_selectboxes


class MainGraphTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.folder.cleanup)
        patcher = patch.object(store, "DATABASE", Path(self.folder.name) / "matrices.sqlite3")
        patcher.start(); self.addCleanup(patcher.stop)
        store.load_matrices(profiles)
        self.profile = dict(categories={"Код": {s: dict(level=3, description="", keywords=[]) for s in ["Docker", "SQL"]}})
        self.matrix = store.save_matrix("Проверка графа", self.profile)
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
        self.app.query_params["view"] = "analysis"
        self.app.session_state["active_matrix_id"] = self.matrix
        self.app.run()
        self.assertFalse(self.app.exception)

    def upload(self, files):
        prepare_selectboxes(self.app)
        self.app.session_state["workspace_files"] = [dict(name=name, content=text.encode("utf-8")) for name, text in files]
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_main_graph_analytics_and_no_heavy_model_imports(self):
        self.upload([("resume.txt", "Я настроил Docker.\n\nЯ писал SQL-запросы к PostgreSQL.")])
        self.assertEqual(self.app.session_state["history"][-1]["score"], 100.)
        self.assertEqual(len(self.app.get("plotly_chart")), 2)
        html = self.app.get("iframe")[0].proto.srcdoc
        self.assertIn("#ffcc58", html)
        self.assertIn("overview_edges", html)
        self.assertNotIn("#21ad78", html)
        self.assertFalse(any("Как граф получил результат" in item.value for item in self.app.markdown))
        self.assertFalse(any('metric-label">Уровень' in item.value for item in self.app.markdown))
        for name in ("transformers", "sentence_transformers", "torch", "keybert"):
            self.assertNotIn(name, sys.modules)
        before = len(self.app.session_state["history"])
        prepare_selectboxes(self.app); self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state["history"]), before)

    def test_main_comparison_duplicate_names_and_zero_score(self):
        self.upload([("resume.txt", "Я настроил Docker.\n\nЯ писал SQL-запросы к PostgreSQL."),
                     ("resume.txt", "Коллега настроил Docker.")])
        self.assertEqual([h["score"] for h in self.app.session_state["history"]], [100., 0.])
        picker = self.app.selectbox(key="comparison_candidate")
        self.assertEqual(len(set(picker.options)), 2)
        first = self.app.get("iframe")[0].proto.srcdoc
        prepare_selectboxes(self.app)
        picker.select("02 · resume.txt").run()
        self.assertFalse(self.app.exception)
        self.assertNotEqual(first, self.app.get("iframe")[0].proto.srcdoc)
        self.assertEqual(len(self.app.session_state["history"]), 2)

    def test_unknown_requirement_is_visible_and_not_removed(self):
        profile = dict(categories={"Проверка": {s: dict(level=1, description="") for s in ["Docker", "Квантовая телепортация"]}})
        result = assess_resume("Я настроил Docker.", profile)
        self.assertEqual(result["total_percentage"], 50.)
        self.assertEqual(result["graph_support"], dict(supported=1, total=2))
        graph = graph_data(result["graph_details"])
        ids = {n["id"] for n in graph["nodes"]}
        self.assertTrue(all(e["source"] in ids and e["target"] in ids for e in graph["edges"] + graph["overview_edges"] + graph["relationships"]))
        self.assertTrue(all(not edge["practical"] for edge in graph["relationships"]))


if __name__ == "__main__":
    unittest.main()
