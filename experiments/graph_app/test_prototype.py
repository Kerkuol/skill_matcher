"""Real graph scoring and existing Streamlit UI, without models/network."""
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESUME_FIXTURES = ROOT / "tests/fixtures/resumes"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from scoring import assess_resume, contribution
from bootstrap import parse_resume


def profile(*skills):
    return dict(categories={"Тест": {skill: dict(level=2, description="") for skill in skills}})


class ScoringTests(unittest.TestCase):
    def test_score_ignores_levels(self):
        self.assertEqual(contribution(dict(status="practical", level=None), 2), 1.)
        self.assertEqual(contribution(dict(status="practical", level=1), 2), 1.)
        self.assertEqual(contribution(dict(status="practical", level=3), 2), 1.)
        for status in ("related", "learning", "negated", "other_actor", "unsupported"):
            self.assertEqual(contribution(dict(status=status, level=3), 2), 0.)

    def test_unknown_requirements_are_not_removed_from_denominator(self):
        result = assess_resume("Я настроил Docker.", profile("Docker", "Квантовая телепортация"))
        self.assertEqual(result["graph_support"], dict(supported=1, total=2))
        self.assertEqual(result["total_percentage"], 50.)

    def test_all_negative_still_produces_chart_categories(self):
        result = assess_resume("Коллега настроил Docker.", profile("Docker"))
        self.assertEqual(result["total_percentage"], 0)
        self.assertEqual(len(result["categories_analysis"]["Тест"]["missing"]), 1)

    def test_pdf_paraphrases_keep_practical_evidence(self):
        expected = profile("Разработка ETL-процессов", "Автоматизация отчётности", "SQL", "Docker", "Контроль качества данных")
        scores = []
        for name in ("01_data_engineer_direct", "02_data_engineer_paraphrased"):
            source = BytesIO((RESUME_FIXTURES / (name + ".pdf")).read_bytes())
            source.name = name + ".pdf"
            result = assess_resume(parse_resume(source), expected)
            self.assertEqual(result["confirmed_skills_count"], 5)
            scores.append(result["total_percentage"])
        self.assertEqual(scores, [100., 100.])

    def test_docx_and_invalid_file(self):
        from docx import Document
        stream = BytesIO()
        doc = Document()
        doc.add_paragraph("Я настроил Docker.")
        doc.save(stream)
        stream.name = "resume.docx"
        self.assertIn("Docker", parse_resume(stream))
        bad = BytesIO(b"not a pdf"); bad.name = "broken.pdf"
        self.assertTrue(parse_resume(bad).startswith("["))

    def test_levels_removed_and_unknown_skills_separated_in_recommendations(self):
        from presentation import missing_requirements
        result = assess_resume("Я настроил Docker.", profile("Docker", "SQL", "Квантовая телепортация"))
        self.assertTrue(all("level" not in row and "required_level" not in row for row in result["graph_details"]))
        self.assertEqual([r["skill"] for r in missing_requirements(result)], ["SQL", "Квантовая телепортация"])
        from network_view import graph_data
        network = graph_data(result["graph_details"])
        ids = {n["id"] for n in network["nodes"]}
        self.assertTrue(all(e["source"] in ids and e["target"] in ids for e in network["edges"]))
        self.assertTrue(any(e["practical"] for e in network["edges"]))


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.app = AppTest.from_file(str(HERE / "app.py"), default_timeout=40).run()
        self.assertFalse(self.app.exception)

    def prepare(self):
        import matrix_storage
        from profiles import profiles
        records = matrix_storage.load_matrices(profiles)
        for widget in self.app.selectbox:
            if widget.label == "Активная матрица компетенций" and widget.value in records:
                widget.set_value(records[widget.value]["name"])
            elif widget.label == "Разобрать требование" and isinstance(widget.value, int):
                # Streamlit 1.32 AppTest serializes formatted options as labels.
                widget.set_value(widget.options[widget.value])

    def select_demo(self):
        self.prepare()
        self.app.selectbox(key="analysis_matrix_picker").select("Тест графа: инженер данных").run()
        self.assertFalse(self.app.exception)

    def upload(self, *names):
        self.prepare()
        self.app.session_state["workspace_files"] = [dict(name=name + ".pdf", content=(RESUME_FIXTURES / (name + ".pdf")).read_bytes()) for name in names]
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_upload_renders_existing_charts_and_explanations_and_caches(self):
        self.select_demo()
        self.upload("02_data_engineer_paraphrased")
        self.assertEqual(len(self.app.get("plotly_chart")), 2)
        self.assertTrue(self.app.get("iframe"))
        self.assertFalse(any('metric-label">Уровень' in item.value for item in self.app.markdown))
        self.assertFalse(any("Как граф получил результат" in item.value for item in self.app.markdown))
        self.assertFalse(any(w.label == "Разобрать требование" for w in self.app.selectbox))
        self.assertTrue(any("Граф навыков" in item.value for item in self.app.markdown))
        self.assertEqual(len(self.app.get("status")), 0)
        self.assertEqual(self.app.session_state["history"][-1]["score"], 100.)
        count = len(self.app.session_state["history"])
        self.prepare(); self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state["history"]), count)

    def test_batch_keeps_comparative_charts_and_each_profile(self):
        self.select_demo()
        self.upload("01_data_engineer_direct", "02_data_engineer_paraphrased")
        self.assertEqual(len(self.app.get("plotly_chart")), 3)
        self.assertEqual(len(self.app.get("iframe")), 1)
        self.assertEqual(len(self.app.session_state["history"]), 2)

    def test_comparison_switches_graph_without_reprocessing(self):
        import re
        self.select_demo()
        self.upload("01_data_engineer_direct", "06_student_negative_control")
        first = self.app.get("iframe")[0].proto.srcdoc
        self.prepare()
        self.app.selectbox(key="comparison_candidate").select("02 · 06_student_negative_control.pdf").run()
        self.assertFalse(self.app.exception)
        second = self.app.get("iframe")[0].proto.srcdoc
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.app.session_state["history"]), 2)
        payload = json.loads(re.search(r"const data=(.*?);\nconst canvas", second).group(1))
        self.assertEqual(len(payload["groups"]), 5)
        self.assertTrue(any(n.get("status") != "practical" for n in payload["nodes"] if n["kind"] == "requirement"))

    def test_matrix_switch_recalculates_and_settings_available(self):
        self.select_demo()
        self.upload("02_data_engineer_paraphrased")
        self.prepare()
        self.app.selectbox(key="analysis_matrix_picker").select("Тест графа: подбор и адаптация").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["history"][-1]["score"], 0.)
        self.prepare()
        next(b for b in self.app.button if b.label == "Управление матрицами").click().run()
        self.assertFalse(self.app.exception)
        # All formatted matrix pickers need the same AppTest label workaround.
        import matrix_storage
        from profiles import profiles
        records = matrix_storage.load_matrices(profiles)
        for widget in self.app.selectbox:
            if widget.value in records:
                widget.set_value(records[widget.value]["name"])
        next(b for b in self.app.button if b.label == "Редактировать матрицу компетенций").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any("уровень" in w.label.lower() for w in self.app.selectbox))

    def test_zero_score_and_foreign_actor_still_show_analytics(self):
        self.select_demo()
        self.prepare()
        self.app.session_state["workspace_files"] = [dict(name="other.txt", content="Коллега настроил Docker.".encode("utf-8"))]
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["history"][-1]["score"], 0.)
        self.assertEqual(len(self.app.get("plotly_chart")), 2)

    def test_invalid_pdf_shows_error_and_removes_loader(self):
        self.select_demo()
        self.prepare()
        self.app.session_state["workspace_files"] = [dict(name="broken.pdf", content=b"not a pdf")]
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.error)
        self.assertEqual(len(self.app.get("status")), 0)


if __name__ == "__main__":
    unittest.main()
