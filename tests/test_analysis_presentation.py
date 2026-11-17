"""Result presentation and processing lifecycle, without loading ML models."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import matrix_storage as store
from analytics_charts import coverage_chart, competency_chart, distribution_chart
from test_navigation import BOOTSTRAP, ROUTER, ROOT
from ui_test_support import prepare_selectboxes


class ChartDataTests(unittest.TestCase):
    def test_all_status_counts_and_zero_score_categories_remain_visible(self):
        categories = {
            "Python": dict(percentage=75, matches=[1, 2], partials=[3], missing=[4]),
            "SQL": dict(percentage=0, matches=[], partials=[], missing=[5, 6]),
        }
        ring = coverage_chart(categories)
        self.assertEqual(list(ring.data[0].values), [2, 1, 3])
        radar = competency_chart(categories)
        self.assertEqual(list(radar.data[0].r), [75, 0, 75])
        bars = distribution_chart(categories)
        self.assertEqual([list(trace.y) for trace in bars.data], [[2, 0], [1, 0], [1, 2]])

    def test_empty_matrix_renders_without_fabricated_values(self):
        self.assertEqual(list(coverage_chart({}).data[0].values), [0, 0, 0])
        self.assertEqual(list(competency_chart({}).data[0].r), [])
        self.assertTrue(all(not trace.y for trace in distribution_chart({}).data))


STUBS = '''
def init_models():
    if st.session_state.get("fail_models"):
        raise RuntimeError("Model failed")
    return None, None
def parse_resume(source):
    return "[Invalid PDF]" if source.read() == b"bad" else "Resume text"
def process_resume_text(text):
    return dict(total_percentage=41, categories_analysis={}, extracted_skills=[])
def update_analytics(*args):
    pass
def display_single_resume_analysis(*args):
    st.write("RESULTS READY")
def display_comparative_analysis(*args):
    st.write("BATCH READY")
'''


class ProcessingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(store, "DATABASE", Path(self.directory.name) / "matrices.sqlite3")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.script = Path(self.directory.name) / "processing.py"
        self.script.write_text(BOOTSTRAP + STUBS + ROUTER, encoding="utf-8")
        self.app = AppTest.from_file(str(self.script), default_timeout=30)
        self.app.query_params["view"] = "analysis"
        self.app.run()

    def process(self, contents):
        self.app.session_state["workspace_files"] = [dict(name="test.pdf", content=contents)]
        prepare_selectboxes(self.app)
        self.app.run()

    def assert_no_loader(self):
        self.assertEqual(len(self.app.get("status")), 0)
        self.assertFalse(any("processing-banner" in item.value for item in self.app.markdown))

    def test_success_removes_loader_and_shows_cached_results(self):
        self.process(b"resume")
        self.assertFalse(self.app.exception)
        self.assert_no_loader()
        self.assertTrue(any(item.value == "RESULTS READY" for item in self.app.markdown))
        prepare_selectboxes(self.app)
        self.app.run()
        self.assert_no_loader()
        self.assertEqual(len(self.app.session_state["history"]), 1)

    def test_parse_error_removes_loader_but_keeps_error_visible(self):
        self.process(b"bad")
        self.assertFalse(self.app.exception)
        self.assert_no_loader()
        self.assertEqual(len(self.app.error), 1)
        self.assertIn("Invalid PDF", self.app.error[0].value)

    def test_model_failure_also_removes_loader(self):
        self.app.session_state["fail_models"] = True
        self.process(b"resume")
        self.assert_no_loader()
        self.assertEqual(len(self.app.exception), 1)
        self.assertFalse(self.app.session_state["workspace_results"])
