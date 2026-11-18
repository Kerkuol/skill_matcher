"""Refresh the isolated prototype entry point from the production application."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def build():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    source = source.replace("PROJECT_ROOT = Path(__file__).resolve().parent", "PROJECT_ROOT = ROOT")
    source = source.replace('page_title="AI Skill Matcher"', 'page_title="Skill Matcher · прототип"')
    source = source.replace('apply_theme()\n', 'apply_theme()\nst.caption("Прототип · отдельная библиотека матриц")\n', 1)
    anchor = "# Screens share the same Streamlit session, but only analysis runs the engine."
    source = source.replace(anchor, 'if "view" not in st.query_params:\n    st.query_params["view"] = "analysis"\n\n' + anchor)
    header = "from pathlib import Path\nimport sys\nHERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nfrom bootstrap import initialize, ROOT\ninitialize()\n"
    output = "\n".join(line.rstrip() for line in (header + source).splitlines()) + "\n"
    (HERE / "app.py").write_text(output, encoding="utf-8")
    (HERE / "matrix_editor.py").write_bytes((ROOT / "matrix_editor.py").read_bytes())
    print("Prototype uses shared application modules and its own matrix database.")

if __name__ == "__main__":
    build()
