import matrix_storage as store
from profiles import profiles


def prepare_selectboxes(app):
    """Streamlit 1.32 AppTest serializes formatted options as raw values.

    Give its frontend simulator the display label; the real selectbox still
    deserializes the selected index to a stable matrix ID. See the note in
    Streamlit 1.32's Selectbox.set_value implementation.
    """
    records = store.load_matrices(profiles)
    for widget in app.selectbox:
        if widget.value in records:
            widget.set_value(records[widget.value]["name"])
