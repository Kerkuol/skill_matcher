from pathlib import Path
import sys
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""Compatibility import for the prototype; implementation is shared with the app."""
from skill_graph.scoring import *
