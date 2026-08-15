"""Assembling the weekly mentor report. CLI entrypoint: ``tools/weekly_report.py``.

``tools/`` holds scripts rather than an installed package, so importing from it takes a path entry.
It is done here, once, because ``tools/experiments_index.py`` is the repo's only reader of the
``notes.md`` registry frontmatter and a second parser in this package would drift from what
``tools/experiments_index.py --check`` enforces.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_TOOLS = str(REPO_ROOT / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
