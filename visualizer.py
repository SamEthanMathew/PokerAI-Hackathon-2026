"""
Shim for the legacy Streamlit match log viewer.

Prefer (from repo root):
  streamlit run tools/visualizer_streamlit.py
"""

import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_script = _root / "tools" / "visualizer_streamlit.py"
raise SystemExit(
    subprocess.call([sys.executable, "-m", "streamlit", "run", str(_script), *sys.argv[1:]])
)
