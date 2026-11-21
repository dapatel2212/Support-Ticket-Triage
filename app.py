"""Root launcher for SmartTriage AI Streamlit app."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "smarttriage-ai-glm" / "smarttriage-ai"

if PROJECT_DIR.exists():
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))
    app_entry = PROJECT_DIR / "app.py"
    with open(app_entry, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, str(app_entry), "exec"))
else:
    raise FileNotFoundError(f"Directory not found: {PROJECT_DIR}")
