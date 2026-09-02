"""
Launch the desktop GUI application.

Usage:
    python scripts/run_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import main

if __name__ == "__main__":
    main()
