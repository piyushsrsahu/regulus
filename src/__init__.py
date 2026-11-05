"""Regulus core package."""
from pathlib import Path

# Repository root, so data/model/DB paths work no matter where the app is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
