"""Pytest configuration: add src/ to sys.path so tests import 'dehazing' directly."""
import sys
from pathlib import Path

# Allow `from dehazing.xxx import ...` and `from scripts.xxx import ...`
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
