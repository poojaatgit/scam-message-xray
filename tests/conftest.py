"""Make the project folder importable so the V4 model can find text_preprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
