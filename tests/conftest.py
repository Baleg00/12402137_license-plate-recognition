import sys
from pathlib import Path

# Add root to Python path so imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
