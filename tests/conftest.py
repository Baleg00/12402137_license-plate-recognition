import sys
from pathlib import Path

# Add repo root to Python path so `import model` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
