import os
import sys
from pathlib import Path

# Force the offline TF-IDF embedding backend so the test-suite is fast
# and does not depend on a model download.
os.environ["TEXTDNA_EMBEDDINGS"] = "tfidf"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
