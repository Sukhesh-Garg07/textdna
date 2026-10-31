"""Sentence embedding backends for the semantic layer.

Two backends, tried in order:

1. ``sentence-transformers`` with ``all-MiniLM-L6-v2`` (dense neural
   embeddings, CPU-friendly). Used when the package is installed and the
   model can be loaded/downloaded.
2. ``tfidf`` fallback: scikit-learn TF-IDF vectors fitted on the sentences
   of the text being analysed. Always available, fully offline, but weaker:
   it only captures lexical overlap, not paraphrase.

The backend that actually ran is reported in every analysis result and in
the README. Set ``TEXTDNA_EMBEDDINGS=tfidf`` to force the fallback (used
by the test-suite so tests do not depend on a model download).
"""

from __future__ import annotations

import os
import threading

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_lock = threading.Lock()
_st_model = None
_st_failed = False


def _load_sentence_transformer():
    """Load the sentence-transformer model once, or return None."""
    global _st_model, _st_failed
    if _st_model is not None:
        return _st_model
    if _st_failed:
        return None
    with _lock:
        if _st_model is not None:
            return _st_model
        if _st_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            _st_model = SentenceTransformer(_MODEL_NAME)
        except Exception:
            _st_failed = True
            return None
    return _st_model


def _tfidf_vectors(sentences: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Every sentence was empty/stop-word-only: return zero vectors.
        return np.zeros((len(sentences), 1), dtype=float)
    return matrix.toarray().astype(float)


def sentence_vectors(sentences: list[str]) -> tuple[np.ndarray, str]:
    """Return (vectors, backend_name) for a list of sentences.

    Vectors are L2-normalised so dot products are cosine similarities.
    """
    if not sentences:
        return np.zeros((0, 1), dtype=float), "none"

    forced = os.environ.get("TEXTDNA_EMBEDDINGS", "").strip().lower()
    if forced != "tfidf":
        model = _load_sentence_transformer()
        if model is not None:
            vectors = model.encode(
                sentences, convert_to_numpy=True, show_progress_bar=False
            )
            vectors = np.asarray(vectors, dtype=float)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vectors / norms, "sentence-transformers:all-MiniLM-L6-v2"

    vectors = _tfidf_vectors(sentences)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms, "tfidf"


def coherence_stats(sentences: list[str]) -> dict:
    """Semantic coherence / drift between consecutive sentences.

    coherence: mean cosine similarity of consecutive sentence pairs.
    drift: the largest single drop neighbourhood, reported as
    ``1 - min(pairwise similarity)`` -- a high value means at least one
    abrupt topic change. With fewer than two sentences both are null.
    """
    if len(sentences) < 2:
        return {
            "backend": "none",
            "pairwise_similarities": [],
            "mean_similarity": None,
            "coherence": None,
            "drift": None,
        }
    vectors, backend = sentence_vectors(sentences)
    sims = [
        float(np.dot(vectors[i], vectors[i + 1]))
        for i in range(len(vectors) - 1)
    ]
    mean_sim = float(np.mean(sims)) if sims else None
    drift = float(1.0 - min(sims)) if sims else None
    return {
        "backend": backend,
        "pairwise_similarities": [round(s, 4) for s in sims],
        "mean_similarity": round(mean_sim, 4) if mean_sim is not None else None,
        "coherence": round(mean_sim, 4) if mean_sim is not None else None,
        "drift": round(drift, 4) if drift is not None else None,
    }
