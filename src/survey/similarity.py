from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SimilarityResult:
    """Nearest-neighbor similarity for each row."""

    nearest_index: np.ndarray  # shape (n,)
    nearest_similarity: np.ndarray  # shape (n,)


def tfidf_nearest_neighbor_similarity(
    texts: List[str],
    *,
    max_features: int = 5000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> SimilarityResult:
    """
    Compute a near-duplicate score using TF-IDF + cosine nearest neighbors.

    Returns, for each row i, the most similar *other* row and the similarity value.
    """
    n = len(texts)
    if n == 0:
        return SimilarityResult(nearest_index=np.array([], dtype=int), nearest_similarity=np.array([], dtype=float))
    if n == 1:
        return SimilarityResult(nearest_index=np.array([-1], dtype=int), nearest_similarity=np.array([0.0], dtype=float))

    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
    )
    X = vec.fit_transform(texts)

    # NearestNeighbors with cosine distance works on sparse matrices efficiently.
    # We ask for 2 neighbors because the closest neighbor is the row itself (distance 0).
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, n_neighbors=2, return_distance=True)

    # For each row, take the second neighbor (index 1), which should be the closest other row.
    other_idx = indices[:, 1].astype(int)
    other_dist = distances[:, 1].astype(float)
    other_sim = 1.0 - other_dist

    # Numerical safety
    other_sim = np.clip(other_sim, 0.0, 1.0)

    return SimilarityResult(nearest_index=other_idx, nearest_similarity=other_sim)

