"""3D node positions from the embeddings via UMAP."""

import warnings

import numpy as np


def umap_3d(emb, n_neighbors=15, min_dist=0.1, seed=42, scale=100.0):
    """Project embeddings to 3D with cosine-metric UMAP, centered and scaled to fit within [-scale, scale]."""
    import umap

    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    with warnings.catch_warnings():
        # Fixing random_state forces single-threaded UMAP; that is fine at this size.
        warnings.filterwarnings("ignore", message="n_jobs value")
        pos = reducer.fit_transform(emb)

    pos = pos - (pos.min(axis=0) + pos.max(axis=0)) / 2
    pos *= scale / np.abs(pos).max()
    return pos.astype(np.float32)
