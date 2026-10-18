"""Cosine-similarity edges between embeddings, plus diagnostics for choosing the threshold."""

import numpy as np


def center(emb):
    """Subtract the mean embedding and re-normalize, which spreads out CLIP's crowded similarity range."""
    emb = emb - emb.mean(axis=0, keepdims=True)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _blocks(emb, block):
    """Yield (start, sims) where sims is the cosine matrix of rows start..start+block against all rows, self-pairs set to -inf."""
    for start in range(0, len(emb), block):
        sims = emb[start:start + block] @ emb.T
        rows = np.arange(len(sims))
        sims[rows, start + rows] = -np.inf
        yield start, sims


def similarity_edges(emb, threshold=0.8, max_degree=10, block=2048):
    """Undirected edges between pairs with cosine > threshold.

    With max_degree > 0, each node only proposes its max_degree strongest neighbors above the threshold;
    the result is the union of all proposals, so a popular node can still end up with more edges.
    Returns (edges [E, 2] int32 with i < j, weights [E] float32).
    """
    n = len(emb)
    src, dst, w = [], [], []
    for start, sims in _blocks(emb, block):
        if 0 < max_degree < n - 1:
            cand = np.argpartition(-sims, max_degree - 1, axis=1)[:, :max_degree]
            vals = np.take_along_axis(sims, cand, axis=1)
            r, c = np.nonzero(vals > threshold)
            cols, vals = cand[r, c], vals[r, c]
        else:
            r, cols = np.nonzero(sims > threshold)
            vals = sims[r, cols]
        src.append(start + r)
        dst.append(cols)
        w.append(vals)

    src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
    i, j = np.minimum(src, dst), np.maximum(src, dst)
    _, first = np.unique(i.astype(np.int64) * n + j, return_index=True)
    edges = np.stack([i[first], j[first]], axis=1).astype(np.int32)
    return edges, w[first].astype(np.float32)


def similarity_report(emb, labels, thresholds=(0.7, 0.75, 0.8, 0.85, 0.9), sample=2000, seed=0, block=2048):
    """Print the distribution of pairwise cosine scores and how many edges each threshold would produce."""
    n = len(emb)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    s = emb[idx] @ emb[idx].T
    pairs = s[np.triu_indices(len(idx), k=1)]
    pct = [1, 5, 25, 50, 75, 95, 99]
    print(f"\nPairwise cosine similarity ({len(pairs):,} sampled pairs):")
    print("  " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(pct, np.percentile(pairs, pct))))
    print(f"  max={pairs.max():.3f}")

    thresholds = np.asarray(thresholds)
    total = np.zeros(len(thresholds), dtype=np.int64)
    same = np.zeros(len(thresholds), dtype=np.int64)
    for start, sims in _blocks(emb, block):
        same_label = labels[start:start + len(sims), None] == labels[None, :]
        for t_idx, t in enumerate(thresholds):
            above = sims > t
            total[t_idx] += above.sum()
            same[t_idx] += (above & same_label).sum()
    total //= 2
    same //= 2

    print("\nEdges per threshold (no degree cap):")
    print(f"  {'threshold':>9}  {'edges':>10}  {'mean degree':>11}  {'same breed':>10}")
    for t, e, sb in zip(thresholds, total, same):
        share = f"{sb / e:.1%}" if e else "-"
        print(f"  {t:>9.2f}  {e:>10,}  {2 * e / n:>11.1f}  {share:>10}")


def graph_summary(edges, weights, labels, n):
    """Print the final graph's size, degree spread, and how often edges join images of the same breed."""
    degree = np.bincount(edges.ravel(), minlength=n)
    same = (labels[edges[:, 0]] == labels[edges[:, 1]]).mean() if len(edges) else 0.0
    print(f"\nGraph: {n:,} nodes, {len(edges):,} edges")
    print(f"  degree mean={degree.mean():.1f} max={degree.max()} isolated={int((degree == 0).sum()):,}")
    if len(edges):
        print(f"  weight min={weights.min():.3f} max={weights.max():.3f}")
    print(f"  same-breed edges: {same:.1%} (chance would be ~{1 / len(np.unique(labels)):.1%})")
