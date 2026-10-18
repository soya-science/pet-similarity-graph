"""Embed Oxford-IIIT Pets with CLIP, link images with cosine > threshold, lay them out in 3D, and export for the viewer."""

import argparse
from pathlib import Path

import numpy as np

from soya_graph.embed import load_or_embed
from soya_graph.export import write_graph
from soya_graph.graph import center, graph_summary, similarity_edges, similarity_report
from soya_graph.layout import umap_3d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.8, help="connect images whose cosine similarity is above this")
    p.add_argument("--max-degree", type=int, default=10, help="each image proposes at most this many strongest edges (0 = no cap)")
    p.add_argument("--center", action="store_true", help="mean-center embeddings before cosine (changes what the threshold means)")
    p.add_argument("--limit", type=int, default=0, help="use a random subset of this many images (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--force-embed", action="store_true", help="recompute embeddings even if cached")
    p.add_argument("--out", type=Path, default=Path("web/graph.json"))
    args = p.parse_args()

    emb, meta = load_or_embed(
        model_name=args.model,
        pretrained=args.pretrained,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
        force=args.force_embed,
    )
    labels = np.asarray(meta["labels"])
    if args.center:
        emb = center(emb)

    similarity_report(emb, labels)
    edges, weights = similarity_edges(emb, args.threshold, args.max_degree)
    graph_summary(edges, weights, labels, len(emb))

    print("\nRunning UMAP to 3D...")
    pos = umap_3d(emb, seed=args.seed)

    write_graph(
        args.out,
        pos=pos,
        edges=edges,
        weights=weights,
        meta=meta,
        params={
            "model": f"{args.model} ({args.pretrained})",
            "threshold": args.threshold,
            "maxDegree": args.max_degree,
            "center": args.center,
            "n": len(emb),
        },
    )


if __name__ == "__main__":
    main()
