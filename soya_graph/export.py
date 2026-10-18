"""Write the graph as compact JSON for the web viewer."""

import json

import numpy as np


def write_graph(path, *, pos, edges, weights, meta, params):
    data = {
        "params": params,
        "classes": meta["classes"],
        "classSpecies": meta["class_species"],
        "nodes": {
            "label": meta["labels"],
            "pos": np.round(pos, 2).ravel().tolist(),
        },
        "edges": {
            "src": edges[:, 0].tolist(),
            "dst": edges[:, 1].tolist(),
            "w": np.round(weights, 4).tolist(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"\nWrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
