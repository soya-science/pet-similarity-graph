"""Write the graph as compact JSON for the web viewer, plus a texture atlas of all thumbnails."""

import json
import math

import numpy as np
from PIL import Image


def write_atlas(thumb_dir, n, path, tile=64):
    """Pack thumbs 0..n-1 into one square-ish JPEG grid, row-major from the top-left. Returns its layout."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    atlas = Image.new("RGB", (cols * tile, rows * tile))
    for i in range(n):
        with Image.open(thumb_dir / f"{i}.jpg") as im:
            atlas.paste(im.resize((tile, tile), Image.LANCZOS), ((i % cols) * tile, (i // cols) * tile))
    atlas.save(path, quality=85)
    print(f"Wrote {path} ({cols}x{rows} tiles of {tile}px, {path.stat().st_size / 1e6:.1f} MB)")
    # The version query makes browsers fetch the new atlas after a rebuild instead of a cached one.
    return {"url": f"{path.name}?v={int(path.stat().st_mtime)}", "cols": cols, "rows": rows, "tile": tile}


def write_graph(path, *, pos, edges, weights, meta, params, atlas=None):
    data = {
        "params": params,
        "classes": meta["classes"],
        "classSpecies": meta["class_species"],
        "atlas": atlas,
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
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
