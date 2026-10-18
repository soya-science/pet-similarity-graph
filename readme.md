# soya-graph

A 3D similarity graph of the [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) photos.

- Each image is embedded with a pretrained CLIP model (open_clip `ViT-B-32`, `laion2b_s34b_b79k`). This is inference only; nothing is trained.
- Two images are connected when the cosine similarity of their embeddings is above a threshold (default 0.8).
- Node positions come from UMAP of the embeddings down to 3D. The edges are drawn on top.
- The result is explored in the browser (three.js). Points turn into the photos themselves as you zoom in, which works by packing every thumbnail into one texture atlas, `web/atlas.jpg`. Search or click a breed to highlight it. Click a photo to fly to it and list its most similar photos.

## Run

```sh
uv sync
uv run python build.py                      # downloads the dataset (~800 MB) on first run
uv run python -m http.server -d web 8000    # then open http://localhost:8000
```

Embeddings are cached in `cache/`, so re-running with a different threshold takes seconds.

## Options

| flag | default | meaning |
|---|---|---|
| `--threshold` | `0.8` | connect pairs with cosine above this |
| `--max-degree` | `10` | each image proposes at most this many strongest edges (`0` = no cap). The graph keeps the union, so a popular image can end up with more |
| `--center` | off | subtract the mean embedding before computing cosine. This spreads out CLIP's crowded score range, so the threshold means something different |
| `--limit` | all | use a random subset of N images |
| `--model`, `--pretrained` | `ViT-B-32`, `laion2b_s34b_b79k` | any open_clip model |
| `--force-embed` | off | ignore the embedding cache |
| `--umap-neighbors`, `--umap-min-dist` | `30`, `0.3` | UMAP layout settings. More neighbors keeps more global structure; a larger min-dist makes clusters less clumped |

Every run prints the distribution of pairwise cosine scores and how many edges each threshold would produce, to help pick the cutoff.
