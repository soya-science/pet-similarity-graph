# pet-similarity-graph

An interactive 3D graph of the 7,349 photos in [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 cat and dog breeds). Photos that a CLIP model sees as similar sit close together and are linked. Zoom in and the points turn into the photos themselves.

## How it works

1. **Embed.** Each photo goes through a pretrained CLIP image encoder (open_clip `ViT-B-32`, `laion2b_s34b_b79k`) to get a 512-dimensional embedding. This is inference only; nothing is trained or fine-tuned.
2. **Link.** Two photos are connected when the cosine similarity of their embeddings is above a threshold (default `0.8`). Each photo proposes at most its 10 strongest matches, which keeps dense clusters from turning into a solid mass of edges.
3. **Lay out.** UMAP reduces the embeddings to 3D, so position reflects similarity across the whole dataset. The links are drawn on top.
4. **Explore.** A static three.js page renders the graph. Every thumbnail is packed into one texture atlas, so thousands of photos can be drawn at once.

The breed labels are never used to build the graph. They only color the points and power the "same breed" statistics, which makes them a check on whether the links make sense.

## What it shows

With the defaults:

- **44,913 links.** 83% of them join two photos of the same breed; picking pairs at random, the breeds would match only about 3% of the time.
- **713 photos have no link** above 0.8.
- **Cats and dogs form two separate clusters**, with breeds grouping inside each.
- **Look-alike breeds link to each other most:** Beagle ↔ Basset Hound, Siamese ↔ Birman, Japanese Chin ↔ Havanese.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). The pinned Python is 3.12, which uv downloads if needed. Embedding uses the Apple Silicon GPU (MPS) or CUDA when available, and falls back to CPU.

```sh
uv sync
uv run python build.py                      # first run downloads the dataset (~800 MB)
uv run python -m http.server -d web 8000    # open http://localhost:8000
```

On an M4 Pro the full build takes about a minute. Embeddings are cached in `cache/`, so later runs with different graph or layout settings take about 20 seconds.

To open it on a phone on the same Wi-Fi, go to `http://<your-computer's-ip>:8000`. On a Mac, `ipconfig getifaddr en0` prints that IP.

## Using the viewer

- **Move around:** drag to orbit, scroll or pinch to zoom, right-drag or two-finger drag to pan. The view rotates slowly until you touch it.
- **Zoom in** and the dots turn into photos.
- **Hover** a point to see its photo. **Click** it to fly there and list its most similar photos with their cosine scores. Click one of those to jump to it. Esc clears the selection.
- **Find a breed** with the search box or the breed chips in the panel. The breed is highlighted, the camera frames it, and the panel shows how many of its links stay inside the breed, which breeds it is most often linked to, and its best-connected photos.
- **Similarity ≥** raises the cutoff for which links are shown. It can only go up from the build threshold; to go lower, rebuild with a smaller `--threshold`.
- **Links / Photos / Rotate** toggle the edges, the zoom-to-photo effect and the auto-rotation. The reset button returns to the starting view.

On phones the controls fold into a one-row toolbar. The slider and toggles sit behind the options button, and details live in a bottom sheet that you swipe or tap to expand. The page follows the system light or dark theme.

## Build options

| flag | default | meaning |
|---|---|---|
| `--threshold` | `0.8` | connect pairs with cosine similarity above this |
| `--max-degree` | `10` | each photo proposes at most this many strongest links (`0` = no cap). The graph keeps the union, so a popular photo can end up with more |
| `--center` | off | subtract the mean embedding before computing cosine. This spreads out CLIP's crowded score range, so the threshold means something different |
| `--limit` | all | use a random subset of N photos (handy for a quick test) |
| `--model`, `--pretrained` | `ViT-B-32`, `laion2b_s34b_b79k` | any open_clip model |
| `--umap-neighbors`, `--umap-min-dist` | `30`, `0.3` | UMAP settings. More neighbors keeps more global structure; a larger min-dist makes clusters less clumped |
| `--force-embed` | off | ignore the embedding cache |
| `--out` | `web/graph.json` | where to write the graph (the atlas goes next to it) |

### Choosing the threshold

Every build prints the spread of pairwise cosine scores and how many links each threshold would give, so you can choose a cutoff from the data instead of guessing:

```
Pairwise cosine similarity (1,999,000 sampled pairs):
  p1=0.327  p5=0.383  p25=0.468  p50=0.532  p75=0.601  p95=0.714  p99=0.793

Edges per threshold (no degree cap):
  threshold       edges  mean degree  same breed
       0.70   1,662,250        452.4       26.7%
       0.75     676,731        184.2       41.7%
       0.80     225,786         61.4       58.5%
       0.85      52,025         14.2       72.6%
       0.90       4,885          1.3       86.3%
```

For this model, 0.8 sits near the top 1% of pairs. Without the degree cap it would still give about 61 links per photo; with the cap of 10 it gives 44,913 links, 83% of them within a breed.

## Project layout

```
build.py                 CLI that runs the whole pipeline
soya_graph/              Python package for the pipeline steps
  embed.py               dataset loading, thumbnails, CLIP embeddings, cache
  graph.py               cosine edges, threshold report, graph summary
  layout.py              UMAP to 3D, aligned to principal axes
  export.py              graph.json and the thumbnail atlas
web/
  index.html             page, styles, three.js import map
  main.js                viewer: rendering, picking, panel, controls
```

Generated files (all gitignored):

| path | size | contents |
|---|---|---|
| `data/` | 1.6 GB | downloaded dataset |
| `cache/` | 14 MB | embeddings and metadata |
| `web/thumbs/` | 130 MB | one 256 px thumbnail per photo |
| `web/atlas.jpg` | 10 MB | all thumbnails at 64 px in one image |
| `web/graph.json` | 1.6 MB | positions, labels and links |
