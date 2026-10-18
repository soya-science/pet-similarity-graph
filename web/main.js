import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Colors are written straight from the CSS tokens (sRGB) into buffers, so skip three's linear conversion.
THREE.ColorManagement.enabled = false;

const POINT_SIZE = 2.6;    // node diameter in world units (layout spans roughly -100..100)
const FOCUS_SIZE = 3.4;
const SELECTED_SIZE = 5.0;
const MAX_PX_PER_UNIT = 6; // zoomed in, a node stops growing at aSize * this many CSS px
const HOVER_RADIUS = 14;   // CSS px: pointer only has to be this close to the nearest node
const FLY_MS = 650;

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const panel = $("panel");
const tooltip = $("tooltip");
const breedSelect = $("breed");
const thresholdInput = $("threshold");

const state = { breed: -1, threshold: 0, showEdges: true, selected: -1, hovered: -1 };

let n, pos, labels, species, classes, classSpecies, params;
let src, dst, w, sameBreedPrefix, adjacency;
let theme, renderer, scene, camera, controls;
let points, lines, highlight, baseSizes;
let fly = null;

const thumb = (i) => `thumbs/${i}.jpg`;
const mix = (a, b, t) => a.clone().lerp(b, t);
const fmt = (x) => x.toLocaleString();

main();

async function main() {
  let data;
  try {
    const res = await fetch("graph.json");
    if (!res.ok) throw new Error(res.statusText);
    data = await res.json();
  } catch {
    panel.innerHTML = `<div id="error">No <code>graph.json</code> found. Run <code>uv run python build.py</code> first, then serve this folder with <code>uv run python -m http.server -d web 8000</code>.</div>`;
    return;
  }
  loadData(data);
  theme = readTheme();
  initScene();
  initControls();
  recolor();
  renderPanel();
  requestAnimationFrame(animate);
}

function loadData(data) {
  params = data.params;
  classes = data.classes;
  classSpecies = data.classSpecies;
  labels = Int32Array.from(data.nodes.label);
  n = labels.length;
  pos = Float32Array.from(data.nodes.pos);
  species = labels.map((l) => (classSpecies[l] === "cat" ? 0 : 1));

  // Sort edges strongest first, so "edges >= t" is always a prefix and the slider is just a draw range.
  const E = data.edges.w.length;
  const order = Array.from({ length: E }, (_, k) => k).sort((a, b) => data.edges.w[b] - data.edges.w[a]);
  src = new Int32Array(E);
  dst = new Int32Array(E);
  w = new Float32Array(E);
  sameBreedPrefix = new Int32Array(E + 1);
  adjacency = Array.from({ length: n }, () => []);
  order.forEach((k, r) => {
    const a = data.edges.src[k], b = data.edges.dst[k];
    src[r] = a; dst[r] = b; w[r] = data.edges.w[k];
    sameBreedPrefix[r + 1] = sameBreedPrefix[r] + (labels[a] === labels[b] ? 1 : 0);
    adjacency[a].push(r);
    adjacency[b].push(r);
  });
  state.threshold = params.threshold;
}

function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const c = (name) => new THREE.Color(cs.getPropertyValue(name).trim());
  const t = {
    surface: c("--surface"),
    primary: c("--text-primary"),
    muted: c("--text-muted"),
    cat: c("--cat"),
    dog: c("--dog"),
    focus: c("--focus"),
    edgeMix: parseFloat(cs.getPropertyValue("--edge-mix")),
  };
  t.mutedPoint = mix(t.surface, t.muted, 0.55);
  t.dimEdge = mix(t.surface, t.muted, t.edgeMix * 0.5);
  return t;
}

// Number of edges with weight >= t (edges are sorted descending).
function visibleCount(t) {
  let lo = 0, hi = w.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (w[mid] >= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function neighborsOf(i) {
  const out = [];
  for (const r of adjacency[i]) {
    if (w[r] < state.threshold) break;
    out.push({ j: src[r] === i ? dst[r] : src[r], w: w[r] });
  }
  return out;
}

// ---------- scene ----------

function initScene() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(theme.surface);
  stage.prepend(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute("aria-label", "3D similarity graph of pet photos");

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
  camera.position.set(0, 0, 260);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotateSpeed = 0.6;
  controls.addEventListener("start", () => { fly = null; });

  // Nodes: one Points object, round sprites with a thin surface-colored ring so overlapping dots stay separable.
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  pg.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  pg.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array(n), 1));
  baseSizes = new Float32Array(n);
  points = new THREE.Points(pg, new THREE.ShaderMaterial({
    uniforms: {
      uScale: { value: 1 },
      uDpr: { value: renderer.getPixelRatio() },
      uRing: { value: theme.surface.clone() },
    },
    vertexShader: `
      attribute vec3 aColor;
      attribute float aSize;
      uniform float uScale;
      uniform float uDpr;
      varying vec3 vColor;
      varying float vPx;
      void main() {
        vColor = aColor;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = clamp(aSize * uScale / -mv.z, 4.0 * uDpr, aSize * ${MAX_PX_PER_UNIT.toFixed(1)} * uDpr);
        vPx = gl_PointSize;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform vec3 uRing;
      uniform float uDpr;
      varying vec3 vColor;
      varying float vPx;
      void main() {
        float R = vPx * 0.5;
        float r = length(gl_PointCoord * 2.0 - 1.0) * R;
        if (r > R) discard;
        float t = smoothstep(R - uDpr - 0.75, R - uDpr + 0.75, r);
        gl_FragColor = vec4(mix(vColor, uRing, t), 1.0);
      }`,
  }));
  points.renderOrder = 2;
  scene.add(points);

  // Edges: one LineSegments in strongest-first order; the threshold slider only changes the draw range.
  const E = w.length;
  const lp = new Float32Array(E * 6);
  for (let r = 0; r < E; r++) {
    lp.set(pos.subarray(src[r] * 3, src[r] * 3 + 3), r * 6);
    lp.set(pos.subarray(dst[r] * 3, dst[r] * 3 + 3), r * 6 + 3);
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.BufferAttribute(lp, 3));
  lg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(E * 6), 3));
  lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, depthWrite: false }));
  lines.renderOrder = 0;
  scene.add(lines);

  // Edges of the selected node, drawn in primary ink above the rest.
  highlight = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: theme.primary, depthWrite: false }),
  );
  highlight.renderOrder = 1;
  scene.add(highlight);

  new ResizeObserver(resize).observe(stage);
  resize();

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    theme = readTheme();
    renderer.setClearColor(theme.surface);
    points.material.uniforms.uRing.value.copy(theme.surface);
    highlight.material.color.copy(theme.primary);
    recolor();
    renderLegend();
  });
}

function resize() {
  const { clientWidth: wpx, clientHeight: hpx } = stage;
  renderer.setSize(wpx, hpx, false);
  camera.aspect = wpx / hpx;
  camera.updateProjectionMatrix();
  const dpr = renderer.getPixelRatio();
  points.material.uniforms.uScale.value = (dpr * hpx) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  points.material.uniforms.uDpr.value = dpr;
}

function nodeColor(i) {
  if (state.breed >= 0) return labels[i] === state.breed ? theme.focus : theme.mutedPoint;
  return species[i] ? theme.dog : theme.cat;
}

// Recompute every node color/size and edge color from the current state.
function recolor() {
  const colors = points.geometry.attributes.aColor.array;
  const sel = state.selected;
  const near = new Set();
  if (sel >= 0) {
    near.add(sel);
    for (const { j } of neighborsOf(sel)) near.add(j);
  }

  const edgeColor = new Array(n);
  for (let i = 0; i < n; i++) {
    let c = nodeColor(i);
    let size = state.breed >= 0 ? (labels[i] === state.breed ? FOCUS_SIZE : POINT_SIZE * 0.8) : POINT_SIZE;
    if (sel >= 0) {
      if (i === sel) size = SELECTED_SIZE;
      else if (near.has(i)) size = FOCUS_SIZE;
      else { c = mix(theme.surface, c, 0.3); size *= 0.85; }
    }
    c.toArray(colors, i * 3);
    baseSizes[i] = size;

    if (sel >= 0) edgeColor[i] = theme.dimEdge;
    else if (state.breed >= 0) edgeColor[i] = labels[i] === state.breed ? mix(theme.surface, theme.focus, 0.6) : theme.dimEdge;
    else edgeColor[i] = mix(theme.surface, c, theme.edgeMix);
  }
  points.geometry.attributes.aColor.needsUpdate = true;
  applySizes();

  const lc = lines.geometry.attributes.color.array;
  for (let r = 0; r < w.length; r++) {
    edgeColor[src[r]].toArray(lc, r * 6);
    edgeColor[dst[r]].toArray(lc, r * 6 + 3);
  }
  lines.geometry.attributes.color.needsUpdate = true;
  lines.geometry.setDrawRange(0, visibleCount(state.threshold) * 2);
  lines.visible = state.showEdges;

  const hp = [];
  if (sel >= 0) {
    for (const { j } of neighborsOf(sel)) hp.push(...pos.subarray(sel * 3, sel * 3 + 3), ...pos.subarray(j * 3, j * 3 + 3));
  }
  highlight.geometry.dispose();
  highlight.geometry = new THREE.BufferGeometry();
  highlight.geometry.setAttribute("position", new THREE.Float32BufferAttribute(hp, 3));
  highlight.visible = state.showEdges;
}

function applySizes() {
  const sizes = points.geometry.attributes.aSize.array;
  sizes.set(baseSizes);
  if (state.hovered >= 0) sizes[state.hovered] = Math.max(baseSizes[state.hovered] * 1.5, FOCUS_SIZE);
  points.geometry.attributes.aSize.needsUpdate = true;
}

function animate(now) {
  if (fly) {
    const k = Math.min(1, (now - fly.t0) / FLY_MS);
    const e = 1 - Math.pow(1 - k, 3);
    controls.target.lerpVectors(fly.from, fly.to, e);
    camera.position.lerpVectors(fly.camFrom, fly.camTo, e);
    if (k === 1) fly = null;
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function flyTo(i) {
  const to = new THREE.Vector3().fromArray(pos, i * 3);
  const offset = camera.position.clone().sub(controls.target);
  offset.setLength(Math.min(offset.length(), 90));
  fly = {
    t0: performance.now(),
    from: controls.target.clone(),
    to,
    camFrom: camera.position.clone(),
    camTo: to.clone().add(offset),
  };
}

// ---------- picking ----------

const projMatrix = new THREE.Matrix4();

// Nearest node to the pointer in screen space, within HOVER_RADIUS; ties go to the node nearer the camera.
function pick(px, py) {
  const rect = renderer.domElement.getBoundingClientRect();
  camera.updateMatrixWorld();
  const m = projMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
  const maxD = HOVER_RADIUS * HOVER_RADIUS;
  let best = -1, bestD = Infinity, bestW = Infinity;
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (cw <= 0) continue;
    const sx = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * 0.5 + 0.5) * rect.width;
    const sy = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * 0.5) * rect.height;
    const d = (sx - px) ** 2 + (sy - py) ** 2;
    if (d > maxD) continue;
    if (d < bestD - 4 || (d <= bestD + 4 && cw < bestW)) { best = i; bestD = d; bestW = cw; }
  }
  return best;
}

function setHovered(i, ev) {
  if (i !== state.hovered) {
    state.hovered = i;
    applySizes();
  }
  renderer.domElement.style.cursor = i >= 0 ? "pointer" : "";
  if (i < 0) { tooltip.style.display = "none"; return; }

  tooltip.querySelector("img").src = thumb(i);
  tooltip.querySelector(".name").textContent = classes[labels[i]];
  const sub = tooltip.querySelector(".sub");
  const link = state.selected >= 0 && i !== state.selected && neighborsOf(state.selected).find((nb) => nb.j === i);
  sub.textContent = link
    ? `cosine ${link.w.toFixed(3)} to selected`
    : `${classSpecies[labels[i]]} · ${neighborsOf(i).length} links`;

  tooltip.style.display = "block";
  const rect = stage.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  tooltip.style.left = `${x + 16 + tw > rect.width ? x - tw - 16 : x + 16}px`;
  tooltip.style.top = `${Math.min(Math.max(8, y - th / 2), rect.height - th - 8)}px`;
}

function select(i, { fly: shouldFly = true } = {}) {
  state.selected = i;
  recolor();
  renderPanel();
  if (i >= 0 && shouldFly) flyTo(i);
}

// ---------- UI ----------

function initControls() {
  const canvas = renderer.domElement;
  let down = null;
  canvas.addEventListener("pointermove", (ev) => {
    if (ev.buttons) { setHovered(-1, ev); return; }
    const rect = canvas.getBoundingClientRect();
    setHovered(pick(ev.clientX - rect.left, ev.clientY - rect.top), ev);
  });
  canvas.addEventListener("pointerleave", (ev) => setHovered(-1, ev));
  canvas.addEventListener("pointerdown", (ev) => { down = [ev.clientX, ev.clientY]; });
  canvas.addEventListener("pointerup", (ev) => {
    if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) return;
    const rect = canvas.getBoundingClientRect();
    select(pick(ev.clientX - rect.left, ev.clientY - rect.top));
  });
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && state.selected >= 0) select(-1);
  });

  const order = classes.map((_, k) => k).sort((a, b) => classes[a].localeCompare(classes[b]));
  const counts = new Array(classes.length).fill(0);
  labels.forEach((l) => counts[l]++);
  for (const k of order) {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = `${classes[k]} (${classSpecies[k]}, ${counts[k]})`;
    breedSelect.append(opt);
  }
  breedSelect.addEventListener("change", () => {
    state.breed = Number(breedSelect.value);
    recolor();
    renderLegend();
    renderPanel();
  });

  thresholdInput.min = params.threshold;
  thresholdInput.max = Math.max(params.threshold + 0.005, Math.ceil((w[0] ?? 1) * 200) / 200);
  thresholdInput.value = params.threshold;
  thresholdInput.addEventListener("input", () => {
    state.threshold = Number(thresholdInput.value);
    recolor();
    renderPanel();
    renderStats();
  });

  $("showEdges").addEventListener("change", (ev) => {
    state.showEdges = ev.target.checked;
    lines.visible = highlight.visible = state.showEdges;
  });
  $("rotate").addEventListener("change", (ev) => { controls.autoRotate = ev.target.checked; });

  renderLegend();
  renderStats();
}

function renderStats() {
  $("thresholdOut").value = state.threshold.toFixed(3);
  $("stats").textContent = `${fmt(n)} images · ${fmt(visibleCount(state.threshold))} edges`;
}

function legendRow(color, text) {
  const row = document.createElement("div");
  const dot = document.createElement("span");
  dot.className = "dot";
  dot.style.background = `#${color.getHexString()}`;
  row.append(dot, document.createTextNode(text));
  return row;
}

function renderLegend() {
  const legend = $("legend");
  legend.replaceChildren();
  if (state.breed >= 0) {
    const count = labels.filter((l) => l === state.breed).length;
    legend.append(
      legendRow(theme.focus, `${classes[state.breed]} (${fmt(count)})`),
      legendRow(theme.mutedPoint, "Other images"),
    );
  } else {
    const cats = species.filter((s) => s === 0).length;
    legend.append(legendRow(theme.cat, `Cats (${fmt(cats)})`), legendRow(theme.dog, `Dogs (${fmt(n - cats)})`));
  }
}

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function dl(rows) {
  return el("dl", {}, ...rows.flatMap(([k, v]) => [el("dt", { textContent: k }), el("dd", { textContent: v })]));
}

function renderPanel() {
  panel.replaceChildren(state.selected >= 0 ? selectionView(state.selected) : overviewView());
}

function overviewView() {
  const count = visibleCount(state.threshold);
  const degree = new Int32Array(n);
  for (let r = 0; r < count; r++) { degree[src[r]]++; degree[dst[r]]++; }
  const isolated = degree.reduce((acc, d) => acc + (d === 0 ? 1 : 0), 0);
  const same = count ? sameBreedPrefix[count] / count : 0;

  return el("div", {},
    el("h2", { textContent: "Overview" }),
    el("div", { className: "sub", textContent: "Each dot is a photo, placed by UMAP of its CLIP embedding. Lines join pairs whose cosine similarity is at least the slider value." }),
    dl([
      ["Images", fmt(n)],
      ["Edges", fmt(count)],
      ["Mean links per image", (2 * count / n).toFixed(1)],
      ["Images with no links", fmt(isolated)],
      ["Same-breed edges", `${(same * 100).toFixed(1)}%`],
      ["Model", params.model],
      ["Build threshold", `> ${params.threshold}${params.center ? " (mean-centered)" : ""}`],
      ["Max proposals per image", params.maxDegree > 0 ? String(params.maxDegree) : "no cap"],
    ]),
    el("div", { className: "muted", textContent: "Click an image to list its most similar neighbors. Esc clears the selection." }),
  );
}

function selectionView(i) {
  const nbs = neighborsOf(i);
  const list = el("ul", { id: "neighbors" });
  for (const { j, w: wj } of nbs) {
    const li = el("li", { tabIndex: 0 },
      el("img", { src: thumb(j), alt: "", loading: "lazy" }),
      el("div", {},
        el("div", { className: "val", textContent: wj.toFixed(3) }),
        el("div", { className: "lbl", textContent: classes[labels[j]] + (labels[j] === labels[i] ? " · same breed" : "") }),
      ),
    );
    const go = () => select(j);
    li.addEventListener("click", go);
    li.addEventListener("keydown", (ev) => { if (ev.key === "Enter") go(); });
    list.append(li);
  }

  const clear = el("button", { className: "link", textContent: "Clear" });
  clear.addEventListener("click", () => select(-1));

  return el("div", {},
    el("div", { className: "panel-head" }, el("h2", { textContent: classes[labels[i]] }), clear),
    el("div", { className: "sub", textContent: `${classSpecies[labels[i]]} · image #${i}` }),
    el("img", { id: "selImg", src: thumb(i), alt: classes[labels[i]] }),
    el("div", { className: "sub", textContent: nbs.length
      ? `${nbs.length} neighbor${nbs.length === 1 ? "" : "s"} with cosine ≥ ${state.threshold.toFixed(3)}`
      : `No neighbors with cosine ≥ ${state.threshold.toFixed(3)}` }),
    list,
  );
}
