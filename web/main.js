import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

// Colors go straight from the CSS tokens (sRGB) into buffers and shaders, so skip three's linear conversion.
THREE.ColorManagement.enabled = false;

// Node sizes are diameters in world units; the layout spans -100..100 along its longest axis.
const POINT_SIZE = 1.6;
const FOCUS_SIZE = 2.3;
const SELECTED_SIZE = 3.6;
const DOT_MAX_PX = 12;      // CSS px cap for a plain dot of POINT_SIZE
const PHOTO_MAX_PX = 110;   // CSS px cap once dots have turned into photos
const PHOTO_FROM_PX = 16;   // on-screen size where a dot starts turning into its photo...
const PHOTO_FULL_PX = 30;   // ...and where it is fully a photo
const ON_TOP = 1.001;       // aDim value that marks the selected node: full color, drawn in front of everything
const HOVER_RADIUS = 12;    // CSS px: the pointer only has to be this close to a small node
const FOV = 45;
const VIEW_DIR = new THREE.Vector3(0.62, 0.38, 0.69).normalize();
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const FLY_MS = REDUCED_MOTION ? 0 : 1000;

const $ = (id) => document.getElementById(id);
const ui = {
  stage: $("stage"), panel: $("panel"), toolbar: $("toolbar"), tooltip: $("tooltip"), legend: $("legend"),
  loading: $("loading"), subtitle: $("subtitle"), search: $("breedSearch"), breedList: $("breedList"),
  threshold: $("threshold"), thresholdOut: $("thresholdOut"),
  links: $("tLinks"), photos: $("tPhotos"), rotate: $("tRotate"), reset: $("resetView"),
  optsBtn: $("optsBtn"), legendInline: $("legendInline"),
  sheetHandle: $("sheetHandle"), grabber: $("grabber"), panelHead: $("panelHead"), panelBody: $("panelBody"),
};
// Matches the CSS breakpoint where the panel becomes a bottom sheet.
const phoneQuery = window.matchMedia("(max-width: 760px)");
const isPhone = () => phoneQuery.matches;

const state = { breed: -1, threshold: 0, links: true, photos: true, rotate: !REDUCED_MOTION, selected: -1, hovered: -1 };

let n, E, pos, labels, species, classes, classSpecies, params, atlas, allNodes, breedMembers;
let src, dst, w, sameBreedPrefix, adjacency;
let theme, renderer, scene, camera, controls, nodes, edges, highlight, halo;
let baseSizes, scaleCss = 1, photoMaxPx = PHOTO_MAX_PX, atlasReady = false, fly = null, home = null, sheetOpen = false;
const fog = { uFogNear: { value: 0 }, uFogFar: { value: 1 } };
let sceneRadius = 100;

const thumb = (i) => `thumbs/${i}.jpg`;
const fmt = (x) => x.toLocaleString();
const pct = (x) => `${(x * 100).toFixed(0)}%`;
const speciesName = (k) => (classSpecies[k] === "cat" ? "Cat" : "Dog");

main();

async function main() {
  let data;
  try {
    const res = await fetch("graph.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.statusText);
    data = await res.json();
  } catch {
    ui.loading.replaceChildren(el("div", {},
      "No ", el("code", { textContent: "graph.json" }), " yet. Run ",
      el("code", { textContent: "uv run python build.py" }), " and reload."));
    return;
  }
  loadData(data);
  theme = readTheme();
  initScene();
  initUI();
  applyTheme();
  loadAtlas();
  renderPanel();
  resize();

  home = homeView();
  controls.target.copy(home.target);
  camera.position.copy(home.target).addScaledVector(home.dir, home.distance * (REDUCED_MOTION ? 1 : 1.8));
  flyTo(home.target, home.distance, home.dir);
  requestAnimationFrame(animate);
  requestAnimationFrame(() => ui.loading.classList.add("done"));
}

// ---------- data ----------

function loadData(data) {
  params = data.params;
  classes = data.classes;
  classSpecies = data.classSpecies;
  atlas = data.atlas;
  labels = Int32Array.from(data.nodes.label);
  n = labels.length;
  pos = Float32Array.from(data.nodes.pos);
  // The layout's longest axis is x. On a portrait screen, turn it upright (90° about z) so it fills the height.
  if (window.innerHeight > window.innerWidth) {
    for (let i = 0; i < pos.length; i += 3) [pos[i], pos[i + 1]] = [-pos[i + 1], pos[i]];
  }
  species = labels.map((l) => (classSpecies[l] === "cat" ? 0 : 1));
  allNodes = Array.from({ length: n }, (_, i) => i);
  breedMembers = classes.map(() => []);
  labels.forEach((l, i) => breedMembers[l].push(i));
  for (let i = 0; i < n; i++) sceneRadius = Math.max(sceneRadius, Math.hypot(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]));

  // Strongest edges first, so "edges >= t" is always a prefix and the slider is just a draw range.
  E = data.edges.w.length;
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

// Number of edges with weight >= t (edges are sorted descending).
function visibleCount(t = state.threshold) {
  let lo = 0, hi = E;
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

function degreesAt() {
  const deg = new Int32Array(n);
  const count = visibleCount();
  for (let r = 0; r < count; r++) { deg[src[r]]++; deg[dst[r]]++; }
  return deg;
}

// ---------- theme ----------

function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const c = (name) => new THREE.Color(cs.getPropertyValue(name).trim());
  const t = {
    bg: c("--bg"),
    primary: c("--text-primary"),
    muted: c("--text-muted"),
    cat: c("--cat"),
    dog: c("--dog"),
    focus: c("--focus"),
    edgeAlpha: parseFloat(cs.getPropertyValue("--edge-alpha")),
    additive: cs.getPropertyValue("--edge-blend").trim() === "additive",
  };
  t.other = t.bg.clone().lerp(t.muted, 0.55); // how non-highlighted nodes end up drawn
  return t;
}

function applyTheme() {
  theme = readTheme();
  renderer.setClearColor(theme.bg);
  nodes.material.uniforms.uBg.value.copy(theme.bg);
  edges.material.blending = theme.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
  edges.material.needsUpdate = true;
  halo.material.uniforms.uColor.value.copy(theme.primary);
  recolor();
  renderLegend();
}

// ---------- scene ----------

function initScene() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  ui.stage.append(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute("aria-label", "3D similarity graph of pet photos");

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(FOV, 1, 0.5, 5000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.7;
  controls.minDistance = 6;
  controls.maxDistance = 1500;
  controls.autoRotateSpeed = 0.45;
  controls.autoRotate = state.rotate;
  controls.addEventListener("start", () => { fly = null; setRotate(false); hideHint(); });

  // Nodes: one Points object. Each sprite is a shaded dot that turns into its photo once it is big enough on screen.
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array(n), 1));
  g.setAttribute("aDim", new THREE.BufferAttribute(new Float32Array(n), 1));
  g.setAttribute("aTile", new THREE.BufferAttribute(new Float32Array(n * 2), 2));
  baseSizes = new Float32Array(n);
  nodes = new THREE.Points(g, new THREE.ShaderMaterial({
    uniforms: {
      ...fog,
      uScale: { value: 1 },
      uDpr: { value: 1 },
      uBaseSize: { value: POINT_SIZE },
      uDotMaxPx: { value: DOT_MAX_PX },
      uPhotoMaxPx: { value: PHOTO_MAX_PX },
      uPhotoFrom: { value: PHOTO_FROM_PX },
      uPhotoFull: { value: PHOTO_FULL_PX },
      uPhotosOn: { value: 0 },
      uBg: { value: new THREE.Color() },
      uAtlas: { value: null },
      uGrid: { value: new THREE.Vector2(1, 1) },
      uTile: { value: 64 },
    },
    vertexShader: /* glsl */ `
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aDim;
      attribute vec2 aTile;
      uniform float uScale;
      uniform float uDpr;
      uniform float uBaseSize;
      uniform float uDotMaxPx;
      uniform float uPhotoMaxPx;
      uniform float uPhotoFrom;
      uniform float uPhotoFull;
      uniform float uPhotosOn;
      uniform float uFogNear;
      uniform float uFogFar;
      varying vec3 vColor;
      varying float vPx;
      varying float vDim;
      varying float vFog;
      varying float vPhoto;
      varying vec2 vTile;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float depth = -mv.z;
        // Only fully visible nodes turn into photos; faded ones stay small dots so the highlighted photos stand out.
        float photoOk = uPhotosOn * step(0.75, aDim);
        float maxPx = mix(uDotMaxPx, uPhotoMaxPx, photoOk) * uDpr * aSize / uBaseSize;
        float px = clamp(aSize * uScale / depth, 4.0 * uDpr, maxPx);
        gl_PointSize = px;
        gl_Position = projectionMatrix * mv;
        // Faded nodes go to the back of the depth range so they never cover a highlighted photo;
        // the selected node (aDim > 1) goes to the front so nothing covers it.
        if (aDim < 0.75) gl_Position.z = 0.9999 * gl_Position.w;
        else if (aDim > 1.0) gl_Position.z = -0.9999 * gl_Position.w;
        vPx = px;
        vColor = aColor;
        vDim = aDim;
        vTile = aTile;
        vFog = 0.85 * smoothstep(uFogNear, uFogFar, depth);
        vPhoto = photoOk * smoothstep(uPhotoFrom * uDpr, uPhotoFull * uDpr, px);
      }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D uAtlas;
      uniform vec2 uGrid;
      uniform float uTile;
      uniform vec3 uBg;
      uniform float uDpr;
      varying vec3 vColor;
      varying float vPx;
      varying float vDim;
      varying float vFog;
      varying float vPhoto;
      varying vec2 vTile;
      float sdRoundBox(vec2 p, float r) {
        vec2 q = abs(p) - (1.0 - r);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
      }
      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float unit = 2.0 / vPx;  // one device pixel in sprite units
        float d = mix(length(p) - 1.0, sdRoundBox(p, 0.16), vPhoto);
        float alpha = 1.0 - smoothstep(-unit, 0.0, d);
        if (alpha <= 0.0) discard;

        vec3 shade = vColor * (1.12 - 0.32 * dot(p, p));
        vec2 c = clamp(gl_PointCoord, 0.5 / uTile, 1.0 - 0.5 / uTile);
        vec3 photo = texture2D(uAtlas, vec2((vTile.x + c.x) / uGrid.x, 1.0 - (vTile.y + c.y) / uGrid.y)).rgb;
        vec3 inner = mix(shade, photo, vPhoto);
        // Dots get a 1px background ring so overlapping dots stay separable; photos get a 2.5px identity-colored frame.
        vec3 rim = mix(uBg, vColor, vPhoto);
        float rimWidth = mix(1.0, 2.5, vPhoto) * uDpr * unit;
        vec3 col = mix(inner, rim, smoothstep(-rimWidth - unit, -rimWidth, d));
        col = mix(uBg, col, min(vDim, 1.0));
        col = mix(col, uBg, vFog);
        gl_FragColor = vec4(col, alpha);
      }`,
    alphaToCoverage: true,
  }));
  const tiles = g.attributes.aTile.array;
  const cols = atlas?.cols ?? 1;
  for (let i = 0; i < n; i++) { tiles[i * 2] = i % cols; tiles[i * 2 + 1] = Math.floor(i / cols); }
  nodes.renderOrder = 2;
  scene.add(nodes);

  // Edges: one LineSegments, strongest first; the threshold slider only changes the draw range.
  const lp = new Float32Array(E * 6);
  for (let r = 0; r < E; r++) {
    lp.set(pos.subarray(src[r] * 3, src[r] * 3 + 3), r * 6);
    lp.set(pos.subarray(dst[r] * 3, dst[r] * 3 + 3), r * 6 + 3);
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.BufferAttribute(lp, 3));
  lg.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(E * 8), 4));
  edges = new THREE.LineSegments(lg, new THREE.ShaderMaterial({
    uniforms: { ...fog },
    vertexShader: /* glsl */ `
      attribute vec4 aColor;
      uniform float uFogNear;
      uniform float uFogFar;
      varying vec4 vColor;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vColor = vec4(aColor.rgb, aColor.a * (1.0 - 0.85 * smoothstep(uFogNear, uFogFar, -mv.z)));
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      varying vec4 vColor;
      void main() { gl_FragColor = vColor; }`,
    transparent: true,
    depthWrite: false,
  }));
  edges.renderOrder = 0;
  scene.add(edges);

  // Links of the selected (or hovered) node, as 2px lines. Opaque and drawn before the nodes without writing depth,
  // so every node (including the selected photo) paints over its own links.
  highlight = new LineSegments2(new LineSegmentsGeometry(), new LineMaterial({ linewidth: 2, depthWrite: false }));
  highlight.renderOrder = 1;
  highlight.visible = false;
  scene.add(highlight);

  // Pulsing ring around the selected node, drawn on top of everything.
  const hg = new THREE.BufferGeometry();
  hg.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0], 3));
  halo = new THREE.Points(hg, new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color() }, uPx: { value: 30 }, uDpr: { value: 1 } },
    vertexShader: /* glsl */ `
      uniform float uPx;
      void main() {
        gl_PointSize = uPx;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uPx;
      uniform float uDpr;
      void main() {
        float r = length(gl_PointCoord * 2.0 - 1.0) * uPx * 0.5;  // device px from center
        float R = uPx * 0.5 - uDpr;
        float a = 1.0 - smoothstep(0.75 * uDpr, 1.75 * uDpr, abs(r - R));
        if (a <= 0.0) discard;
        gl_FragColor = vec4(uColor, a * 0.9);
      }`,
    transparent: true,
    depthTest: false,
  }));
  halo.renderOrder = 10;
  halo.visible = false;
  halo.frustumCulled = false;
  scene.add(halo);

  new ResizeObserver(resize).observe(ui.stage);
  new ResizeObserver(updateViewOffset).observe(ui.panel);
  new ResizeObserver(updateViewOffset).observe(ui.toolbar);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
}

function loadAtlas() {
  const max = renderer.capabilities.maxTextureSize;
  if (!atlas || atlas.cols * atlas.tile > max || atlas.rows * atlas.tile > max) {
    ui.photos.disabled = true;
    setPhotos(false);
    return;
  }
  new THREE.TextureLoader().load(atlas.url, (tex) => {
    tex.colorSpace = THREE.NoColorSpace;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    const u = nodes.material.uniforms;
    u.uAtlas.value = tex;
    u.uGrid.value.set(atlas.cols, atlas.rows);
    u.uTile.value = atlas.tile;
    atlasReady = true;
    setPhotos(state.photos);
  }, undefined, () => { ui.photos.disabled = true; });
}

function resize() {
  const W = ui.stage.clientWidth, H = ui.stage.clientHeight;
  const dpr = renderer.getPixelRatio();
  renderer.setSize(W, H, false);
  camera.aspect = W / H;
  scaleCss = H / (2 * Math.tan(THREE.MathUtils.degToRad(FOV / 2)));
  // On small screens, photos stop growing sooner so a handful don't fill the whole view.
  photoMaxPx = Math.round(Math.min(PHOTO_MAX_PX, Math.max(56, Math.min(W, H) * 0.16)));
  nodes.material.uniforms.uPhotoMaxPx.value = photoMaxPx;
  nodes.material.uniforms.uScale.value = dpr * scaleCss;
  nodes.material.uniforms.uDpr.value = dpr;
  halo.material.uniforms.uDpr.value = dpr;
  highlight.material.resolution.set(W, H);
  updateViewOffset();
}

// The unobstructed part of the canvas: below the toolbar, and left of the side panel or above the bottom sheet.
function freeArea() {
  const W = ui.stage.clientWidth, H = ui.stage.clientHeight;
  const p = ui.panel.getBoundingClientRect(), t = ui.toolbar.getBoundingClientRect();
  const panelOnRight = p.width < W * 0.6;
  const pad = isPhone() ? 12 : 24;
  const hintRoom = isPhone() ? 0 : 16;
  const left = pad, right = (panelOnRight ? p.left : W) - pad;
  const top = t.bottom + pad, bottom = (panelOnRight ? H : p.top) - pad - hintRoom;
  return { W, H, fw: right - left, fh: bottom - top, cx: (left + right) / 2, cy: (top + bottom) / 2 };
}

// Shift the projection so the camera's axis lands in the middle of the free area instead of under the panel.
function updateViewOffset() {
  if (!camera) return;
  const { W, H, cx, cy } = freeArea();
  camera.setViewOffset(W, H, W / 2 - cx, H / 2 - cy, W, H);
}

// Distance at which the given nodes fit the free area when viewed from `dir` toward `center`.
function fitDistance(indices, center, dir, quantile = 0.98) {
  const { H, fw, fh } = freeArea();
  const tanV = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
  const tx = (tanV * fw) / H, ty = (tanV * fh) / H;
  const forward = dir.clone().negate();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward);
  const need = indices.map((i) => {
    const x = pos[i * 3] - center.x, y = pos[i * 3 + 1] - center.y, z = pos[i * 3 + 2] - center.z;
    const u = Math.abs(x * right.x + y * right.y + z * right.z);
    const v = Math.abs(x * up.x + y * up.y + z * up.z);
    return x * dir.x + y * dir.y + z * dir.z + Math.max(u / tx, v / ty);
  }).sort((a, b) => a - b);
  return Math.max(15, need[Math.min(need.length - 1, Math.floor(quantile * need.length))] * 1.04);
}

function centroid(indices) {
  const c = new THREE.Vector3();
  for (const i of indices) c.x += pos[i * 3], c.y += pos[i * 3 + 1], c.z += pos[i * 3 + 2];
  return c.divideScalar(Math.max(1, indices.length));
}

function homeView() {
  const target = new THREE.Vector3();
  return { target, dir: VIEW_DIR.clone(), distance: fitDistance(allNodes, target, VIEW_DIR, 0.995) };
}

function currentDir() {
  return camera.position.clone().sub(controls.target).normalize();
}

// On-screen diameter in CSS px of a node of `size` and `dim` at `depth`, matching the vertex shader.
function displayPx(size, depth, dim = 1) {
  const cap = (photosActive() && dim >= 0.75 ? photoMaxPx : DOT_MAX_PX) * size / POINT_SIZE;
  return Math.min(Math.max(size * scaleCss / depth, 4), cap);
}

const photosActive = () => state.photos && atlasReady;

function nodeColor(i) {
  if (state.breed >= 0) return labels[i] === state.breed ? theme.focus : theme.muted;
  return species[i] ? theme.dog : theme.cat;
}

// Recompute node colors/sizes/dimming and edge colors from the current state.
function recolor() {
  const A = nodes.geometry.attributes;
  const sel = state.selected, focus = state.breed;
  const near = sel >= 0 ? new Set([sel, ...neighborsOf(sel).map((nb) => nb.j)]) : null;
  for (let i = 0; i < n; i++) {
    const inFocus = focus < 0 || labels[i] === focus;
    let size = focus < 0 ? POINT_SIZE : inFocus ? FOCUS_SIZE : POINT_SIZE * 0.85;
    let dim = inFocus ? 1 : 0.55;
    if (near) {
      if (i === sel) { size = SELECTED_SIZE; dim = ON_TOP; }
      else if (near.has(i)) { size = Math.max(size, FOCUS_SIZE); dim = 1; }
      else { dim = Math.min(dim, 0.3); size *= 0.9; }
    }
    nodeColor(i).toArray(A.aColor.array, i * 3);
    A.aDim.array[i] = dim;
    baseSizes[i] = size;
  }
  A.aColor.needsUpdate = true;
  A.aDim.needsUpdate = true;
  applySizes();

  const col = edges.geometry.attributes.aColor.array;
  const wmin = params.threshold, span = Math.max(1e-6, (w[0] ?? 1) - wmin);
  const setEnd = (o, node, strength) => {
    let c = species[node] ? theme.dog : theme.cat;
    let a = theme.edgeAlpha * strength;
    if (focus >= 0) {
      if (labels[node] === focus) { c = theme.focus; a = Math.min(1, a * 2); }
      else { c = theme.muted; a *= 0.3; }
    }
    if (near) a *= 0.3;
    col[o] = c.r; col[o + 1] = c.g; col[o + 2] = c.b; col[o + 3] = a;
  };
  for (let r = 0; r < E; r++) {
    const strength = 0.25 + 0.75 * (w[r] - wmin) / span; // stronger similarity, brighter link
    setEnd(r * 8, src[r], strength);
    setEnd(r * 8 + 4, dst[r], strength);
  }
  edges.geometry.attributes.aColor.needsUpdate = true;
  edges.geometry.setDrawRange(0, visibleCount() * 2);
  edges.visible = state.links;
  updateHighlight();
}

function applySizes() {
  const sizes = nodes.geometry.attributes.aSize.array;
  sizes.set(baseSizes);
  if (state.hovered >= 0) sizes[state.hovered] = baseSizes[state.hovered] * 1.35;
  nodes.geometry.attributes.aSize.needsUpdate = true;
}

function updateHighlight() {
  const i = state.selected >= 0 ? state.selected : state.hovered;
  const nbs = i >= 0 && state.links ? neighborsOf(i) : [];
  highlight.visible = nbs.length > 0;
  if (!nbs.length) return;
  const arr = new Float32Array(nbs.length * 6);
  nbs.forEach(({ j }, k) => {
    arr.set(pos.subarray(i * 3, i * 3 + 3), k * 6);
    arr.set(pos.subarray(j * 3, j * 3 + 3), k * 6 + 3);
  });
  highlight.geometry.dispose();
  highlight.geometry = new LineSegmentsGeometry().setPositions(arr);
  highlight.material.color.copy(theme.bg).lerp(theme.primary, state.selected >= 0 ? 0.8 : 0.45);
}

// ---------- animation ----------

function animate(now) {
  if (fly) stepFly(now);
  controls.update();
  camera.updateMatrixWorld();
  const dist = camera.position.distanceTo(controls.target);
  fog.uFogNear.value = Math.max(0, dist - sceneRadius * 0.25);
  fog.uFogFar.value = dist + sceneRadius * 1.3;
  updateHalo(now);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function updateHalo(now) {
  const i = state.selected;
  halo.visible = i >= 0;
  if (i < 0) return;
  halo.position.fromArray(pos, i * 3);
  const depth = -halo.position.clone().applyMatrix4(camera.matrixWorldInverse).z;
  const pulse = REDUCED_MOTION ? 0 : 3 * Math.sin(now / 320);
  halo.material.uniforms.uPx.value = (displayPx(SELECTED_SIZE, depth) + 14 + pulse) * renderer.getPixelRatio();
}

function flyTo(target, distance, dir = currentDir()) {
  const fromDir = currentDir();
  fly = {
    t0: performance.now(),
    fromTarget: controls.target.clone(),
    toTarget: target.clone(),
    fromDist: camera.position.distanceTo(controls.target),
    toDist: distance,
    fromDir,
    rot: new THREE.Quaternion().setFromUnitVectors(fromDir, dir.clone().normalize()),
  };
  if (!FLY_MS) stepFly(fly.t0 + 1);
}

function stepFly(now) {
  const k = FLY_MS ? Math.min(1, Math.max(0, (now - fly.t0) / FLY_MS)) : 1;
  const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
  controls.target.lerpVectors(fly.fromTarget, fly.toTarget, e);
  const q = new THREE.Quaternion().slerp(fly.rot, e);
  const dist = fly.fromDist * Math.pow(fly.toDist / fly.fromDist, e);
  camera.position.copy(controls.target).addScaledVector(fly.fromDir.clone().applyQuaternion(q), dist);
  if (k === 1) fly = null;
}

function flyHome() {
  home = homeView();
  flyTo(home.target, home.distance, home.dir);
}

// Distance at which a node of `size` shows at about `px` CSS px wide.
const distanceForPx = (size, px) => (size * scaleCss) / px;

function flyToNode(i) {
  flyTo(new THREE.Vector3().fromArray(pos, i * 3), distanceForPx(SELECTED_SIZE, 80));
}

function flyToBreed(k) {
  const members = breedMembers[k];
  const target = centroid(members);
  // Extra margin, since up close nodes are photos that extend well past their centers; and never so close
  // that the breed's photos get bigger than ~44px and pile on top of each other.
  flyTo(target, Math.max(fitDistance(members, target, currentDir(), 0.9) * 1.6, distanceForPx(FOCUS_SIZE, 44)));
}

// ---------- picking ----------

const projMatrix = new THREE.Matrix4();

// The node under the pointer: of the nodes whose drawn sprite covers it, the one nearest the camera;
// failing that, the closest node within `radius` CSS px.
function pick(px, py, radius = HOVER_RADIUS) {
  const W = ui.stage.clientWidth, H = ui.stage.clientHeight;
  camera.updateMatrixWorld();
  const m = projMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
  const sizes = nodes.geometry.attributes.aSize.array;
  const dims = nodes.geometry.attributes.aDim.array;
  let cover = -1, coverDepth = Infinity, near = -1, nearD = radius * radius;
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (cw <= 0) continue;
    const sx = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * 0.5 + 0.5) * W;
    const sy = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * 0.5) * H;
    const d = (sx - px) ** 2 + (sy - py) ** 2;
    const r = displayPx(sizes[i], cw, dims[i]) / 2;
    if (d <= r * r) {
      const drawnDepth = dims[i] < 0.75 ? cw + 1e6 : dims[i] > 1 ? -1 : cw; // matches the depth layering in the shader
      if (drawnDepth < coverDepth) { cover = i; coverDepth = drawnDepth; }
    } else if (d < nearD) {
      near = i; nearD = d;
    }
  }
  return cover >= 0 ? cover : near;
}

function setHovered(i, ev) {
  if (i !== state.hovered) {
    state.hovered = i;
    applySizes();
    if (state.selected < 0) updateHighlight();
  }
  renderer.domElement.style.cursor = i >= 0 ? "pointer" : "";
  const tip = ui.tooltip;
  if (i < 0) { tip.style.display = "none"; return; }

  tip.querySelector("img").src = thumb(i);
  tip.querySelector(".name").textContent = classes[labels[i]];
  const link = state.selected >= 0 && i !== state.selected && neighborsOf(state.selected).find((nb) => nb.j === i);
  const links = neighborsOf(i).length;
  tip.querySelector(".sub").textContent = link
    ? `cosine ${link.w.toFixed(3)} to selected`
    : `${speciesName(labels[i])} · ${links} link${links === 1 ? "" : "s"}`;

  tip.style.display = "flex";
  const W = ui.stage.clientWidth, H = ui.stage.clientHeight;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  const x = ev.clientX, y = ev.clientY;
  tip.style.left = `${x + 18 + tw > W - 8 ? x - tw - 18 : x + 18}px`;
  tip.style.top = `${Math.min(Math.max(8, y - th / 2), H - th - 8)}px`;
}

// ---------- state changes ----------

const hideHint = () => $("hint").classList.add("hidden");

function select(i) {
  hideHint();
  state.selected = i;
  recolor();
  renderPanel();
  if (i >= 0) {
    setRotate(false);
    flyToNode(i);
  }
}

function setBreed(k) {
  hideHint();
  state.breed = k;
  state.selected = -1;
  ui.search.value = k >= 0 ? classes[k] : "";
  recolor();
  renderLegend();
  renderPanel();
  if (k >= 0) {
    setRotate(false);
    if (isPhone()) {
      ui.search.blur();
      setSheet(false); // first, so the fly-in fits the breed into the full graph area
    }
    flyToBreed(k);
  }
}

function setSheet(open) {
  sheetOpen = open;
  ui.panel.classList.toggle("expanded", open);
  ui.grabber.setAttribute("aria-expanded", String(open));
  ui.grabber.setAttribute("aria-label", open ? "Hide details" : "Show details");
}

function setOptions(open) {
  ui.toolbar.classList.toggle("open", open);
  ui.optsBtn.setAttribute("aria-expanded", String(open));
}

function setRotate(on) {
  state.rotate = on;
  controls.autoRotate = on;
  ui.rotate.setAttribute("aria-pressed", String(on));
}

function setPhotos(on) {
  state.photos = on;
  ui.photos.setAttribute("aria-pressed", String(on && !ui.photos.disabled));
  const u = nodes.material.uniforms;
  u.uPhotosOn.value = photosActive() ? 1 : 0;
}

function setLinks(on) {
  state.links = on;
  ui.links.setAttribute("aria-pressed", String(on));
  edges.visible = on;
  updateHighlight();
}

// ---------- UI ----------

function initUI() {
  ui.subtitle.textContent = `${fmt(n)} photos · ${fmt(E)} links · CLIP ${params.model.split(" ")[0]}`;

  const canvas = renderer.domElement;
  let down = null;
  canvas.addEventListener("pointermove", (ev) => {
    if (ev.buttons) { setHovered(-1, ev); return; }
    setHovered(pick(ev.clientX, ev.clientY), ev);
  });
  canvas.addEventListener("pointerleave", (ev) => setHovered(-1, ev));
  canvas.addEventListener("pointerdown", (ev) => {
    down = [ev.clientX, ev.clientY];
    setOptions(false);
  });
  canvas.addEventListener("pointerup", (ev) => {
    const touch = ev.pointerType === "touch";
    if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > (touch ? 10 : 4)) return;
    select(pick(ev.clientX, ev.clientY, touch ? 24 : HOVER_RADIUS));
  });

  // Bottom sheet (phones): tap or swipe the handle/header to expand and collapse.
  let drag = null;
  ui.sheetHandle.addEventListener("pointerdown", (ev) => {
    if (!isPhone() || ev.target.closest(".close")) return;
    drag = ev.clientY;
  });
  ui.sheetHandle.addEventListener("pointerup", (ev) => {
    if (drag === null) return;
    const dy = ev.clientY - drag;
    drag = null;
    setSheet(dy < -24 ? true : dy > 24 ? false : !sheetOpen);
  });
  ui.sheetHandle.addEventListener("pointercancel", () => { drag = null; });
  ui.grabber.addEventListener("click", (ev) => { if (ev.detail === 0) setSheet(!sheetOpen); }); // keyboard only

  ui.optsBtn.addEventListener("click", () => setOptions(!ui.toolbar.classList.contains("open")));
  window.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || ev.target === ui.search) return;
    if (state.selected >= 0) select(-1);
    else if (state.breed >= 0) setBreed(-1);
  });

  for (const name of [...classes].sort((a, b) => a.localeCompare(b))) {
    ui.breedList.append(el("option", { value: name }));
  }
  const findBreed = (text) => {
    const q = text.trim().toLowerCase();
    if (!q) return -1;
    const idx = (pred) => classes.findIndex((c) => pred(c.toLowerCase()));
    const exact = idx((c) => c === q);
    if (exact >= 0) return exact;
    const prefix = idx((c) => c.startsWith(q));
    return prefix >= 0 ? prefix : idx((c) => c.includes(q));
  };
  ui.search.addEventListener("input", () => {
    const exact = classes.findIndex((c) => c.toLowerCase() === ui.search.value.trim().toLowerCase());
    if (exact >= 0 && exact !== state.breed) setBreed(exact);
    else if (!ui.search.value && state.breed >= 0) setBreed(-1);
  });
  ui.search.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      const k = findBreed(ui.search.value);
      if (k >= 0) setBreed(k);
    } else if (ev.key === "Escape") {
      setBreed(-1);
      ui.search.blur();
    }
  });

  ui.threshold.min = params.threshold;
  ui.threshold.max = Math.max(params.threshold + 0.005, Math.ceil((w[0] ?? 1) * 200) / 200);
  ui.threshold.value = params.threshold;
  ui.threshold.addEventListener("input", () => {
    state.threshold = Number(ui.threshold.value);
    renderThreshold();
    recolor();
    renderPanel();
  });
  renderThreshold();

  ui.links.addEventListener("click", () => setLinks(!state.links));
  ui.photos.addEventListener("click", () => setPhotos(!state.photos));
  ui.rotate.addEventListener("click", () => setRotate(!state.rotate));
  ui.reset.addEventListener("click", () => {
    if (state.selected >= 0 || state.breed >= 0) setBreed(-1);
    flyHome();
  });
  setRotate(state.rotate);
}

function renderThreshold() {
  ui.thresholdOut.value = state.threshold.toFixed(3);
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "style") Object.assign(node.style, v);
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  node.append(...children.filter((c) => c != null && c !== false));
  return node;
}

const swatch = (color) => el("span", { className: "dot", style: { background: `#${color.getHexString()}` } });
const speciesColor = (k) => (classSpecies[k] === "cat" ? theme.cat : theme.dog);

// The floating legend (desktop) and the one inside the toolbar (phones) show the same entries.
function renderLegend() {
  let items;
  if (state.breed >= 0) {
    items = [[theme.focus, `${classes[state.breed]} (${fmt(breedMembers[state.breed].length)})`], [theme.other, "Other photos"]];
  } else {
    const cats = species.filter((s) => s === 0).length;
    items = [[theme.cat, `Cats (${fmt(cats)})`], [theme.dog, `Dogs (${fmt(n - cats)})`]];
  }
  ui.legend.replaceChildren(...items.map(([c, text]) => el("div", {}, swatch(c), text)));
  ui.legendInline.replaceChildren(...items.map(([c, text]) => el("span", {}, swatch(c), text)));
}

// Each view is a head (title row; on phones also the collapsed bottom sheet) and a body.
function renderPanel() {
  const view = state.selected >= 0 ? selectionView(state.selected) : state.breed >= 0 ? breedView(state.breed) : overviewView();
  ui.panelHead.replaceChildren(view.head);
  ui.panelBody.replaceChildren(...view.body);
  ui.panel.scrollTop = 0;
  ui.panelBody.scrollTop = 0;
}

function head({ title, meta, onclose = null, thumbSrc = null, metaPeekOnly = false }) {
  let close = null;
  if (onclose) {
    close = el("button", { className: "icon-btn close", "aria-label": "Close", onclick: onclose });
    close.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="M2 2l8 8M10 2l-8 8"/></svg>`;
  }
  return el("div", { className: "head" },
    thumbSrc && el("img", { className: "head-thumb", src: thumbSrc, alt: "" }),
    el("div", { className: "head-text" },
      el("h2", { textContent: title }),
      el("div", { className: metaPeekOnly ? "meta peek-only" : "meta" }, ...meta)),
    close);
}

function statTiles(items) {
  return el("div", { className: "tiles" }, ...items.map(([v, k]) =>
    el("div", { className: "tile" }, el("div", { className: "v", textContent: v }), el("div", { className: "k", textContent: k }))));
}

function photoGrid(items) {
  return el("div", { className: "grid" }, ...items.map(({ i, badge, caption }) =>
    el("button", { onclick: () => select(i), "aria-label": `${classes[labels[i]]}${badge ? `, cosine ${badge}` : ""}` },
      el("div", { className: "ph" },
        el("img", { src: thumb(i), alt: "", loading: "lazy" }),
        badge && el("span", { className: "badge", textContent: badge })),
      el("div", { className: "cap", textContent: caption }))));
}

function overviewView() {
  const count = visibleCount();
  const deg = degreesAt();
  const isolated = deg.reduce((acc, d) => acc + (d === 0 ? 1 : 0), 0);
  const chips = (sp) => el("div", { className: "chips" }, ...classes
    .map((name, k) => ({ name, k }))
    .filter(({ k }) => classSpecies[k] === sp)
    .sort((a, b) => a.name.localeCompare(b.name))
    .map(({ name, k }) => el("button", { className: "chip", onclick: () => setBreed(k) },
      swatch(speciesColor(k)), name, el("span", { className: "count", textContent: fmt(breedMembers[k].length) }))));

  return {
    head: head({ title: "Overview", meta: [`${fmt(n)} photos · ${fmt(count)} links`], metaPeekOnly: true }),
    body: [
      el("p", { textContent: "Every point is a photo, placed by UMAP of its CLIP embedding, so similar photos sit close together. Links join pairs whose cosine similarity clears the slider. Zoom in and the points turn into the photos themselves." }),
      statTiles([
        [fmt(n), "photos"],
        [fmt(count), "links"],
        [count ? pct(sameBreedPrefix[count] / count) : "–", "of links stay within a breed"],
        [fmt(isolated), "photos with no link"],
      ]),
      el("h3", { textContent: "Cat breeds" }), chips("cat"),
      el("h3", { textContent: "Dog breeds" }), chips("dog"),
      el("div", { className: "foot", textContent: `${params.model} · built with cosine > ${params.threshold}, up to ${params.maxDegree || "∞"} links proposed per photo` }),
    ],
  };
}

function breedView(b) {
  const members = breedMembers[b];
  const deg = degreesAt();
  const count = visibleCount();
  let within = 0, across = 0;
  const other = new Map();
  for (let r = 0; r < count; r++) {
    const la = labels[src[r]], lb = labels[dst[r]];
    if (la === b && lb === b) within++;
    else if (la === b || lb === b) {
      across++;
      const o = la === b ? lb : la;
      other.set(o, (other.get(o) ?? 0) + 1);
    }
  }
  const unlinked = members.filter((i) => deg[i] === 0).length;
  const top = [...other].sort((x, y) => y[1] - x[1]).slice(0, 5);
  const best = [...members].sort((x, y) => deg[y] - deg[x]).slice(0, 9);

  return {
    head: head({ title: classes[b], meta: [swatch(speciesColor(b)), `${speciesName(b)} · ${fmt(members.length)} photos`], onclose: () => setBreed(-1) }),
    body: [
      statTiles([
        [within + across ? pct(within / (within + across)) : "–", "of its links stay in the breed"],
        [fmt(unlinked), "photos with no link"],
      ]),
      el("h3", { textContent: "Most often linked to" }),
      top.length
        ? el("div", { className: "rows" }, ...top.map(([k, c]) =>
            el("button", { onclick: () => setBreed(k) }, el("span", { textContent: classes[k] }), el("span", { className: "count", textContent: `${fmt(c)} link${c === 1 ? "" : "s"}` }))))
        : el("div", { className: "empty", textContent: "No links to other breeds at this threshold." }),
      el("h3", { textContent: "Best-connected photos" }),
      photoGrid(best.map((i) => ({ i, caption: `${deg[i]} link${deg[i] === 1 ? "" : "s"}` }))),
    ],
  };
}

function selectionView(i) {
  const nbs = neighborsOf(i);
  const k = labels[i];
  return {
    head: head({
      title: classes[k],
      meta: [swatch(speciesColor(k)), `${speciesName(k)} · photo #${i} · ${nbs.length} similar`],
      onclose: () => select(-1),
      thumbSrc: thumb(i),
    }),
    body: [
      el("img", { className: "hero", src: thumb(i), alt: `${classes[k]} photo` }),
      el("p", {}, el("button", { className: "link-btn", textContent: `Highlight all ${classes[k]} photos`, onclick: () => setBreed(k) })),
      el("h3", { textContent: nbs.length ? `${nbs.length} most similar` : "Most similar" }),
      nbs.length
        ? photoGrid(nbs.map(({ j, w: wj }) => ({ i: j, badge: wj.toFixed(3), caption: classes[labels[j]] })))
        : el("div", { className: "empty", textContent: `No photo is above ${state.threshold.toFixed(3)} cosine with this one. Lower the slider, or rebuild with a lower --threshold.` }),
    ],
  };
}
