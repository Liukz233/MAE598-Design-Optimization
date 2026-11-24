// 3D comparison of the initial and optimized plates. Everything shown is read
// from results.json, which run_all.py writes.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const root = document.getElementById("viewer");
const R = await (await fetch("assets/results.json")).json();
const C = R.config;
const nx = C.nx, ny = C.ny, tx = C.tiles_x, ty = C.tiles_y, rf = C.refine;
const dx = C.Lx_mm / nx, dy = C.Ly_mm / ny;
const SIGMA = 5.670374419e-8;
const parts = [
  { key: "chip", name: "Power module", box: R.components.chip },
  { key: "optic", name: "Transceiver", box: R.components.optic },
];

// ---- colour scale: one hue, light to dark -----------------------------------
const stops = ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"].map((h) => new THREE.Color(h));
const allT = [R.designs.initial, R.designs.optimized, R.designs.linearized].flatMap((d) => d.T_C);
const Tlo = Math.floor(Math.min(...allT) / 5) * 5, Thi = Math.ceil(Math.max(...allT) / 5) * 5;
const rgb = stops.map((c) => { const o = {}; c.getRGB(o, THREE.SRGBColorSpace); return o; });  // sRGB values
function colorOf(T) {
  const s = Math.min(Math.max((T - Tlo) / (Thi - Tlo), 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(s), stops.length - 2), f = s - i;
  return { r: rgb[i].r + f * (rgb[i + 1].r - rgb[i].r), g: rgb[i].g + f * (rgb[i + 1].g - rgb[i].g),
           b: rgb[i].b + f * (rgb[i + 1].b - rgb[i].b) };
}

// ---- page structure -----------------------------------------------------------
root.innerHTML = `
  <div class="v-controls">
    <label>Right panel
      <select id="v-design">
        <option value="optimized">Optimized (nonlinear model)</option>
        <option value="linearized">Optimized with linearized radiation</option>
      </select></label>
    <button id="v-play" type="button">▶ Play optimization history</button>
    <label class="v-grow">Iteration <span id="v-iter"></span>
      <input id="v-frame" type="range" min="0" max="${R.frames.length - 1}" value="${R.frames.length - 1}"></label>
    <label>Thickness drawn ×<span id="v-zlabel">5</span>
      <input id="v-z" type="range" min="1" max="10" step="1" value="5"></label>
    <button id="v-reset" type="button">Reset view</button>
  </div>
  <div class="v-panels">
    <div class="v-panel"><div class="v-title" id="v-title-0"></div><div class="v-host" id="v-host-0"></div><div class="v-metrics" id="v-metrics-0"></div></div>
    <div class="v-panel"><div class="v-title" id="v-title-1"></div><div class="v-host" id="v-host-1"></div><div class="v-metrics" id="v-metrics-1"></div></div>
  </div>
  <div class="v-legend">
    <div class="v-bar-wrap"><div class="v-bar" id="v-bar"></div><div class="v-ticks" id="v-ticks"></div></div>
    <div class="v-key">Temperature of the plate, °C. Solid outline: power module. Dashed outline: transceiver.
      <b>×</b> marks nodes above the component's limit. Hover or tap a tile to inspect it.</div>
  </div>
  <div class="v-tip" id="v-tip" hidden></div>`;
const $ = (id) => document.getElementById(id);

$("v-bar").style.background = `linear-gradient(to right, ${stops.map((c) => "#" + c.getHexString()).join(",")})`;
const ticks = [[Tlo, `${Tlo}`], [Thi, `${Thi}`],
  ...parts.map((p) => [p.box.limit_C, `${p.box.limit_C} (${p.name.toLowerCase()} limit)`])];
$("v-ticks").innerHTML = ticks.map(([T, label]) =>
  `<span class="v-tick" style="left:${(100 * (T - Tlo)) / (Thi - Tlo)}%">${label}</span>`).join("");

// ---- geometry of one design -----------------------------------------------------
const node = (i, j) => i + j * (nx + 1);
function inBox(x, y, b) {
  return x >= 1e3 * b.x0 - 1e-6 && x <= 1e3 * b.x1 + 1e-6 && y >= 1e3 * b.y0 - 1e-6 && y <= 1e3 * b.y1 + 1e-6;
}

function build(design, zs) {
  const t = design.t_mm, T = design.T_C;
  const pos = [], col = [], edge = [], dashed = [], solid = [], cross = [];
  const push = (x, y, z, T_, shade) => {
    const c = colorOf(T_);
    pos.push(x, y, z);
    col.push(c.r * shade, c.g * shade, c.b * shade);
  };
  const tile = (a, b) => (a < 0 || b < 0 || a >= tx || b >= ty ? 0 : t[a + b * tx]);

  // top faces: four triangles per element around its centre value
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const z = zs * t[Math.floor(i / rf) + Math.floor(j / rf) * tx];
    const corner = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
    const Tc = corner.reduce((s, [a, b]) => s + T[node(a, b)], 0) / 4;
    for (let k = 0; k < 4; k++) {
      const [a0, b0] = corner[k], [a1, b1] = corner[(k + 1) % 4];
      push(a0 * dx, b0 * dy, z, T[node(a0, b0)], 1);
      push(a1 * dx, b1 * dy, z, T[node(a1, b1)], 1);
      push((i + 0.5) * dx, (j + 0.5) * dy, z, Tc, 1);
    }
  }
  // side walls where a tile stands above its neighbour (or the plate edge)
  const wall = (i0, j0, i1, j1, zLow, zHigh, shade) => {
    for (let s = 0; s < rf; s++) {
      const a0 = i0 + ((i1 - i0) * s) / rf, b0 = j0 + ((j1 - j0) * s) / rf;
      const a1 = i0 + ((i1 - i0) * (s + 1)) / rf, b1 = j0 + ((j1 - j0) * (s + 1)) / rf;
      const T0 = T[node(a0, b0)], T1 = T[node(a1, b1)];
      push(a0 * dx, b0 * dy, zLow, T0, shade); push(a1 * dx, b1 * dy, zLow, T1, shade); push(a1 * dx, b1 * dy, zHigh, T1, shade);
      push(a0 * dx, b0 * dy, zLow, T0, shade); push(a1 * dx, b1 * dy, zHigh, T1, shade); push(a0 * dx, b0 * dy, zHigh, T0, shade);
    }
    edge.push(i0 * dx, j0 * dy, zHigh, i1 * dx, j1 * dy, zHigh);
  };
  for (let b = 0; b < ty; b++) for (let a = 0; a < tx; a++) {
    const z = zs * tile(a, b), i0 = a * rf, i1 = (a + 1) * rf, j0 = b * rf, j1 = (b + 1) * rf;
    const sides = [[a - 1, b, i0, j0, i0, j1, 0.8], [a + 1, b, i1, j0, i1, j1, 0.8],
                   [a, b - 1, i0, j0, i1, j0, 0.66], [a, b + 1, i0, j1, i1, j1, 0.66]];
    for (const [na, nb, p, q, r, s, shade] of sides) {
      const zn = zs * tile(na, nb);
      if (z > zn + 1e-9) wall(p, q, r, s, zn, z, shade);
    }
  }
  // flat mounting face
  const g = 0.75, L = C.Lx_mm, W = C.Ly_mm;
  for (const [x, y] of [[0, 0], [L, W], [L, 0], [0, 0], [0, W], [L, W]]) { pos.push(x, y, 0); col.push(g, g, g); }

  // component outlines and over-limit marks
  for (const p of parts) {
    const b = p.box, x0 = 1e3 * b.x0, x1 = 1e3 * b.x1, y0 = 1e3 * b.y0, y1 = 1e3 * b.y1;
    let zTop = 0;
    for (let bj = 0; bj < ty; bj++) for (let a = 0; a < tx; a++) {
      const cx = (a + 0.5) * rf * dx, cy = (bj + 0.5) * rf * dy;
      if (inBox(cx, cy, b)) zTop = Math.max(zTop, zs * tile(a, bj));
    }
    const z = zTop + 0.5;
    (p.key === "chip" ? solid : dashed).push(x0, y0, z, x1, y0, z, x1, y0, z, x1, y1, z, x1, y1, z, x0, y1, z, x0, y1, z, x0, y0, z);
    for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
      const x = i * dx, y = j * dy;
      if (inBox(x, y, b) && T[node(i, j)] > b.limit_C) {
        const r = 1.6;
        cross.push(x - r, y - r, z + 0.3, x + r, y + r, z + 0.3, x - r, y + r, z + 0.3, x + r, y - r, z + 0.3);
      }
    }
  }
  return { pos, col, edge, dashed, solid, cross };
}

// ---- one renderer per panel, one shared camera ----------------------------------
const camera = new THREE.PerspectiveCamera(32, 1.6, 1, 5000);
camera.up.set(0, 0, 1);
const target = new THREE.Vector3(C.Lx_mm / 2, C.Ly_mm / 2, 0);
const home = new THREE.Vector3(C.Lx_mm / 2 + 30, -215, 290);

function makeLines(material) {
  const obj = new THREE.LineSegments(new THREE.BufferGeometry(), material);
  obj.frustumCulled = false;
  return obj;
}
const panels = [0, 1].map((k) => {
  const host = $(`v-host-${k}`);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf6f8fa);
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(),
    new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  const edges = makeLines(new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.3 }));
  const solid = makeLines(new THREE.LineBasicMaterial({ color: 0x111111 }));
  const dashed = makeLines(new THREE.LineDashedMaterial({ color: 0x111111, dashSize: 3, gapSize: 2 }));
  const cross = makeLines(new THREE.LineBasicMaterial({ color: 0x000000 }));
  const pick = makeLines(new THREE.LineBasicMaterial({ color: 0x0b5fff }));
  pick.visible = false;
  scene.add(mesh, edges, solid, dashed, cross, pick);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.target.copy(target);
  controls.addEventListener("change", render);
  return { host, renderer, scene, mesh, edges, solid, dashed, cross, pick, controls, design: null };
});

function render() {
  for (const p of panels) p.renderer.render(p.scene, camera);
}
function resize() {
  const w = panels[0].host.clientWidth, h = Math.round(Math.min(Math.max(w * 0.62, 230), 430));
  for (const p of panels) p.renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  render();
}
function resetView() {
  camera.position.copy(home);
  for (const p of panels) p.controls.update();
  render();
}

// ---- showing a design -------------------------------------------------------------
let zs = 5;
const setLines = (obj, array) =>
  obj.geometry.setAttribute("position", new THREE.Float32BufferAttribute(array, 3));

function show(k, design, title, m) {
  const p = panels[k], g = build(design, zs);
  p.design = design;
  p.mesh.geometry.dispose();
  p.mesh.geometry = new THREE.BufferGeometry();
  p.mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(g.pos, 3));
  // colours are given in sRGB; three.js works in linear RGB
  const lin = new Float32Array(g.col.length);
  for (let i = 0; i < lin.length; i++) {
    const c = g.col[i];
    lin[i] = c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }
  p.mesh.geometry.setAttribute("color", new THREE.BufferAttribute(lin, 3));
  setLines(p.edges, g.edge); setLines(p.solid, g.solid); setLines(p.cross, g.cross);
  setLines(p.dashed, g.dashed); p.dashed.computeLineDistances();
  p.pick.visible = false;
  $(`v-title-${k}`).textContent = title;
  const rows = parts.map((c) => {
    const T = m[`T_${c.key}_C`], ok = T <= c.box.limit_C + 1e-6;
    return `<tr class="${ok ? "ok" : "bad"}"><td>${c.name} peak</td><td>${T.toFixed(1)} °C</td>
      <td>limit ${c.box.limit_C} °C</td><td>${ok ? "✓ pass" : "✗ FAIL"}</td></tr>`;
  }).join("");
  $(`v-metrics-${k}`).innerHTML = `<table><tr><td>Mass (objective)</td><td>${m.mass_g.toFixed(1)} g</td><td></td><td></td></tr>${rows}</table>`;
}

const lastFrame = R.frames.length - 1;
let frame = lastFrame, timer = null;
function showRight() {
  const choice = $("v-design").value;
  if (frame < lastFrame) {
    const f = R.frames[frame];
    show(1, f, `Iteration ${f.iteration} of ${R.frames[lastFrame].iteration}`, f);
  } else {
    const d = R.designs[choice];
    show(1, d, d.label, d.metrics);
  }
  $("v-iter").textContent = R.frames[frame].iteration;
  render();
}
function showAll() {
  show(0, R.designs.initial, R.designs.initial.label, R.designs.initial.metrics);
  showRight();
}

$("v-design").addEventListener("change", () => { stop(); frame = lastFrame; $("v-frame").value = frame; showRight(); });
$("v-frame").addEventListener("input", (e) => { stop(); $("v-design").value = "optimized"; frame = +e.target.value; showRight(); });
$("v-z").addEventListener("input", (e) => { zs = +e.target.value; $("v-zlabel").textContent = zs; showAll(); });
$("v-reset").addEventListener("click", resetView);
function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  $("v-play").textContent = "▶ Play optimization history";
}
$("v-play").addEventListener("click", () => {
  if (timer) return stop();
  $("v-design").value = "optimized";
  frame = 0;
  $("v-play").textContent = "■ Stop";
  timer = setInterval(() => {
    $("v-frame").value = frame;
    showRight();
    if (frame >= lastFrame) stop(); else frame += 1;
  }, 160);
});

// ---- inspection on hover or tap ----------------------------------------------------
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = $("v-tip");
function inspect(k, ev) {
  const p = panels[k], rect = p.renderer.domElement.getBoundingClientRect();
  mouse.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObject(p.mesh)[0];
  for (const q of panels) q.pick.visible = false;
  if (!hit) { tip.hidden = true; return render(); }
  const x = Math.min(Math.max(hit.point.x, 0), C.Lx_mm - 1e-9), y = Math.min(Math.max(hit.point.y, 0), C.Ly_mm - 1e-9);
  const i = Math.floor(x / dx), j = Math.floor(y / dy), a = Math.floor(i / rf), b = Math.floor(j / rf);
  const u = x / dx - i, v = y / dy - j, T = p.design.T_C;
  // bilinear interpolation with the element's shape functions
  const Tp = (1 - u) * (1 - v) * T[node(i, j)] + u * (1 - v) * T[node(i + 1, j)] + u * v * T[node(i + 1, j + 1)] + (1 - u) * v * T[node(i, j + 1)];
  const TK = Tp + 273.15, TinfK = C.T_inf_C + 273.15;
  const conv = C.h * (TK - TinfK), rad = C.eps * SIGMA * (TK ** 4 - TinfK ** 4);
  let where = "";
  for (const c of parts) if (inBox(x, y, c.box)) {
    where = `<br>${c.name}, limit ${c.box.limit_C} °C: <b>${Tp > c.box.limit_C ? "over the limit" : "within the limit"}</b>`;
  }
  tip.innerHTML = `<b>Tile (${a + 1}, ${b + 1})</b> · thickness ${p.design.t_mm[a + b * tx].toFixed(2)} mm<br>
    Temperature ${Tp.toFixed(1)} °C at x = ${x.toFixed(0)} mm, y = ${y.toFixed(0)} mm<br>
    Surface heat loss ${(conv + rad).toFixed(0)} W/m² (convection ${conv.toFixed(0)}, radiation ${rad.toFixed(0)})${where}`;
  tip.hidden = false;
  const box = root.getBoundingClientRect();
  tip.style.left = `${Math.min(ev.clientX - box.left + 14, box.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${ev.clientY - box.top + 14}px`;
  // outline the same tile in both panels
  for (const q of panels) {
    const z = zs * q.design.t_mm[a + b * tx] + 0.4, w = rf * dx, hgt = rf * dy, x0 = a * w, y0 = b * hgt;
    setLines(q.pick, [x0, y0, z, x0 + w, y0, z, x0 + w, y0, z, x0 + w, y0 + hgt, z, x0 + w, y0 + hgt, z, x0, y0 + hgt, z, x0, y0 + hgt, z, x0, y0, z]);
    q.pick.visible = true;
  }
  render();
}
panels.forEach((p, k) => {
  const el = p.renderer.domElement;
  let down = null;
  el.addEventListener("pointermove", (ev) => { if (ev.pointerType === "mouse" && ev.buttons === 0) inspect(k, ev); });
  el.addEventListener("pointerdown", (ev) => { down = [ev.clientX, ev.clientY]; });
  el.addEventListener("pointerup", (ev) => {
    if (down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) < 6) inspect(k, ev);
    down = null;
  });
  el.addEventListener("pointerleave", (ev) => {
    if (ev.pointerType !== "mouse") return;
    tip.hidden = true;
    for (const q of panels) q.pick.visible = false;
    render();
  });
});

window.addEventListener("resize", resize);
resize();
showAll();
resetView();
root.dataset.ready = "1";
