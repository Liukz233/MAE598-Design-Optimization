// 3D comparison of the initial and optimized beams. Everything shown is read
// from results.json, which run_all.py writes. The analysis covers half the beam;
// the other half is its mirror image about midspan (x = 0).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const root = document.getElementById("viewer");
const R = await (await fetch("assets/results.json")).json();
const C = R.config;
const nx = C.nx, tx = C.tiles_x, ty = C.tiles_y, rf = C.refine, a = C.element_mm;
const L = C.half_span_mm, H = C.depth_mm;
const limits = { stress: C.sigma_allow_MPa, deflection: C.deflection_limit_mm };

// ---- colour scale: one hue, light to dark -----------------------------------
const stops = ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"].map((h) => new THREE.Color(h));
const rgb = stops.map((c) => { const o = {}; c.getRGB(o, THREE.SRGBColorSpace); return o; });
const allU = [R.designs.initial, R.designs.optimized, R.designs.linear].flatMap((d) => d.utilization);
const Umax = Math.max(1.2, Math.ceil(Math.max(...allU) * 10) / 10);
function colorOf(u) {
  const s = Math.min(Math.max(u / Umax, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(s), stops.length - 2), f = s - i;
  return [rgb[i].r + f * (rgb[i + 1].r - rgb[i].r), rgb[i].g + f * (rgb[i + 1].g - rgb[i].g),
          rgb[i].b + f * (rgb[i + 1].b - rgb[i].b)];
}

// ---- page structure -----------------------------------------------------------
root.innerHTML = `
  <div class="v-controls">
    <label>Right panel
      <select id="v-design">
        <option value="optimized">Optimized, nonlinear material</option>
        <option value="linear">Optimized assuming a linear material</option>
      </select></label>
    <button id="v-play" type="button">▶ Play optimization history</button>
    <label class="v-grow">Iteration <span id="v-iter"></span>
      <input id="v-frame" type="range" min="0" max="${R.frames.length - 1}" value="${R.frames.length - 1}"></label>
    <label>Deformation ×<span id="v-slabel">40</span>
      <input id="v-s" type="range" min="0" max="100" step="5" value="40"></label>
    <button id="v-reset" type="button">Reset view</button>
  </div>
  <div class="v-panels">
    <div class="v-panel"><div class="v-title" id="v-title-0"></div><div class="v-host" id="v-host-0"></div><div class="v-metrics" id="v-metrics-0"></div></div>
    <div class="v-panel"><div class="v-title" id="v-title-1"></div><div class="v-host" id="v-host-1"></div><div class="v-metrics" id="v-metrics-1"></div></div>
  </div>
  <div class="v-legend">
    <div class="v-bar-wrap"><div class="v-bar" id="v-bar"></div><div class="v-ticks" id="v-ticks"></div></div>
    <div class="v-key">Colour: von Mises stress of each tile divided by the allowable ${limits.stress.toFixed(0)} MPa.
      <b>×</b> marks tiles above the allowable. Dashed line under midspan: the deflection limit, drawn at the same
      exaggeration as the deformation. Hover or tap a tile to inspect it.</div>
  </div>
  <div class="v-tip" id="v-tip" hidden></div>`;
const $ = (id) => document.getElementById(id);

$("v-bar").style.background = `linear-gradient(to right, ${stops.map((c) => "#" + c.getHexString()).join(",")})`;
$("v-ticks").innerHTML = [[0, "0"], [0.5, "0.5"], [1, "1.0 (allowable)"]].map(([u, label]) =>
  `<span class="v-tick" style="left:${(100 * u) / Umax}%">${label}</span>`).join("");

// ---- geometry of one design -----------------------------------------------------
const node = (i, j) => i + j * (nx + 1);          // rows from the bottom of the beam
const tileAt = (t, i, j) => (j < 0 || j >= ty || i >= tx ? 0 : t[Math.max(i, 0) + j * tx]);  // i < 0: mirror image

function build(design, scale) {
  const t = design.t_mm, U = design.utilization;
  const ux = design.ux_mm, uy = design.uy_mm;
  const P = (i, j) => [i * a + scale * ux[node(i, j)], j * a + scale * uy[node(i, j)]];  // deformed node
  const pos = [], col = [], faceTile = [], edge = [], cross = [];
  for (const side of [1, -1]) {                   // the half that was analysed, then its mirror image
    const tri = (p, q, r, z0, z1, z2, c, shade, tile) => {
      pos.push(side * p[0], p[1], z0, side * q[0], q[1], z1, side * r[0], r[1], z2);
      for (let k = 0; k < 3; k++) col.push(c[0] * shade, c[1] * shade, c[2] * shade);
      faceTile.push(side * (tile + 1));
    };
    for (let j = 0; j < ty; j++) for (let i = 0; i < tx; i++) {
      const tile = i + j * tx, z = t[tile] / 2, c = colorOf(U[tile]);
      // front and back faces: four triangles per element around its centre (bilinear value there)
      for (let ej = j * rf; ej < (j + 1) * rf; ej++) for (let ei = i * rf; ei < (i + 1) * rf; ei++) {
        const q = [P(ei, ej), P(ei + 1, ej), P(ei + 1, ej + 1), P(ei, ej + 1)];
        const m = [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4];
        for (let k = 0; k < 4; k++) for (const zz of [z, -z]) tri(q[k], q[(k + 1) % 4], m, zz, zz, zz, c, 1, tile);
      }
      // walls where this tile is thicker than its neighbour, or on the outside of the beam
      const sides = [[i - 1, j, i * rf, j * rf, 0, 1, 0.8], [i + 1, j, (i + 1) * rf, j * rf, 0, 1, 0.8],
                     [i, j - 1, i * rf, j * rf, 1, 0, 0.66], [i, j + 1, i * rf, (j + 1) * rf, 1, 0, 0.66]];
      for (const [ni, nj, i0, j0, di, dj, shade] of sides) {
        const zn = tileAt(t, ni, nj) / 2;
        if (z <= zn + 1e-9) continue;
        for (let s = 0; s < rf; s++) {
          const p = P(i0 + di * s, j0 + dj * s), q = P(i0 + di * (s + 1), j0 + dj * (s + 1));
          for (const [zl, zh] of zn > 0 ? [[zn, z], [-z, -zn]] : [[-z, z]]) {
            tri(p, q, q, zl, zl, zh, c, shade, tile);
            tri(p, q, p, zl, zh, zh, c, shade, tile);
          }
          edge.push(side * p[0], p[1], z, side * q[0], q[1], z);
        }
      }
      if (U[tile] > 1) {                          // over the allowable: a cross that does not rely on colour
        const p = P(i * rf, j * rf), q = P((i + 1) * rf, (j + 1) * rf), r = P((i + 1) * rf, j * rf), s = P(i * rf, (j + 1) * rf);
        const k = 0.25, zc = z + 0.4, mix = (u, v) => [u[0] + k * (v[0] - u[0]), u[1] + k * (v[1] - u[1])];
        for (const [u, v] of [[mix(p, q), mix(q, p)], [mix(r, s), mix(s, r)]]) cross.push(side * u[0], u[1], zc, side * v[0], v[1], zc);
      }
    }
  }
  const zl = C.t_max_mm / 2 + 1;                  // in front of the plate, so the beam does not hide it
  const limitLine = [-120, -scale * limits.deflection, zl, 120, -scale * limits.deflection, zl];
  return { pos, col, faceTile, edge, cross, limitLine };
}

// ---- one renderer per panel, one shared camera ----------------------------------
const camera = new THREE.PerspectiveCamera(30, 1.6, 10, 20000);
const target = new THREE.Vector3(0, H / 2 - 30, 0);
const home = new THREE.Vector3(430, 340, 1480);

function makeLines(material) {
  const obj = new THREE.LineSegments(new THREE.BufferGeometry(), material);
  obj.frustumCulled = false;
  return obj;
}
function fixtures(scene) {
  // load at midspan and the two supports, drawn at their undeformed positions
  scene.add(new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, H + 95, 0), 85, 0x1c4fd6, 30, 18));
  const mat = new THREE.MeshBasicMaterial({ color: 0x555555 });
  for (const x of [-(L - C.pad_mm / 2), L - C.pad_mm / 2]) {
    const cone = new THREE.Mesh(new THREE.ConeGeometry(18, 30, 4), mat);
    cone.position.set(x, -17, 0);
    scene.add(cone);
  }
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
  const edges = makeLines(new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }));
  const cross = makeLines(new THREE.LineBasicMaterial({ color: 0x000000 }));
  const limit = makeLines(new THREE.LineDashedMaterial({ color: 0x111111, dashSize: 8, gapSize: 5 }));
  const pick = makeLines(new THREE.LineBasicMaterial({ color: 0x0b5fff }));
  pick.visible = false;
  scene.add(mesh, edges, cross, limit, pick);
  fixtures(scene);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.target.copy(target);
  controls.addEventListener("change", render);
  return { host, renderer, scene, mesh, edges, cross, limit, pick, controls, design: null, faceTile: [] };
});

function render() {
  for (const p of panels) p.renderer.render(p.scene, camera);
}
function resize() {
  const w = panels[0].host.clientWidth, h = Math.round(Math.min(Math.max(w * 0.6, 220), 420));
  for (const p of panels) p.renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  render();
}
function resetView() {
  // move back on narrow panels so that the whole span stays in view
  camera.position.copy(home).sub(target).multiplyScalar(Math.max(1, 1.8 / camera.aspect)).add(target);
  for (const p of panels) p.controls.update();
  render();
}

// ---- showing a design -------------------------------------------------------------
let scale = 40;
const setLines = (obj, array) =>
  obj.geometry.setAttribute("position", new THREE.Float32BufferAttribute(array, 3));

function show(k, design, title) {
  const p = panels[k], g = build(design, scale), m = design.metrics;
  p.design = design;
  p.faceTile = g.faceTile;
  p.mesh.geometry.dispose();
  p.mesh.geometry = new THREE.BufferGeometry();
  p.mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(g.pos, 3));
  const lin = new Float32Array(g.col.length);     // colours are given in sRGB; three.js works in linear RGB
  for (let i = 0; i < lin.length; i++) {
    const c = g.col[i];
    lin[i] = c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }
  p.mesh.geometry.setAttribute("color", new THREE.BufferAttribute(lin, 3));
  setLines(p.edges, g.edge); setLines(p.cross, g.cross);
  setLines(p.limit, g.limitLine); p.limit.computeLineDistances();
  p.pick.visible = false;
  $(`v-title-${k}`).textContent = title;
  const row = (name, value, unit, limit, digits) => {
    const ok = value <= limit * (1 + 1e-6);
    return `<tr class="${ok ? "ok" : "bad"}"><td>${name}</td><td>${value.toFixed(digits)} ${unit}</td>
      <td>limit ${limit.toFixed(digits)} ${unit}</td><td>${ok ? "✓ pass" : "✗ FAIL"}</td></tr>`;
  };
  $(`v-metrics-${k}`).innerHTML = `<table><tr><td>Mass (objective)</td><td>${m.mass_kg.toFixed(2)} kg</td><td></td><td></td></tr>
    ${row("Midspan deflection", m.deflection_mm, "mm", limits.deflection, 2)}
    ${row("Peak tile stress", m.stress_MPa, "MPa", limits.stress, 0)}</table>`;
}

const lastFrame = R.frames.length - 1;
let frame = lastFrame, timer = null;
function showRight() {
  if (frame < lastFrame) {
    const f = R.frames[frame];
    show(1, f, `Iteration ${f.iteration} of ${R.frames[lastFrame].iteration}`);
  } else {
    const d = R.designs[$("v-design").value];
    show(1, d, d.label);
  }
  $("v-iter").textContent = R.frames[frame].iteration;
  render();
}
function showAll() {
  show(0, R.designs.initial, R.designs.initial.label);
  showRight();
}

$("v-design").addEventListener("change", () => { stop(); frame = lastFrame; $("v-frame").value = frame; showRight(); });
$("v-frame").addEventListener("input", (e) => { stop(); $("v-design").value = "optimized"; frame = +e.target.value; showRight(); });
$("v-s").addEventListener("input", (e) => { scale = +e.target.value; $("v-slabel").textContent = scale; showAll(); });
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
  }, 180);
});

// ---- inspection on hover or tap ----------------------------------------------------
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = $("v-tip");
function outline(p, tile, side) {
  const d = p.design, i = tile % tx, j = Math.floor(tile / tx), z = d.t_mm[tile] / 2 + 0.6;
  const at = (ii, jj) => [side * (ii * a + scale * d.ux_mm[node(ii, jj)]), jj * a + scale * d.uy_mm[node(ii, jj)], z];
  const c = [at(i * rf, j * rf), at((i + 1) * rf, j * rf), at((i + 1) * rf, (j + 1) * rf), at(i * rf, (j + 1) * rf)];
  setLines(p.pick, [0, 1, 2, 3].flatMap((k) => [...c[k], ...c[(k + 1) % 4]]));
  p.pick.visible = true;
}
function inspect(k, ev) {
  const p = panels[k], rect = p.renderer.domElement.getBoundingClientRect();
  mouse.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObject(p.mesh)[0];
  for (const q of panels) q.pick.visible = false;
  if (!hit) { tip.hidden = true; return render(); }
  const id = p.faceTile[hit.faceIndex], side = Math.sign(id), tile = Math.abs(id) - 1;
  const i = tile % tx, j = Math.floor(tile / tx), d = p.design;
  const corners = [node(i * rf, j * rf), node((i + 1) * rf, j * rf), node((i + 1) * rf, (j + 1) * rf), node(i * rf, (j + 1) * rf)];
  const mean = (v) => corners.reduce((s, n) => s + v[n], 0) / 4;
  const u = d.utilization[tile];
  tip.innerHTML = `<b>Tile (${i + 1}, ${j + 1})</b> at x = ${(side * (i + 0.5) * rf * a).toFixed(0)} mm, y = ${((j + 0.5) * rf * a).toFixed(0)} mm<br>
    Thickness ${d.t_mm[tile].toFixed(2)} mm<br>
    Von Mises stress ${(u * limits.stress).toFixed(0)} MPa, ${(100 * u).toFixed(0)}% of the allowable: <b>${u > 1 ? "over the limit" : "within the limit"}</b><br>
    Displacement ${(side * mean(d.ux_mm)).toFixed(2)} mm along the span, ${mean(d.uy_mm).toFixed(2)} mm vertical`;
  tip.hidden = false;
  const box = root.getBoundingClientRect();
  tip.style.left = `${Math.max(4, Math.min(ev.clientX - box.left + 14, box.width - tip.offsetWidth - 4))}px`;
  tip.style.top = `${ev.clientY - box.top + 14}px`;
  for (const q of panels) outline(q, tile, side);   // the same tile in both panels
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
