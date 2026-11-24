// 3D comparison of the initial and optimized grippers. Everything shown is read
// from results.json, which run_all.py writes. The analysis covers the upper half
// of the gripper; the lower half is its mirror image about the centre line y = 0.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const root = document.getElementById("viewer");
const R = await (await fetch("assets/results.json")).json();
const C = R.config;
const nx = C.nx, ny = C.ny, a = C.element_mm, T = C.thickness_mm;
const L = C.length_mm, H = C.half_height_mm;
const limits = { stroke: C.stroke_max_mm, stress: C.sigma_allow_MPa, volume: C.volume_fraction };
const SHOWN = 0.05;                                // elements below this density are not drawn

// ---- colour scale: one hue, light to dark -----------------------------------
const stops = ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"].map((h) => new THREE.Color(h));
const rgb = stops.map((c) => { const o = {}; c.getRGB(o, THREE.SRGBColorSpace); return o; });
const Umax = 1.2;
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
        <option value="optimized">Optimized, large-displacement analysis</option>
        <option value="linear">Optimized with linear analysis</option>
      </select></label>
    <button id="v-play" type="button">▶ Play actuation</button>
    <label class="v-grow">Actuator force <span id="v-load"></span>%
      <input id="v-act" type="range" min="0" max="100" step="1" value="100"></label>
    <label>Optimizer iteration <span id="v-iter"></span>
      <input id="v-frame" type="range" min="0" max="${R.frames.length - 1}" value="${R.frames.length - 1}"></label>
    <button id="v-reset" type="button">Reset view</button>
  </div>
  <div class="v-panels">
    <div class="v-panel"><div class="v-title" id="v-title-0"></div><div class="v-host" id="v-host-0"></div><div class="v-metrics" id="v-metrics-0"></div></div>
    <div class="v-panel"><div class="v-title" id="v-title-1"></div><div class="v-host" id="v-host-1"></div><div class="v-metrics" id="v-metrics-1"></div></div>
  </div>
  <div class="v-legend">
    <div class="v-bar-wrap"><div class="v-bar" id="v-bar"></div><div class="v-ticks" id="v-ticks"></div></div>
    <div class="v-key">Colour: von Mises stress of each element at full force, divided by the allowable ${limits.stress.toFixed(1)} MPa.
      <b>×</b> marks elements above the allowable. The height of each element is its density times the plate thickness.
      Hover or tap an element to inspect it.</div>
  </div>
  <div class="v-tip" id="v-tip" hidden></div>`;
const $ = (id) => document.getElementById(id);

$("v-bar").style.background = `linear-gradient(to right, ${stops.map((c) => "#" + c.getHexString()).join(",")})`;
$("v-ticks").innerHTML = [[0, "0"], [0.5, "0.5"], [1, "1.0 (allowable)"]].map(([u, label]) =>
  `<span class="v-tick" style="left:${(100 * u) / Umax}%">${label}</span>`).join("");

// ---- displacement at a given fraction of the actuator force ------------------------
function displacement(design, fraction) {
  const path = design.path;
  if (!path) return null;                           // history frames are drawn undeformed
  const s = fraction * (path.length - 1), k = Math.min(Math.floor(s), path.length - 2), f = s - k;
  const mix = (p, q) => p.map((v, i) => v + f * (q[i] - v));
  return { ux: mix(path[k].ux_mm, path[k + 1].ux_mm), uy: mix(path[k].uy_mm, path[k + 1].uy_mm) };
}

// ---- geometry of one design -----------------------------------------------------
const node = (i, j) => i + j * (nx + 1);            // rows from the centre line upwards
const rhoAt = (rho, i, j) => (i < 0 || i >= nx || j >= ny ? 0 : rho[i + Math.max(j, 0) * nx]);  // j < 0: mirror image
const heightOf = (r) => (r < SHOWN ? 0 : r * T);

function build(design, fraction) {
  const rho = design.rho, U = design.utilization, d = displacement(design, fraction);
  const P = (i, j) => [i * a + (d ? d.ux[node(i, j)] : 0), j * a + (d ? d.uy[node(i, j)] : 0)];
  const pos = [], col = [], faceElement = [], edge = [], cross = [];
  for (const side of [1, -1]) {                     // the half that was analysed, then its mirror image
    const tri = (p, q, r, z0, z1, z2, c, shade, e) => {
      pos.push(p[0], side * p[1], z0, q[0], side * q[1], z1, r[0], side * r[1], z2);
      for (let k = 0; k < 3; k++) col.push(c[0] * shade, c[1] * shade, c[2] * shade);
      faceElement.push(side * (e + 1));
    };
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const e = i + j * nx, z = heightOf(rho[e]) / 2;
      if (z === 0) continue;
      const c = colorOf(U ? U[e] : 0);
      const q = [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)];
      const m = [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4];
      // front and back faces: four triangles around the centre point of the bilinear element
      for (let k = 0; k < 4; k++) for (const zz of [z, -z]) tri(q[k], q[(k + 1) % 4], m, zz, zz, zz, c, 1, e);
      // walls where this element stands above its neighbour
      const sides = [[i, j - 1, 0, 1, 0.66], [i + 1, j, 1, 2, 0.8], [i, j + 1, 2, 3, 0.66], [i - 1, j, 3, 0, 0.8]];
      for (const [ni, nj, k0, k1, shade] of sides) {
        const zn = heightOf(rhoAt(rho, ni, nj)) / 2;
        if (z <= zn + 1e-9) continue;
        for (const [zl, zh] of zn > 0 ? [[zn, z], [-z, -zn]] : [[-z, z]]) {
          tri(q[k0], q[k1], q[k1], zl, zl, zh, c, shade, e);
          tri(q[k0], q[k1], q[k0], zl, zh, zh, c, shade, e);
        }
        if (zn === 0) edge.push(q[k0][0], side * q[k0][1], z, q[k1][0], side * q[k1][1], z);
      }
      if (U && U[e] > 1) {                          // over the allowable: a cross that does not rely on colour
        const zc = z + 0.15, k = 0.2, mix = (u, v) => [u[0] + k * (v[0] - u[0]), u[1] + k * (v[1] - u[1])];
        for (const [u, v] of [[mix(q[0], q[2]), mix(q[2], q[0])], [mix(q[1], q[3]), mix(q[3], q[1])]])
          cross.push(u[0], side * u[1], zc, v[0], side * v[1], zc);
      }
    }
  }
  return { pos, col, faceElement, edge, cross };
}

// ---- one renderer per panel, one shared camera ----------------------------------
const camera = new THREE.PerspectiveCamera(30, 1.4, 1, 5000);
const target = new THREE.Vector3(L / 2, 0, 0);
const home = new THREE.Vector3(L / 2 + 27, -54, 232);

function makeLines(material) {
  const obj = new THREE.LineSegments(new THREE.BufferGeometry(), material);
  obj.frustumCulled = false;
  return obj;
}
function fixtures(scene) {
  // actuator force on the centre line, the two clamps, and the object between the jaws
  scene.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-22, 0, 0), 20, 0x1c4fd6, 7, 5));
  const grey = new THREE.MeshBasicMaterial({ color: 0x555555 });
  for (const s of [1, -1]) {
    const clamp = new THREE.Mesh(new THREE.BoxGeometry(3, C.support_height_mm, T + 2), grey);
    clamp.position.set(-1.6, s * (H - C.support_height_mm / 2), 0);
    scene.add(clamp);
  }
  const object = new THREE.Mesh(new THREE.BoxGeometry(C.gap_length_mm * 0.5, 2 * C.gap_half_height_mm - 0.4, T * 0.6),
    new THREE.MeshBasicMaterial({ color: 0x9aa7b4, transparent: true, opacity: 0.45 }));
  object.position.set(L - C.gap_length_mm * 0.3, 0, 0);
  scene.add(object);
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
  const edges = makeLines(new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.45 }));
  const cross = makeLines(new THREE.LineBasicMaterial({ color: 0x000000 }));
  const pick = makeLines(new THREE.LineBasicMaterial({ color: 0x0b5fff }));
  pick.visible = false;
  scene.add(mesh, edges, cross, pick);
  fixtures(scene);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.target.copy(target);
  controls.addEventListener("change", render);
  return { host, renderer, scene, mesh, edges, cross, pick, controls, design: null, faceElement: [] };
});

function render() {
  for (const p of panels) p.renderer.render(p.scene, camera);
}
function resize() {
  const w = panels[0].host.clientWidth, h = Math.round(Math.min(Math.max(w * 0.75, 240), 430));
  for (const p of panels) p.renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  render();
}
function resetView() {
  // move back on narrow panels so that the whole gripper stays in view
  camera.position.copy(home).sub(target).multiplyScalar(Math.max(1, 1.3 / camera.aspect)).add(target);
  for (const p of panels) p.controls.update();
  render();
}

// ---- showing a design -------------------------------------------------------------
let fraction = 1;
const setLines = (obj, array) =>
  obj.geometry.setAttribute("position", new THREE.Float32BufferAttribute(array, 3));

function show(k, design, title) {
  const p = panels[k], g = build(design, fraction), m = design.metrics;
  p.design = design;
  p.faceElement = g.faceElement;
  p.mesh.geometry.dispose();
  p.mesh.geometry = new THREE.BufferGeometry();
  p.mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(g.pos, 3));
  const lin = new Float32Array(g.col.length);       // colours are given in sRGB; three.js works in linear RGB
  for (let i = 0; i < lin.length; i++) {
    const c = g.col[i];
    lin[i] = c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }
  p.mesh.geometry.setAttribute("color", new THREE.BufferAttribute(lin, 3));
  setLines(p.edges, g.edge); setLines(p.cross, g.cross);
  p.pick.visible = false;
  $(`v-title-${k}`).textContent = title;
  const row = (name, value, unit, limit, digits) => {
    const ok = value <= limit * (1 + 1e-3);
    return `<tr class="${ok ? "ok" : "bad"}"><td>${name}</td><td>${value.toFixed(digits)} ${unit}</td>
      <td>limit ${limit.toFixed(digits)} ${unit}</td><td>${ok ? "✓ pass" : "✗ FAIL"}</td></tr>`;
  };
  $(`v-metrics-${k}`).innerHTML = `<table>
    <tr><td>Jaw closing (objective)</td><td>${m.closing_mm.toFixed(2)} mm</td><td>per jaw, at full force</td><td></td></tr>
    ${row("Actuator stroke", m.stroke_mm, "mm", limits.stroke, 2)}
    ${row("Peak stress", m.stress_MPa, "MPa", limits.stress, 1)}
    ${row("Material used", 100 * m.volume_fraction, "%", 100 * limits.volume, 1)}</table>`;
}

const lastFrame = R.frames.length - 1;
let frame = lastFrame, timer = null;
function showRight() {
  if (frame < lastFrame) {
    const f = R.frames[frame];
    show(1, f, `Iteration ${f.iteration} of ${R.frames[lastFrame].iteration}, drawn undeformed`);
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
  $("v-load").textContent = Math.round(100 * fraction);
}

$("v-design").addEventListener("change", () => { stop(); frame = lastFrame; $("v-frame").value = frame; showRight(); });
$("v-frame").addEventListener("input", (e) => { stop(); $("v-design").value = "optimized"; frame = +e.target.value; showRight(); });
$("v-act").addEventListener("input", (e) => { stop(); fraction = +e.target.value / 100; showAll(); });
$("v-reset").addEventListener("click", resetView);
function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  $("v-play").textContent = "▶ Play actuation";
}
$("v-play").addEventListener("click", () => {
  if (timer) return stop();
  frame = lastFrame; $("v-frame").value = frame;
  let t = 0;
  $("v-play").textContent = "■ Stop";
  timer = setInterval(() => {                       // force up to 100% and back, twice
    t += 1;
    fraction = 0.5 - 0.5 * Math.cos((t / 40) * Math.PI);
    $("v-act").value = Math.round(100 * fraction);
    showAll();
    if (t >= 160) { fraction = 1; $("v-act").value = 100; showAll(); stop(); }
  }, 60);
});

// ---- inspection on hover or tap ----------------------------------------------------
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = $("v-tip");
function outline(p, e, side) {
  const d = displacement(p.design, fraction), i = e % nx, j = Math.floor(e / nx);
  const z = heightOf(p.design.rho[e]) / 2 + 0.25;
  const at = (ii, jj) => [ii * a + (d ? d.ux[node(ii, jj)] : 0), side * (jj * a + (d ? d.uy[node(ii, jj)] : 0)), z];
  const c = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
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
  const id = p.faceElement[hit.faceIndex], side = Math.sign(id), e = Math.abs(id) - 1;
  const i = e % nx, j = Math.floor(e / nx), d = p.design, full = d.path ? d.path[d.path.length - 1] : null;
  const corners = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
  const mean = (v) => corners.reduce((s, n) => s + v[n], 0) / 4;
  const u = d.utilization ? d.utilization[e] : 0;
  tip.innerHTML = `<b>Element (${i + 1}, ${j + 1})</b> at x = ${((i + 0.5) * a).toFixed(1)} mm, y = ${(side * (j + 0.5) * a).toFixed(1)} mm<br>
    Density ${d.rho[e].toFixed(2)}<br>
    Von Mises stress at full force ${(u * limits.stress).toFixed(1)} MPa, ${(100 * u).toFixed(0)}% of the allowable: <b>${u > 1 ? "over the limit" : "within the limit"}</b>` +
    (full ? `<br>Displacement at full force ${mean(full.ux_mm).toFixed(2)} mm in x, ${(side * mean(full.uy_mm)).toFixed(2)} mm in y` : "");
  tip.hidden = false;
  const box = root.getBoundingClientRect();
  tip.style.left = `${Math.max(4, Math.min(ev.clientX - box.left + 14, box.width - tip.offsetWidth - 4))}px`;
  tip.style.top = `${ev.clientY - box.top + 14}px`;
  for (const q of panels) if (heightOf(q.design.rho[e]) > 0) outline(q, e, side);   // the same element in both panels
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
