// Placeholder 3D view of the design domain, read from results.json.
// It checks the site pipeline (three.js import map, fetch, rendering) before the solver exists.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const host = document.getElementById("setup-viewer");
const data = await (await fetch("assets/results.json")).json();
const { Lx, Ly, thickness: t } = data.domain;
const yTop = Ly / 2;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f8fa);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(1, 2, 3);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
camera.position.set(Lx * 0.5 + 60, -150, 170);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(Lx * 0.5, 0, 0);
controls.update();

// Design domain = full rectangle minus the passive voids, drawn as extruded blocks.
const solidMat = new THREE.MeshStandardMaterial({ color: 0x9fb3c8, transparent: true, opacity: 0.55 });
const edgeMat = new THREE.LineBasicMaterial({ color: 0x334155 });
function block(x0, x1, y0, y1) {
  const geo = new THREE.BoxGeometry(x1 - x0, y1 - y0, t);
  const mesh = new THREE.Mesh(geo, solidMat);
  mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0);
  scene.add(mesh);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
  edges.position.copy(mesh.position);
  scene.add(edges);
}
const v = data.passive_void[0];
block(0, v.x0, -yTop, yTop);
block(v.x0, Lx, v.y1, yTop);
block(v.x0, Lx, -yTop, v.y0);

// Supports: dark pyramids along the clamped strips on the left edge.
const supMat = new THREE.MeshStandardMaterial({ color: 0x1f2937 });
for (const s of data.supports) {
  for (let y = s.y0; y <= s.y1 + 1e-9; y += (s.y1 - s.y0) / 2) {
    const cone = new THREE.Mesh(new THREE.ConeGeometry(3, 7, 4), supMat);
    cone.rotation.z = -Math.PI / 2;
    cone.position.set(s.x - 3.5, y, 0);
    scene.add(cone);
  }
}

// Input force (orange) and jaw closing directions (blue).
function arrow(p, dir, color, len) {
  const d = new THREE.Vector3(dir[0], dir[1], 0).normalize();
  const origin = new THREE.Vector3(p.x, p.y, 0).addScaledVector(d, -len);
  scene.add(new THREE.ArrowHelper(d, origin, len, color, 6, 4));
}
arrow(data.input, data.input.dir, 0xe8590c, 30);
for (const o of data.outputs) arrow({ x: o.x + 8, y: o.y }, o.dir, 0x1c7ed6, 14);

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
