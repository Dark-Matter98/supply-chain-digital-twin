// Inbound twin control tower: one simulated day from a replay (docs/REPLAY.md), shown twice
// side by side: the plan run untouched (top floor) and with the twin re-planning (bottom floor).
// Every position, time and number comes from the replay; this file only lays it out and
// interpolates. ?video plays the scripted story in story.js frame by frame for recording.
import * as THREE from 'three';
import { STORY } from './story.js';

const qs = new URLSearchParams(location.search);
const VIDEO = qs.has('video');
if (VIDEO) document.body.classList.add('video');
const DATA = window.TWIN_DATA || qs.get('data') || './data/';
const SCENARIO = VIDEO ? null : qs.get('scenario');
const REP = await fetch(DATA + (SCENARIO ? `replay_${SCENARIO}.json` : 'replay_R2_s115_d10.json')).then(r => r.json());
const STRESS_SUMMARY = await fetch(DATA + 'stress_summary.json').then(r => (r.ok ? r.json() : null)).catch(() => null);
const STRESS = REP.meta.stress;   // the disruption in this replay, or null

const RUNS = { plan: REP.runs.static, twin: REP.runs.twin };
const ST = REP.story, FP = ST.focus_part, FS = ST.focus_supplier, FT = ST.focus_truck;
const PROD0 = REP.meta.line.prod_start_h * 60, PROD1 = REP.meta.line.prod_end_h * 60, TAKT = REP.meta.line.takt_min;
const RATE = REP.meta.line_down_usd_per_min;
const NP = REP.network.parts.length;
const KM = REP.network.road_km;                       // free-flow minutes = km at 60 km/h
const SPEED_KMH = 60;

// ------------------------------------------------------------------ helpers
const hhmm = t => { t = Math.max(0, t); return `${String(Math.floor(t / 60) % 24).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`; };
const usd = x => '$' + Math.round(x).toLocaleString('en-US');
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function bisect(arr, t, key = x => x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (key(arr[m]) <= t) lo = m + 1; else hi = m; } return lo - 1; }
function interp(ts, ys, t) {
  if (t <= ts[0]) return ys[0];
  if (t >= ts[ts.length - 1]) return ys[ys.length - 1];
  const i = bisect(ts, t), f = (t - ts[i]) / (ts[i + 1] - ts[i]);
  return ys[i] + (ys[i + 1] - ys[i]) * f;
}
function along(pts, f) {   // point and heading at fraction f of a polyline
  const lens = pts.slice(1).map((p, i) => p.distanceTo(pts[i]));
  let d = clamp(f, 0, 1) * lens.reduce((a, b) => a + b, 0);
  for (let i = 0; i < lens.length; i++) {
    if (d <= lens[i] || i === lens.length - 1) {
      const a = pts[i], b = pts[i + 1], g = lens[i] ? d / lens[i] : 0;
      return { p: a.clone().lerp(b, clamp(g, 0, 1)), h: Math.atan2(b.z - a.z, b.x - a.x) };
    }
    d -= lens[i];
  }
}

// ------------------------------------------------------------------ layout (world units, ground = XZ)
const MAP = { x0: -55, x1: -8, z0: -25, z1: 25 };
const geo = REP.network.suppliers.map(s => [s.x_km, s.y_km]).concat([[0, 0]]);
const G = { x0: Math.min(...geo.map(g => g[0])) - 5, x1: Math.max(...geo.map(g => g[0])) + 5,
            y0: Math.min(...geo.map(g => g[1])) - 5, y1: Math.max(...geo.map(g => g[1])) + 5 };
const MS = Math.min((MAP.x1 - MAP.x0) / (G.x1 - G.x0), (MAP.z1 - MAP.z0) / (G.y1 - G.y0));
const mapXZ = (x, y) => new THREE.Vector3((MAP.x0 + MAP.x1) / 2 + (x - (G.x0 + G.x1) / 2) * MS, 0,
                                          (MAP.z0 + MAP.z1) / 2 - (y - (G.y0 + G.y1) / 2) * MS);
const NODE = { 0: mapXZ(0, 0) };
for (const s of REP.network.suppliers) NODE[s.id] = mapXZ(s.x_km, s.y_km);

const FW = 50, FH = 23.2;
const FLOORS = { plan: { x0: 0, z0: -25.8 }, twin: { x0: 0, z0: 2.6 } };   // gap between them for the titles
const fp = (run, u, v, y = 0) => new THREE.Vector3(FLOORS[run].x0 + u, y, FLOORS[run].z0 + v);
const DOOR_V = [4.8, 9.8, 14.8, 19.8], DOOR_WALL = 7.6, AT_DOOR_U = 4.7;
const YARD = k => [1.3, 4.0 + 4.3 * (k % 5)];
const GATE = [1.3, 22.4];
const laneUV = p => [14.6 + 1.75 * (p % 6), 3.6 + 3.9 * Math.floor(p / 6)];
const rackUV = p => [41.3, 2.6 + 19.6 * p / (NP - 1)];
const LOOP = [[26.6, 12.2], [26.6, 3.2], [38.6, 3.2], [38.6, 12.2], [38.6, 21.2], [26.6, 21.2], [26.6, 12.2]];
const CONV = { u0: 43.3, u1: 46.3, v0: 1.8, v1: 22.9 };

// ------------------------------------------------------------------ renderer, camera
const stage = document.getElementById('stage');
const labelsEl = document.getElementById('labels');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(VIDEO ? 1 : Math.min(2, devicePixelRatio));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9edf2);
const camera = new THREE.PerspectiveCamera(18, 1, 1, 2000);
const VIEW = { x0: -59, x1: 52, z0: -28.6, z1: 26.4 };

function fitCamera() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  const tilt = THREE.MathUtils.degToRad(24);
  const cx = (VIEW.x0 + VIEW.x1) / 2, cz = (VIEW.z0 + VIEW.z1) / 2;
  const vw = VIEW.x1 - VIEW.x0, vh = (VIEW.z1 - VIEW.z0) * Math.cos(tilt);
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const d = Math.max(vh, vw / camera.aspect) / 2 / Math.tan(fov / 2) * (camera.aspect < 1.8 ? 1.07 : 1.02);
  camera.position.set(cx, d * Math.cos(tilt), cz + d * Math.sin(tilt));
  camera.lookAt(cx, 0, cz);
  camera.updateProjectionMatrix();
}

scene.add(new THREE.HemisphereLight(0xffffff, 0xcbd5e1, 1.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(-40, 80, -30);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
Object.assign(sun.shadow.camera, { left: -70, right: 70, top: 50, bottom: -50, near: 1, far: 300 });
sun.shadow.bias = -0.0004;
scene.add(sun);

const matCache = {};
const mat = (c, emit = 0) => matCache[`${c}_${emit}`] ||= new THREE.MeshStandardMaterial({ color: c, roughness: 0.85, emissive: emit ? c : 0x000000, emissiveIntensity: emit });
const BOX = new THREE.BoxGeometry(1, 1, 1);
BOX.translate(0, 0.5, 0);   // origin at the bottom
function box(w, h, d, color, pos, { shadow = true, material } = {}) {
  const m = new THREE.Mesh(BOX, material || mat(color));
  m.scale.set(w, h, d);
  if (pos) m.position.copy(pos);
  m.castShadow = shadow; m.receiveShadow = true;
  scene.add(m);
  return m;
}
function slab(x0, z0, x1, z1, color, y = 0, h = 0.08) { return box(x1 - x0, h, z1 - z0, color, new THREE.Vector3((x0 + x1) / 2, y, (z0 + z1) / 2), { shadow: false }); }
function strip(a, b, w, color, y, material) {
  const len = a.distanceTo(b), m = new THREE.Mesh(BOX, material || mat(color));
  m.scale.set(len, 0.06, w);
  m.position.set((a.x + b.x) / 2, y, (a.z + b.z) / 2);
  m.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x);
  m.receiveShadow = true;
  scene.add(m);
  return m;
}

// Text painted on the floor: it scales with the scene, so zone names never collide at any size.
function floorText(text, center, maxW, h, color = '#64748b') {
  const c = document.createElement('canvas'), ctx = c.getContext('2d'), px = 96;
  const fontCss = `700 ${px}px "Segoe UI", Inter, system-ui, sans-serif`;
  ctx.font = fontCss;
  c.width = Math.ceil(ctx.measureText(text).width) + 24; c.height = px + 32;
  ctx.font = fontCss; ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 12, c.height / 2 + 4);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  let pw = h * c.width / c.height, ph = h;
  if (pw > maxW) { ph *= maxW / pw; pw = maxW; }
  const m = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false }));
  m.renderOrder = 10;   // drawn over the racks in front of it, which the camera tilt would otherwise let cover it
  m.rotation.x = -Math.PI / 2;
  m.position.set(center.x, 0.03, center.z);
  scene.add(m);
  return m;
}

// ------------------------------------------------------------------ labels (DOM, projected each frame)
const LABELS = [];
function label(html, pos, cls = '', obj = null, dy = 0) {
  const el = document.createElement('div');
  el.className = 'lbl ' + cls;
  el.innerHTML = html;
  labelsEl.appendChild(el);
  const L = { el, pos: pos ? pos.clone() : new THREE.Vector3(), obj, dy, show: true };
  LABELS.push(L);
  return L;
}
const _v = new THREE.Vector3();
function placeLabels() {
  const w = stage.clientWidth, h = stage.clientHeight;
  for (const L of LABELS) {
    if (!L.show || (L.obj && !L.obj.visible)) { L.el.style.display = 'none'; continue; }
    _v.copy(L.obj ? L.obj.position : L.pos);
    _v.y += L.dy;
    _v.project(camera);
    L.el.style.display = '';
    L.el.style.left = `${((_v.x + 1) / 2 * w).toFixed(1)}px`;
    L.el.style.top = `${((1 - _v.y) / 2 * h).toFixed(1)}px`;
  }
}

// ------------------------------------------------------------------ static world
slab(-200, -200, 200, 200, 0xe9edf2, -0.2, 0.1);
const NZ = Object.values(NODE).map(p => p.z), MAPZ0 = Math.min(...NZ) - 3.5, MAPZ1 = Math.max(...NZ) + 3;
slab(MAP.x0 - 2, MAPZ0, MAP.x1 + 2, MAPZ1, 0xf8fafc, -0.08);
label('SUPPLIER NETWORK <small>roads coloured by live traffic</small>', new THREE.Vector3(MAP.x0 - 1.5, 0, MAPZ0 - 1.2), 'floor plan');

// roads: every leg driven that day, coloured by the corridor's true congestion
const legSet = new Set();
for (const run of Object.values(RUNS)) {
  for (const tr of run.trucks) for (const e of tr.track) if (e.state === 'travel') legSet.add([e.from, e.to].sort((a, b) => a - b).join('-'));
  for (const c of run.couriers) legSet.add(`0-${c.supplier}`);
}
const ROADS = [...legSet].map(k => {
  const [i, j] = k.split('-').map(Number);
  const m = new THREE.MeshStandardMaterial({ color: 0xc3cbd6, roughness: 0.9 });
  return { i, j, m, mesh: strip(NODE[i], NODE[j], 0.55, null, 0.0, m) };
});
const ROAD_BASE = new THREE.Color(0xc3cbd6), AMBER = new THREE.Color(0xf59e0b), RED = new THREE.Color(0xdc2626);
function paintTraffic(t) {
  const k = clamp(Math.floor(t / 15), 0, 95), mult = REP.traffic.multiplier;
  const closed = stressOn(t) && STRESS.road_supplier ? STRESS.road_supplier : 0;
  for (const r of ROADS) {
    let x = Math.max(...[r.i, r.j].filter(n => n).map(n => mult[n][k]));
    if (closed && (r.i === closed || r.j === closed)) x = Math.max(x, STRESS.road_multiplier);
    const c = x < 1.15 ? ROAD_BASE.clone() : x < 1.45 ? ROAD_BASE.clone().lerp(AMBER, (x - 1.15) / 0.3) : AMBER.clone().lerp(RED, clamp((x - 1.45) / 0.25, 0, 1));
    r.m.color.copy(c);
    r.mesh.scale.z = x < 1.15 ? 0.55 : 0.85;
  }
}

// suppliers and the plant node
const SITE_LABEL = {}, SITE_MAT = {};
for (const s of REP.network.suppliers) {
  const p = NODE[s.id];
  SITE_MAT[s.id] = new THREE.MeshStandardMaterial({ color: s.id === FS ? 0x64748b : 0x94a3b8, roughness: 0.85 });
  box(1.6, 0.9, 1.2, null, p, { material: SITE_MAT[s.id] });
  SITE_LABEL[s.id] = label(s.id === FS ? `Supplier ${s.id}` : `S${s.id}`, p, 'site', null, 1.4);
}
const stressOn = t => !!STRESS && t >= STRESS.start && t < STRESS.end;
const ROAD_TAG = STRESS && STRESS.road_supplier
  ? label(`Road at a crawl (×${STRESS.road_multiplier})`, NODE[0].clone().lerp(NODE[STRESS.road_supplier], 0.5), 'stop', null, 1.0) : null;
box(3.2, 1.6, 2.4, 0x334155, NODE[0]);
label('PLANT', NODE[0], 'big', null, 2.4);

// two plant floors: the same day under the plan, and with the twin
const FLOOR = {};
for (const run of ['plan', 'twin']) {
  const F = FLOOR[run] = {};
  const { x0, z0 } = FLOORS[run];
  slab(x0, z0, x0 + FW, z0 + FH, 0xdde3ea, -0.08);                 // yard
  slab(x0 + DOOR_WALL, z0, x0 + FW, z0 + FH, 0xffffff, -0.06);     // building floor
  const wallC = 0xcbd5e1;
  box(FW - DOOR_WALL, 0.7, 0.25, wallC, fp(run, (DOOR_WALL + FW) / 2, 0)).castShadow = true;
  box(FW - DOOR_WALL, 0.7, 0.25, wallC, fp(run, (DOOR_WALL + FW) / 2, FH));
  box(0.25, 0.7, FH, wallC, fp(run, FW, FH / 2));
  let v = 0;   // dock wall with four door gaps
  for (const dv of DOOR_V) { box(0.25, 0.7, dv - 1.4 - v, wallC, fp(run, DOOR_WALL, (v + dv - 1.4) / 2)); v = dv + 1.4; }
  box(0.25, 0.7, FH - v, wallC, fp(run, DOOR_WALL, (v + FH) / 2));
  DOOR_V.forEach((dv, k) => { box(0.18, 0.12, 2.6, 0x0284c7, fp(run, DOOR_WALL, dv)); });
  F.blocked = DOOR_V.map(dv => box(2.6, 0.5, 2.4, null, fp(run, AT_DOOR_U + 0.6, dv), { material: mat(0xdc2626, 0.25) }));
  F.blockedLbl = DOOR_V.map((dv, k) => label('Door out', null, 'stop', F.blocked[k], 1.0));
  // supermarket lanes (height = hours of cover), line-side racks, tugger loop, conveyor, andon
  F.lanes = []; F.racks = [];
  for (let p = 0; p < NP; p++) {
    const [u, w] = laneUV(p);
    box(1.45, 0.12, 3.3, 0xe2e8f0, fp(run, u, w), { shadow: false });
    F.lanes.push(box(1.15, 1, 3.0, p === FP ? 0x0284c7 : 0x60a5fa, fp(run, u, w, 0.12), { material: new THREE.MeshStandardMaterial({ color: p === FP ? 0x0284c7 : 0x93c5fd, roughness: 0.8 }) }));
    const [ru, rv] = rackUV(p);
    F.racks.push(box(p === FP ? 2.0 : 1.4, 1, p === FP ? 0.62 : 0.5, 0x60a5fa, fp(run, ru, rv), { material: new THREE.MeshStandardMaterial({ color: 0x93c5fd, roughness: 0.8 }) }));
  }
  const loopPts = LOOP.map(([u, w]) => fp(run, u, w, 0.02));
  for (let i = 0; i < loopPts.length - 1; i++) strip(loopPts[i], loopPts[i + 1], 0.35, 0xcbd5e1, 0.0);
  F.loop = loopPts;
  box(CONV.u1 - CONV.u0, 0.45, CONV.v1 - CONV.v0, 0xcbd5e1, fp(run, (CONV.u0 + CONV.u1) / 2, (CONV.v0 + CONV.v1) / 2));
  F.cars = Array.from({ length: 11 }, () => box(2.1, 0.75, 1.15, 0x475569, fp(run, 44.8, 2, 0.45)));
  box(0.25, 3.6, 0.25, 0x94a3b8, fp(run, 48.6, 2.2));
  F.andon = new THREE.MeshStandardMaterial({ color: 0x16a34a, emissive: 0x16a34a, emissiveIntensity: 0.6 });
  box(1.3, 1.3, 1.3, null, fp(run, 48.6, 2.2, 3.6), { material: F.andon });
  // pallets in receiving, one per container waiting to be received (capped)
  F.pallets = [];
  for (let d = 0; d < 4; d++) for (let k = 0; k < 8; k++) F.pallets.push(box(0.75, 0.6, 0.75, 0xd6b98c, fp(run, 9.0 + 0.95 * (k % 4), DOOR_V[d] - 0.6 + 1.1 * Math.floor(k / 4))));
  // vehicles inside
  F.trucks = REP.runs.static.trucks.map((_, k) => makeTruck(k === FT ? 0xf59e0b : 0x334155));
  F.bg = Array.from({ length: 8 }, () => makeTruck(0x94a3b8));
  F.tugger = makeTugger();
  F.forks = Array.from({ length: 3 }, () => box(0.9, 0.6, 0.7, 0x0284c7, null, { material: mat(0x0284c7, 0.35) }));
  // labels
  F.title = label(run === 'plan' ? 'FIXED PLAN <small>no re-planning during the day</small>'
                                 : 'FIXED PLAN + SHORTAGE EARLY WARNING <small>forecasts run-outs, acts hours ahead</small>',
                  new THREE.Vector3(x0, 0, z0 - 1.5), 'floor ' + (run === 'plan' ? 'plan' : 'twin'));
  for (const [txt, u, w] of [['DOCK', 3.9, 5.6], ['RECEIVING', 10.8, 4.6], ['SUPERMARKET', 19.4, 9.5], ['TUGGER LOOP', 32.6, 10], ['LINE', 44.8, 4.5]])
    floorText(txt, fp(run, u, 0.95), w, 1.05);
  F.lineTag = label('', fp(run, 44.8, 12.4), 'ok', null, 4.5);
  F.partTag = label('', fp(run, rackUV(FP)[0] - 1.2, rackUV(FP)[1]), 'big', null, 1.5);
  F.partTag.el.style.transform = 'translate(-100%, -50%)';
  F.truckLbl = F.trucks.map((m, k) => label(`T${k}`, null, 'truck' + (k === FT ? ' focus' : ''), m, 2.0));
}

function makeTruck(color) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(BOX, mat(0xffffff)); body.scale.set(2.3, 1.0, 1.15); body.position.x = -0.35; body.castShadow = true;
  const cab = new THREE.Mesh(BOX, mat(color)); cab.scale.set(0.8, 1.0, 1.1); cab.position.x = 1.25; cab.castShadow = true;
  const roof = new THREE.Mesh(BOX, mat(color)); roof.scale.set(2.3, 0.12, 1.15); roof.position.set(-0.35, 1.0, 0);
  g.add(body, cab, roof);
  scene.add(g);
  return g;
}
function makeTugger() {
  const g = new THREE.Group();
  const head = new THREE.Mesh(BOX, mat(0x0f172a)); head.scale.set(0.9, 0.7, 0.75); g.add(head);
  g.load = [];
  for (let i = 1; i <= 3; i++) {
    const cart = new THREE.Mesh(BOX, mat(0x94a3b8)); cart.scale.set(0.9, 0.18, 0.75); cart.position.x = -1.05 * i; g.add(cart);
    const load = new THREE.Mesh(BOX, mat(0x60a5fa)); load.scale.set(0.75, 0.5, 0.6); load.position.set(-1.05 * i, 0.18, 0); g.add(load); g.load.push(load);
  }
  g.traverse(o => { o.castShadow = true; });
  g.scale.setScalar(1.5);
  scene.add(g);
  return g;
}

// map vehicles: milk-run trucks (from the twin run; identical to the plan until the twin acts) and couriers
const MAP_TRUCKS = REP.runs.twin.trucks.map((_, k) => {
  const g = makeTruck(k === FT ? 0xf59e0b : 0x334155);
  g.scale.setScalar(0.85);
  return { g, lbl: label(`T${k}`, null, 'truck' + (k === FT ? ' focus' : ''), g, 1.8) };
});
const VANS = Array.from({ length: 3 }, () => {
  const g = new THREE.Group();
  const b = new THREE.Mesh(BOX, mat(0x0284c7, 0.3)); b.scale.set(1.5, 0.9, 0.9); b.castShadow = true; g.add(b);
  scene.add(g);
  return { g, lbl: label('Courier', null, 'twin', g, 1.7) };
});

// spotlight rings the story can switch on
const RING = new THREE.RingGeometry(1.0, 1.35, 48); RING.rotateX(-Math.PI / 2);
const SPOTS = {};
function spot(name, color) {
  const m = new THREE.Mesh(RING, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false }));
  m.position.y = 0.15; m.visible = false; scene.add(m);
  SPOTS[name] = m;
}
spot('truck', 0xf59e0b); spot('courier', 0x0284c7); spot('planLine', 0xdc2626); spot('twinLine', 0x16a34a);
spot('planRack', 0xdc2626); spot('twinRack', 0x0284c7); spot('supplier', 0x0284c7);

// ------------------------------------------------------------------ state at time t
function truckState(run, k, t) {
  const track = run.trucks[k].track, i = bisect(track, t, e => e.t0);
  if (i < 0 || t >= track[i].t1) return { state: 'idle' };
  const e = track[i];
  return { ...e, f: (t - e.t0) / Math.max(e.t1 - e.t0, 1e-6) };
}
const dockOf = (run, k, t0) => run.dock.find(d => d.truck === k && d.t_arrive !== null && d.t_arrive <= t0 + 1e-6 && (d.t_leave === null || t0 <= d.t_leave + 1e-6));
const lineStatus = (run, t) => (t < PROD0 || t >= PROD1) ? 'off' : run.line_stops.some(s => s.t0 <= t && t < s.t1) ? 'stopped' : 'running';
const downSoFar = (run, t) => run.line_stops.reduce((a, s) => a + Math.max(0, Math.min(t, s.t1) - s.t0) * (s.t0 < t), 0);
function runningMinutes(run, t) { const hi = clamp(t, PROD0, PROD1); return Math.max(0, hi - PROD0 - downSoFar(run, hi)); }
const cover = (run, where, p, t) => interp(run.cover_h.t, run.cover_h[where][p], t);
const legRatio = e => (e.t1 - e.t0) / Math.max(KM[e.from][e.to] / SPEED_KMH * 60, 1e-6);

// ------------------------------------------------------------------ per-frame scene update
function setPose(obj, p, h) { obj.position.set(p.x, p.y || 0, p.z); obj.rotation.y = -h; }
function updateMap(t) {
  paintTraffic(t);
  if (ROAD_TAG) ROAD_TAG.show = stressOn(t);
  if (STRESS && STRESS.supplier_down) {
    const s = STRESS.supplier_down, down = stressOn(t);
    SITE_MAT[s].color.setHex(down ? 0xdc2626 : s === FS ? 0x64748b : 0x94a3b8);
    SITE_LABEL[s].el.className = 'lbl site' + (down ? ' stop' : '');
    SITE_LABEL[s].el.textContent = down ? `Supplier ${s} · shut until ${hhmm(STRESS.end)}` : `Supplier ${s}`;
  }
  const run = RUNS.twin;
  MAP_TRUCKS.forEach(({ g }, k) => {
    const s = truckState(run, k, t);
    if (s.state === 'travel') { const a = along([NODE[s.from], NODE[s.to]], s.f); setPose(g, a.p, a.h); g.visible = true; }
    else if (s.state === 'load' || s.state === 'late') { setPose(g, NODE[s.at].clone().add(new THREE.Vector3(1.6, 0, 1.2)), 0); g.visible = true; }
    else g.visible = false;
  });
  const vans = run.couriers.filter(c => c.t_dispatch <= t && t < c.t_at_plant);
  VANS.forEach(({ g }, k) => {
    const c = vans[k];
    g.visible = !!c;
    if (!c) return;
    let a;
    if (t < c.t_at_supplier) a = along([NODE[0], NODE[c.supplier]], (t - c.t_dispatch) / (c.t_at_supplier - c.t_dispatch));
    else if (t < c.t_pickup) a = { p: NODE[c.supplier].clone().add(new THREE.Vector3(-1.8, 0, 1.4 + 1.3 * k)), h: 0 };
    else a = along([NODE[c.supplier], NODE[0]], (t - c.t_pickup) / (c.t_at_plant - c.t_pickup));
    setPose(g, a.p, a.h);
  });
}

function updateFloor(name, t) {
  const run = RUNS[name], F = FLOOR[name];
  const out = new Set((run.doors_out || []).filter(d => d.t0 <= t && t < d.t1).map(d => d.door));
  F.blocked.forEach((b, k) => { b.visible = out.has(k); });
  // milk-run trucks at the dock
  F.trucks.forEach((g, k) => {
    const s = truckState(run, k, t);
    let uv = null, h = 0;
    if (s.state === 'gate') { uv = GATE; h = -Math.PI / 2; }
    else if (s.state === 'yard') { uv = YARD(k); h = -Math.PI / 2; }
    else if (['door_wait', 'unload', 'dock_out'].includes(s.state)) {
      const d = dockOf(run, k, s.t0);
      uv = [AT_DOOR_U, DOOR_V[d && d.door !== null ? d.door : k % 4]]; h = 0;
    }
    g.visible = !!uv;
    if (uv) setPose(g, fp(name, uv[0], uv[1]), h);
  });
  // other carriers' trucks
  const live = run.dock.filter(d => d.kind === 'background' && d.t_arrive <= t && t < (d.t_leave ?? 1e9));
  F.bg.forEach((g, k) => {
    const d = live[k];
    g.visible = !!d;
    if (!d) return;
    if (d.t_yard === null || t < d.t_yard) setPose(g, fp(name, GATE[0], GATE[1]), -Math.PI / 2);
    else if (d.t_door === null || t < d.t_door) setPose(g, fp(name, ...YARD(k + 2)), -Math.PI / 2);
    else setPose(g, fp(name, AT_DOOR_U, DOOR_V[d.door]), 0);
  });
  // stock: supermarket lanes and line-side racks
  for (let p = 0; p < NP; p++) {
    const sm = cover(run, 'supermarket', p, t), ln = cover(run, 'line', p, t);
    F.lanes[p].scale.y = 0.08 + 0.42 * Math.min(sm, 6);
    F.racks[p].scale.y = 0.08 + 0.75 * Math.min(ln, 3);
    const m = F.racks[p].material;
    m.color.setHex(ln < 0.05 ? 0xdc2626 : ln < 0.5 ? 0xf59e0b : p === FP ? 0x0284c7 : 0x93c5fd);
    m.emissive.setHex(ln < 0.05 ? 0xdc2626 : 0x000000); m.emissiveIntensity = 0.5;
    if (ln < 0.05) F.racks[p].scale.y = 0.5;
  }
  // the focus part's live stock, written next to its rack
  const c19 = cover(run, 'line', FP, t);
  F.partTag.el.className = 'lbl big ' + (c19 < 0.05 ? 'stop' : c19 < 0.5 ? 'focus' : 'twin');
  F.partTag.el.textContent = c19 < 0.05 ? `Part ${FP}: OUT` : `Part ${FP}: ${c19.toFixed(1)} h of stock`;
  // line, andon, cars
  const status = lineStatus(run, t);
  F.andon.color.setHex(status === 'stopped' ? 0xdc2626 : status === 'running' ? 0x16a34a : 0x94a3b8);
  F.andon.emissive.copy(F.andon.color);
  const span = CONV.v1 - CONV.v0 - 1.4, gap = span / F.cars.length, off = runningMinutes(run, t) / TAKT * gap * 0.5;
  F.cars.forEach((c, k) => { c.position.copy(fp(name, (CONV.u0 + CONV.u1) / 2, CONV.v1 - 0.7 - ((k * gap + off) % span), 0.45)); });
  F.lineTag.el.className = 'lbl ' + (status === 'stopped' ? 'stop' : 'ok');
  F.lineTag.el.textContent = status === 'stopped' ? `LINE STOPPED · ${downSoFar(run, t).toFixed(0)} min` : status === 'running' ? 'Line running' : 'Line off shift';
  // tugger: on its loop since the last departure
  const deps = run.tugger.filter(x => x.t_depart <= t), dep = deps[deps.length - 1];
  if (dep) {
    const a = along(F.loop, (t - dep.t_depart) / REP.meta.plant.tugger_loop_min);
    setPose(F.tugger, a.p, a.h);
    F.tugger.load.forEach((l, i) => { l.visible = dep.containers > i * 5 && t < dep.t_drop; });
  }
  // hot runs: a forklift from the part's supermarket lane straight to its rack
  const runsNow = run.hot_runs.filter(h => h.t_start <= t && t < h.t_drop + 1);
  F.forks.forEach((f, k) => {
    const h = runsNow[k];
    f.visible = !!h;
    if (!h) return;
    const [lu, lv] = laneUV(h.part), [ru, rv] = rackUV(h.part);
    const a = along([fp(name, lu + 1.2, lv), fp(name, ru - 1.4, rv)], (t - h.t_start) / Math.max(h.t_drop - h.t_start, 1e-6));
    setPose(f, a.p, a.h);
  });
  // receiving: containers unloaded and not yet received
  const unl = {};
  for (const d of run.dock) if (d.kind === 'milk' && d.t_unloaded !== null) unl[`${d.truck}_${d.t_arrive}`] = d.t_unloaded;
  const n = run.receipts.reduce((a, r) => a + ((unl[`${r.truck}_${r.t_arrive}`] ?? 1e9) <= t && t < r.t_received ? r.containers : 0), 0);
  const show = Math.min(F.pallets.length, Math.ceil(n / 2));
  F.pallets.forEach((p, k) => { p.visible = k < show; });
}

// ------------------------------------------------------------------ sidebar
const $ = id => document.getElementById(id);
// The two runs: the fixed plan, and the same plan with the Shortage Early Warning (the
// re-planning policy, SPEC section 8.2) running on top of the twin.
const WHO = { plan: 'Fixed plan', twin: 'Early warning' };
function lineRows(t) {
  return ['plan', 'twin'].map(name => {
    const run = RUNS[name], s = lineStatus(run, t);
    const cars = Math.floor(runningMinutes(run, t) / TAKT);
    const pill = s === 'running' ? '<span class="pill run">RUNNING</span>' : s === 'stopped' ? '<span class="pill stop">STOPPED</span>' : '<span class="pill off">OFF SHIFT</span>';
    return `<div class="row"><span class="who ${name}">${WHO[name]}</span>${pill}<span class="num">${cars} cars</span></div>`;
  }).join('');
}
function costRows(t) {
  return ['plan', 'twin'].map(name => {
    const m = downSoFar(RUNS[name], t);
    return `<div class="row"><span class="who ${name}">${WHO[name]}</span><span class="num ${m > 0 ? 'bad' : ''}">${m.toFixed(1)} min</span><span class="num ${m > 0 ? 'bad' : ''}">${usd(m * RATE)}</span></div>`;
  }).join('') + (() => {
    const spent = RUNS.twin.actions.filter(a => a.t <= t).reduce((a, x) => a + x.premium_usd, 0);
    return `<div class="row" style="color:#64748b;font-size:13px"><span>Courier cost</span><span></span><span class="num">${usd(spent)}</span></div>`;
  })();
}

const CH = { w: 400, h: 150, l: 34, r: 8, t: 10, b: 22, t0: 6 * 60, t1: 18 * 60, y1: 4 };
const cx = t => CH.l + (t - CH.t0) / (CH.t1 - CH.t0) * (CH.w - CH.l - CH.r);
const cy = y => CH.t + (1 - Math.min(y, CH.y1) / CH.y1) * (CH.h - CH.t - CH.b);
function pathOf(run) {
  const ts = run.cover_h.t, ys = run.cover_h.line[FP];
  return ts.map((t, i) => (t >= CH.t0 && t <= CH.t1) ? `${cx(t).toFixed(1)},${cy(ys[i]).toFixed(1)}` : null).filter(Boolean).join(' ');
}
const P_PLAN = pathOf(RUNS.plan), P_TWIN = pathOf(RUNS.twin);
function chart(t) {
  const x = cx(clamp(t, CH.t0, CH.t1));
  const stop = ST.static_stop ? ST.static_stop[0] : Infinity, act = ST.first_twin_action ?? Infinity;
  let s = `<svg width="${CH.w}" height="${CH.h}" viewBox="0 0 ${CH.w} ${CH.h}" font-family="Inter, Segoe UI" font-size="11">
  <defs><clipPath id="now"><rect x="0" y="0" width="${x}" height="${CH.h}"/></clipPath></defs>`;
  for (const y of [0, 1, 2, 3, 4]) s += `<line x1="${CH.l}" x2="${CH.w - CH.r}" y1="${cy(y)}" y2="${cy(y)}" stroke="${y ? '#eef2f6' : '#fca5a5'}" stroke-width="${y ? 1 : 1.5}"/><text x="${CH.l - 6}" y="${cy(y) + 4}" text-anchor="end" fill="#94a3b8">${y}h</text>`;
  for (const h of [6, 9, 12, 15, 18]) s += `<text x="${cx(h * 60)}" y="${CH.h - 6}" text-anchor="${h === 18 ? 'end' : 'middle'}" fill="#94a3b8">${String(h).padStart(2, '0')}:00</text>`;
  s += `<g clip-path="url(#now)"><polyline points="${P_PLAN}" fill="none" stroke="#94a3b8" stroke-width="2.5"/>`;
  s += `<polyline points="${P_TWIN}" fill="none" stroke="#0284c7" stroke-width="2.5" stroke-dasharray="${t < act ? '0' : '0'}"/></g>`;
  if (t >= act) s += `<line x1="${cx(act)}" x2="${cx(act)}" y1="${CH.t}" y2="${CH.h - CH.b}" stroke="#0284c7" stroke-dasharray="3 3"/><text x="${cx(act) - 4}" y="${CH.t + 9}" text-anchor="end" fill="#0284c7" font-weight="700">warning</text>`;
  if (t >= stop) s += `<circle cx="${cx(stop)}" cy="${cy(0)}" r="5" fill="#dc2626"/><text x="${cx(stop) + 7}" y="${cy(0) - 6}" fill="#dc2626" font-weight="700">line stops</text>`;
  s += `<line x1="${x}" x2="${x}" y1="${CH.t}" y2="${CH.h - CH.b}" stroke="#0f172a" stroke-opacity="0.25"/>`;
  s += `<text x="${CH.w - CH.r}" y="${CH.t + 9}" text-anchor="end" fill="#475569"><tspan fill="#94a3b8" font-weight="700">━ fixed plan</tspan>  <tspan fill="#0284c7" font-weight="700">━ early warning</tspan></text>`;
  return s + '</svg>';
}

// alerts: built once from the replay, shown as their time passes
const ALERTS = [];
if (FT !== null && FT !== undefined) for (const e of RUNS.plan.trucks[FT].track) if (e.state === 'travel' && legRatio(e) > 1.5)
  ALERTS.push({ t: e.t1, tag: 'risk', tagText: 'TRAFFIC', text: `T${FT} ${e.from ? 'S' + e.from : 'plant'} → ${e.to ? 'S' + e.to : 'plant'}: ${Math.round(e.t1 - e.t0)} min, normally ${Math.round(KM[e.from][e.to] / SPEED_KMH * 60)}` });
if (STRESS) {
  ALERTS.push({ t: STRESS.start, tag: 'stop', tagText: 'STRESS', text: STRESS.label });
  ALERTS.push({ t: STRESS.end, tag: 'plan', tagText: 'STRESS', text: 'Disruption over' });
}
const firstRisk = RUNS.twin.risk.find(r => r.least_slack_min < 0);
if (firstRisk) ALERTS.push({ t: firstRisk.t, tag: 'risk', tagText: 'WARNING', text: `Part ${firstRisk.part} will run ${Math.round(-firstRisk.least_slack_min)} min short at the line` });
const WORDS = { priority: a => `Priority dock door for T${a.truck}`, hot_run: a => `Rush part ${a.part} to the line`,
  courier: a => `Courier to S${a.supplier} for part ${a.part} (${usd(a.premium_usd)})`, reroute: a => `Re-route T${a.truck}` };
for (const a of RUNS.twin.actions) ALERTS.push({ t: a.t, tag: 'twin', tagText: 'ACTION', text: (WORDS[a.kind] || (x => x.kind))(a) });
for (const c of RUNS.twin.couriers) ALERTS.push({ t: c.t_received, tag: 'twin', tagText: 'ACTION', text: `Courier delivers ${c.containers} containers from S${c.supplier}` });
for (const s of RUNS.plan.line_stops) {
  ALERTS.push({ t: s.t0, tag: 'stop', tagText: 'STOP', text: `Fixed plan: line stopped, part ${s.parts.join(', ')} ran out` });
  ALERTS.push({ t: s.t1, tag: 'plan', tagText: 'PLAN', text: `Fixed plan: line restarts after ${Math.round(s.t1 - s.t0)} min` });
}
ALERTS.sort((a, b) => a.t - b.t);
const alertsHTML = t => ALERTS.filter(a => a.t <= t).slice(-8).reverse().map(a =>
  `<div class="alert"><span class="t">${hhmm(a.t)}</span><span class="tag ${a.tag}">${a.tagText}</span><span>${a.text}</span></div>`).join('');

function truckCards(t) {
  const run = RUNS.twin;
  return run.trucks.map((_, k) => {
    const s = truckState(run, k, t);
    let txt = 'Parked at the plant', late = false;
    if (s.state === 'travel') { const r = legRatio(s); late = r > 1.5; txt = `${s.from ? 'S' + s.from : 'Plant'} → ${s.to ? 'S' + s.to : 'plant'}${late ? ` · traffic ×${r.toFixed(1)}` : ''}`; }
    else if (s.state === 'load') txt = `Loading at S${s.at}`;
    else if (s.state === 'late') { txt = `Waiting: S${s.at} not ready`; late = true; }
    else if (s.state === 'gate') txt = 'At the gate';
    else if (s.state === 'yard') txt = 'Waiting for a door';
    else if (s.state === 'door_wait') txt = 'At a door, waiting for a crew';
    else if (s.state === 'unload') txt = 'Unloading';
    else if (s.state === 'dock_out') txt = 'Leaving the dock';
    const f = s.state === 'idle' ? 0 : s.f;
    return `<div class="truck ${late ? 'late' : ''}"><b>T${k}</b>${k === FT ? ' <span style="color:#d97706;font-weight:700">· carries part ' + FP + '</span>' : ''}<div class="st">${txt}</div><div class="bar"><i style="width:${(f * 100).toFixed(0)}%;background:${late ? '#f59e0b' : '#64748b'}"></i></div></div>`;
  }).join('');
}

// ------------------------------------------------------------------ results and end card
const H = REP.headline, KP = { plan: RUNS.plan.day_kpis, twin: RUNS.twin.day_kpis };
function bars(rows, max, fmt) {
  return rows.map(([name, v, cls]) => `<div class="bar2"><span>${name}</span><div class="track"><div class="fill ${cls}" style="width:${(100 * v / max).toFixed(1)}%"></div><span class="val">${fmt(v)}</span></div></div>`).join('');
}
function resultsHTML(p) {
  const k = x => x * clamp(p, 0, 1);
  const d = (a, b) => `${Math.round(100 * (b / a - 1))}%`;
  const rp = H.re_planning, op = H.optimization;
  const tot = Math.max(rp.R1.never_usd_per_day, rp.R2.never_usd_per_day);
  return `<div class="res"><h1>What the Shortage Early Warning changed</h1><p class="sub">Simulated lean plant: local suppliers, milk runs 1–4 times a day, 2 h of safety stock at the line. One day shown above; the study runs 20 held-out test days in each of two traffic worlds.</p>
  <div class="grid">
   <div class="card"><h3>This day</h3>
    <div class="metric"><div class="name">Cars built <span class="delta">+${KP.twin.vehicles - KP.plan.vehicles}</span></div>${bars([['Fixed plan', k(KP.plan.vehicles), 'plan'], ['Early warning', k(KP.twin.vehicles), 'twin']], 800, v => Math.round(v))}</div>
    <div class="metric"><div class="name">Line down <span class="delta">−${KP.plan.line_down_min.toFixed(0)} min</span></div>${bars([['Fixed plan', k(KP.plan.line_down_min), 'plan'], ['Early warning', k(KP.twin.line_down_min), 'twin']], KP.plan.line_down_min, v => `${v.toFixed(0)} min`)}</div>
    <div class="metric"><div class="name">Total cost of the day <span class="delta">${d(KP.plan.total_usd, KP.twin.total_usd)}</span></div>${bars([['Fixed plan', k(KP.plan.total_usd), 'plan'], ['Early warning', k(KP.twin.total_usd), 'twin']], KP.plan.total_usd, usd)}</div>
    <div class="note">Its actions that day (couriers, rush to the line, dock priority) cost ${usd(KP.twin.premium_usd + KP.twin.disruption_usd)}.</div></div>
   <div class="card"><h3>20 held-out test days, per day</h3>
    <div class="metric"><div class="name">Normal traffic <span class="delta">${rp.R1.change_pct}%</span></div>${bars([['Fixed plan', k(rp.R1.never_usd_per_day), 'plan'], ['Early warning', k(rp.R1.twin_usd_per_day), 'twin']], tot, usd)}</div>
    <div class="metric"><div class="name">Drifting traffic <span class="delta">${rp.R2.change_pct}%</span></div>${bars([['Fixed plan', k(rp.R2.never_usd_per_day), 'plan'], ['Early warning', k(rp.R2.twin_usd_per_day), 'twin']], tot, usd)}</div>
    <div class="note">Total cost = logistics + $5,000 per line-down minute. Where it does not pay: with 3–4 h of stock the line rarely stops, and it saves only $40–210 a day.</div></div>
   <div class="card"><h3>The plan itself, optimized (OR-Tools)</h3>
    <div class="metric"><div class="name">Supplier pickups per day <span class="delta">${op.pickups_per_day.optimized - op.pickups_per_day.naive}</span></div>${bars([['Naive plan', k(op.pickups_per_day.naive), 'plan'], ['Optimized', k(op.pickups_per_day.optimized), 'twin']], op.pickups_per_day.naive, v => Math.round(v))}</div>
    <div class="metric"><div class="name">Logistics cost <span class="delta">${op.logistics_pct}%</span></div><div style="font-size:30px;font-weight:800;margin-top:6px">${usd(-op.logistics_usd_per_day)} <span style="font-size:15px;color:#64748b;font-weight:600">saved per day</span></div></div>
    <div class="note">Routes, pickup frequency and fleet, checked in the twin on 20 test days.</div></div>
  </div></div>`;
}
const NAME = qs.get('name') || 'Prashant Dhakal';
const endHTML = () => `<div class="box"><h1>Inbound Supply Chain Digital Twin</h1>
 <div class="tagline">Supply chain · plant process · optimization · system decisions</div>
 <div class="stack">Discrete-event simulation (SimPy) · Optimization (OR-Tools)<br>Probabilistic forecasting (Kalman filter, Monte Carlo) · Python · three.js</div>
 <div class="name">${NAME}</div><div class="fine">Synthetic plant and data. Every movement replayed from the simulation.</div></div>`;

// ------------------------------------------------------------------ frame
let lastUI = '', lastSize = '';
function frame(t, opts = {}) {
  const size = `${stage.clientWidth}x${stage.clientHeight}`;
  if (size !== lastSize) { lastSize = size; fitCamera(); }   // the grid settles after the sidebar fills
  updateMap(t);
  updateFloor('plan', t);
  updateFloor('twin', t);
  // spotlights
  const on = new Set(opts.spots || []);
  for (const [k, m] of Object.entries(SPOTS)) m.visible = on.has(k);
  const pulse = 1 + 0.18 * Math.sin((opts.sec ?? performance.now() / 1000) * 6);
  if (on.has('truck') && MAP_TRUCKS[FT]) { SPOTS.truck.position.copy(MAP_TRUCKS[FT].g.position).setY(0.15); SPOTS.truck.scale.setScalar(2.2 * pulse); SPOTS.truck.visible = MAP_TRUCKS[FT].g.visible; }
  if (on.has('courier')) { const v = VANS.find(x => x.g.visible); SPOTS.courier.visible = !!v; if (v) { SPOTS.courier.position.copy(v.g.position).setY(0.15); SPOTS.courier.scale.setScalar(2.0 * pulse); } }
  if (on.has('supplier')) { SPOTS.supplier.position.copy(NODE[FS]).setY(0.15); SPOTS.supplier.scale.setScalar(2.6 * pulse); }
  const [ru, rv] = rackUV(FP);
  SPOTS.planRack.position.copy(fp('plan', ru, rv, 0.15)); SPOTS.planRack.scale.setScalar(1.6 * pulse);
  SPOTS.twinRack.position.copy(fp('twin', ru, rv, 0.15)); SPOTS.twinRack.scale.setScalar(1.6 * pulse);
  SPOTS.planLine.position.copy(fp('plan', 48.6, 2.2, 0.15)); SPOTS.planLine.scale.setScalar(2.2 * pulse);
  SPOTS.twinLine.position.copy(fp('twin', 48.6, 2.2, 0.15)); SPOTS.twinLine.scale.setScalar(2.2 * pulse);
  SITE_LABEL[FS].el.className = 'lbl site' + (on.has('supplier') || on.has('courier') ? ' twin' : '');

  // sidebar (rebuilt only when the minute changes)
  const key = `${Math.floor(t)}|${(opts.glow || []).join()}`;
  if (key !== lastUI) {
    lastUI = key;
    $('clock').textContent = hhmm(t);
    $('lineRows').innerHTML = lineRows(t);
    $('costRows').innerHTML = costRows(t);
    $('chart').innerHTML = chart(t);
    $('alerts').innerHTML = alertsHTML(t);
    $('trucks').innerHTML = truckCards(t);
    for (const id of ['lineCard', 'chartCard', 'costCard', 'alertCard']) $(id).classList.toggle('glow', (opts.glow || []).includes(id));
  }
  // caption, results, end card
  const cap = $('caption');
  if (opts.caption) { cap.innerHTML = opts.caption + (opts.sub ? `<small>${opts.sub}</small>` : ''); cap.style.opacity = opts.capAlpha ?? 1; }
  else if (VIDEO) cap.style.opacity = 0;
  const res = $('results');
  if (opts.results !== undefined && opts.results !== null) { res.style.display = 'flex'; res.innerHTML = resultsHTML(opts.results); res.style.opacity = clamp(opts.results * 4, 0, 1); }
  else if (VIDEO || !showingResults) res.style.display = 'none';
  const end = $('endcard');
  if (opts.endcard !== undefined && opts.endcard !== null) { end.style.display = 'flex'; end.innerHTML = endHTML(); end.style.opacity = clamp(opts.endcard * 3, 0, 1); }
  else end.style.display = 'none';
  renderer.render(scene, camera);
  placeLabels();
}

// ------------------------------------------------------------------ stress-test card and picker
function stressCard() {
  const row = STRESS_SUMMARY && STRESS_SUMMARY.scenarios.find(s => s.scenario === STRESS.id);
  const card = document.createElement('div');
  card.className = 'card stress';
  const k = x => `$${Math.round(x / 1000).toLocaleString('en-US')}k`;
  card.innerHTML = `<h3>Stress test</h3><div class="stitle">${STRESS.label}</div><div class="sdetail">${STRESS.detail}</div>` + (row ? `
    <div class="row"><span class="who plan">Fixed plan</span><span>line stopped ${row.fixed_stop_days} of ${row.days} days</span><span class="num">${row.fixed_line_down_min} min</span></div>
    <div class="row"><span class="who twin">Early warning</span><span>line stopped ${row.warning_stop_days} of ${row.days} days</span><span class="num">${row.warning_line_down_min} min</span></div>
    <div class="row"><span class="who">Cost of the day</span><span>${k(row.fixed_total_usd)} → ${k(row.warning_total_usd)}</span><span class="num" style="color:var(--green)">−${row.saving_pct}%</span></div>
    <div class="sdetail">Averages over ${row.days} held-out test days. Shown: the median day (seed ${row.replay_seed}).</div>` : '');
  $('lineCard').before(card);
}
function scenarioPicker() {
  const sel = document.createElement('select');
  sel.id = 'scenario';
  const opts = [['', 'Day 10: truck delayed by traffic']].concat(
    (STRESS_SUMMARY ? STRESS_SUMMARY.scenarios : []).filter(s => s.scenario !== 'baseline').map(s => [s.scenario, `Stress test: ${s.label}`]));
  sel.innerHTML = opts.map(([v, l]) => `<option value="${v}" ${v === (SCENARIO || '') ? 'selected' : ''}>${l}</option>`).join('');
  sel.onchange = () => { const q = new URLSearchParams(location.search); if (sel.value) q.set('scenario', sel.value); else q.delete('scenario'); q.delete('t'); location.search = q.toString(); };
  $('play').before(sel);
}

// ------------------------------------------------------------------ run
addEventListener('resize', () => { fitCamera(); lastUI = ''; });
fitCamera();
$('chartTitle').textContent = `Part ${FP} at the line · hours of stock`;
if (STRESS) stressCard();
let showingResults = false;
if (VIDEO) {
  $('speed').textContent = `Day ${REP.meta.day} · replay`;
  window.__story = { fps: STORY.fps, frames: STORY.frames() };
  window.__frame = f => { const s = STORY.at(f / STORY.fps); frame(s.t, s); $('speed').textContent = s.speed || `Day ${REP.meta.day}`; return true; };
  window.__ready = true;
} else {
  scenarioPicker();
  let t = Number(qs.get('t') || (STRESS ? STRESS.start - 60 : 8 * 60)), playing = true, rate = 10;   // simulated minutes per second
  const scrub = $('scrub'), play = $('play'), fast = $('fast');
  const RATES = [5, 10, 30];
  fast.textContent = `${rate} min/s`;
  play.onclick = () => { playing = !playing; play.textContent = playing ? 'Pause' : 'Play'; };
  fast.onclick = () => { rate = RATES[(RATES.indexOf(rate) + 1) % RATES.length]; fast.textContent = `${rate} min/s`; };
  scrub.oninput = () => { t = Number(scrub.value); lastUI = ''; };
  $('showResults').onclick = () => { showingResults = !showingResults; $('results').style.display = showingResults ? 'flex' : 'none'; if (showingResults) $('results').innerHTML = resultsHTML(1); };
  $('results').onclick = () => { showingResults = false; $('results').style.display = 'none'; };
  $('speed').textContent = STRESS ? `Day ${REP.meta.day} · normal traffic` : `Day ${REP.meta.day} · drifting traffic`;
  let last = performance.now();
  const loop = now => {
    const dt = (now - last) / 1000; last = now;
    if (playing) { t += dt * rate; if (t > 18 * 60) t = 5 * 60; }
    scrub.value = t;
    frame(t, { caption: null });
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
