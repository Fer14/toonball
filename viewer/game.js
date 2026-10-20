// toonball game view: replays scene.json (tracking recovered from broadcast video) as a
// cartoon match. Themes: "wii" (Mii stadium) and "toy" (toy figures in a kid's bedroom).
//
// URL: /viewer/index.html?run=/runs/<clip>&theme=wii|toy&layout=pip|side|game&record=1
import * as THREE from "three";

const P = new URLSearchParams(location.search);
const RUN = P.get("run") || "/runs/08fd33_0";
const THEME = P.get("theme") || "wii";
const LAYOUT = P.get("layout") || "pip";
const RECORD = P.has("record");
const USE_POSE = P.get("pose") !== "0";     // ?pose=0 → procedural animation only
document.body.classList.add(THEME, LAYOUT, ...(RECORD ? ["record"] : []));

const L = 105, W = 68;                     // pitch metres
const toWorld = (x, y) => [x - L / 2, y - W / 2];
const SCALE = 1.2;                         // cartoon players are drawn a bit larger than life
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const SKIN = ["#f7d1b5", "#e9b48f", "#c98e62", "#a26a42", "#7a4b2c", "#f2c6a0"];
const HAIR = ["#2b1b12", "#5a3a1e", "#c8a165", "#111111", "#8a4b23", "#d9b26f"];
const GK = ["#39ff88", "#ff7a00", "#b04bff", "#ffe600"];

function rng(seed) {                        // deterministic per-player variety
  let s = (seed * 9301 + 49297) % 233280;
  return () => (s = (s * 9301 + 49297) % 233280) / 233280;
}

// ───────────────────────────── data ─────────────────────────────
const scene_ = await (await fetch(`${RUN}/scene.json`)).json();
const FPS = scene_.fps, N = scene_.n_frames;

function fillNulls(a) {                     // nearest-valid fill for camera arrays
  const out = a.slice();
  let last = out.find((v) => v != null) ?? 0;
  for (let i = 0; i < out.length; i++) out[i] = out[i] ?? last, last = out[i];
  return out;
}
function smooth(a, win) {
  const out = new Float32Array(a.length), h = Math.floor(win / 2);
  for (let i = 0; i < a.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - h); j <= Math.min(a.length - 1, i + h); j++) s += a[j], n++;
    out[i] = s / n;
  }
  return out;
}
const ball = scene_.ball;
// follow-cam target: halfway between the real camera's view centre and the ball
const camX = fillNulls(scene_.camera.x), camY = fillNulls(scene_.camera.y);
const ballX = fillNulls(ball.x), ballY = fillNulls(ball.y);
const cam = {
  x: smooth(camX.map((v, i) => lerp(v, ballX[i], 0.5)), Math.round(FPS * 1.5)),
  y: smooth(camY.map((v, i) => lerp(v, ballY[i], 0.6)), Math.round(FPS * 1.5)),
  w: smooth(fillNulls(scene_.camera.w), Math.round(FPS * 2)),
};
const kicksAt = new Map();                  // player id -> [frames]
for (const [f, id] of scene_.kicks) (kicksAt.get(id) ?? kicksAt.set(id, []).get(id)).push(f);

// ball speed per frame (m/s), used for goalkeeper dives
const ballSpeed = new Float32Array(N);
for (let f = 1; f < N; f++) {
  const a = ball.x[f - 1], b = ball.x[f];
  if (a != null && b != null) ballSpeed[f] = Math.hypot(b - a, ball.y[f] - ball.y[f - 1]) * FPS;
}

// per-player derived tracks: heading faces motion, or the ball when standing still
// (goalkeepers always face the ball unless sprinting); gait phase from real stride lengths
for (const p of scene_.players) {
  const n = p.x.length, hd = new Float32Array(n), ph = new Float32Array(n);
  const gk = p.role === "goalkeeper";
  let h = 0, phase = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 4), b = Math.min(n - 1, i + 4);
    const dx = p.x[b] - p.x[a], dy = p.y[b] - p.y[a];
    const sp = p.s[i] ?? 0;
    const f = p.start + i, bx = ball.x[f], by = ball.y[f];
    let target = h;
    if (sp > (gk ? 4 : 0.8)) target = Math.atan2(dx, dy);
    else if (bx != null) target = Math.atan2(bx - p.x[i], by - p.y[i]);
    let d = target - h;
    d = Math.atan2(Math.sin(d), Math.cos(d));
    h += d * (i === 0 ? 1 : 0.15);
    hd[i] = h;
    // one gait cycle (two steps) covers ~1.8 m walking up to ~4.4 m sprinting, i.e. about
    // 3 footsteps/s when jogging (same formula as toonball/pose/angles.stride_length)
    phase += (2 * Math.PI * sp) / FPS / (1.2 + 0.4 * sp);
    ph[i] = phase;
  }
  p.heading = hd; p.phase = ph;

  // goalkeeper dives: a fast ball passing close to a keeper near their goal
  p.dives = [];
  if (gk) {
    for (let i = 0; i < n; i++) {
      const f = p.start + i, bx = ball.x[f], by = ball.y[f];
      if (bx == null || ballSpeed[f] < 11) continue;
      const near = p.x[i] < 25 || p.x[i] > L - 25;
      const ox = bx - p.x[i], oy = by - p.y[i], dist = Math.hypot(ox, oy);
      const last = p.dives[p.dives.length - 1];
      if (near && dist < 3.2 && dist > 0.5 && (!last || f - last.f > 1.5 * FPS)) {
        // side: + = keeper's left, from the cross product with the facing direction
        const fx = Math.sin(hd[i]), fy = Math.cos(hd[i]);
        p.dives.push({ f: f - Math.round(0.15 * FPS), side: Math.sign(fx * oy - fy * ox) || 1 });
      }
    }
  }
}

// ─────────────────────────── renderer ───────────────────────────
const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: RECORD });
renderer.setPixelRatio(RECORD ? 1 : Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.0;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.5, 2000);

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

// ─────────────────────────── textures ───────────────────────────
function canvasTex(w, h, draw, repeat) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  return t;
}

function drawPitchLines(g, s, ox, oy, lw) {
  const X = (x) => ox + x * s, Y = (y) => oy + y * s;
  g.strokeStyle = "rgba(255,255,255,.95)";
  g.lineWidth = lw * s;
  g.strokeRect(X(0), Y(0), L * s, W * s);
  g.beginPath(); g.moveTo(X(L / 2), Y(0)); g.lineTo(X(L / 2), Y(W)); g.stroke();
  g.beginPath(); g.arc(X(L / 2), Y(W / 2), 9.15 * s, 0, Math.PI * 2); g.stroke();
  for (const side of [0, 1]) {
    const x0 = side ? L : 0, dir = side ? -1 : 1;
    const pb = [x0, W / 2 - 20.16, 16.5, 40.32], gb = [x0, W / 2 - 9.16, 5.5, 18.32];
    for (const [x, y, l, w] of [pb, gb]) {
      g.strokeRect(dir > 0 ? X(x) : X(x - l), Y(y), l * s, w * s);
    }
    g.beginPath();
    const a = Math.acos(5.5 / 9.15);
    const cx = X(x0 + dir * 11);
    if (dir > 0) g.arc(cx, Y(W / 2), 9.15 * s, -a, a);
    else g.arc(cx, Y(W / 2), 9.15 * s, Math.PI - a, Math.PI + a);
    g.stroke();
    g.fillStyle = "#fff";
    g.beginPath(); g.arc(cx, Y(W / 2), 0.25 * s, 0, Math.PI * 2); g.fill();
    for (const y of [0, W]) {
      g.beginPath(); g.arc(X(x0), Y(y), 1 * s, 0, Math.PI * 2); g.stroke();
    }
  }
  g.fillStyle = "#fff";
  g.beginPath(); g.arc(X(L / 2), Y(W / 2), 0.3 * s, 0, Math.PI * 2); g.fill();
}

function pitchTexture(theme) {
  const M = 6, s = 20;                       // margin metres, px per metre
  const w = (L + 2 * M) * s, h = (W + 2 * M) * s;
  return [canvasTex(w, h, (g) => {
    const stripes = 18;
    for (let i = 0; i < stripes; i++) {
      g.fillStyle = theme === "toy"
        ? (i % 2 ? "#4cae4f" : "#43a047")
        : (i % 2 ? "#5cc85a" : "#4fb84d");
      g.fillRect(M * s + (i * L * s) / stripes, 0, (L * s) / stripes + 1, h);
    }
    g.fillStyle = theme === "toy" ? "#3d8f40" : "#47a845";
    g.fillRect(0, 0, M * s, h); g.fillRect(w - M * s, 0, M * s, h);
    if (theme === "toy") {                   // felt fibres
      const r = rng(7);
      for (let i = 0; i < 60000; i++) {
        g.fillStyle = `rgba(${r() > .5 ? "255,255,255" : "0,40,0"},${0.05 + r() * 0.06})`;
        g.fillRect(r() * w, r() * h, 2, 2);
      }
    }
    drawPitchLines(g, s, M * s, M * s, theme === "toy" ? 0.22 : 0.15);
    if (theme === "toy") {                   // stitched rug border
      g.strokeStyle = "#f5e6b8"; g.setLineDash([18, 12]); g.lineWidth = 6;
      g.strokeRect(14, 14, w - 28, h - 28);
    }
  }), L + 2 * M, W + 2 * M];
}

// ─────────────────────────── world build ───────────────────────────
const sun = new THREE.DirectionalLight(0xffffff, THEME === "toy" ? 2.2 : 2.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, near: 1, far: 300 });
sun.shadow.bias = -0.0005;
scene.add(sun, sun.target);
scene.add(new THREE.HemisphereLight(THEME === "toy" ? 0xfff1d6 : 0xcfe8ff,
                                    THEME === "toy" ? 0x6b4a2b : 0x4f7a3a, 1.1));

{
  const [tex, pw, ph] = pitchTexture(THEME);
  const pitchMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(pw, ph),
    new THREE.MeshStandardMaterial({ map: tex, roughness: THEME === "toy" ? 1 : 0.85 }),
  );
  pitchMesh.rotation.x = -Math.PI / 2;
  pitchMesh.receiveShadow = true;
  if (THEME === "toy") pitchMesh.position.y = 0.05;
  scene.add(pitchMesh);
}

function goal(side) {
  const g = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 });
  const r = 0.12, gw = 7.32, gh = 2.44, depth = 2.2;
  const post = new THREE.CylinderGeometry(r, r, gh, 12);
  for (const z of [-gw / 2, gw / 2]) {
    const m = new THREE.Mesh(post, white); m.position.set(0, gh / 2, z); m.castShadow = true; g.add(m);
  }
  const bar = new THREE.Mesh(new THREE.CylinderGeometry(r, r, gw + 2 * r, 12), white);
  bar.rotation.x = Math.PI / 2; bar.position.y = gh; bar.castShadow = true; g.add(bar);
  const net = new THREE.Mesh(
    new THREE.BoxGeometry(depth, gh, gw, 6, 6, 18),
    new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.35 }),
  );
  net.position.set(-depth / 2, gh / 2, 0);
  g.add(net);
  g.position.set(side ? L / 2 : -L / 2, 0, 0);
  if (side) g.rotation.y = Math.PI;
  return g;
}
scene.add(goal(0), goal(1));

function crowdTexture() {
  return canvasTex(512, 128, (g, w, h) => {
    g.fillStyle = "#39415a"; g.fillRect(0, 0, w, h);
    const r = rng(3), cols = ["#ff595e", "#ffca3a", "#1982c4", "#ffffff", "#8ac926", "#6a4c93",
                              "#ff924c", "#e9f5ff"];
    for (let row = 0; row < 4; row++) for (let i = 0; i < 21; i++) {
      const x = i * 24 + 12 + (row % 2) * 12, y = row * 32 + 22;
      g.fillStyle = cols[Math.floor(r() * cols.length)];
      g.beginPath(); g.ellipse(x, y, 9, 11, 0, 0, Math.PI * 2); g.fill();
      g.fillStyle = SKIN[Math.floor(r() * SKIN.length)];
      g.beginPath(); g.arc(x, y - 13, 6, 0, Math.PI * 2); g.fill();
    }
  }, [10, 1]);
}

function adBoards(len, z, colours) {
  const tex = canvasTex(1024, 64, (g, w, h) => {
    const words = ["TOONBALL", "OPEN DATA", "GOAL!", "TOONBALL", "PLAY ON", "FAIR PLAY"];
    words.forEach((t, i) => {
      g.fillStyle = colours[i % colours.length]; g.fillRect(i * w / 6, 0, w / 6, h);
      g.fillStyle = "#fff"; g.font = "bold 34px Trebuchet MS, sans-serif";
      g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(t, (i + 0.5) * w / 6, h / 2 + 2);
    });
  }, [len / 60, 1]);
  const m = new THREE.Mesh(new THREE.BoxGeometry(len, 1, 0.2),
                           new THREE.MeshStandardMaterial({ map: tex, emissive: 0x222222, roughness: 0.5 }));
  m.position.set(0, 0.5, z);
  m.castShadow = true;
  return m;
}

function buildStadium() {
  scene.background = canvasTex(4, 512, (g, w, h) => {
    const gr = g.createLinearGradient(0, 0, 0, h);
    gr.addColorStop(0, "#3d9be9"); gr.addColorStop(0.6, "#9fd3ff"); gr.addColorStop(1, "#e8f6ff");
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
  });
  scene.fog = new THREE.Fog(0xcfe9ff, 180, 420);
  const crowd = crowdTexture();
  const mat = new THREE.MeshStandardMaterial({ map: crowd, roughness: 1 });
  const roofMat = new THREE.MeshStandardMaterial({ color: 0xf4f6fb, roughness: 0.6 });
  const board = new THREE.MeshStandardMaterial({ color: 0x2b8cff, emissive: 0x0b3a80, roughness: 0.4 });
  const sides = [
    { len: L + 30, pos: [0, -W / 2 - 9], rot: 0 },
    { len: W + 20, pos: [-L / 2 - 10, 0], rot: Math.PI / 2 },
    { len: W + 20, pos: [L / 2 + 10, 0], rot: -Math.PI / 2 },
  ];
  for (const s of sides) {
    const g = new THREE.Group();
    for (let t = 0; t < 4; t++) {
      const tier = new THREE.Mesh(new THREE.BoxGeometry(s.len, 3, 5), mat);
      tier.position.set(0, 1.5 + t * 3, -t * 5);
      tier.receiveShadow = true;
      g.add(tier);
    }
    const roof = new THREE.Mesh(new THREE.BoxGeometry(s.len, 0.6, 14), roofMat);
    roof.position.set(0, 15, -12); g.add(roof);
    const ad = new THREE.Mesh(new THREE.BoxGeometry(s.len - 20, 1, 0.3), board);
    ad.position.set(0, 0.5, 4.2); g.add(ad);
    g.position.set(s.pos[0], 0, s.pos[1]);
    g.rotation.y = s.rot;
    scene.add(g);
  }
  scene.add(adBoards(L + 10, W / 2 + 5, ["#2b8cff", "#e5293b", "#1d2433"]));
  const track = new THREE.Mesh(new THREE.PlaneGeometry(L + 40, 14),
                               new THREE.MeshStandardMaterial({ color: 0x3f8f45, roughness: 1 }));
  track.rotation.x = -Math.PI / 2; track.position.set(0, -0.01, W / 2 + 12);
  scene.add(track);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(800, 800),
                                new THREE.MeshStandardMaterial({ color: 0x7cc576 }));
  ground.rotation.x = -Math.PI / 2; ground.position.y = -0.02;
  scene.add(ground);
}

function letterBlock(letter, color, size) {
  const tex = canvasTex(256, 256, (g) => {
    g.fillStyle = color; g.fillRect(0, 0, 256, 256);
    g.strokeStyle = "rgba(255,255,255,.7)"; g.lineWidth = 16; g.strokeRect(16, 16, 224, 224);
    g.fillStyle = "#fff"; g.font = "bold 170px Comic Sans MS, sans-serif";
    g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(letter, 128, 140);
  });
  const m = new THREE.Mesh(new THREE.BoxGeometry(size, size, size),
                           new THREE.MeshStandardMaterial({ map: tex, roughness: 0.55 }));
  m.castShadow = m.receiveShadow = true;
  return m;
}

function buildBedroom() {
  // walls with a blue-sky cloud wallpaper, wooden floor, giant toys around the rug
  const wallTex = canvasTex(1024, 512, (g, w, h) => {
    g.fillStyle = "#6fb6ee"; g.fillRect(0, 0, w, h);
    const r = rng(11);
    g.fillStyle = "#ffffff";
    for (let i = 0; i < 14; i++) {
      const x = r() * w, y = r() * h, s = 28 + r() * 22;
      for (const [dx, dy, k] of [[0, 0, 1], [s, 6, .8], [-s, 8, .75], [s * .5, -s * .5, .8]]) {
        g.beginPath(); g.arc(x + dx, y + dy, s * k, 0, Math.PI * 2); g.fill();
      }
    }
  }, [3, 1]);
  scene.background = new THREE.Color(0x6fb6ee);
  const floorTex = canvasTex(512, 512, (g, w, h) => {
    const r = rng(5);
    for (let i = 0; i < 8; i++) {
      const c = 150 + Math.floor(r() * 30);
      g.fillStyle = `rgb(${c},${Math.floor(c * 0.62)},${Math.floor(c * 0.33)})`;
      g.fillRect(0, i * 64, w, 64);
      g.fillStyle = "rgba(60,30,10,.5)"; g.fillRect(0, i * 64, w, 3);
      g.fillRect(Math.floor(r() * w), i * 64, 3, 64);
      for (let k = 0; k < 40; k++) {
        g.fillStyle = `rgba(80,40,10,${r() * .12})`;
        g.fillRect(r() * w, i * 64 + r() * 64, 40 + r() * 80, 2);
      }
    }
  }, [30, 30]);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(700, 700),
                               new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.6 }));
  floor.rotation.x = -Math.PI / 2; floor.position.y = -0.02; floor.receiveShadow = true;
  scene.add(floor);
  const wallMat = new THREE.MeshStandardMaterial({ map: wallTex, roughness: 1 });
  const back = new THREE.Mesh(new THREE.PlaneGeometry(700, 160), wallMat);
  back.position.set(0, 80, -W / 2 - 60);
  scene.add(back);
  for (const sx of [-1, 1]) {
    const side = new THREE.Mesh(new THREE.PlaneGeometry(500, 160), wallMat);
    side.position.set(sx * (L / 2 + 90), 80, 100);
    side.rotation.y = -sx * Math.PI / 2;
    scene.add(side);
  }
  const skirting = new THREE.Mesh(new THREE.BoxGeometry(700, 4, 1),
                                  new THREE.MeshStandardMaterial({ color: 0xffffff }));
  skirting.position.set(0, 2, -W / 2 - 59.5);
  scene.add(skirting);

  // giant ABC blocks stacked along the far side, like toy grandstands
  const cols = ["#e8542c", "#2b8cff", "#f6c026", "#3bb273", "#9b5de5"];
  const letters = "TOONBALLTOYBOX";
  let k = 0;
  for (let x = -L / 2 - 4; x < L / 2 + 6; x += 9) {
    const n = 1 + ((k * 7) % 2);
    for (let j = 0; j < n; j++) {
      const b = letterBlock(letters[(k + j) % letters.length], cols[(k + j) % cols.length], 8);
      b.position.set(x + (j % 2) * 1.5, 4 + j * 8, -W / 2 - 22 - (k % 2) * 4);
      b.rotation.y = ((k * 13 + j * 7) % 10 - 5) * 0.03;
      scene.add(b);
    }
    k++;
  }
  // corner toys: a big ball, a crayon, a stack of rings
  const bigBall = new THREE.Mesh(new THREE.SphereGeometry(9, 32, 24), new THREE.MeshPhysicalMaterial({
    map: canvasTex(512, 256, (g, w, h) => {
      const c = ["#e63946", "#ffd166", "#118ab2", "#ffffff"];
      for (let i = 0; i < 8; i++) { g.fillStyle = c[i % 4]; g.fillRect(i * w / 8, 0, w / 8 + 1, h); }
    }), roughness: 0.25, clearcoat: 1,
  }));
  bigBall.position.set(-L / 2 - 22, 9, -W / 2 - 4);
  bigBall.castShadow = true;
  scene.add(bigBall);
  const crayon = new THREE.Group();
  const cMat = new THREE.MeshStandardMaterial({ color: 0x7b2cbf, roughness: 0.5 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 1.6, 26, 6), cMat);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(1.6, 4, 6), cMat);
  tip.position.y = 15;
  crayon.add(body, tip);
  crayon.rotation.z = Math.PI / 2; crayon.rotation.y = 0.4;
  crayon.position.set(L / 2 + 20, 1.6, -W / 2 - 4);
  crayon.traverse((o) => (o.castShadow = true));
  scene.add(crayon);
  const ringCols = [0xe63946, 0xf77f00, 0xfcbf49, 0x06d6a0, 0x118ab2];
  ringCols.forEach((c, i) => {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(5 - i * 0.7, 1.4, 16, 32),
                                new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.3, clearcoat: 1 }));
    ring.rotation.x = Math.PI / 2;
    ring.position.set(L / 2 + 24, 1.4 + i * 2.6, -W / 2 + 18);
    ring.castShadow = true;
    scene.add(ring);
  });
}

THEME === "toy" ? buildBedroom() : buildStadium();

// ─────────────────────────── characters ───────────────────────────

function shade(hex, k) {
  const c = new THREE.Color(hex);
  return c.multiplyScalar(k);
}

function makePlayer(p) {
  const r = rng(p.id + 1);
  const role = p.role;
  // goalkeepers wear their real kit colour (measured from their crops) when we have it
  let shirt = role === "referee" ? "#1b1b1b"
    : role === "goalkeeper" ? (p.colour ?? GK[(p.team + 2) % GK.length])
    : (scene_.teams[p.team] ?? "#bbbbbb");
  const shorts = role === "referee" ? "#1b1b1b" : shade(shirt, 0.35);
  const toy = THEME === "toy";
  const mats = [];
  const M = (color, extra = {}) => {
    const m = toy
      ? new THREE.MeshPhysicalMaterial({ color, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.15, ...extra })
      : new THREE.MeshStandardMaterial({ color, roughness: 0.55, ...extra });
    m.transparent = true;
    mats.push(m);
    return m;
  };
  const skin = M(SKIN[Math.floor(r() * SKIN.length)]);
  const shirtM = M(shirt), shortsM = M(shorts), hairM = M(HAIR[Math.floor(r() * HAIR.length)]);
  const sockM = M(role === "referee" ? "#1b1b1b" : shirt), bootM = M("#202020");
  const eyeM = M("#151515");

  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  const add = (parent, geo, mat, x, y, z) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z); m.castShadow = true;
    parent.add(m);
    return m;
  };
  // two-segment limbs: hip → knee → boot, shoulder → elbow → hand
  const legs = [], knees = [], arms = [], elbows = [];
  for (const sx of [-1, 1]) {
    const hip = new THREE.Group(); hip.position.set(0.14 * sx, 0.88, 0); body.add(hip);
    add(hip, new THREE.CapsuleGeometry(0.1, 0.2, 4, 10), skin, 0, -0.2, 0);
    const knee = new THREE.Group(); knee.position.set(0, -0.4, 0); hip.add(knee);
    add(knee, new THREE.SphereGeometry(0.1, 10, 8), skin, 0, 0, 0);
    add(knee, new THREE.CylinderGeometry(0.105, 0.095, 0.3, 10), sockM, 0, -0.16, 0);
    add(knee, new THREE.BoxGeometry(0.16, 0.1, 0.3), bootM, 0, -0.36, 0.06);
    legs.push(hip); knees.push(knee);
    const sh = new THREE.Group(); sh.position.set(0.34 * sx, 1.42, 0); body.add(sh);
    add(sh, new THREE.SphereGeometry(0.12, 12, 10), shirtM, 0, 0, 0);
    add(sh, new THREE.CapsuleGeometry(0.075, 0.13, 4, 10), skin, 0, -0.14, 0);
    const elbow = new THREE.Group(); elbow.position.set(0, -0.28, 0); sh.add(elbow);
    add(elbow, new THREE.CapsuleGeometry(0.07, 0.12, 4, 10), skin, 0, -0.12, 0);
    if (role === "goalkeeper") add(elbow, new THREE.SphereGeometry(0.11, 12, 10), M("#f2f2f2"), 0, -0.24, 0);
    sh.rotation.z = 0.12 * sx;
    arms.push(sh); elbows.push(elbow);
  }
  add(body, new THREE.CylinderGeometry(0.28, 0.26, 0.26, 16), shortsM, 0, 0.92, 0);
  add(body, new THREE.CapsuleGeometry(0.28, 0.3, 6, 16), shirtM, 0, 1.25, 0);
  const head = new THREE.Group(); head.position.set(0, 1.92, 0); body.add(head);
  add(head, new THREE.SphereGeometry(0.33, 24, 18), skin, 0, 0, 0);
  add(head, new THREE.SphereGeometry(0.345, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.45),
      hairM, 0, 0.02, -0.03);
  for (const sx of [-1, 1]) add(head, new THREE.SphereGeometry(0.045, 10, 8), eyeM, 0.11 * sx, 0.04, 0.3);
  if (role === "referee") {
    add(body, new THREE.BoxGeometry(0.58, 0.08, 0.5), M("#ffd400"), 0, 1.45, 0);
  }
  let base = null;
  if (toy) {                                  // table-football style base
    base = add(root, new THREE.CylinderGeometry(0.42, 0.5, 0.16, 24), M(shade(shirt, 0.8)), 0, 0.08, 0);
    body.position.y = 0.14;
  }
  root.scale.setScalar(SCALE);
  scene.add(root);
  return { p, root, body, legs, knees, arms, elbows, head, base, mats, kicks: kicksAt.get(p.id) ?? [] };
}

const players = scene_.players.map(makePlayer);

// ball + blob shadow
const ballMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.22 * SCALE * (THEME === "toy" ? 1.3 : 1.0), 24, 16),
  new THREE.MeshStandardMaterial({
    roughness: 0.35,
    map: canvasTex(256, 128, (g, w, h) => {
      g.fillStyle = "#fff"; g.fillRect(0, 0, w, h);
      g.fillStyle = THEME === "toy" ? "#e63946" : "#222";
      for (let i = 0; i < 6; i++) for (let j = 0; j < 3; j++) {
        g.beginPath(); g.arc(i * 44 + (j % 2) * 22 + 10, j * 44 + 20, 11, 0, Math.PI * 2); g.fill();
      }
    }),
  }),
);
ballMesh.castShadow = true;
scene.add(ballMesh);
// blob shadow straight below the ball: makes its height readable when it is in the air
const ballBlob = new THREE.Mesh(new THREE.CircleGeometry(0.35, 24),
  new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }));
ballBlob.rotation.x = -Math.PI / 2;
scene.add(ballBlob);

const marker = new THREE.Mesh(new THREE.ConeGeometry(0.35, 0.6, 3),
                              new THREE.MeshBasicMaterial({ color: 0xffe14d }));
marker.rotation.x = Math.PI;
scene.add(marker);

// ─────────────────────────── per-frame update ───────────────────────────
const at = (arr, i) => arr[clamp(i, 0, arr.length - 1)];
function sample(arr, f) {                     // linear interpolation, null aware
  const i = Math.floor(f), t = f - i;
  const a = at(arr, i), b = at(arr, i + 1);
  if (a == null) return b;
  if (b == null) return a;
  return lerp(a, b, t);
}

function updatePlayer(pl, f) {
  const p = pl.p, i = f - p.start;
  if (i < 0 || i > p.x.length - 1) { pl.root.visible = false; return; }
  const alpha = sample(p.a, i);
  pl.root.visible = alpha > 0.02;
  if (!pl.root.visible) return;
  const [x, z] = toWorld(sample(p.x, i), sample(p.y, i));
  pl.root.position.set(x, 0, z);
  pl.root.rotation.y = sample(p.heading, i);
  // fade in / out when someone enters or leaves the camera view
  for (const m of pl.mats) { m.opacity = alpha; m.depthWrite = alpha > 0.99; }

  // pose from the video (toonball/pose): gait timing + stride size, knee / elbow bend, lean,
  // sideways tilt — blended with the procedural cycle by how reliable the pose is
  const pose = USE_POSE ? p.pose : null;
  const pw = pose ? clamp(sample(pose.w, i) ?? 0, 0, 1) : 0;
  const speed = sample(p.s, i) ?? 0;
  const ph = pose ? sample(pose.phase, i) : sample(p.phase, i);
  const amp = clamp(speed / 8, 0, 1);
  const moving = clamp((speed - 0.3) / 0.7, 0, 1);       // blend into standing still
  const swingAmp = lerp((0.12 + 0.45 * amp) * moving, clamp(sample(pose?.amp ?? [], i) ?? 0, 0, 0.8), pw);
  const swing = Math.sin(ph) * swingAmp;
  const baseY = pl.base ? 0.14 : 0;
  const kneePose = sample(pose?.knee ?? [], i), elbowPose = sample(pose?.elbow ?? [], i);
  for (const [k, sx] of [[0, -1], [1, 1]]) {
    pl.legs[k].rotation.set(-sx * swing, 0, 0);
    pl.arms[k].rotation.set(sx * swing * 0.8, 0, 0.12 * sx);
    // knees bend most while the leg swings back; elbows stay bent more when running fast
    const kProc = 0.08 + (0.2 + 0.9 * amp) * moving * Math.max(0, sx * Math.sin(ph));
    pl.knees[k].rotation.set(kneePose == null ? kProc : kProc + pw * (kneePose - 0.08 - 0.3 * amp * moving), 0, 0);
    const eProc = 0.25 + 0.9 * amp * moving;
    pl.elbows[k].rotation.set(-(elbowPose == null ? eProc : lerp(eProc, elbowPose, pw)), 0, 0);
  }
  const leanPose = sample(pose?.lean ?? [], i);
  pl.body.rotation.set(leanPose == null ? 0.1 * amp : lerp(0.1 * amp, clamp(leanPose, -0.3, 0.5), pw), 0, 0);
  const rollPose = sample(pose?.roll ?? [], i), rw = pose ? clamp(sample(pose.wr, i) ?? 0, 0, 1) : 0;
  if (rollPose != null) pl.body.rotation.z = -clamp(rollPose, -0.3, 0.3) * rw * 0.8;
  pl.body.position.y = baseY + Math.abs(Math.cos(ph)) * 0.05 * amp;
  pl.root.position.y = THEME === "toy" ? Math.abs(Math.sin(ph)) * 0.18 * amp : 0;
  pl.head.rotation.y = Math.sin(f * 0.05 + p.id) * 0.15;

  if (p.role === "goalkeeper") animateKeeper(pl, f, i, speed, ph, baseY);

  for (const kf of pl.kicks) {                // kick: right leg winds back then strikes
    const k = (f - kf) / (0.35 * FPS);
    if (k >= -0.4 && k <= 1) {
      const a = k < 0 ? -1.1 * (k + 0.4) / 0.4 : lerp(-1.1, 1.3, Math.sin(k * Math.PI / 2));
      pl.legs[1].rotation.x = -a;
      pl.arms[0].rotation.x = 0.7;
      pl.arms[1].rotation.x = -0.5;
      pl.body.rotation.x = -0.15;
    }
  }
}

// goalkeeper: ready stance facing the ball, side-steps, and dives at fast shots
function animateKeeper(pl, f, i, speed, ph, baseY) {
  const p = pl.p;
  const ballNear = (() => {
    const bx = ball.x[Math.round(f)];
    return bx != null && Math.hypot(bx - p.x[Math.round(i)], ball.y[Math.round(f)] - p.y[Math.round(i)]) < 35;
  })();
  if (speed < 4 && ballNear) {
    // lateral component of the motion relative to the facing direction → side-steps
    const j = Math.round(i), a = Math.max(0, j - 2), b = Math.min(p.x.length - 1, j + 2);
    const vx = (p.x[b] - p.x[a]) / ((b - a) / FPS || 1), vy = (p.y[b] - p.y[a]) / ((b - a) / FPS || 1);
    const h = p.heading[j];
    const lateral = Math.abs(vx * Math.cos(h) - vy * Math.sin(h));
    const step = clamp(lateral / 2.5, 0, 1) * Math.sin(ph * 1.6);
    for (const [k, sx] of [[0, -1], [1, 1]]) {
      pl.legs[k].rotation.set(0, 0, sx * (0.2 + 0.25 * Math.max(0, sx * step)));
      pl.arms[k].rotation.set(-0.45, 0, sx * 0.75);
    }
    pl.body.rotation.set(0.28, 0, 0);
    pl.body.position.y = baseY - 0.12;
    for (let k = 0; k < 2; k++) { pl.knees[k].rotation.x = 0.55; pl.elbows[k].rotation.x = -0.6; }
  }
  for (const d of p.dives) {
    const k = (f - d.f) / (0.85 * FPS);
    if (k < 0 || k > 1) continue;
    const reach = Math.sin(Math.min(k / 0.45, 1) * Math.PI / 2);   // launch, then hold
    const fall = k > 0.7 ? (k - 0.7) / 0.3 : 0;                     // land
    pl.body.rotation.set(0, 0, -d.side * 1.35 * reach);
    pl.body.position.y = baseY + 0.55 * Math.sin(Math.min(k, 0.7) / 0.7 * Math.PI) * (1 - fall);
    for (const [kk, sx] of [[0, -1], [1, 1]]) {
      pl.arms[kk].rotation.set(0, 0, sx * lerp(0.75, 2.9, reach));
      pl.legs[kk].rotation.set(0, 0, sx * 0.08);
    }
  }
}

const clockEl = document.getElementById("clock");
document.getElementById("chip0").style.background = scene_.teams[0];
document.getElementById("chip1").style.background = scene_.teams[1];
if (THEME === "toy") document.getElementById("brand").textContent = "TOY BOX FOOTBALL";

const camPos = new THREE.Vector3(), camTarget = new THREE.Vector3();
function update(f) {
  f = clamp(f, 0, N - 1);
  for (const pl of players) updatePlayer(pl, f);

  const bx = sample(ball.x, f), by = sample(ball.y, f), bz = sample(ball.z, f) ?? 0;
  ballMesh.visible = ballBlob.visible = bx != null;
  if (bx != null) {
    const [x, z] = toWorld(bx, by);
    const r = ballMesh.geometry.parameters.radius;
    ballMesh.position.set(x, r + bz + (THEME === "toy" ? 0.05 : 0), z);
    ballBlob.position.set(x, THEME === "toy" ? 0.07 : 0.02, z);
    ballBlob.scale.setScalar(1 + bz * 0.08);
    ballBlob.material.opacity = 0.35 / (1 + bz * 0.25);
    const i = Math.floor(f);
    const dx = (at(ball.x, i + 1) ?? bx) - bx, dz = (at(ball.y, i + 1) ?? by) - by;
    ballMesh.rotation.x += dz / r; ballMesh.rotation.z -= dx / r;
  }
  const owner = scene_.owner[Math.round(f)];
  const op = owner >= 0 ? players.find((q) => q.p.id === owner) : null;
  marker.visible = !!(op && op.root.visible);
  if (marker.visible) {
    marker.position.copy(op.root.position);
    marker.position.y = 3.3 * SCALE + Math.sin(f * 0.25) * 0.15;
    marker.rotation.y = f * 0.08;
    marker.material.color.set(scene_.teams[op.p.team] ?? "#ffe14d").offsetHSL(0, 0, 0.15);
  }

  // broadcast-style camera following the real camera's view on the pitch
  const i = Math.round(f);
  const cx = cam.x[i] - L / 2, cz = cam.y[i] - W / 2;
  const viewW = clamp(cam.w[i] * 1.3, 45, 70);
  const hfov = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * camera.aspect);
  const dist = viewW / 2 / Math.tan(hfov / 2);
  const elev = THREE.MathUtils.degToRad(THEME === "toy" ? 17 : 20);
  // keep the bottom edge of the screen at the near touchline (no empty foreground)
  const hgt = dist * Math.sin(elev), back = dist * Math.cos(elev);
  const nearReach = back - hgt / Math.tan(elev + THREE.MathUtils.degToRad(camera.fov) / 2);
  const zMax = W / 2 + 3 - nearReach;
  camTarget.set(clamp(cx, -L / 2 + 5, L / 2 - 5), 0, clamp(cz, Math.min(-W / 2 + 10, zMax), zMax));
  camPos.set(lerp(camTarget.x, 0, 0.25), dist * Math.sin(elev), camTarget.z + dist * Math.cos(elev));
  camera.position.copy(camPos);
  camera.lookAt(camTarget);
  sun.position.set(camTarget.x - 30, 60, camTarget.z + 20);
  sun.target.position.copy(camTarget);

  const t = f / FPS;
  clockEl.textContent = `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  renderer.render(scene, camera);
}

// ─────────────────────────── playback / recording ───────────────────────────
window.__N = N;
window.__fps = FPS;
window.__step = (f) => { update(f); return true; };
// recording with the real match inset: headless Chromium doesn't repaint a seeked <video>
// before screenshots, so the inset shows one extracted image per frame instead
window.__seekStep = async (f) => {
  const img = document.getElementById("broadcast");
  img.src = `${RUN}/${P.get("frames")}/${String(f + 1).padStart(5, "0")}.jpg`;
  await img.decode();
  update(f);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return true;
};
window.__frame = (f, q = 0.92) => { update(f); return renderer.domElement.toDataURL("image/jpeg", q); };

if (!RECORD) {
  const video = document.getElementById("broadcast");
  video.src = `${RUN}/${scene_.video}`;
  const seek = document.getElementById("seek"), play = document.getElementById("play");
  const themeSel = document.getElementById("theme"), layoutSel = document.getElementById("layout");
  const poseBox = document.getElementById("pose");
  themeSel.value = THEME; layoutSel.value = LAYOUT; poseBox.checked = USE_POSE;
  const go = () => {
    P.set("theme", themeSel.value); P.set("layout", layoutSel.value);
    P.set("pose", poseBox.checked ? "1" : "0");
    location.search = P;
  };
  themeSel.onchange = layoutSel.onchange = poseBox.onchange = go;
  seek.max = N - 1;
  play.onclick = () => (video.paused ? video.play() : video.pause());
  video.onplay = () => (play.textContent = "❚❚");
  video.onpause = () => (play.textContent = "▶");
  seek.oninput = () => (video.currentTime = seek.value / FPS);
  video.loop = true;
  const loop = () => {
    const f = video.currentTime * FPS;
    seek.value = Math.floor(f);
    update(f);
    requestAnimationFrame(loop);
  };
  loop();
  video.play().catch(() => {});
}
if (RECORD && LAYOUT !== "game") {
  const img = document.createElement("img");  // same id → same CSS as the <video>
  img.id = "broadcast";
  document.getElementById("broadcast").replaceWith(img);
}
window.__ready = true;
