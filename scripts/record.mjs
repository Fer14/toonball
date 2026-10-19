// Frame-stepped capture of the game viewer → MP4, then composite with the broadcast.
// usage: node scripts/record.mjs <run_dir> [wii|toy] [pip|side|game]
import { spawn } from "node:child_process";
import { createReadStream, existsSync, statSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [runDir, theme = "wii", layout = "pip"] = process.argv.slice(2);
const W = 1600, H = 900;
const runRel = "/" + path.relative(ROOT, runDir).split(path.sep).join("/");

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json",
                ".mp4": "video/mp4" };
const server = http.createServer((req, res) => {
  const p = path.join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!p.startsWith(ROOT) || !existsSync(p) || statSync(p).isDirectory()) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { "content-type": TYPES[path.extname(p)] ?? "application/octet-stream" });
  createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const browser = await chromium.launch({
  args: ["--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=gl", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: W, height: H } });
page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
page.on("pageerror", (e) => console.error("[page]", e.message));
let videoParam = "";
if (layout !== "game") {
  // the page itself shows the real match (exactly as in the viewer); Chromium needs WebM
  const webm = path.join(runDir, "clip_rec.webm");
  if (!existsSync(webm)) {
    await new Promise((r, j) => spawn("ffmpeg", ["-v", "error", "-y", "-i", path.join(runDir, "clip.mp4"),
      "-vf", "scale=960:-2", "-c:v", "libvpx-vp9", "-b:v", "3M", "-deadline", "realtime", "-cpu-used", "8",
      "-row-mt", "1", "-an", webm], { stdio: "inherit" }).on("close", (c) => (c ? j(new Error(`ffmpeg ${c}`)) : r())));
  }
  videoParam = "&video=clip_rec.webm";
}
await page.goto(`http://127.0.0.1:${port}/viewer/index.html?run=${runRel}&theme=${theme}&layout=${layout}&record=1${videoParam}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 120000 });
const n = Math.min(await page.evaluate(() => window.__N), Number(process.env.MAX_FRAMES || Infinity));
const fps = await page.evaluate(() => window.__fps);

const outMp4 = path.join(runDir, `${layout}_${theme}.mp4`);
const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "image2pipe", "-framerate", String(fps),
  "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
  "-pix_fmt", "yuv420p", outMp4], { stdio: ["pipe", "inherit", "inherit"] });

const t0 = Date.now();
for (let f = 0; f < n; f++) {
  await page.evaluate(([f, seek]) => (seek ? window.__seekStep(f) : window.__step(f)), [f, layout !== "game"]);
  const buf = await page.screenshot({ type: "jpeg", quality: 92 });  // canvas + HTML HUD
  if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
  if (f % 100 === 0) {
    process.stdout.write(`\rframe ${f}/${n}  ${((Date.now() - t0) / (f + 1)).toFixed(0)} ms/frame`);
  }
}
ff.stdin.end();
await new Promise((r) => ff.on("close", r));
await browser.close();
server.close();
console.log(`\nwrote ${outMp4}`);
