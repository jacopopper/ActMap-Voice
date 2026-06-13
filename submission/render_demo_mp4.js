#!/usr/bin/env node

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const WIDTH = 1920;
const HEIGHT = 1080;
const FPS = 24;
const SECONDS = 30;
const FRAME_COUNT = FPS * SECONDS;
const PORT = 9333;
const HTML_PATH = path.join(ROOT, "submission", "actmap_demo.html");
const AUDIO_PATH = path.join(ROOT, "submission", "demo_artifacts", "actmap_voiceover.mp3");
const MANIFEST_PATH = path.join(ROOT, "submission", "demo_artifacts", "actmap_voiceover.json");
const FRAMES_DIR = path.join(ROOT, "submission", "demo_artifacts", "actmap_demo_frames");
const OUTPUT_PATH = path.join(ROOT, "submission", "demo_artifacts", "actmap_voice_demo.mp4");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function httpJson(url, method = "GET") {
  return new Promise((resolve, reject) => {
    const req = http.request(url, { method }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        body += chunk;
      });
      res.on("end", () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error(`HTTP ${res.statusCode}: ${body.slice(0, 300)}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on("error", reject);
    req.end();
  });
}

async function waitForChrome() {
  const url = `http://127.0.0.1:${PORT}/json/version`;
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      await httpJson(url);
      return;
    } catch {
      await sleep(125);
    }
  }
  throw new Error("Chrome remote debugging port did not become ready.");
}

function cdpClient(webSocketUrl) {
  let id = 0;
  const pending = new Map();
  const ws = new WebSocket(webSocketUrl);

  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) {
        reject(new Error(`${message.error.message}: ${message.error.data || ""}`));
      } else {
        resolve(message.result);
      }
    }
  });

  return {
    ready: new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, { once: true });
      ws.addEventListener("error", reject, { once: true });
    }),
    send(method, params = {}) {
      const requestId = ++id;
      const payload = JSON.stringify({ id: requestId, method, params });
      return new Promise((resolve, reject) => {
        pending.set(requestId, { resolve, reject });
        ws.send(payload);
      });
    },
    close() {
      ws.close();
    },
  };
}

function fileUrl(filePath) {
  return `file://${filePath}`;
}

function findChrome() {
  const candidates = [
    process.env.CHROME_BIN,
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error("Could not find Chrome.");
}

function findFfmpeg() {
  const candidates = [
    process.env.FFMPEG_BIN,
    "/home/jacopodardini/.cache/ms-playwright/ffmpeg-1011/ffmpeg-linux",
    "/home/jacopodardini/.local/share/flatpak/runtime/org.freedesktop.Platform/x86_64/25.08/18c4d2cd492ef63f2ba38cd06de0149d2542404144a0e0f6ebe0dfe5f7dbb680/files/bin/ffmpeg",
    "/home/jacopodardini/.local/share/flatpak/runtime/org.gnome.Platform/x86_64/46/1adbb849eea4915c4ca9b5a7718d26a881d7f67f469a11822b52605933bcbe70/files/bin/ffmpeg",
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error("Could not find ffmpeg.");
}

function prepareFramesDir() {
  fs.mkdirSync(FRAMES_DIR, { recursive: true });
  for (const entry of fs.readdirSync(FRAMES_DIR)) {
    if (entry.endsWith(".jpg")) {
      fs.unlinkSync(path.join(FRAMES_DIR, entry));
    }
  }
}

async function openPage() {
  const target = await httpJson(
    `http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(
      `${fileUrl(HTML_PATH)}?t=0&freeze=1&export=1`
    )}`,
    "PUT"
  );
  const client = cdpClient(target.webSocketDebuggerUrl);
  await client.ready;
  await client.send("Page.enable");
  await client.send("Runtime.enable");
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: WIDTH,
    height: HEIGHT,
    deviceScaleFactor: 1,
    mobile: false,
  });

  for (let attempt = 0; attempt < 100; attempt += 1) {
    const result = await client.send("Runtime.evaluate", {
      expression: "Boolean(window.__actmapDemoReady)",
      returnByValue: true,
    });
    if (result.result.value) {
      return client;
    }
    await sleep(50);
  }
  throw new Error("Demo page did not become ready.");
}

async function captureFrames(client) {
  for (let frame = 0; frame < FRAME_COUNT; frame += 1) {
    const seconds = frame / FPS;
    await client.send("Runtime.evaluate", {
      expression: `window.__actmapDemoSetTime(${seconds.toFixed(6)})`,
      awaitPromise: true,
    });
    const screenshot = await client.send("Page.captureScreenshot", {
      format: "jpeg",
      quality: 92,
      fromSurface: true,
      clip: {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
        scale: 1,
      },
    });
    const filename = path.join(FRAMES_DIR, `frame_${String(frame).padStart(4, "0")}.jpg`);
    fs.writeFileSync(filename, Buffer.from(screenshot.data, "base64"));
    if ((frame + 1) % FPS === 0) {
      process.stdout.write(`captured ${(frame + 1) / FPS}s / ${SECONDS}s\r`);
    }
  }
  process.stdout.write(`captured ${SECONDS}s / ${SECONDS}s\n`);
}

function muxMp4(ffmpegPath) {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  const audioDuration = Number(manifest.duration_seconds || 30);
  const atempo = Math.max(0.5, Math.min(2, audioDuration / SECONDS));
  const framePattern = path.join(FRAMES_DIR, "frame_%04d.jpg");
  const args = [
    "-y",
    "-framerate",
    String(FPS),
    "-i",
    framePattern,
    "-i",
    AUDIO_PATH,
    "-filter:a",
    `atempo=${atempo.toFixed(6)},atrim=duration=${SECONDS}`,
    "-t",
    String(SECONDS),
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-b:a",
    "192k",
    "-movflags",
    "+faststart",
    OUTPUT_PATH,
  ];
  const result = spawnSync(ffmpegPath, args, { encoding: "utf8" });
  if (result.status !== 0) {
    process.stderr.write(result.stdout);
    process.stderr.write(result.stderr);
    throw new Error(`ffmpeg failed with exit code ${result.status}`);
  }
  return { output: OUTPUT_PATH, atempo };
}

async function main() {
  if (!fs.existsSync(AUDIO_PATH)) {
    throw new Error(`Missing voiceover audio: ${AUDIO_PATH}`);
  }
  prepareFramesDir();

  const chromePath = findChrome();
  const ffmpegPath = findFfmpeg();
  const profileDir = "/tmp/actmap-demo-chrome-profile";
  fs.rmSync(profileDir, { recursive: true, force: true });
  const chrome = spawn(chromePath, [
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-crash-reporter",
    "--disable-crashpad",
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profileDir}`,
    `--window-size=${WIDTH},${HEIGHT}`,
    "--force-device-scale-factor=1",
    "about:blank",
  ], { stdio: ["ignore", "ignore", "pipe"] });

  let chromeStderr = "";
  chrome.stderr.on("data", (chunk) => {
    chromeStderr += chunk.toString();
  });

  let client;
  try {
    await waitForChrome();
    client = await openPage();
    await captureFrames(client);
    const result = muxMp4(ffmpegPath);
    console.log(`wrote ${result.output}`);
    console.log(`audio_atempo=${result.atempo.toFixed(6)}`);
  } catch (error) {
    if (chromeStderr) {
      process.stderr.write(chromeStderr);
    }
    throw error;
  } finally {
    if (client) {
      client.close();
    }
    chrome.kill("SIGTERM");
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
