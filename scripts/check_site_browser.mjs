import {spawn} from 'node:child_process';
import {writeFile, mkdir, readFile, rm} from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';

const shots = process.argv[2] || '/tmp/shots-atlas';
const siteDir = process.argv[3] || '/tmp/site-atlas';
await mkdir(shots, {recursive: true});

// 1. Start static server on free port
const server = http.createServer((req, res) => {
  let filePath = path.join(siteDir, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
  fs.stat(filePath, (err, stats) => {
    if (err || stats.isDirectory()) {
      filePath = path.join(filePath, 'index.html');
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('Not found');
      } else {
        const ext = path.extname(filePath);
        const map = {'.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png'};
        res.writeHead(200, {'Content-Type': map[ext] || 'text/plain'});
        res.end(data);
      }
    });
  });
});

await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const baseUrl = `http://127.0.0.1:${port}`;

// 2. Launch Chrome with random user data dir and remote debugging port 0
const userDataDir = `/tmp/neurosush-chrome-${Date.now()}`;
await mkdir(userDataDir, {recursive: true});
const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new',
  '--enable-unsafe-swiftshader',
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-background-networking',
  '--disable-component-update',
  '--disable-sync',
  '--disable-extensions',
  '--metrics-recording-only',
  '--disable-features=MediaRouter,OptimizationHints',
  '--remote-debugging-port=0',
  `--user-data-dir=${userDataDir}`,
  'about:blank'
], {stdio: 'ignore'});

// Wait for DevToolsActivePort
let wsUrl;
for (let i = 0; i < 80; i++) {
  try {
    const content = await readFile(`${userDataDir}/DevToolsActivePort`, 'utf-8');
    const lines = content.trim().split('\n');
    const dbgPort = lines[0];
    const targets = await (await fetch(`http://127.0.0.1:${dbgPort}/json`)).json();
    const t = targets.find(t => t.type === 'page') || targets[0];
    if (t) {
      wsUrl = t.webSocketDebuggerUrl;
      break;
    }
  } catch {
    await new Promise(r => setTimeout(r, 100));
  }
}

if (!wsUrl) throw Error('Could not connect to Chrome DevTools');

const ws = new WebSocket(wsUrl);
await new Promise(r => ws.addEventListener('open', r, {once: true}));

let next = 0;
const pending = new Map();
ws.addEventListener('message', ({data}) => {
  const m = JSON.parse(data);
  if (m.id) {
    const p = pending.get(m.id);
    pending.delete(m.id);
    if (m.error) p.reject(Error(JSON.stringify(m.error)));
    else p.resolve(m.result);
  }
});

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++next;
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
}

async function evaluate(expression) {
  const res = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (res.exceptionDetails) throw Error(JSON.stringify(res.exceptionDetails));
  return res.result.value;
}

async function ready() {
  for (let i = 0; i < 100; i++) {
    try {
      if (await evaluate(`document.readyState === 'complete' && !!document.querySelector('#theme')`)) return;
    } catch {}
    await new Promise(r => setTimeout(r, 100));
  }
  throw Error('Page did not become ready');
}

await send('Page.enable');
await send('Network.enable');

const viewports = [
  ['mobile-390', 390, 844],
  ['desktop-1440', 1440, 900]
];

const pages = ['gallery.html', 'index.html', 'results.html', 'research.html', 'studio-live.html', 'plates/N01.html', 'plates/Y01.html'];

for (const [label, width, height] of viewports) {
  await send('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: width <= 768});
  for (const page of pages) {
    for (const theme of ['dark', 'light']) {
      try {
        await send('Page.navigate', {url: `${baseUrl}/${page}`});
        await ready();
        await new Promise(r => setTimeout(r, 200));

        // Apply theme
        await evaluate(`document.documentElement.dataset.theme === '${theme}' || document.querySelector('#theme')?.click()`);
        await new Promise(r => setTimeout(r, 100));

        // Check horizontal overflow (scrollWidth <= innerWidth)
        const scrollW = await evaluate('document.documentElement.scrollWidth');
        const innerW = await evaluate('window.innerWidth');
        if (scrollW > innerW + 1.5) {
          const offenders = await evaluate('(() => { const bad = []; for (const el of document.querySelectorAll("*")) { const r = el.getBoundingClientRect(); if (r.right > window.innerWidth + 1.5) bad.push({tag: el.tagName, id: el.id, class: el.className, right: Math.round(r.right)}); } return bad; })()');
          throw Error(`Horizontal overflow: scrollWidth ${scrollW} > innerWidth ${innerW}, offender: ` + JSON.stringify(offenders[0]));
        }

        if (page === 'gallery.html') {
          // Check prereq containment within .atlas-card
          const badPrereqs = await evaluate(`(() => {
            const bad = [];
            for (const card of document.querySelectorAll('.atlas-card')) {
              const prereq = card.querySelector('.prereq');
              if (prereq) {
                const cRect = card.getBoundingClientRect();
                const pRect = prereq.getBoundingClientRect();
                if (pRect.top < cRect.top - 1 || pRect.bottom > cRect.bottom + 1 || pRect.left < cRect.left - 1 || pRect.right > cRect.right + 1) {
                  bad.push({cardClass: card.className});
                }
              }
            }
            return bad;
          })()`);
          if (badPrereqs.length > 0) throw Error('Prerequisite outside card bounding box');
        }

        // Save screenshot
        const snap = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
        await writeFile(`${shots}/${label}-${theme}-${page.replaceAll('/', '-')}.png`, Buffer.from(snap.data, 'base64'));

        console.log(`PASS | page: ${page.padEnd(16)} | viewport: ${label.padEnd(12)} | theme: ${theme}`);
      } catch (err) {
        console.error(`FAIL | page: ${page.padEnd(16)} | viewport: ${label.padEnd(12)} | theme: ${theme} | error: ${err.message}`);
        process.exitCode = 1;
      }
    }
  }
}

ws.close();
chrome.kill();
server.close();
await rm(userDataDir, {recursive: true, force: true});
