import {panels as templates} from './panels.mjs';
import {ZoomEngine, cameraAt, validateReplay} from './engine.mjs';

const q = selector => document.querySelector(selector);
const response = await fetch(new URL('./facts.json', import.meta.url));
if (!response.ok) throw Error('Fact ledger unavailable');
const ledger = await response.json();
const facts = Object.fromEntries(ledger.filter(f => f.status === 'VERIFIED').map(f => [f.id, f.value]));
const panels = templates.map(p => ({
  ...p,
  copy: p.copy.replace(/\{([a-z0-9_]+)\}/g, (_, id) => {
    if (!(id in facts)) throw Error('Missing fact: ' + id);
    return ((id.startsWith('ideal_') && id.endsWith('_score')) || id.endsWith('_margin'))
      ? facts[id].toFixed(3) : facts[id];
  }),
}));

// Paper figures supplement the canvas. A missing optional manifest must not stop the story.
let paperFigures = {};
try {
  const figuresResponse = await fetch(new URL('./paper-figures/manifest.json', import.meta.url));
  if (figuresResponse.ok) paperFigures = await figuresResponse.json();
} catch { /* Keep the schematic presentation available. */ }

// A failed replay is isolated to its panel; the rest of the sourced story still runs.
let replays = [];
try {
  replays = await Promise.all(['v1', 'v2'].map(async variant => {
    const replayResponse = await fetch(new URL(`../data/tem-run-${variant}-s0.json`, import.meta.url));
    if (!replayResponse.ok) throw Error('Recorded run unavailable');
    return validateReplay(await replayResponse.json());
  }));
} catch {
  const replayPanel = panels.find(p => p.scene === 'replay');
  if (replayPanel) {
    replayPanel.scene = 'arena';
    replayPanel.badge = 'SCHEMATIC';
    replayPanel.caption = 'Recorded run unavailable · schematic arena';
    replayPanel.note = 'The recorded run could not load. This arena is a schematic, not model activity. ' + replayPanel.note;
  }
}

let chapter = -1;
let presenting = false;
let showDetails = false;
let showFigure = false;
let calloutIndex = -1;
const paperImage = q('#paper-image');
const figureElement = q('#paper-figure');
const figureButton = q('#figure-toggle');
const highlightButton = q('#highlight-next');
const calloutLayer = q('#paper-callout-layer');

function syncCalloutLayer() {
  if (figureElement.hidden || calloutLayer.hidden) return;
  const imageBox = paperImage.getBoundingClientRect();
  const figureBox = figureElement.getBoundingClientRect();
  calloutLayer.style.left = `${imageBox.left - figureBox.left}px`;
  calloutLayer.style.top = `${imageBox.top - figureBox.top}px`;
  calloutLayer.style.width = `${imageBox.width}px`;
  calloutLayer.style.height = `${imageBox.height}px`;
}
function renderCallout() {
  const callouts = paperFigures?.[chapter]?.callouts;
  const available = presenting && !figureElement.hidden && Array.isArray(callouts) && callouts.length > 0;
  highlightButton.hidden = !available;
  const current = available ? callouts[calloutIndex] : null;
  calloutLayer.hidden = !current;
  const text = q('#paper-highlight-text');
  text.hidden = !current;
  if (!current) {
    highlightButton.textContent = 'Highlight (H)';
    highlightButton.setAttribute('aria-label', 'Show first figure highlight');
    return;
  }
  const ring = q('#paper-callout-ring');
  for (const [property, value] of Object.entries({left: current.x, top: current.y, width: current.w, height: current.h})) {
    ring.style[property] = `${value * 100}%`;
  }
  q('#paper-callout-number').textContent = String(calloutIndex + 1);
  text.textContent = `${calloutIndex + 1} / ${callouts.length} · ${current.label}`;
  highlightButton.textContent = calloutIndex + 1 === callouts.length ? 'Clear highlight (H)' : 'Next highlight (H)';
  highlightButton.setAttribute('aria-label', calloutIndex + 1 === callouts.length ? 'Clear figure highlight' : 'Show next figure highlight');
  requestAnimationFrame(syncCalloutLayer);
}
function cycleCallout() {
  const callouts = paperFigures?.[chapter]?.callouts;
  if (highlightButton.hidden || !Array.isArray(callouts)) return;
  calloutIndex = calloutIndex + 1 >= callouts.length ? -1 : calloutIndex + 1;
  renderCallout();
}

function compactCopy(copy) {
  const sentences = copy.split(/(?<=[.!?])\s+(?=[A-Z])/);
  return sentences.length > 2 ? sentences.slice(0, 2).join(' ') : copy;
}
function renderCopy() {
  if (chapter < 0) return;
  const p = panels[chapter];
  q('#panel-copy').textContent = presenting && !showDetails ? compactCopy(p.copy) : p.copy;
  q('#details-toggle').hidden = !presenting;
  q('#details-toggle').setAttribute('aria-pressed', String(showDetails));
  q('#details-toggle').textContent = showDetails ? 'Short text' : 'Details';
}
function renderFigure() {
  const item = paperFigures?.[chapter];
  const available = Boolean(item && typeof item.src === 'string' && item.src.startsWith('paper-figures/'));
  figureButton.hidden = !presenting || !available;
  const visible = presenting && showFigure && available;
  figureElement.hidden = !visible;
  figureButton.setAttribute('aria-pressed', String(visible));
  figureButton.textContent = visible ? 'Animation' : 'Paper figure';
  if (!visible) { renderCallout(); return; }
  paperImage.src = new URL(item.src, import.meta.url).href;
  paperImage.alt = item.caption || 'Figure from the presentation deck';
  q('#paper-caption').textContent = item.caption || '';
  q('#paper-source').textContent = item.source || 'Adapted from the presentation deck';
  renderCallout();
}
paperImage.addEventListener('load', syncCalloutLayer);
new ResizeObserver(syncCalloutLayer).observe(paperImage);
addEventListener('resize', syncCalloutLayer);
paperImage.addEventListener('error', () => {
  figureElement.hidden = true;
  figureButton.hidden = true;
  figureButton.setAttribute('aria-pressed', 'false');
  calloutIndex = -1;
  renderCallout();
});
function setPresenting(value) {
  presenting = Boolean(value);
  document.body.classList.toggle('presenting', presenting);
  q('#presenter-toggle').setAttribute('aria-pressed', String(presenting));
  q('#presenter-toggle').textContent = presenting ? 'Exit present' : 'Present';
  showDetails = false;
  if (presenting) showFigure = true;
  calloutIndex = -1;
  renderCopy();
  renderFigure();
  engine.resize();
}
function onState(active) {
  if (panels[active].scene === 'replay' && replays.length) {
    const world = engine.replays[engine.replayVariant].worlds[engine.replayWorld];
    q('#recorded-step').value = (Math.floor(engine.phase * 4) % world.walk.length) / (world.walk.length - 1);
  }
  if (chapter === active) return;
  chapter = active;
  q('#story-panel').scrollTop = 0;
  const p = panels[active];
  q('#panel-title').innerHTML = p.title;
  q('#panel-label').textContent = 'THE SPIKING TEM · ' + p.label.toUpperCase();
  q('#replay-controls').hidden = p.scene !== 'replay' || !replays.length;
  q('#panel-copy').dataset.factIds = p.facts.join(' ');
  q('#panel-note').textContent = p.note;
  q('#citation').textContent = p.source;
  q('#camera-badge').textContent = p.badge;
  q('#scene-caption').textContent = p.caption || 'Illustrated mechanisms · not a trained network';
  q('#position').textContent = p.label;
  q('#panel-count').textContent = `${active + 1} / ${panels.length}`;
  q('#previous').disabled = active === 0;
  q('#next').disabled = active === panels.length - 1;
  showDetails = false;
  calloutIndex = -1;
  renderCopy();
  renderFigure();
}

const engine = new ZoomEngine(q('#scene'), q('#overlay'), panels, facts, onState);
engine.replays = replays;
engine.replayVariant = 0;
engine.replayWorld = 0;
for (const [id, key] of [['recording', 'replayVariant'], ['recorded-world', 'replayWorld']]) {
  q('#' + id).addEventListener('change', event => {
    engine[key] = Number(event.target.value);
    engine.phase = 0;
    engine.wake();
  });
}
q('#recorded-step').addEventListener('input', event => {
  if (!replays.length) return;
  engine.setPlaying(false);
  q('#play').textContent = 'Animate';
  q('#play').setAttribute('aria-pressed', 'false');
  const world = engine.replays[engine.replayVariant].worlds[engine.replayWorld];
  engine.phase = Number(event.target.value) * (world.walk.length - 1) / 4;
  engine.wake();
});

q('#scroll-track').style.height = panels.length * 100 + 'svh';
let range = 1;
function resize() { range = Math.max(1, q('#scroll-track').offsetHeight - innerHeight); }
resize();
addEventListener('resize', resize);
addEventListener('scroll', () => { if (!engine.reduced) engine.setTarget(scrollY / range); }, {passive: true});
function navigate(i) {
  const index = Math.max(0, Math.min(panels.length - 1, i));
  scrollTo({top: engine.keys[index].t * range, behavior: 'instant'});
  engine.setTarget(engine.keys[index].t);
}
q('#previous').addEventListener('click', () => navigate(chapter - 1));
q('#next').addEventListener('click', () => navigate(chapter + 1));
addEventListener('keydown', event => {
  if (engine.reduced || q('dialog').open || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.target.closest('input,select,textarea,[contenteditable]')) return;
  const key = event.key;
  if (['ArrowDown', 'ArrowRight', 'PageDown', 'ArrowUp', 'ArrowLeft', 'PageUp', 'Home', 'End'].includes(key)) {
    event.preventDefault();
    navigate(key === 'Home' ? 0 : key === 'End' ? panels.length - 1 : chapter +
      (['ArrowDown', 'ArrowRight', 'PageDown'].includes(key) ? 1 : -1));
  } else if (key === ' ' && !event.target.closest('button,a')) {
    event.preventDefault();
    navigate(chapter + 1);
  } else if (key.toLowerCase() === 'p') setPresenting(!presenting);
  else if (key.toLowerCase() === 'i' && !figureButton.hidden) { showFigure = !showFigure; calloutIndex = -1; renderFigure(); }
  else if (key.toLowerCase() === 'h') cycleCallout();
});
q('#play').addEventListener('click', () => {
  engine.setPlaying(!engine.playing);
  q('#play').setAttribute('aria-pressed', String(engine.playing));
  q('#play').textContent = engine.playing ? 'Pause' : 'Animate';
});
q('#figure-toggle').addEventListener('click', () => { showFigure = !showFigure; calloutIndex = -1; renderFigure(); });
highlightButton.addEventListener('click', cycleCallout);
q('#details-toggle').addEventListener('click', () => { showDetails = !showDetails; renderCopy(); });
q('#presenter-toggle').addEventListener('click', () => setPresenting(!presenting));
q('#fullscreen-toggle').addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch { /* Fullscreen may be disabled by the browser or embed. */ }
});
addEventListener('fullscreenchange', () => {
  const full = Boolean(document.fullscreenElement);
  q('#fullscreen-toggle').textContent = full ? 'Exit full screen' : 'Full screen';
  q('#fullscreen-toggle').setAttribute('aria-pressed', String(full));
  resize();
  engine.resize();
});
q('#theme').addEventListener('click', () => {
  document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  engine.updateTheme();
});
q('#references-toggle').addEventListener('click', () => q('#references').showModal());
q('#close-references').addEventListener('click', () => q('#references').close());
for (const p of panels) {
  const section = document.createElement('p');
  section.textContent = p.label + ': ' + p.facts.map(id => {
    const f = ledger.find(row => row.id === id);
    if (!f || f.status !== 'VERIFIED') throw Error('Unverified fact: ' + id);
    return f.source;
  }).join('; ');
  q('#panel-sources').append(section);
}
const motion = matchMedia('(prefers-reduced-motion: reduce)');
let stillsBuilt = false;
function reducedMotion() {
  engine.reduced = motion.matches;
  resize();
  if (motion.matches) {
    cancelAnimationFrame(engine.raf);
    engine.raf = 0;
    if (!stillsBuilt) {
      const saved = engine.t, savedPhase = engine.phase, savedX = engine.cx, savedY = engine.cy;
      engine.phase = 3;
      engine.cx = engine.w / 2;
      engine.cy = engine.h / 2;
      for (let i = 0; i < panels.length; i++) {
        const p = panels[i];
        engine.t = engine.keys[i].t;
        cameraAt(engine.t, engine.keys, engine.camera);
        engine.draw();
        const section = document.createElement('section'), heading = document.createElement('h2');
        const img = document.createElement('img'), copy = document.createElement('p'), note = document.createElement('p');
        heading.innerHTML = p.title;
        img.src = engine.canvas.toDataURL();
        img.alt = p.label + ' — ' + p.badge;
        img.width = engine.w;
        img.height = engine.h;
        copy.textContent = p.copy;
        note.textContent = p.badge + ' · ' + p.note + ' ' + p.source;
        section.append(heading, img, copy, note);
        q('#reduced-story').append(section);
      }
      const footer = document.createElement('footer'), heading = document.createElement('h2');
      heading.textContent = 'Sources';
      const ledgerLink = document.createElement('a');
      ledgerLink.href = 'facts.json';
      ledgerLink.textContent = 'Complete fact ledger ↗';
      footer.append(heading, q('#references ol').cloneNode(true), ledgerLink);
      q('#reduced-story').append(footer);
      engine.t = saved;
      engine.phase = savedPhase;
      engine.cx = savedX;
      engine.cy = savedY;
      cameraAt(saved, engine.keys, engine.camera);
      stillsBuilt = true;
    }
  } else engine.setTarget(scrollY / range);
}
motion.addEventListener('change', reducedMotion);
addEventListener('visibilitychange', () => { engine.lastTime = 0; if (!document.hidden) engine.wake(); });
window.storyEngine = engine;
onState(0);
reducedMotion();
engine.wake();
document.documentElement.dataset.ready = 'true';
