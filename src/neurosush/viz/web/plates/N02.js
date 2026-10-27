/** PLATE N02 — Exponential LIF at rheobase: a ball in a potential well. */
(function (root) {

  'use strict';


  // All student-facing copy is kept together for translation and review.
  const S = {

    title: 'ELIF at rheobase: a ball in a potential well',
source: 'live',
    mirror: 'neurosush.neurons.dynamics.elif_derivative · exponential_boost',
    equation: 'τ dv/dt = (v_rest − v) + Δ exp((v − θ_rh)/Δ) + R I',
    chapter: 'CHAPTER',
back: '← Back',
next: 'Next →',
play: 'Play',
pause: 'Pause',
    theory: 'Theory ghost [T]',
reset: 'Reset ball',
free: 'Free play',
    current: 'Tilt / injected current I',
kick: 'Poke ball (+1.8 mV)',
    prediction: 'Predict, then reveal',
question: 'At 90% of rheobase, will the ball escape?',
    stays: 'It stays in the well',
escapes: 'It escapes over the hump',
commit: 'Commit prediction',
    reveal: 'Reveal: 90% of rheobase still has a stable well. The ball relaxes into it.',
    difference: 'fixed-point difference',
glossary: 'Glossary',
    gWell: 'Stable fixed point: a voltage where drift is zero and nearby voltages return.',
    gHump: 'Unstable fixed point: a zero-drift balance that pushes a disturbed voltage away.',
    gRheo: 'Rheobase: the current at which the well and hump merge in a saddle-node.',
    gDelta: 'Δ: the scale of the exponential spike-initiation current.',
    ch1t: '1. A voltage becomes a rolling ball',
ch1b: 'The ball’s horizontal position is voltage. The curve is a potential: its slope comes from the ELIF voltage equation.',
    ch2t: '2. The hump is a balance that cannot hold',
ch2b: 'The filled dot is stable; the hollow dot is unstable. Poke the ball on either side of the hump and watch the direction reverse.',
    ch3t: '3. Current tilts the landscape',
ch3b: 'Drag the amber handle or the faucet. Input current raises the drift everywhere and tips the landscape toward escape.',
    ch4t: '4. Rheobase is a saddle-node',
ch4b: 'At I_rh the two fixed points touch at θ_rh. Above it, no balance remains: voltage accelerates into a spike.',
    ch5t: '5. Why exponential LIF runs away',
ch5b: 'The dashed gray comparison removes the exponential term. Ordinary LIF keeps one passive balance; ELIF has a genuine regenerative onset.',
    landscape: 'potential landscape U(v)',
phase: 'phase line: dv/dt versus v',
stable: 'stable',
unstable: 'unstable',
    lif: 'LIF comparison (no exponential)',
spike: 'ESCAPE / spike initiation',
footer: 'live · JS port of ELIF equations',
  }
;


  const DEFAULTS = Object.freeze({
 tau: 20,
v_rest: -70,
theta_rh: -50,
delta: 2,
resistance: 10,
dt: 0.04 }
);

  function mulberry32(seed) {
 let a = seed >>> 0;
 return function () {
 a = (a + 0x6D2B79F5) >>> 0;
 let t = a;
 t = Math.imul(t ^ (t >>> 15),
t | 1);
 t ^= t + Math.imul(t ^ (t >>> 7),
t | 61);
 return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
 }
;
 }

  function params(overrides) {
 return {
 ...DEFAULTS,
...(overrides || {
}
) }
;
 }

  function boost(v,
p) {
 const q = params(p);
 return q.delta * Math.exp((v - q.theta_rh) / q.delta);
 }

  function numerator(v,
current,
p) {
 const q = params(p);
 return (q.v_rest - v) + boost(v,
q) + q.resistance * current;
 }

  function derivative(v,
current,
p) {
 const q = params(p);
 return numerator(v,
current,
q) / q.tau;
 }

  function potential(v,
current,
p) {
 const q = params(p);
 return 0.5 * (v - q.v_rest) ** 2 - q.delta ** 2 * Math.exp((v - q.theta_rh) / q.delta) - q.resistance * current * v;
 }

  function rheobase(p) {
 const q = params(p);
 return (q.theta_rh - q.v_rest - q.delta) / q.resistance;
 }

  function bisection(fn,
lo,
hi,
tolerance) {
 let a = lo;
 let b = hi;
 let fa = fn(a);
 const tol = tolerance || 1e-10;
 if (fa === 0) return a;
 if (fa * fn(b) > 0) return null;
 for (let i = 0;
 i < 100;
 i++) {
 const mid = (a + b) / 2;
 const fm = fn(mid);
 if (Math.abs(fm) < tol || (b - a) < tol) return mid;
 if (fa * fm <= 0) b = mid;
 else {
 a = mid;
 fa = fm;
 }
 }
 return (a + b) / 2;
 }

  function fixedPoints(current,
p) {
 const q = params(p);
 const rh = rheobase(q);
 if (current > rh + 1e-10) return [];
 if (Math.abs(current - rh) <= 1e-8) return [{
 v: q.theta_rh,
stable: false,
merged: true }
];
 const f = (v) => numerator(v,
current,
q);
 const stable = bisection(f,
q.v_rest - 60,
q.theta_rh);
 const unstable = bisection(f,
q.theta_rh,
q.theta_rh + 25);
 return [stable,
unstable].filter((v) => v !== null).map((v,
i) => ({
 v,
stable: i === 0,
merged: false }
));
 }

  function euler(v,
current,
p) {
 const q = params(p);
 return v + q.dt * derivative(v,
current,
q);
 }

  function simulate(v0,
current,
duration,
p) {
 const q = params(p);
 let v = v0;
 const values = [];
 for (let t = 0;
 t < duration;
 t += q.dt) {
 v = Math.min(30,
euler(v,
current,
q));
 values.push(v);
 if (v >= 29.9) break;
 }
 return values;
 }

  function theoryRoots(current,
p) {
 return fixedPoints(current,
p);
 }

  const core = {
 DEFAULTS,
mulberry32,
boost,
numerator,
derivative,
potential,
rheobase,
bisection,
fixedPoints,
theoryRoots,
euler,
simulate }
;


  function mount(rootEl,
ctx) {

    const figure = ctx.figure || rootEl;

    const margin = ctx.margin || document.getElementById('atlas-margin');

    const section = document.getElementById('atlas-figure') || figure.parentElement || figure;

    const colors = () => ctx.colors ? ctx.colors() : {
 paper: '#f6f2e9',
ink: '#1b1a17',
muted: '#6d6960',
border: '#bcb5a6',
volt: '#bd6d15',
exc: '#167f83',
inh: '#ad4d64',
trace: '#3b6c9e',
theory: '#57544e' }
;

    const state = {
 p: params(),
current: 0.9 * rheobase(),
v: -69,
running: true,
theory: true,
chapter: 1,
drag: false,
last: performance.now(),
history: [],
answer: '',
revealed: false,
lif: false,
flash: 0 }
;

    const chapters = [
      [S.ch1t,
S.ch1b,
0],
[S.ch2t,
S.ch2b,
0.55],
[S.ch3t,
S.ch3b,
0.82],
[S.ch4t,
S.ch4b,
1],
[S.ch5t,
S.ch5b,
0.90],
    ];

    margin.innerHTML = `<section id="n02-margin" style="font-size:12px;
line-height:1.4"><div class="atlas-margin-label">GOVERNING EQUATION</div><code style="display:block;
white-space:normal;
margin:6px 0">${
S.equation}
</code><div class="atlas-margin-label">LIVE TERMS</div><div id="n02-terms" class="atlas-mono"></div><div class="atlas-margin-label" style="margin-top:10px">${
S.glossary}
</div><p><b>well</b> — ${
S.gWell}
</p><p><b>hump</b> — ${
S.gHump}
</p><p><b>rheobase</b> — ${
S.gRheo}
</p><p><b>Δ</b> — ${
S.gDelta}
</p><div id="n02-diff" class="atlas-mono"></div><small>${
S.footer}
<br>${
S.mirror}
</small></section>`;

    let chapterBar = document.getElementById('n02-chapter');

    if (!chapterBar) {
 chapterBar = document.createElement('section');
 chapterBar.id = 'n02-chapter';
 section.insertBefore(chapterBar,
figure);
 }

    chapterBar.style.cssText = 'border-top:1px solid var(--border);padding:8px 0 6px;margin-bottom:6px;';

    chapterBar.innerHTML = `<div style="display:flex;
justify-content:space-between;
gap:8px;
align-items:center"><span class="atlas-margin-label" id="n02-badge"></span><span><button class="atlas-btn" id="n02-prev">${
S.back}
</button> <button class="atlas-btn" id="n02-next">${
S.next}
</button></span></div><div class="atlas-serif" id="n02-copy"></div>`;

    let tools = document.getElementById('n02-tools');

    if (!tools) {
 tools = document.createElement('div');
 tools.id = 'n02-tools';
 section.insertBefore(tools,
figure);
 }

    tools.style.cssText = 'display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:5px 0 8px;';

    tools.innerHTML = `<button class="atlas-btn" id="n02-play">${
S.pause}
</button><button class="atlas-btn" id="n02-kick">${
S.kick}
</button><label class="atlas-serif">${
S.current}
: <input id="n02-current" aria-label="${
S.current}
" type="range" min="0" max="1.25" step="0.01" value="0.9"></label><output id="n02-iv" class="atlas-mono"></output><button class="atlas-btn" id="n02-theory">${
S.theory}
</button><button class="atlas-btn" id="n02-lif">${
S.lif}
</button><button class="atlas-btn" id="n02-reset">${
S.reset}
</button>`;

    const canvas = figure.querySelector('canvas') || ctx.canvas(920,
480);
 const c = canvas.getContext('2d');

    figure.style.setProperty('aspect-ratio', 'auto', 'important');
    figure.style.setProperty('height', '420px', 'important');
    figure.style.setProperty('min-height', '420px', 'important');
    figure.style.setProperty('max-height', 'none', 'important');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
 canvas.setAttribute('aria-label',
'Interactive potential well and phase line. Drag the amber tilt handle or click the landscape to poke the ball.');

    const el = (id) => document.getElementById(id);

    function setChapter(n) {
 state.chapter = Math.max(1,
Math.min(5,
n));
 const ch = chapters[state.chapter - 1];
 state.current = ch[2] * rheobase();
 el('n02-current').value = ch[2];
 el('n02-badge').textContent = `${
S.chapter}
 ${
state.chapter}
 / 5`;
 el('n02-copy').innerHTML = `<b>${
ch[0]}
</b> — ${
ch[1]}
`;
 el('n02-prev').disabled = state.chapter === 1;
 el('n02-next').disabled = state.chapter === 5;
      if (state.chapter === 5) state.lif = true;
      resetBall();
      draw();
    }

    function resetBall() {
      const roots = fixedPoints(state.current, state.p);
      state.v = roots[0] ? roots[0].v - 1.5 : state.p.theta_rh - 4;
      state.history = [];
    }

    function setCurrent(frac) {
      state.current = Math.max(0, Math.min(1.25, frac)) * rheobase(state.p);
      el('n02-current').value = Math.max(0, Math.min(1.25, frac));
      resetBall();
      draw();
    }

    el('n02-current').oninput = (e) => setCurrent(Number(e.target.value));
    el('n02-play').onclick = () => {
      state.running = !state.running;
      el('n02-play').textContent = state.running ? S.pause : S.play;
      draw();
    };

    el('n02-kick').onclick = () => {
      state.v += 1.8;
      state.flash = 18;
      draw();
    };
    el('n02-reset').onclick = () => { resetBall(); draw(); };
    el('n02-theory').onclick = () => {
      state.theory = !state.theory;
      el('n02-theory').classList.toggle('active', state.theory);
      draw();
    };
    el('n02-lif').onclick = () => {
      state.lif = !state.lif;
      el('n02-lif').classList.toggle('active', state.lif);
      draw();
    };
;

    el('n02-prev').onclick = () => setChapter(state.chapter - 1);
 el('n02-next').onclick = () => setChapter(state.chapter + 1);

    const predict = document.createElement('div');
 predict.id = 'n02-predict';
 predict.style.cssText = 'margin:6px 0;padding:7px;border:1px solid var(--border);font-size:12px;';
 predict.innerHTML = `<b>${
S.prediction}
</b> — ${
S.question}
<label><input type="radio" name="n02-answer" value="stay"> ${
S.stays}
</label><label><input type="radio" name="n02-answer" value="escape"> ${
S.escapes}
</label><button class="atlas-btn" id="n02-commit" disabled>${
S.commit}
</button><span id="n02-reveal" style="display:none"></span>`;
 section.insertBefore(predict,
figure);

    predict.querySelectorAll('input').forEach((x) => x.addEventListener('change',
(e) => {
 state.answer = e.target.value;
 el('n02-commit').disabled = false;
 }
));
 el('n02-commit').onclick = () => {
      state.revealed = true;
      el('n02-reveal').style.display = 'inline';
      el('n02-reveal').textContent = (state.answer === 'stay' ? ' ✓ ' : ' — ') + S.reveal;
      state.theory = true;
      draw();
    };

    function pointer(e) {
      const r = canvas.getBoundingClientRect();
      const dpr = devicePixelRatio || 1;
      return {
        x: (e.clientX - r.left) * (canvas.width / dpr) / r.width,
        y: (e.clientY - r.top) * (canvas.height / dpr) / r.height
      };
    }

    let currentLayout = {
      isMobile: false,
      landX0: 18,
      landW: 400,
      splitY: 265,
      sliderTrackX: 800,
      sliderTop: 60,
      sliderBottom: 370,
      sliderH: 310
    };

    function updateSlider(y) {
      const lay = currentLayout;
      const frac = Math.max(0, Math.min(1, (lay.sliderBottom - y) / lay.sliderH));
      setCurrent(frac * 1.25);
    }

    canvas.addEventListener('pointerdown', (e) => {
      const p = pointer(e);
      const lay = currentLayout;
      if (p.x >= lay.sliderTrackX - 35) {
        state.drag = true;
        canvas.setPointerCapture(e.pointerId);
        updateSlider(p.y);
      } else if (!lay.isMobile || p.y < lay.splitY) {
        if (p.x >= lay.landX0 && p.x <= lay.landX0 + lay.landW) {
          state.v = ((p.x - lay.landX0 - 4) / (lay.landW - 8)) * 52 - 92;
          state.flash = 12;
        }
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      if (state.drag) updateSlider(pointer(e).y);
    });

    canvas.addEventListener('pointerup', () => {
      state.drag = false;
    });

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      if (e.key === ' ') {
        e.preventDefault();
        el('n02-play').click();
      }
      if (e.key.toLowerCase() === 't') el('n02-theory').click();
      if (e.key === 'ArrowRight') setChapter(state.chapter + 1);
      if (e.key === 'ArrowLeft') setChapter(state.chapter - 1);
    });

    function line(x1, y1, x2, y2, style, dash) {
      c.save();
      c.strokeStyle = style;
      c.lineWidth = 1.25;
      c.setLineDash(dash || []);
      c.beginPath();
      c.moveTo(x1, y1);
      c.lineTo(x2, y2);
      c.stroke();
      c.restore();
    }

    function text(t, x, y, style, align) {
      c.fillStyle = style;
      c.font = '12px ui-monospace,monospace';
      c.textAlign = align || 'left';
      c.fillText(t, x, y);
    }

    function draw() {
      const dpr = devicePixelRatio || 1,
        rect = canvas.getBoundingClientRect(),
        W = Math.max(300, Math.round(rect.width)),
        isMobile = W < 620,
        targetH = 420;

      if (figure && figure.style) {
        figure.style.setProperty('aspect-ratio', 'auto', 'important');
        figure.style.setProperty('height', `${targetH}px`, 'important');
        figure.style.setProperty('min-height', `${targetH}px`, 'important');
        figure.style.setProperty('max-height', 'none', 'important');
      }
      canvas.style.height = `${targetH}px`;

      const w = Math.round(W * dpr),
        h = Math.round(targetH * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      const H = targetH,
        co = colors(),
        muted = co.muted || co.inkMuted || '#57534e',
        border = co.border || '#d6cfbf',
        paperCard = co.paperCard || (co.paper === '#15171a' ? '#1c2024' : '#eeeae0');

      c.fillStyle = co.paper;
      c.fillRect(0, 0, W, H);

      const current = state.current,
        rh = rheobase(state.p),
        roots = fixedPoints(current, state.p);

      let landX0, landW, landTop, landBottom;
      let hx, hy, pw, ph;
      let sliderTrackX, sliderTop, sliderBottom, sliderH, handleY;
      const splitY = 220;

      if (isMobile) {
        landX0 = 18;
        landW = W - 36;
        landTop = 28;
        landBottom = 195;
        line(18, splitY - 5, W - 18, splitY - 5, border);

        sliderTrackX = W - 48;
        sliderTop = splitY + 28;
        sliderBottom = H - 42;
        sliderH = sliderBottom - sliderTop;
        handleY = sliderBottom - (current / (rh * 1.25)) * sliderH;

        hx = 18;
        hy = splitY + 30;
        pw = Math.max(160, sliderTrackX - 100 - hx);
        ph = H - splitY - 80;
      } else {
        sliderTrackX = W - 52;
        sliderTop = 60;
        sliderBottom = H - 50;
        sliderH = sliderBottom - sliderTop;
        handleY = sliderBottom - (current / (rh * 1.25)) * sliderH;

        pw = Math.min(235, Math.max(170, Math.round(W * 0.27)));
        hx = sliderTrackX - 98 - pw;
        hy = 52;
        ph = H - 125;

        landX0 = 18;
        landW = hx - 42;
        landTop = 32;
        landBottom = H - 58;
      }

      currentLayout = {
        isMobile,
        landX0,
        landW,
        splitY: isMobile ? splitY : H,
        sliderTrackX,
        sliderTop,
        sliderBottom,
        sliderH
      };

      function vToX(v) {
        return landX0 + 4 + ((v + 92) / 52) * (landW - 8);
      }

      const samples = [];
      let lo = Infinity, hi = -Infinity;
      for (let v = -92; v <= -40; v += 0.25) {
        const u = potential(v, current, state.p);
        samples.push([v, u]);
        lo = Math.min(lo, u);
        hi = Math.max(hi, u);
      }
      const yU = (u) => landBottom - ((u - lo) / (hi - lo || 1)) * (landBottom - landTop);

      text(S.landscape, landX0, 18, muted);
      line(landX0, landBottom, landX0 + landW, landBottom, border);

      samples.forEach((a, i) => {
        if (!i) c.beginPath();
        const x = vToX(a[0]), y = yU(a[1]);
        if (!i) c.moveTo(x, y);
        else c.lineTo(x, y);
      });
      c.strokeStyle = co.ink;
      c.lineWidth = 2;
      c.stroke();

      const closeRoots = roots.length === 2 && (vToX(roots[1].v) - vToX(roots[0].v) < 65);
      roots.forEach((r, idx) => {
        const x = vToX(r.v);
        const y = yU(potential(r.v, current, state.p));
        c.beginPath();
        c.arc(x, y, 6, 0, Math.PI * 2);
        c.fillStyle = r.stable ? co.exc : co.paper;
        c.fill();
        c.strokeStyle = co.ink;
        c.lineWidth = 1.25;
        c.stroke();
        const labelY = (closeRoots && idx === 1) ? y - 25 : y - 13;
        text(r.merged ? 'saddle-node' : (r.stable ? S.stable : S.unstable), x, labelY, co.ink, 'center');
        if (state.theory) {
          c.save();
          c.setLineDash([4, 4]);
          c.strokeStyle = co.theory;
          c.beginPath();
          c.arc(x, y, 11, 0, Math.PI * 2);
          c.stroke();
          c.restore();
        }
      });

      const ballClampedV = Math.max(-92, Math.min(-40, state.v));
      const ballX = vToX(ballClampedV);
      const ballY = yU(potential(ballClampedV, current, state.p));
      c.beginPath();
      c.arc(ballX, ballY, 10 + (state.flash ? 2 : 0), 0, Math.PI * 2);
      c.fillStyle = co.volt;
      c.fill();
      c.strokeStyle = co.ink;
      c.lineWidth = 1.5;
      c.stroke();
      text('V = ' + state.v.toFixed(2) + ' mV', ballX, ballY + 26, co.ink, 'center');

      // Phase line inset card box
      c.save();
      c.fillStyle = paperCard;
      c.strokeStyle = border;
      c.lineWidth = 1;
      c.beginPath();
      if (typeof c.roundRect === 'function') {
        c.roundRect(hx - 8, hy - 18, pw + 16, ph + 38, 4);
      } else {
        c.rect(hx - 8, hy - 18, pw + 16, ph + 38);
      }
      c.fill();
      c.stroke();
      c.restore();

      text(S.phase, hx, hy - 4, muted);
      line(hx, hy + ph / 2, hx + pw, hy + ph / 2, border);
      line(hx + pw * 0.2, hy, hx + pw * 0.2, hy + ph, border);

      c.save();
      c.beginPath();
      c.rect(hx - 1, hy - 1, pw + 2, ph + 2);
      c.clip();
      for (let v = -90; v <= -40; v += 0.5) {
        const x = hx + ((v + 92) / 52) * pw;
        const y = hy + ph / 2 - derivative(v, current, state.p) * 18;
        if (v === -90) { c.beginPath(); c.moveTo(x, y); }
        else c.lineTo(x, y);
      }
      c.strokeStyle = co.trace;
      c.lineWidth = 1.5;
      c.stroke();

      roots.forEach((r) => {
        const x = hx + ((r.v + 92) / 52) * pw;
        const y = hy + ph / 2;
        c.beginPath();
        c.arc(x, y, 5, 0, Math.PI * 2);
        c.fillStyle = r.stable ? co.exc : co.paper;
        c.fill();
        c.strokeStyle = co.ink;
        c.lineWidth = 1.25;
        c.stroke();
      });
      c.restore();

      text('dv/dt', hx + 4, hy + 12, co.trace);
      text('v (mV)', hx + pw - 2, hy + ph + 14, muted, 'right');

      // Tilt slider
      line(sliderTrackX, sliderTop, sliderTrackX, sliderBottom, border);
      c.fillStyle = co.volt;
      c.fillRect(sliderTrackX - 8, handleY - 5, 16, 10);
      text('drag tilt', sliderTrackX - 12, handleY + 4, co.volt, 'right');
      text('I = ' + current.toFixed(3) + ' nA', sliderTrackX, sliderBottom + 20, co.volt, 'center');

      if (state.lif) {
        c.save();
        c.setLineDash([5, 4]);
        c.strokeStyle = co.theory;
        c.beginPath();
        for (let v = -92; v <= -40; v += 0.4) {
          const x = vToX(v);
          const y = landBottom - ((0.5 * (v - state.p.v_rest) ** 2 - state.p.resistance * current * v - lo) / (hi - lo || 1)) * (landBottom - landTop);
          if (v === -92) c.moveTo(x, y);
          else c.lineTo(x, y);
        }
        c.stroke();
        c.restore();
        text(S.lif, landX0 + 4, landBottom - 8, co.theory);
      }
      if (state.v > -43) text(S.spike, ballX, landTop + 12, co.inh, 'center');

      const n=numerator(state.v,current,state.p),
nearest=roots.length?Math.min(...roots.map(r=>Math.abs(r.v-state.v))):Infinity;
 el('n02-terms').innerHTML=`leak ${
(state.p.v_rest-state.v).toFixed(2)}
 mV<br>boost ${
boost(state.v,state.p).toFixed(3)}
 mV<br>input ${
(state.p.resistance*current).toFixed(3)}
 mV<br>dv/dt ${
(n/state.p.tau).toFixed(4)}
 mV/ms`;
el('n02-iv').textContent=`${
(current/rh*100).toFixed(0)}
% I_rh`;
el('n02-diff').textContent=`${
S.difference}
: ${
nearest.toFixed(4)}
 mV`;

    }

    function loop() {
      const now = performance.now(),
        elapsed = Math.min(35, now - state.last);
      state.last = now;
      if (state.running) {
        const steps = Math.max(1, Math.round(elapsed / 3));
        for (let i = 0; i < steps; i++) {
          const dv = state.lif
            ? ((state.p.v_rest - state.v) + state.p.resistance * state.current) / state.p.tau
            : derivative(state.v, state.current, state.p);
          state.v = Math.min(32, state.v + state.p.dt * dv);
          if (state.v > 30) state.v = -40;
        }
      }
      state.flash = Math.max(0, state.flash - 1);
      draw();
    }

    if (typeof requestAnimationFrame === 'function') {
      function raf() {
        loop();
        requestAnimationFrame(raf);
      }
      requestAnimationFrame(raf);
    }
    setInterval(loop, 16);
    if (typeof ctx.onFrame === 'function') {
      ctx.onFrame(loop);
    }

    setChapter(1);
    loop();

  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {
 core }
;

  if (typeof Atlas !== 'undefined') Atlas.Plate.register({
 id: 'N02',
title: {
 en: S.title }
,
source: S.source,
mount }
);

}
)(typeof globalThis !== 'undefined' ? globalThis : this);
