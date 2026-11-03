/** PLATE N02 — Exponential LIF at rheobase: a ball in a potential well. */
(function (root) {

  'use strict';


  // All student-facing copy is kept together for translation and review.
  const S = {

    title: 'ELIF at rheobase: a ball in a potential well',
source: 'live',
    mirror: 'neurosush.neurons.dynamics.elif_terms · exponential_boost',
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
    reveal: 'Reveal: At 90% of rheobase, initial states below the unstable fixed point relax into the stable well; a sufficient kick can still cause escape.',
    difference: 'fixed-point difference',
glossary: 'Glossary',
    gWell: 'Stable fixed point: a voltage where drift is zero and nearby voltages return.',
    gHump: 'Unstable fixed point: a zero-drift balance that pushes a disturbed voltage away.',
    gRheo: 'Rheobase: the current at which the well and hump merge in a saddle-node.',
    gDelta: 'Δ (mV): exponential voltage slope factor; the associated current scale is Δ/R.',
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
    landscape: 'effective potential U(v) (mV²); dv/dt = −U′(v)/τ',
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
dt: 0.04 } // ms; voltage mV, current nA, resistance MΩ; illustrative parameters
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

  function euler(v, current, p) {
    const q = params(p);
    return v + q.dt * derivative(v, current, q);
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
   merged: true
 }];
 const lo = q.v_rest - 20;
 const hi = q.theta_rh + 25;
 const steps = 150;
 const dx = (hi - lo) / steps;
 let brackets = [];
 let prevX = lo;
 let prevY = numerator(lo, current, q);
 for (let i = 1;
 i <= steps;
 i++) {
 const x = lo + i * dx;
 const y = numerator(x, current, q);
 if (prevY * y <= 0) {
 brackets.push([prevX, x]);
 }
 prevX = x;
 prevY = y;
 }
 let roots = [];
 for (const b of brackets) {
 const root = bisection((v) => numerator(v, current, q), b[0], b[1], 1e-10);
 if (root !== null) {
 if (!roots.some((r) => Math.abs(r - root) < 1e-8)) {
 roots.push(root);
 }
 }
 }
 return roots.map((v) => {
 const eps = 1e-5;
 const slope = (numerator(v + eps, current, q) - numerator(v - eps, current, q)) / (2 * eps);
 return {
 v,
 stable: slope < 0 };
 }
).sort((a,
b) => a.v - b.v);
 }

  function simulate(vStart,
current,
steps,
p) {
 const q = params(p);
 let v = vStart;
 let traj = [v];
 for (let i = 0;
 i < steps;
 i++) {
 const dv = derivative(v, current, q);
 v += q.dt * dv;
 traj.push(v);
 }
 return traj;
 }

  const core = {
    DEFAULTS,
    boost,
    numerator,
    derivative,
    potential,
    rheobase,
    fixedPoints,
    simulate,
    euler,
    mulberry32
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core };
  }

  if (typeof Atlas === 'undefined') return;

  Atlas.Plate.register({
    id: 'N02',
    title: {
      en: 'ELIF at rheobase: a ball in a potential well',
      fa: 'ELIF در ریوبیس: گوی در چاه پتانسیل'
    },
    source: 'live',
    mount(figure,
ctx) {
    const section = figure.parentElement || figure;
    const margin = ctx.margin || document.getElementById('atlas-margin') || section;

    const state = {
      chapter: 1,
      current: 0.9 * rheobase(),
      v: -68,
      running: true,
      theory: true,
      last: performance.now(),
      drag: false,
      history: [],
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
</p><div id="n02-diff" class="atlas-mono"></div>

<details id="n02-learn" ${typeof window !== 'undefined' && window.innerWidth >= 620 ? 'open' : ''} style="margin-top:10px;border-top:1px solid var(--border);padding-top:8px;">
  <summary style="cursor:pointer;font-weight:600;" class="atlas-margin-label">Learn: the math, the brain, the ML analogy</summary>
  <div style="font-size:11px;line-height:1.4;margin-top:6px;display:flex;flex-direction:column;gap:6px;">
    <div><b>1. What you see:</b> A ball rolling in a 1D effective potential landscape representing membrane voltage dynamics in the Exponential Integrate-and-Fire (EIF) model. Tilt and current push fixed points together.</div>
    <div><b>2. The math:</b> Governed by τ · dv/dt = (v_rest - v) + Δ · exp((v - θ_rh)/Δ) + R · I. Effective potential U(v) = 0.5(v - v_rest)² - Δ² · exp((v - θ_rh)/Δ) - R·I·v satisfies dv/dt = -U'(v)/τ. Parameters: τ = 20 ms, v_rest = -70 mV, θ_rh = -50 mV, Δ = 2 mV, R = 10 MΩ. Rheobase I_rh = (θ_rh - v_rest - Δ)/R = 1.8 nA.</div>
    <div><b>3. In the brain:</b> Captures rapid regenerative spike initiation in cortical neurons. The exponential term models fast sodium channel activation near threshold (θ_rh = -50 mV), producing sharp action potential onset without full Hodgkin-Huxley gating variables.</div>
    <div><b>4. Like classic ML:</b> Equivalent to 1D gradient descent in a tilted potential energy landscape where input current tilts the loss function. However, unlike optimization landscapes, there is no trained objective or backpropagation, and escape triggers a discontinuous reset rather than convergence.</div>
  </div>
</details>

<small>${
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
 el('n02-commit').disabled = false;
 }));

    el('n02-commit').onclick = () => {
      const val = predict.querySelector('input:checked')?.value;
      const reveal = el('n02-reveal');
      reveal.style.display = 'inline';
      reveal.textContent = ' ' + S.reveal;
      el('n02-commit').disabled = true;
    };

    let currentLayout = null;

    function pointer(e) {
      const r = canvas.getBoundingClientRect();
      return {
        x: e.clientX - r.left,
        y: e.clientY - r.top }
      ;
    }

    canvas.addEventListener('pointerdown', (e) => {
      const pt = pointer(e);
      if (!currentLayout) return;
      const l = currentLayout;
      if (pt.x >= l.sliderTrackX - 25 && pt.x <= l.sliderTrackX + 25 && pt.y >= l.sliderTop - 10 && pt.y <= l.sliderBottom + 10) {
        state.drag = true;
        updateSlider(pt.y);
      } else if (pt.x >= l.landX0 && pt.x <= l.landX0 + l.landW && pt.y >= l.landTop && pt.y <= l.landBottom) {
        const vMin = -80, vMax = -20;
        state.v = vMin + ((pt.x - l.landX0) / l.landW) * (vMax - vMin);
        state.history = [];
        draw();
      }
    });

    function updateSlider(y) {
      if (!currentLayout) return;
      const l = currentLayout;
      const frac = 1 - Math.max(0, Math.min(1, (y - l.sliderTop) / l.sliderH));
      setCurrent(frac * 1.25);
    }

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
        sliderTrackX,
        sliderTop,
        sliderBottom,
        sliderH
      };

      // Draw landscape potential U(v)
      c.save();
      c.strokeStyle = border;
      c.strokeRect(landX0, landTop, landW, landBottom - landTop);
      text(S.landscape, landX0 + 8, landTop + 18, muted);

      const vMin = -80, vMax = -20;
      const vToX = (v) => landX0 + ((v - vMin) / (vMax - vMin)) * landW;
      const potentials = [];
      let uMin = Infinity, uMax = -Infinity;
      for (let x = 0; x <= landW; x++) {
        const v = vMin + (x / landW) * (vMax - vMin);
        const u = potential(v, current, state.p);
        potentials.push(u);
        if (u < uMin) uMin = u;
        if (u > uMax) uMax = u;
      }
      const uSpan = Math.max(1, uMax - uMin);
      const uToY = (u) => landBottom - 18 - ((u - uMin) / uSpan) * (landBottom - landTop - 45);

      c.strokeStyle = co.ink || '#111';
      c.lineWidth = 2;
      c.beginPath();
      for (let x = 0; x <= landW; x++) {
        const px = landX0 + x;
        const py = uToY(potentials[x]);
        if (x === 0) c.moveTo(px, py); else c.lineTo(px, py);
      }
      c.stroke();

      // Ball position
      const ballX = vToX(state.v);
      const ballU = potential(state.v, current, state.p);
      const ballY = uToY(ballU);
      c.fillStyle = co.exc || '#b45309';
      c.beginPath();
      c.arc(ballX, ballY - 6, 8, 0, Math.PI * 2);
      c.fill();
      c.restore();

      // Draw phase line dv/dt vs v
      c.save();
      c.strokeStyle = border;
      c.strokeRect(hx, hy, pw, ph);
      text(S.phase, hx + 8, hy + 18, muted);

      const pMidY = hy + ph / 2;
      line(hx, pMidY, hx + pw, pMidY, border, [2, 2]);

      c.strokeStyle = co.volt || '#0f766e';
      c.lineWidth = 1.75;
      c.beginPath();
      for (let x = 0; x <= pw; x++) {
        const v = vMin + (x / pw) * (vMax - vMin);
        const dv = derivative(v, current, state.p);
        const px = hx + x;
        const py = pMidY - dv * 12;
        if (x === 0) c.moveTo(px, py); else c.lineTo(px, py);
      }
      c.stroke();

      // Fixed point markers on phase line
      for (const pt of roots) {
        const px = vToX(pt.v);
        const py = pMidY;
        c.fillStyle = pt.stable ? (co.volt || '#0f766e') : paperCard;
        c.strokeStyle = co.volt || '#0f766e';
        c.lineWidth = 2;
        c.beginPath();
        c.arc(px, py, 5, 0, Math.PI * 2);
        c.fill();
        c.stroke();
      }
      c.restore();

      // Draw slider
      c.save();
      c.strokeStyle = border;
      c.lineWidth = 4;
      c.beginPath();
      c.moveTo(sliderTrackX, sliderTop);
      c.lineTo(sliderTrackX, sliderBottom);
      c.stroke();

      c.fillStyle = co.exc || '#b45309';
      c.beginPath();
      c.arc(sliderTrackX, handleY, 9, 0, Math.PI * 2);
      c.fill();
      text(`I = ${current.toFixed(2)} nA`, sliderTrackX + 16, handleY + 4, muted);
      c.restore();

      // Update terms readout
      const nearest = roots.length > 0 ? Math.min(...roots.map(r => Math.abs(r.v - state.v))) : 0;
      const n = numerator(state.v, current, state.p);
      el('n02-terms').innerHTML = `v: ${state.v.toFixed(2)} mV<br>I: ${current.toFixed(2)} nA<br>R·I: ${(state.p.resistance * current).toFixed(3)} mV<br>dv/dt: ${(n / state.p.tau).toFixed(4)} mV/ms`;
      el('n02-iv').textContent = `${(current / rh * 100).toFixed(0)}% I_rh`;
      el('n02-diff').textContent = `${S.difference}: ${nearest.toFixed(4)} mV`;

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
          if (state.v > 30) {
            state.running = false;
            state.v = 30;
            const playBtn = el('n02-play');
            if (playBtn) playBtn.textContent = S.play;
          }
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
});
})(typeof globalThis !== 'undefined' ? globalThis : this);
