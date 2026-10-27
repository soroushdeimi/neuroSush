/**
 * PLATE M01 — Why (1 - dt/tau)^k? Euler's method, convergence and instability.
 * A decay ball slides along Euler's staircase against the exact exponential ghost;
 * the student drags dt and watches it hug the curve, overshoot, oscillate, blow up.
 */
(function (root) {
  'use strict';

  // ---- strings (English) ----
  const S = {
    title: 'Why (1 - dt/tau)^k · Euler step, convergence and instability',
    chapters: [
      {
        name: 'A decay and its tangent',
        cap: 'y obeys dy/dt = -y/tau: it always slides downhill, steeper when it is high. Watch small Euler steps follow the tangent and hug the exact curve.',
        dt: 1,
        mode: 'euler',
        pin: 'tangent',
      },
      {
        name: 'One Euler step by hand',
        cap: 'One giant step of dt = tau = 20 ms: the factor is (1 - 20/20) = 0, so y lands flat on zero in a single jump, while the exact answer is e^-1 = 0.368.',
        dt: 20,
        mode: 'euler',
        pin: 'numbers',
      },
      {
        name: 'Many steps: the staircase',
        cap: 'With dt = 5 ms the tangent segments form a staircase: each corner lands on (1 - dt/tau)^k. The corners sit just above the exact curve.',
        dt: 5,
        mode: 'euler',
        pin: null,
      },
      {
        name: 'Smaller dt, smaller error',
        cap: 'The error after 100 ms is proportional to dt: halve dt and the error halves too. The pane below shows the fitted slope ~= 1 (first order).',
        dt: 1,
        mode: 'euler',
        pin: 'error-pane',
      },
      {
        name: 'dt too big: oscillation and blow-up',
        cap: 'Past dt = tau the corners alternate sign; past dt = 2tau = 40 ms the magnitude grows every step and the method blows up. Drag dt across the limit.',
        dt: 44,
        mode: 'euler',
        pin: 'stability',
      },
      {
        name: 'The fix: exact exponential step',
        cap: 'Replacing the factor (1 - dt/tau) with e^(-dt/tau) is exact for this decay and unconditionally stable: the ball rides the ghost for any dt, to machine precision.',
        dt: 45,
        mode: 'exact',
        pin: 'fix',
      },
    ],
    eqTitle: 'the equations',
    eqEuler: 'Euler',
    eqExact: 'exact',
    liveTitle: 'live values',
    glossTitle: 'glossary',
    footer: 'source: live · mirrors neurosush.neurons.dynamics.euler_step and the trace decay (1 - dt/tau)^k',
    btnPlay: 'Play',
    btnPause: 'Pause',
    btnHalve: 'Halve dt',
    btnPredict: 'Predict',
    btnTheory: 'Theory',
    btnEuler: 'Euler',
    btnExact: 'exact step',
    btnNext: 'Next',
    btnBack: 'Back',
    chapterOf: 'Chapter',
    predictQuestion: 'With dt = {dt} ms, where does the Euler staircase land on its last corner in [0, 100] ms?',
    predictHint: 'Pick a value, commit, then watch the reveal.',
    predictCorrect: 'the staircase corner lands there.',
    predictOff: 'the staircase lands on {euler}; the exact decay is {exact} (difference {diff}).',
    predictAgain: 'Guess again',
    lblEulerLanding: 'Euler corner',
    lblExactLanding: 'exact',
    lblYourPick: 'your pick',
    valDt: 'dt',
    valTau: 'tau',
    valFactor: '(1 - dt/tau)',
    valK: 'k (steps taken)',
    valY: 'y at playhead',
    valDiff: 'difference',
    valStable: 'stable? dt < 2*tau',
    valYes: 'yes',
    valNo: 'NO',
    valSlope: 'slope',
    valFirstOrder: 'first order: error ~ dt',
    glosTau: 'tau: the decay time; y falls to e^-1 = 37% of its value after tau = 20 ms.',
    glosEuler: 'Euler step: follow the tangent line for one step dt, take the landing point, repeat.',
    glosGhost: 'theory ghost: the exact answer y = exp(-t/tau), dashed. Toggle with T.',
    glosOrder: 'first order: the error after a fixed total time shrinks in proportion to dt.',
    glosStability: 'stability: Euler explodes once dt >= 2*tau; between tau and 2*tau it only swings sign.',
    glosExact: 'exact step: multiplying by e^(-dt/tau) solves this ODE exactly at the grid points.',
    ariaCanvas: 'Animated decay: Euler staircase of y(t) against the exact exponential exp(-t/tau); drag the dt ruler or scrub the playhead.',
    errAt100: 'error at t = 100 ms',
    axisT: 't (ms)',
    axisY: 'y',
    axisDt: 'dt (ms, log)',
    zoneSafe: 'safe zone: dt < 2tau',
    lblBlown: 'blown up',
    chipStable: 'stable: dt < 2tau',
    chipSwing: 'swinging: tau < dt < 2tau',
    chipBlown: 'blown up: dt > 2tau',
    signFlipNote: 'sign flips every step',
    lblYour: 'your dt',
    scrubHint: 'drag the plot to scrub the ball · drag the ruler band below to change dt',
    scrubHintShort: 'drag to scrub · drag the dt ruler below',
    legGhost: 'exact exp(-t/tau)',
    legStair: 'Euler staircase',
  };

  // ---- core: pure math, no DOM, exported for node tests ----
  const core = {
    T_END: 100,
    TAU: 20,
    DT_MIN: 0.05,
    DT_MAX: 45,

    /** Deterministic PRNG (mulberry32). */
    mulberry32(seed) {
      let a = seed >>> 0;
      return function () {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    },

    /** Exact solution y(t) = exp(-t/tau) of dy/dt = -y/tau. */
    expDecay(t, tau) {
      return Math.exp(-t / tau);
    },

    /** One forward Euler step y_{k+1} = (1 - dt/tau) y_k. */
    eulerStep(y, dt, tau) {
      return y * (1 - dt / tau);
    },

    /** One exact exponential-integrator step y_{k+1} = exp(-dt/tau) y_k. */
    exactStep(y, dt, tau) {
      return y * Math.exp(-dt / tau);
    },

    /** Number of uniform steps whose corner grid covers [0, tEnd]. */
    stepsTo(dt, tau, tEnd) {
      return Math.max(1, Math.round(tEnd / dt));
    },

    /** Landing value on the corner grid near tEnd: (1 - dt/tau)^k or e^(-k dt/tau). */
    landingValue(mode, dt, tau, tEnd) {
      const k = core.stepsTo(dt, tau, tEnd);
      if (mode === 'exact') return Math.exp((-k * dt) / tau);
      return Math.pow(1 - dt / tau, k);
    },

    /** |corner value - exact value| at the corner covering tEnd. */
    errorAt(dt, tau, tEnd) {
      const k = core.stepsTo(dt, tau, tEnd);
      const y = Math.pow(1 - dt / tau, k);
      return Math.abs(y - Math.exp((-k * dt) / tau));
    },

    /** Log-spaced sweep of (dt, error) pairs. */
    errorSweep(dtMin, dtMax, tau, tEnd, nPts) {
      const dts = [];
      const errs = [];
      const lo = Math.log(dtMin);
      const hi = Math.log(dtMax);
      for (let i = 0; i < nPts; i++) {
        const dt = Math.exp(lo + ((hi - lo) * i) / (nPts - 1));
        dts.push(dt);
        errs.push(core.errorAt(dt, tau, tEnd));
      }
      return { dts, errs };
    },

    /** Least-squares slope of ln(err) against ln(dt). */
    fitSlopeLogLog(xs, ys) {
      if (xs.length !== ys.length || xs.length < 2) {
        throw new Error(`fit needs >= 2 paired points, got ${xs.length} and ${ys.length}`);
      }
      let sx = 0;
      let sy = 0;
      let sxx = 0;
      let sxy = 0;
      const n = xs.length;
      for (let i = 0; i < n; i++) {
        const lx = Math.log(xs[i]);
        const ly = Math.log(ys[i]);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
      }
      const den = n * sxx - sx * sx;
      if (den === 0) {
        throw new Error('degenerate fit: all points share one dt');
      }
      return (n * sxy - sx * sy) / den;
    },

    /** Euler is stable only below the boundary dt = 2 tau. */
    isStable(dt, tau) {
      return dt > 0 && dt < 2 * tau;
    },
  };

  // ---- ui: only runs when a browser Atlas shell exists ----
  function mount(rootEl, ctx) {
    const doc = rootEl.ownerDocument;
    if (!doc || typeof document === 'undefined') return;
    const TA = core.TAU;
    const T0 = core.T_END;
    // fixed viewing window for the hero plot: blow-up is clipped, never zoomed
    const YCLIP = 1.5;
    const YMIN = -1.62;
    const YMAX = 1.62;

    if (!doc.getElementById('m01-css')) {
      const styleEl = doc.createElement('style');
      styleEl.id = 'm01-css';
      styleEl.textContent = `
        .m01-wrap { display: grid; grid-template-columns: 238px minmax(0,1fr); gap: 10px; width: 100%; height: 100%; min-height: 0; }
        .m01-margin { display: flex; flex-direction: column; gap: 12px; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink-muted, #57534e);
           border-inline-end: 1px solid var(--border, #d6cfbf); padding-inline-end: 8px; overflow-y: auto; min-height: 0; }
        .m01-margin.m01-margin--host { border-inline-end: none; padding-inline-end: 0; }
        .m01-margin::-webkit-scrollbar { width: 6px; }
        .m01-margin::-webkit-scrollbar-thumb { background: var(--border, #d6cfbf); border-radius: 3px; }
        .m01-stage { display: flex; flex-direction: column; gap: 6px; width: 100%; height: 100%; min-height: 0; min-width: 0; }
        .m01-stage.m01-stage--phone { height: auto; }
        .m01-stage.m01-stage--phone .m01-canvasbox { flex: none; height: 470px; min-height: 470px; }
        .m01-chap { display: flex; align-items: center; gap: 8px; flex: none; }
        .m01-btn { font: 600 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink, #1b1a17);
          background: var(--paper-card, #eeeae0); border: 1px solid var(--border, #d6cfbf); border-radius: 7px;
          padding: 4px 10px; cursor: pointer; white-space: nowrap; }
        .m01-btn:hover { border-color: var(--ink-muted, #57534e); }
        .m01-btn:disabled { opacity: .55; cursor: default; }
        .m01-btn.m01-on { background: var(--mod, #7e22ce); color: var(--paper, #f6f2e9); border-color: transparent; }
        .m01-btn:focus-visible { outline: 2px solid var(--mod, #7e22ce); outline-offset: 1px; }
        .m01-chapname { font-family: ui-serif, Georgia, serif; font-size: 14px; color: var(--ink, #1b1a17); min-width: 0; }
        .m01-chapno { font: 12px ui-monospace, monospace; color: var(--ink-muted, #57534e); white-space: nowrap; }
        .m01-cap { font-family: ui-serif, Georgia, serif; font-size: 13px; line-height: 1.35;
          color: var(--ink-muted, #57534e); flex: none; }
        .m01-canvasbox { flex: 1 1 auto; min-height: 120px; position: relative; }
        .m01-canvasbox canvas { display: block; border: 1px solid var(--border, #d6cfbf); border-radius: 6px;
          background: var(--paper-card, #eeeae0); touch-action: none; }
        .m01-ctrls { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; flex: none; }
        .m01-readout { font: 12px ui-monospace, monospace; color: var(--ink, #1b1a17); font-variant-numeric: tabular-nums;
          background: var(--paper-inset, #e7e2d6); border: 1px solid var(--border, #d6cfbf); border-radius: 6px;
          padding: 3px 8px; white-space: nowrap; }
        .m01-readout b { color: var(--volt, #b45309); font-weight: 600; }
        .m01-predict { background: var(--paper-card, #eeeae0); border: 1px dashed var(--mod, #7e22ce); border-radius: 8px;
          padding: 8px 10px; font-family: ui-serif, Georgia, serif; font-size: 13px; color: var(--ink, #1b1a17); flex: none; }
        .m01-predict .m01-q { margin-bottom: 6px; }
        .m01-predict .m01-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
        .m01-predict .m01-res { margin: 6px 0; font-family: ui-monospace, monospace; font-size: 12px; }
        .m01-hint { font: 12px ui-monospace, monospace; color: var(--ink-muted, #57534e); }
        @media (max-width: 720px) {
          .m01-wrap { display: flex; flex-direction: column; height: 100%; min-height: 0; overflow-y: auto; }
          .m01-stage { order: 1; flex: none; }
          .m01-wrap .m01-canvasbox { aspect-ratio: 620 / 380; min-height: 150px; height: auto; }
          .m01-margin { order: 2; flex: none; border-inline-end: none; border-top: 1px solid var(--border, #d6cfbf);
            padding: 12px 0 0; }
          .m01-btn { font-size: 11.5px; padding: 3px 7px; }
          .m01-chapname { font-size: 12.5px; }
          .m01-cap { font-size: 12px; }
        }
      `;
      doc.head.appendChild(styleEl);
    }

    // Marginalia live in the shell's own margin column when the shell provides one;
    // otherwise the plate falls back to an internal margin column.
    const marginHost = (ctx && ctx.margin) || (typeof doc.getElementById === 'function' ? doc.getElementById('atlas-margin') : null) || null;
    const hostMode = Boolean(marginHost);
    const marginEl = doc.createElement('div');
    marginEl.className = 'm01-margin' + (hostMode ? ' m01-margin--host' : '');
    const stage = doc.createElement('div');
    stage.className = 'm01-stage';
    if (hostMode) {
      marginHost.appendChild(marginEl);
      rootEl.appendChild(stage);
    } else {
      const wrap = doc.createElement('div');
      wrap.className = 'm01-wrap';
      wrap.appendChild(marginEl);
      wrap.appendChild(stage);
      rootEl.appendChild(wrap);
    }

    // On narrow screens the shell's fixed-aspect figure box is too shallow for the
    // stacked bands; there the plate lets the box grow with the content instead.
    const win = doc.defaultView;
    const mqNarrow = win && typeof win.matchMedia === 'function' ? win.matchMedia('(max-width: 720px)') : null;
    function applySizeMode() {
      if (!hostMode) return false;
      const phone = Boolean(mqNarrow && mqNarrow.matches);
      stage.classList.toggle('m01-stage--phone', phone);
      if (phone) {
        rootEl.style.aspectRatio = 'auto';
        rootEl.style.maxHeight = 'none';
        rootEl.style.height = 'auto';
      } else {
        rootEl.style.aspectRatio = '';
        rootEl.style.maxHeight = '';
        rootEl.style.height = '';
      }
      return phone;
    }
    applySizeMode();

    const cv = ctx.canvas(1000, 620);
    cv.style.position = 'absolute';
    cv.style.left = '0';
    cv.style.top = '0';
    const canvasBox = doc.createElement('div');
    canvasBox.className = 'm01-canvasbox';
    stage.appendChild(canvasBox);
    canvasBox.appendChild(cv);
    cv.setAttribute('role', 'img');
    cv.setAttribute('aria-label', S.ariaCanvas);

    // margin: equations with hoverable terms, live values, glossary, honest footer
    marginEl.innerHTML = `
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.eqTitle.toUpperCase()}</div>
        <div class="atlas-equation-box atlas-mono">
          <div style="margin-bottom:4px;">dy/dt = <span class="m01-term" data-term="tau" style="cursor:pointer; color:var(--volt);">&minus;y/&tau;</span></div>
          <div style="margin-bottom:4px;"><span style="color:var(--ink-muted); font-family:var(--font-serif);">${S.eqEuler}</span> y<sub>k+1</sub> =
            (<span class="m01-term" data-term="dt" style="cursor:pointer; color:var(--exc);">dt</span>/&tau;) &middot;
            <span class="m01-term" data-term="factor" style="cursor:pointer; color:var(--volt);">(1 &minus; dt/&tau;)</span> &middot; y<sub>k</sub> =
            <span class="atlas-margin-value" id="m01-vf" style="font-size:12px;">?</span> &middot; y<sub>k</sub></div>
          <div style="margin-bottom:4px;"><span style="color:var(--ink-muted); font-family:var(--font-serif);">${S.eqExact}</span> y<sub>k+1</sub> =
            <span class="m01-term" data-term="fix" style="cursor:pointer; color:var(--mod);">e<sup>&minus;dt/&tau;</sup></span> &middot; y<sub>k</sub> =
            <span class="atlas-margin-value" id="m01-vfe" style="font-size:12px;">?</span> &middot; y<sub>k</sub></div>
          <div><span class="m01-term" data-term="staircase" style="cursor:pointer; color:var(--volt);">staircase</span>:
            y<sub>k</sub> = (1 &minus; dt/&tau;)<sup class="m01-term" data-term="k" style="cursor:pointer; color:var(--exc);">k</sup> =
            <span class="atlas-margin-value" id="m01-vpow" style="font-size:12px;">?</span></div>
          <div id="m01-stair-exp" style="display:none; margin-top:4px;"><span style="color:var(--ink-muted); font-family:var(--font-serif);">e-fold</span>:
            y<sub>k</sub> = e<sup>&minus;k&middot;dt/&tau;</sup> =
            <span class="atlas-margin-value" id="m01-vexq" style="font-size:12px;">?</span></div>
        </div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.liveTitle.toUpperCase()}</div>
        <div style="display:flex; flex-direction:column; gap:4px; font-family:var(--font-mono); font-size:11px;">
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="color:var(--ink-muted);">${S.valDt}:</span>
            <span><b class="atlas-margin-value" id="m01-vdt" style="font-size:12px;">?</b> ms</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="color:var(--ink-muted);">${S.valTau}:</span>
            <span class="atlas-margin-value" style="font-size:12px;">${TA} ms</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="color:var(--ink-muted);">${S.valK}:</span>
            <span class="atlas-margin-value" id="m01-vk" style="font-size:12px;">?</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="color:var(--ink-muted);">${S.valY}:</span>
            <span class="atlas-margin-value" id="m01-vy" style="font-size:12px;">?</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="color:var(--ink-muted);">${S.valDiff}:</span>
            <span class="atlas-margin-value" id="m01-vdiff" style="font-size:12px;">?</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="color:var(--ink-muted);">${S.valStable}:</span>
            <span class="atlas-margin-value" id="m01-vstab" style="font-size:12px; font-weight:600;">?</span>
          </div>
        </div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.glossTitle.toUpperCase()}</div>
        <div style="display:flex; flex-direction:column; gap:6px; font-size:11.5px; line-height:1.4;">
          <div>${S.glosTau}</div>
          <div>${S.glosEuler}</div>
          <div>${S.glosGhost}</div>
          <div>${S.glosOrder}</div>
          <div>${S.glosStability}</div>
          <div>${S.glosExact}</div>
        </div>
      </div>
      <div class="atlas-margin-section">
        <div class="m01-hint">${S.footer}</div>
      </div>
    `;

    const vEls = {};
    for (const id of ['m01-vf', 'm01-vfe', 'm01-vpow', 'm01-vdt', 'm01-vk', 'm01-vy', 'm01-vdiff', 'm01-vstab']) {
      vEls[id] = marginEl.querySelector('#' + id);
    }
    vEls['m01-stair-exp'] = marginEl.querySelector('#m01-stair-exp');
    vEls['m01-vexq'] = marginEl.querySelector('#m01-vexq');

    // chapter bar
    const btnBack = doc.createElement('button');
    btnBack.className = 'm01-btn';
    btnBack.innerHTML = '&larr; ' + S.btnBack;
    btnBack.setAttribute('aria-label', 'Previous chapter');
    const chapNoEl = doc.createElement('span');
    chapNoEl.className = 'm01-chapno';
    const chapNameEl = doc.createElement('span');
    chapNameEl.className = 'm01-chapname';
    const btnNext = doc.createElement('button');
    btnNext.className = 'm01-btn';
    btnNext.innerHTML = S.btnNext + ' &rarr;';
    btnNext.setAttribute('aria-label', 'Next chapter');
    const chapRow = doc.createElement('div');
    chapRow.className = 'm01-chap';
    chapRow.appendChild(btnBack);
    chapRow.appendChild(chapNoEl);
    chapRow.appendChild(chapNameEl);
    chapRow.appendChild(btnNext);
    const capEl = doc.createElement('div');
    capEl.className = 'm01-cap';
    stage.appendChild(chapRow);
    stage.appendChild(capEl);

    // predict panel
    const predictEl = doc.createElement('div');
    predictEl.className = 'm01-predict';
    predictEl.style.display = 'none';
    stage.appendChild(predictEl);

    // control row
    const btnPlay = doc.createElement('button');
    btnPlay.className = 'm01-btn';
    btnPlay.setAttribute('aria-label', 'Play or pause: Space');
    const btnHalve = doc.createElement('button');
    btnHalve.className = 'm01-btn';
    btnHalve.textContent = S.btnHalve;
    const btnTheory = doc.createElement('button');
    btnTheory.className = 'm01-btn';
    btnTheory.setAttribute('aria-pressed', 'false');
    const btnPredict = doc.createElement('button');
    btnPredict.className = 'm01-btn';
    btnPredict.textContent = S.btnPredict;
    btnPredict.setAttribute('aria-pressed', 'false');
    const btnEuler = doc.createElement('button');
    btnEuler.className = 'm01-btn';
    btnEuler.textContent = S.btnEuler;
    btnEuler.setAttribute('aria-pressed', 'true');
    const btnExact = doc.createElement('button');
    btnExact.className = 'm01-btn';
    btnExact.textContent = S.btnExact;
    btnExact.setAttribute('aria-pressed', 'false');
    const readoutEl = doc.createElement('span');
    readoutEl.className = 'm01-readout';
    const ctrlRow = doc.createElement('div');
    ctrlRow.className = 'm01-ctrls';
    ctrlRow.appendChild(btnPlay);
    ctrlRow.appendChild(btnHalve);
    ctrlRow.appendChild(btnTheory);
    ctrlRow.appendChild(btnPredict);
    ctrlRow.appendChild(btnEuler);
    ctrlRow.appendChild(btnExact);
    ctrlRow.appendChild(readoutEl);
    stage.appendChild(ctrlRow);

    // ---- state ----
    const st = {
      dt: 5,
      mode: 'euler',
      theory: true,
      chapter: 0,
      pin: null,
      hover: null,
      tp: 0,
      lastNow: -1,
      predict: null,
      halveFlash: -1e9,
      dtFlash: -1e9,
      ys: [1],
      ts: [0],
      n: 1,
    };

    function fmtSig(v) {
      if (!Number.isFinite(v)) return String(v);
      if (v === 0) return '0';
      const a = Math.abs(v);
      if (a >= 1000 || a < 0.001) return v.toExponential(1);
      return String(Number(v.toPrecision(3)));
    }

    function fmtVal(v) {
      if (Number.isFinite(v) && Math.abs(v) < 1000) return v.toFixed(3);
      return fmtSig(v);
    }

    // ---- predict choices (deterministic shuffle from dt) ----
    function makeChoices(dt) {
      const a = core.landingValue('euler', dt, TA, T0);
      const b = core.landingValue('exact', dt, TA, T0);
      let c = core.eulerStep(a, dt, TA);
      if (Math.abs(c - a) < 1e-6 && Math.abs(c - b) < 1e-6) c = a + 0.25;
      else if (Math.abs(c - a) < 1e-6) c = a + Math.max(0.05, 0.1 * Math.abs(a) + 0.02);
      if (Math.abs(c - b) < 1e-6) c = b + 0.15;
      const arr = [
        { v: a, ok: true },
        { v: b, ok: false },
        { v: c, ok: false },
      ];
      if (Math.abs(arr[0].v - arr[2].v) < 1e-9) arr[2].v = arr[1].v + 0.2;
      const rnd = core.mulberry32(Math.round(dt * 7919) >>> 4);
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
      }
      return arr;
    }

    // ---- chapters ----
    const CH = S.chapters;
    function refreshButtons() {
      btnPlay.textContent = ctx.clock && ctx.clock.isPlaying ? '\u275A\u275A ' + S.btnPause : '\u25B6 ' + S.btnPlay;
      btnEuler.classList.toggle('m01-on', st.mode === 'euler');
      btnEuler.setAttribute('aria-pressed', String(st.mode === 'euler'));
      btnExact.classList.toggle('m01-on', st.mode === 'exact');
      btnExact.setAttribute('aria-pressed', String(st.mode === 'exact'));
      btnTheory.classList.toggle('m01-on', st.theory);
      btnTheory.setAttribute('aria-pressed', String(st.theory));
      btnTheory.textContent = st.theory ? S.btnTheory + ' \u2713' : S.btnTheory;
    }

    function simulate() {
      // the staircase for the current settings; cached until inputs change
      const n = core.stepsTo(st.dt, TA, T0);
      if (st.n === n && st.mode === st.ysMode && st.dt === st.dtAt) return;
      const f = st.mode === 'exact' ? core.exactStep : core.eulerStep;
      const ys = [1];
      const ts = [0];
      let y = 1;
      for (let i = 1; i <= n; i++) {
        y = f(y, st.dt, TA);
        ys.push(y);
        ts.push(i * st.dt);
      }
      st.ys = ys;
      st.ts = ts;
      st.n = n;
      st.ysMode = st.mode;
      st.dtAt = st.dt;
    }

    function setChapter(i) {
      st.chapter = ((i % CH.length) + CH.length) % CH.length;
      const ch = CH[st.chapter];
      st.dt = ch.dt;
      st.mode = ch.mode;
      st.pin = ch.pin || null;
      st.predict = null;
      predictEl.style.display = 'none';
      btnPredict.classList.remove('m01-on');
      btnPredict.setAttribute('aria-pressed', 'false');
      chapNoEl.textContent = S.chapterOf + ' ' + (st.chapter + 1) + '/' + CH.length;
      chapNameEl.textContent = ch.name;
      capEl.textContent = ch.cap;
      st.tp = 0;
      st.dtFlash = performance.now();
      refreshButtons();
      renderPredict();
      render();
    }

    btnBack.addEventListener('click', () => setChapter(st.chapter - 1));
    btnNext.addEventListener('click', () => setChapter(st.chapter + 1));
    stage.tabIndex = 0;
    stage.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        e.stopPropagation();
        setChapter(st.chapter - 1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        e.stopPropagation();
        setChapter(st.chapter + 1);
      }
    });

    btnPlay.addEventListener('click', () => {
      if (ctx.clock) ctx.clock.toggle();
      refreshButtons();
      render();
    });
    btnTheory.addEventListener('click', toggleTheory);
    function toggleTheory() {
      st.theory = !st.theory;
      refreshButtons();
      render();
    }

    function setDt(v, fromHalf) {
      st.dt = Math.max(core.DT_MIN, Math.min(core.DT_MAX, v));
      if (fromHalf) {
        st.halveFlash = performance.now();
        st.halvePrevErr = core.errorAt(st.dt * 2, TA, T0);
      }
      if (st.predict && st.predict.picked === -1) {
        st.predict.choices = makeChoices(st.dt);
        renderPredict();
      }
      st.dtFlash = performance.now();
      simulate();
      refreshButtons();
      render();
    }
    btnHalve.addEventListener('click', () => setDt(st.dt / 2, true));

    function setMode(m) {
      st.mode = m;
      if (st.predict && st.predict.picked === -1) {
        st.predict.choices = makeChoices(st.dt);
        renderPredict();
      }
      simulate();
      refreshButtons();
      render();
    }
    btnEuler.addEventListener('click', () => setMode('euler'));
    btnExact.addEventListener('click', () => setMode('exact'));

    btnPredict.addEventListener('click', () => {
      if (st.predict) {
        st.predict = null;
      } else {
        st.predict = { choices: makeChoices(st.dt), picked: -1 };
      }
      predictEl.style.display = st.predict ? '' : 'none';
      btnPredict.classList.toggle('m01-on', Boolean(st.predict));
      btnPredict.setAttribute('aria-pressed', String(Boolean(st.predict)));
      renderPredict();
      render();
    });

    function renderPredict() {
      const p = st.predict;
      if (!p) {
        predictEl.style.display = 'none';
        return;
      }
      predictEl.style.display = '';
      const committed = p.picked !== -1;
      const btns = p.choices
        .map((c, i) => `<button class="m01-btn m01-pick" data-i="${i}" ${committed ? 'disabled' : ''}>${fmtSig(c.v)}</button>`)
        .join('');
      let body = `<div class="m01-q">${S.predictQuestion.replace('{dt}', fmtSig(st.dt))}<br>` +
        `<span style="font-size:11.5px;color:var(--ink-muted,#57534e)">${S.predictHint}</span></div>` +
        `<div class="m01-row">${committed ? '' : btns}</div>`;
      if (committed) {
        const pick = p.choices[p.picked];
        const eu = core.landingValue('euler', st.dt, TA, T0);
        const ex = core.landingValue('exact', st.dt, TA, T0);
        const mark = pick.ok ? '\u2713' : '\u2717';
        const txt = pick.ok
          ? S.predictCorrect
          : S.predictOff
              .replace('{euler}', fmtSig(eu))
              .replace('{exact}', fmtSig(ex))
              .replace('{diff}', fmtSig(Math.abs(eu - ex)));
        body += `<div class="m01-res">${mark} ${S.lblYourPick}: ${fmtSig(pick.v)} &mdash; ${txt}</div>` +
          `<div class="m01-row"><button class="m01-btn" data-again="1">${S.predictAgain}</button></div>`;
      }
      predictEl.innerHTML = body;
      predictEl.querySelectorAll('.m01-pick').forEach((b) => {
        b.addEventListener('click', () => {
          if (!st.predict) return;
          st.predict.picked = parseInt(b.dataset.i, 10);
          renderPredict();
          render();
        });
      });
      const again = predictEl.querySelector('[data-again]');
      if (again) {
        again.addEventListener('click', () => {
          st.predict.picked = -1;
          st.predict.choices = makeChoices(st.dt);
          renderPredict();
          render();
        });
      }
    }

    // hover a marginalia term to highlight where it acts in the figure
    marginEl.querySelectorAll('.m01-term').forEach((el) => {
      el.addEventListener('mouseenter', () => {
        st.hover = el.dataset.term;
        el.classList.add('m01-hot');
        render();
      });
      el.addEventListener('mouseleave', () => {
        st.hover = null;
        el.classList.remove('m01-hot');
        render();
      });
    });

    // the shell's T key / Theory button broadcasts here
    if (typeof window !== 'undefined') {
      window.addEventListener('atlas:theory', (e) => {
        const active = e.detail && typeof e.detail.active === 'boolean' ? e.detail.active : !st.theory;
        if (active !== st.theory) {
          st.theory = active;
          refreshButtons();
          render();
        }
      });
    }

    // ---- logical layout: five stacked bands, each with its own margins ----
    //   [ scrub-hint heard ][ hero plot ][ x-tick strip ][ dt ruler band ][ log-log error band ]
    let L = { cssW: 800, cssH: 500, scale: 1, orientWide: true, geom: null };
    let dtFromX = null; // inverse maps installed each render

    function installMaps() {
      dtFromX = (x, r) => {
        const f = Math.max(0, Math.min(1, (x - r.trackX) / r.trackW));
        return Math.exp(Math.log(core.DT_MIN) + f * (Math.log(core.DT_MAX) - Math.log(core.DT_MIN)));
      };
    }

    function dtFrac(r, dt) {
      return Math.max(0, Math.min(1, (Math.log(dt) - Math.log(core.DT_MIN)) / (Math.log(core.DT_MAX) - Math.log(core.DT_MIN))));
    }
    function dtToX(r, dt) {
      return r.trackX + dtFrac(r, dt) * r.trackW;
    }

    function layout(lw, lh, wide) {
      const gutter = wide ? 58 : 46;          // left gutter: y tick labels + axis titles
      const padR = Math.max(gutter, 16);      // symmetric right margin
      const hintH = 26;                       // scrub-hint heard above the plot
      const tickH = 30;                       // x tick strip under the hero
      const rulerH = 78;                      // dt ruler band
      const errH = Math.round(Math.max(122, Math.min(170, lh * 0.22)));
      const errCap = 16;                      // caption strip under the error pane
      const gap = 8;
      const heroY = hintH + 2;
      const heroH = Math.max(170, lh - heroY - tickH - rulerH - gap - errH - errCap - gap - 4);
      const rulerY = heroY + heroH + tickH;
      const errY = rulerY + rulerH + gap;
      const w = lw - gutter - padR;
      return {
        hero: { x: gutter, y: heroY, w, h: heroH },
        xticks: { x: gutter, y: heroY + heroH, w, h: tickH },
        ruler: { x: gutter, y: rulerY, w, h: rulerH },
        err: { x: gutter, y: errY, w, h: Math.min(errH, Math.max(84, lh - errY - errCap - 4)) },
        errCapY: errY + errH + errCap,
      };
    }

    // ---- pointer interaction: dt ruler drag + playhead scrub ----
    let drag = null;
    function evPos(e) {
      const rect = cv.getBoundingClientRect();
      return {
        x: rect.width > 0 ? ((e.clientX - rect.left) / rect.width) * L.geom.lw : 0,
        y: rect.height > 0 ? ((e.clientY - rect.top) / rect.height) * L.geom.lh : 0,
      };
    }

    function hitTest(p) {
      const r = L.geom;
      if (!r) return null;
      if (p.x >= r.ruler.x - 8 && p.x <= r.ruler.x + r.ruler.w + 8 && p.y >= r.ruler.y - 12 && p.y <= r.ruler.y + r.ruler.h + 10) {
        return { kind: 'ruler', rect: r.ruler };
      }
      if (p.x >= r.hero.x - 4 && p.x <= r.hero.x + r.hero.w + 4 && p.y >= 0 && p.y <= r.hero.y + r.hero.h + 14) {
        return { kind: 'time', rect: r.hero };
      }
      return null;
    }

    cv.addEventListener('pointerdown', (e) => {
      const p = evPos(e);
      drag = hitTest(p);
      if (drag) {
        cv.setPointerCapture(e.pointerId);
        if (drag.kind === 'ruler') {
          setDt(installAndRuler(p));
        } else {
          st.tp = scrubTime(p, drag.rect);
        }
        e.preventDefault();
      }
    });
    cv.addEventListener('pointermove', (e) => {
      const p = evPos(e);
      if (drag && drag.kind === 'ruler') {
        setDt(installAndRuler(p));
      } else if (drag && drag.kind === 'time') {
        st.tp = scrubTime(p, drag.rect);
        st.lastNow = performance.now();
        render();
      } else {
        const hit = hitTest(p);
        cv.style.cursor = hit ? (hit.kind === 'ruler' ? 'ew-resize' : 'crosshair') : 'default';
      }
    });
    cv.addEventListener('pointerup', () => {
      drag = null;
    });
    cv.addEventListener('pointercancel', () => {
      drag = null;
    });

    function installAndRuler(p) {
      installMaps();
      return dtFromX(p.x, drag.rect);
    }
    function scrubTime(p, hr) {
      const tMax = Math.max(T0, st.ts[st.n]);
      const f = Math.max(0, Math.min(1, (p.x - hr.x) / hr.w));
      return f * tMax;
    }

    // ---- ball along the polygon path ----
    function ballState(tp) {
      const n = st.n;
      const dt = st.dt;
      const tMaxPath = n * dt;
      const tc = Math.max(0, Math.min(tp, tMaxPath));
      let k = Math.floor(tc / dt + 1e-9);
      if (k >= n) k = n - 1;
      if (k < 0) k = 0;
      const frac = (tc - k * dt) / dt;
      const y = st.ys[k] + frac * (st.ys[k + 1] - st.ys[k]);
      return { k, t: tc, y, frac };
    }

    // ---- live values in margin ----
    function updateReadouts() {
      const b = ballState(st.tp);
      const yExact = core.expDecay(b.t, TA);
      const diff = Math.abs(b.y - yExact);
      const stable = core.isStable(st.dt, TA);
      vEls['m01-vdt'].textContent = fmtSig(st.dt);
      vEls['m01-vf'].textContent = fmtSig(core.eulerStep(1, st.dt, TA));
      vEls['m01-vfe'].textContent = fmtSig(core.exactStep(1, st.dt, TA));
      vEls['m01-vk'].textContent = String(b.k);
      vEls['m01-vpow'].textContent = fmtSig(core.landingValue('euler', st.dt, TA, T0));
      vEls['m01-stair-exp'].style.display = st.mode === 'exact' ? '' : 'none';
      vEls['m01-vexq'].textContent = fmtSig(core.landingValue('exact', st.dt, TA, T0));
      vEls['m01-vy'].textContent = fmtVal(b.y);
      vEls['m01-vdiff'].textContent = fmtSig(diff);
      vEls['m01-vstab'].textContent = stable ? S.valYes : S.valNo;
      vEls['m01-vstab'].style.color = stable ? 'var(--exc)' : 'var(--inh)';
      const err = core.errorAt(st.dt, TA, T0);
      readoutEl.innerHTML = `${S.errAt100}: <b>${fmtSig(err)}</b>&nbsp;&nbsp;dt = ${fmtSig(st.dt)} ms&nbsp;&nbsp;` +
        `${S.valSlope}: ${fitCache.toFixed(2)}`;
    }

    let fitCache = 1;
    let fitPointAt1 = core.errorAt(1, TA, T0); // anchor C for the fitted line err = C dt^slope
    function fitSlopeCache() {
      fitCache = core.fitSlopeLogLog(core.errorSweep(0.05, 2, TA, T0, 41).dts, core.errorSweep(0.05, 2, TA, T0, 41).errs);
      return fitCache;
    }
    fitCache = fitSlopeCache();

    // ---- shared drawing helpers ----
    function rounded(g2, x, y, w, h, rad) {
      g2.beginPath();
      if (g2.roundRect) g2.roundRect(x, y, w, h, rad);
      else g2.rect(x, y, w, h);
    }

    function chipBox(g2, text, fontM) {
      g2.font = fontM(12);
      return { w: g2.measureText(text).width + 14, h: 21 };
    }

    function drawChip(g2, xRight, yTop, text, color, fontM, c) {
      const box = chipBox(g2, text, fontM);
      const x = xRight - box.w;
      g2.fillStyle = c.paperCard;
      g2.strokeStyle = color;
      g2.lineWidth = 1;
      rounded(g2, x, yTop, box.w, box.h, 6);
      g2.globalAlpha = 0.94;
      g2.fill();
      g2.globalAlpha = 1;
      g2.stroke();
      g2.fillStyle = color;
      g2.textAlign = 'center';
      g2.textBaseline = 'middle';
      g2.fillText(text, x + box.w / 2, yTop + box.h / 2 + 0.5);
      return box;
    }

    function note(g2, x, y, text, color, fontM, clampRect, c) {
      g2.font = fontM(12);
      const w = g2.measureText(text).width + 12;
      let half = w / 2;
      if (clampRect) {
        x = Math.max(clampRect.x + 4 + half, Math.min(clampRect.x + clampRect.w - 4 - half, x));
        y = Math.max(clampRect.y + 13, Math.min(clampRect.y + clampRect.h - 13, y));
      }
      g2.fillStyle = c.paperCard;
      g2.strokeStyle = c.border;
      g2.lineWidth = 1;
      rounded(g2, x - half, y - 10, w, 20, 5);
      g2.globalAlpha = 0.94;
      g2.fill();
      g2.globalAlpha = 1;
      g2.stroke();
      g2.fillStyle = color;
      g2.textAlign = 'center';
      g2.textBaseline = 'middle';
      g2.fillText(text, x, y);
    }

    // ---- rendering ----
    function render() {
      simulate();
      const cssW = Math.max(300, Math.round(canvasBox.clientWidth || 800) - 2);
      const cssH = Math.max(260, Math.round(canvasBox.clientHeight || 500) - 2);
      const dpr = Math.min(2.5, (typeof window !== 'undefined' && window.devicePixelRatio) || 1);
      const wide = cssW >= 620;
      const scale = 1; // logical units = CSS px, so every font is drawn at its real size
      cv.style.width = cssW + 'px';
      cv.style.height = cssH + 'px';
      const bw = Math.round(cssW * dpr);
      const bh = Math.round(cssH * dpr);
      if (cv.width !== bw || cv.height !== bh) {
        cv.width = bw;
        cv.height = bh;
      }
      const g2 = cv.getContext('2d');
      g2.setTransform(1, 0, 0, 1, 0, 0);
      g2.clearRect(0, 0, cv.width, cv.height);
      g2.setTransform(dpr, 0, 0, dpr, 0, 0);

      const geom = layout(cssW, cssH, wide);
      geom.lw = cssW;
      geom.lh = cssH;
      L = { cssW, cssH, scale, orientWide: wide, geom };

      const c = ctx.colors();
      const px = (n) => `${n}px`;
      const fontM = (n) => `${px(n)} ui-monospace, SFMono-Regular, Menlo, monospace`;
      const fontS = (n) => `${px(n)} ui-serif, Georgia, serif`;
      void fontS;

      // scrub-hint heard: its own strip above the plot
      g2.fillStyle = c.inkMuted;
      g2.font = fontM(12);
      g2.textAlign = 'right';
      g2.textBaseline = 'bottom';
      g2.globalAlpha = 0.8;
      g2.fillText(cssW < 560 ? S.scrubHintShort : S.scrubHint, geom.lw - 6, geom.hero.y - 4);
      g2.globalAlpha = 1;

      const hr = geom.hero;
      const tMaxView = Math.max(T0, st.ts[st.n]);
      const X = (t) => hr.x + (t / tMaxView) * hr.w;
      const Y = (v) => hr.y + ((YMAX - Math.min(YMAX, Math.max(YMIN, v))) / (YMAX - YMIN)) * hr.h;

      drawHero(g2, { hr, geom, c, fontM, X, Y, tMaxView });
      drawXTicks(g2, geom.xticks, X, tMaxView, c, fontM);

      drawRuler(g2, geom.ruler, c, fontM);
      drawError(g2, geom.err, c, fontM);
      updateReadouts();
    }

    function gridTicks(ymin, ymax, target) {
      const span = ymax - ymin;
      const raw = span / target;
      const pow = Math.pow(10, Math.floor(Math.log10(raw || 1)));
      let nice = pow;
      if (raw / pow > 2.2) nice = 2.5 * pow;
      if (raw / pow > 5) nice = 5 * pow;
      const out = [];
      for (let v = Math.ceil(ymin / nice) * nice; v <= ymax + 1e-9; v += nice) {
        out.push(Number(v.toFixed(6)));
      }
      return { step: nice, ticks: out };
    }

    // the Euler staircase is the hero: corner-to-corner steps, an animated tangent arm,
    // the drop into the next point, sign flips past tau, clipped blow-up past 2tau
    function drawHero(g2, o) {
      const { hr, geom, c, fontM, X, Y, tMaxView } = o;
      const unstableFactor = st.mode === 'euler' && st.dt > TA;
      const blown = st.mode === 'euler' && !core.isStable(st.dt, TA);

      // frame
      g2.strokeStyle = c.ink;
      g2.lineWidth = 1;
      g2.strokeRect(hr.x, hr.y, hr.w, hr.h);

      // y grid + ticks (labels live in the left gutter, the rotated axis title at the far left)
      const yTicks = gridTicks(YMIN, YMAX, 6);
      g2.font = fontM(12);
      g2.fillStyle = c.inkMuted;
      g2.textAlign = 'right';
      g2.textBaseline = 'middle';
      for (const v of yTicks.ticks) {
        g2.globalAlpha = 0.45;
        g2.strokeStyle = c.border;
        g2.lineWidth = 0.8;
        g2.beginPath();
        g2.moveTo(hr.x, Y(v));
        g2.lineTo(hr.x + hr.w, Y(v));
        g2.stroke();
        g2.globalAlpha = 1;
        g2.fillStyle = c.inkMuted;
        g2.fillText(v.toFixed(Math.abs(v) < 10 && v !== Math.round(v) ? 1 : 0), hr.x - 7, Y(v));
      }
      // zero axis: heavier so sign flips past tau read as crossings
      g2.save();
      g2.strokeStyle = c.inkMuted;
      g2.globalAlpha = unstableFactor ? 0.8 : 0.4;
      g2.lineWidth = unstableFactor ? 1.4 : 1;
      g2.setLineDash([6, 4]);
      g2.beginPath();
      g2.moveTo(hr.x, Y(0));
      g2.lineTo(hr.x + hr.w, Y(0));
      g2.stroke();
      g2.restore();
      g2.save();
      g2.translate(hr.x - 40, hr.y + hr.h / 2);
      g2.rotate(-Math.PI / 2);
      g2.textAlign = 'center';
      g2.textBaseline = 'middle';
      g2.fillStyle = c.inkMuted;
      g2.font = fontM(12);
      g2.fillText(S.axisY, 0, 0);
      g2.restore();

      // clip: nothing drawn inside the plot may escape the frame (blow-up is cut, not zoomed)
      g2.save();
      g2.beginPath();
      g2.rect(hr.x, hr.y, hr.w, hr.h);
      g2.clip();

      // ghost curve (theory); hover 'fix' forces it visible
      if (st.theory || st.hover === 'fix') {
        g2.strokeStyle = c.theory;
        g2.globalAlpha = st.theory ? 1 : 0.45;
        g2.setLineDash([5, 4]);
        g2.lineWidth = st.hover === 'fix' ? 2.2 : 1.5;
        g2.beginPath();
        const nG = 240;
        for (let i = 0; i <= nG; i++) {
          const t = (tMaxView * i) / nG;
          const xx = X(t);
          const yy = Y(core.expDecay(t, TA));
          if (i === 0) g2.moveTo(xx, yy);
          else g2.lineTo(xx, yy);
        }
        g2.stroke();
        g2.setLineDash([]);
        g2.globalAlpha = 1;
      }

      // full staircase path, faint
      g2.strokeStyle = c.volt;
      g2.lineWidth = 2;
      g2.lineJoin = 'round';
      g2.lineCap = 'round';
      g2.globalAlpha = 0.3;
      g2.beginPath();
      for (let i = 0; i <= st.n; i++) {
        const xx = X(st.ts[i]);
        const yy = Y(st.ys[i]);
        if (i === 0) g2.moveTo(xx, yy);
        else g2.lineTo(xx, yy);
      }
      g2.stroke();
      g2.globalAlpha = 1;

      // played steps, solid; the current tangent arm extended past its landing
      const b = ballState(st.tp);
      const kp = Math.min(st.n, b.k + (b.frac > 1e-9 ? 1 : 0));
      g2.lineWidth = 2.6;
      g2.strokeStyle = c.volt;
      g2.beginPath();
      g2.moveTo(X(st.ts[0]), Y(st.ys[0]));
      for (let i = 1; i <= kp; i++) {
        g2.lineTo(X(st.ts[i]), Y(st.ys[i]));
      }
      g2.stroke();

      // tangent arm: from the current corner, the slope -y/tau extended 1.45 dt,
      // then a dashed drop from the arm's tip down into the next corner
      const showTangent = st.pin === 'tangent' || st.hover === 'dt' || (st.mode === 'euler' && st.n <= 12);
      if (showTangent && st.mode === 'euler' && st.n >= 2) {
        const kC = Math.min(st.n - 1, kp);
        if (kC < st.n) {
          const yk = st.ys[kC];
          const slope = (-yk / TA) * st.dt;
          const x0 = X(st.ts[kC]);
          const y0 = Y(yk);
          const x1 = X(st.ts[kC] + 1.45 * st.dt);
          const y1 = Y(yk + slope * 1.45);
          g2.save();
          g2.strokeStyle = c.exc;
          g2.lineWidth = 2;
          g2.setLineDash([5, 4]);
          g2.beginPath();
          g2.moveTo(x0, y0);
          g2.lineTo(x1, y1);
          g2.stroke();
          // the drop to the landed corner
          g2.lineWidth = 1.4;
          g2.setLineDash([2.5, 2.5]);
          g2.beginPath();
          g2.moveTo(x1, y1);
          g2.lineTo(X(st.ts[kC + 1]), Y(st.ys[kC + 1]));
          g2.stroke();
          g2.restore();
          if (st.n <= 4) {
            note(g2, (x0 + X(st.ts[kC] + st.dt)) / 2, Y(yk + slope * 0.62) - 14,
              `slope\u00b7dt = ${fmtSig(slope)}`, c.exc, fontM, hr, c);
          }
        }
      }

      // drop ticks from every few corners to the ghost (accumulating local error)
      if (st.n <= 64 && st.theory) {
        g2.strokeStyle = c.inkMuted;
        g2.lineWidth = 1;
        g2.globalAlpha = 0.55;
        const every = Math.max(1, Math.ceil(st.n / 24));
        for (let i = every; i <= st.n; i += every) {
          g2.beginPath();
          g2.moveTo(X(st.ts[i]), Y(st.ys[i]));
          g2.lineTo(X(st.ts[i]), Y(core.expDecay(st.ts[i], TA)));
          g2.stroke();
        }
        g2.globalAlpha = 1;
      }

      // sign glyphs past dt = tau: every corner flips side of the zero axis
      if (unstableFactor && st.n <= 8) {
        g2.font = fontM(13);
        g2.fillStyle = c.mod;
        g2.textAlign = 'center';
        g2.textBaseline = 'middle';
        for (let i = 2; i <= st.n; i++) {
          const off = st.ys[i] >= 0 ? -11 : 11;
          g2.fillText(st.ys[i] >= 0 ? '+' : '\u2212', X(st.ts[i]), Y(st.ys[i]) + off);
        }
        g2.font = fontM(12);
        g2.fillStyle = c.mod;
        g2.textAlign = 'left';
        g2.textBaseline = 'bottom';
        g2.fillText(S.signFlipNote, hr.x + 8, hr.y + hr.h - 6);
      }

      // clipped blow-up: open arrows where corners leave the window
      if (blown) {
        g2.strokeStyle = c.inh;
        g2.lineWidth = 2;
        for (let i = 0; i <= st.n; i++) {
          const yv = st.ys[i];
          if (Math.abs(yv) > YCLIP) {
            const dir = yv > 0 ? -1 : 1;
            const ax = X(st.ts[i]);
            const ay = Y(dir > 0 ? -YCLIP : YCLIP);
            g2.beginPath();
            g2.moveTo(ax - 4, ay - 3 * dir);
            g2.lineTo(ax, ay);
            g2.lineTo(ax + 4, ay - 3 * dir);
            g2.stroke();
          }
        }
      }

      // corner dots
      const dotSkip = Math.max(1, Math.round(st.n / 110));
      for (let i = 0; i <= st.n; i += dotSkip) {
        g2.fillStyle = c.volt;
        g2.beginPath();
        g2.arc(X(st.ts[i]), Y(st.ys[i]), 2.1, 0, Math.PI * 2);
        g2.fill();
      }

      // difference gauge: drop line from the ball to the ghost at the playhead
      const yGho = Y(core.expDecay(b.t, TA));
      const gapHot = st.hover === 'err' || st.pin === 'error-pane';
      const gapVal = Math.abs(b.y - core.expDecay(b.t, TA));
      if (st.theory || st.hover === 'fix') {
        g2.strokeStyle = gapHot ? c.inh : c.inkMuted;
        g2.lineWidth = gapHot ? 2.2 : 1.1;
        g2.setLineDash([2.5, 2.5]);
        g2.beginPath();
        g2.moveTo(X(b.t), Y(b.y));
        g2.lineTo(X(b.t), yGho);
        g2.stroke();
        g2.setLineDash([]);
        if (gapHot || gapVal > (YMAX - YMIN) / 26) {
          g2.fillStyle = c.inh;
          g2.font = fontM(12);
          g2.textAlign = 'left';
          g2.textBaseline = 'middle';
          g2.fillText(`|dy| = ${fmtSig(gapVal)}`, X(b.t) + 7, (Y(b.y) + yGho) / 2);
        }
      }

      g2.restore(); // end clip

      // legend (top-left, wraps under the status chip reserved on the right)
      let chipText = null;
      let chipColor = c.exc;
      if (st.mode === 'euler') {
        if (blown) {
          chipText = S.chipBlown;
          chipColor = c.inh;
        } else if (st.dt > TA) {
          chipText = S.chipSwing;
          chipColor = c.mod;
        } else {
          chipText = S.chipStable;
          chipColor = c.exc;
        }
      }
      const chipW = chipText ? chipBox(g2, chipText, fontM).w + 10 : 0;
      const chipX = hr.x + hr.w - 5 - chipW;
      g2.textBaseline = 'middle';
      g2.textAlign = 'left';
      let lx = hr.x + 10;
      let ly = hr.y + 12;
      const legends = [[c.volt, null, S.legStair], [c.theory, [4, 3], S.legGhost]];
      // a narrow plot pushes the legend below the status chip instead of under it
      const firstNeed = 27 + g2.measureText(legends[0][2]).width + 14;
      if (chipText && lx + firstNeed > chipX - 8) ly = hr.y + 36;
      const wrapW = hr.x + hr.w - 6 - chipW;
      if (!chipText) ly = hr.y + 12;
      for (const [col, dash, lbl] of legends) {
        const need = 27 + g2.measureText(lbl).width + 14;
        if (lx + need > wrapW) {
          lx = hr.x + 10;
          if (chipText && ly < hr.y + 34) ly = hr.y + 36;
          else ly += 18;
        }
        g2.strokeStyle = col;
        g2.lineWidth = 2;
        g2.setLineDash(dash || []);
        g2.beginPath();
        g2.moveTo(lx, ly);
        g2.lineTo(lx + 22, ly);
        g2.stroke();
        g2.setLineDash([]);
        g2.fillStyle = c.ink;
        g2.font = fontM(12);
        g2.fillText(lbl, lx + 27, ly);
        lx += need;
      }
      // status chip (top-right): stable / swinging / blown up
      if (chipText) drawChip(g2, hr.x + hr.w - 5, hr.y + 8, chipText, chipColor, fontM, c);

      // chapter 2: numbers on the single giant step
      if (st.pin === 'numbers' && st.n >= 2) {
        note(g2, X((st.ts[0] + st.ts[1]) / 2) - 46, Y((st.ys[0] + st.ys[1]) / 2) - 16,
          `y1 = (1 - ${fmtSig(st.dt)}/${TA}) x 1 = ${fmtSig(st.ys[1])}`, c.exc, fontM, hr, c);
        note(g2, X(st.ts[1]), Y(core.expDecay(st.ts[1], TA)) + 16,
          `exact e^-1 = ${fmtSig(core.expDecay(st.ts[1], TA))}`, c.theory, fontM, hr, c);
      }

      // hover 'factor': label the multiplier at the played corner
      if (st.hover === 'factor' && st.n >= 1) {
        const kC = Math.min(st.n - 1, Math.max(0, kp - 1));
        note(g2, X(st.ts[kC] + st.dt / 2), Y(Math.min(st.ys[kC], st.ys[kC + 1]) - 0.12),
          `${S.valFactor} = ${fmtSig(core.eulerStep(1, st.dt, TA))}`, c.mod, fontM, hr, c);
      }
      // hover 'k': highlight the corner count
      if (st.hover === 'k' && st.n >= 1) {
        const kC = Math.min(st.n, kp);
        note(g2, X(st.ts[kC]), Y(st.ys[kC]) - 16,
          `k = ${kC}  y_k = ${fmtSig(st.ys[kC])}`, c.mod, fontM, hr, c);
      }
      // hover 'tau': annotate the slope law at the ball
      if (st.hover === 'tau') {
        note(g2, X(b.t + st.dt / 2), Y(b.y) - 18, `slope = -y/tau = ${fmtSig(-b.y / TA)}`, c.exc, fontM, hr, c);
      }

      // predict reveal markers
      const p = st.predict;
      if (p && p.picked !== -1) {
        const tLand = st.ts[st.n];
        const xL = X(tLand);
        const marker = (yv, col, lbl, dash, yOff) => {
          g2.strokeStyle = col;
          g2.lineWidth = 1.2;
          g2.setLineDash(dash || []);
          g2.beginPath();
          g2.moveTo(xL, hr.y);
          g2.lineTo(xL, hr.y + hr.h);
          g2.stroke();
          g2.setLineDash([]);
          g2.fillStyle = c.paperCard;
          g2.beginPath();
          g2.arc(xL, yv, 5.5, 0, Math.PI * 2);
          g2.fill();
          g2.strokeStyle = col;
          g2.lineWidth = 2;
          g2.stroke();
          g2.fillStyle = col;
          g2.font = fontM(12);
          g2.textBaseline = 'middle';
          const tw = g2.measureText(lbl).width + 8;
          const left = xL + tw < hr.x + hr.w - 4;
          g2.textAlign = left ? 'left' : 'right';
          let tys = yv + (yOff || 0);
          if (tys - 9 < hr.y + 26) tys = hr.y + 26;
          if (tys + 9 > hr.y + hr.h - 14) tys = hr.y + hr.h - 14;
          g2.fillText(lbl, xL + (left ? 8 : -8), tys);
        };
        marker(Y(core.landingValue('euler', st.dt, TA, T0)), c.volt, `${S.lblEulerLanding}: ${fmtSig(core.landingValue('euler', st.dt, TA, T0))}`, null, -12);
        if (st.theory) marker(Y(core.landingValue('exact', st.dt, TA, T0)), c.theory, `${S.lblExactLanding}: ${fmtSig(core.landingValue('exact', st.dt, TA, T0))}`, [3, 3], 13);
      }

      // ball with halo
      const xBall = X(b.t);
      const yBall = Y(b.y);
      g2.save();
      g2.shadowColor = c.volt;
      g2.shadowBlur = 14;
      g2.fillStyle = c.volt;
      g2.beginPath();
      g2.arc(xBall, yBall, 6, 0, Math.PI * 2);
      g2.fill();
      g2.restore();
      g2.strokeStyle = c.paper;
      g2.lineWidth = 1.4;
      g2.beginPath();
      g2.arc(xBall, yBall, 6, 0, Math.PI * 2);
      g2.stroke();
      // corner-arrival pulse: a ring while the ball has just crossed a corner
      if (b.frac > 0 && b.frac < 0.2 && b.k >= 1) {
        const ph = b.frac / 0.2;
        g2.strokeStyle = c.volt;
        g2.globalAlpha = 1 - ph;
        g2.lineWidth = 2;
        g2.beginPath();
        g2.arc(X(st.ts[b.k]), Y(st.ys[b.k]), 4 + 12 * ph, 0, Math.PI * 2);
        g2.stroke();
        g2.globalAlpha = 1;
      }
    }

    // x tick strip: its own band between the hero and the ruler
    function drawXTicks(g2, xt, X, tMaxView, c, fontM) {
      const xStep = tMaxView > 160 ? 40 : tMaxView > 80 ? 20 : 10;
      g2.font = fontM(12);
      g2.fillStyle = c.inkMuted;
      g2.textAlign = 'center';
      g2.textBaseline = 'top';
      for (let t = 0; t <= tMaxView + 1e-9; t += xStep) {
        const x = X(t);
        g2.beginPath();
        g2.strokeStyle = c.border;
        g2.lineWidth = 1;
        g2.moveTo(x, xt.y - 4);
        g2.lineTo(x, xt.y + 2);
        g2.stroke();
        g2.fillText(String(Math.round(t)), x, xt.y + 3);
      }
      g2.textAlign = 'right';
      g2.textBaseline = 'bottom';
      g2.fillText(S.axisT, xt.x + xt.w, xt.y + xt.h - 1);
    }

    function drawRuler(g2, r, c, fontM) {
      // band anatomy: dt label (top) | track | decade ticks | captions (bottom)
      const trackX = r.x + 2;
      const trackW = r.w - 4;
      const trackY = r.y + 30;
      const trackH = 16;
      r.trackX = trackX;
      r.trackW = trackW;

      // dt readout label, above the handle inside the band
      const hx = dtToX(r, st.dt);
      const flash = performance.now() - st.dtFlash < 500;
      g2.fillStyle = c.volt;
      g2.font = fontM(12);
      g2.textAlign = 'center';
      g2.textBaseline = 'bottom';
      const labelX = Math.min(trackX + trackW - 40, Math.max(trackX + 40, hx));
      g2.fillText(`dt = ${fmtSig(st.dt)} ms`, labelX, trackY - 9);

      // track with stability tint + zone names (never color-only)
      g2.fillStyle = c.paper;
      g2.strokeStyle = c.border;
      g2.lineWidth = 1;
      g2.fillRect(trackX, trackY, trackW, trackH);
      g2.strokeRect(trackX, trackY, trackW, trackH);
      const xLimit = dtToX(r, 2 * TA);
      g2.fillStyle = c.exc;
      g2.globalAlpha = 0.16;
      g2.fillRect(trackX, trackY, xLimit - trackX, trackH);
      g2.globalAlpha = 0.14;
      g2.fillStyle = c.inh;
      g2.fillRect(xLimit, trackY, trackX + trackW - xLimit, trackH);
      g2.globalAlpha = 1;
      // zone name inside the tinted band, on whichever side clears the handle
      g2.fillStyle = c.exc;
      g2.font = fontM(12);
      g2.textBaseline = 'middle';
      const zoneW = g2.measureText(S.zoneSafe).width;
      const hxNow = dtToX(r, st.dt);
      const fitsLeft = hxNow - (trackX + 5 + zoneW) > 14;
      const fitsRight = xLimit - 9 - (hxNow + 14) > zoneW;
      if (fitsLeft) {
        g2.textAlign = 'left';
        g2.fillText(S.zoneSafe, trackX + 5, trackY + trackH / 2);
      } else if (fitsRight) {
        g2.textAlign = 'right';
        g2.fillText(S.zoneSafe, xLimit - 9, trackY + trackH / 2);
      }

      // the dt = 2tau boundary
      const bHot = st.pin === 'stability' || st.hover === 'stability' || st.dt > 2 * TA;
      g2.strokeStyle = c.inh;
      g2.lineWidth = bHot ? 2 : 1.4;
      if (bHot) {
        g2.save();
        g2.shadowColor = c.inh;
        g2.shadowBlur = 6;
      }
      g2.beginPath();
      g2.moveTo(xLimit, trackY - 7);
      g2.lineTo(xLimit, trackY + trackH + 7);
      g2.stroke();
      if (bHot) g2.restore();

      // log decade ticks under the track
      g2.fillStyle = c.inkMuted;
      g2.font = fontM(12);
      g2.strokeStyle = c.inkMuted;
      g2.lineWidth = 1;
      g2.textAlign = 'center';
      g2.textBaseline = 'top';
      for (const dv of [0.1, 1, 10]) {
        const x = Math.min(trackX + trackW, Math.max(trackX, dtToX(r, dv)));
        g2.beginPath();
        g2.moveTo(x, trackY + trackH);
        g2.lineTo(x, trackY + trackH + 3);
        g2.stroke();
        g2.fillText(String(dv), Math.min(trackX + trackW - 12, Math.max(trackX + 12, x)), trackY + trackH + 4);
      }

      // band captions, bottom line (left title, right stability note)
      g2.textAlign = 'left';
      g2.textBaseline = 'bottom';
      g2.fillStyle = c.inkMuted;
      g2.fillText(S.axisDt, trackX + 2, r.y + r.h - 1);
      g2.textAlign = 'right';
      g2.fillStyle = c.inh;
      g2.fillText(r.w < 420 ? `2tau = ${2 * TA} ms` : `2tau = ${2 * TA} ms \u00b7 blow-up beyond`, trackX + trackW, r.y + r.h - 1);

      // draggable handle at dt
      g2.save();
      if (flash || st.hover === 'dt' || (drag && drag.kind === 'ruler')) {
        g2.shadowColor = c.volt;
        g2.shadowBlur = 9;
      }
      g2.fillStyle = c.volt;
      g2.strokeStyle = c.paper;
      g2.lineWidth = 1.4;
      g2.beginPath();
      g2.moveTo(hx, trackY + 4);
      g2.lineTo(hx - 7, trackY - 6);
      g2.lineTo(hx + 7, trackY - 6);
      g2.closePath();
      g2.fill();
      g2.stroke();
      g2.restore();

      // halve-flash ring on the track
      const now2 = performance.now();
      if (now2 - st.halveFlash < 600) {
        const ph = (now2 - st.halveFlash) / 600;
        g2.strokeStyle = c.volt;
        g2.globalAlpha = 1 - ph;
        g2.lineWidth = 2;
        g2.beginPath();
        g2.arc(hx, trackY + trackH / 2, 6 + 30 * ph, 0, Math.PI * 2);
        g2.stroke();
        g2.globalAlpha = 1;
      }
    }

    function drawError(g2, r, c, fontM) {
      if (!r || r.w < 90 || r.h < 70) return;
      const dtLo = Math.log(core.DT_MIN);
      const dtHi = Math.log(core.DT_MAX);
      const XE = (dt) => r.x + ((Math.log(dt) - dtLo) / (dtHi - dtLo)) * r.w;
      const eLo = Math.log(2e-6);
      const eHi = Math.log(6);
      const YE = (err) => r.y + r.h - ((Math.log(Math.max(err, 2e-6)) - eLo) / (eHi - eLo)) * r.h;

      // frame
      const hot = st.pin === 'error-pane' || st.hover === 'staircase';
      if (hot) {
        g2.strokeStyle = c.mod;
        g2.lineWidth = 2;
        g2.strokeRect(r.x - 3, r.y - 3, r.w + 6, r.h + 6);
      }
      g2.strokeStyle = c.ink;
      g2.lineWidth = 1;
      g2.strokeRect(r.x, r.y, r.w, r.h);

      // pane title, top-left inside the frame
      g2.fillStyle = c.ink;
      const title = S.errAt100;
      g2.font = fontM(12);
      g2.textAlign = 'left';
      g2.textBaseline = 'top';
      g2.fillText(title, r.x + 8, r.y + 6);

      // unstable band past dt = 2tau, tagged with rotated text so it never collides
      const xl = XE(2 * TA);
      if (xl < r.x + r.w - 1) {
        g2.fillStyle = c.inh;
        g2.globalAlpha = 0.08;
        g2.fillRect(xl, r.y, r.x + r.w - xl, r.h);
        g2.globalAlpha = 1;
        g2.strokeStyle = c.inh;
        g2.setLineDash([3, 3]);
        g2.lineWidth = 1;
        g2.beginPath();
        g2.moveTo(xl, r.y);
        g2.lineTo(xl, r.y + r.h);
        g2.stroke();
        g2.setLineDash([]);
        g2.save();
        g2.translate(Math.min(xl + 10, r.x + r.w - 10), r.y + r.h - 8);
        g2.rotate(-Math.PI / 2);
        g2.fillStyle = c.inh;
        g2.font = fontM(12);
        g2.textAlign = 'left';
        g2.textBaseline = 'middle';
        g2.fillText(S.lblBlown, 0, 0);
        g2.restore();
      }

      // decade labels under their ticks, bottom edge (clear of the sweep)
      g2.font = fontM(12);
      for (const dv of [0.1, 1, 10]) {
        if (dv < core.DT_MIN || dv > core.DT_MAX) continue;
        const x = XE(dv);
        g2.strokeStyle = c.border;
        g2.globalAlpha = 0.5;
        g2.lineWidth = 0.8;
        g2.beginPath();
        g2.moveTo(x, r.y);
        g2.lineTo(x, r.y + r.h);
        g2.stroke();
        g2.globalAlpha = 1;
        g2.fillStyle = c.inkMuted;
        g2.textAlign = 'center';
        g2.textBaseline = 'bottom';
        g2.fillText(String(dv), x, r.y + r.h - 2);
      }
      // y tick labels in the left gutter, rotated axis title at the far left
      g2.textAlign = 'right';
      g2.textBaseline = 'middle';
      for (let e = -6; e <= 0; e++) {
        const v = Math.pow(10, e);
        if (v < 2e-6 || v > 6) continue;
        const yv = YE(v);
        g2.strokeStyle = c.border;
        g2.globalAlpha = 0.5;
        g2.lineWidth = 0.8;
        g2.beginPath();
        g2.moveTo(r.x, yv);
        g2.lineTo(r.x + r.w, yv);
        g2.stroke();
        g2.globalAlpha = 1;
        g2.fillStyle = c.inkMuted;
        g2.fillText('1e' + e, r.x - 5, yv + (yv > r.y + 8 ? 0 : 8));
      }

      // sweep points; rose where the method explodes
      const sw = core.errorSweep(core.DT_MIN, core.DT_MAX, TA, T0, 60);
      for (let i = 0; i < sw.dts.length; i++) {
        const stablePt = core.isStable(sw.dts[i], TA);
        g2.globalAlpha = stablePt ? 0.5 : 0.85;
        g2.fillStyle = stablePt ? c.trace : c.inh;
        g2.beginPath();
        g2.arc(XE(sw.dts[i]), YE(sw.errs[i]), stablePt ? 1.8 : 2.2, 0, Math.PI * 2);
        g2.fill();
      }
      g2.globalAlpha = 1;

      // fitted line err(dt) = err(1) * dt^slope over the fit window [0.05, 2]
      g2.strokeStyle = c.exc;
      g2.lineWidth = 1.6;
      g2.setLineDash([5, 3]);
      g2.beginPath();
      g2.moveTo(XE(0.05), YE(fitPointAt1 * Math.pow(0.05, fitCache)));
      g2.lineTo(XE(2), YE(fitPointAt1 * Math.pow(2, fitCache)));
      g2.stroke();
      g2.setLineDash([]);

      // caption strip below the frame: x caption (left) and the fitted slope (right,
      // colour-matched to the fit line so the note cannot collide with the sweep)
      g2.fillStyle = c.inkMuted;
      g2.font = fontM(12);
      g2.textAlign = 'left';
      g2.textBaseline = 'bottom';
      g2.fillText(S.axisDt, r.x + 2, r.y + r.h + 14);
      const slopeTxt = r.w < 420
        ? `${S.valSlope} = ${fitCache.toFixed(2)}`
        : `${S.valSlope} = ${fitCache.toFixed(2)} \u2014 ${S.valFirstOrder}`;
      g2.fillStyle = c.exc;
      g2.textAlign = 'right';
      let slw = g2.measureText(slopeTxt).width;
      if (slw > r.w - 100) g2.font = fontM(Math.max(12, (12 * (r.w - 100)) / slw));
      g2.fillText(slopeTxt, r.x + r.w, r.y + r.h + 14);

      // the student's dt marker
      const mX = Math.min(r.x + r.w, Math.max(r.x, XE(st.dt)));
      const mY = YE(core.errorAt(st.dt, TA, T0));
      const now3 = performance.now();
      if (now3 - st.halveFlash < 700) {
        const ph = (now3 - st.halveFlash) / 700;
        g2.strokeStyle = c.volt;
        g2.globalAlpha = 1 - ph;
        g2.lineWidth = 2;
        g2.beginPath();
        g2.arc(mX, mY, 6 + 22 * ph, 0, Math.PI * 2);
        g2.stroke();
        g2.globalAlpha = 1;
      }
      g2.fillStyle = c.volt;
      g2.strokeStyle = c.paper;
      g2.lineWidth = 1.3;
      g2.beginPath();
      g2.arc(mX, mY, 4.2, 0, Math.PI * 2);
      g2.fill();
      g2.stroke();
      g2.fillStyle = c.volt;
      g2.font = fontM(12);
      const yourTxt = `${S.lblYour}: ${fmtSig(core.errorAt(st.dt, TA, T0))}`;
      const ytw = g2.measureText(yourTxt).width;
      const fromRight = mX > r.x + r.w - ytw - 14;
      g2.textAlign = fromRight ? 'right' : 'left';
      g2.textBaseline = 'top';
      const lyx = fromRight ? mX - 8 : mX + 8;
      g2.fillText(yourTxt, lyx, Math.min(mY + 6, r.y + r.h - 18));
    }

    // ---- playhead animation (simulated time advances ~24 ms per wall second) ----
    const TPLAY = 24;
    function frame() {
      const now = performance.now();
      if (st.lastNow < 0) st.lastNow = now;
      const dtw = Math.min(0.12, (now - st.lastNow) / 1000);
      st.lastNow = now;
      const playing = ctx.clock ? ctx.clock.isPlaying : true;
      if (playing) {
        const endT = st.n * st.dt;
        st.tp += dtw * TPLAY;
        if (st.tp > endT) st.tp = 0;
        render();
        refreshButtons();
      }
    }
    ctx.onFrame(frame);
    if (mqNarrow) {
      const onWidthChange = () => {
        applySizeMode();
        render();
      };
      if (typeof mqNarrow.addEventListener === 'function') mqNarrow.addEventListener('change', onWidthChange);
      else if (typeof mqNarrow.addListener === 'function') mqNarrow.addListener(onWidthChange);
    }
    setChapter(0);
    render();
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { core };
  if (typeof Atlas !== 'undefined' && Atlas.Plate) {
    Atlas.Plate.register({ id: 'M01', title: { en: S.title }, source: 'live', mount });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
