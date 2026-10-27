/**
 * PLATE Y02 — Triplet STDP and frequency dependence.
 * A live JavaScript port of neurosush.synapses.triplet.TripletSTDP.
 */
(function (root) {
  'use strict';

  // ---- String table (English only, no scattered literals) ----
  const S = {
    id: 'PLATE Y02',
    title: 'Triplet STDP: frequency leaves a memory',
    source: 'live',
    mirrors: 'neurosush.synapses.triplet.TripletSTDP · trace_decay · trace_increment',

    // Chapters
    chapter: 'Chapter',
    of: 'of',
    prev: '◀ Back',
    next: 'Next ▶',
    run: '⚡ Run 24 pairs',
    running: 'Running sweep...',
    reset: 'Reset experiment',

    // Controls & Dial
    dial: 'Pairing frequency:',
    hz: 'Hz',
    dragHint: 'drag dial',
    clickPreHint: 'Click PRE soma to fire',
    clickPostHint: 'Click POST soma to fire',

    // Hero anatomy
    preSoma: 'PRE SOMA',
    postSoma: 'POST SOMA',
    preAxon: 'Axon with Myelin Sheaths',
    terminalBouton: 'Terminal Bouton',
    synapticCleft: 'Synaptic Cleft',
    postSpine: 'Dendritic Spine',
    weightLabel: 'Synaptic weight w =',

    // Oscilloscope band
    scopeTitle: 'TRACE OSCILLOSCOPE: FAST FADES · SLOW SURVIVES',
    r1Label: 'r₁ fast pre',
    r2Label: 'r₂ slow pre',
    o1Label: 'o₁ fast post',
    o2Label: 'o₂ slow post',
    r1Formula: 'τ₊ = 16.8 ms',
    r2Formula: 'τ_x = 101 ms',
    o1Formula: 'τ₋ = 33.7 ms',
    o2Formula: 'τ_y = 125 ms',
    tickPre: 'pre spike',
    tickPost: 'post spike',

    // Plot 1: Pairing train
    plot1Title: 'Pairing train · weight w over time',
    weightUnit: 'w (unitless)',
    axisPairings: 'Pairings (1 to 24)',
    legendTriplet: 'Triplet STDP',
    legendPair: 'Pair STDP',
    legendGhost: 'Theory ghost',

    // Plot 2: Frequency sweep
    plot2Title: 'Frequency sweep · Δw vs frequency',
    deltaUnit: 'Δw (unitless)',
    axisFreq: 'Frequency (Hz)',
    sweepHint: 'Press Run to sweep 1–50 Hz',
    currentFreqTag: 'Live dial',

    // Predict-then-reveal
    predictTitle: 'Predict, then reveal',
    predictQuestion: 'At 40 Hz, will triplet STDP potentiate more or less than at 5 Hz?',
    btnMore: '▲ More potentiation',
    btnLess: '▼ Less potentiation',
    answerCorrect: 'Correct! At 40 Hz the interval (25 ms) is much shorter than slow trace τ_y (125 ms). o₂ stays elevated and multiplies the next LTP term.',
    answerIncorrect: 'Not quite. Because τ_y (125 ms) is long, post spikes at 40 Hz leave o₂ elevated, multiplying the next LTP term and producing more potentiation.',

    // Marginalia
    eqTitle: 'Governing Equations',
    eqDecay: 'r₁, o₁, r₂, o₂ ← x · (1 − dt/τ)',
    eqLtp: 'LTP = r₁ · post · (a₂⁺ + a₃⁺ · o₂_before)',
    eqLtd: 'LTD = o₁ · pre · (a₂⁻ + a₃⁻ · r₂_before)',
    eqWeight: 'w ← clamp(w + LTP − LTD, 0, 1)',

    liveTitle: 'Live Values',
    r1Live: 'fast pre trace r₁',
    o1Live: 'fast post trace o₁',
    r2Live: 'slow pre trace r₂',
    o2Live: 'slow post trace o₂',
    dwLive: 'this update Δw',
    diffLive: 'difference |live − ghost|',

    glossTitle: 'Glossary',
    glTrace: 'Trace',
    glTraceDef: 'Exponential memory of a spike: x(t) = x₀(1 − dt/τ)ᵏ.',
    glTriplet: 'Triplet STDP',
    glTripletDef: 'Plasticity where a second post spike sees the slow trace o₂, gating potentiation by rate.',
    glLtp: 'Potentiation (LTP)',
    glLtpDef: 'Synaptic strengthening triggered when pre precedes post.',

    ch1t: '1. One pair, one rule',
    ch1b: 'At 1 Hz, each pair arrives after old traces have faded. Pair and triplet rules nearly agree.',
    ch2t: '2. Repeat faster',
    ch2b: 'Turn the dial up to 10 Hz. The pair-only reference stays nearly flat; the triplet curve starts rising.',
    ch3t: '3. A slow trace remembers',
    ch3b: 'Watch the violet halo around the post cell. o₂ remains after each post spike and is sampled before the next one.',
    ch4t: '4. The third spike matters',
    ch4b: 'At 40 Hz, r₁ at a post spike is multiplied by a₂⁺ + a₃⁺o₂. Higher frequency produces dramatically more potentiation.',
    ch5t: '5. Sweep the frequencies',
    ch5b: 'Run the sweep across 1–50 Hz. Each dot is a simulated train; the dashed line is the frequency-independent pair reference.'
  };

  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), t | 1);
      r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  function requirePositive(name, value) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(name + ' must be positive, got ' + value);
    }
  }

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  const core = {
    DEFAULT: {
      a2_plus: 0.005,
      a3_plus: 0.006,
      a2_minus: 0.007,
      a3_minus: 0.0023,
      tau_plus: 16.8,
      tau_minus: 33.7,
      tau_x: 101.0,
      tau_y: 125.0,
      dt: 1.0,
      w0: 0.5
    },

    mulberry32,

    traceDecay(trace, tau, dt) {
      requirePositive('tau', tau);
      requirePositive('dt', dt);
      return trace * (1.0 - dt / tau);
    },

    traceIncrement(decayed, spike, interaction) {
      return interaction === 'nearest' ? (spike ? 1.0 : decayed) : decayed + (spike ? 1.0 : 0.0);
    },

    traceStep(trace, spike, options = {}) {
      const p = { ...core.DEFAULT, ...options };
      return core.traceIncrement(core.traceDecay(trace, p.tau, p.dt), spike, p.interaction || 'all');
    },

    initialState(w = core.DEFAULT.w0) {
      return {
        r1: 0.0,
        o1: 0.0,
        r2: 0.0,
        o2: 0.0,
        w,
        ltp: 0.0,
        ltd: 0.0,
        dw: 0.0,
        r2Before: 0.0,
        o2Before: 0.0
      };
    },

    step(state, pre, post, options = {}) {
      const p = { ...core.DEFAULT, ...options };
      ['tau_plus', 'tau_minus', 'tau_x', 'tau_y', 'dt'].forEach((k) => requirePositive(k, p[k]));

      const interaction = p.interaction || 'all';
      const r2Before = core.traceDecay(state.r2, p.tau_x, p.dt);
      const o2Before = core.traceDecay(state.o2, p.tau_y, p.dt);

      const r1 = core.traceIncrement(core.traceDecay(state.r1, p.tau_plus, p.dt), pre, interaction);
      const o1 = core.traceIncrement(core.traceDecay(state.o1, p.tau_minus, p.dt), post, interaction);
      const r2 = core.traceIncrement(r2Before, pre, interaction);
      const o2 = core.traceIncrement(o2Before, post, interaction);

      const ltp = post ? r1 * (p.a2_plus + p.a3_plus * o2Before) : 0.0;
      const ltd = pre ? o1 * (p.a2_minus + p.a3_minus * r2Before) : 0.0;
      const dw = ltp - ltd;

      return {
        r1,
        o1,
        r2,
        o2,
        w: clamp(state.w + dw, 0.0, 1.0),
        ltp,
        ltd,
        dw,
        r2Before,
        o2Before
      };
    },

    pairDelta(options = {}) {
      const p = { ...core.DEFAULT, ...options };
      const c = 1.0 - p.dt / p.tau_plus;
      return p.a2_plus * Math.pow(c, p.lag ?? 10);
    },

    train(frequency, pairs = 24, options = {}) {
      requirePositive('frequency', frequency);
      if (!Number.isInteger(pairs) || pairs < 1) {
        throw new Error('pairs must be a positive integer, got ' + pairs);
      }

      const p = { ...core.DEFAULT, ...options };
      const lag = p.lag ?? 10;
      const period = Math.max(lag + 1, Math.round(1000 / frequency));

      let s = core.initialState(p.w0);
      const history = [{ t: 0, ...s }];
      const events = [];

      for (let t = 0; t < (pairs - 1) * period + lag + 1; t += 1) {
        const pre = t % period === 0 && t / period < pairs;
        const post = t >= lag && (t - lag) % period === 0 && (t - lag) / period < pairs;

        s = core.step(s, pre, post, p);
        history.push({ t: t + 1, ...s });
        if (pre || post) {
          events.push({ t, pre, post, dw: s.dw, ...s });
        }
      }

      return {
        frequency,
        period,
        lag,
        pairs,
        state: s,
        delta: s.w - p.w0,
        history,
        events
      };
    },

    sweep(frequencies, pairs = 24, options = {}) {
      return frequencies.map((f) => ({
        frequency: f,
        ...core.train(f, pairs, options)
      }));
    },

    pairTrain(frequency, pairs = 24, options = {}) {
      return core.train(frequency, pairs, { ...options, a3_plus: 0, a3_minus: 0 });
    },

    deterministicDemo(seed = 7) {
      const rng = mulberry32(seed);
      return Array.from({ length: 8 }, () => Math.round(1 + rng() * 49));
    }
  };

  // ---- ui: only runs when a browser Atlas shell exists ----
  function mount(rootEl, ctx) {
    const margin = ctx.margin;
    const figure = ctx.figure || rootEl;
    const colors = ctx.colors;

    // Simulation state
    let frequency = 5;
    let chapter = 1;
    let showTheory = true;
    let isRunning = true;
    let elapsed = 0;
    let accumulator = 0.0;
    let hoveredTerm = null;
    let userPrediction = null;

    // Biological animation state
    let axonPulseProgress = -1.0;
    let backpropPulseProgress = -1.0;
    let plasticityPulse = 0.0;
    let preFlash = 0.0;
    let postFlash = 0.0;
    const vesiclesReleased = [];

    // Protocol isolating rate-sensitive LTP triplet
    const protocol = { a2_minus: 0, a3_minus: 0 };
    let experiment = core.train(frequency, 24, protocol);
    let pairExperiment = core.pairTrain(frequency, 24, protocol);
    let lastState = experiment.history[0];

    // Frequency sweep state: pre-calculate full sweep for instant rendering and animation
    const sweepFrequencies = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
    const fullSweepData = core.sweep(sweepFrequencies, 24, protocol);
    let sweepRevealedCount = fullSweepData.length; // Prefill so curves are visible!
    let isSweepAnimating = false;
    let sweepAnimProgress = 0;

    const chapters = [
      [S.ch1t, S.ch1b, 1],
      [S.ch2t, S.ch2b, 10],
      [S.ch3t, S.ch3b, 20],
      [S.ch4t, S.ch4b, 40],
      [S.ch5t, S.ch5b, 50]
    ];

    // Populate Margin
    if (margin) {
      margin.innerHTML = `
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.eqTitle}</div>
          <div class="atlas-equation-box atlas-mono">
            <div class="y02-term" data-term="decay">${S.eqDecay}</div>
            <div class="y02-term" data-term="ltp">${S.eqLtp}</div>
            <div class="y02-term" data-term="ltd">${S.eqLtd}</div>
            <div class="y02-term" data-term="weight">${S.eqWeight}</div>
          </div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.liveTitle}</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px;">
            <div><span style="color:var(--ink-muted);">${S.r1Live}</span><br><b class="atlas-mono" id="y02-r1">0.000</b></div>
            <div><span style="color:var(--ink-muted);">${S.o1Live}</span><br><b class="atlas-mono" id="y02-o1">0.000</b></div>
            <div><span style="color:var(--ink-muted);">${S.r2Live}</span><br><b class="atlas-mono" id="y02-r2">0.000</b></div>
            <div><span style="color:var(--ink-muted);">${S.o2Live}</span><br><b class="atlas-mono" id="y02-o2">0.000</b></div>
            <div><span style="color:var(--ink-muted);">${S.dwLive}</span><br><b class="atlas-mono" id="y02-dw">+0.00000</b></div>
            <div><span style="color:var(--ink-muted);">${S.diffLive}</span><br><b class="atlas-mono" id="y02-diff">0.0000</b></div>
          </div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.glossTitle}</div>
          <div style="font-size:12px; line-height:1.5;">
            <b>${S.glTrace}:</b> ${S.glTraceDef}<br>
            <b>${S.glTriplet}:</b> ${S.glTripletDef}<br>
            <b>${S.glLtp}:</b> ${S.glLtpDef}
          </div>
        </div>
      `;

      margin.querySelectorAll('.y02-term').forEach((el) => {
        el.style.cursor = 'help';
        el.addEventListener('mouseenter', () => { hoveredTerm = el.dataset.term; });
        el.addEventListener('mouseleave', () => { hoveredTerm = null; });
      });
    }

    // Story Chapter Banner
    const banner = document.createElement('section');
    banner.className = 'y02-banner';
    banner.style.cssText = 'border:1px solid var(--border); background:var(--paper-card); padding:8px 12px; border-radius:3px; font-size:13px; display:grid; gap:4px;';
    banner.innerHTML = `
      <div style="display:flex; justify-content:space-between; gap:8px; align-items:center; flex-wrap:wrap;">
        <span class="atlas-mono" id="y02-index" style="font-weight:600; color:var(--volt);">${S.chapter} 1 ${S.of} 5</span>
        <b id="y02-ct" style="font-family:var(--font-serif); font-size:14px;">${S.ch1t}</b>
        <span>
          <button class="atlas-btn" id="y02-prev">${S.prev}</button>
          <button class="atlas-btn" id="y02-next">${S.next}</button>
        </span>
      </div>
      <span id="y02-cb" style="color:var(--ink-muted); line-height:1.4;">${S.ch1b}</span>
    `;

    // Predict-then-reveal Card
    const predictCard = document.createElement('section');
    predictCard.className = 'y02-predict';
    predictCard.style.cssText = 'border:1px solid var(--border); background:var(--paper-inset); padding:8px 12px; border-radius:3px; font-size:12px; display:grid; gap:6px;';
    predictCard.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
        <strong style="font-family:var(--font-serif); color:var(--ink);">${S.predictTitle}</strong>
        <span style="color:var(--ink-muted);">${S.predictQuestion}</span>
      </div>
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <button class="atlas-btn" id="y02-more" style="border-color:var(--exc); color:var(--exc);">${S.btnMore}</button>
        <button class="atlas-btn" id="y02-less" style="border-color:var(--inh); color:var(--inh);">${S.btnLess}</button>
        <span id="y02-answer" style="margin-left:8px; font-weight:500;"></span>
      </div>
    `;

    // Controls Bar
    const controls = document.createElement('section');
    controls.className = 'y02-controls';
    controls.style.cssText = 'border:1px solid var(--border); background:var(--paper-card); padding:8px 12px; border-radius:3px; font-size:12px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; justify-content:space-between;';
    controls.innerHTML = `
      <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
        <button class="atlas-btn" id="y02-run" style="font-weight:600;">${S.run}</button>
        <button class="atlas-btn" id="y02-reset">${S.reset}</button>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <label for="y02-slider" style="color:var(--ink-muted);">${S.dial}</label>
        <input id="y02-slider" type="range" min="1" max="50" value="5" style="width:110px; cursor:pointer;">
        <b class="atlas-mono" id="y02-freq" style="min-width:44px; color:var(--exc);">5 Hz</b>
      </div>
    `;

    // Canvas Container placement
    const host = rootEl.classList && rootEl.classList.contains('atlas-canvas-container')
      ? rootEl
      : (rootEl.querySelector('.atlas-canvas-container') || rootEl);
    const section = host.parentElement || figure;

    host.style.setProperty('aspect-ratio', 'auto', 'important');
    host.style.setProperty('height', '550px', 'important');
    host.style.setProperty('min-height', '550px', 'important');
    host.style.setProperty('max-height', 'none', 'important');
    host.innerHTML = '';

    section.insertBefore(banner, host);
    if (host.nextSibling) {
      section.insertBefore(predictCard, host.nextSibling);
      section.insertBefore(controls, predictCard.nextSibling);
    } else {
      section.appendChild(predictCard);
      section.appendChild(controls);
    }

    const canvas = document.createElement('canvas');
    canvas.className = 'atlas-canvas';
    canvas.style.cssText = 'width:100%; height:100%; display:block; cursor:default;';
    host.appendChild(canvas);
    const g = canvas.getContext('2d');

    const $ = (id) => document.getElementById(id);
    const slider = $('y02-slider');

    function resetSimulation() {
      experiment = core.train(frequency, 24, protocol);
      pairExperiment = core.pairTrain(frequency, 24, protocol);
      elapsed = 0;
      accumulator = 0.0;
      lastState = experiment.history[0];
      axonPulseProgress = -1.0;
      backpropPulseProgress = -1.0;
      plasticityPulse = 0.0;
      vesiclesReleased.length = 0;
    }

    function setFrequency(val) {
      frequency = clamp(Math.round(val), 1, 50);
      slider.value = frequency;
      $('y02-freq').textContent = frequency + ' ' + S.hz;
      resetSimulation();
    }

    function setChapter(n) {
      chapter = clamp(n, 1, 5);
      const c = chapters[chapter - 1];
      $('y02-index').textContent = `${S.chapter} ${chapter} ${S.of} 5`;
      $('y02-ct').textContent = c[0];
      $('y02-cb').textContent = c[1];
      setFrequency(c[2]);

      if (chapter === 5) {
        startSweepAnimation();
      }
    }

    function startSweepAnimation() {
      isSweepAnimating = true;
      sweepAnimProgress = 0;
      sweepRevealedCount = 0;
      $('y02-run').textContent = S.running;
      $('y02-run').disabled = true;
    }

    function firePre() {
      preFlash = 1.0;
      axonPulseProgress = 0.0;
    }

    function firePost() {
      postFlash = 1.0;
      plasticityPulse = 1.0;
      backpropPulseProgress = 0.0;
    }

    $('y02-prev').onclick = () => setChapter(chapter - 1);
    $('y02-next').onclick = () => setChapter(chapter + 1);

    $('y02-run').onclick = () => {
      startSweepAnimation();
    };

    $('y02-reset').onclick = () => {
      sweepRevealedCount = fullSweepData.length;
      isSweepAnimating = false;
      $('y02-run').textContent = S.run;
      $('y02-run').disabled = false;
      resetSimulation();
    };

    slider.oninput = (e) => setFrequency(+e.target.value);

    $('y02-more').onclick = () => {
      userPrediction = true;
      $('y02-answer').textContent = S.answerCorrect;
      $('y02-answer').style.color = 'var(--exc)';
      showTheory = true;
      setFrequency(40);
    };

    $('y02-less').onclick = () => {
      userPrediction = false;
      $('y02-answer').textContent = S.answerIncorrect;
      $('y02-answer').style.color = 'var(--inh)';
      showTheory = true;
      setFrequency(40);
    };

    window.addEventListener('atlas:theory', (e) => {
      showTheory = e.detail && typeof e.detail.active === 'boolean' ? e.detail.active : !showTheory;
    });

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === ' ') {
        isRunning = !isRunning;
        e.preventDefault();
      } else if (e.key === 'ArrowRight') {
        setChapter(chapter + 1);
      } else if (e.key === 'ArrowLeft') {
        setChapter(chapter - 1);
      } else if (e.key === 't' || e.key === 'T') {
        showTheory = !showTheory;
      }
    });

    // Pointer Dragging for Dial and Somata clicking
    let isDraggingDial = false;
    let isDraggingSweepMarker = false;

    // Cache of layout coordinates
    let layout = {
      preX: 90,
      postX: 700,
      somaY: 75,
      somaR: 26,
      boutonX: 395,
      dial: { x: 20, y: 512, w: 760, h: 28 },
      trackX: 200,
      trackW: 400,
      plot1: { x: 20, y: 340, w: 370, h: 100 },
      plot2: { x: 410, y: 340, w: 370, h: 100 }
    };

    const padYAxis = 34;

    function getCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const x = (e.clientX - rect.left) * (canvas.width / (rect.width * dpr));
      const y = (e.clientY - rect.top) * (canvas.height / (rect.height * dpr));
      return { x, y, width: rect.width, height: rect.height };
    }

    canvas.addEventListener('pointerdown', (e) => {
      const p = getCoords(e);

      // Check click on Pre Soma
      if (Math.hypot(p.x - layout.preX, p.y - layout.somaY) <= layout.somaR + 10) {
        firePre();
        lastState = core.step(lastState, true, false, protocol);
        return;
      }

      // Check click on Post Soma
      if (Math.hypot(p.x - layout.postX, p.y - layout.somaY) <= layout.somaR + 10) {
        firePost();
        lastState = core.step(lastState, false, true, protocol);
        return;
      }

      // Check click/drag on Dial Track
      const dial = layout.dial;
      const trackX = layout.trackX;
      const trackW = layout.trackW;
      if (p.x >= trackX - 12 && p.x <= trackX + trackW + 12 && p.y >= dial.y - 12 && p.y <= dial.y + dial.h + 12) {
        isDraggingDial = true;
        canvas.setPointerCapture(e.pointerId);
        const f = 1 + clamp((p.x - trackX) / trackW, 0.0, 1.0) * 49;
        setFrequency(f);
        return;
      }

      // Check click/drag on Frequency Sweep plot marker
      const plot2 = layout.plot2;
      const ax2 = plot2.x + padYAxis;
      const pw2 = plot2.w - padYAxis;
      if (p.x >= ax2 - 5 && p.x <= ax2 + pw2 + 5 && p.y >= plot2.y && p.y <= plot2.y + plot2.h + 20) {
        isDraggingSweepMarker = true;
        canvas.setPointerCapture(e.pointerId);
        const f = 1 + clamp((p.x - ax2) / pw2, 0.0, 1.0) * 49;
        setFrequency(f);
        return;
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      const p = getCoords(e);
      if (isDraggingDial) {
        const trackX = layout.trackX;
        const trackW = layout.trackW;
        const f = 1 + clamp((p.x - trackX) / trackW, 0.0, 1.0) * 49;
        setFrequency(f);
      } else if (isDraggingSweepMarker) {
        const plot2 = layout.plot2;
        const ax2 = plot2.x + padYAxis;
        const pw2 = plot2.w - padYAxis;
        const f = 1 + clamp((p.x - ax2) / pw2, 0.0, 1.0) * 49;
        setFrequency(f);
      } else {
        const distPre = Math.hypot(p.x - layout.preX, p.y - layout.somaY);
        const distPost = Math.hypot(p.x - layout.postX, p.y - layout.somaY);
        const overDial = p.x >= layout.trackX - 10 && p.x <= layout.trackX + layout.trackW + 10 &&
                         p.y >= layout.dial.y - 10 && p.y <= layout.dial.y + layout.dial.h + 10;
        const ax2 = layout.plot2.x + padYAxis;
        const pw2 = layout.plot2.w - padYAxis;
        const overPlot2 = p.x >= ax2 - 5 && p.x <= ax2 + pw2 + 5 &&
                          p.y >= layout.plot2.y && p.y <= layout.plot2.y + layout.plot2.h + 15;

        if (distPre <= layout.somaR + 10 || distPost <= layout.somaR + 10) {
          canvas.style.cursor = 'pointer';
        } else if (overDial || overPlot2) {
          canvas.style.cursor = 'ew-resize';
        } else {
          canvas.style.cursor = 'default';
        }
      }
    });

    const stopDragging = (e) => {
      if (isDraggingDial || isDraggingSweepMarker) {
        try { canvas.releasePointerCapture(e.pointerId); } catch {}
      }
      isDraggingDial = false;
      isDraggingSweepMarker = false;
    };
    canvas.addEventListener('pointerup', stopDragging);
    canvas.addEventListener('pointercancel', stopDragging);

    // Drawing helper primitives
    function line(x1, y1, x2, y2, stroke, width = 1, dash = []) {
      g.save();
      g.strokeStyle = stroke;
      g.lineWidth = width;
      g.setLineDash(dash);
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
      g.restore();
    }

    function text(s, x, y, color, size = 12, align = 'left', bold = false) {
      g.fillStyle = color;
      g.font = `${bold ? '600 ' : ''}${size}px ui-monospace, monospace`;
      g.textAlign = align;
      g.fillText(s, x, y);
    }

    let lastTimestamp = performance.now();

    function render(timestamp) {
      const dtMs = Math.min(60, Math.max(1, timestamp - lastTimestamp));
      lastTimestamp = timestamp;

      const c = colors();
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = Math.max(1, Math.round(rect.width));
      const H = Math.max(1, Math.round(rect.height));

      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
      }

      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);

      const isMobile = W < 600;
      const targetHeight = isMobile ? '660px' : '550px';
      if (host.style.height !== targetHeight) {
        host.style.setProperty('aspect-ratio', 'auto', 'important');
        host.style.setProperty('height', targetHeight, 'important');
        host.style.setProperty('min-height', targetHeight, 'important');
        host.style.setProperty('max-height', 'none', 'important');
      }

      // Advance train simulation physics (simulated rate calibrated for lively 60 fps action)
      if (isRunning) {
        accumulator += dtMs * 0.28;
        while (accumulator >= 1.0) {
          accumulator -= 1.0;
          elapsed = (elapsed + 1) % experiment.history.length;
          lastState = experiment.history[elapsed];

          // Trigger physical events on spikes
          if (elapsed < experiment.history.length) {
            const ev = experiment.events.find((e) => e.t === elapsed);
            if (ev) {
              if (ev.pre) firePre();
              if (ev.post) firePost();
            }
          }
        }
      }

      // Advance sweep animation
      if (isSweepAnimating) {
        sweepAnimProgress += dtMs * 0.015;
        sweepRevealedCount = Math.min(fullSweepData.length, 1 + Math.floor(sweepAnimProgress));
        if (sweepRevealedCount >= fullSweepData.length) {
          isSweepAnimating = false;
          $('y02-run').textContent = S.run;
          $('y02-run').disabled = false;
        }
      }

      // Decay physical pulses
      preFlash = Math.max(0.0, preFlash - dtMs * 0.005);
      postFlash = Math.max(0.0, postFlash - dtMs * 0.005);
      plasticityPulse = Math.max(0.0, plasticityPulse - dtMs * 0.004);

      // Backpropagating dendritic pulse wave
      if (backpropPulseProgress >= 0.0) {
        backpropPulseProgress += dtMs * 0.004;
        if (backpropPulseProgress >= 1.0) backpropPulseProgress = -1.0;
      }

      // =========================================================================
      // DYNAMIC GEOMETRIC LAYOUT
      // =========================================================================
      const heroH = isMobile ? 140 : Math.round(H * 0.35); // ~190px on desktop
      const hero = { x: 12, y: 8, w: W - 24, h: heroH };

      const scopeH = 110; // Exactly meets >= 110px requirement
      const scopeY = hero.y + hero.h + 8;
      const scope = { x: 12, y: scopeY, w: W - 24, h: scopeH };

      let plot1, plot2, dial;
      let trackX, trackW;

      if (isMobile) {
        // Vertical stacked layout for phone
        const p1Y = scope.y + scope.h + 28;
        const pH = 95;
        plot1 = { x: 12, y: p1Y, w: W - 24, h: pH };

        const p2Y = plot1.y + plot1.h + 52;
        plot2 = { x: 12, y: p2Y, w: W - 24, h: pH };

        dial = { x: 12, y: plot2.y + plot2.h + 46, w: W - 24, h: 28 };
        trackX = dial.x + 10;
        trackW = dial.w - 20;
      } else {
        // Desktop 2-column side-by-side plots
        const dialH = 26;
        const dialY = H - 32;
        dial = { x: 16, y: dialY, w: W - 32, h: dialH };

        const plotsY = scope.y + scope.h + 28; // Room for plot titles & legends
        const plotsAvailH = Math.max(65, dial.y - plotsY - 24);

        const plotW = Math.floor((W - 38) / 2);
        plot1 = { x: 12, y: plotsY, w: plotW, h: plotsAvailH };
        plot2 = { x: plot1.x + plot1.w + 14, y: plotsY, w: plotW, h: plotsAvailH };

        const labelW = 160;
        const readoutW = 140;
        trackX = dial.x + labelW + 10;
        trackW = Math.max(80, dial.w - labelW - readoutW - 20);
      }

      // Cache layout for pointer hits
      const preX = hero.x + Math.max(isMobile ? 55 : 75, hero.w * (isMobile ? 0.18 : 0.17));
      const postX = hero.x + hero.w - Math.max(isMobile ? 55 : 75, hero.w * (isMobile ? 0.18 : 0.17));
      const somaY = hero.y + Math.round(hero.h * 0.44);
      const somaR = isMobile ? 22 : Math.min(27, Math.max(20, Math.round(hero.w * 0.038)));
      const boutonX = (preX + postX) / 2;

      layout = {
        preX,
        postX,
        somaY,
        somaR,
        boutonX,
        dial,
        trackX,
        trackW,
        plot1,
        plot2
      };

      // =========================================================================
      // 1. HERO BAND (40% of Figure Height): BIOLOGICAL NEURONS & 4 TRACE HALOS
      // =========================================================================
      const baseR = somaR;
      const fastPreR = baseR + 10 + lastState.r1 * (isMobile ? 18 : 26);
      const slowPreR = baseR + 16 + lastState.r2 * (isMobile ? 16 : 22);
      const fastPostR = baseR + 10 + lastState.o1 * (isMobile ? 18 : 26);
      const slowPostR = baseR + 18 + lastState.o2 * (isMobile ? 28 : 45);

      // Draw Trace Halos (4 Distinct Color Roles)
      // Pre Halos: r1 (--trace blue), r2 (--exc teal)
      // Post Halos: o1 (--inh rose), o2 (--mod violet)
      [
        [preX, somaY, slowPreR, c.exc, lastState.r2],
        [preX, somaY, fastPreR, c.trace, lastState.r1],
        [postX, somaY, slowPostR, c.mod, lastState.o2],
        [postX, somaY, fastPostR, c.inh, lastState.o1]
      ].forEach(([hx, hy, hr, color, val]) => {
        const gr = g.createRadialGradient(hx, hy, baseR * 0.5, hx, hy, hr);
        gr.addColorStop(0, color);
        gr.addColorStop(0.6, color);
        gr.addColorStop(1, 'rgba(0,0,0,0)');
        g.save();
        g.globalAlpha = Math.min(0.48, 0.06 + val * 0.38 + preFlash * 0.12);
        g.fillStyle = gr;
        g.beginPath();
        g.arc(hx, hy, hr, 0, Math.PI * 2);
        g.fill();
        g.restore();
      });

      // Anatomical Dendritic Arbors
      function drawArbor(originX, originY, dir, strokeColor) {
        g.save();
        g.strokeStyle = strokeColor;
        g.lineWidth = 1.4;
        g.lineCap = 'round';
        for (let i = -3; i <= 3; i++) {
          const dy = i * baseR * 0.44;
          const startX = originX + dir * baseR * 0.5;
          const startY = originY + dy * 0.3;
          const endX = originX + dir * (baseR * 1.5 + Math.abs(i) * 5);
          const endY = originY + dy;
          g.beginPath();
          g.moveTo(startX, startY);
          g.quadraticCurveTo((startX + endX) / 2, (startY + endY) / 2 + (i % 2 === 0 ? 4 : -4), endX, endY);
          g.stroke();

          // Sub-branch
          const midX = (startX + endX) / 2;
          const midY = (startY + endY) / 2;
          g.beginPath();
          g.moveTo(midX, midY);
          g.lineTo(midX + dir * 8, midY + (dy > 0 ? 8 : -8));
          g.stroke();
        }
        g.restore();
      }

      drawArbor(preX, somaY, -1, c.inkMuted);
      drawArbor(postX, somaY, 1, c.inkMuted);

      // Terminal Bouton geometry (grows with synaptic weight w)
      const boutonW = 12 + clamp(lastState.w, 0.0, 1.0) * (isMobile ? 18 : 24);
      const boutonGrowth = 1.0 + plasticityPulse * 0.25;
      const effectiveBoutonW = boutonW * boutonGrowth;

      // Axon Hillock (smooth funnel from pre soma to axon)
      g.save();
      g.fillStyle = c.border;
      g.beginPath();
      g.moveTo(preX + baseR - 2, somaY - 6);
      g.quadraticCurveTo(preX + baseR + 8, somaY - 2, preX + baseR + 16, somaY - 2);
      g.lineTo(boutonX - effectiveBoutonW, somaY - 2);
      g.lineTo(boutonX - effectiveBoutonW, somaY + 2);
      g.lineTo(preX + baseR + 16, somaY + 2);
      g.quadraticCurveTo(preX + baseR + 8, somaY + 2, preX + baseR - 2, somaY + 6);
      g.closePath();
      g.fill();
      g.restore();

      // Axon Cable & Myelin Sheaths from Pre to Bouton
      const axonStart = preX + baseR + 16;
      const axonEnd = boutonX - effectiveBoutonW - 4;
      line(axonStart, somaY, axonEnd, somaY, c.border, 2);

      const nSheaths = Math.max(1, Math.floor((axonEnd - axonStart) / 40));
      const sheathW = Math.max(14, (axonEnd - axonStart - (nSheaths - 1) * 8) / nSheaths);
      for (let s = 0; s < nSheaths; s++) {
        const sx = axonStart + s * (sheathW + 8);
        g.save();
        g.fillStyle = c.paperCard;
        g.strokeStyle = c.border;
        g.lineWidth = 1.2;
        g.beginPath();
        g.roundRect(sx, somaY - 6, sheathW, 12, 3);
        g.fill();
        g.stroke();
        g.restore();
      }

      // Traveling Action Potential Pulse along Axon
      if (axonPulseProgress >= 0.0) {
        axonPulseProgress += dtMs * 0.0035;
        const pulseX = preX + baseR + (boutonX - (preX + baseR)) * clamp(axonPulseProgress, 0.0, 1.0);
        g.save();
        g.fillStyle = c.volt;
        g.shadowColor = c.volt;
        g.shadowBlur = 10;
        g.beginPath();
        g.arc(pulseX, somaY, 5, 0, Math.PI * 2);
        g.fill();
        g.restore();

        if (axonPulseProgress >= 1.0) {
          axonPulseProgress = -1.0;
          // Emit neurotransmitter particles into cleft
          for (let vi = 0; vi < 12; vi++) {
            vesiclesReleased.push({
              x: boutonX + 2,
              y: somaY + (Math.random() - 0.5) * 16,
              vx: 0.07 + Math.random() * 0.10,
              vy: (Math.random() - 0.5) * 0.05,
              life: 1.0
            });
          }
        }
      }

      // Update & draw neurotransmitter particles in cleft
      for (let vi = vesiclesReleased.length - 1; vi >= 0; vi--) {
        const v = vesiclesReleased[vi];
        v.x += v.vx * dtMs;
        v.y += v.vy * dtMs;
        v.life -= dtMs * 0.003;
        if (v.life <= 0 || v.x > boutonX + 30) {
          vesiclesReleased.splice(vi, 1);
        } else {
          g.save();
          g.fillStyle = c.volt;
          g.globalAlpha = Math.max(0.2, v.life);
          g.beginPath();
          g.arc(v.x, v.y, 2.2, 0, Math.PI * 2);
          g.fill();
          g.restore();
        }
      }

      // Backpropagating Dendritic Spike along Postsynaptic Shaft
      if (backpropPulseProgress >= 0.0) {
        const bpX = postX - baseR - (postX - baseR - (boutonX + 16)) * clamp(backpropPulseProgress, 0.0, 1.0);
        g.save();
        g.fillStyle = c.volt;
        g.shadowColor = c.volt;
        g.shadowBlur = 8;
        g.beginPath();
        g.arc(bpX, somaY, 4.5, 0, Math.PI * 2);
        g.fill();
        g.restore();
      }

      // Postsynaptic Dendritic Spine & Shaft (facing cleft)
      const spineHeadX = boutonX + 16;
      g.save();
      g.fillStyle = c.paperCard;
      g.strokeStyle = c.border;
      g.lineWidth = 1.6;
      g.beginPath();
      g.moveTo(spineHeadX, somaY - 4);
      g.lineTo(postX - baseR, somaY - 4);
      g.lineTo(postX - baseR, somaY + 4);
      g.lineTo(spineHeadX, somaY + 4);
      g.closePath();
      g.fill();
      g.stroke();

      // Postsynaptic density (thick receptive membrane facing cleft)
      g.strokeStyle = c.mod;
      g.lineWidth = 3.2;
      g.beginPath();
      g.arc(spineHeadX + 2, somaY, 14, Math.PI * 0.65, Math.PI * 1.35);
      g.stroke();
      g.restore();

      // Terminal Bouton (Bulb growing with w - teardrop shape)
      g.save();
      if (plasticityPulse > 0.05) {
        g.shadowColor = c.exc;
        g.shadowBlur = 14 * plasticityPulse;
      }
      g.beginPath();
      g.moveTo(boutonX - effectiveBoutonW * 0.8, somaY - 3);
      g.quadraticCurveTo(boutonX - effectiveBoutonW * 0.3, somaY - baseR * 0.75, boutonX, somaY - baseR * 0.75);
      g.arc(boutonX, somaY, effectiveBoutonW, -Math.PI / 2, Math.PI / 2);
      g.quadraticCurveTo(boutonX - effectiveBoutonW * 0.3, somaY + baseR * 0.75, boutonX - effectiveBoutonW * 0.8, somaY + 3);
      g.closePath();
      g.fillStyle = plasticityPulse > 0.1 ? c.exc : c.paperCard;
      g.fill();
      g.strokeStyle = hoveredTerm === 'weight' ? c.volt : (plasticityPulse > 0.05 ? c.exc : c.ink);
      g.lineWidth = 2.2;
      g.stroke();
      g.restore();

      // Synaptic Vesicles inside bouton
      for (let vi = 0; vi < 8; vi++) {
        const ang = (vi / 8) * Math.PI * 2;
        const vx = boutonX - 2 + Math.cos(ang) * (effectiveBoutonW * 0.45);
        const vy = somaY + Math.sin(ang) * (baseR * 0.35);
        g.fillStyle = axonPulseProgress > 0.8 ? c.volt : c.inkMuted;
        g.beginPath();
        g.arc(vx, vy, 2, 0, Math.PI * 2);
        g.fill();
      }

      // Somata Bodies with Anatomical Nucleus
      [
        [preX, S.preSoma, c.trace, preFlash],
        [postX, S.postSoma, c.mod, postFlash]
      ].forEach(([sx, label, themeCol, flash]) => {
        g.save();
        if (flash > 0.05) {
          g.shadowColor = c.volt;
          g.shadowBlur = 14 * flash;
        }
        g.fillStyle = flash > 0.1 ? c.volt : c.paperCard;
        g.strokeStyle = themeCol;
        g.lineWidth = hoveredTerm === 'decay' ? 3.5 : 2.2;
        g.beginPath();
        g.arc(sx, somaY, baseR, 0, Math.PI * 2);
        g.fill();
        g.stroke();

        // Cell Nucleus
        g.beginPath();
        g.arc(sx - 2, somaY - 1, baseR * 0.38, 0, Math.PI * 2);
        g.fillStyle = c.border;
        g.fill();
        g.beginPath();
        g.arc(sx - 1, somaY - 1, baseR * 0.16, 0, Math.PI * 2);
        g.fillStyle = c.inkMuted;
        g.fill();

        g.restore();

        text(label, sx, somaY - baseR - 8, c.ink, 12, 'center', true);
      });

      // Hero Text Readouts
      text(isMobile ? 'PRE (click)' : S.clickPreHint, preX, hero.y + hero.h - 8, c.inkMuted, 12, 'center');
      text(isMobile ? 'POST (click)' : S.clickPostHint, postX, hero.y + hero.h - 8, c.inkMuted, 12, 'center');
      text(`${S.weightLabel} ${lastState.w.toFixed(4)}`, boutonX, somaY + baseR + 24, c.exc, 13, 'center', true);

      // =========================================================================
      // 2. OSCILLOSCOPE BAND (>= 110 px tall): 4 TRACE MEMORIES WITH GUTTER
      // =========================================================================
      line(scope.x, scope.y, scope.x + scope.w, scope.y, c.border, 1);
      line(scope.x, scope.y + scope.h, scope.x + scope.w, scope.y + scope.h, c.border, 1);

      // Left Gutter (wide enough for labels and live numbers without overlap!)
      const gutterW = isMobile ? 125 : 175;
      const waveX = scope.x + gutterW + 10;
      const waveW = scope.w - gutterW - 14;
      const waveH = scope.h - 22;
      const waveY = scope.y + 16;

      line(scope.x + gutterW, scope.y, scope.x + gutterW, scope.y + scope.h, c.border, 1);

      // Scope Title
      text(S.scopeTitle, scope.x + 8, scope.y + 11, c.inkMuted, 12, 'left', true);

      // Gutter Labels with Live Values & Colors
      const traceDefs = [
        { label: isMobile ? 'r₁ pre' : S.r1Label, key: 'r1', color: c.trace, dash: [], val: lastState.r1 },
        { label: isMobile ? 'r₂ pre' : S.r2Label, key: 'r2', color: c.exc, dash: [4, 3], val: lastState.r2 },
        { label: isMobile ? 'o₁ post' : S.o1Label, key: 'o1', color: c.inh, dash: [], val: lastState.o1 },
        { label: isMobile ? 'o₂ post' : S.o2Label, key: 'o2', color: c.mod, dash: [4, 3], val: lastState.o2 }
      ];

      traceDefs.forEach((td, idx) => {
        const rowY = scope.y + 24 + idx * 20;
        // Swatch
        line(scope.x + 8, rowY + 5, scope.x + 24, rowY + 5, td.color, 2, td.dash);
        // Text label
        text(td.label, scope.x + 28, rowY + 8, td.color, 12, 'left', true);
        // Live numeric value
        text(td.val.toFixed(3), scope.x + gutterW - 6, rowY + 8, td.color, 12, 'right');
      });

      // Oscilloscope background subtle grid lines
      line(waveX, waveY + waveH, waveX + waveW, waveY + waveH, c.border, 1);
      line(waveX, waveY + waveH * 0.5, waveX + waveW, waveY + waveH * 0.5, c.border, 0.7, [3, 4]);

      // Rolling window of trace samples
      const windowSteps = 160;
      const histSlice = experiment.history.slice(Math.max(0, elapsed - windowSteps + 1), elapsed + 1);

      // Draw Waveforms
      traceDefs.forEach((td) => {
        g.save();
        g.strokeStyle = td.color;
        g.lineWidth = 1.8;
        g.setLineDash(td.dash);
        g.beginPath();
        histSlice.forEach((samp, i) => {
          const x = waveX + (waveW * i) / Math.max(1, histSlice.length - 1);
          const y = waveY + waveH - clamp(samp[td.key], 0.0, 2.0) * (waveH * 0.45);
          if (i === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        });
        g.stroke();
        g.restore();
      });

      // Spike tick markers on oscilloscope
      const recentEvents = experiment.events.filter(
        (e) => e.t <= elapsed && e.t >= elapsed - windowSteps
      );
      recentEvents.forEach((ev) => {
        const tickX = waveX + waveW * (ev.t - (elapsed - windowSteps)) / windowSteps;
        line(tickX, waveY, tickX, waveY + waveH, c.volt, 1, [2, 3]);
        if (ev.pre) {
          text('▲ pre', tickX, waveY + 9, c.trace, 12, 'center');
        }
        if (ev.post) {
          text('▼ post', tickX, waveY + waveH - 4, c.mod, 12, 'center');
        }
      });

      // =========================================================================
      // 3. LOWER PLOTS BAND: WEIGHT OVER TIME & FREQUENCY SWEEP
      // =========================================================================

      // --- Plot 1: Pairing Train - Weight over Time ---
      const ax1 = plot1.x + padYAxis;
      const pw1 = plot1.w - padYAxis;
      line(ax1, plot1.y, ax1, plot1.y + plot1.h, c.border, 1);
      line(ax1, plot1.y + plot1.h, ax1 + pw1, plot1.y + plot1.h, c.border, 1);

      // Top title on Row 1 (plot1.y - 18)
      text(S.plot1Title, plot1.x, plot1.y - 18, c.ink, 12, 'left', true);

      // Legend on Row 2 (plot1.y - 4) with zero overlap
      line(plot1.x, plot1.y - 8, plot1.x + 14, plot1.y - 8, c.exc, 2);
      text(S.legendTriplet, plot1.x + 18, plot1.y - 5, c.exc, 12);

      line(plot1.x + 115, plot1.y - 8, plot1.x + 129, plot1.y - 8, c.trace, 1.8, [3, 2]);
      text(S.legendPair, plot1.x + 133, plot1.y - 5, c.trace, 12);

      if (showTheory) {
        line(plot1.x + 215, plot1.y - 8, plot1.x + 229, plot1.y - 8, c.theory, 1.5, [4, 3]);
        text(S.legendGhost, plot1.x + 233, plot1.y - 5, c.theory, 12);
      }

      // Numeric Y-axis ticks - cleanly placed to the left of the axis
      text('0.50', ax1 - 5, plot1.y + plot1.h - 2, c.inkMuted, 11, 'right');
      text('0.60', ax1 - 5, plot1.y + 12, c.inkMuted, 11, 'right');

      // X-axis label & ticks
      text(S.axisPairings, ax1 + pw1 / 2, plot1.y + plot1.h + 16, c.inkMuted, 12, 'center');
      text('1', ax1 + 2, plot1.y + plot1.h + 16, c.inkMuted, 12);
      text('24', ax1 + pw1 - 2, plot1.y + plot1.h + 16, c.inkMuted, 12, 'right');

      // Prefilled curves: draw full 24-pairing weight trajectory
      const maxW = Math.max(0.60, experiment.state.w, pairExperiment.state.w) + 0.01;
      const minW = 0.495;

      // Draw Pair Rule Trajectory (dotted/dashed)
      g.save();
      g.strokeStyle = c.trace;
      g.lineWidth = 1.6;
      g.setLineDash([4, 3]);
      g.beginPath();
      pairExperiment.history.forEach((h, i) => {
        const x = ax1 + (i / (pairExperiment.history.length - 1)) * pw1;
        const y = plot1.y + plot1.h - ((h.w - minW) / (maxW - minW)) * plot1.h;
        if (i === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      });
      g.stroke();
      g.restore();

      // Draw Triplet STDP Trajectory (solid curve)
      g.save();
      g.strokeStyle = c.exc;
      g.lineWidth = 2.2;
      g.beginPath();
      experiment.history.forEach((h, i) => {
        const x = ax1 + (i / (experiment.history.length - 1)) * pw1;
        const y = plot1.y + plot1.h - ((h.w - minW) / (maxW - minW)) * plot1.h;
        if (i === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      });
      g.stroke();
      g.restore();

      // Theory Ghost for Pair STDP (dashed horizontal line at final pair theoretical weight)
      if (showTheory) {
        const pairTheoryY = plot1.y + plot1.h - ((pairExperiment.state.w - minW) / (maxW - minW)) * plot1.h;
        line(ax1, pairTheoryY, ax1 + pw1, pairTheoryY, c.theory, 1.5, [5, 4]);
      }

      // Animated live cursor tracking current step
      const cursorX = ax1 + (elapsed / Math.max(1, experiment.history.length - 1)) * pw1;
      const cursorY = plot1.y + plot1.h - ((lastState.w - minW) / (maxW - minW)) * plot1.h;
      g.save();
      g.fillStyle = c.volt;
      g.beginPath();
      g.arc(cursorX, cursorY, 4.5, 0, Math.PI * 2);
      g.fill();
      g.restore();

      // --- Plot 2: Frequency Sweep ---
      const ax2 = plot2.x + padYAxis;
      const pw2 = plot2.w - padYAxis;
      line(ax2, plot2.y, ax2, plot2.y + plot2.h, c.border, 1);
      line(ax2, plot2.y + plot2.h, ax2 + pw2, plot2.y + plot2.h, c.border, 1);

      // Top title on Row 1 (plot2.y - 18)
      text(S.plot2Title, plot2.x, plot2.y - 18, c.ink, 12, 'left', true);

      // Legend on Row 2 (plot2.y - 4) with zero overlap
      line(plot2.x, plot2.y - 8, plot2.x + 14, plot2.y - 8, c.exc, 2);
      g.beginPath();
      g.arc(plot2.x + 7, plot2.y - 8, 3, 0, Math.PI * 2);
      g.fillStyle = c.exc;
      g.fill();
      text(S.legendTriplet, plot2.x + 18, plot2.y - 5, c.exc, 12);

      if (showTheory) {
        line(plot2.x + 120, plot2.y - 8, plot2.x + 134, plot2.y - 8, c.theory, 1.5, [4, 3]);
        text(S.legendGhost, plot2.x + 138, plot2.y - 5, c.theory, 12);
      }

      // Numeric Y-axis ticks - cleanly placed to the left of the axis
      const maxDelta = 0.60;
      text('0.00', ax2 - 5, plot2.y + plot2.h - 2, c.inkMuted, 11, 'right');
      text('0.55', ax2 - 5, plot2.y + 12, c.inkMuted, 11, 'right');

      // X-axis label & ticks
      text(S.axisFreq, ax2 + pw2 / 2, plot2.y + plot2.h + 16, c.inkMuted, 12, 'center');
      text('1 Hz', ax2 + 2, plot2.y + plot2.h + 16, c.inkMuted, 12);
      text('50 Hz', ax2 + pw2 - 2, plot2.y + plot2.h + 16, c.inkMuted, 12, 'right');

      // Pre-plot Pair Line (Frequency-independent theory reference, dashed line across 1 to 50 Hz)
      const pairTheoryDelta = core.pairDelta() * 24;
      const pairLineY = plot2.y + plot2.h - (pairTheoryDelta / maxDelta) * plot2.h;
      if (showTheory) {
        line(ax2, pairLineY, ax2 + pw2, pairLineY, c.theory, 1.5, [5, 4]);
      }

      // Current dial frequency marker on sweep plot
      const dialMarkerX = ax2 + ((frequency - 1) / 49) * pw2;
      line(dialMarkerX, plot2.y, dialMarkerX, plot2.y + plot2.h, c.volt, 1.2, [3, 3]);

      // Plot Triplet Sweep points and connecting curve
      const revealedPoints = fullSweepData.slice(0, sweepRevealedCount);
      if (revealedPoints.length > 0) {
        g.save();
        g.strokeStyle = c.exc;
        g.lineWidth = 1.8;
        g.beginPath();
        revealedPoints.forEach((p, idx) => {
          const px = ax2 + ((p.frequency - 1) / 49) * pw2;
          const py = plot2.y + plot2.h - (p.delta / maxDelta) * plot2.h;
          if (idx === 0) g.moveTo(px, py);
          else g.lineTo(px, py);
        });
        g.stroke();

        revealedPoints.forEach((p) => {
          const px = ax2 + ((p.frequency - 1) / 49) * pw2;
          const py = plot2.y + plot2.h - (p.delta / maxDelta) * plot2.h;
          g.fillStyle = c.exc;
          g.beginPath();
          g.arc(px, py, 3.8, 0, Math.PI * 2);
          g.fill();
        });
        g.restore();
      }

      // Live marker dot on the curve and clean label position
      const currDeltaVal = experiment.delta;
      const markerCurveY = plot2.y + plot2.h - (currDeltaVal / maxDelta) * plot2.h;
      g.save();
      g.fillStyle = c.volt;
      g.beginPath();
      g.arc(dialMarkerX, markerCurveY, 5, 0, Math.PI * 2);
      g.fill();
      g.restore();

      if (dialMarkerX > ax2 + pw2 * 0.75) {
        text(`${frequency} Hz`, dialMarkerX - 8, markerCurveY - 6, c.volt, 12, 'right', true);
      } else {
        text(`${frequency} Hz`, dialMarkerX + 8, markerCurveY - 6, c.volt, 12, 'left', true);
      }

      // =========================================================================
      // 4. DIAL SLIDER ROW: DEDICATED SEPARATE ROW WITH LABELS >= 12 PX
      // =========================================================================
      const trackY = dial.y + 13;

      if (isMobile) {
        // Mobile dial row: centered track with label above
        text(`${S.dial} ${frequency} Hz (${S.dragHint})`, dial.x + dial.w / 2, dial.y - 2, c.exc, 12, 'center', true);
        line(trackX, trackY, trackX + trackW, trackY, c.border, 3);
        const thumbX = trackX + ((frequency - 1) / 49) * trackW;
        line(trackX, trackY, thumbX, trackY, c.exc, 3.5);

        text('1 Hz', trackX, trackY + 16, c.inkMuted, 12, 'left');
        text('50 Hz', trackX + trackW, trackY + 16, c.inkMuted, 12, 'right');

        g.save();
        g.fillStyle = c.paperCard;
        g.strokeStyle = c.exc;
        g.lineWidth = 2.2;
        g.beginPath();
        g.arc(thumbX, trackY, 9, 0, Math.PI * 2);
        g.fill();
        g.stroke();
        g.restore();
      } else {
        // Desktop dial row
        text(S.dial, dial.x, trackY + 4, c.ink, 12, 'left', true);

        // Slider Track
        line(trackX, trackY, trackX + trackW, trackY, c.border, 3);
        const thumbX = trackX + ((frequency - 1) / 49) * trackW;
        line(trackX, trackY, thumbX, trackY, c.exc, 3.5);

        // Track min/max labels below track ends
        text('1 Hz', trackX, trackY + 16, c.inkMuted, 12, 'center');
        text('50 Hz', trackX + trackW, trackY + 16, c.inkMuted, 12, 'center');

        // Draggable thumb knob
        g.save();
        g.fillStyle = c.paperCard;
        g.strokeStyle = c.exc;
        g.lineWidth = 2.2;
        g.beginPath();
        g.arc(thumbX, trackY, 9, 0, Math.PI * 2);
        g.fill();
        g.stroke();
        g.restore();

        // Right Readout & Hint
        text(`${frequency} Hz (${S.dragHint})`, trackX + trackW + 14, trackY + 4, c.exc, 12, 'left', true);
      }

      // Update Marginalia readouts
      const livePairTheory = core.pairDelta() * 24;
      $('y02-r1').textContent = lastState.r1.toFixed(3);
      $('y02-o1').textContent = lastState.o1.toFixed(3);
      $('y02-r2').textContent = lastState.r2.toFixed(3);
      $('y02-o2').textContent = lastState.o2.toFixed(3);
      $('y02-dw').textContent = (lastState.dw >= 0 ? '+' : '') + lastState.dw.toFixed(5);
      $('y02-diff').textContent = Math.abs(experiment.delta - livePairTheory).toFixed(4);

      // Request next frame continuously at 60 fps!
      requestAnimationFrame(render);
    }

    if (typeof ctx.onFrame === 'function') {
      ctx.onFrame(() => render(performance.now()));
    }
    render(performance.now());
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core };
  }

  if (typeof Atlas !== 'undefined') {
    Atlas.Plate.register({
      id: 'Y02',
      title: { en: S.title },
      source: 'live',
      mount
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
