/**
 * PLATE N03 · AdEx: adaptation in the phase plane
 * Live JS simulation of the adaptive exponential integrate-and-fire model (Brette & Gerstner 2005).
 * Mirrors neurosush.neurons.dynamics.adex_terms, adaptation_step, and fire.
 * Units: mV (voltage), ms (time), nA (current), MΩ (resistance), nA/mV (conductance).
 */
(function (root) {
  'use strict';

  // Defensive shell compatibility shim
  if (typeof globalThis !== 'undefined' && typeof window !== 'undefined') {
    if (typeof globalThis.setLang === 'undefined') {
      globalThis.activeLang = 'en';
      globalThis.setLang = function (l) { globalThis.activeLang = l; };
      globalThis.getLang = function () { return globalThis.activeLang || 'en'; };
      globalThis.onLangChange = function () {};
    }
  }

  // ---- String table (English only, no scattered literals) ----
  const S = {
    plateId: 'N03',
    title: 'AdEx: adaptation in the phase plane',
    source: 'live',
    libraryMirror: 'neurosush.neurons.dynamics.adex_terms · adaptation_step · fire',

    // Marginalia equations
    eqTitle: 'Governing Equations',
    eqV: 'τ · dv/dt = (v_rest − v) + ΔT · exp((v − θ_rh) / ΔT) + R · (I − ω)',
    eqW: 'τ_w · dω/dt = a · (v − v_rest) − ω  +  b · ∑ δ(t − t_spk)',
    eqTermV: 'v: membrane potential (mV)',
    eqTermW: 'ω: adaptation current (nA)',
    eqTermI: 'I: injected current drive (nA)',
    eqTermB: 'b: spike-triggered jump in ω (nA)',
    eqTermTauW: 'τ_w: adaptation decay time constant (ms)',
    eqTermDelta: 'ΔT: exponential slope factor (mV)',
    eqTermTheta: 'θ_rh: rheobase threshold (mV)',

    // Live Readouts
    liveTitle: 'Live Dynamical State',
    readoutV: 'Potential V',
    readoutW: 'Adaptation ω',
    readoutI: 'Drive I',
    readoutB: 'Jump b',
    readoutTauW: 'Decay τ_w',
    readoutNullDist: 'Distance to v-nullcline',
    readoutLastISI: 'Latest ISI',

    // Units
    unitsMv: 'mV',
    unitsNa: 'nA',
    unitsMs: 'ms',
    unitsMohm: 'MΩ',
    unitsHz: 'Hz',

    // Direct Controls
    controlsTitle: 'Direct Controls',
    labelCurrent: 'Injected Current Step I',
    labelJump: 'Spike Adaptation Jump b',
    labelTauW: 'Brake Decay Time τ_w',
    btnTheory: 'Theory Ghost [T]',
    btnTheoryOn: 'Theory Ghost [T]: ON',
    btnTheoryOff: 'Theory Ghost [T]: OFF',
    btnReset: '↺ Reset Simulation',
    btnPerturb: '⚡ Kick Membrane (+10 mV)',
    btnPause: '⏸ Pause [Space]',
    btnResume: '▶ Play [Space]',

    // Presets
    presetsTitle: 'Dynamics Presets',
    presetTonic: 'Tonic',
    presetAdapting: 'Adapting',
    presetBursting: 'Bursting',

    // Visual Panel Titles
    panelTraceTitle: 'Membrane Potential V(t)',
    panelPhaseTitle: 'Phase Plane: V vs Adaptation ω',
    panelIsiTitle: 'Live Inter-Spike Intervals (ISIs)',

    // Phase Plane Marks & Legends
    nullclineV: 'dv/dt = 0 (v-nullcline)',
    nullclineW: 'dω/dt = 0 (ω-nullcline)',
    jumpArrowLabel: 'Reset & jump (+b)',
    phaseClickHint: 'Click phase plane to place state',
    traceClickHint: 'Click trace to perturb (+10 mV)',

    // ISI Panel Labels
    isiEmpty: 'No spikes fired yet — increase current drive I to trigger action potentials.',
    isiAdaptNote: 'Spike-frequency adaptation: ISIs lengthen as brake accumulates',
    isiTonicNote: 'Tonic spiking: intervals remain completely constant',
    isiBurstNote: 'Bursting: rapid intra-burst spikes separated by inter-burst pauses',

    // Chapters
    chapterLabel: 'CHAPTER',
    ofChapters: 'OF 5',
    btnPrev: '← Back',
    btnNext: 'Next →',
    ch1Title: '1. Plain ELIF: Exponential Escape',
    ch1Body: 'With adaptation disengaged (b = 0 nA, ω = 0 nA), a steady current step of 2.5 nA carries voltage smoothly past rheobase (-50 mV) into the exponential knee. The neuron fires regular, unadapting tonic spikes with identical intervals.',
    ch2Title: '2. Add ω: The Adaptation Brake',
    ch2Body: 'Engaging adaptation (b = 0.2 nA) introduces the negative feedback current ω. In the phase plane, ω lifts the orbit upward, opposing inward current and bending the trajectory away from immediate re-excitation.',
    ch3Title: '3. Each Spike Strengthens the Brake (b)',
    ch3Body: 'Every action potential delivers a discrete jump +b (0.5 nA) to ω at reset, seen as a purple jump arrow. As ω accumulates, subsequent spikes take longer to reach threshold: the ISI bars step upward.',
    ch4Title: '4. The Brake Leaks Away (τ_w)',
    ch4Body: 'Between action potentials, ω decays back toward baseline with time constant τ_w (240 ms). A long τ_w preserves adaptation across consecutive spikes; a short τ_w allows fast recovery.',
    ch5Title: '5. Bursting: Depolarized Reset',
    ch5Body: 'When reset voltage sits above the rheobase knee (v_reset = -47 mV > -50 mV), each reset immediately triggers another spike. A rapid burst fires until accumulated ω finally quenches the cluster, permitting a recovery pause.',

    // Predict-then-reveal
    predictTitle: 'Predict-then-Reveal Challenge',
    predictQuestion: 'Question: If you increase the spike jump b, do the inter-spike intervals (ISIs) grow faster or slower?',
    predictOptFaster: 'Grow faster (stronger adaptation per spike)',
    predictOptSlower: 'Grow slower (weaker adaptation per spike)',
    predictOptConstant: 'Stay constant (no change in interval growth)',
    predictRevealCorrect: 'Correct! Each action potential injects a larger jump +b directly into the adaptation current ω. A stronger inhibitory brake builds up earlier, delaying subsequent threshold crossings and causing ISIs to lengthen more rapidly.',
    predictRevealWrong: 'Not quite. Because b is added to the inhibitory adaptation current ω at each spike reset, a larger b applies a stronger brake immediately, causing later intervals to lengthen faster.',
    predictTheoryNote: 'The dashed theory nullcline confirms that a larger ω shifts the threshold crossing later.',

    // Glossary
    glossaryTitle: 'Glossary',
    glossaryW: 'Adaptation current (ω): A slow hyperpolarizing outward current that subtracts from injected drive.',
    glossaryNull: 'Nullcline: The locus where dv/dt = 0 (exponential U-curve) or dω/dt = 0 (linear line); their intersection sets the fixed point.',
    glossaryISI: 'Inter-spike interval (ISI): The elapsed time from one action potential peak to the next.',
    glossaryB: 'Jump parameter (b): Discrete increment added to ω at each spike, establishing spike-frequency adaptation.',
    glossaryTauW: 'Adaptation time constant (τ_w): Relaxation timescale over which ω decays back toward steady state.',

    // Honest footer
    footer: 'live · JS port of neurosush.neurons.dynamics.adex_terms, adaptation_step, and fire',
  };

  // ---- core: pure math, no DOM, exported for node tests ----
  const DEFAULTS = Object.freeze({
    tau: 20.0, // ms, membrane time constant
    v_rest: -65.0, // mV, resting potential
    resistance: 10.0, // MΩ, membrane input resistance
    delta: 2.0, // mV, exponential slope factor
    theta_rh: -50.0, // mV, rheobase threshold
    threshold: 20.0, // mV, spike peak cutoff
    v_peak: 20.0, // mV, spike peak cutoff
    v_reset: -58.0, // mV, post-spike reset potential
    current: 2.5, // nA, injected step current
    a: 0.02, // nA/mV, subthreshold adaptation conductance
    b: 0.5, // nA, spike-triggered adaptation jump
    tau_w: 120.0, // ms, adaptation decay time constant
    dt: 0.1, // ms, integration time step
    seed: 103, // PRNG seed
  });

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function finite(name, n) {
    if (!Number.isFinite(n)) {
      throw new Error(name + ' must be finite, got ' + n);
    }
  }

  function positive(name, n) {
    finite(name, n);
    if (n <= 0) {
      throw new Error(name + ' must be positive, got ' + n);
    }
  }

  function merged(params) {
    return { ...DEFAULTS, ...(params || {}) };
  }

  /**
   * Decompose one AdEx Euler step into leak, input, spike initiation, and adaptation terms.
   * Mirrors neurosush.neurons.dynamics.adex_terms.
   */
  function adex_terms(v, current, omega, params) {
    const p = merged(params);
    finite('v', v);
    finite('current', current);
    finite('omega', omega);
    positive('tau', p.tau);
    positive('delta', p.delta);
    positive('dt', p.dt);

    const h = p.dt / p.tau;
    const clampedVExp = Math.min(25.0, (v - p.theta_rh) / p.delta);
    return {
      leak: (p.v_rest - v) * h,
      input: p.resistance * current * h,
      spike_initiation: p.delta * Math.exp(clampedVExp) * h,
      adaptation: -p.resistance * omega * h,
    };
  }

  /**
   * Compute instantaneous derivatives dv/dt and dω/dt.
   */
  function derivative(v, omega, current, params) {
    const p = merged(params);
    positive('tau_w', p.tau_w);
    const t = adex_terms(v, current, omega, p);
    const dv = (t.leak + t.input + t.spike_initiation + t.adaptation) / p.dt;
    const dw = (p.a * (v - p.v_rest) - omega) / p.tau_w;
    return { dv, dw, terms: t };
  }

  /**
   * Create an initial state object.
   */
  function createState(params) {
    const p = merged(params);
    return {
      v: p.v_rest,
      omega: 0.0,
      t: 0.0,
      spikes: [],
      history: [],
      isis: [],
      lastSpikeTime: -Infinity,
    };
  }

  /**
   * Advance simulation state by one Euler step.
   * Mirrors neurosush.neurons.dynamics.adaptation_step and fire.
   */
  function step(state, params, current) {
    const p = merged(params);
    positive('dt', p.dt);
    positive('tau_w', p.tau_w);

    const I = current === undefined ? p.current : current;
    const beforeV = state.v;
    const beforeOmega = state.omega;
    const d = derivative(beforeV, beforeOmega, I, p);

    let nextV = beforeV + d.dv * p.dt;
    let nextOmega = beforeOmega + d.dw * p.dt;
    let spiked = false;

    const spikeThreshold = p.threshold !== undefined ? p.threshold : p.v_peak;
    if (nextV >= spikeThreshold) {
      spiked = true;
      nextV = p.v_reset;
      nextOmega += p.b;
      const spikeT = Math.round((state.t + p.dt) * 1e4) / 1e4;
      state.spikes.push(spikeT);
      if (state.lastSpikeTime > -Infinity) {
        const isi = Math.round((spikeT - state.lastSpikeTime) * 10) / 10;
        state.isis.push(isi);
      }
      state.lastSpikeTime = spikeT;
    }

    state.v = nextV;
    state.omega = nextOmega;
    state.t = Math.round((state.t + p.dt) * 1e4) / 1e4;

    const sample = {
      t: state.t,
      v: nextV,
      omega: nextOmega,
      spiked,
      beforeV,
      beforeOmega,
      terms: d.terms,
      dv: d.dv,
      dw: d.dw,
    };
    state.history.push(sample);
    return sample;
  }

  /**
   * Simulate AdEx neuron for a specified duration in ms.
   */
  function simulate(params, duration, stimulus) {
    const p = merged(params);
    const state = createState(p);
    const n = Math.round(duration / p.dt);
    for (let i = 0; i < n; i++) {
      const curI = stimulus ? stimulus(state.t) : p.current;
      step(state, p, curI);
    }
    const isi = state.spikes.slice(1).map((t, i) => Math.round((t - state.spikes[i]) * 10) / 10);
    return { ...state, isi };
  }

  /**
   * Closed-form v-nullcline: dv/dt = 0 solved for ω as a function of v.
   * ω_V(v) = I + [(v_rest - v) + ΔT·exp((v - θ_rh)/ΔT)] / R
   */
  function nullclineV(v, params) {
    const p = merged(params);
    const expTerm = p.delta * Math.exp(Math.min(25.0, (v - p.theta_rh) / p.delta));
    return (p.v_rest - v + expTerm + p.resistance * p.current) / p.resistance;
  }

  /**
   * Closed-form ω-nullcline: dω/dt = 0 solved for ω as a function of v.
   * ω_W(v) = a · (v - v_rest)
   */
  function nullclineW(v, params) {
    const p = merged(params);
    return p.a * (v - p.v_rest);
  }

  /**
   * Empirical steady-state firing rate f-I calculation (in Hz).
   */
  function theoryFI(current, params) {
    const p = merged(params);
    const simResult = simulate({ ...p, current }, 1200);
    const intervals = simResult.isi;
    if (intervals.length < 2) return 0.0;
    const tail = intervals.slice(-4);
    const meanISI = tail.reduce((sum, v) => sum + v, 0) / tail.length;
    return meanISI > 0 ? 1000.0 / meanISI : 0.0;
  }

  /**
   * Closed-form steady-state adaptation current estimation for firing frequency f.
   */
  function theorySteadyStateOmega(f, vMean, params) {
    const p = merged(params);
    return p.tau_w * (p.a * (vMean - p.v_rest) + p.b * (f / 1000.0));
  }

  /**
   * Standard dynamic parameter presets.
   */
  function presets(name) {
    if (name === 'tonic') {
      return {
        current: 2.5,
        b: 0.0,
        a: 0.0,
        tau_w: 100.0,
        v_reset: -58.0,
      };
    }
    if (name === 'bursting') {
      return {
        current: 2.2,
        b: 0.35,
        a: 0.04,
        tau_w: 150.0,
        v_reset: -47.0,
      };
    }
    // 'adapting' is default
    return {
      current: 2.5,
      b: 0.5,
      a: 0.02,
      tau_w: 120.0,
      v_reset: -58.0,
    };
  }

  const core = {
    DEFAULTS,
    S,
    mulberry32,
    finite,
    positive,
    adex_terms,
    derivative,
    createState,
    step,
    simulate,
    nullclineV,
    nullclineW,
    theoryFI,
    theorySteadyStateOmega,
    presets,
  };

  // ---- ui: only runs when a browser Atlas shell exists ----
  function mount(rootEl, ctx) {
    const figureContainer = ctx.figure || rootEl;
    const marginContainer = ctx.margin || document.getElementById('atlas-margin');

    // Make figure container have an explicit height so canvas fits neatly
    if (figureContainer && figureContainer.style) {
      const isNarrow = typeof window !== 'undefined' && window.innerWidth < 640;
      figureContainer.style.setProperty('aspect-ratio', 'auto', 'important');
      figureContainer.style.setProperty('height', isNarrow ? '540px' : '460px', 'important');
      figureContainer.style.setProperty('min-height', isNarrow ? '500px' : '440px', 'important');
      figureContainer.style.setProperty('max-height', '600px', 'important');
    }

    // Simulation parameter and dynamic state
    let p = { ...DEFAULTS, ...presets('adapting') };
    let state = createState(p);
    let paused = false;
    let theoryActive = true;
    let currentChapter = 1;
    let hoveredTerm = null;
    let predictAnswer = null;
    let predictRevealed = false;

    // Rolling visual history
    const traceHistory = [];
    const recentJumpArrows = []; // { x1, y1, x2, y2, time }

    function formatNumber(val, decimals = 2) {
      return Number.isFinite(val) ? val.toFixed(decimals) : '—';
    }

    function resetSimulation() {
      state = createState(p);
      traceHistory.length = 0;
      recentJumpArrows.length = 0;
    }

    // Apply Chapter settings
    const chaptersData = [
      {
        title: S.ch1Title,
        body: S.ch1Body,
        params: { current: 2.5, b: 0.0, a: 0.0, tau_w: 100.0, v_reset: -58.0 },
      },
      {
        title: S.ch2Title,
        body: S.ch2Body,
        params: { current: 2.5, b: 0.2, a: 0.02, tau_w: 100.0, v_reset: -58.0 },
      },
      {
        title: S.ch3Title,
        body: S.ch3Body,
        params: { current: 2.5, b: 0.5, a: 0.02, tau_w: 120.0, v_reset: -58.0 },
      },
      {
        title: S.ch4Title,
        body: S.ch4Body,
        params: { current: 2.5, b: 0.5, a: 0.02, tau_w: 240.0, v_reset: -58.0 },
      },
      {
        title: S.ch5Title,
        body: S.ch5Body,
        params: { current: 2.2, b: 0.35, a: 0.04, tau_w: 150.0, v_reset: -47.0 },
      },
    ];

    // ---- 1. Build Marginalia (Equations, Live State, Controls, Glossary) ----
    if (marginContainer) {
      marginContainer.innerHTML = `
        <div class="atlas-margin-section" id="n03-margin-eqs">
          <div class="atlas-margin-label">${S.eqTitle}</div>
          <div class="atlas-equation-box n03-term" data-term="v" style="cursor: pointer; margin-bottom: 6px;">
            ${S.eqV}
          </div>
          <div class="atlas-equation-box n03-term" data-term="w" style="cursor: pointer;">
            ${S.eqW}
          </div>
        </div>

        <div class="atlas-margin-section" id="n03-margin-readouts">
          <div class="atlas-margin-label">${S.liveTitle}</div>
          <div style="display: flex; flex-direction: column; gap: 4px; font-family: var(--font-mono); font-size: 11px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span class="n03-term" data-term="v" style="cursor: pointer; color: var(--volt);">${S.readoutV}:</span>
              <span class="atlas-margin-value" id="n03-val-v" style="color: var(--volt); font-weight: 600;">-65.00 mV</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span class="n03-term" data-term="w" style="cursor: pointer; color: var(--mod);">${S.readoutW}:</span>
              <span class="atlas-margin-value" id="n03-val-w" style="color: var(--mod); font-weight: 600;">0.00 nA</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span class="n03-term" data-term="i" style="cursor: pointer; color: var(--exc);">${S.readoutI}:</span>
              <span class="atlas-margin-value" id="n03-val-i">2.50 nA</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span class="n03-term" data-term="b" style="cursor: pointer;">${S.readoutB}:</span>
              <span class="atlas-margin-value" id="n03-val-b">0.50 nA</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span class="n03-term" data-term="tau" style="cursor: pointer;">${S.readoutTauW}:</span>
              <span class="atlas-margin-value" id="n03-val-tau">120 ms</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span class="n03-term" data-term="diff" style="cursor: pointer;">${S.readoutNullDist}:</span>
              <span class="atlas-margin-value" id="n03-val-diff">0.00 nA</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
              <span>${S.readoutLastISI}:</span>
              <span class="atlas-margin-value" id="n03-val-isi">—</span>
            </div>
          </div>
        </div>

        <div class="atlas-margin-section" id="n03-margin-controls">
          <div class="atlas-margin-label">${S.controlsTitle}</div>
          <div style="display: flex; flex-direction: column; gap: 8px;">
            <div>
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span>${S.labelCurrent}</span>
                <span class="atlas-mono" id="n03-out-current">2.50 nA</span>
              </div>
              <input type="range" id="n03-slider-current" min="0.5" max="4.5" step="0.05" value="2.5" style="width: 100%; accent-color: var(--exc);">
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span>${S.labelJump}</span>
                <span class="atlas-mono" id="n03-out-b">0.50 nA</span>
              </div>
              <input type="range" id="n03-slider-b" min="0" max="1.5" step="0.02" value="0.5" style="width: 100%; accent-color: var(--mod);">
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span>${S.labelTauW}</span>
                <span class="atlas-mono" id="n03-out-tau">120 ms</span>
              </div>
              <input type="range" id="n03-slider-tau" min="30" max="300" step="5" value="120" style="width: 100%; accent-color: var(--inh);">
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 2px;">
              <button class="atlas-btn" id="n03-btn-theory" style="font-size: 11px;">${S.btnTheory}</button>
              <button class="atlas-btn" id="n03-btn-pause" style="font-size: 11px;">${S.btnPause}</button>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
              <button class="atlas-btn" id="n03-btn-perturb" style="font-size: 11px;">${S.btnPerturb}</button>
              <button class="atlas-btn" id="n03-btn-reset" style="font-size: 11px;">${S.btnReset}</button>
            </div>

            <div style="margin-top: 4px;">
              <div style="font-size: 10px; color: var(--ink-muted); text-transform: uppercase; margin-bottom: 4px;">${S.presetsTitle}</div>
              <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px;">
                <button class="atlas-btn n03-preset-btn" data-preset="tonic" style="font-size: 10px; padding: 4px 2px;">${S.presetTonic}</button>
                <button class="atlas-btn n03-preset-btn" data-preset="adapting" style="font-size: 10px; padding: 4px 2px;">${S.presetAdapting}</button>
                <button class="atlas-btn n03-preset-btn" data-preset="bursting" style="font-size: 10px; padding: 4px 2px;">${S.presetBursting}</button>
              </div>
            </div>
          </div>
        </div>

        <div class="atlas-margin-section" id="n03-margin-glossary">
          <div class="atlas-margin-label">${S.glossaryTitle}</div>
          <div style="display: flex; flex-direction: column; gap: 5px; font-size: 11px; line-height: 1.35;">
            <div>• <b>ω:</b> ${S.glossaryW}</div>
            <div>• <b>Nullcline:</b> ${S.glossaryNull}</div>
            <div>• <b>ISI:</b> ${S.glossaryISI}</div>
            <div>• <b>b:</b> ${S.glossaryB}</div>
            <div>• <b>τ_w:</b> ${S.glossaryTauW}</div>
          </div>
        </div>

        <div style="font-size: 10px; color: var(--ink-muted); margin-top: auto; padding: 4px;">
          ${S.footer}
        </div>
      `;

      // Term hover events
      marginContainer.querySelectorAll('.n03-term').forEach((el) => {
        el.addEventListener('mouseenter', () => {
          hoveredTerm = el.getAttribute('data-term');
          el.style.backgroundColor = 'var(--paper-inset)';
        });
        el.addEventListener('mouseleave', () => {
          hoveredTerm = null;
          el.style.backgroundColor = 'transparent';
        });
      });
    }

    // Control sync helper
    function syncControlsUI() {
      const elI = document.getElementById('n03-slider-current');
      const outI = document.getElementById('n03-out-current');
      if (elI) elI.value = p.current;
      if (outI) outI.textContent = `${p.current.toFixed(2)} nA`;

      const elB = document.getElementById('n03-slider-b');
      const outB = document.getElementById('n03-out-b');
      if (elB) elB.value = p.b;
      if (outB) outB.textContent = `${p.b.toFixed(2)} nA`;

      const elTau = document.getElementById('n03-slider-tau');
      const outTau = document.getElementById('n03-out-tau');
      if (elTau) elTau.value = p.tau_w;
      if (outTau) outTau.textContent = `${Math.round(p.tau_w)} ms`;

      const btnTheory = document.getElementById('n03-btn-theory');
      if (btnTheory) {
        btnTheory.textContent = theoryActive ? S.btnTheoryOn : S.btnTheoryOff;
        btnTheory.classList.toggle('active', theoryActive);
      }

      const btnPause = document.getElementById('n03-btn-pause');
      if (btnPause) {
        btnPause.textContent = paused ? S.btnResume : S.btnPause;
      }

      if (marginContainer) {
        marginContainer.querySelectorAll('.n03-preset-btn').forEach((btn) => {
          const kind = btn.getAttribute('data-preset');
          const pre = presets(kind);
          const isMatch = Math.abs(pre.current - p.current) < 0.01 &&
                          Math.abs(pre.b - p.b) < 0.01 &&
                          Math.abs(pre.v_reset - p.v_reset) < 0.01;
          btn.classList.toggle('active', isMatch);
        });
      }
    }

    // Bind margin inputs
    if (marginContainer) {
      document.getElementById('n03-slider-current')?.addEventListener('input', (e) => {
        p.current = parseFloat(e.target.value);
        syncControlsUI();
      });
      document.getElementById('n03-slider-b')?.addEventListener('input', (e) => {
        p.b = parseFloat(e.target.value);
        syncControlsUI();
      });
      document.getElementById('n03-slider-tau')?.addEventListener('input', (e) => {
        p.tau_w = parseFloat(e.target.value);
        syncControlsUI();
      });
      document.getElementById('n03-btn-theory')?.addEventListener('click', () => {
        theoryActive = !theoryActive;
        syncControlsUI();
      });
      document.getElementById('n03-btn-pause')?.addEventListener('click', () => {
        paused = !paused;
        syncControlsUI();
      });
      document.getElementById('n03-btn-perturb')?.addEventListener('click', () => {
        state.v = Math.min(p.v_peak, state.v + 10.0);
      });
      document.getElementById('n03-btn-reset')?.addEventListener('click', () => {
        resetSimulation();
      });
      marginContainer.querySelectorAll('.n03-preset-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
          const kind = btn.getAttribute('data-preset');
          p = { ...DEFAULTS, ...presets(kind) };
          syncControlsUI();
          resetSimulation();
        });
      });
    }

    // ---- 2. Build Chapter Banner (Inserted above Canvas) ----
    const chapterBar = document.createElement('div');
    chapterBar.className = 'n03-chapter-bar';
    chapterBar.style.cssText =
      'background: var(--paper-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 12px; margin-bottom: 8px; display: flex; flex-direction: column; gap: 4px;';
    chapterBar.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap;">
        <div style="font-family: var(--font-serif); font-weight: 600; font-size: 14px;" id="n03-ch-title">${S.ch1Title}</div>
        <div style="display: flex; align-items: center; gap: 6px;">
          <div style="display: flex; gap: 4px;" id="n03-ch-dots">
            <span class="n03-dot" style="cursor: pointer; font-size: 13px;">●</span>
            <span class="n03-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4;">○</span>
            <span class="n03-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4;">○</span>
            <span class="n03-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4;">○</span>
            <span class="n03-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4;">○</span>
          </div>
          <button class="atlas-btn" id="n03-ch-prev" style="padding: 2px 8px; font-size: 11px;">${S.btnPrev}</button>
          <button class="atlas-btn" id="n03-ch-next" style="padding: 2px 8px; font-size: 11px;">${S.btnNext}</button>
        </div>
      </div>
      <div style="font-size: 12px; color: var(--ink-muted); line-height: 1.4;" id="n03-ch-body">
        ${S.ch1Body}
      </div>
      <div id="n03-predict-box" style="display: none; background: var(--paper-inset); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; margin-top: 4px;">
        <div style="font-weight: 600; font-size: 11px; color: var(--mod); margin-bottom: 4px;">${S.predictTitle}: ${S.predictQuestion}</div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
          <button class="atlas-btn n03-pred-btn" data-choice="faster" style="font-size: 11px;">${S.predictOptFaster}</button>
          <button class="atlas-btn n03-pred-btn" data-choice="slower" style="font-size: 11px;">${S.predictOptSlower}</button>
          <button class="atlas-btn n03-pred-btn" data-choice="constant" style="font-size: 11px;">${S.predictOptConstant}</button>
        </div>
        <div id="n03-predict-reveal" style="display: none; margin-top: 6px; font-size: 11px; line-height: 1.4;"></div>
      </div>
    `;

    if (figureContainer.parentElement) {
      figureContainer.parentElement.insertBefore(chapterBar, figureContainer);
    } else {
      figureContainer.appendChild(chapterBar);
    }

    function setChapter(chIndex) {
      currentChapter = Math.max(1, Math.min(5, chIndex));
      const z = chaptersData[currentChapter - 1];
      p = { ...p, ...z.params };

      const titleEl = document.getElementById('n03-ch-title');
      const bodyEl = document.getElementById('n03-ch-body');
      const dots = document.querySelectorAll('.n03-dot');
      const predictBox = document.getElementById('n03-predict-box');

      if (titleEl) titleEl.textContent = z.title;
      if (bodyEl) bodyEl.textContent = z.body;
      dots.forEach((d, i) => {
        d.textContent = i + 1 === currentChapter ? '●' : '○';
        d.style.opacity = i + 1 === currentChapter ? '1.0' : '0.4';
      });

      if (predictBox) {
        predictBox.style.display = currentChapter === 3 ? 'block' : 'none';
      }

      syncControlsUI();
      resetSimulation();
    }

    document.getElementById('n03-ch-prev')?.addEventListener('click', () => setChapter(currentChapter - 1));
    document.getElementById('n03-ch-next')?.addEventListener('click', () => setChapter(currentChapter + 1));
    document.querySelectorAll('.n03-dot').forEach((dot, idx) => {
      dot.addEventListener('click', () => setChapter(idx + 1));
    });

    // Predict-then-reveal buttons
    const predReveal = document.getElementById('n03-predict-reveal');
    chapterBar.querySelectorAll('.n03-pred-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        predictAnswer = btn.getAttribute('data-choice');
        predictRevealed = true;
        theoryActive = true;
        syncControlsUI();
        if (predReveal) {
          predReveal.style.display = 'block';
          const isCorrect = predictAnswer === 'faster';
          predReveal.innerHTML = `
            <div style="font-weight: 600; color: ${isCorrect ? 'var(--exc)' : 'var(--inh)'}; margin-bottom: 2px;">
              ${isCorrect ? S.predictRevealCorrect : S.predictRevealWrong}
            </div>
            <div style="color: var(--ink-muted);">${S.predictTheoryNote}</div>
          `;
        }
      });
    });

    // ---- 3. Canvas Setup & Layout ----
    const canvas = ctx.canvas();
    const g = canvas.getContext('2d');
    let colors = {};

    function computeLayout(w, h) {
      const mobile = w < 640;
      const pad = 10;

      if (mobile) {
        const traceH = Math.round(h * 0.26);
        const phaseH = Math.round(h * 0.44);
        const isiH = h - traceH - phaseH - pad * 4;

        return {
          mobile: true,
          trace: { x: pad, y: pad, w: w - pad * 2, h: traceH },
          phase: { x: pad, y: pad * 2 + traceH, w: w - pad * 2, h: phaseH },
          isi: { x: pad, y: pad * 3 + traceH + phaseH, w: w - pad * 2, h: Math.max(100, isiH) },
        };
      }

      const topH = Math.round(h * 0.62);
      const bottomH = Math.round(h * 0.33);
      const leftW = Math.round((w - pad * 3) * 0.47);
      const rightW = w - pad * 3 - leftW;

      return {
        mobile: false,
        trace: { x: pad, y: pad, w: leftW, h: topH },
        phase: { x: pad * 2 + leftW, y: pad, w: rightW, h: topH },
        isi: { x: pad, y: pad * 2 + topH, w: w - pad * 2, h: bottomH },
      };
    }

    function scaleLinear(val, domainMin, domainMax, rangeMin, rangeMax) {
      if (domainMax === domainMin) return rangeMin;
      return rangeMin + ((val - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
    }

    function drawCardBox(r, title, subtitle) {
      g.save();
      g.fillStyle = colors.paperCard || 'rgba(0,0,0,0.02)';
      g.strokeStyle = colors.border || '#d0c8b8';
      g.lineWidth = 1;

      g.beginPath();
      if (typeof g.roundRect === 'function') {
        g.roundRect(r.x, r.y, r.w, r.h, 4);
      } else {
        g.rect(r.x, r.y, r.w, r.h);
      }
      g.stroke();

      g.fillStyle = colors.ink;
      g.font = '600 12px ui-serif, Georgia, serif';
      g.textAlign = 'left';
      g.textBaseline = 'top';
      g.fillText(title, r.x + 10, r.y + 8);

      if (subtitle) {
        g.font = '10px ui-sans-serif, system-ui, sans-serif';
        const titleW = g.measureText(title).width;
        const subW = g.measureText(subtitle).width;
        if (titleW + subW + 28 <= r.w) {
          g.fillStyle = colors.inkMuted || '#7a756b';
          g.textAlign = 'right';
          g.fillText(subtitle, r.x + r.w - 10, r.y + 9);
        }
      }
      g.restore();
    }

    // ---- 4. Draw Voltage Trace Panel ----
    function drawTracePanel(r) {
      const isMobile = r.w < 560;
      drawCardBox(r, S.panelTraceTitle, isMobile ? null : S.traceClickHint);

      const x0 = r.x + 36;
      const y0 = r.y + 24;
      const w = r.w - 48;
      const h = r.h - 44;

      const vMin = -80.0;
      const vMax = 25.0;

      // Axes grid guides
      g.save();
      g.strokeStyle = colors.border || '#e0d8cc';
      g.lineWidth = 0.75;
      g.setLineDash([3, 3]);

      const yRest = scaleLinear(p.v_rest, vMin, vMax, y0 + h, y0);
      g.beginPath();
      g.moveTo(x0, yRest);
      g.lineTo(x0 + w, yRest);
      g.stroke();

      const yTheta = scaleLinear(p.theta_rh, vMin, vMax, y0 + h, y0);
      g.beginPath();
      g.moveTo(x0, yTheta);
      g.lineTo(x0 + w, yTheta);
      g.stroke();

      const yTh = scaleLinear(p.v_peak, vMin, vMax, y0 + h, y0);
      g.beginPath();
      g.moveTo(x0, yTh);
      g.lineTo(x0 + w, yTh);
      g.stroke();
      g.restore();

      // Axis lines
      g.save();
      g.strokeStyle = colors.inkMuted || '#888';
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x0, y0);
      g.lineTo(x0, y0 + h);
      g.lineTo(x0 + w, y0 + h);
      g.stroke();

      // Y-axis tick labels
      g.fillStyle = colors.inkMuted || '#888';
      g.font = '10px ui-monospace, monospace';
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillText('+20', x0 - 4, yTh);
      if (h >= 115) {
        g.fillText('-50', x0 - 4, yTheta);
        g.fillText('-65', x0 - 4, yRest);
      } else {
        g.fillText('-50', x0 - 4, yTheta);
      }
      g.fillText('-80', x0 - 4, y0 + h);

      // Injected current indicator badge with background so spikes don't occlude it
      const currentBadge = `I = ${p.current.toFixed(2)} nA`;
      g.font = '600 10px ui-monospace, monospace';
      const cbW = g.measureText(currentBadge).width;
      const cbX = x0 + w - cbW - 8;
      const cbY = y0 + 4;
      g.fillStyle = colors.paperCard || '#fbf8f2';
      g.strokeStyle = colors.border || '#d0c8b8';
      g.lineWidth = 1;
      g.beginPath();
      if (typeof g.roundRect === 'function') g.roundRect(cbX, cbY, cbW + 8, 16, 2);
      else g.rect(cbX, cbY, cbW + 8, 16);
      g.fill();
      g.stroke();
      g.fillStyle = colors.exc;
      g.textAlign = 'left';
      g.textBaseline = 'middle';
      g.fillText(currentBadge, cbX + 4, cbY + 8);
      g.restore();

      if (!traceHistory.length) return;

      const tLatest = traceHistory[traceHistory.length - 1].t;
      const timeWindow = 120.0; // 120 ms displayed
      const tStart = Math.max(0.0, tLatest - timeWindow);
      const tEnd = tStart + timeWindow;

      // Draw Voltage Trace curve
      g.save();
      g.beginPath();
      let started = false;
      for (let i = 0; i < traceHistory.length; i++) {
        const q = traceHistory[i];
        if (q.t < tStart) continue;
        const px = scaleLinear(q.t, tStart, tEnd, x0, x0 + w);
        const py = scaleLinear(q.v, vMin, vMax, y0 + h, y0);
        if (!started) {
          g.moveTo(px, py);
          started = true;
        } else {
          if (q.spiked) {
            g.lineTo(px, yTh);
          }
          g.lineTo(px, py);
        }
      }
      g.strokeStyle = hoveredTerm === 'v' ? '#ff8c00' : colors.volt;
      g.lineWidth = hoveredTerm === 'v' ? 2.5 : 1.8;
      g.shadowColor = colors.volt;
      g.shadowBlur = hoveredTerm === 'v' ? 8 : 0;
      g.stroke();
      g.restore();

      // Vertical spike marker lines
      g.save();
      for (let i = 0; i < traceHistory.length; i++) {
        const q = traceHistory[i];
        if (q.t < tStart || !q.spiked) continue;
        const px = scaleLinear(q.t, tStart, tEnd, x0, x0 + w);
        g.strokeStyle = colors.exc;
        g.lineWidth = 1.0;
        g.setLineDash([2, 2]);
        g.beginPath();
        g.moveTo(px, yTh);
        g.lineTo(px, y0 + h);
        g.stroke();

        g.fillStyle = colors.volt;
        g.beginPath();
        g.arc(px, yTh, 3, 0, Math.PI * 2);
        g.fill();
      }
      g.restore();

      // X-axis label
      g.fillStyle = colors.inkMuted;
      g.font = '10px ui-monospace, monospace';
      g.textAlign = 'right';
      g.fillText('time (ms) →', x0 + w, y0 + h + 14);
    }

    // ---- 5. Draw HERO Phase Plane Panel ----
    function drawPhasePanel(r) {
      const isMobile = r.w < 560;
      drawCardBox(r, isMobile ? 'Phase Plane: V vs ω' : S.panelPhaseTitle, isMobile ? null : S.phaseClickHint);

      const x0 = r.x + (isMobile ? 42 : 50);
      const y0 = r.y + 28;
      const w = r.w - (isMobile ? 54 : 62);
      const h = r.h - 52;

      const vMin = -80.0;
      const vMax = 25.0;
      const wMin = -0.2;
      const wMax = 3.2;

      // Coordinate axes
      g.save();
      g.strokeStyle = colors.inkMuted || '#888';
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x0, y0);
      g.lineTo(x0, y0 + h);
      g.lineTo(x0 + w, y0 + h);
      g.stroke();

      // Vertical Y-axis label
      g.save();
      g.translate(x0 - (isMobile ? 26 : 34), y0 + h / 2);
      g.rotate(-Math.PI / 2);
      g.textAlign = 'center';
      g.textBaseline = 'bottom';
      g.fillStyle = colors.inkMuted || '#888';
      g.font = (isMobile ? '9px' : '10px') + ' ui-monospace, monospace';
      g.fillText(isMobile ? 'ω (nA)' : 'Adaptation ω (nA)', 0, 0);
      g.restore();

      // Y-axis tick labels
      g.fillStyle = colors.inkMuted || '#888';
      g.font = '10px ui-monospace, monospace';
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillText('3.0', x0 - 4, scaleLinear(3.0, wMin, wMax, y0 + h, y0));
      g.fillText('2.0', x0 - 4, scaleLinear(2.0, wMin, wMax, y0 + h, y0));
      g.fillText('1.0', x0 - 4, scaleLinear(1.0, wMin, wMax, y0 + h, y0));
      g.fillText('0.0', x0 - 4, scaleLinear(0.0, wMin, wMax, y0 + h, y0));

      // X-axis tick labels
      g.textAlign = 'center';
      g.textBaseline = 'top';
      g.fillText('-80', x0, y0 + h + 4);
      g.fillText('-65', scaleLinear(-65, vMin, vMax, x0, x0 + w), y0 + h + 4);
      g.fillText('-50', scaleLinear(-50, vMin, vMax, x0, x0 + w), y0 + h + 4);
      g.fillText('+20', scaleLinear(20, vMin, vMax, x0, x0 + w), y0 + h + 4);

      // X-axis label
      g.textAlign = 'right';
      g.textBaseline = 'top';
      g.fillText('V (mV) →', x0 + w, y0 + h + 15);
      g.restore();

      // Theory Ghost: Nullclines (strictly clipped to coordinate bounds)
      if (theoryActive) {
        g.save();
        g.beginPath();
        g.rect(x0, y0, w, h);
        g.clip();

        // 1. v-nullcline (exponential U-curve)
        g.strokeStyle = colors.theory || '#222';
        g.lineWidth = 1.5;
        g.setLineDash([5, 4]);
        g.beginPath();
        let vFirst = true;
        for (let vVal = -80.0; vVal <= 2.0; vVal += 0.5) {
          const wVal = nullclineV(vVal, p);
          if (wVal > wMax + 0.5) continue;
          const px = scaleLinear(vVal, vMin, vMax, x0, x0 + w);
          const py = scaleLinear(wVal, wMin, wMax, y0 + h, y0);
          if (vFirst) {
            g.moveTo(px, py);
            vFirst = false;
          } else {
            g.lineTo(px, py);
          }
        }
        g.stroke();

        // 2. ω-nullcline (linear line)
        g.strokeStyle = colors.mod || '#7b42bc';
        g.lineWidth = 1.25;
        g.setLineDash([3, 3]);
        g.beginPath();
        let wFirst = true;
        for (let vVal = -80.0; vVal <= 20.0; vVal += 2.0) {
          const wVal = nullclineW(vVal, p);
          const px = scaleLinear(vVal, vMin, vMax, x0, x0 + w);
          const py = scaleLinear(wVal, wMin, wMax, y0 + h, y0);
          if (wFirst) {
            g.moveTo(px, py);
            wFirst = false;
          } else {
            g.lineTo(px, py);
          }
        }
        g.stroke();
        g.restore();

        // Nullclines legend badge with protective background
        g.save();
        g.font = '9px ui-monospace, monospace';
        const legW = isMobile ? 108 : 138;
        const legH = 30;
        const legX = x0 + w - legW - 4;
        const legY = y0 + 4;

        g.fillStyle = colors.paperCard || '#fbf8f2';
        g.strokeStyle = colors.border || '#d0c8b8';
        g.lineWidth = 1;
        g.beginPath();
        if (typeof g.roundRect === 'function') g.roundRect(legX, legY, legW, legH, 3);
        else g.rect(legX, legY, legW, legH);
        g.fill();
        g.stroke();

        // v-nullcline entry
        g.strokeStyle = colors.theory || '#222';
        g.lineWidth = 1.5;
        g.setLineDash([4, 3]);
        g.beginPath();
        g.moveTo(legX + 6, legY + 9);
        g.lineTo(legX + 20, legY + 9);
        g.stroke();

        g.setLineDash([]);
        g.fillStyle = colors.theory || '#222';
        g.textAlign = 'left';
        g.textBaseline = 'middle';
        g.fillText(isMobile ? 'v-nullcline' : 'dv/dt = 0 (v)', legX + 24, legY + 9);

        // w-nullcline entry
        g.strokeStyle = colors.mod || '#7b42bc';
        g.lineWidth = 1.25;
        g.setLineDash([3, 3]);
        g.beginPath();
        g.moveTo(legX + 6, legY + 21);
        g.lineTo(legX + 20, legY + 21);
        g.stroke();

        g.setLineDash([]);
        g.fillStyle = colors.mod || '#7b42bc';
        g.fillText(isMobile ? 'ω-nullcline' : 'dω/dt = 0 (ω)', legX + 24, legY + 21);
        g.restore();
      }

      // Trajectory Comet (gradient trail of recent points)
      if (traceHistory.length > 1) {
        const cometPoints = traceHistory.slice(-200);
        const count = cometPoints.length;

        g.save();
        for (let i = 1; i < count; i++) {
          const prev = cometPoints[i - 1];
          const curr = cometPoints[i];

          if (curr.spiked) continue;

          const x1 = scaleLinear(prev.v, vMin, vMax, x0, x0 + w);
          const y1 = scaleLinear(prev.omega, wMin, wMax, y0 + h, y0);
          const x2 = scaleLinear(curr.v, vMin, vMax, x0, x0 + w);
          const y2 = scaleLinear(curr.omega, wMin, wMax, y0 + h, y0);

          const alpha = (i / count) * 0.85 + 0.15;
          g.beginPath();
          g.moveTo(x1, y1);
          g.lineTo(x2, y2);
          g.strokeStyle = colors.trace || '#2563eb';
          g.globalAlpha = alpha;
          g.lineWidth = 1.0 + (i / count) * 2.2;
          g.stroke();
        }
        g.restore();

        // Comet Head (current state)
        const latest = traceHistory[traceHistory.length - 1];
        const hx = scaleLinear(latest.v, vMin, vMax, x0, x0 + w);
        const hy = scaleLinear(latest.omega, wMin, wMax, y0 + h, y0);

        g.save();
        g.fillStyle = colors.volt;
        g.shadowColor = colors.volt;
        g.shadowBlur = 10;
        g.beginPath();
        g.arc(hx, hy, 4.5, 0, Math.PI * 2);
        g.fill();

        g.strokeStyle = colors.exc;
        g.lineWidth = 1.5;
        g.beginPath();
        g.arc(hx, hy, 8, 0, Math.PI * 2);
        g.stroke();
        g.restore();
      }

      // Spike Reset & Jump Arrows
      const nowTime = state.t;
      for (let i = recentJumpArrows.length - 1; i >= 0; i--) {
        const arrow = recentJumpArrows[i];
        const age = nowTime - arrow.time;
        if (age > 450.0) {
          recentJumpArrows.splice(i, 1);
          continue;
        }

        const alpha = Math.max(0.15, 1.0 - age / 450.0);
        const x1 = scaleLinear(arrow.x1, vMin, vMax, x0, x0 + w);
        const y1 = scaleLinear(arrow.y1, wMin, wMax, y0 + h, y0);
        const x2 = scaleLinear(arrow.x2, vMin, vMax, x0, x0 + w);
        const y2 = scaleLinear(arrow.y2, wMin, wMax, y0 + h, y0);

        g.save();
        g.strokeStyle = colors.mod;
        g.fillStyle = colors.mod;
        g.globalAlpha = alpha;
        g.lineWidth = 1.75;
        g.setLineDash([4, 3]);

        g.beginPath();
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
        g.stroke();

        const angle = Math.atan2(y2 - y1, x2 - x1);
        const headLen = 8;
        g.setLineDash([]);
        g.beginPath();
        g.moveTo(x2, y2);
        g.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
        g.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
        g.closePath();
        g.fill();

        // Label only the latest reset marker with a clean pill badge so it never collides with nullclines or orbits
        if (i === recentJumpArrows.length - 1) {
          const jumpLabel = p.b > 0.01 ? `reset + b (+${p.b.toFixed(2)} nA)` : `reset (${p.v_reset.toFixed(0)} mV)`;
          g.font = '600 10px ui-monospace, monospace';
          const jw = g.measureText(jumpLabel).width;
          const badgeX = Math.min(x0 + w - jw - 10, Math.max(x0 + 4, x2 - jw / 2));
          const badgeY = Math.max(y0 + 4, y2 - 20);

          g.fillStyle = colors.paperCard || '#fbf8f2';
          g.strokeStyle = colors.mod;
          g.lineWidth = 1;
          g.beginPath();
          if (typeof g.roundRect === 'function') g.roundRect(badgeX - 4, badgeY - 2, jw + 8, 15, 2);
          else g.rect(badgeX - 4, badgeY - 2, jw + 8, 15);
          g.fill();
          g.stroke();

          g.fillStyle = colors.mod;
          g.textAlign = 'left';
          g.textBaseline = 'top';
          g.fillText(jumpLabel, badgeX, badgeY);
        }
        g.restore();
      }
    }

    // ---- 6. Draw Live ISI Bar Chart Panel ----
    function drawISIPanel(r) {
      const isMobile = r.w < 560;
      const regime = p.b === 0 ? 'Tonic' : (p.v_reset > -50.0 ? 'Bursting' : 'Adapting');
      drawCardBox(r, isMobile ? 'Live ISIs' : `${S.panelIsiTitle}  ·  ${regime}`, null);

      const x0 = r.x + 44;
      const y0 = r.y + 24;
      const w = r.w - 56;
      const h = r.h - 44;

      const intervals = state.isis.slice(-16);

      // Coordinate axes
      g.save();
      g.strokeStyle = colors.inkMuted || '#888';
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x0, y0);
      g.lineTo(x0, y0 + h);
      g.lineTo(x0 + w, y0 + h);
      g.stroke();

      g.fillStyle = colors.inkMuted || '#888';
      g.font = '10px ui-monospace, monospace';
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillText('0', x0 - 4, y0 + h);
      g.restore();

      if (!intervals.length) {
        g.save();
        g.fillStyle = colors.inkMuted;
        g.font = isMobile ? '10px ui-sans-serif, system-ui, sans-serif' : '11px ui-sans-serif, system-ui, sans-serif';
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillText(isMobile ? 'No spikes yet — increase I' : S.isiEmpty, x0 + w / 2, y0 + h / 2);
        g.restore();
        return;
      }

      const maxIsiVal = Math.max(90.0, Math.max(...intervals) * 1.15);

      // Top Y-axis tick
      g.save();
      g.fillStyle = colors.inkMuted || '#888';
      g.font = '10px ui-monospace, monospace';
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillText(`${Math.round(maxIsiVal)}`, x0 - 4, y0 + 6);
      g.fillText('ms', x0 - 4, y0 + 16);
      g.restore();

      const slotW = w / Math.max(8, intervals.length);
      const barW = Math.max(6, Math.min(28, slotW - (isMobile ? 3 : 6)));

      g.save();
      intervals.forEach((isiVal, idx) => {
        const barH = scaleLinear(isiVal, 0, maxIsiVal, 0, h - 16);
        const bx = x0 + (slotW - barW) / 2 + idx * slotW;
        const by = y0 + h - barH;

        const isLatest = idx === intervals.length - 1;
        g.fillStyle = isLatest ? colors.volt : colors.mod;
        g.beginPath();
        if (typeof g.roundRect === 'function') {
          g.roundRect(bx, by, barW, barH, [3, 3, 0, 0]);
        } else {
          g.rect(bx, by, barW, barH);
        }
        g.fill();

        // Bar value
        g.fillStyle = colors.ink;
        g.font = (isMobile ? '8px' : '9px') + ' ui-monospace, monospace';
        g.textAlign = 'center';
        g.textBaseline = 'bottom';
        if (!isMobile || idx % 2 === 0 || intervals.length <= 8) {
          g.fillText(`${isiVal.toFixed(1)}`, bx + barW / 2, by - 2);
        }

        // Spike index below axis
        g.fillStyle = colors.inkMuted;
        g.font = (isMobile ? '8px' : '9px') + ' ui-monospace, monospace';
        g.textAlign = 'center';
        g.textBaseline = 'top';
        if (!isMobile || idx % 2 === 0 || intervals.length <= 8) {
          g.fillText(`#${state.isis.length - intervals.length + idx + 1}`, bx + barW / 2, y0 + h + 3);
        }
      });

      // Mean ISI badge placed in card header safely above plot area
      const meanISI = intervals.reduce((s, v) => s + v, 0) / intervals.length;
      const rateHz = 1000.0 / meanISI;
      const meanText = isMobile ? `Mean: ${meanISI.toFixed(1)} ms (${rateHz.toFixed(0)} Hz)` : `Mean ISI: ${meanISI.toFixed(1)} ms (${rateHz.toFixed(1)} Hz)`;
      g.font = '600 10px ui-monospace, monospace';
      g.fillStyle = colors.ink;
      g.textAlign = 'right';
      g.textBaseline = 'top';
      g.fillText(meanText, r.x + r.w - 10, r.y + 8);
      g.restore();
    }

    // ---- 7. Main Render Loop ----
    function render() {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || canvas.width / dpr;
      const h = canvas.clientHeight || canvas.height / dpr;

      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }

      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      colors = ctx.colors ? ctx.colors() : {
        paper: '#f6f2e9',
        paperCard: 'rgba(0,0,0,0.02)',
        ink: '#1b1a17',
        inkMuted: '#7a756b',
        border: '#d0c8b8',
        volt: '#d97706',
        exc: '#0d9488',
        inh: '#e11d48',
        mod: '#7c3aed',
        trace: '#2563eb',
        theory: '#1b1a17',
      };

      g.clearRect(0, 0, w, h);

      const L = computeLayout(w, h);
      drawTracePanel(L.trace);
      drawPhasePanel(L.phase);
      drawISIPanel(L.isi);

      // Update marginalia readouts
      if (traceHistory.length) {
        const q = traceHistory[traceHistory.length - 1];
        const valV = document.getElementById('n03-val-v');
        const valW = document.getElementById('n03-val-w');
        const valI = document.getElementById('n03-val-i');
        const valB = document.getElementById('n03-val-b');
        const valTau = document.getElementById('n03-val-tau');
        const valDiff = document.getElementById('n03-val-diff');
        const valISI = document.getElementById('n03-val-isi');

        if (valV) valV.textContent = `${formatNumber(q.v, 2)} mV`;
        if (valW) valW.textContent = `${formatNumber(q.omega, 2)} nA`;
        if (valI) valI.textContent = `${formatNumber(p.current, 2)} nA`;
        if (valB) valB.textContent = `${formatNumber(p.b, 2)} nA`;
        if (valTau) valTau.textContent = `${Math.round(p.tau_w)} ms`;
        if (valDiff) {
          const vNull = nullclineV(q.v, p);
          valDiff.textContent = `${formatNumber(Math.abs(q.omega - vNull), 2)} nA`;
        }
        if (valISI) {
          valISI.textContent = state.isis.length ? `${state.isis[state.isis.length - 1].toFixed(1)} ms` : '—';
        }
      }
    }

    // Direct Canvas Pointer Interaction
    function getCanvasCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      return {
        x: (e.clientX - rect.left) * (canvas.width / dpr) / (rect.width || 1),
        y: (e.clientY - rect.top) * (canvas.height / dpr) / (rect.height || 1),
      };
    }

    canvas.addEventListener('pointerdown', (e) => {
      const pt = getCanvasCoords(e);
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const L = computeLayout(w, h);

      if (pt.x >= L.trace.x && pt.x <= L.trace.x + L.trace.w &&
          pt.y >= L.trace.y && pt.y <= L.trace.y + L.trace.h) {
        state.v = Math.min(p.v_peak, state.v + 10.0);
        return;
      }

      if (pt.x >= L.phase.x && pt.x <= L.phase.x + L.phase.w &&
          pt.y >= L.phase.y && pt.y <= L.phase.y + L.phase.h) {
        const fracX = (pt.x - (L.phase.x + 40)) / (L.phase.w - 52);
        const fracY = 1.0 - (pt.y - (L.phase.y + 26)) / (L.phase.h - 50);

        if (fracX >= 0 && fracX <= 1 && fracY >= 0 && fracY <= 1) {
          state.v = -80.0 + fracX * 105.0;
          state.omega = -0.2 + fracY * 3.4;
        }
      }
    });

    // Keyboard controls
    window.addEventListener('keydown', (e) => {
      if (e.target && e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        paused = !paused;
        syncControlsUI();
      } else if (e.key === 't' || e.key === 'T') {
        theoryActive = !theoryActive;
        syncControlsUI();
      } else if (e.key === 'ArrowRight') {
        setChapter(currentChapter + 1);
      } else if (e.key === 'ArrowLeft') {
        setChapter(currentChapter - 1);
      } else if (e.key === 'r' || e.key === 'R') {
        resetSimulation();
      }
    });

    // Initial controls sync & chapter start
    syncControlsUI();
    setChapter(1);

    // 1:1 real-time biological stepping
    let lastTime = performance.now();
    let simAccumulator = 0;

    function renderFrame(now) {
      if (!now) now = performance.now();
      const dtReal = Math.min(64.0, Math.max(1.0, now - lastTime));
      lastTime = now;

      if (!paused) {
        simAccumulator += dtReal;
        // Step in increments of p.dt (0.1 ms)
        const stepsToRun = Math.min(300, Math.floor(simAccumulator / p.dt));
        simAccumulator -= stepsToRun * p.dt;

        for (let s = 0; s < stepsToRun; s++) {
          const sample = step(state, p);
          traceHistory.push(sample);

          if (sample.spiked) {
            recentJumpArrows.push({
              x1: p.v_peak,
              y1: sample.beforeOmega,
              x2: p.v_reset,
              y2: sample.omega,
              time: state.t,
            });
          }
        }

        // Bounded trace window
        const tCutoff = state.t - 220.0;
        while (traceHistory.length > 0 && traceHistory[0].t < tCutoff) {
          traceHistory.shift();
        }
      }
      render();
    }

    // Initial immediate render
    renderFrame(performance.now());

    // Both rAF and setInterval so virtual-time-budget in headless chrome advances the animation
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(function raf(now) {
        renderFrame(now);
        requestAnimationFrame(raf);
      });
    }
    setInterval(() => renderFrame(performance.now()), 16);
    ctx.onFrame(() => renderFrame(performance.now()));
  }

  // Export module for Node unit tests and Atlas browser shell
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core };
  }
  if (typeof Atlas !== 'undefined') {
    Atlas.Plate.register({
      id: 'N03',
      title: { en: S.title },
      source: 'live',
      mount,
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
