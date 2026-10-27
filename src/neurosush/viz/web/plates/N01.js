(function (root) {
  'use strict';

  // ---- Strings Table (English only, no scattered literals) ----
  const S = {
    title: 'LIF: The Leaky Bucket',
    equation_label: 'Governing Equation',
    equation_formula: 'τ · dv/dt = (v_rest - v) + R · I',
    term_tau: 'τ = {tau} ms (time constant)',
    term_v: 'v = {v} mV (membrane potential)',
    term_vrest: 'v_rest = {v_rest} mV (resting potential)',
    term_R: 'R = {R} MΩ (membrane resistance)',
    term_I: 'I = {I} nA (injected current)',
    term_dvdt: 'dv/dt = {dvdt} mV/ms (voltage drift)',

    theory_box_title: 'Theory Ghost Readout',
    theory_isi: 'Theory ISI: {isi} ms ({rate} Hz)',
    meas_isi: 'Measured ISI: {isi} ms ({rate} Hz)',
    difference_readout: 'Difference: {diff} ms ({pct}%)',
    no_spiking: 'Subthreshold (No Spiking)',

    controls_title: 'Direct Controls',
    faucet_current: 'Injected Current I (Faucet)',
    threshold_slider: 'Threshold v_th (Spill Line)',
    btn_noise_off: 'Noise: OFF',
    btn_noise_on: 'Noise: ON',
    btn_kick: '⚡ Kick Membrane (+10 mV)',
    btn_theory_off: 'Theory Ghost [T]: OFF',
    btn_theory_on: 'Theory Ghost [T]: ON',
    btn_reset: '↺ Reset Simulation',

    ruler_label: 'ISI Calipers: {isi} ms ({rate} Hz)',
    ruler_hint: 'Drag calipers to measure inter-spike interval',
    f_I_title: 'f-I Transfer Function & Bifurcation',
    f_I_xaxis: 'Current I (nA) — click or drag to set',
    f_I_yaxis: 'Rate f (Hz)',
    rheobase_label: 'Rheobase {val} nA',
    live_point_label: 'Live: {I} nA, {f} Hz',
    subthreshold_zone: 'Subthreshold (0 Hz)',
    spiking_zone: 'Active Spiking',

    bucket_label: 'Membrane Bucket',
    faucet_label: 'Inflow: I = {I} nA (R·I = {val} mV)',
    leak_label: 'Passive Leak: (v - v_rest) = {val} mV',
    rest_mark: 'Rest -70 mV',
    reset_mark: 'Reset -75 mV',
    th_mark: 'v_th = {val} mV',
    refractory_lid: 'Refractory Clamped ({t} ms hold)',
    spike_pulse: 'Spike Pulse ⚡',
    bucket_hint: 'Click bucket to perturb (+10 mV)',

    ch_prev: '◀ Back',
    ch_next: 'Next ▶',
    ch_free: 'Free Play',

    ch1_title: '1. Resting Equilibrium · Leak Holds at Rest',
    ch1_desc:
      'With zero current (I = 0 nA), the membrane rests at -70 mV. A +10 mV perturbation leaks out through passive ion channels, pulling the level back down to equilibrium.',
    ch2_title: '2. Subthreshold Drive · Passive Balance',
    ch2_desc:
      'Opening the faucet to 2.0 nA (R·I = 20 mV) adds water. The level rises toward -50 mV, where inflow exactly matches outflow leak. Because R·I < v_th - v_rest (rheobase 3.0 nA), it never spills.',
    ch3_title: '3. Crossing Rheobase · Periodic Spiking',
    ch3_desc:
      'Increasing current to 3.5 nA pushes R·I = 35 mV above the 30 mV threshold distance. The bucket fills to threshold (-40 mV), trips an action potential spike, resets to -75 mV, and fires periodically.',
    ch4_title: '4. Stronger Drive · Faster Spiking',
    ch4_desc:
      'With current at 6.0 nA, inflow pours rapidly, shortening the charging interval. Notice the spilling water and spray. Drag calipers on the trace to measure the shorter inter-spike interval.',
    ch5_title: '5. The f-I Curve · Logarithmic Saturation',
    ch5_desc:
      'The firing rate curve reveals the transfer function: zero firing below rheobase (3.0 nA), a steep onset bifurcation, and sublinear logarithmic growth bounded by the refractory period.',

    predict_title: 'Predict-then-Reveal',
    predict_q: 'Double current from 3.5 nA to 7.0 nA: does the firing rate double?',
    predict_opt1: 'Yes — rate doubles (linear response)',
    predict_opt2: 'No — rate less than doubles (sublinear)',
    predict_opt3: 'No — rate more than doubles (superlinear)',
    predict_reveal_correct: 'Correct! The LIF firing rate is sublinear.',
    predict_reveal_wrong: 'Not quite! The LIF firing rate is sublinear.',
    predict_reveal_body:
      'Because T = τ · ln((R·I - (v_reset - v_rest)) / (R·I - (v_th - v_rest))) + t_ref, the rate f = 1/T saturates logarithmically as I increases and is strictly capped by the refractory period (1/t_ref = 500 Hz). The dashed theory curve confirms this.',

    glossary_title: 'Glossary',
    glossary_v: 'Membrane Potential (v): Electrical potential across the cell membrane.',
    glossary_vrest: 'Resting Potential (v_rest): Baseline steady-state voltage without input (-70 mV).',
    glossary_vth: 'Threshold (v_th): Critical voltage triggering an explosive action potential (-40 mV).',
    glossary_vreset: 'Reset Potential (v_reset): Hyperpolarized voltage immediately following a spike (-75 mV).',
    glossary_tau: 'Time Constant (τ): Timescale of passive membrane RC charging (τ = R·C = 20 ms).',
    glossary_R: 'Resistance (R): Input impedance opposing transmembrane ion flow (10 MΩ).',
    glossary_I: 'Injected Current (I): Intracellular depolarizing current from a microelectrode (nA).',
    glossary_tref: 'Refractory Period (t_ref): Absolute refractory window clamping the membrane (2.0 ms).',
    glossary_rheobase: 'Rheobase: Minimal steady current required to trigger a spike: (v_th - v_rest)/R = 3.0 nA.',

    footer_source: 'live (JS port: neurosush.neurons.dynamics.lif_derivative + euler_step + fire + Refractory)',
  };

  // ---- core: pure math, no DOM, exported for node tests ----
  const DEFAULTS = Object.freeze({
    tau: 20.0, // ms
    v_rest: -70.0, // mV
    threshold: -40.0, // mV
    v_reset: -75.0, // mV
    resistance: 10.0, // MOhm
    current: 0.0, // nA
    dt: 0.1, // ms
    t_ref: 2.0, // ms
    noise_enabled: false,
    noise_sigma: 0.5, // nA
    seed: 42,
  });

  function mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rng) {
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function lif_derivative(v, current, params = DEFAULTS) {
    const v_rest = params.v_rest ?? DEFAULTS.v_rest;
    const resistance = params.resistance ?? DEFAULTS.resistance;
    return (v_rest - v) + resistance * current;
  }

  function euler_step(v, tau_dv_dt, params = DEFAULTS) {
    const tau = params.tau ?? DEFAULTS.tau;
    const dt = params.dt ?? DEFAULTS.dt;
    return v + tau_dv_dt * (dt / tau);
  }

  function fire(v, params = DEFAULTS) {
    const threshold = params.threshold ?? DEFAULTS.threshold;
    const v_reset = params.v_reset ?? DEFAULTS.v_reset;
    const spiked = v >= threshold;
    return { spiked, v: spiked ? v_reset : v };
  }

  function createState(overrides = {}) {
    const p = { ...DEFAULTS, ...overrides };
    return {
      t: 0,
      v: p.v_rest,
      refractory_time: 0,
      spiked: false,
      spikeCount: 0,
      lastSpikeTime: -Infinity,
      spikeTimes: [],
      isiList: [],
      params: p,
      rng: mulberry32(p.seed ?? 42),
    };
  }

  function step(state, dt = state.params.dt) {
    const p = state.params;
    let current_eff = p.current;
    if (p.noise_enabled) {
      current_eff += gaussian(state.rng) * (p.noise_sigma ?? 0.5);
    }

    let v = state.v;
    let refractory_time = state.refractory_time;
    let spiked = false;

    if (refractory_time > 1e-9) {
      refractory_time = Math.max(0, refractory_time - dt);
      v = p.v_reset;
    } else {
      const tau_dv_dt = lif_derivative(v, current_eff, p);
      v = euler_step(v, tau_dv_dt, { ...p, dt });
    }

    // Threshold crossing check
    if (state.refractory_time <= 1e-9 && v >= p.threshold) {
      spiked = true;
      v = p.v_reset;
      refractory_time = p.t_ref;
      if (state.lastSpikeTime > -Infinity) {
        state.isiList.push(state.t - state.lastSpikeTime);
      }
      state.lastSpikeTime = state.t;
      state.spikeTimes.push(state.t);
      state.spikeCount++;
    }

    state.t += dt;
    state.v = v;
    state.refractory_time = refractory_time;
    state.spiked = spiked;

    return {
      t: state.t,
      v,
      spiked,
      refractory_time,
      current_eff,
      tau_dv_dt: lif_derivative(v, current_eff, p),
    };
  }

  function theoryISI(current, params = DEFAULTS) {
    const p = { ...DEFAULTS, ...params };
    const RI = p.resistance * current;
    const d_reset = p.v_reset - p.v_rest;
    const d_th = p.threshold - p.v_rest;
    if (RI <= d_th) return Infinity;
    return p.tau * Math.log((RI - d_reset) / (RI - d_th)) + p.t_ref;
  }

  function theoryRate(current, params = DEFAULTS) {
    const T = theoryISI(current, params);
    return Number.isFinite(T) && T > 0 ? 1000.0 / T : 0;
  }

  function theoryVoltage(t, current, params = DEFAULTS, v_start = null) {
    const p = { ...DEFAULTS, ...params };
    const v_inf = p.v_rest + p.resistance * current;
    const v0 = v_start !== null ? v_start : p.v_rest;
    return v_inf + (v0 - v_inf) * Math.exp(-t / p.tau);
  }

  function simulateRun(params, totalMs) {
    const p = { ...DEFAULTS, ...params };
    const dt = p.dt;
    const totalSteps = Math.round(totalMs / dt);
    const state = createState(p);
    for (let s = 0; s < totalSteps; s++) {
      step(state, dt);
    }
    const spikes = state.spikeCount;
    const rate = spikes / (totalMs / 1000.0);
    const isis = state.isiList;
    const minISI = isis.length > 0 ? Math.min(...isis) : Infinity;
    const meanISI = isis.length > 0 ? isis.reduce((a, b) => a + b, 0) / isis.length : Infinity;
    return {
      spikes,
      rate,
      spikeTimes: state.spikeTimes,
      isiList: isis,
      minISI,
      meanISI,
      finalV: state.v,
    };
  }

  // Deterministic physical droplet simulation with seeded PRNG
  function createDroplets(seed = 42) {
    const rng = mulberry32(seed);
    const inflow = [];
    const leak = [];
    const splashes = [];
    let inflowTimer = 0;
    let leakTimer = 0;

    function reset() {
      inflow.length = 0;
      leak.length = 0;
      splashes.length = 0;
      inflowTimer = 0;
      leakTimer = 0;
    }

    function update(dtMs, current, v, v_rest, faucetX, faucetY, waterTopY, leakX, leakY, basinY) {
      const dtFrac = dtMs / 16.667;

      // Inflow droplets when faucet is open
      if (current > 0.05) {
        inflowTimer += dtMs * (current / 6.0);
        if (inflowTimer >= 14.0) {
          inflowTimer = 0;
          inflow.push({
            x: faucetX + (rng() - 0.5) * 6,
            y: faucetY + 6,
            vy: 2.0 + (current / 6.0) * 2.5,
            r: 1.2 + (current / 6.0) * 1.5,
          });
        }
      }
      for (let i = inflow.length - 1; i >= 0; i--) {
        const d = inflow[i];
        d.y += d.vy * dtFrac;
        d.vy += 0.18 * dtFrac;
        if (d.y >= waterTopY) {
          inflow.splice(i, 1);
        }
      }

      // Leak droplets when v > v_rest
      const leakDelta = Math.max(0, v - v_rest);
      if (leakDelta > 0.05) {
        leakTimer += dtMs * (leakDelta / 30.0);
        if (leakTimer >= 16.0) {
          leakTimer = 0;
          leak.push({
            x: leakX + (rng() - 0.5) * 4,
            y: leakY + 2,
            vy: 1.5 + (leakDelta / 30.0) * 2.0,
            r: 1.0 + (leakDelta / 30.0) * 1.4,
          });
        }
      }
      for (let i = leak.length - 1; i >= 0; i--) {
        const d = leak[i];
        d.y += d.vy * dtFrac;
        d.vy += 0.18 * dtFrac;
        if (d.y >= basinY) {
          leak.splice(i, 1);
        }
      }

      // Splash droplets (from spill or reset splash)
      for (let i = splashes.length - 1; i >= 0; i--) {
        const s = splashes[i];
        s.x += s.vx * dtFrac;
        s.y += s.vy * dtFrac;
        s.vy += 0.22 * dtFrac;
        s.life -= dtMs / 350.0;
        if (s.life <= 0) {
          splashes.splice(i, 1);
        }
      }
    }

    function addSplash(x, y, count = 12, dirX = 1) {
      for (let i = 0; i < count; i++) {
        splashes.push({
          x: x + (rng() - 0.5) * 6,
          y: y,
          vx: dirX * (1.8 + rng() * 3.5) + (rng() - 0.5) * 1.5,
          vy: -2.5 - rng() * 3.8,
          r: 1.2 + rng() * 1.6,
          life: 1.0,
        });
      }
    }

    return {
      inflow,
      leak,
      splashes,
      update,
      addSplash,
      reset,
    };
  }

  const core = {
    DEFAULTS,
    mulberry32,
    gaussian,
    lif_derivative,
    euler_step,
    fire,
    createState,
    step,
    theoryISI,
    theoryRate,
    theoryVoltage,
    simulateRun,
    createDroplets,
    S,
  };

  // ---- ui: only runs when a browser Atlas shell exists ----
  function mount(rootEl, ctx) {
    const figureContainer = ctx.figure || rootEl;
    const marginContainer = ctx.margin || document.getElementById('atlas-margin');

    const targetHeight = (typeof window !== 'undefined' && window.innerWidth < 560) ? '480px' : '520px';
    if (figureContainer && figureContainer.style) {
      figureContainer.style.setProperty('aspect-ratio', 'auto', 'important');
      figureContainer.style.setProperty('height', targetHeight, 'important');
      figureContainer.style.setProperty('min-height', targetHeight, 'important');
      figureContainer.style.setProperty('flex-shrink', '0', 'important');
    }

    // Simulation state
    const simParams = { ...DEFAULTS };
    const simState = createState(simParams);
    simState.params = simParams;
    const droplets = createDroplets(42);

    let theoryActive = false;
    let hoveredTerm = null;
    let currentChapter = 1;
    let predictAnswer = null;
    let predictSubmitted = false;

    // Direct manipulation drag states
    let draggingHandle = null; // 'faucet' | 'threshold' | 'caliper1' | 'caliper2' | 'fi_marker'
    let caliper1Time = 40.0;
    let caliper2Time = 62.2;

    // Visual animation states
    let tipAngle = 0; // Bucket tip angle in radians
    let spikePulse = null; // { x, y, progress }
    let splashRipple = 1.0; // Starts with a splash ripple for chapter 1 kick
    let refractoryLidAnim = 0; // Refractory lid clamp display timer
    let waterLevelAnim = simParams.v_rest + 10.0; // Start with +10 mV kick so student sees leak

    const traceHistory = []; // { t, v, spiked }
    const MAX_TRACE_MS = 160;

    function fastForwardTrace(durationMs) {
      const steps = Math.round(durationMs / 0.1);
      for (let s = 0; s < steps; s++) {
        const out = step(simState, 0.1);
        traceHistory.push({ t: out.t, v: out.v, spiked: out.spiked });
      }
      while (traceHistory.length > 0 && traceHistory[traceHistory.length - 1].t - traceHistory[0].t > MAX_TRACE_MS) {
        traceHistory.shift();
      }
    }

    // Precomputed empirical f-I points
    const fiPoints = [
      { I: 0.0, f: 0 },
      { I: 1.0, f: 0 },
      { I: 2.0, f: 0 },
      { I: 2.9, f: 0 },
      { I: 3.2, f: 12.5 },
      { I: 3.5, f: 23.9 },
      { I: 4.0, f: 33.5 },
      { I: 5.0, f: 45.0 },
      { I: 6.0, f: 57.8 },
      { I: 7.0, f: 69.5 },
      { I: 8.0, f: 79.2 },
      { I: 10.0, f: 98.4 },
      { I: 12.0, f: 116.2 },
    ];

    // ---- DOM: Mount Margin Components ----
    if (marginContainer) {
      marginContainer.innerHTML = `
        <div class="atlas-margin-section" id="n01-equation-section">
          <div class="atlas-margin-label">${S.equation_label}</div>
          <div class="atlas-equation-box" style="font-size: 13px; font-weight: 600;">
            ${S.equation_formula}
          </div>
          <div style="display: flex; flex-direction: column; gap: 4px; margin-top: 6px;">
            <div class="n01-term" data-term="tau" style="cursor: pointer; padding: 2px 4px; border-radius: 3px;" id="n01-term-tau">τ = 20.0 ms</div>
            <div class="n01-term" data-term="v" style="cursor: pointer; padding: 2px 4px; border-radius: 3px;" id="n01-term-v">v = -60.0 mV</div>
            <div class="n01-term" data-term="v_rest" style="cursor: pointer; padding: 2px 4px; border-radius: 3px;" id="n01-term-vrest">v_rest = -70.0 mV</div>
            <div class="n01-term" data-term="R" style="cursor: pointer; padding: 2px 4px; border-radius: 3px;" id="n01-term-R">R = 10.0 MΩ</div>
            <div class="n01-term" data-term="I" style="cursor: pointer; padding: 2px 4px; border-radius: 3px;" id="n01-term-I">I = 0.00 nA</div>
            <div class="n01-term" data-term="dvdt" style="cursor: pointer; padding: 2px 4px; border-radius: 3px;" id="n01-term-dvdt">dv/dt = -0.50 mV/ms</div>
          </div>
        </div>

        <div class="atlas-margin-section" id="n01-theory-section">
          <div class="atlas-margin-label">${S.theory_box_title}</div>
          <div style="font-family: var(--font-mono); font-size: 12px; display: flex; flex-direction: column; gap: 4px;" id="n01-theory-readout">
            <div id="n01-theory-isi">Theory ISI: -</div>
            <div id="n01-meas-isi">Measured ISI: -</div>
            <div id="n01-diff-isi">Difference: -</div>
          </div>
        </div>

        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.controls_title}</div>
          <div style="display: flex; flex-direction: column; gap: 10px;">
            <div>
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span>${S.faucet_current}</span>
                <span class="atlas-mono" id="n01-val-current">0.0 nA</span>
              </div>
              <input type="range" id="n01-slider-current" min="0" max="12" step="0.1" value="0" style="width: 100%; accent-color: var(--exc);">
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span>${S.threshold_slider}</span>
                <span class="atlas-mono" id="n01-val-th">-40.0 mV</span>
              </div>
              <input type="range" id="n01-slider-th" min="-55" max="-35" step="0.5" value="-40" style="width: 100%; accent-color: var(--volt);">
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
              <button class="atlas-btn" id="n01-btn-noise" style="font-size: 11px; min-height: 32px;">${S.btn_noise_off}</button>
              <button class="atlas-btn" id="n01-btn-theory" style="font-size: 11px; min-height: 32px;">${S.btn_theory_off}</button>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
              <button class="atlas-btn" id="n01-btn-kick" style="font-size: 11px; min-height: 32px;">${S.btn_kick}</button>
              <button class="atlas-btn" id="n01-btn-reset" style="font-size: 11px; min-height: 32px;">${S.btn_reset}</button>
            </div>
          </div>
        </div>

        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.glossary_title}</div>
          <div style="display: flex; flex-direction: column; gap: 6px; font-size: 11px; line-height: 1.4;">
            <div>• <b>v:</b> ${S.glossary_v}</div>
            <div>• <b>v_rest:</b> ${S.glossary_vrest}</div>
            <div>• <b>v_th:</b> ${S.glossary_vth}</div>
            <div>• <b>v_reset:</b> ${S.glossary_vreset}</div>
            <div>• <b>τ:</b> ${S.glossary_tau}</div>
            <div>• <b>R:</b> ${S.glossary_R}</div>
            <div>• <b>I:</b> ${S.glossary_I}</div>
            <div>• <b>t_ref:</b> ${S.glossary_tref}</div>
            <div>• <b>Rheobase:</b> ${S.glossary_rheobase}</div>
          </div>
        </div>

        <div style="font-size: 10px; color: var(--ink-muted); margin-top: auto; padding: 4px;">
          ${S.footer_source}
        </div>
      `;

      // Term hover events
      marginContainer.querySelectorAll('.n01-term').forEach((el) => {
        el.addEventListener('mouseenter', () => {
          hoveredTerm = el.getAttribute('data-term');
          el.style.backgroundColor = 'var(--paper-inset)';
        });
        el.addEventListener('mouseleave', () => {
          hoveredTerm = null;
          el.style.backgroundColor = 'transparent';
        });
      });

      // Slider & Button inputs
      const sliderCurrent = document.getElementById('n01-slider-current');
      const valCurrent = document.getElementById('n01-val-current');
      sliderCurrent.addEventListener('input', (e) => {
        simParams.current = parseFloat(e.target.value);
        valCurrent.textContent = `${simParams.current.toFixed(1)} nA`;
      });

      const sliderTh = document.getElementById('n01-slider-th');
      const valTh = document.getElementById('n01-val-th');
      sliderTh.addEventListener('input', (e) => {
        simParams.threshold = parseFloat(e.target.value);
        valTh.textContent = `${simParams.threshold.toFixed(1)} mV`;
      });

      const btnNoise = document.getElementById('n01-btn-noise');
      btnNoise.addEventListener('click', () => {
        simParams.noise_enabled = !simParams.noise_enabled;
        btnNoise.textContent = simParams.noise_enabled ? S.btn_noise_on : S.btn_noise_off;
        btnNoise.classList.toggle('active', simParams.noise_enabled);
      });

      const btnTheory = document.getElementById('n01-btn-theory');
      btnTheory.addEventListener('click', () => {
        theoryActive = !theoryActive;
        btnTheory.textContent = theoryActive ? S.btn_theory_on : S.btn_theory_off;
        btnTheory.classList.toggle('active', theoryActive);
      });

      const triggerKick = () => {
        simState.v = Math.min(simParams.threshold + 2.0, simState.v + 10.0);
        splashRipple = 1.0;
        const rect = canvas ? canvas.getBoundingClientRect() : null;
        const cssW = (rect && rect.width) ? rect.width : (typeof window !== 'undefined' && window.innerWidth < 560 ? 400 : 860);
        const cssH = (rect && rect.height) ? rect.height : 500;
        const L = getLayout(cssW, cssH);
        droplets.addSplash(L.bucketCenterX, L.bucketBottomY - 10, 10, 0);
      };

      const btnKick = document.getElementById('n01-btn-kick');
      btnKick.addEventListener('click', triggerKick);

      const btnReset = document.getElementById('n01-btn-reset');
      btnReset.addEventListener('click', () => {
        simState.v = simParams.v_rest;
        simState.refractory_time = 0;
        simState.spikeCount = 0;
        simState.lastSpikeTime = -Infinity;
        simState.isiList = [];
        traceHistory.length = 0;
        droplets.reset();
      });
    }

    // ---- DOM: Mount Chapter Banner in Figure Section ----
    const chapterBar = document.createElement('div');
    chapterBar.className = 'n01-chapter-bar';
    chapterBar.style.cssText =
      'background: var(--paper-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 12px; margin-bottom: 6px; display: flex; flex-direction: column; gap: 4px;';
    chapterBar.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px;">
        <div style="font-family: var(--font-serif); font-weight: 600; font-size: 14px;" id="n01-ch-title">${S.ch1_title}</div>
        <div style="display: flex; align-items: center; gap: 6px;">
          <div style="display: flex; gap: 2px;" id="n01-ch-dots">
            <span class="n01-dot" style="cursor: pointer; font-size: 13px; padding: 4px 5px; display: inline-flex; align-items: center; justify-content: center; min-width: 22px; min-height: 22px;">●</span>
            <span class="n01-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4; padding: 4px 5px; display: inline-flex; align-items: center; justify-content: center; min-width: 22px; min-height: 22px;">○</span>
            <span class="n01-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4; padding: 4px 5px; display: inline-flex; align-items: center; justify-content: center; min-width: 22px; min-height: 22px;">○</span>
            <span class="n01-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4; padding: 4px 5px; display: inline-flex; align-items: center; justify-content: center; min-width: 22px; min-height: 22px;">○</span>
            <span class="n01-dot" style="cursor: pointer; font-size: 13px; opacity: 0.4; padding: 4px 5px; display: inline-flex; align-items: center; justify-content: center; min-width: 22px; min-height: 22px;">○</span>
          </div>
          <button class="atlas-btn" id="n01-ch-prev" style="padding: 4px 10px; font-size: 11px; min-height: 28px;">${S.ch_prev}</button>
          <button class="atlas-btn" id="n01-ch-next" style="padding: 4px 10px; font-size: 11px; min-height: 28px;">${S.ch_next}</button>
        </div>
      </div>
      <div style="font-size: 12px; color: var(--ink-muted); line-height: 1.35;" id="n01-ch-desc">
        ${S.ch1_desc}
      </div>
      <div id="n01-predict-box" style="display: none; background: var(--paper-inset); border: 1px solid var(--border); border-radius: 3px; padding: 6px 8px; margin-top: 4px;">
        <div style="font-weight: 600; font-size: 11px; color: var(--volt); margin-bottom: 4px;">${S.predict_title}: ${S.predict_q}</div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
          <button class="atlas-btn n01-pred-btn" data-choice="1" style="font-size: 11px; padding: 6px 10px; min-height: 32px;">${S.predict_opt1}</button>
          <button class="atlas-btn n01-pred-btn" data-choice="2" style="font-size: 11px; padding: 6px 10px; min-height: 32px;">${S.predict_opt2}</button>
          <button class="atlas-btn n01-pred-btn" data-choice="3" style="font-size: 11px; padding: 6px 10px; min-height: 32px;">${S.predict_opt3}</button>
        </div>
        <div id="n01-predict-reveal" style="display: none; margin-top: 6px; font-size: 11px; line-height: 1.35;"></div>
      </div>
    `;

    if (figureContainer.parentElement) {
      figureContainer.parentElement.insertBefore(chapterBar, figureContainer);
    } else {
      figureContainer.appendChild(chapterBar);
    }

    function setChapter(ch) {
      currentChapter = Math.max(1, Math.min(5, ch));
      const titleEl = document.getElementById('n01-ch-title');
      const descEl = document.getElementById('n01-ch-desc');
      const predictBox = document.getElementById('n01-predict-box');
      const dots = document.querySelectorAll('.n01-dot');

      dots.forEach((d, idx) => {
        d.textContent = idx + 1 === currentChapter ? '●' : '○';
        d.style.opacity = idx + 1 === currentChapter ? '1.0' : '0.4';
      });

      const sliderCurrent = document.getElementById('n01-slider-current');
      const valCurrent = document.getElementById('n01-val-current');
      droplets.reset();

      if (currentChapter === 1) {
        titleEl.textContent = S.ch1_title;
        descEl.textContent = S.ch1_desc;
        predictBox.style.display = 'none';
        simParams.current = 0.0;
        simState.v = simParams.v_rest;
        traceHistory.length = 0;
        simState.t = 0;
        // Baseline at -70 mV for 60 ms
        for (let s = 0; s < 600; s++) {
          const out = step(simState, 0.1);
          traceHistory.push({ t: out.t, v: out.v, spiked: out.spiked });
        }
        // Scripted perturbation: kick +10 mV at t = 60 ms so leak is active immediately
        simState.v = simParams.v_rest + 10.0;
        splashRipple = 1.0;
        // Decay for 80 ms
        for (let s = 0; s < 800; s++) {
          const out = step(simState, 0.1);
          traceHistory.push({ t: out.t, v: out.v, spiked: out.spiked });
        }
      } else if (currentChapter === 2) {
        titleEl.textContent = S.ch2_title;
        descEl.textContent = S.ch2_desc;
        predictBox.style.display = 'none';
        simParams.current = 2.0;
        fastForwardTrace(150);
      } else if (currentChapter === 3) {
        titleEl.textContent = S.ch3_title;
        descEl.textContent = S.ch3_desc;
        predictBox.style.display = 'none';
        simParams.current = 3.5;
        fastForwardTrace(200);
        const T3 = theoryISI(3.5, simParams);
        caliper1Time = 30.0;
        caliper2Time = 30.0 + (Number.isFinite(T3) ? T3 : 42.0);
      } else if (currentChapter === 4) {
        titleEl.textContent = S.ch4_title;
        descEl.textContent = S.ch4_desc;
        predictBox.style.display = 'block';
        simParams.current = 6.0;
        fastForwardTrace(200);
        const T4 = theoryISI(6.0, simParams);
        caliper1Time = 30.0;
        caliper2Time = 30.0 + (Number.isFinite(T4) ? T4 : 17.5);
      } else if (currentChapter === 5) {
        titleEl.textContent = S.ch5_title;
        descEl.textContent = S.ch5_desc;
        predictBox.style.display = 'block';
        simParams.current = 5.0;
        theoryActive = true;
        fastForwardTrace(200);
        const T5 = theoryISI(5.0, simParams);
        caliper1Time = 30.0;
        caliper2Time = 30.0 + (Number.isFinite(T5) ? T5 : 22.2);
        const btnTh = document.getElementById('n01-btn-theory');
        if (btnTh) {
          btnTh.textContent = S.btn_theory_on;
          btnTh.classList.add('active');
        }
      }

      if (sliderCurrent && valCurrent) {
        sliderCurrent.value = String(simParams.current);
        valCurrent.textContent = `${simParams.current.toFixed(1)} nA`;
      }

      const prevBtn = document.getElementById('n01-ch-prev');
      const nextBtn = document.getElementById('n01-ch-next');
      if (prevBtn) prevBtn.disabled = currentChapter === 1;
      if (nextBtn) nextBtn.disabled = currentChapter === 5;
    }

    document.getElementById('n01-ch-prev').addEventListener('click', () => setChapter(currentChapter - 1));
    document.getElementById('n01-ch-next').addEventListener('click', () => setChapter(currentChapter + 1));
    document.querySelectorAll('.n01-dot').forEach((d, idx) => {
      d.addEventListener('click', () => setChapter(idx + 1));
    });

    // Predict buttons
    document.querySelectorAll('.n01-pred-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        predictAnswer = btn.getAttribute('data-choice');
        predictSubmitted = true;
        theoryActive = true;
        const btnTh = document.getElementById('n01-btn-theory');
        if (btnTh) {
          btnTh.textContent = S.btn_theory_on;
          btnTh.classList.add('active');
        }
        const revealEl = document.getElementById('n01-predict-reveal');
        revealEl.style.display = 'block';
        const isCorrect = predictAnswer === '2';
        revealEl.innerHTML = `
          <div style="font-weight: 600; color: ${isCorrect ? 'var(--exc)' : 'var(--inh)'};">
            ${isCorrect ? S.predict_reveal_correct : S.predict_reveal_wrong}
          </div>
          <div>${S.predict_reveal_body}</div>
        `;
      });
    });

    // Initialize Chapter 1 state
    setChapter(1);

    // ---- Canvas Setup ----
    const canvas = ctx.canvas();
    const c2d = canvas.getContext('2d');

    // Global keyboard listener for Theory & Kick
    window.addEventListener('atlas:theory', (e) => {
      theoryActive = e.detail.active;
      const btn = document.getElementById('n01-btn-theory');
      if (btn) {
        btn.textContent = theoryActive ? S.btn_theory_on : S.btn_theory_off;
        btn.classList.toggle('active', theoryActive);
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 't' || e.key === 'T') {
        theoryActive = !theoryActive;
        const btn = document.getElementById('n01-btn-theory');
        if (btn) {
          btn.textContent = theoryActive ? S.btn_theory_on : S.btn_theory_off;
          btn.classList.toggle('active', theoryActive);
        }
      } else if (e.key === 'k' || e.key === 'K') {
        triggerKick();
      } else if (e.key === 's' || e.key === 'S') {
        simState.v = simParams.threshold + 1.0;
        tipAngle = 0.38;
        spikePulse = { progress: 0 };
        splashRipple = 1.0;
        const rect = canvas.getBoundingClientRect();
        const L = getLayout(rect.width || 860, rect.height || 500);
        droplets.addSplash(L.bucketRight + 4, L.bucketTopY + 14, 18, 1);
      } else if (e.key >= '1' && e.key <= '5') {
        setChapter(parseInt(e.key, 10));
      }
    });

    function getLayout(cssW, cssH) {
      const isMobile = cssW < 560;
      if (!isMobile) {
        const leftColW = Math.round(cssW * 0.46);
        const bucketCenterX = Math.round(leftColW * 0.46);
        const bucketW = Math.min(270, Math.round(leftColW * 0.74));
        const bucketLeft = bucketCenterX - bucketW / 2;
        const bucketRight = bucketCenterX + bucketW / 2;
        const bucketTopY = 115;
        const bucketBottomY = 365;
        const bucketH = bucketBottomY - bucketTopY;
        const faucetX = bucketCenterX;
        const faucetY = 55;
        const dividerX = leftColW + 12;
        const traceLeft = dividerX + 32;
        const traceRight = cssW - 18;
        const traceTop = 32;
        const traceBottom = 225;
        const fiLeft = dividerX + 32;
        const fiRight = cssW - 18;
        const fiTop = 275;
        const fiBottom = cssH - 35;
        return {
          isMobile,
          leftColW,
          bucketCenterX,
          bucketW,
          bucketLeft,
          bucketRight,
          bucketTopY,
          bucketBottomY,
          bucketH,
          faucetX,
          faucetY,
          dividerX,
          traceLeft,
          traceRight,
          traceTop,
          traceBottom,
          traceW: traceRight - traceLeft,
          traceH: traceBottom - traceTop,
          fiLeft,
          fiRight,
          fiTop,
          fiBottom,
          fiW: fiRight - fiLeft,
          fiH: fiBottom - fiTop,
        };
      } else {
        const bucketCenterX = Math.round(cssW * 0.50);
        const bucketW = Math.min(230, cssW - 70);
        const bucketLeft = bucketCenterX - bucketW / 2;
        const bucketRight = bucketCenterX + bucketW / 2;
        const bucketTopY = 54;
        const bucketBottomY = 146;
        const bucketH = bucketBottomY - bucketTopY;
        const faucetX = bucketCenterX;
        const faucetY = 36;
        const dividerX = -999;
        const traceLeft = 36;
        const traceRight = cssW - 16;
        const traceTop = 196;
        const traceBottom = cssH - 28;
        const fiLeft = 36;
        const fiRight = cssW - 16;
        const fiTop = 196;
        const fiBottom = cssH - 30;
        return {
          isMobile,
          leftColW: cssW,
          bucketCenterX,
          bucketW,
          bucketLeft,
          bucketRight,
          bucketTopY,
          bucketBottomY,
          bucketH,
          faucetX,
          faucetY,
          dividerX,
          traceLeft,
          traceRight,
          traceTop,
          traceBottom,
          traceW: traceRight - traceLeft,
          traceH: traceBottom - traceTop,
          fiLeft,
          fiRight,
          fiTop,
          fiBottom,
          fiW: fiRight - fiLeft,
          fiH: fiBottom - fiTop,
        };
      }
    }

    // Coordinates conversion
    function toCanvasCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX ?? (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
      const clientY = e.clientY ?? (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
      const scaleX = (canvas.width / (window.devicePixelRatio || 1)) / (rect.width || 1);
      const scaleY = (canvas.height / (window.devicePixelRatio || 1)) / (rect.height || 1);
      return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
      };
    }

    // Direct manipulation pointer handlers
    canvas.addEventListener('pointerdown', (e) => {
      const pos = toCanvasCoords(e);
      const cssW = canvas.width / (window.devicePixelRatio || 1);
      const cssH = canvas.height / (window.devicePixelRatio || 1);
      const L = getLayout(cssW, cssH);

      // 1. Faucet handle
      const faucetHitR = L.isMobile ? 32 : 26;
      if (Math.hypot(pos.x - L.faucetX, pos.y - (L.faucetY - 22)) < faucetHitR) {
        draggingHandle = 'faucet';
        canvas.setPointerCapture(e.pointerId);
        return;
      }

      // 2. Threshold chevron handle at bucket right edge
      const thFrac = (simParams.threshold - -80.0) / 45.0;
      const thY = L.bucketBottomY - thFrac * L.bucketH;
      const thHitX = L.isMobile ? 32 : 24;
      const thHitY = L.isMobile ? 22 : 18;
      if (Math.abs(pos.x - (L.bucketRight + 8)) < thHitX && Math.abs(pos.y - thY) < thHitY) {
        draggingHandle = 'threshold';
        canvas.setPointerCapture(e.pointerId);
        return;
      }

      // 3. Calipers on oscilloscope trace (only when spiking is active)
      const tThDown = theoryISI(simParams.current, simParams);
      const showCalipersDown = Number.isFinite(tThDown) && tThDown > 0;
      const caliperHit = L.isMobile ? 22 : 14;
      if (showCalipersDown && (!L.isMobile || currentChapter !== 5) && pos.x >= L.traceLeft && pos.x <= L.traceRight && pos.y >= L.traceTop - 12 && pos.y <= L.traceBottom + 12) {
        const c1X = L.traceLeft + (caliper1Time / MAX_TRACE_MS) * L.traceW;
        const c2X = L.traceLeft + (caliper2Time / MAX_TRACE_MS) * L.traceW;
        if (Math.abs(pos.x - c1X) < caliperHit) {
          draggingHandle = 'caliper1';
          canvas.setPointerCapture(e.pointerId);
          return;
        }
        if (Math.abs(pos.x - c2X) < caliperHit) {
          draggingHandle = 'caliper2';
          canvas.setPointerCapture(e.pointerId);
          return;
        }
      }

      // 4. f-I plot direct manipulation (click or drag to set current)
      if ((!L.isMobile || currentChapter === 5) && pos.x >= L.fiLeft - 8 && pos.x <= L.fiRight + 8 && pos.y >= L.fiTop - 10 && pos.y <= L.fiBottom + 15) {
        draggingHandle = 'fi_marker';
        canvas.setPointerCapture(e.pointerId);
        const newI = Math.max(0, Math.min(12.0, ((pos.x - L.fiLeft) / L.fiW) * 12.0));
        simParams.current = Math.round(newI * 10) / 10;
        const sliderCurrent = document.getElementById('n01-slider-current');
        const valCurrent = document.getElementById('n01-val-current');
        if (sliderCurrent && valCurrent) {
          sliderCurrent.value = String(simParams.current);
          valCurrent.textContent = `${simParams.current.toFixed(1)} nA`;
        }
        return;
      }

      // 5. Click inside bucket: Kick membrane!
      if (pos.x >= L.bucketLeft && pos.x <= L.bucketRight && pos.y >= L.bucketTopY && pos.y <= L.bucketBottomY) {
        simState.v = Math.min(simParams.threshold + 2.0, simState.v + 10.0);
        splashRipple = 1.0;
        droplets.addSplash(pos.x, pos.y, 12, 0);
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      const pos = toCanvasCoords(e);
      const cssW = canvas.width / (window.devicePixelRatio || 1);
      const cssH = canvas.height / (window.devicePixelRatio || 1);
      const L = getLayout(cssW, cssH);
      const thFrac = (simParams.threshold - -80.0) / 45.0;
      const thY = L.bucketBottomY - thFrac * L.bucketH;
      const c1X = L.traceLeft + (caliper1Time / MAX_TRACE_MS) * L.traceW;
      const c2X = L.traceLeft + (caliper2Time / MAX_TRACE_MS) * L.traceW;
      const tThMove = theoryISI(simParams.current, simParams);
      const showCalipersMove = Number.isFinite(tThMove) && tThMove > 0;

      if (!draggingHandle) {
        const faucetHitR = L.isMobile ? 32 : 26;
        const thHitX = L.isMobile ? 32 : 24;
        const thHitY = L.isMobile ? 22 : 18;
        const caliperHit = L.isMobile ? 22 : 14;
        if (Math.hypot(pos.x - L.faucetX, pos.y - (L.faucetY - 22)) < faucetHitR) {
          canvas.style.cursor = 'grab';
        } else if (Math.abs(pos.x - (L.bucketRight + 8)) < thHitX && Math.abs(pos.y - thY) < thHitY) {
          canvas.style.cursor = 'ns-resize';
        } else if (showCalipersMove && (!L.isMobile || currentChapter !== 5) && (Math.abs(pos.x - c1X) < caliperHit || Math.abs(pos.x - c2X) < caliperHit)) {
          canvas.style.cursor = 'ew-resize';
        } else if ((!L.isMobile || currentChapter === 5) && pos.x >= L.fiLeft && pos.x <= L.fiRight && pos.y >= L.fiTop && pos.y <= L.fiBottom) {
          canvas.style.cursor = 'pointer';
        } else if (pos.x >= L.bucketLeft && pos.x <= L.bucketRight && pos.y >= L.bucketTopY && pos.y <= L.bucketBottomY) {
          canvas.style.cursor = 'pointer';
        } else {
          canvas.style.cursor = 'default';
        }
        return;
      }

      if (draggingHandle === 'faucet') {
        canvas.style.cursor = 'grabbing';
        const newI = Math.max(0, Math.min(12.0, (pos.x - (L.faucetX - 60)) / 12.0));
        simParams.current = Math.round(newI * 10) / 10;
        const sliderCurrent = document.getElementById('n01-slider-current');
        const valCurrent = document.getElementById('n01-val-current');
        if (sliderCurrent && valCurrent) {
          sliderCurrent.value = String(simParams.current);
          valCurrent.textContent = `${simParams.current.toFixed(1)} nA`;
        }
      } else if (draggingHandle === 'threshold') {
        canvas.style.cursor = 'ns-resize';
        const frac = (L.bucketBottomY - pos.y) / L.bucketH;
        const newTh = Math.max(-55.0, Math.min(-35.0, -80.0 + frac * 45.0));
        simParams.threshold = Math.round(newTh * 2) / 2;
        const sliderTh = document.getElementById('n01-slider-th');
        const valTh = document.getElementById('n01-val-th');
        if (sliderTh && valTh) {
          sliderTh.value = String(simParams.threshold);
          valTh.textContent = `${simParams.threshold.toFixed(1)} mV`;
        }
      } else if (draggingHandle === 'caliper1') {
        canvas.style.cursor = 'ew-resize';
        const tFrac = Math.max(0, Math.min(1, (pos.x - L.traceLeft) / L.traceW));
        caliper1Time = tFrac * MAX_TRACE_MS;
      } else if (draggingHandle === 'caliper2') {
        canvas.style.cursor = 'ew-resize';
        const tFrac = Math.max(0, Math.min(1, (pos.x - L.traceLeft) / L.traceW));
        caliper2Time = tFrac * MAX_TRACE_MS;
      } else if (draggingHandle === 'fi_marker') {
        canvas.style.cursor = 'grabbing';
        const newI = Math.max(0, Math.min(12.0, ((pos.x - L.fiLeft) / L.fiW) * 12.0));
        simParams.current = Math.round(newI * 10) / 10;
        const sliderCurrent = document.getElementById('n01-slider-current');
        const valCurrent = document.getElementById('n01-val-current');
        if (sliderCurrent && valCurrent) {
          sliderCurrent.value = String(simParams.current);
          valCurrent.textContent = `${simParams.current.toFixed(1)} nA`;
        }
      }
    });

    const stopDragging = (e) => {
      if (draggingHandle) {
        try {
          canvas.releasePointerCapture(e.pointerId);
        } catch {}
        draggingHandle = null;
        canvas.style.cursor = 'default';
      }
    };
    canvas.addEventListener('pointerup', stopDragging);
    canvas.addEventListener('pointercancel', stopDragging);

    // ---- 60 FPS Simulation & Draw Loop ----
    let lastTimestamp = 0;
    let simSubstepAccumulator = 0;
    let wavePhase = 0;

    function renderFrame(now) {
      if (!now) now = performance.now();
      const dtReal = lastTimestamp === 0 ? 16 : Math.min(64, now - lastTimestamp);
      if (dtReal < 2) return;
      lastTimestamp = now;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const cssW = Math.round(rect.width || 860);
      const cssH = Math.round(rect.height || 520);
      const L = getLayout(cssW, cssH);

      // Advance simulation
      simSubstepAccumulator += dtReal;
      const stepsToRun = Math.min(400, Math.floor(simSubstepAccumulator / 0.1));
      simSubstepAccumulator -= stepsToRun * 0.1;

      for (let s = 0; s < stepsToRun; s++) {
        const stepOut = step(simState, 0.1);

        traceHistory.push({
          t: stepOut.t,
          v: stepOut.v,
          spiked: stepOut.spiked,
        });

        if (stepOut.spiked) {
          tipAngle = 0.22; // radians tilt on threshold crossing
          spikePulse = { progress: 0 };
          splashRipple = 1.0;
          refractoryLidAnim = 1.0;
          droplets.addSplash(L.bucketRight + 4, L.bucketTopY + 14, 16, 1); // Water spill cascade from rim
          droplets.addSplash(L.bucketCenterX, L.bucketBottomY - 10, 10, 0); // Reset splash inside bucket
        }
      }

      // Maintain traceHistory length
      while (traceHistory.length > 0 && traceHistory[traceHistory.length - 1].t - traceHistory[0].t > MAX_TRACE_MS) {
        traceHistory.shift();
      }

      // Update visual animations
      wavePhase += 0.08 + (simParams.current / 6.0) * 0.08;
      tipAngle *= 0.90; // smooth tilt recovery
      splashRipple *= 0.94; // ripple decay
      refractoryLidAnim *= 0.94; // lid clamp hold animation decay
      if (spikePulse) {
        spikePulse.progress += 0.04;
        if (spikePulse.progress >= 1.0) spikePulse = null;
      }

      // Smooth water level animation
      waterLevelAnim += (simState.v - waterLevelAnim) * 0.28;

      // Update Margin Readouts
      const deriv = lif_derivative(simState.v, simParams.current, simParams);
      const termTau = document.getElementById('n01-term-tau');
      const termV = document.getElementById('n01-term-v');
      const termI = document.getElementById('n01-term-I');
      const termDvdt = document.getElementById('n01-term-dvdt');
      if (termTau) termTau.textContent = `τ = ${simParams.tau.toFixed(1)} ms`;
      if (termV) termV.textContent = `v = ${simState.v.toFixed(1)} mV`;
      if (termI) termI.textContent = `I = ${simParams.current.toFixed(2)} nA`;
      if (termDvdt) termDvdt.textContent = `dv/dt = ${(deriv / simParams.tau).toFixed(2)} mV/ms`;

      const tTheory = theoryISI(simParams.current, simParams);
      const fTheory = theoryRate(simParams.current, simParams);
      const recentISIs = simState.isiList.slice(-5);
      const measISI = recentISIs.length > 0 ? recentISIs[recentISIs.length - 1] : null;
      const measRate = measISI && measISI > 0 ? 1000.0 / measISI : 0;

      const isiTheoryEl = document.getElementById('n01-theory-isi');
      const isiMeasEl = document.getElementById('n01-meas-isi');
      const diffEl = document.getElementById('n01-diff-isi');

      if (isiTheoryEl && isiMeasEl && diffEl) {
        if (Number.isFinite(tTheory) && tTheory > 0) {
          isiTheoryEl.textContent = `Theory ISI: ${tTheory.toFixed(2)} ms (${fTheory.toFixed(1)} Hz)`;
          if (measISI !== null) {
            const diff = measISI - tTheory;
            const diffPct = (Math.abs(diff) / tTheory) * 100.0;
            isiMeasEl.textContent = `Measured ISI: ${measISI.toFixed(2)} ms (${measRate.toFixed(1)} Hz)`;
            diffEl.textContent = `Diff: ${diff >= 0 ? '+' : ''}${diff.toFixed(2)} ms (${diffPct.toFixed(1)}%)`;
            diffEl.style.color = diffPct < 3.0 ? 'var(--exc)' : 'var(--volt)';
          } else {
            isiMeasEl.textContent = 'Measured ISI: collecting...';
            diffEl.textContent = 'Diff: -';
          }
        } else {
          isiTheoryEl.textContent = 'Theory ISI: ∞ (Subthreshold)';
          isiMeasEl.textContent = 'Measured ISI: ∞ (No spikes)';
          diffEl.textContent = 'Diff: 0.0 ms (Exact)';
          diffEl.style.color = 'var(--ink-muted)';
        }
      }

      // ---- Drawing onto Canvas ----
      if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
      }
      c2d.resetTransform();
      c2d.scale(dpr, dpr);

      const colors = ctx.colors();
      c2d.clearRect(0, 0, cssW, cssH);

      // Background paper card
      c2d.fillStyle = colors.paper;
      c2d.fillRect(0, 0, cssW, cssH);

      // Divider line (desktop only)
      if (!L.isMobile) {
        c2d.strokeStyle = colors.border;
        c2d.lineWidth = 1;
        c2d.beginPath();
        c2d.moveTo(L.dividerX, 16);
        c2d.lineTo(L.dividerX, cssH - 16);
        c2d.stroke();
      }

      // =======================================================
      // SECTION 1: THE HERO BUCKET (LEFT PANEL - >= 45% WIDTH)
      // =======================================================
      const bucketCenterX = L.bucketCenterX;
      const bucketW = L.bucketW;
      const bucketLeft = L.bucketLeft;
      const bucketRight = L.bucketRight;
      const bucketTopY = L.bucketTopY;
      const bucketBottomY = L.bucketBottomY;
      const bucketH = L.bucketH;

      // Overhead Faucet
      const faucetX = L.faucetX;
      const faucetY = L.faucetY;
      const flowRateFrac = Math.min(1.0, simParams.current / 6.0);

      c2d.save();
      // Faucet glow if hovered
      if (hoveredTerm === 'I' || hoveredTerm === 'R') {
        c2d.fillStyle = 'rgba(15, 118, 110, 0.18)';
        c2d.beginPath();
        c2d.arc(faucetX, faucetY, 36, 0, Math.PI * 2);
        c2d.fill();
      }

      // Faucet pipe from top left
      c2d.strokeStyle = colors.ink;
      c2d.lineWidth = 3;
      c2d.beginPath();
      c2d.moveTo(bucketLeft - 10, faucetY - 14);
      c2d.lineTo(faucetX, faucetY - 14);
      c2d.lineTo(faucetX, faucetY);
      c2d.stroke();

      // Faucet nozzle
      c2d.fillStyle = colors.ink;
      c2d.fillRect(faucetX - 8, faucetY, 16, 7);

      // Draggable valve handle
      c2d.strokeStyle = colors.exc;
      c2d.lineWidth = 3.5;
      c2d.beginPath();
      c2d.moveTo(faucetX - 20, faucetY - 22);
      c2d.lineTo(faucetX + 20, faucetY - 22);
      c2d.stroke();
      c2d.fillStyle = colors.exc;
      c2d.beginPath();
      c2d.arc(faucetX, faucetY - 22, 6, 0, Math.PI * 2);
      c2d.fill();

      // Faucet inflow label
      c2d.font = '600 11px var(--font-mono)';
      c2d.fillStyle = colors.exc;
      c2d.textAlign = 'center';
      const inflowVolt = (simParams.resistance * simParams.current).toFixed(1);
      c2d.fillText(`Faucet: I = ${simParams.current.toFixed(1)} nA (R·I = ${inflowVolt} mV)`, faucetX, faucetY - (L.isMobile ? 24 : 30));

      // Water stream from faucet nozzle into bucket
      const streamW = simParams.current > 0.05 ? 1.2 + flowRateFrac * 5.5 : 0;
      if (streamW > 0) {
        c2d.fillStyle = colors.exc;
        c2d.fillRect(faucetX - streamW / 2, faucetY + 7, streamW, bucketTopY - (faucetY + 7) + 12);
      }
      c2d.restore();

      // Bucket Tipping Transformation (Spill on Spike)
      c2d.save();
      c2d.translate(bucketCenterX, bucketBottomY);
      c2d.rotate(tipAngle);
      c2d.translate(-bucketCenterX, -bucketBottomY);

      // Highlight halo if term hovered
      if (hoveredTerm === 'v' || hoveredTerm === 'tau') {
        c2d.fillStyle = 'rgba(180, 83, 9, 0.14)';
        c2d.fillRect(bucketLeft - 10, bucketTopY - 8, bucketW + 20, bucketH + 16);
      }

      // Water Level calculation: v in [-80, -35] mV
      const vFrac = Math.max(0, Math.min(1.0, (waterLevelAnim - -80.0) / 45.0));
      const waterTopY = bucketBottomY - vFrac * bucketH;

      // Update droplets simulation
      const leakHoleX = bucketRight - 28;
      const basinY = bucketBottomY + (L.isMobile ? 12 : 36);
      droplets.update(
        dtReal,
        simParams.current,
        simState.v,
        simParams.v_rest,
        faucetX,
        faucetY,
        waterTopY,
        leakHoleX,
        bucketBottomY,
        basinY
      );

      // Draw water body with gentle animated wave surface
      const waveAmp = 0.8 + flowRateFrac * 2.2 + splashRipple * 5.0;
      c2d.fillStyle = 'rgba(180, 83, 9, 0.42)';
      c2d.beginPath();
      c2d.moveTo(bucketLeft + 2, bucketBottomY);
      c2d.lineTo(bucketLeft + 2, waterTopY);

      // Multi-sine wave surface
      const waveSegments = 32;
      const segW = (bucketW - 4) / waveSegments;
      for (let i = 0; i <= waveSegments; i++) {
        const wx = bucketLeft + 2 + i * segW;
        const wy =
          waterTopY +
          Math.sin(i * 0.35 + wavePhase) * waveAmp +
          Math.cos(i * 0.65 - wavePhase * 1.2) * (waveAmp * 0.4);
        c2d.lineTo(wx, wy);
      }
      c2d.lineTo(bucketRight - 2, bucketBottomY);
      c2d.closePath();
      c2d.fill();

      // Top water surface meniscus line
      c2d.strokeStyle = colors.volt;
      c2d.lineWidth = 2.2;
      c2d.beginPath();
      c2d.moveTo(bucketLeft + 2, waterTopY);
      for (let i = 0; i <= waveSegments; i++) {
        const wx = bucketLeft + 2 + i * segW;
        const wy =
          waterTopY +
          Math.sin(i * 0.35 + wavePhase) * waveAmp +
          Math.cos(i * 0.65 - wavePhase * 1.2) * (waveAmp * 0.4);
        c2d.lineTo(wx, wy);
      }
      c2d.stroke();

      // Spill cascade over the right rim during tipping
      if (tipAngle > 0.03) {
        c2d.fillStyle = 'rgba(180, 83, 9, 0.75)';
        c2d.beginPath();
        c2d.moveTo(bucketRight - 6, bucketTopY + 2);
        c2d.quadraticCurveTo(bucketRight + 24, bucketTopY + 12, bucketRight + 32, bucketTopY + 46);
        c2d.quadraticCurveTo(bucketRight + 16, bucketTopY + 28, bucketRight - 2, bucketTopY + 10);
        c2d.closePath();
        c2d.fill();

        // Spray droplets
        c2d.fillStyle = colors.volt;
        c2d.beginPath();
        c2d.arc(bucketRight + 26, bucketTopY + 32, 2.8, 0, Math.PI * 2);
        c2d.arc(bucketRight + 34, bucketTopY + 48, 2.2, 0, Math.PI * 2);
        c2d.arc(bucketRight + 20, bucketTopY + 22, 2.5, 0, Math.PI * 2);
        c2d.fill();
      }

      // Draw Inflow Droplets
      c2d.fillStyle = colors.exc;
      for (const d of droplets.inflow) {
        c2d.beginPath();
        c2d.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        c2d.fill();
      }

      // Thick Glass Beaker Walls
      c2d.strokeStyle = colors.ink;
      c2d.lineWidth = 2.5;
      c2d.beginPath();
      c2d.moveTo(bucketLeft, bucketTopY);
      c2d.lineTo(bucketLeft, bucketBottomY);
      c2d.lineTo(bucketRight, bucketBottomY);
      c2d.lineTo(bucketRight, bucketTopY);
      c2d.stroke();

      // Graduation ticks and labels on left wall
      c2d.font = '500 10px var(--font-mono)';
      c2d.textAlign = 'right';
      c2d.textBaseline = 'middle';
      const graduationTicks = L.isMobile ? [-70, -55, -40] : [-75, -70, -65, -60, -55, -50, -45, -40, -35];
      for (const tVal of graduationTicks) {
        const ty = bucketBottomY - ((tVal - -80.0) / 45.0) * bucketH;
        c2d.strokeStyle = colors.inkMuted;
        c2d.lineWidth = 1;
        c2d.beginPath();
        c2d.moveTo(bucketLeft, ty);
        c2d.lineTo(bucketLeft + 8, ty);
        c2d.stroke();
        c2d.fillStyle = colors.inkMuted;
        c2d.fillText(`${tVal}`, bucketLeft - 5, ty);
      }

      // Resting potential reference line (-70 mV)
      const restY = bucketBottomY - ((-70.0 - -80.0) / 45.0) * bucketH;
      c2d.strokeStyle = colors.inkMuted;
      c2d.setLineDash([3, 3]);
      c2d.beginPath();
      c2d.moveTo(bucketLeft, restY);
      c2d.lineTo(bucketRight, restY);
      c2d.stroke();
      c2d.setLineDash([]);
      c2d.fillStyle = colors.inkMuted;
      c2d.textAlign = 'left';
      c2d.fillText(S.rest_mark, bucketRight + 6, restY + (L.isMobile ? -2 : 9));

      // Reset potential reference line (-75 mV)
      const resetY = bucketBottomY - ((-75.0 - -80.0) / 45.0) * bucketH;
      c2d.strokeStyle = colors.inkMuted;
      c2d.setLineDash([2, 3]);
      c2d.beginPath();
      c2d.moveTo(bucketLeft, resetY);
      c2d.lineTo(bucketRight, resetY);
      c2d.stroke();
      c2d.setLineDash([]);
      c2d.fillStyle = colors.inkMuted;
      c2d.textAlign = 'left';
      c2d.fillText(S.reset_mark, bucketRight + 6, resetY + (L.isMobile ? 12 : 9));

      // Threshold line and draggable chevron handle (-40 mV)
      const thFracLive = (simParams.threshold - -80.0) / 45.0;
      const thY = bucketBottomY - thFracLive * bucketH;
      c2d.strokeStyle = colors.volt;
      c2d.lineWidth = 2.0;
      c2d.beginPath();
      c2d.moveTo(bucketLeft, thY);
      c2d.lineTo(bucketRight, thY);
      c2d.stroke();

      // Threshold draggable handle chevron
      c2d.fillStyle = colors.volt;
      c2d.beginPath();
      c2d.moveTo(bucketRight + 2, thY);
      c2d.lineTo(bucketRight + 12, thY - 7);
      c2d.lineTo(bucketRight + 12, thY + 7);
      c2d.closePath();
      c2d.fill();
      c2d.font = '600 11px var(--font-mono)';
      c2d.fillText(`v_th = ${simParams.threshold.toFixed(0)}`, bucketRight + 16, thY);

      // Refractory Lid (clamping during t_ref) - inside tipping transform!
      if (refractoryLidAnim > 0.05 || simState.refractory_time > 1e-9) {
        c2d.fillStyle = colors.ink;
        c2d.fillRect(bucketLeft - 6, bucketTopY - 6, bucketW + 12, 10);
        c2d.font = '600 11px var(--font-mono)';
        c2d.fillStyle = colors.volt;
        c2d.textAlign = 'center';
        c2d.fillText(`Refractory Clamped (${simParams.t_ref.toFixed(1)} ms hold)`, bucketCenterX, bucketTopY - 14);
      }

      // Leak Orifice & Outflow Stream
      const leakDelta = Math.max(0, simState.v - simParams.v_rest);
      const leakStreamW = leakDelta > 0.05 ? 1.0 + (leakDelta / 30.0) * 4.2 : 0;
      if (leakStreamW > 0) {
        c2d.fillStyle = colors.inh;
        c2d.fillRect(leakHoleX - leakStreamW / 2, bucketBottomY, leakStreamW, L.isMobile ? 12 : 26);
      }

      // Draw Leak Droplets
      c2d.fillStyle = colors.inh;
      for (const d of droplets.leak) {
        c2d.beginPath();
        c2d.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        c2d.fill();
      }

      // Draw Splash Droplets
      for (const s of droplets.splashes) {
        c2d.fillStyle = colors.volt;
        c2d.beginPath();
        c2d.arc(s.x, s.y, s.r * s.life, 0, Math.PI * 2);
        c2d.fill();
      }

      c2d.restore(); // End bucket tipping transform

      const basinOffset = L.isMobile ? 8 : 24;
      const leakTextOffset = L.isMobile ? 22 : 52;
      const hintOffset = L.isMobile ? 36 : 68;

      // Catch Basin below bucket for passive leak
      c2d.fillStyle = colors.paperCard;
      c2d.strokeStyle = colors.border;
      c2d.lineWidth = 1.2;
      c2d.strokeRect(bucketLeft + (L.isMobile ? 12 : 25), bucketBottomY + basinOffset, bucketW - (L.isMobile ? 24 : 40), L.isMobile ? 10 : 14);

      // Leak Rate Readout Label
      c2d.font = (L.isMobile ? '500 9px' : '500 11px') + ' var(--font-mono)';
      c2d.fillStyle = colors.inh;
      c2d.textAlign = 'center';
      c2d.fillText(`Leak: (v - v_rest) = ${leakDelta.toFixed(1)} mV`, bucketCenterX, bucketBottomY + leakTextOffset);

      // Bucket click hint (desktop only)
      if (!L.isMobile) {
        c2d.font = '500 10px var(--font-sans)';
        c2d.fillStyle = colors.inkMuted;
        c2d.fillText(S.bucket_hint, bucketCenterX, bucketBottomY + hintOffset);
      }

      // Spike Pulse Burst Shooting from bucket rim to trace
      if (spikePulse) {
        const pulseStartX = bucketRight;
        const pulseEndX = (!L.isMobile ? L.dividerX + 30 : bucketRight + 40);
        const px = pulseStartX + spikePulse.progress * (pulseEndX - pulseStartX);
        const py = thY - spikePulse.progress * (L.isMobile ? 8 : 15);
        c2d.fillStyle = colors.volt;
        c2d.beginPath();
        c2d.arc(px, py, 7 * (1 - spikePulse.progress * 0.3), 0, Math.PI * 2);
        c2d.fill();
        c2d.font = '700 11px var(--font-sans)';
        c2d.fillText(S.spike_pulse, px, py - 10);
      }

      // =======================================================
      // SECTION 2: SCROLLING VOLTAGE TRACE v(t)
      // =======================================================
      if (!L.isMobile || currentChapter !== 5) {
        const traceLeft = L.traceLeft;
        const traceRight = L.traceRight;
        const traceTop = L.traceTop;
        const traceBottom = L.traceBottom;
        const traceW = L.traceW;
        const traceH = L.traceH;

        const vMin = -80.0;
        const vMax = 25.0;
        const vToY = (v) => traceBottom - ((v - vMin) / (vMax - vMin)) * traceH;

        // Panel Header & Live Voltage Badge
        c2d.font = '600 11px var(--font-sans)';
        c2d.fillStyle = colors.ink;
        c2d.textAlign = 'left';
        c2d.fillText('Membrane Voltage Trace v(t)', traceLeft, traceTop - 10);

        c2d.font = '600 11px var(--font-mono)';
        c2d.fillStyle = colors.volt;
        c2d.textAlign = 'right';
        c2d.fillText(`v = ${simState.v.toFixed(1)} mV`, traceRight, traceTop - 10);

        // Horizontal Voltage Reference Grids
        const gridVs = L.isMobile ? [-70, -40, 20] : [-75, -70, -40, 0, 20];
        for (const gv of gridVs) {
          const gy = vToY(gv);
          c2d.strokeStyle = colors.border;
          c2d.lineWidth = 0.8;
          c2d.beginPath();
          c2d.moveTo(traceLeft, gy);
          c2d.lineTo(traceRight, gy);
          c2d.stroke();

          c2d.font = '500 9px var(--font-mono)';
          c2d.fillStyle = colors.inkMuted;
          c2d.textAlign = 'right';
          const labelY = gv === -75 ? gy + 4 : gv === -70 ? gy - 2 : gy + 3;
          c2d.fillText(`${gv}${gv === 20 ? ' mV' : ''}`, traceLeft - 4, labelY);
        }

        // Threshold and Rest Reference Dashes
        c2d.strokeStyle = colors.volt;
        c2d.setLineDash([3, 3]);
        const traceThY = vToY(simParams.threshold);
        c2d.beginPath();
        c2d.moveTo(traceLeft, traceThY);
        c2d.lineTo(traceRight, traceThY);
        c2d.stroke();
        c2d.setLineDash([]);

        // Draw Voltage Trace Polyline
        if (traceHistory.length > 1) {
          const latestT = traceHistory[traceHistory.length - 1].t;
          const tToX = (t) => traceRight - ((latestT - t) / MAX_TRACE_MS) * traceW;

          c2d.save();
          c2d.beginPath();
          c2d.rect(traceLeft, traceTop - 4, traceW, traceH + 8);
          c2d.clip();

          // Theory Ghost Exponential Charging Curve on Trace
          if (theoryActive) {
            c2d.strokeStyle = colors.theory;
            c2d.lineWidth = 1.6;
            c2d.setLineDash([4, 3]);
            c2d.beginPath();
            let started = false;
            let segmentT0 = traceHistory[0].t;
            for (let i = 0; i < traceHistory.length; i++) {
              const h = traceHistory[i];
              if (h.spiked) segmentT0 = h.t;
              const dtFromReset = h.t - segmentT0;
              const vTheory = core.theoryVoltage(dtFromReset, simParams.current, simParams, simParams.v_reset);
              const gx = tToX(h.t);
              const gy = vToY(vTheory);
              if (!started) {
                c2d.moveTo(gx, gy);
                started = true;
              } else {
                c2d.lineTo(gx, gy);
              }
            }
            c2d.stroke();
            c2d.setLineDash([]);
          }

          // Continuous Simulated Voltage Curve with Action Potential Spikes to +20 mV
          c2d.strokeStyle = colors.volt;
          c2d.lineWidth = 1.8;
          c2d.beginPath();
          let first = true;
          for (let i = 0; i < traceHistory.length; i++) {
            const h = traceHistory[i];
            const x = tToX(h.t);
            const y = vToY(h.spiked ? 20.0 : h.v);

            if (first) {
              c2d.moveTo(x, y);
              first = false;
            } else {
              if (h.spiked) {
                // Action potential upstroke to +20 mV, then reset downstroke
                c2d.lineTo(x, vToY(20.0));
                c2d.lineTo(x, vToY(simParams.v_reset));
              } else {
                c2d.lineTo(x, y);
              }
            }
          }
          c2d.stroke();

          // Spike peak markers (amber dots at +20 mV)
          c2d.fillStyle = colors.volt;
          for (let i = 0; i < traceHistory.length; i++) {
            const h = traceHistory[i];
            if (h.spiked) {
              const sx = tToX(h.t);
              c2d.beginPath();
              c2d.arc(sx, vToY(20.0), 3.0, 0, Math.PI * 2);
              c2d.fill();
              c2d.strokeStyle = colors.paper;
              c2d.lineWidth = 1.0;
              c2d.stroke();
            }
          }

          c2d.restore();
        }

        // Draggable ISI Calipers on Trace (only shown when neuron is actively spiking)
        if (Number.isFinite(tTheory) && tTheory > 0) {
          const c1X = traceLeft + (caliper1Time / MAX_TRACE_MS) * traceW;
          const c2X = traceLeft + (caliper2Time / MAX_TRACE_MS) * traceW;
          const minCX = Math.min(c1X, c2X);
          const maxCX = Math.max(c1X, c2X);
          const dtCaliper = Math.abs(caliper2Time - caliper1Time);
          const fCaliper = dtCaliper > 0 ? (1000.0 / dtCaliper).toFixed(1) : '0';

          // Caliper vertical needle lines
          c2d.strokeStyle = colors.trace;
          c2d.lineWidth = 1.5;
          c2d.setLineDash([3, 2]);
          c2d.beginPath();
          c2d.moveTo(c1X, traceTop);
          c2d.lineTo(c1X, traceBottom);
          c2d.moveTo(c2X, traceTop);
          c2d.lineTo(c2X, traceBottom);
          c2d.stroke();
          c2d.setLineDash([]);

          // Caliper horizontal dimension bar & handles
          const rulerY = traceTop + 14;
          c2d.strokeStyle = colors.trace;
          c2d.lineWidth = 1.4;
          c2d.beginPath();
          c2d.moveTo(minCX, rulerY);
          c2d.lineTo(maxCX, rulerY);
          c2d.moveTo(minCX, rulerY - 5);
          c2d.lineTo(minCX, rulerY + 5);
          c2d.moveTo(maxCX, rulerY - 5);
          c2d.lineTo(maxCX, rulerY + 5);
          c2d.stroke();

          c2d.font = '600 10px var(--font-mono)';
          c2d.fillStyle = colors.trace;
          c2d.textAlign = 'center';
          c2d.fillText(`Δt = ${dtCaliper.toFixed(1)} ms (${fCaliper} Hz)`, (minCX + maxCX) / 2, rulerY - 6);
        }
      }

      // =======================================================
      // SECTION 3: f-I TRANSFER FUNCTION & BIFURCATION
      // =======================================================
      if (!L.isMobile || currentChapter === 5) {
        const fiLeft = L.fiLeft;
        const fiRight = L.fiRight;
        const fiTop = L.fiTop;
        const fiBottom = L.fiBottom;
        const fiW = L.fiW;
        const fiH = L.fiH;

        const iToX = (I) => fiLeft + (I / 12.0) * fiW;
        const fToY = (f) => fiBottom - (f / 120.0) * fiH;

        // Section Header
        c2d.font = '600 11px var(--font-sans)';
        c2d.fillStyle = colors.ink;
        c2d.textAlign = 'left';
        c2d.fillText(S.f_I_title, fiLeft, fiTop - 8);

        // Subthreshold vs Active Spiking Zones
        const rheobaseI = (simParams.threshold - simParams.v_rest) / simParams.resistance;
        const rheoX = iToX(rheobaseI);

        c2d.fillStyle = 'rgba(87, 83, 78, 0.05)';
        c2d.fillRect(fiLeft, fiTop, rheoX - fiLeft, fiH);

        c2d.font = '500 9px var(--font-sans)';
        c2d.fillStyle = colors.inkMuted;
        c2d.textAlign = 'center';
        c2d.fillText(S.subthreshold_zone, (fiLeft + rheoX) / 2, fiTop + 24);

        // Axes lines
        c2d.strokeStyle = colors.border;
        c2d.lineWidth = 1;
        c2d.beginPath();
        c2d.moveTo(fiLeft, fiTop);
        c2d.lineTo(fiLeft, fiBottom);
        c2d.lineTo(fiRight, fiBottom);
        c2d.stroke();

        // X Ticks & Labels (0, 3, 6, 9, 12 nA)
        c2d.font = '500 9px var(--font-mono)';
        c2d.fillStyle = colors.inkMuted;
        c2d.textAlign = 'center';
        for (const tickI of [0, 3, 6, 9, 12]) {
          const tx = iToX(tickI);
          c2d.beginPath();
          c2d.moveTo(tx, fiBottom);
          c2d.lineTo(tx, fiBottom + 4);
          c2d.stroke();
          c2d.fillText(`${tickI}`, tx, fiBottom + 14);
        }
        c2d.fillText(S.f_I_xaxis, fiLeft + fiW / 2, fiBottom + 26);

        // Y Ticks & Labels (0, 30, 60, 90, 120 Hz)
        c2d.textAlign = 'right';
        for (const tickF of [30, 60, 90, 120]) {
          const ty = fToY(tickF);
          c2d.beginPath();
          c2d.moveTo(fiLeft - 4, ty);
          c2d.lineTo(fiLeft, ty);
          c2d.stroke();
          c2d.fillText(`${tickF}`, fiLeft - 5, ty + 3);
        }

        // Vertical Rheobase Boundary Line
        c2d.strokeStyle = colors.inh;
        c2d.lineWidth = 1.2;
        c2d.setLineDash([3, 3]);
        c2d.beginPath();
        c2d.moveTo(rheoX, fiTop);
        c2d.lineTo(rheoX, fiBottom);
        c2d.stroke();
        c2d.setLineDash([]);
        c2d.font = '600 9px var(--font-mono)';
        c2d.fillStyle = colors.inh;
        c2d.textAlign = 'center';
        c2d.fillText(`Rheobase (${rheobaseI.toFixed(1)} nA)`, rheoX, fiTop + 10);

        // Theoretical Continuous Curve f(I) as Dashed Line
        c2d.strokeStyle = colors.theory;
        c2d.lineWidth = 1.8;
        c2d.setLineDash([4, 3]);
        c2d.beginPath();
        let curveStarted = false;
        for (let curI = 0.0; curI <= 12.0; curI += 0.15) {
          const rate = core.theoryRate(curI, simParams);
          const px = iToX(curI);
          const py = fToY(rate);
          if (!curveStarted) {
            c2d.moveTo(px, py);
            curveStarted = true;
          } else {
            c2d.lineTo(px, py);
          }
        }
        c2d.stroke();
        c2d.setLineDash([]);

        // Empirical Simulation Points
        c2d.fillStyle = colors.ink;
        for (const pt of fiPoints) {
          const px = iToX(pt.I);
          const py = fToY(pt.f);
          c2d.beginPath();
          c2d.arc(px, py, 2.5, 0, Math.PI * 2);
          c2d.fill();
        }

        // Direct Manipulation Live Operating Point Marker (draggable)
        const liveRate = fTheory;
        const liveX = iToX(simParams.current);
        const liveY = fToY(liveRate);

        // Glowing outer ring
        c2d.fillStyle = 'rgba(180, 83, 9, 0.22)';
        c2d.beginPath();
        c2d.arc(liveX, liveY, 10, 0, Math.PI * 2);
        c2d.fill();

        // Operating Point Dot
        c2d.fillStyle = colors.volt;
        c2d.beginPath();
        c2d.arc(liveX, liveY, 5.5, 0, Math.PI * 2);
        c2d.fill();
        c2d.strokeStyle = colors.paper;
        c2d.lineWidth = 1.5;
        c2d.stroke();

        // Readout badge
        c2d.font = '600 10px var(--font-mono)';
        c2d.fillStyle = colors.volt;
        c2d.textAlign = liveX > fiRight - 70 ? 'right' : 'left';
        c2d.fillText(
          `(I=${simParams.current.toFixed(1)}, f=${liveRate.toFixed(1)} Hz)`,
          liveX + (liveX > fiRight - 70 ? -10 : 10),
          liveY - 5
        );
      }

    }

    ctx.onFrame(() => renderFrame(performance.now()));
    requestAnimationFrame(function raf(now) {
      renderFrame(now);
      requestAnimationFrame(raf);
    });
    renderFrame(performance.now());
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core };
  }

  if (typeof Atlas !== 'undefined') {
    Atlas.Plate.register({
      id: 'N01',
      title: { en: 'LIF: the leaky bucket', fa: 'LIF: سطل آب نشتی' },
      source: 'live',
      mount,
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
