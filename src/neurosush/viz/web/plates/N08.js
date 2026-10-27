/**
 * PLATE N08 · f-I curve and the bifurcation
 * Live JS simulation of the leaky integrate-and-fire (LIF) and exponential LIF (ELIF)
 * rate-current (f-I) relations. Units: ms, mV, nA (R in MΩ so 1 MΩ·nA = 1 mV), Hz.
 *
 * The hero figure is an f-I curve traced in real time by an automated triangular current
 * ramp; a membrane-voltage strip below shows the spike train densifying. The closed-form
 * (LIF) / quadrature (ELIF) rate is the dashed theory ghost; a draggable current marker lets
 * the student pick any holding current and watch the live spiking at that point.
 */
(function (root) {
  'use strict';

  // ---- String table (English only, no scattered literals) ----
  const S = {
    plateId: 'N08',
    title: 'f-I Curve & Bifurcation: rate versus current',
    source: 'live',
    libraryMirror:
      'neurosush.neurons.dynamics.lif_derivative · dynamics.exponential_boost · ' +
      'dynamics.lif_terms · dynamics.elif_terms · neurons.models.LIF · neurons.models.ELIF',

    // ---- Margins ----
    eqHeading: 'Governing equations',
    eqLif: 'τ dV/dt = (V_rest − V) + R·I',
    eqElif: 'τ dV/dt = (V_rest − V) + R·I + Δ·exp((V−θ)/Δ)',
    eqIsi: 'T = τ · ln( I / (I − I_rh) )',
    eqRate: 'f = 1 / T',
    eqRheoLif: 'I_rh = (V_th − V_rest) / R',
    eqRheoElif: 'I_rh = (θ − V_rest − Δ) / R',
    eqSqrt: 'near onset:  f ~ √(I − I_rh)   (ELIF, type-I)',
    eqLog: 'LIF onset is logarithmic:  f → 0 like 1 / ln(1/(I−I_rh))',
    eqGain: 'gain  S = df/dI  (Hz/nA)',

    readoutsHeading: 'Live values',
    roModel: 'model',
    roI: 'holding current I',
    roIrh: 'rheobase I_rh',
    roT: 'interspike interval T',
    roF: 'firing rate f',
    roFsim: 'simulated rate',
    roNotYet: 'not yet',
    roGain: 'gain  S = df/dI',
    roDiff: 'theory − sim  difference',
    roNoise: 'current noise σ_I',
    roSigmaV: 'voltage noise σ_V = R·σ_I',

    unitsNa: 'nA',
    unitsHz: 'Hz',
    unitsMs: 'ms',
    unitsMv: 'mV',
    unitsHzPerNa: 'Hz/nA',

    glossHeading: 'Glossary',
    glRheobase: 'Rheobase — the smallest steady current that makes the neuron fire.',
    glFI: 'f-I curve — steady firing rate f as a function of a constant input current I.',
    glTypeI: 'Type-I onset — rate rises continuously from zero at I_rh (e.g. ELIF, √ law).',
    glIsi: 'ISI — interspike interval; f = 1/T.',
    glGain: 'Gain — slope df/dI of the f-I curve, in Hz per nA.',
    glHyst: 'Dynamical hysteresis — the trace lags a changing current, so the up- and down-ramp loops differ.',
    glNoise: 'Noise smoothing — random current lets the neuron fire below I_rh, rounding the sharp knee.',

    // ---- Hero / figure ----
    figTitle: 'Rate–current relation  f(I)',
    axisI: 'input current  I  (nA)',
    axisF: 'firing rate  f  (Hz)',
    axisV: 'V  (mV)',
    axisTs: 'simulated time  (ms)',
    legendTheory: 'theory (dashed)',
    legendMeas: 'measured trail',
    legendNow: 'current marker',
    legendNoise: 'noise theory',
    stripTitle: 'membrane voltage & spikes',
    insetTitle: 'onset zoom  (log–log)',
    insetX: 'log₁₀ (I − I_rh)',
    insetY: 'log₁₀ f',
    slopeHalf: 'slope ½',
    rheoTag: 'I_rh',
    silentTag: 'silent — no spikes',
    firingTag: 'firing',
    rampUpTag: 'ramp ▲',
    rampDownTag: 'ramp ▼',
    pokeHint: 'click the voltage strip to inject a brief current pulse',
    dragHint: 'drag the violet marker to pick a holding current',

    // ---- Controls ----
    chLabel: 'Chapter',
    ofLabel: 'of',
    btnPrev: '← Back',
    btnNext: 'Next →',
    btnModelLif: 'Model: LIF',
    btnModelElif: 'Model: ELIF',
    btnAuto: 'Auto ramp',
    btnHold: 'Hold (drag)',
    btnPredict: 'Predict [P]',
    btnTheory: 'Theory [T]',
    lblNoise: 'noise σ_I',
    btnReset: 'Clear trail',
    predictOpen: 'Test yourself',

    // ---- Chapters ----
    ch1Title: '1 · Ramp the current',
    ch1Body:
      'A slow current ramp sweeps I from 0 upward and back. The neuron stays silent until I crosses the rheobase, then the spike train in the strip densifies and the measured point climbs the curve.',
    ch2Title: '2 · The rheobase',
    ch2Body:
      'I_rh is the smallest steady current that fires the cell. Below it the voltage settles at a stable rest; above it there is no rest and every trajectory escapes to threshold. ELIF reaches rheobase below LIF because the exponential term helps.',
    ch3Title: '3 · The curve builds',
    ch3Body:
      'Each sampled point places the measured rate at its current. The dashed ink line is the closed-form prediction: for LIF it is exact, for ELIF it is the ISI quadrature. The difference readout compares the two.',
    ch4Title: '4 · √ onset versus log onset',
    ch4Body:
      'Zoom on the onset with a log–log inset. ELIF is type-I: f grows as √(I−I_rh), a straight line of slope ½. LIF onset is logarithmic, f → 0 like 1/ln(1/(I−I_rh)), a curve no power law follows. Predict: will doubling I double f?',
    ch5Title: '5 · Noise smooths the knee',
    ch5Body:
      'Add a seeded random current. Spikes now appear below I_rh and the sharp corner is rounded: the measured curve meets the noise-theory curve (dash-dot) and lifts off the noiseless ghost. The seed makes the run reproducible.',

    // ---- Predict ----
    predictTitle: 'Predict  ·  then reveal',
    predictQ: 'If the input current is doubled, does the firing rate double?',
    predictA: 'A) Yes — the rate always doubles.',
    predictB: 'B) Not exactly: near onset ELIF grows as √(I−I_rh) and LIF is logarithmic, so the factor depends on I.',
    predictC: 'C) No — the rate is independent of the current.',
    predictSubmit: 'Commit',
    predictCorrect:
      'Correct. Near the rheobase the ELIF rate grows as √(I−I_rh): doubling the drive to only 1.5–1.8 nA multiplies the rate by about 1.4×, not 2×. Far above I_rh both models approach a nearly linear f(I), so the factor approaches 2 only asymptotically.',
    predictWrong:
      'Not quite. The relation is nonlinear. Near onset ELIF grows as √(I−I_rh) (a factor ~1.4 for a doubling), and LIF onset is logarithmic (a factor ~2.4 at twice its rheobase). Only far above I_rh does the factor approach 2.',
    predictRun: 'Run it',
    predictTryAgain: 'Try again',
    predictDone: 'Revealed.'
  };

  // ---- small pure helpers ----
  function mulberry32(a) {
    let t = a >>> 0;
    return function () {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), t | 1);
      r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Box-Muller normal generator with its own cached pair (deterministic under a seed).
  function makeGaussian(rng) {
    let has = false;
    let cache = 0;
    return function () {
      if (has) {
        has = false;
        return cache;
      }
      let u = 0;
      let v = 0;
      while (u === 0) u = rng();
      while (v === 0) v = rng();
      const r = Math.sqrt(-2 * Math.log(u));
      const th = 2 * Math.PI * v;
      cache = r * Math.sin(th);
      has = true;
      return r * Math.cos(th);
    };
  }

  // ---- core: pure math, no DOM, exported for node tests ----
  const P = {
    TAU: 20, // membrane time constant, ms
    R: 10, // membrane resistance, MΩ   (1 MΩ·nA = 1 mV)
    V_REST: -65, // resting potential, mV
    V_RESET: -65, // reset potential after a spike, mV
    LIF_THRESHOLD: -50, // LIF spike threshold, mV
    ELIF_THRESHOLD: -40, // ELIF spike threshold, mV
    THETA_RH: -50, // ELIF exponential reference voltage θ, mV
    DELTA: 2 // ELIF exponential sharpness Δ, mV
  };

  const core = {
    P: P,

    // --- drift: the right-hand side of τ dV/dt, in mV (divide by τ for mV/ms) ---
    lifDrive(V, I) {
      return (P.V_REST - V) + P.R * I;
    },
    elifDrive(V, I) {
      return (P.V_REST - V) + P.R * I + P.DELTA * Math.exp((V - P.THETA_RH) / P.DELTA);
    },
    drive(model, V, I) {
      return model === 'elif' ? this.elifDrive(V, I) : this.lifDrive(V, I);
    },
    threshold(model) {
      return model === 'elif' ? P.ELIF_THRESHOLD : P.LIF_THRESHOLD;
    },

    // --- rheobase, where drift has a saddle-node at the exponential knee ---
    lifRheobase() {
      return (P.LIF_THRESHOLD - P.V_REST) / P.R;
    },
    elifRheobase() {
      return (P.THETA_RH - P.V_REST - P.DELTA) / P.R;
    },
    rheobase(model) {
      return model === 'elif' ? this.elifRheobase() : this.lifRheobase();
    },

    // --- closed-form LIF rate (exact ISI), Hz ---
    lifRate(I) {
      if (!Number.isFinite(I)) throw new Error('lifRate: current must be finite, got ' + I);
      const Irh = this.lifRheobase();
      if (I <= Irh) return 0;
      const T = P.TAU * Math.log(I / (I - Irh)); // ms
      return 1000 / T;
    },

    // --- ELIF ISI by quadrature T = τ ∫ dV / drive(V) over (V_reset, V_th), ms ---
    elifIsi(I, opts) {
      if (!Number.isFinite(I)) throw new Error('elifIsi: current must be finite, got ' + I);
      const Irh = this.elifRheobase();
      if (I <= Irh) return Infinity;
      const n = (opts && opts.n) || 20000;
      const a = P.V_RESET;
      const b = P.ELIF_THRESHOLD;
      const h = (b - a) / n;
      let s = 0;
      for (let i = 0; i < n; i++) {
        s += h / this.elifDrive(a + (i + 0.5) * h, I);
      }
      return P.TAU * s;
    },
    elifRate(I, opts) {
      const T = this.elifIsi(I, opts);
      return Number.isFinite(T) && T > 0 ? 1000 / T : 0;
    },

    // --- steady-state (noiseless) theory rate for either model, Hz ---
    theoryRate(model, I) {
      return model === 'elif' ? this.elifRate(I) : this.lifRate(I);
    },

    // --- one integration step (RK4 for both models; exact in the limit dt→0) ---
    step(model, V, I, dt) {
      if (model === 'elif') {
        const f = (v) => this.elifDrive(v, I) / P.TAU;
        const k1 = f(V);
        const k2 = f(V + 0.5 * dt * k1);
        const k3 = f(V + 0.5 * dt * k2);
        const k4 = f(V + dt * k3);
        return V + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
      }
      // LIF: linear in V, RK4 collapses to the exact exponential map
      const f = (v) => this.lifDrive(v, I) / P.TAU;
      const k1 = f(V);
      const k2 = f(V + 0.5 * dt * k1);
      const k3 = f(V + 0.5 * dt * k2);
      const k4 = f(V + dt * k3);
      return V + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    },

    // --- simulate to a steady rate; sigma is current noise in nA, seeded ---
    simulateSteady(model, I, opts) {
      const o = opts || {};
      const dt = o.dt === undefined ? 0.02 : o.dt;
      const warmup = o.warmup === undefined ? 200 : o.warmup;
      const duration = o.duration === undefined ? 2000 : o.duration;
      const sigma = o.sigma === undefined ? 0 : o.sigma; // nA
      const seed = o.seed === undefined ? 1 : o.seed;
      const vth = this.threshold(model);
      const rng = mulberry32((seed >>> 0) || 1);
      const gn = makeGaussian(rng);
      let V = P.V_RESET;
      let t = 0;
      const steps = Math.round((warmup + duration) / dt);
      const spikes = [];
      for (let k = 0; k < steps; k++) {
        let In = I;
        if (sigma > 0) In += sigma * gn();
        V = this.step(model, V, In, dt);
        t += dt;
        if (V >= vth) {
          V = P.V_RESET;
          if (t >= warmup) spikes.push(t - warmup);
        }
      }
      let rate = 0;
      let isiMean = 0;
      if (spikes.length >= 2) {
        let sum = 0;
        for (let i = 1; i < spikes.length; i++) sum += spikes[i] - spikes[i - 1];
        isiMean = sum / (spikes.length - 1);
        rate = 1000 / isiMean;
      }
      return { rate: rate, isiMean: isiMean, spikes: spikes, n: spikes.length };
    },

    // --- noise theory: stationary rate of a white-noise-driven IF neuron ---
    // dV = D(V) dt + σ dW with D in mV/ms and σ in mV/√ms.  Risken's double integral:
    //   r = S / ( 2 ∫_{V_r}^{V_th} e^{Φ(v)} ∫_v^{V_th} e^{-Φ(x)} dx dv ),  Φ = ∫ 2D/S, S = σ².
    noiseRate(model, I, sigma, opts) {
      if (!(sigma > 0)) return this.theoryRate(model, I);
      const n = (opts && opts.n) || 1200;
      const vth = this.threshold(model);
      const a = P.V_RESET;
      const b = vth;
      const S = sigma * sigma;
      const h = (b - a) / n;
      const D = new Float64Array(n + 1);
      const F = new Float64Array(n + 1);
      for (let i = 0; i <= n; i++) D[i] = this.drive(model, a + i * h, I) / P.TAU;
      F[0] = 0;
      for (let i = 1; i <= n; i++) F[i] = F[i - 1] + (2 / S) * 0.5 * (D[i] + D[i - 1]) * h;
      let Idd = 0;
      for (let i = 0; i <= n; i++) {
        let inner = 0;
        for (let j = i; j <= n; j++) {
          inner += Math.exp(F[i] - F[j]) * h;
        }
        Idd += inner * h;
      }
      if (!(Idd > 0)) return Infinity;
      const r = S / (2 * Idd); // 1/ms
      return r * 1000; // Hz
    },
    // convenience: current noise in nA → voltage noise σ = R·σ_I
    noiseRateFromCurrent(model, I, sigmaI, opts) {
      return this.noiseRate(model, I, P.R * sigmaI, opts);
    },

    // --- triangular ramp on simulated time; returns current and direction ---
    rampCurrent(tsim, opts) {
      const o = opts || {};
      const Imin = o.Imin === undefined ? 0 : o.Imin;
      const Imax = o.Imax === undefined ? 5 : o.Imax;
      const half = o.half === undefined ? 500 : o.half;
      const cycle = 2 * half;
      let p = (tsim % cycle) / half;
      const up = p <= 1;
      if (!up) p = 2 - p;
      return { I: Imin + (Imax - Imin) * p, up: up };
    },
    // --- micro sweep around a model's rheobase (rheobase chapter) ---
    rheoSweep(tsim, model, opts) {
      const o = opts || {};
      const w = o.width === undefined ? 0.22 : o.width;
      const half = o.half === undefined ? 320 : o.half;
      const c = this.rheobase(model);
      const r = this.rampCurrent(tsim, { Imin: c - w, Imax: c + w, half: half });
      return r;
    },

    mulberry32: mulberry32,
    makeGaussian: makeGaussian
  };

  // ---- mount: only runs when the browser Atlas shell exists ----
  function mount(rootEl, ctx) {
    if (!ctx || typeof ctx.colors !== 'function') return;

    const figureContainer = ctx.figure || rootEl;
    const marginContainer = ctx.margin || document.getElementById('atlas-margin');
    const figureSection =
      document.getElementById('atlas-figure') || (figureContainer && figureContainer.parentElement);

    if (ctx.clock) ctx.clock.stepsPerSecond = 60;

    // Give the figure a little more vertical room than the default 16:10.
    if (figureContainer && figureContainer.style) {
      figureContainer.style.aspectRatio = '4 / 3';
      // Desktop: the shell lays the plate out in 100vh; keep the canvas within the flex area.
      if (typeof window !== 'undefined' && window.innerWidth > 720) {
        figureContainer.style.height = '430px';
      } else {
        // A fixed reading height keeps the trace band legible on a narrow screen.
        figureContainer.style.height = '500px';
      }
    }

    // ---------- string helpers for the DOM ----------
    const dim = (text, extra) =>
      `<span style="color:var(--ink-muted)${extra ? ';' + extra : ''}">${text}</span>`;

    // ---------- 1. Marginalia ----------
    if (marginContainer) {
      marginContainer.innerHTML = `
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.eqHeading}</div>
          <div class="atlas-equation-box" id="n08-eq-box">
            <div><span class="n08-term" data-term="elif" style="color:var(--exc);cursor:pointer;">${S.eqElif}</span></div>
            <div style="margin-top:4px;">${dim(S.eqLif)}</div>
            <div style="margin-top:6px;border-top:1px dashed var(--border);padding-top:6px;">
              <span class="n08-term" data-term="T" style="cursor:pointer;">${S.eqIsi}</span>
              <span class="n08-term" data-term="f" style="cursor:pointer;"> , ${S.eqRate}</span>
            </div>
            <div style="margin-top:6px;">
              <span class="n08-term" data-term="Irh" style="color:var(--volt);cursor:pointer;">${S.eqRheoLif}</span>
              <span style="color:var(--ink-muted)">  ·  </span>
              <span class="n08-term" data-term="Irh" style="color:var(--exc);cursor:pointer;">${S.eqRheoElif}</span>
            </div>
            <div style="margin-top:6px;color:var(--exc);">${S.eqSqrt}</div>
            <div style="color:var(--ink-muted);">${S.eqLog}</div>
          </div>
        </div>

        <div class="atlas-margin-section" id="n08-readouts">
          <div class="atlas-margin-label">${S.readoutsHeading}</div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span class="atlas-serif">${S.roModel}</span>
            <span class="atlas-mono" id="n08-val-model" style="color:var(--exc);font-weight:600;">ELIF</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span class="atlas-serif" style="color:var(--mod);font-weight:600;">${S.roI}</span>
            <span class="atlas-mono" id="n08-val-I">0.00 nA</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span class="atlas-serif">${S.roIrh}</span>
            <span class="atlas-mono" id="n08-val-Irh">1.30 nA</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;border-top:1px dashed var(--border);padding-top:4px;">
            <span class="atlas-serif">${S.roT}</span>
            <span class="atlas-mono" id="n08-val-T">—</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span class="atlas-serif" style="font-weight:600;">${S.roF}</span>
            <span class="atlas-mono" id="n08-val-f" style="color:var(--trace);font-weight:600;">0.0 Hz</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span class="atlas-serif">${S.roFsim}</span>
            <span class="atlas-mono" id="n08-val-fsim">0.0 Hz</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span class="atlas-serif">${S.roGain}</span>
            <span class="atlas-mono" id="n08-val-gain">—</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;" id="n08-diff-row">
            <span class="atlas-serif">${S.roDiff}</span>
            <span class="atlas-mono" id="n08-val-diff">0.0 Hz</span>
          </div>
          <div id="n08-noise-readout" style="display:none;border-top:1px dashed var(--border);padding-top:4px;">
            <div style="display:flex;justify-content:space-between;align-items:baseline;">
              <span class="atlas-serif">${S.roNoise}</span>
              <span class="atlas-mono" id="n08-val-sigmaI">0.00 nA</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;">
              <span class="atlas-serif">${S.roSigmaV}</span>
              <span class="atlas-mono" id="n08-val-sigmaV">0.00 mV</span>
            </div>
          </div>
        </div>

        <div class="atlas-margin-section" style="font-size:11px;line-height:1.45;">
          <div class="atlas-margin-label">${S.glossHeading}</div>
          <p style="margin:2px 0;">• ${S.glRheobase}</p>
          <p style="margin:0 0 2px;">• ${S.glFI}</p>
          <p style="margin:0 0 2px;">• ${S.glTypeI}</p>
          <p style="margin:0 0 2px;">• ${S.glIsi}</p>
          <p style="margin:0 0 2px;">• ${S.glGain}</p>
          <p style="margin:0 0 2px;">• ${S.glHyst}</p>
          <p style="margin:0 0 2px;">• ${S.glNoise}</p>
        </div>

        <div class="atlas-margin-section" style="font-size:10px;line-height:1.4;color:var(--ink-muted);">
          <div class="atlas-margin-label">Source</div>
          <div>${S.source} · mirrors <span class="atlas-mono">${S.libraryMirror}</span></div>
        </div>
      `;

      marginContainer.querySelectorAll('.n08-term').forEach((el) => {
        el.addEventListener('pointerenter', () => {
          state.hoverTerm = el.getAttribute('data-term');
          el.style.textDecoration = 'underline';
        });
        el.addEventListener('pointerleave', () => {
          state.hoverTerm = null;
          el.style.textDecoration = 'none';
        });
      });
    }

    // ---------- 2. Chapter bar + toolbar + predict panel (above the canvas) ----------
    let chapterBar = document.getElementById('n08-chapter-bar');
    if (!chapterBar) {
      chapterBar = document.createElement('div');
      chapterBar.id = 'n08-chapter-bar';
      chapterBar.style.cssText =
        'background:var(--paper-card);border:1px solid var(--border);border-radius:var(--radius);' +
        'padding:8px 12px;margin-bottom:6px;display:flex;flex-direction:column;gap:5px;';
      if (figureSection) figureSection.insertBefore(chapterBar, figureContainer);
    }
    chapterBar.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span class="atlas-badge" id="n08-ch-badge" style="font-size:10px;padding:1px 5px;">${S.chLabel} 1 ${S.ofLabel} 5</span>
          <span class="atlas-serif" id="n08-ch-title" style="font-size:13px;font-weight:600;">${S.ch1Title}</span>
          <div id="n08-ch-dots" style="display:flex;gap:4px;"></div>
        </div>
        <div style="display:flex;align-items:center;gap:4px;flex-shrink:0;">
          <button class="atlas-btn" id="n08-btn-prev" disabled style="padding:4px 9px;font-size:11px;min-height:28px;">${S.btnPrev}</button>
          <button class="atlas-btn" id="n08-btn-next" style="padding:4px 9px;font-size:11px;min-height:28px;">${S.btnNext}</button>
          <button class="atlas-btn" id="n08-btn-predict" aria-pressed="false" style="padding:4px 9px;font-size:11px;min-height:28px;">${S.btnPredict}</button>
        </div>
      </div>
      <div class="atlas-serif" id="n08-ch-body" style="font-size:12px;color:var(--ink);line-height:1.4;">${S.ch1Body}</div>
      <div id="n08-predict-box" style="display:none;background:var(--paper-inset);border:1px solid var(--border);border-radius:3px;padding:8px 10px;">
        <div class="atlas-margin-label" style="font-size:10px;color:var(--mod);">${S.predictTitle}</div>
        <div class="atlas-serif" style="font-size:12px;margin:3px 0 6px;">${S.predictQ}</div>
        <div style="display:flex;flex-direction:column;gap:4px;">
          <label style="cursor:pointer;display:flex;gap:6px;font-size:12px;"><input type="radio" name="n08-predict" value="A"><span>${S.predictA}</span></label>
          <label style="cursor:pointer;display:flex;gap:6px;font-size:12px;"><input type="radio" name="n08-predict" value="B"><span>${S.predictB}</span></label>
          <label style="cursor:pointer;display:flex;gap:6px;font-size:12px;"><input type="radio" name="n08-predict" value="C"><span>${S.predictC}</span></label>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap;">
          <button class="atlas-btn" id="n08-btn-commit" disabled style="padding:4px 10px;font-size:11px;min-height:28px;">${S.predictSubmit}</button>
          <button class="atlas-btn" id="n08-btn-runit" style="padding:4px 10px;font-size:11px;min-height:28px;display:none;">${S.predictRun}</button>
          <span id="n08-predict-feedback" class="atlas-serif" style="display:none;font-size:11px;font-weight:600;"></span>
        </div>
      </div>
    `;

    let toolbar = document.getElementById('n08-toolbar');
    if (!toolbar) {
      toolbar = document.createElement('div');
      toolbar.id = 'n08-toolbar';
      toolbar.style.cssText =
        'display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;';
      if (figureSection) figureSection.insertBefore(toolbar, figureContainer);
    }
    toolbar.innerHTML = `
      <button class="atlas-btn" id="n08-btn-model" aria-pressed="true" style="font-weight:600;font-size:12px;border-color:var(--exc);color:var(--exc);">${S.btnModelElif}</button>
      <button class="atlas-btn" id="n08-btn-mode" style="font-size:12px;">${S.btnHold}</button>
      <button class="atlas-btn active" id="n08-btn-theory" aria-pressed="true" style="font-size:12px;">${S.btnTheory}</button>
      <div style="display:inline-flex;align-items:center;gap:5px;">
        <label for="n08-noise" class="atlas-serif" style="font-size:11px;">${S.lblNoise}</label>
        <input type="range" id="n08-noise" min="0" max="0.25" step="0.01" value="0" style="width:84px;cursor:pointer;">
        <span class="atlas-mono" id="n08-noise-txt" style="font-size:11px;min-width:52px;">0.00 nA</span>
      </div>
      <button class="atlas-btn" id="n08-btn-clear" style="font-size:12px;">${S.btnReset}</button>
    `;

    // ---------- 3. Canvas ----------
    const targetH = (typeof window !== 'undefined' && window.innerWidth < 560) ? '520px' : '440px';
    if (figureContainer && figureContainer.style) {
      figureContainer.style.setProperty('aspect-ratio', 'auto', 'important');
      figureContainer.style.setProperty('height', targetH, 'important');
      figureContainer.style.setProperty('min-height', targetH, 'important');
      figureContainer.style.setProperty('flex-shrink', '0', 'important');
    }
    const canvas = figureContainer.querySelector('canvas') || ctx.canvas();
    const g = canvas.getContext('2d');

    const $ = (id) => document.getElementById(id);
    const btnModel = $('n08-btn-model');
    const btnMode = $('n08-btn-mode');
    const btnTheory = $('n08-btn-theory');
    const noiseInput = $('n08-noise');
    const noiseTxt = $('n08-noise-txt');
    const btnClear = $('n08-btn-clear');
    const btnPrev = $('n08-btn-prev');
    const btnNext = $('n08-btn-next');
    const btnPredict = $('n08-btn-predict');
    const btnCommit = $('n08-btn-commit');
    const btnRunIt = $('n08-btn-runit');
    const predictBox = $('n08-predict-box');
    const predictFeedback = $('n08-predict-feedback');
    const chBadge = $('n08-ch-badge');
    const chTitle = $('n08-ch-title');
    const chBody = $('n08-ch-body');
    const chDots = $('n08-ch-dots');
    const noiseReadout = $('n08-noise-readout');

    // ---------- state ----------
    const state = {
      model: 'elif',
      mode: 'ramp', // 'ramp' | 'hold' | 'rheo'
      viz: 'hold', // derived button label cache: 'ramp' | 'hold'
      theoryOn: true,
      chapter: 1,
      hoverTerm: null,
      highlight: null,
      simT: 0,
      V: P.V_RESET,
      lastI: 0,
      rampUp: true,
      rampHalf: 500,
      Imax: 5.0,
      holdI: P.V_REST !== 0 ? 1.3 : 1.3,
      pulseMs: 0,
      noiseI: 0, // σ_I in nA
      dragging: false,
      trail: [],
      trailClock: 0,
      spikes: [], // sim times (ms)
      vBuf: [], // {t, V}
      seed: 12345,
      rng: mulberry32(12345),
      gauss: null,
      predictAnswer: null,
      predictRevealed: false,
      theoryCache: {},
      noiseCacheKey: '',
      noiseCurve: null,
      W: 800,
      H: 470
    };
    state.gauss = makeGaussian(state.rng);
    state.holdI = core.elifRheobase() * 1.0;

    // precompute a chunky noiseless theory curve per model (cheap, cached)
    function theoryCurve(model) {
      if (state.theoryCache[model]) return state.theoryCache[model];
      const N = 220;
      const pts = [];
      for (let i = 0; i <= N; i++) {
        const I = (i / N) * state.Imax;
        pts.push({ I: I, f: core.theoryRate(model, I) });
      }
      state.theoryCache[model] = pts;
      return pts;
    }

    // onset sample set for the log-log inset (fine near the knee)
    function onsetCurve(model) {
      const key = 'onset:' + model;
      if (state.theoryCache[key]) return state.theoryCache[key];
      const Irh = core.rheobase(model);
      const pts = [];
      const N = 48;
      for (let i = 0; i < N; i++) {
        const dI = Math.pow(10, -3 + (i / (N - 1)) * 3); // 1e-3 … 1 nA
        const f = core.theoryRate(model, Irh + dI);
        if (f > 0) pts.push({ dI: dI, f: f });
      }
      state.theoryCache[key] = pts;
      return pts;
    }

    function noiseCurve(sigmaI) {
      const key = sigmaI.toFixed(3);
      if (state.noiseCacheKey === key && state.noiseCurve) return state.noiseCurve;
      const pts = [];
      const N = 44;
      for (let i = 0; i <= N; i++) {
        const I = (i / N) * state.Imax;
        pts.push({ I: I, f: core.noiseRateFromCurrent(state.model, I, sigmaI, { n: 700 }) });
      }
      state.noiseCacheKey = key;
      state.noiseCurve = pts;
      return pts;
    }

    function modelColor(c) {
      return state.model === 'elif' ? c.exc : c.volt;
    }

    // ---------- UI wiring ----------
    function setModel(model) {
      state.model = model;
      state.theoryCache = {};
      state.noiseCurve = null;
      state.noiseCacheKey = '';
      state.spikes = [];
      state.vBuf = [];
      state.trail = [];
      state.V = P.V_RESET;
      state.simT = 0;
      state.holdI = core.rheobase(model);
      btnModel.textContent = model === 'elif' ? S.btnModelElif : S.btnModelLif;
      btnModel.style.borderColor = model === 'elif' ? 'var(--exc)' : 'var(--volt)';
      btnModel.style.color = model === 'elif' ? 'var(--exc)' : 'var(--volt)';
      const mc = $('n08-val-model');
      if (mc) {
        mc.textContent = model === 'elif' ? 'ELIF' : 'LIF';
        mc.style.color = model === 'elif' ? 'var(--exc)' : 'var(--volt)';
      }
      const irh = $('n08-val-Irh');
      if (irh) irh.textContent = core.rheobase(model).toFixed(2) + ' ' + S.unitsNa;
      updateNoiseReadout();
      resetSimulation(true);
    }

    function setTheory(on) {
      state.theoryOn = on;
      btnTheory.classList.toggle('active', on);
      btnTheory.setAttribute('aria-pressed', String(on));
    }

    function setMode(mode) {
      state.mode = mode;
      resetSimulation(true);
      if (mode === 'ramp') {
        state.highlight = 'trail';
        btnMode.classList.remove('active');
      } else if (mode === 'rheo') {
        btnMode.classList.add('active');
      } else {
        btnMode.classList.add('active');
      }
    }

    function clearTrail() {
      state.trail = [];
      state.trailClock = 0;
    }

    function updateNoiseReadout() {
      if (!noiseReadout) return;
      noiseReadout.style.display = state.noiseI > 0 ? 'block' : 'none';
      const si = $('n08-val-sigmaI');
      const sv = $('n08-val-sigmaV');
      if (si) si.textContent = state.noiseI.toFixed(2) + ' ' + S.unitsNa;
      if (sv) sv.textContent = (P.R * state.noiseI).toFixed(2) + ' ' + S.unitsMv;
    }

    btnModel.addEventListener('click', () => setModel(state.model === 'elif' ? 'lif' : 'elif'));
    btnMode.addEventListener('click', () =>
      setMode(state.mode === 'ramp' ? 'hold' : 'ramp')
    );
    btnTheory.addEventListener('click', () => setTheory(!state.theoryOn));
    btnClear.addEventListener('click', clearTrail);

    noiseInput.addEventListener('input', (e) => {
      state.noiseI = parseFloat(e.target.value) || 0;
      noiseTxt.textContent = state.noiseI.toFixed(2) + ' ' + S.unitsNa;
      if (state.noiseI === 0) {
        state.noiseCurve = null;
        state.noiseCacheKey = '';
      }
      updateNoiseReadout();
    });

    // theory ghost is toggled from the shell key 'T' / toolbar button
    window.addEventListener('atlas:theory', (e) => {
      if (e.detail && e.detail.active !== undefined) setTheory(e.detail.active);
    });

    // ---------- predict ----------
    const predictOptions = document.querySelectorAll('input[name="n08-predict"]');
    predictOptions.forEach((opt) => {
      opt.addEventListener('change', (e) => {
        state.predictAnswer = e.target.value;
        btnCommit.disabled = false;
      });
    });
    btnPredict.addEventListener('click', () => {
      const vis = predictBox.style.display !== 'none';
      predictBox.style.display = vis ? 'none' : 'block';
      btnPredict.setAttribute('aria-pressed', String(!vis));
    });
    btnCommit.addEventListener('click', () => {
      state.predictRevealed = true;
      btnCommit.style.display = 'none';
      predictFeedback.style.display = 'inline-block';
      const correct = state.predictAnswer === 'B';
      predictFeedback.style.color = correct ? 'var(--exc)' : 'var(--inh)';
      predictFeedback.textContent = (correct ? '✓ ' : '✗ ') + (correct ? S.predictCorrect : S.predictWrong);
      btnRunIt.style.display = 'inline-block';
      setTheory(true);
    });
    btnRunIt.addEventListener('click', () => {
      // run the doubling experiment for the current model near onset
      const Irh = core.rheobase(state.model);
      const I1 = Irh + 0.3;
      const f1 = core.theoryRate(state.model, I1);
      const f2 = core.theoryRate(state.model, 2 * I1);
      const ratio = f1 > 0 ? f2 / f1 : NaN;
      predictFeedback.textContent =
        `I = ${I1.toFixed(2)} → ${f1.toFixed(1)} Hz;  ` +
        `2I = ${(2 * I1).toFixed(2)} → ${f2.toFixed(1)} Hz;  ` +
        `ratio = ${ratio.toFixed(2)}× (not 2).`;
      setMode('hold');
      state.holdI = 2 * I1;
      state.trail = [];
    });

    // ---------- chapters ----------
    const chapters = [
      {
        num: 1,
        title: S.ch1Title,
        body: S.ch1Body,
        setup: () => {
          state.noiseI = 0;
          noiseInput.value = '0';
          noiseTxt.textContent = '0.00 nA';
          updateNoiseReadout();
          setTheory(true);
          setMode('ramp');
          state.highlight = 'trail';
        }
      },
      {
        num: 2,
        title: S.ch2Title,
        body: S.ch2Body,
        setup: () => {
          state.noiseI = 0;
          noiseInput.value = '0';
          noiseTxt.textContent = '0.00 nA';
          updateNoiseReadout();
          setTheory(true);
          setMode('rheo');
          state.highlight = 'rheo';
        }
      },
      {
        num: 3,
        title: S.ch3Title,
        body: S.ch3Body,
        setup: () => {
          state.noiseI = 0;
          noiseInput.value = '0';
          noiseTxt.textContent = '0.00 nA';
          updateNoiseReadout();
          setTheory(true);
          setMode('ramp');
          state.highlight = 'trail';
        }
      },
      {
        num: 4,
        title: S.ch4Title,
        body: S.ch4Body,
        setup: () => {
          setTheory(true);
          setMode('hold');
          state.holdI = core.rheobase(state.model) + 0.35;
          state.highlight = 'inset';
          predictBox.style.display = 'block';
          btnPredict.setAttribute('aria-pressed', 'true');
        }
      },
      {
        num: 5,
        title: S.ch5Title,
        body: S.ch5Body,
        setup: () => {
          state.noiseI = 0.08;
          noiseInput.value = '0.08';
          noiseTxt.textContent = '0.08 nA';
          updateNoiseReadout();
          setTheory(true);
          setMode('ramp');
          state.highlight = 'noise';
        }
      }
    ];

    function renderDots() {
      let html = '';
      for (let i = 0; i < chapters.length; i++) {
        const on = i + 1 === state.chapter;
        html += `<span class="n08-dot" data-ch="${i + 1}" style="cursor:pointer;font-size:12px;padding:3px 5px;display:inline-flex;align-items:center;justify-content:center;min-width:20px;min-height:20px;color:${
          on ? 'var(--mod)' : 'var(--ink-muted)'
        };opacity:${on ? '1' : '0.45'};">●</span>`;
      }
      chDots.innerHTML = html;
      chDots.querySelectorAll('.n08-dot').forEach((d) => {
        d.addEventListener('click', () => applyChapter(parseInt(d.getAttribute('data-ch'), 10)));
      });
    }

    function applyChapter(idx) {
      if (idx < 1 || idx > chapters.length) return;
      state.chapter = idx;
      const ch = chapters[idx - 1];
      chBadge.textContent = `${S.chLabel} ${ch.num} ${S.ofLabel} ${chapters.length}`;
      chTitle.textContent = ch.title;
      chBody.textContent = ch.body;
      btnPrev.disabled = idx === 1;
      btnNext.disabled = idx === chapters.length;
      renderDots();
      ch.setup();
    }

    btnPrev.addEventListener('click', () => applyChapter(state.chapter - 1));
    btnNext.addEventListener('click', () => applyChapter(state.chapter + 1));

    // keyboard: arrows step chapters (space / T are handled by the shell)
    window.addEventListener('keydown', (e) => {
      const tag = e.target && e.target.tagName ? e.target.tagName.toUpperCase() : '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'ArrowRight') applyChapter(state.chapter + 1);
      else if (e.key === 'ArrowLeft') applyChapter(state.chapter - 1);
      else if (e.key === 'p' || e.key === 'P') btnPredict.click();
    });

    // ---------- pointer interaction on the canvas ----------
    function plotRect(W, H) {
      const small = W < 560;
      return { x0: 62, x1: W - 16, y0: 40, y1: Math.round(H * (small ? 0.49 : 0.60)) };
    }
    function stripRect(W, H) {
      const r = plotRect(W, H);
      const top = r.y1 + 34;
      return { x0: 62, x1: W - 16, y0: top, y1: H - (W < 560 ? 28 : 16), split: 0.60 };
    }

    function pointerToCanvas(e) {
      const rect = canvas.getBoundingClientRect();
      const sx = rect.width > 0 ? canvas.clientWidth / rect.width : 1;
      return { x: (e.clientX - rect.left) * (state.W / rect.width), y: (e.clientY - rect.top) * (state.H / rect.height) };
    }

    canvas.addEventListener('pointerdown', (e) => {
      const p = pointerToCanvas(e);
      const r = plotRect(state.W, state.H);
      const sr = stripRect(state.W, state.H);
      if (p.x >= r.x0 && p.x <= r.x1 && p.y >= r.y0 && p.y <= r.y1) {
        const I = ((p.x - r.x0) / (r.x1 - r.x0)) * state.Imax;
        state.mode = 'hold';
        btnMode.classList.add('active');
        state.holdI = Math.max(0, Math.min(state.Imax, I));
        state.dragging = true;
        state.simT = 0;
        state.trail = [];
        try {
          canvas.setPointerCapture(e.pointerId);
        } catch (err) {}
      } else if (p.x >= sr.x0 && p.x <= sr.x1 && p.y >= sr.y0 && p.y <= sr.y1) {
        state.pulseMs = 3.0; // brief depolarizing pulse in the strip
      }
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!state.dragging) return;
      const p = pointerToCanvas(e);
      const r = plotRect(state.W, state.H);
      const I = ((p.x - r.x0) / (r.x1 - r.x0)) * state.Imax;
      state.holdI = Math.max(0, Math.min(state.Imax, I));
    });
    const endDrag = () => {
      state.dragging = false;
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    // ---------- simulation ----------
    const SIM_DT = 0.02; // ms
    const STEPS_PER_FRAME = 90; // 1.8 ms simulated per animation frame
    const V_SAMPLE_MS = 0.6;
    const WINDOW_MS = 170;
    const PREFILL_MS = 190;

    function vth() {
      return core.threshold(state.model);
    }

    function currentAt() {
      if (state.mode === 'ramp') {
        const r = core.rampCurrent(state.simT, { Imax: state.Imax, half: state.rampHalf });
        state.rampUp = r.up;
        return r.I;
      }
      if (state.mode === 'rheo') {
        const r = core.rheoSweep(state.simT, state.model, {});
        state.rampUp = r.up;
        return r.I;
      }
      return state.holdI;
    }

    function resetSimulation(prefill) {
      state.simT = 0;
      state.V = P.V_RESET;
      state.spikes = [];
      state.vBuf = [];
      state.trail = [];
      state.trailClock = 0;
      state.rng = mulberry32(state.seed);
      state.gauss = makeGaussian(state.rng);
      if (!prefill) return;

      // Seed the visible window near rheobase, rather than making a new visit begin blank.
      const sampleEvery = Math.round(V_SAMPLE_MS / SIM_DT);
      const n = Math.round(PREFILL_MS / SIM_DT);
      for (let k = 0; k < n; k++) {
        let I = currentAt();
        if (state.pulseMs > 0) {
          I += 3.0;
          state.pulseMs -= SIM_DT;
        }
        let In = I;
        if (state.noiseI > 0) In += state.noiseI * state.gauss();
        state.lastI = I;
        state.V = core.step(state.model, state.V, In, SIM_DT);
        state.simT += SIM_DT;
        if (state.V >= vth()) {
          state.V = P.V_RESET;
          state.spikes.push(state.simT);
        }
        if (k % sampleEvery === 0) state.vBuf.push({ t: state.simT, V: state.V });
      }
      state.trailClock = 10;
    }

    function advance() {
      const th = vth();
      const vSampleEvery = Math.round(V_SAMPLE_MS / SIM_DT);
      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        let I = currentAt();
        if (state.pulseMs > 0) {
          I += 3.0;
          state.pulseMs -= SIM_DT;
        }
        let In = I;
        if (state.noiseI > 0) In += state.noiseI * state.gauss();
        state.lastI = I;
        state.V = core.step(state.model, state.V, In, SIM_DT);
        state.simT += SIM_DT;
        if (state.V >= th) {
          state.V = P.V_RESET;
          state.spikes.push(state.simT);
          if (state.spikes.length > 4000) state.spikes.shift();
        }
        if (k % vSampleEvery === 0) {
          state.vBuf.push({ t: state.simT, V: state.V });
        }
      }
      // trim buffers
      const tmin = state.simT - WINDOW_MS;
      while (state.vBuf.length && state.vBuf[0].t < tmin) state.vBuf.shift();
      while (state.spikes.length && state.spikes[0] < state.simT - 1600) state.spikes.shift();

      // measured rate sample
      state.trailClock += STEPS_PER_FRAME * SIM_DT;
      if (state.trailClock >= 10) {
        state.trailClock = 0;
        const recent = state.spikes.filter((s) => s > state.simT - 420);
        if (recent.length >= 2) {
          const k = Math.min(4, recent.length - 1);
          let sum = 0;
          for (let i = recent.length - k; i < recent.length; i++) sum += recent[i] - recent[i - 1];
          const isi = sum / k;
          const f = isi > 0 ? 1000 / isi : 0;
          state.trail.push({ I: state.lastI, f: f, up: state.rampUp, noise: state.noiseI > 0 });
          if (state.trail.length > 900) state.trail.shift();
        }
      }
    }

    function measuredRate() {
      const recent = state.spikes.filter((s) => s > state.simT - 1200);
      if (recent.length < 2) return null;
      const k = Math.min(4, recent.length - 1);
      let sum = 0;
      for (let i = recent.length - k; i < recent.length; i++) sum += recent[i] - recent[i - 1];
      return sum > 0 ? 1000 / (sum / k) : null;
    }

    // ---------- readout DOM ----------
    function updateReadouts() {
      const I = state.mode === 'hold' ? state.holdI : state.lastI;
      const Irh = core.rheobase(state.model);
      const thr = I > Irh ? core.theoryRate(state.model, I) : 0;
      const isi = state.model === 'elif' ? core.elifIsi(I, { n: 6000 }) : I > Irh ? 1000 / thr : Infinity;
      const sim = measuredRate();
      const set = (id, v) => {
        const el = $(id);
        if (el) el.textContent = v;
      };
      set('n08-val-I', I.toFixed(2) + ' ' + S.unitsNa);
      set('n08-val-T', Number.isFinite(isi) && isi > 0 ? isi.toFixed(1) + ' ' + S.unitsMs : '—');
      set('n08-val-f', thr.toFixed(1) + ' ' + S.unitsHz);

      if (I <= Irh && (state.noiseI <= 0 || sim === null)) {
        set('n08-val-fsim', '0.0 ' + S.unitsHz);
        set('n08-val-diff', '0.0 ' + S.unitsHz);
      } else if (sim !== null && sim > 0) {
        set('n08-val-fsim', sim.toFixed(1) + ' ' + S.unitsHz);
        set('n08-val-diff', (thr - sim).toFixed(1) + ' ' + S.unitsHz);
      } else {
        set('n08-val-fsim', S.roNotYet || 'not yet');
        set('n08-val-diff', '—');
      }

      // local gain df/dI by central difference
      const dI = 0.01;
      const fa = I - dI > Irh ? core.theoryRate(state.model, I - dI) : 0;
      const fb = core.theoryRate(state.model, I + dI);
      set('n08-val-gain', ((fb - fa) / (2 * dI)).toFixed(1) + ' ' + S.unitsHzPerNa);
    }

    // ---------- drawing helpers ----------
    function rr(c, x, y, w, h, r) {
      const rad = Math.min(r, w / 2, h / 2);
      c.beginPath();
      c.moveTo(x + rad, y);
      c.arcTo(x + w, y, x + w, y + h, rad);
      c.arcTo(x + w, y + h, x, y + h, rad);
      c.arcTo(x, y + h, x, y, rad);
      c.arcTo(x, y, x + w, y, rad);
      c.closePath();
    }

    function drawAxes(c, colors, rect, xMax, yMax, xTicks, yTicks, xLabel, yLabel, opts) {
      const { x0, x1, y0, y1 } = rect;
      c.save();
      // panel
      c.fillStyle = colors.paper;
      rr(c, x0 - 44, y0 - 26, x1 - x0 + 60, y1 - y0 + 44, 4);
      c.fill();
      // grid
      c.strokeStyle = colors.border;
      c.lineWidth = 0.6;
      c.setLineDash([2, 3]);
      for (const t of yTicks) {
        const y = y1 - (t / yMax) * (y1 - y0);
        c.beginPath();
        c.moveTo(x0, y);
        c.lineTo(x1, y);
        c.stroke();
      }
      for (const t of xTicks) {
        const x = x0 + (t / xMax) * (x1 - x0);
        c.beginPath();
        c.moveTo(x, y0);
        c.lineTo(x, y1);
        c.stroke();
      }
      c.setLineDash([]);
      // axes
      c.strokeStyle = colors.ink;
      c.lineWidth = 1.1;
      c.beginPath();
      c.moveTo(x0, y0);
      c.lineTo(x0, y1);
      c.lineTo(x1, y1);
      c.stroke();
      // ticks + labels
      c.fillStyle = colors.inkMuted;
      c.font = '10px ui-monospace, monospace';
      c.textAlign = 'right';
      c.textBaseline = 'middle';
      for (const t of yTicks) {
        const y = y1 - (t / yMax) * (y1 - y0);
        c.fillText(String(t), x0 - 5, y);
      }
      c.textAlign = 'center';
      c.textBaseline = 'top';
      for (const t of xTicks) {
        const x = x0 + (t / xMax) * (x1 - x0);
        c.fillText(String(t), x, y1 + 4);
      }
      // axis titles
      c.fillStyle = colors.inkMuted;
      c.font = '11px ui-serif, Georgia, serif';
      c.textAlign = 'center';
      c.textBaseline = 'top';
      c.fillText(xLabel, (x0 + x1) / 2, y1 + 16);
      c.save();
      c.translate(x0 - 40, (y0 + y1) / 2);
      c.rotate(-Math.PI / 2);
      c.textBaseline = 'top';
      c.fillText(yLabel, 0, 0);
      c.restore();
      c.restore();
    }

    function drawPolyline(c, pts, mapX, mapY, color, width, dash) {
      if (!pts || pts.length < 2) return;
      c.save();
      c.strokeStyle = color;
      c.lineWidth = width;
      if (dash) c.setLineDash(dash);
      c.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const x = mapX(pts[i].I);
        const y = mapY(pts[i].f);
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
      c.stroke();
      c.restore();
    }

    // ---------- main render ----------
    function render() {
      if (!g) return;
      const colors = ctx.colors();
      const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
      const rect = canvas.getBoundingClientRect();
      const W = Math.max(280, Math.round(rect.width || state.W));
      const H = Math.max(200, Math.round(rect.height || state.H));
      state.W = W;
      state.H = H;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
      }
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.scale(dpr, dpr);
      g.clearRect(0, 0, W, H);
      g.fillStyle = colors.paperCard;
      g.fillRect(0, 0, W, H);

      const small = W < 560;
      if (small && figureContainer && figureContainer.style) figureContainer.style.height = '500px';

      const model = state.model;
      const Irh = core.rheobase(model);
      const mcol = modelColor(colors);

      // ---- rate scale ----
      const tcurve = theoryCurve(model);
      let fmaxRaw = 1;
      for (const p of tcurve) if (p.f > fmaxRaw) fmaxRaw = p.f;
      if (state.noiseI > 0) {
        for (const p of noiseCurve(state.noiseI)) if (p.f > 0.5 && p.f < 1e5 && p.f > fmaxRaw) fmaxRaw = p.f;
      }
      const fmax = Math.ceil((fmaxRaw * 1.12) / 20) * 20;

      const r = plotRect(W, H);
      const mapI = (I) => r.x0 + (I / state.Imax) * (r.x1 - r.x0);
      const mapF = (f) => r.y1 - (Math.min(f, fmax) / fmax) * (r.y1 - r.y0);

      // ---- main plot ----
      drawAxes(
        g,
        colors,
        r,
        state.Imax,
        fmax,
        [1, 2, 3, 4, 5],
        [fmax * 0.25, fmax * 0.5, fmax * 0.75, fmax].map((v) => Math.round(v)),
        S.axisI,
        S.axisF
      );

      // The silent side is lightly tinted; the measured trail itself never paints the plot.
      g.save();
      g.globalAlpha = 0.045;
      g.fillStyle = colors.trace;
      g.fillRect(r.x0, r.y0, mapI(Irh) - r.x0, r.y1 - r.y0);
      g.restore();

      // highlight band for chapters 2/5
      if (state.highlight === 'rheo' || state.highlight === 'inset') {
        g.save();
        g.globalAlpha = 0.10;
        g.fillStyle = colors.volt;
        g.fillRect(mapI(Math.max(0, Irh - 0.35)), r.y0, mapI(Irh + 0.35) - mapI(Math.max(0, Irh - 0.35)), r.y1 - r.y0);
        g.restore();
      }

      // ribeobase vertical line
      g.save();
      g.strokeStyle = colors.volt;
      g.lineWidth = state.highlight === 'rheo' ? 2 : 1.1;
      g.setLineDash([5, 3]);
      g.beginPath();
      g.moveTo(mapI(Irh), r.y0);
      g.lineTo(mapI(Irh), r.y1);
      g.stroke();
      g.restore();
      g.fillStyle = colors.volt;
      g.font = '10px ui-monospace, monospace';
      g.textAlign = 'left';
      g.textBaseline = 'bottom';
      g.fillText(`${S.rheoTag} = ${Irh.toFixed(2)} nA`, mapI(Irh) + 4, r.y0 + 12);

      // theory ghost
      if (state.theoryOn) {
        drawPolyline(g, tcurve, mapI, mapF, colors.theory, 1.5, [6, 4]);
        // noise theory ghost (dash-dot)
        if (state.noiseI > 0) {
          const nc = noiseCurve(state.noiseI).filter((p) => Number.isFinite(p.f) && p.f < 1e5);
          drawPolyline(g, nc, mapI, mapF, colors.inh, 1.4, [7, 3, 1.5, 3]);
        }
      }

      // measured trail
      for (const p of state.trail) {
        const x = mapI(p.I);
        const y = mapF(p.f);
        if (y < r.y0 - 1 || y > r.y1 + 1) continue;
        if (p.noise) {
          g.fillStyle = colors.inh;
        } else {
          g.fillStyle = colors.trace;
        }
        g.beginPath();
        g.arc(x, y, 2.0, 0, Math.PI * 2);
        g.fill();
      }

      // current marker (violet) + guide
      const Icur = state.mode === 'hold' ? state.holdI : state.lastI;
      const xCur = mapI(Icur);
      g.save();
      g.strokeStyle = colors.mod;
      g.lineWidth = 1.6;
      if (state.dragging) {
        g.shadowColor = colors.mod;
        g.shadowBlur = 8;
      }
      g.beginPath();
      g.moveTo(xCur, r.y0);
      g.lineTo(xCur, r.y1);
      g.stroke();
      g.restore();
      // handle knob at the axis base
      g.fillStyle = colors.mod;
      g.beginPath();
      g.arc(xCur, r.y1, 4.5, 0, Math.PI * 2);
      g.fill();
      g.strokeStyle = colors.paper;
      g.lineWidth = 1.2;
      g.stroke();
      // small "grab" glyph
      g.fillStyle = colors.paper;
      g.font = '7px ui-sans-serif, sans-serif';
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText('↔', xCur, r.y1);
      // measured dot at current marker
      const simNow = measuredRate();
      if (simNow !== null && simNow > 0) {
        const yM = mapF(simNow);
        if (yM > r.y0 - 1 && yM < r.y1 + 1) {
          g.fillStyle = colors.mod;
          g.beginPath();
          g.arc(xCur, yM, 3.4, 0, Math.PI * 2);
          g.fill();
          g.strokeStyle = colors.paper;
          g.lineWidth = 1.1;
          g.stroke();
        }
      }
      // theory dot at current marker
      if (Icur > Irh) {
        const yT = mapF(core.theoryRate(model, Icur));
        g.strokeStyle = colors.theory;
        g.lineWidth = 1.2;
        g.beginPath();
        g.arc(xCur, yT, 3, 0, Math.PI * 2);
        g.stroke();
      }

      // current value label
      g.fillStyle = colors.mod;
      g.font = '11px ui-monospace, monospace';
      g.textAlign = xCur > r.x0 + (r.x1 - r.x0) * 0.6 ? 'right' : 'left';
      g.textBaseline = 'top';
      g.fillText(`I = ${Icur.toFixed(2)} nA`, xCur + (g.textAlign === 'right' ? -6 : 6), r.y0 + 26);
      if (Icur <= Irh) {
        g.fillStyle = colors.inkMuted;
        g.font = '11px ui-serif, Georgia, serif';
        g.fillText(S.silentTag, xCur + (g.textAlign === 'right' ? -6 : 6), r.y0 + 42);
      }

      // ramp direction arrow
      if (state.mode !== 'hold') {
        g.save();
        g.strokeStyle = colors.inkMuted;
        g.fillStyle = colors.inkMuted;
        g.lineWidth = 1.2;
        const ax = r.x1 - 40;
        const ay = r.y0 + 14;
        g.beginPath();
        g.moveTo(ax - 8, ay);
        g.lineTo(ax + 8, ay);
        g.stroke();
        const dir = state.rampUp ? 1 : -1;
        g.beginPath();
        g.moveTo(ax + dir * 8, ay);
        g.lineTo(ax + dir * 2, ay - 4);
        g.lineTo(ax + dir * 2, ay + 4);
        g.closePath();
        g.fill();
        g.font = '9px ui-monospace, monospace';
        g.textAlign = 'right';
        g.textBaseline = 'bottom';
        g.fillText(state.rampUp ? 'up' : 'down', ax - 14, ay + 4);
        g.restore();
      }

      // legend (top-left inside plot)
      drawLegend(g, colors, r, model, mcol);

      // ---- bottom strip ----
      const sr = stripRect(W, H);
      renderStrip(g, colors, sr, W, H);

      // ---- log-log inset: in the quiet upper-left of f-I space, never on the voltage trace ----
      const insetW = small ? Math.min(136, Math.round((r.x1 - r.x0) * 0.44)) : Math.round((r.x1 - r.x0) * 0.29);
      const insetY0 = r.y0 + (small ? 42 : 44);
      const insetMaxH = r.y1 - insetY0 - (small ? 8 : 14);
      const insetH = small ? Math.min(72, insetMaxH) : 122;
      const inset = {
        x0: r.x0 + 8,
        x1: r.x0 + 8 + insetW,
        y0: insetY0,
        y1: insetY0 + insetH
      };
      renderInset(g, colors, inset, small);

      updateReadouts();
    }

    function drawLegend(c, colors, r, model, mcol) {
      const compact = r.x1 - r.x0 < 380;
      const items = [
        { color: colors.theory, dash: true, label: compact ? 'theory' : S.legendTheory },
        { color: colors.trace, dot: true, label: compact ? 'trail' : S.legendMeas },
        { color: colors.mod, line: true, label: compact ? 'marker' : S.legendNow }
      ];
      if (state.noiseI > 0) items.push({ color: colors.inh, dashdot: true, label: compact ? 'noise' : S.legendNoise });

      c.save();
      c.font = (compact ? '9px' : '10px') + ' ui-sans-serif, sans-serif';
      c.textBaseline = 'middle';
      c.textAlign = 'left';

      const pad = compact ? 8 : 12;
      const markW = 16;
      const widths = items.map((it) => markW + 5 + c.measureText(it.label).width + pad);
      const total = widths.reduce((a, b) => a + b, 0);

      // Position above the plot border (y0 - 13), so it NEVER overlaps the curves
      let x = Math.max(r.x0 + 6, r.x1 - total - 2);
      let y = r.y0 - 13;

      for (const it of items) {
        if (it.dash) {
          c.strokeStyle = it.color;
          c.lineWidth = 1.4;
          c.setLineDash([5, 3]);
          c.beginPath();
          c.moveTo(x, y);
          c.lineTo(x + markW, y);
          c.stroke();
          c.setLineDash([]);
        } else if (it.dashdot) {
          c.strokeStyle = it.color;
          c.lineWidth = 1.4;
          c.setLineDash([6, 2, 1.5, 2]);
          c.beginPath();
          c.moveTo(x, y);
          c.lineTo(x + markW, y);
          c.stroke();
          c.setLineDash([]);
        } else if (it.dot) {
          c.fillStyle = it.color;
          c.beginPath();
          c.arc(x + markW / 2, y, 2.4, 0, Math.PI * 2);
          c.fill();
        } else {
          c.strokeStyle = it.color;
          c.lineWidth = 1.8;
          c.beginPath();
          c.moveTo(x + markW / 2, y - 4);
          c.lineTo(x + markW / 2, y + 4);
          c.stroke();
        }
        c.fillStyle = colors.inkMuted;
        c.fillText(it.label, x + markW + 5, y);
        x += markW + 5 + c.measureText(it.label).width + pad;
      }
      c.restore();
    }

    function renderStrip(c, colors, sr, W, H) {
      const th = vth();
      const hi = th + 6;
      const lo = P.V_REST - 14;
      // panel
      c.save();
      c.fillStyle = colors.paper;
      rr(c, sr.x0 - 44, sr.y0 - 22, W - (sr.x0 - 44) - 16, sr.y1 - sr.y0 + 34, 4);
      c.fill();
      c.restore();

      const titleY = sr.y0 - 8;
      c.fillStyle = colors.inkMuted;
      c.font = '10px ui-serif, Georgia, serif';
      c.textAlign = 'left';
      c.textBaseline = 'bottom';
      c.fillText(S.stripTitle, sr.x0 - 38, titleY);

      const mapV = (v) => sr.y1 - ((v - lo) / (hi - lo)) * (sr.y1 - sr.y0);

      // threshold + reset lines
      c.save();
      c.setLineDash([4, 3]);
      c.lineWidth = 1;
      c.strokeStyle = colors.inkMuted;
      c.beginPath();
      c.moveTo(sr.x0, mapV(th));
      c.lineTo(sr.x1, mapV(th));
      c.stroke();
      c.beginPath();
      c.moveTo(sr.x0, mapV(P.V_RESET));
      c.lineTo(sr.x1, mapV(P.V_RESET));
      c.stroke();
      c.restore();
      c.fillStyle = colors.inkMuted;
      c.font = '9px ui-monospace, monospace';
      c.textAlign = 'left';
      c.textBaseline = 'bottom';
      c.fillText(`V_th ${th}`, sr.x0 + 3, mapV(th) - 1);
      c.fillText(`V_reset ${P.V_RESET}`, sr.x0 + 3, mapV(P.V_RESET) - 1);

      // voltage trace
      const tmin = state.simT - WINDOW_MS;
      if (state.vBuf.length > 1) {
        c.save();
        c.strokeStyle = colors.volt;
        c.lineWidth = 1.4;
        c.beginPath();
        let started = false;
        for (const s of state.vBuf) {
          if (s.t < tmin) continue;
          const x = sr.x0 + ((s.t - tmin) / WINDOW_MS) * (sr.x1 - sr.x0);
          const y = mapV(s.V);
          if (!started) {
            c.moveTo(x, y);
            started = true;
          } else {
            c.lineTo(x, y);
          }
        }
        c.stroke();
        c.restore();
      }

      // spike ticks in the strip top and a raster row
      c.save();
      c.strokeStyle = colors.volt;
      c.lineWidth = 1.1;
      for (const t of state.spikes) {
        if (t < tmin) continue;
        const x = sr.x0 + ((t - tmin) / WINDOW_MS) * (sr.x1 - sr.x0);
        c.beginPath();
        c.moveTo(x, sr.y0);
        c.lineTo(x, sr.y0 + 7);
        c.stroke();
      }
      c.restore();

      // noise shading when active
      if (state.noiseI > 0) {
        c.save();
        c.globalAlpha = 0.12;
        c.fillStyle = colors.inh;
        c.fillRect(sr.x0, sr.y0, sr.x1 - sr.x0, sr.y1 - sr.y0);
        c.restore();
        c.fillStyle = colors.inh;
        c.font = '10px ui-monospace, monospace';
        c.textAlign = 'right';
        c.textBaseline = 'bottom';
        c.fillText(`noise σ_I = ${state.noiseI.toFixed(2)} nA`, sr.x1 - 4, sr.y0 + 10);
      }

      // pulse indicator
      if (state.pulseMs > 0) {
        c.fillStyle = colors.mod;
        c.font = '10px ui-monospace, monospace';
        c.textAlign = 'right';
        c.textBaseline = 'top';
        c.fillText('+ pulse', sr.x1 - 4, sr.y0 + 4);
      }

      // time axis ticks
      c.fillStyle = colors.inkMuted;
      c.font = '9px ui-monospace, monospace';
      c.textAlign = 'center';
      c.textBaseline = 'top';
      c.fillText(S.axisTs, (sr.x0 + sr.x1) / 2, sr.y1 + 7);
      c.textAlign = 'right';
      c.fillText('now', sr.x1, sr.y1 + 7);
    }

    function renderInset(c, colors, box, small) {
      const { x0, x1, y0, y1 } = box;
      c.save();
      c.fillStyle = colors.paperElevated || colors.paper;
      rr(c, x0, y0, x1 - x0, y1 - y0, 3);
      c.fill();
      c.strokeStyle = state.highlight === 'inset' ? colors.mod : colors.border;
      c.lineWidth = state.highlight === 'inset' ? 1.6 : 0.8;
      c.stroke();
      c.clip();

      const padL = small ? 18 : 36;
      const padR = 6;
      const padT = small ? 14 : 20;
      const padB = small ? 10 : 22;
      const px0 = x0 + padL;
      const px1 = x1 - padR;
      const py0 = y0 + padT;
      const py1 = y1 - padB;

      // gather points for both models
      const lifPts = onsetCurve('lif');
      const elifPts = onsetCurve('elif');
      const all = lifPts.concat(elifPts);
      let lgxmin = 1e9;
      let lgxmax = -1e9;
      let lgymin = 1e9;
      let lgymax = -1e9;
      for (const p of all) {
        const lx = Math.log10(p.dI);
        const ly = Math.log10(p.f);
        if (lx < lgxmin) lgxmin = lx;
        if (lx > lgxmax) lgxmax = lx;
        if (ly < lgymin) lgymin = ly;
        if (ly > lgymax) lgymax = ly;
      }
      lgxmin = Math.max(lgxmin, -3);
      lgxmax = Math.min(lgxmax, 0.1);
      const mapX = (dI) => px0 + ((Math.log10(dI) - lgxmin) / (lgxmax - lgxmin)) * (px1 - px0);
      const mapY = (f) => py1 - ((Math.log10(f) - lgymin) / (lgymax - lgymin)) * (py1 - py0);

      // reference slope 1/2 line (drawn through the ELIF region)
      c.save();
      c.strokeStyle = colors.inkMuted;
      c.lineWidth = 1;
      c.setLineDash([3, 3]);
      const refx = -2.2;
      const refy = -0.4;
      const slope = 0.5;
      const xA = refx - 0.7;
      const yA = refy - 0.7 * slope;
      const xB = refx + 0.7;
      const yB = refy + 0.7 * slope;
      c.beginPath();
      c.moveTo(mapX(Math.pow(10, xA)), mapY(Math.pow(10, yA)));
      c.lineTo(mapX(Math.pow(10, xB)), mapY(Math.pow(10, yB)));
      c.stroke();
      c.restore();
      // slope label
      c.fillStyle = colors.inkMuted;
      c.font = '9px ui-monospace, monospace';
      c.textAlign = 'left';
      c.textBaseline = 'top';
      c.fillText(S.slopeHalf, mapX(Math.pow(10, xB)) - 44, mapY(Math.pow(10, yB)) - 12);

      // curves
      const drawOnset = (pts, color, w) => {
        c.strokeStyle = color;
        c.lineWidth = w;
        c.beginPath();
        let first = true;
        for (const p of pts) {
          const x = mapX(p.dI);
          const y = mapY(p.f);
          if (!first) c.lineTo(x, y);
          else {
            c.moveTo(x, y);
            first = false;
          }
        }
        c.stroke();
      };
      drawOnset(lifPts, colors.volt, 1.8);
      drawOnset(elifPts, colors.exc, 1.8);

      // axis box
      c.strokeStyle = colors.border;
      c.lineWidth = 0.7;
      c.beginPath();
      c.moveTo(px0, py0);
      c.lineTo(px0, py1);
      c.lineTo(px1, py1);
      c.stroke();

      // labels
      c.fillStyle = colors.inkMuted;
      c.font = small ? '9px ui-monospace, monospace' : '10px ui-monospace, monospace';
      c.textAlign = 'left';
      c.textBaseline = 'top';
      if (!small) c.fillText(S.insetTitle + '  ·  ' + S.insetX, x0 + 8, y0 + 5);
      else c.fillText(S.insetTitle, x0 + 6, y0 + 4);
      c.textAlign = 'center';
      if (!small) c.fillText(S.insetX, (px0 + px1) / 2, y1 - 14);
      c.save();
      c.translate(x0 + 8, (py0 + py1) / 2);
      c.rotate(-Math.PI / 2);
      c.textAlign = 'center';
      c.textBaseline = 'top';
      c.fillText(S.insetY, 0, 0);
      c.restore();

      // legend for models inside the inset
      c.font = '9px ui-monospace, monospace';
      c.textAlign = 'left';
      c.textBaseline = 'top';
      const ly = py0 + 3;
      c.fillStyle = colors.exc;
      c.fillText('ELIF √', px0 + 4, ly);
      c.fillStyle = colors.volt;
      c.fillText('LIF log', px0 + 52, ly);

      c.restore();
    }

    // ---------- expose current state for tests/debug (harmless) ----------
    rootEl.__n08 = state;

    // ---------- init ----------
    setModel('elif');
    renderDots();
    applyChapter(1);
    render();

    ctx.onFrame(() => {
      advance();
      render();
    });
  }

  // Export for node tests
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core: core, S: S };
  }

  // Register in the browser Atlas shell
  const AtlasRef = typeof Atlas !== 'undefined' ? Atlas : root && root.Atlas;
  if (AtlasRef && AtlasRef.Plate) {
    AtlasRef.Plate.register({
      id: 'N08',
      title: { en: S.title },
      source: S.source,
      mount: mount
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
