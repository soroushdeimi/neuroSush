/**
 * PLATE N05 · Hodgkin-Huxley: how a neuron fires
 * Live JS simulation of the squid giant axon Hodgkin-Huxley model (1952).
 * Units: ms, mV, µA/cm², mS/cm², µF/cm².
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
    plateId: 'N05',
    title: 'Hodgkin-Huxley: how a neuron fires',
    titleFa: 'هادگکین–هاکسلی: مقطع غشا، کانال‌های یونی سدیم و پتاسیم، دروازه‌های m و h و n',
    source: 'live',
    libraryMirror: 'neurosush.neurons.hodgkin_huxley (when released) — JS port; the library model is validated against the same values.',

    // Marginalia
    eqTitle: 'Governing Equation',
    eqCm: 'Cm dV/dt',
    eqIext: 'I_ext',
    eqINa: 'gNa·m³h·(V - ENa)',
    eqIK: 'gK·n⁴·(V - EK)',
    eqIL: 'gL·(V - EL)',
    eqMinus: ' - ',
    eqEqual: ' = ',

    readoutV: 'Membrane Potential V',
    readoutIext: 'Stimulus Current I_ext',
    readoutINa: 'Sodium Current I_Na',
    readoutIK: 'Potassium Current I_K',
    readoutIL: 'Leak Current I_L',
    readoutItot: 'Net Current C·dV/dt',
    readoutGates: 'Gating Probabilities',
    readoutTheoryDiff: 'Theory Distance |x - x_inf|',

    unitsMv: 'mV',
    unitsCurrent: 'µA/cm²',
    unitsCond: 'mS/cm²',
    unitsMs: 'ms',

    // Glossary
    glossaryTitle: 'Glossary & Kinetics',
    glossaryM: 'm gate: Fast Na⁺ activation; 3 subunits (P_open = m³). Opens sharply on depolarization.',
    glossaryH: 'h gate: Slow Na⁺ inactivation; ball-and-chain plug (P_open = h). Shuts pore at positive voltages.',
    glossaryN: 'n gate: Delayed-rectifier K⁺ activation; 4 subunits (P_open = n⁴). Repolarizes the spike.',
    glossaryTTX: 'TTX (Tetrodotoxin): Selective Na⁺ channel blocker; abolishes regenerative spikes.',
    glossaryTEA: 'TEA (Tetraethylammonium): Selective K⁺ channel blocker; prevents repolarization.',
    glossaryDriving: 'Driving force: (V - E_ion) sets the direction and magnitude of passive ion flux.',

    // Hero figure labels
    outside: 'OUTSIDE (Extracellular fluid: high Na⁺, [Na⁺] = 145 mM)',
    inside: 'INSIDE (Intracellular cytosol: high K⁺, [K⁺] = 140 mM)',
    lipidBilayer: 'Lipid Bilayer Patch',
    naChannel: 'Na⁺ Channel (m³h)',
    kChannel: 'K⁺ Channel (n⁴)',
    leakChannel: 'Leak',
    batteryTitle: 'Membrane Voltmeter',
    electrodeActive: 'Electrode Stimulating',
    pokePrompt: 'Click & hold membrane to inject current',

    // Controls
    btnStimulate: '⚡ Stimulate',
    btnStimulating: '⚡ Injecting...',
    labelCurrentSlider: 'I_ext',
    modeCurrentClamp: 'Current Clamp (Normal)',
    modeVoltageClamp: 'Voltage Clamp (H&H)',
    btnClampStep: 'Trigger Clamp Step',
    clampStepLabel: 'Step to',
    toggleTTX: 'TTX (Block Na⁺)',
    toggleTEA: 'TEA (Block K⁺)',
    toggleTheory: 'Theory [T]',
    theoryOn: 'Theory Ghost: ON',
    theoryOff: 'Theory Ghost: OFF',
    traceVoltage: 'Membrane Potential V(t)',
    traceConductance: 'Conductances gNa(t) & gK(t)',
    traceGates: 'Gating Variables m, h, n',
    traceClampCurrent: 'Clamp Current I_clamp(t)',
    steadyStateCurves: 'Steady-State Curves x_inf(V)',

    // Chapters
    chapterLabel: 'CHAPTER',
    ofChapters: 'OF 5',
    btnPrev: '← Back',
    btnNext: 'Next →',
    btnChallenge: 'Challenge [P]',
    ch1Title: 'Resting Equilibrium',
    ch1Body: 'At rest (-65 mV), K⁺ leak balances the tiny Na⁺ window current. Gating particles are quiet: m ≈ 0.05, h ≈ 0.60, n ≈ 0.32. The membrane sits close to EK (-77 mV).',
    ch2Title: 'Depolarization & Explosive Na⁺ Influx',
    ch2Body: 'Stimulus pushes V above threshold. Fast 3-subunit m gates swing open wide, allowing Na⁺ ions to rush in down their electrochemical gradient (positive feedback).',
    ch3Title: 'Peak & Inactivation Ball Docking',
    ch3Body: 'At the peak (+30 to +40 mV), the slow h inactivation ball-and-chain docks into the Na⁺ pore, shutting off influx. Delayed 4-subunit n gates open, releasing outward K⁺.',
    ch4Title: 'Undershoot & Slow Recovery',
    ch4Body: 'Delayed-rectifier n gates close slowly, leaving K⁺ conductance elevated even after V drops below rest. This hyperpolarizes the membrane toward EK (-77 mV).',
    ch5Title: 'Refractory State',
    ch5Body: 'The h inactivation balls recover slowly at negative potentials. While h is docked, a second stimulus cannot open Na⁺ pores, enforcing the refractory period.',

    // Predict-then-reveal
    predictTitle: 'Predict-Then-Reveal',
    predictQuestion: 'If we block the sodium channels with TTX, what happens to a strong current stimulus?',
    predictOptA: 'A) The neuron fires a broader action potential',
    predictOptB: 'B) The neuron fires a smaller, shorter action potential',
    predictOptC: 'C) No action potential fires; only a small passive bump occurs',
    predictSubmit: 'Commit Prediction',
    predictRevealedTitle: 'Experimental Reveal',
    predictExplanation: 'Correct! Regenerative spikes require fast Na⁺ influx through voltage-gated channels. With TTX blocking Na⁺ pores, only passive RC charging and outward leak remain.',
    predictTryBtn: 'Run TTX Experiment Now',
  };

  // ---- core: pure math, no DOM, exported for node tests ----
  const core = {
    C_M: 1.0,       // µF/cm²
    G_NA: 120.0,    // mS/cm²
    G_K: 36.0,      // mS/cm²
    G_L: 0.3,       // mS/cm²
    E_NA: 50.0,     // mV
    E_K: -77.0,     // mV
    E_L: -54.387,   // mV
    REST_V: -65.0,  // mV
    REST_M: 0.052932485,
    REST_H: 0.596120753,
    REST_N: 0.317676914,

    // Rate equations with removable singularities
    alpha_m(V) {
      const u = (V + 40) / 10;
      if (Math.abs(u) < 1e-6) return 1 + 0.5 * u + (u ** 2) / 12;
      return -u / Math.expm1(-u);
    },
    beta_m(V) {
      return 4 * Math.exp(-(V + 65) / 18);
    },
    alpha_h(V) {
      return 0.07 * Math.exp(-(V + 65) / 20);
    },
    beta_h(V) {
      return 1 / (1 + Math.exp(-(V + 35) / 10));
    },
    alpha_n(V) {
      const u = (V + 55) / 10;
      if (Math.abs(u) < 1e-6) return 0.1 * (1 + 0.5 * u + (u ** 2) / 12);
      return 0.1 * (-u / Math.expm1(-u));
    },
    beta_n(V) {
      return 0.125 * Math.exp(-(V + 65) / 80);
    },

    m_inf(V) {
      const a = this.alpha_m(V);
      const b = this.beta_m(V);
      return a / (a + b);
    },
    tau_m(V) {
      return 1 / (this.alpha_m(V) + this.beta_m(V));
    },
    h_inf(V) {
      const a = this.alpha_h(V);
      const b = this.beta_h(V);
      return a / (a + b);
    },
    tau_h(V) {
      return 1 / (this.alpha_h(V) + this.beta_h(V));
    },
    n_inf(V) {
      const a = this.alpha_n(V);
      const b = this.beta_n(V);
      return a / (a + b);
    },
    tau_n(V) {
      return 1 / (this.alpha_n(V) + this.beta_n(V));
    },

    derivatives(state, I_ext = 0, opts = {}) {
      const { V, m, h, n } = state;
      const gNa = opts.ttx ? 0 : this.G_NA;
      const gK = opts.tea ? 0 : this.G_K;
      const gL = this.G_L;

      const I_Na = gNa * (m * m * m) * h * (V - this.E_NA);
      const I_K = gK * (n * n * n * n) * (V - this.E_K);
      const I_L = gL * (V - this.E_L);

      const dV = (I_ext - I_Na - I_K - I_L) / this.C_M;
      const dm = this.alpha_m(V) * (1 - m) - this.beta_m(V) * m;
      const dh = this.alpha_h(V) * (1 - h) - this.beta_h(V) * h;
      const dn = this.alpha_n(V) * (1 - n) - this.beta_n(V) * n;

      return { dV, dm, dh, dn, I_Na, I_K, I_L, I_ext };
    },

    rk4Step(state, I_ext, dt, opts = {}) {
      const isClamp = Boolean(opts.clamp);
      const clampV = opts.clampV !== undefined ? opts.clampV : state.V;

      const evalDeriv = (s) => {
        const cur = isClamp ? { ...s, V: clampV } : s;
        const d = this.derivatives(cur, I_ext, opts);
        if (isClamp) d.dV = 0;
        return d;
      };

      const k1 = evalDeriv(state);

      const s2 = {
        V: isClamp ? clampV : state.V + 0.5 * dt * k1.dV,
        m: Math.max(0, Math.min(1, state.m + 0.5 * dt * k1.dm)),
        h: Math.max(0, Math.min(1, state.h + 0.5 * dt * k1.dh)),
        n: Math.max(0, Math.min(1, state.n + 0.5 * dt * k1.dn)),
      };
      const k2 = evalDeriv(s2);

      const s3 = {
        V: isClamp ? clampV : state.V + 0.5 * dt * k2.dV,
        m: Math.max(0, Math.min(1, state.m + 0.5 * dt * k2.dm)),
        h: Math.max(0, Math.min(1, state.h + 0.5 * dt * k2.dh)),
        n: Math.max(0, Math.min(1, state.n + 0.5 * dt * k2.dn)),
      };
      const k3 = evalDeriv(s3);

      const s4 = {
        V: isClamp ? clampV : state.V + dt * k3.dV,
        m: Math.max(0, Math.min(1, state.m + dt * k3.dm)),
        h: Math.max(0, Math.min(1, state.h + dt * k3.dh)),
        n: Math.max(0, Math.min(1, state.n + dt * k3.dn)),
      };
      const k4 = evalDeriv(s4);

      const nextV = isClamp ? clampV : state.V + (dt / 6) * (k1.dV + 2 * k2.dV + 2 * k3.dV + k4.dV);
      const nextM = Math.max(0, Math.min(1, state.m + (dt / 6) * (k1.dm + 2 * k2.dm + 2 * k3.dm + k4.dm)));
      const nextH = Math.max(0, Math.min(1, state.h + (dt / 6) * (k1.dh + 2 * k2.dh + 2 * k3.dh + k4.dh)));
      const nextN = Math.max(0, Math.min(1, state.n + (dt / 6) * (k1.dn + 2 * k2.dn + 2 * k3.dn + k4.dn)));

      const finalDerivs = this.derivatives({ V: nextV, m: nextM, h: nextH, n: nextN }, I_ext, opts);

      return {
        V: nextV,
        m: nextM,
        h: nextH,
        n: nextN,
        I_Na: finalDerivs.I_Na,
        I_K: finalDerivs.I_K,
        I_L: finalDerivs.I_L,
        I_ext,
      };
    },

    simulate(durationMs, dt = 0.025, I_ext = 0, initial = null, opts = {}) {
      let state = initial
        ? { ...initial }
        : { V: this.REST_V, m: this.REST_M, h: this.REST_H, n: this.REST_N };
      const steps = Math.round(durationMs / dt);
      const history = [];
      for (let i = 0; i < steps; i++) {
        const t = i * dt;
        const curI = typeof I_ext === 'function' ? I_ext(t) : I_ext;
        state = this.rk4Step(state, curI, dt, opts);
        history.push({ t, ...state });
      }
      return { state, history };
    },

    createRng(seed = 42) {
      let a = (seed ^ 0x12345678) >>> 0;
      return function () {
        let t = (a += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    },

    initChannels(numNa = 10, numK = 6, numLeak = 2, rng = null) {
      const rand = rng || this.createRng(101);
      const channels = [];
      const total = numNa + numK + numLeak;

      for (let i = 0; i < numNa; i++) {
        channels.push({
          type: 'na',
          id: `na_${i}`,
          xRatio: (i + 0.5 + (rand() - 0.5) * 0.35) / total,
          mOpen: false,
          hOpen: true,
          isOpen: false,
          mSubunits: [false, false, false],
          hSwingPhase: rand() * Math.PI * 2,
        });
      }
      for (let i = 0; i < numK; i++) {
        channels.push({
          type: 'k',
          id: `k_${i}`,
          xRatio: (numNa + i + 0.5 + (rand() - 0.5) * 0.35) / total,
          nOpen: false,
          isOpen: false,
          nSubunits: [false, false, false, false],
        });
      }
      for (let i = 0; i < numLeak; i++) {
        channels.push({
          type: 'leak',
          id: `leak_${i}`,
          xRatio: (numNa + numK + i + 0.5 + (rand() - 0.5) * 0.35) / total,
          isOpen: true,
        });
      }

      channels.sort((a, b) => a.xRatio - b.xRatio);
      return channels;
    },

    sampleChannels(channels, m, h, n, rng) {
      const pM = Math.max(0, Math.min(1, m * m * m));
      const pH = Math.max(0, Math.min(1, h));
      const pN = Math.max(0, Math.min(1, n * n * n * n));

      for (const ch of channels) {
        if (ch.type === 'na') {
          ch.mOpen = rng() < pM;
          ch.hOpen = rng() < pH;
          ch.isOpen = ch.mOpen && ch.hOpen;
          ch.mSubunits[0] = rng() < m;
          ch.mSubunits[1] = rng() < m;
          ch.mSubunits[2] = rng() < m;
        } else if (ch.type === 'k') {
          ch.nOpen = rng() < pN;
          ch.isOpen = ch.nOpen;
          ch.nSubunits[0] = rng() < n;
          ch.nSubunits[1] = rng() < n;
          ch.nSubunits[2] = rng() < n;
          ch.nSubunits[3] = rng() < n;
        } else if (ch.type === 'leak') {
          ch.isOpen = true;
        }
      }
      return channels;
    },

    initIons(numNa = 85, numK = 85, rng = null) {
      const rand = rng || this.createRng(777);
      const ions = [];

      for (let i = 0; i < numNa; i++) {
        const isOutside = rand() < 0.90;
        ions.push({
          type: 'na',
          x: rand(),
          y: isOutside ? 0.05 + rand() * 0.33 : 0.62 + rand() * 0.32,
          vx: (rand() - 0.5) * 0.003,
          vy: (rand() - 0.5) * 0.003,
          inPore: false,
        });
      }

      for (let i = 0; i < numK; i++) {
        const isInside = rand() < 0.90;
        ions.push({
          type: 'k',
          x: rand(),
          y: isInside ? 0.62 + rand() * 0.32 : 0.05 + rand() * 0.33,
          vx: (rand() - 0.5) * 0.003,
          vy: (rand() - 0.5) * 0.003,
          inPore: false,
        });
      }

      return ions;
    },
  };

  // ---- UI / Mount (atlas shell integration) ----
  function mount(rootEl, ctx) {
    if (!ctx || !ctx.colors) return;

    // Prefill 200 history steps at resting equilibrium
    const prefillLength = 200;
    const initialVHistory = new Array(prefillLength).fill(core.REST_V);
    const initialGNa = core.G_NA * (core.REST_M ** 3) * core.REST_H;
    const initialGK = core.G_K * (core.REST_N ** 4);
    const initialGNaHistory = new Array(prefillLength).fill(initialGNa);
    const initialGKHistory = new Array(prefillLength).fill(initialGK);
    const initialMHistory = new Array(prefillLength).fill(core.REST_M);
    const initialHHistory = new Array(prefillLength).fill(core.REST_H);
    const initialNHistory = new Array(prefillLength).fill(core.REST_N);
    const initialClampHistory = new Array(prefillLength).fill(0);

    const state = {
      sim: {
        V: core.REST_V,
        m: core.REST_M,
        h: core.REST_H,
        n: core.REST_N,
        I_Na: 0,
        I_K: 0,
        I_L: 0,
        I_ext: 0,
      },
      I_ext_slider: 0,
      pulseRemainingMs: 0,
      isPoking: false,
      voltageClamp: false,
      clampHoldV: -65,
      clampStepV: 0,
      clampTimeMs: 0,
      ttx: false,
      tea: false,
      theoryActive: false,
      hoverTerm: null,
      chapter: 1,
      predictAnswer: null,
      predictRevealed: false,
      rng: core.createRng(999),
      channels: core.initChannels(10, 6, 2),
      ions: core.initIons(85, 85),
      voltageHistory: initialVHistory,
      gNaHistory: initialGNaHistory,
      gKHistory: initialGKHistory,
      mHistory: initialMHistory,
      hHistory: initialHHistory,
      nHistory: initialNHistory,
      clampHistory: initialClampHistory,
      spikes: [],
      maxHistory: 200,
    };

    const figureContainer = ctx.figure || rootEl;
    const marginContainer = ctx.margin || document.getElementById('atlas-margin');
    const figureSection = document.getElementById('atlas-figure') || figureContainer.parentElement || figureContainer;

    if (ctx.clock) {
      ctx.clock.stepsPerSecond = 60;
    }

    // Set canvas container height to fit in window
    if (figureContainer && figureContainer.style) {
      figureContainer.style.aspectRatio = 'auto';
      figureContainer.style.height = (typeof window !== 'undefined' && window.innerWidth < 560) ? '460px' : '420px';
      figureContainer.style.maxHeight = '520px';
    }

    // 1. Build Marginalia (Equation, Live Terms, Glossary)
    if (marginContainer) {
      marginContainer.innerHTML = `
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.eqTitle}</div>
          <div class="atlas-equation-box" id="n05-eq-box">
            <span class="n05-term" data-term="dV" style="color:var(--volt); cursor:pointer;">${S.eqCm}</span>
            <span>${S.eqEqual}</span>
            <span class="n05-term" data-term="Iext" style="cursor:pointer;">${S.eqIext}</span>
            <span>${S.eqMinus}</span>
            <span class="n05-term" data-term="INa" style="color:var(--exc); font-weight:600; cursor:pointer;">${S.eqINa}</span>
            <span>${S.eqMinus}</span>
            <span class="n05-term" data-term="IK" style="color:var(--inh); font-weight:600; cursor:pointer;">${S.eqIK}</span>
            <span>${S.eqMinus}</span>
            <span class="n05-term" data-term="IL" style="cursor:pointer;">${S.eqIL}</span>
          </div>
        </div>

        <div class="atlas-margin-section" id="n05-live-readouts">
          <div class="atlas-margin-label">Live Membrane State</div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif">${S.readoutV}:</span>
            <span class="atlas-margin-value" id="val-V" style="color:var(--volt); font-size:18px;">-65.0 mV</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif">I_ext:</span>
            <span class="atlas-margin-value" id="val-Iext">0.00 µA/cm²</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif" style="color:var(--exc); font-weight:600;">I_Na:</span>
            <span class="atlas-margin-value" id="val-INa" style="color:var(--exc);">0.00 µA/cm²</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif" style="color:var(--inh); font-weight:600;">I_K:</span>
            <span class="atlas-margin-value" id="val-IK" style="color:var(--inh);">0.00 µA/cm²</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif">I_Leak:</span>
            <span class="atlas-margin-value" id="val-IL">0.00 µA/cm²</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; border-top:1px dashed var(--border); padding-top:4px;">
            <span class="atlas-serif">C·dV/dt:</span>
            <span class="atlas-margin-value" id="val-Itot">0.00 µA/cm²</span>
          </div>
        </div>

        <div class="atlas-margin-section" id="n05-gates-readout">
          <div class="atlas-margin-label">${S.readoutGates}</div>
          <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:6px; text-align:center;">
            <div>
              <div class="atlas-mono" style="color:var(--exc); font-weight:bold;">m</div>
              <div class="atlas-mono" id="val-m">0.053</div>
            </div>
            <div>
              <div class="atlas-mono" style="color:var(--mod); font-weight:bold;">h</div>
              <div class="atlas-mono" id="val-h">0.596</div>
            </div>
            <div>
              <div class="atlas-mono" style="color:var(--inh); font-weight:bold;">n</div>
              <div class="atlas-mono" id="val-n">0.318</div>
            </div>
          </div>
          <div id="theory-diff-container" style="display:none; margin-top:6px; border-top:1px dashed var(--border); padding-top:4px; font-size:11px;">
            <div class="atlas-margin-label" style="font-size:10px;">${S.readoutTheoryDiff}</div>
            <div id="val-theory-diff" class="atlas-mono" style="color:var(--theory); font-size:11px;">|m - m_inf| = 0.000</div>
          </div>
        </div>

        <div class="atlas-margin-section" style="font-size:11px; line-height:1.4;">
          <div class="atlas-margin-label">${S.glossaryTitle}</div>
          <p style="margin:2px 0 4px 0;">• ${S.glossaryM}</p>
          <p style="margin:0 0 4px 0;">• ${S.glossaryH}</p>
          <p style="margin:0 0 4px 0;">• ${S.glossaryN}</p>
          <p style="margin:0 0 4px 0;">• ${S.glossaryTTX}</p>
          <p style="margin:0 0 2px 0;">• ${S.glossaryTEA}</p>
        </div>
      `;

      marginContainer.querySelectorAll('.n05-term').forEach((el) => {
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

    // 2. Mount Chapter Banner AT TOP of Figure Section (before canvas container)
    let chapterBar = document.getElementById('n05-chapter-bar');
    if (!chapterBar) {
      chapterBar = document.createElement('div');
      chapterBar.id = 'n05-chapter-bar';
      chapterBar.style.cssText =
        'background:var(--paper-card); border:1px solid var(--border); border-radius:var(--radius); padding:8px 12px; margin-bottom:6px; display:flex; flex-direction:column; gap:4px;';
      figureSection.insertBefore(chapterBar, figureContainer);
    }

    chapterBar.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
          <span class="atlas-badge" id="n05-ch-badge" style="font-size:10px; padding:1px 5px;">${S.chapterLabel} 1 ${S.ofChapters}</span>
          <span class="atlas-serif" id="n05-ch-title" style="font-size:13px; font-weight:600;">${S.ch1Title}</span>
          <div id="n05-ch-dots" style="display:flex; gap:3px;">
            <span class="n05-dot" data-ch="1" style="cursor:pointer; font-size:11px;">●</span>
            <span class="n05-dot" data-ch="2" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
            <span class="n05-dot" data-ch="3" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
            <span class="n05-dot" data-ch="4" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
            <span class="n05-dot" data-ch="5" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:4px; flex-shrink:0;">
          <button class="atlas-btn" id="n05-btn-ch-prev" disabled style="padding:2px 6px; font-size:11px;">${S.btnPrev}</button>
          <button class="atlas-btn" id="n05-btn-ch-next" style="padding:2px 6px; font-size:11px;">${S.btnNext}</button>
          <button class="atlas-btn" id="n05-btn-challenge" style="padding:2px 6px; font-size:11px;">Challenge</button>
        </div>
      </div>
      <div class="atlas-serif" id="n05-ch-body" style="font-size:12px; color:var(--ink); line-height:1.4;">
        ${S.ch1Body}
      </div>
      <div id="n05-predict-box" style="display:none; background:var(--paper-inset); border:1px solid var(--border); border-radius:3px; padding:6px 10px; margin-top:4px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span class="atlas-margin-label" style="font-size:10px; color:var(--volt);">${S.predictTitle}</span>
          <span class="atlas-serif" style="font-size:11px; color:var(--ink-muted);">${S.predictQuestion}</span>
        </div>
        <div style="display:flex; gap:10px; flex-wrap:wrap; font-size:11px; align-items:center;">
          <label style="cursor:pointer; display:flex; align-items:center; gap:4px;">
            <input type="radio" name="n05-predict" value="A" />
            <span>Broader spike</span>
          </label>
          <label style="cursor:pointer; display:flex; align-items:center; gap:4px;">
            <input type="radio" name="n05-predict" value="B" />
            <span>Smaller spike</span>
          </label>
          <label style="cursor:pointer; display:flex; align-items:center; gap:4px;">
            <input type="radio" name="n05-predict" value="C" />
            <span><b>No spike (passive bump only)</b></span>
          </label>
          <button class="atlas-btn" id="n05-btn-predict-submit" disabled style="padding:1px 6px; font-size:11px;">${S.predictSubmit}</button>
          <span id="n05-predict-feedback" style="display:none; font-size:11px; font-weight:600;"></span>
        </div>
      </div>
    `;

    // 3. Mount Compact Toolbar (Between chapter banner and canvas)
    let toolbar = document.getElementById('n05-toolbar');
    if (!toolbar) {
      toolbar = document.createElement('div');
      toolbar.id = 'n05-toolbar';
      toolbar.style.cssText =
        'display:flex; align-items:center; justify-content:space-between; gap:6px; margin-bottom:6px; flex-wrap:wrap;';
      figureSection.insertBefore(toolbar, figureContainer);
    }

    toolbar.innerHTML = `
      <div style="display:inline-flex; align-items:center; gap:6px;">
        <button class="atlas-btn" id="n05-btn-stim" style="font-weight:600; padding:3px 10px; background:var(--volt); color:var(--paper); border-color:var(--volt); font-size:12px;">
          ${S.btnStimulate}
        </button>
        <div style="display:inline-flex; align-items:center; gap:4px;">
          <label for="n05-input-current" class="atlas-serif" style="font-size:11px; font-weight:600;">${S.labelCurrentSlider}:</label>
          <input type="range" id="n05-input-current" min="0" max="20" step="0.5" value="0" style="width:75px; cursor:pointer;" />
          <span class="atlas-mono" id="n05-txt-current" style="font-size:11px; min-width:44px;">0.0 µA</span>
        </div>
      </div>

      <div style="display:inline-flex; align-items:center; gap:6px; flex-wrap:wrap;">
        <button class="atlas-btn" id="n05-btn-clamp" style="font-size:11px; padding:3px 8px;">
          ${S.modeCurrentClamp}
        </button>
        <button class="atlas-btn" id="n05-toggle-ttx" aria-pressed="false" style="font-size:11px; padding:3px 8px;">
          ${S.toggleTTX}
        </button>
        <button class="atlas-btn" id="n05-toggle-tea" aria-pressed="false" style="font-size:11px; padding:3px 8px;">
          ${S.toggleTEA}
        </button>
        <button class="atlas-btn" id="n05-toggle-theory" aria-pressed="false" style="font-size:11px; padding:3px 8px;">
          ${S.toggleTheory}
        </button>
      </div>
    `;

    // 4. Main Canvas inside figureContainer
    const canvas = figureContainer.querySelector('canvas') || ctx.canvas();
    const cContext = canvas.getContext('2d');

    const oldControls = document.getElementById('n05-controls-wrapper');
    if (oldControls) oldControls.remove();

    // DOM References
    const btnStim = document.getElementById('n05-btn-stim');
    const inputCurrent = document.getElementById('n05-input-current');
    const txtCurrent = document.getElementById('n05-txt-current');
    const btnClamp = document.getElementById('n05-btn-clamp');
    const toggleTTX = document.getElementById('n05-toggle-ttx');
    const toggleTEA = document.getElementById('n05-toggle-tea');
    const toggleTheory = document.getElementById('n05-toggle-theory');
    const theoryDiffContainer = document.getElementById('theory-diff-container');

    const chBadge = document.getElementById('n05-ch-badge');
    const chTitle = document.getElementById('n05-ch-title');
    const chBody = document.getElementById('n05-ch-body');
    const btnChPrev = document.getElementById('n05-btn-ch-prev');
    const btnChNext = document.getElementById('n05-btn-ch-next');
    const btnChallenge = document.getElementById('n05-btn-challenge');
    const predictBox = document.getElementById('n05-predict-box');

    const predictOptions = document.querySelectorAll('input[name="n05-predict"]');
    const btnPredictSubmit = document.getElementById('n05-btn-predict-submit');
    const txtPredictFeedback = document.getElementById('n05-predict-feedback');

    // Trigger stimulus helper
    const triggerStimulate = () => {
      state.pulseRemainingMs = 1.2;
      btnStim.textContent = S.btnStimulating;
      setTimeout(() => {
        btnStim.textContent = S.btnStimulate;
      }, 400);
    };

    btnStim.addEventListener('click', triggerStimulate);

    inputCurrent.addEventListener('input', (e) => {
      state.I_ext_slider = parseFloat(e.target.value);
      txtCurrent.textContent = `${state.I_ext_slider.toFixed(1)} µA`;
    });

    btnClamp.addEventListener('click', () => {
      state.voltageClamp = !state.voltageClamp;
      btnClamp.classList.toggle('active', state.voltageClamp);
      btnClamp.textContent = state.voltageClamp ? S.modeVoltageClamp : S.modeCurrentClamp;
      state.clampTimeMs = 0;
      state.clampHistory = new Array(prefillLength).fill(0);
    });

    toggleTTX.addEventListener('click', () => {
      state.ttx = !state.ttx;
      toggleTTX.classList.toggle('active', state.ttx);
      toggleTTX.setAttribute('aria-pressed', String(state.ttx));
    });

    toggleTEA.addEventListener('click', () => {
      state.tea = !state.tea;
      toggleTEA.classList.toggle('active', state.tea);
      toggleTEA.setAttribute('aria-pressed', String(state.tea));
    });

    const setTheory = (active) => {
      state.theoryActive = active;
      toggleTheory.classList.toggle('active', state.theoryActive);
      toggleTheory.setAttribute('aria-pressed', String(state.theoryActive));
      if (theoryDiffContainer) {
        theoryDiffContainer.style.display = state.theoryActive ? 'block' : 'none';
      }
    };

    toggleTheory.addEventListener('click', () => {
      setTheory(!state.theoryActive);
    });

    window.addEventListener('atlas:theory', (e) => {
      if (e.detail && e.detail.active !== undefined) {
        setTheory(e.detail.active);
      }
    });

    btnChallenge.addEventListener('click', () => {
      const isVisible = predictBox.style.display !== 'none';
      predictBox.style.display = isVisible ? 'none' : 'block';
    });

    // Pointer-hold on membrane canvas to inject current directly
    canvas.addEventListener('pointerdown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const clickY = (e.clientY - rect.top) / rect.height;
      if (clickY < 0.58) {
        state.isPoking = true;
      }
    });

    const stopPoking = () => {
      state.isPoking = false;
    };
    canvas.addEventListener('pointerup', stopPoking);
    canvas.addEventListener('pointercancel', stopPoking);
    canvas.addEventListener('pointerleave', stopPoking);

    // Chapters setup
    const chapters = [
      {
        num: 1,
        title: S.ch1Title,
        body: S.ch1Body,
        setup: () => {
          state.I_ext_slider = 0;
          inputCurrent.value = '0';
          txtCurrent.textContent = '0.0 µA';
          state.ttx = false;
          state.tea = false;
          toggleTTX.classList.remove('active');
          toggleTEA.classList.remove('active');
          state.voltageClamp = false;
          btnClamp.classList.remove('active');
          btnClamp.textContent = S.modeCurrentClamp;
          predictBox.style.display = 'none';
        },
      },
      {
        num: 2,
        title: S.ch2Title,
        body: S.ch2Body,
        setup: () => {
          state.I_ext_slider = 10;
          inputCurrent.value = '10';
          txtCurrent.textContent = '10.0 µA';
          state.ttx = false;
          state.tea = false;
          toggleTTX.classList.remove('active');
          toggleTEA.classList.remove('active');
          state.voltageClamp = false;
          btnClamp.classList.remove('active');
          btnClamp.textContent = S.modeCurrentClamp;
          predictBox.style.display = 'none';
        },
      },
      {
        num: 3,
        title: S.ch3Title,
        body: S.ch3Body,
        setup: () => {
          state.pulseRemainingMs = 1.2;
          predictBox.style.display = 'none';
        },
      },
      {
        num: 4,
        title: S.ch4Title,
        body: S.ch4Body,
        setup: () => {
          state.I_ext_slider = 2.5;
          inputCurrent.value = '2.5';
          txtCurrent.textContent = '2.5 µA';
          predictBox.style.display = 'none';
        },
      },
      {
        num: 5,
        title: S.ch5Title,
        body: S.ch5Body,
        setup: () => {
          predictBox.style.display = 'block';
        },
      },
    ];

    const applyChapter = (idx) => {
      state.chapter = idx;
      const ch = chapters[idx - 1];
      chBadge.textContent = `${S.chapterLabel} ${ch.num} ${S.ofChapters}`;
      chTitle.textContent = ch.title;
      chBody.textContent = ch.body;
      btnChPrev.disabled = ch.num === 1;
      btnChNext.disabled = ch.num === chapters.length;

      const dots = document.querySelectorAll('.n05-dot');
      dots.forEach((d, i) => {
        d.textContent = i + 1 === idx ? '●' : '○';
        d.style.opacity = i + 1 === idx ? '1.0' : '0.4';
      });

      ch.setup();
      renderFigure(cContext, canvas, state, ctx.colors());
    };

    btnChPrev.addEventListener('click', () => {
      if (state.chapter > 1) applyChapter(state.chapter - 1);
    });

    btnChNext.addEventListener('click', () => {
      if (state.chapter < chapters.length) applyChapter(state.chapter + 1);
    });

    document.querySelectorAll('.n05-dot').forEach((d) => {
      d.addEventListener('click', () => {
        const c = parseInt(d.getAttribute('data-ch'), 10);
        if (c >= 1 && c <= 5) applyChapter(c);
      });
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'ArrowRight' && !e.shiftKey) {
        if (state.chapter < chapters.length) applyChapter(state.chapter + 1);
      } else if (e.key === 'ArrowLeft' && !e.shiftKey) {
        if (state.chapter > 1) applyChapter(state.chapter - 1);
      } else if (e.key === 's' || e.key === 'S') {
        triggerStimulate();
      } else if (e.key === 'p' || e.key === 'P') {
        btnChallenge.click();
      }
    });

    // Predict-then-reveal
    predictOptions.forEach((opt) => {
      opt.addEventListener('change', (e) => {
        state.predictAnswer = e.target.value;
        btnPredictSubmit.disabled = false;
      });
    });

    btnPredictSubmit.addEventListener('click', () => {
      state.predictRevealed = true;
      btnPredictSubmit.style.display = 'none';
      txtPredictFeedback.style.display = 'inline-block';
      if (state.predictAnswer === 'C') {
        txtPredictFeedback.style.color = 'var(--exc)';
        txtPredictFeedback.textContent = `Correct! ${S.predictExplanation}`;
      } else {
        txtPredictFeedback.style.color = 'var(--inh)';
        txtPredictFeedback.textContent = `Not quite. ${S.predictExplanation}`;
      }
      state.ttx = true;
      toggleTTX.classList.add('active');
      toggleTTX.setAttribute('aria-pressed', 'true');
      state.pulseRemainingMs = 2.0;
    });

    // 5. Main Simulation & Animation Loop
    const dt = 0.025; // 0.025 ms RK4 timestep
    const substepsPerFrame = 20; // 0.50 ms per frame: captures peak spike and dynamic ion fluxes majestically

    ctx.onFrame(() => {
      let prevV = state.sim.V;

      for (let s = 0; s < substepsPerFrame; s++) {
        let curI = state.I_ext_slider;
        if (state.pulseRemainingMs > 0) {
          curI += 15.0; // 15 µA/cm² depolarizing pulse
          state.pulseRemainingMs -= dt;
        }
        if (state.isPoking) {
          curI += 15.0;
        }

        const opts = {
          ttx: state.ttx,
          tea: state.tea,
          clamp: state.voltageClamp,
          clampV: state.voltageClamp ? (state.clampTimeMs > 2.0 ? state.clampStepV : state.clampHoldV) : undefined,
        };

        state.sim = core.rk4Step(state.sim, curI, dt, opts);

        if (state.voltageClamp) {
          state.clampTimeMs += dt;
          if (state.clampTimeMs > 25.0) {
            state.clampTimeMs = 0;
          }
        }
      }

      const currentSim = state.sim;

      // Spike peak detection for markers
      if (prevV > 0 && prevV >= currentSim.V && state.voltageHistory.length > 2) {
        const lastV = state.voltageHistory[state.voltageHistory.length - 1];
        if (lastV > 15) {
          state.spikes.push({
            step: state.voltageHistory.length - 1,
            peakV: lastV,
          });
          if (state.spikes.length > 5) state.spikes.shift();
        }
      }

      // Update trace histories
      state.voltageHistory.push(currentSim.V);
      if (state.voltageHistory.length > state.maxHistory) {
        state.voltageHistory.shift();
        for (const sp of state.spikes) {
          sp.step -= 1;
        }
        state.spikes = state.spikes.filter((sp) => sp.step >= 0);
      }

      const gNa = state.ttx ? 0 : core.G_NA * (currentSim.m ** 3) * currentSim.h;
      const gK = state.tea ? 0 : core.G_K * (currentSim.n ** 4);
      state.gNaHistory.push(gNa);
      state.gKHistory.push(gK);
      if (state.gNaHistory.length > state.maxHistory) {
        state.gNaHistory.shift();
        state.gKHistory.shift();
      }

      state.mHistory.push(currentSim.m);
      state.hHistory.push(currentSim.h);
      state.nHistory.push(currentSim.n);
      if (state.mHistory.length > state.maxHistory) {
        state.mHistory.shift();
        state.hHistory.shift();
        state.nHistory.shift();
      }

      if (state.voltageClamp) {
        const netClampI = currentSim.I_Na + currentSim.I_K + currentSim.I_L;
        state.clampHistory.push(netClampI);
        if (state.clampHistory.length > state.maxHistory) state.clampHistory.shift();
      }

      // Stochastic channel gate sampling
      core.sampleChannels(state.channels, currentSim.m, currentSim.h, currentSim.n, state.rng);

      // Ion Particle Flow Simulation
      updateIonFlow(state, currentSim);

      // Update Marginalia DOM readouts
      const valV = document.getElementById('val-V');
      const valIext = document.getElementById('val-Iext');
      const valINa = document.getElementById('val-INa');
      const valIK = document.getElementById('val-IK');
      const valIL = document.getElementById('val-IL');
      const valItot = document.getElementById('val-Itot');
      const valM = document.getElementById('val-m');
      const valH = document.getElementById('val-h');
      const valN = document.getElementById('val-n');
      const valTheoryDiff = document.getElementById('val-theory-diff');

      if (valV) valV.textContent = `${currentSim.V.toFixed(1)} mV`;
      if (valIext) valIext.textContent = `${currentSim.I_ext.toFixed(2)} µA/cm²`;
      if (valINa) valINa.textContent = `${currentSim.I_Na.toFixed(2)} µA/cm²`;
      if (valIK) valIK.textContent = `${currentSim.I_K.toFixed(2)} µA/cm²`;
      if (valIL) valIL.textContent = `${currentSim.I_L.toFixed(2)} µA/cm²`;
      if (valItot) {
        const netI = currentSim.I_ext - currentSim.I_Na - currentSim.I_K - currentSim.I_L;
        valItot.textContent = `${netI.toFixed(2)} µA/cm²`;
      }
      if (valM) valM.textContent = currentSim.m.toFixed(3);
      if (valH) valH.textContent = currentSim.h.toFixed(3);
      if (valN) valN.textContent = currentSim.n.toFixed(3);

      if (valTheoryDiff && state.theoryActive) {
        const mInf = core.m_inf(currentSim.V);
        const diffM = Math.abs(currentSim.m - mInf);
        valTheoryDiff.textContent = `|m - m_inf| = ${diffM.toFixed(3)}`;
      }

      // Render graphics to Canvas
      renderFigure(cContext, canvas, state, ctx.colors());
    });

    // Initial render on mount so canvas is immediately complete
    renderFigure(cContext, canvas, state, ctx.colors());

    // Update ion positions and active streaming through open pores
    function updateIonFlow(s, sim) {
      const ions = s.ions;
      const channels = s.channels;

      // Current driving magnitudes
      const naFlowSpeed = Math.min(0.025, 0.004 + Math.abs(sim.I_Na) * 0.00005);
      const kFlowSpeed = Math.min(0.025, 0.004 + Math.abs(sim.I_K) * 0.00005);

      for (const ion of ions) {
        let nearCh = null;
        let minDist = 999;
        for (const ch of channels) {
          const d = Math.abs(ion.x - ch.xRatio);
          if (d < minDist) {
            minDist = d;
            nearCh = ch;
          }
        }

        const isPoreZone = minDist < 0.032;
        const inBilayerY = ion.y >= 0.42 && ion.y <= 0.58;

        if (ion.type === 'na') {
          const isOpenNaPore = nearCh && nearCh.type === 'na' && nearCh.isOpen && !s.ttx;

          if (isPoreZone && isOpenNaPore) {
            // Sucked downward into intracellular fluid (inward current)
            ion.x += (nearCh.xRatio - ion.x) * 0.28;
            ion.vy = naFlowSpeed;
            ion.y += ion.vy;
            ion.inPore = true;

            if (ion.y > 0.60) {
              ion.inPore = false;
              if (s.rng() < 0.15) {
                ion.y = 0.06 + s.rng() * 0.25;
                ion.x = s.rng();
                ion.vy = (s.rng() - 0.5) * 0.003;
              }
            }
          } else {
            ion.inPore = false;
            ion.x += ion.vx;
            ion.y += ion.vy;

            if (inBilayerY) {
              if (ion.y < 0.50) ion.vy = -Math.abs(ion.vy);
              else ion.vy = Math.abs(ion.vy);
            }
          }
        } else if (ion.type === 'k') {
          const isOpenKPore = nearCh && nearCh.type === 'k' && nearCh.isOpen && !s.tea;

          if (isPoreZone && isOpenKPore) {
            // Sucked upward into extracellular fluid (outward current)
            ion.x += (nearCh.xRatio - ion.x) * 0.28;
            ion.vy = -kFlowSpeed;
            ion.y += ion.vy;
            ion.inPore = true;

            if (ion.y < 0.40) {
              ion.inPore = false;
              if (s.rng() < 0.15) {
                ion.y = 0.65 + s.rng() * 0.28;
                ion.x = s.rng();
                ion.vy = (s.rng() - 0.5) * 0.003;
              }
            }
          } else {
            ion.inPore = false;
            ion.x += ion.vx;
            ion.y += ion.vy;

            if (inBilayerY) {
              if (ion.y < 0.50) ion.vy = -Math.abs(ion.vy);
              else ion.vy = Math.abs(ion.vy);
            }
          }
        }

        // Boundary containment
        if (ion.x < 0.02) ion.x = 0.98;
        if (ion.x > 0.98) ion.x = 0.02;
        if (ion.y < 0.04) {
          ion.y = 0.04;
          ion.vy = Math.abs(ion.vy);
        } else if (ion.y > 0.95) {
          ion.y = 0.95;
          ion.vy = -Math.abs(ion.vy);
        }
      }
    }

    // 6. Canvas 2D Drawing Engine
    function renderFigure(ctx2d, cnv, s, rawColors) {
      if (!ctx2d || !cnv) return;

      const colors = {
        ...rawColors,
        paperInset: rawColors.paperCard || '#e7e2d6',
        borderSubtle: rawColors.border || '#e5dfd3',
      };

      const dpr = window.devicePixelRatio || 1;
      const rect = cnv.getBoundingClientRect();
      const W = rect.width || cnv.width / dpr || 800;
      const H = rect.height || cnv.height / dpr || 400;

      if (cnv.width !== Math.round(W * dpr) || cnv.height !== Math.round(H * dpr)) {
        cnv.width = Math.round(W * dpr);
        cnv.height = Math.round(H * dpr);
      }
      ctx2d.resetTransform();
      ctx2d.scale(dpr, dpr);

      ctx2d.fillStyle = colors.paperCard;
      ctx2d.fillRect(0, 0, W, H);

      // Top 52% = Hero membrane patch; Bottom 48% = 3 Oscilloscope trace panes
      const topH = Math.round(H * 0.52);
      const bottomH = H - topH;

      renderMembranePatch(ctx2d, W, topH, s, colors);
      renderOscilloscope(ctx2d, 0, topH, W, bottomH, s, colors);
    }

    // A. Hero Membrane Patch
    function renderMembranePatch(ctx2d, W, H, s, colors) {
      ctx2d.save();

      const isMobile = W < 520;
      const thermW = isMobile ? 74 : 118;
      const patchW = W - thermW - 14;
      const midY = H * 0.50;
      const bilayerThickness = 30;
      const topLeafletY = midY - bilayerThickness / 2;
      const btmLeafletY = midY + bilayerThickness / 2;

      const isSpiking = s.sim.V > 0;
      const vNorm = Math.max(0, Math.min(1, (s.sim.V - core.E_K) / (core.E_NA - core.E_K)));

      // Extracellular fluid (top)
      ctx2d.fillStyle = colors.paperElevated;
      ctx2d.fillRect(8, 6, patchW, topLeafletY - 6);

      // Intracellular fluid (bottom) with voltage charge glow
      if (isSpiking) {
        ctx2d.fillStyle = `rgba(180, 83, 9, ${0.08 + vNorm * 0.16})`;
      } else {
        ctx2d.fillStyle = colors.paper;
      }
      ctx2d.fillRect(8, btmLeafletY, patchW, H - btmLeafletY - 6);

      // Hydrophobic core between leaflets
      ctx2d.fillStyle = colors.paperCard;
      ctx2d.fillRect(8, topLeafletY, patchW, bilayerThickness);

      // Lipid Bilayer Heads & Hairline Tails
      const headRadius = 4.2;
      const headSpacing = 11;
      const numHeads = Math.floor((patchW - 16) / headSpacing);

      // Tails
      ctx2d.lineWidth = 0.8;
      ctx2d.strokeStyle = colors.border;
      for (let i = 0; i < numHeads; i++) {
        const x = 14 + i * headSpacing;
        ctx2d.beginPath();
        ctx2d.moveTo(x - 1.5, topLeafletY + 2);
        ctx2d.lineTo(x - 1, midY - 2);
        ctx2d.moveTo(x + 1.5, topLeafletY + 2);
        ctx2d.lineTo(x + 1, midY - 2);
        ctx2d.stroke();

        ctx2d.beginPath();
        ctx2d.moveTo(x - 1.5, btmLeafletY - 2);
        ctx2d.lineTo(x - 1, midY + 2);
        ctx2d.moveTo(x + 1.5, btmLeafletY - 2);
        ctx2d.lineTo(x + 1, midY + 2);
        ctx2d.stroke();
      }

      // Heads
      ctx2d.fillStyle = colors.paperElevated;
      ctx2d.strokeStyle = colors.inkMuted;
      ctx2d.lineWidth = 1;
      for (let i = 0; i < numHeads; i++) {
        const x = 14 + i * headSpacing;
        ctx2d.beginPath();
        ctx2d.arc(x, topLeafletY - 2, headRadius, 0, Math.PI * 2);
        ctx2d.fill();
        ctx2d.stroke();

        ctx2d.beginPath();
        ctx2d.arc(x, btmLeafletY + 2, headRadius, 0, Math.PI * 2);
        ctx2d.fill();
        ctx2d.stroke();
      }

      // Draw Channels
      const channelH = bilayerThickness + 14;

      for (const ch of s.channels) {
        const cx = 14 + ch.xRatio * patchW;
        const cy = midY;

        ctx2d.save();

        if (ch.type === 'na') {
          const isHighlighted = s.hoverTerm === 'INa';
          const isBlocked = s.ttx;

          // Protein barrel walls
          ctx2d.fillStyle = isHighlighted ? 'rgba(15, 118, 110, 0.25)' : colors.paperElevated;
          ctx2d.strokeStyle = isHighlighted ? colors.volt : colors.exc;
          ctx2d.lineWidth = 1.4;

          // Left subunit
          ctx2d.beginPath();
          ctx2d.moveTo(cx - 15, cy - channelH / 2);
          ctx2d.quadraticCurveTo(cx - 6, cy, cx - 15, cy + channelH / 2);
          ctx2d.lineTo(cx - 7, cy + channelH / 2);
          ctx2d.quadraticCurveTo(cx - 4, cy, cx - 7, cy - channelH / 2);
          ctx2d.closePath();
          ctx2d.fill();
          ctx2d.stroke();

          // Right subunit
          ctx2d.beginPath();
          ctx2d.moveTo(cx + 15, cy - channelH / 2);
          ctx2d.quadraticCurveTo(cx + 6, cy, cx + 15, cy + channelH / 2);
          ctx2d.lineTo(cx + 7, cy + channelH / 2);
          ctx2d.quadraticCurveTo(cx + 4, cy, cx + 7, cy - channelH / 2);
          ctx2d.closePath();
          ctx2d.fill();
          ctx2d.stroke();

          // Open pore visible gap luminous glow
          if (ch.isOpen && !isBlocked) {
            ctx2d.fillStyle = 'rgba(15, 118, 110, 0.35)';
            ctx2d.fillRect(cx - 5, cy - channelH / 2 + 2, 10, channelH - 4);
          }

          // 3 visible "m" activation gates at extracellular mouth
          ctx2d.strokeStyle = colors.exc;
          ctx2d.lineWidth = 2.0;

          if (ch.mOpen && !isBlocked) {
            // Gates swung open outward leaving wide gap
            ctx2d.beginPath();
            ctx2d.moveTo(cx - 6, cy - channelH / 2);
            ctx2d.lineTo(cx - 4, cy - channelH / 2 - 8);
            ctx2d.moveTo(cx, cy - channelH / 2);
            ctx2d.lineTo(cx, cy - channelH / 2 - 9);
            ctx2d.moveTo(cx + 6, cy - channelH / 2);
            ctx2d.lineTo(cx + 4, cy - channelH / 2 - 8);
            ctx2d.stroke();
          } else {
            // Closed: 3 m gates meet across the mouth
            ctx2d.beginPath();
            ctx2d.moveTo(cx - 6, cy - channelH / 2);
            ctx2d.lineTo(cx - 1, cy - channelH / 2);
            ctx2d.moveTo(cx - 1, cy - channelH / 2 + 1);
            ctx2d.lineTo(cx + 2, cy - channelH / 2 + 1);
            ctx2d.moveTo(cx + 2, cy - channelH / 2);
            ctx2d.lineTo(cx + 6, cy - channelH / 2);
            ctx2d.stroke();
          }

          // "h" inactivation gate: ball-and-chain
          const chainAnchorX = cx - 9;
          const chainAnchorY = cy + channelH / 2;
          ctx2d.strokeStyle = colors.mod;
          ctx2d.fillStyle = colors.mod;
          ctx2d.lineWidth = 1.2;

          if (!ch.hOpen || isBlocked) {
            // Ball docked inside cytoplasmic pore mouth
            ctx2d.beginPath();
            ctx2d.moveTo(chainAnchorX, chainAnchorY);
            ctx2d.quadraticCurveTo(cx - 4, chainAnchorY + 4, cx, chainAnchorY);
            ctx2d.stroke();

            // Docked inactivation ball (plugs the pore)
            ctx2d.beginPath();
            ctx2d.arc(cx, chainAnchorY - 2, 4.5, 0, Math.PI * 2);
            ctx2d.fill();
            ctx2d.strokeStyle = colors.paper;
            ctx2d.stroke();
          } else {
            // Open: ball swings freely in intracellular fluid
            const ballX = cx - 13;
            const ballY = chainAnchorY + 11;
            ctx2d.beginPath();
            ctx2d.moveTo(chainAnchorX, chainAnchorY);
            ctx2d.quadraticCurveTo(cx - 11, chainAnchorY + 6, ballX, ballY);
            ctx2d.stroke();

            ctx2d.beginPath();
            ctx2d.arc(ballX, ballY, 4.5, 0, Math.PI * 2);
            ctx2d.fill();
          }

          // TTX blocker molecule wedged in outer pore
          if (isBlocked) {
            ctx2d.fillStyle = colors.inh;
            ctx2d.strokeStyle = colors.paper;
            ctx2d.lineWidth = 1;
            ctx2d.fillRect(cx - 6, cy - channelH / 2 - 4, 12, 7);
            ctx2d.strokeRect(cx - 6, cy - channelH / 2 - 4, 12, 7);
            ctx2d.font = '8px ui-monospace, monospace';
            ctx2d.fillStyle = colors.paper;
            ctx2d.fillText('TTX', cx - 7, cy - channelH / 2 + 2);
          }

        } else if (ch.type === 'k') {
          const isHighlighted = s.hoverTerm === 'IK';
          const isBlocked = s.tea;

          ctx2d.fillStyle = isHighlighted ? 'rgba(190, 18, 60, 0.25)' : colors.paperElevated;
          ctx2d.strokeStyle = isHighlighted ? colors.volt : colors.inh;
          ctx2d.lineWidth = 1.4;

          // Protein barrel walls
          ctx2d.beginPath();
          ctx2d.moveTo(cx - 14, cy - channelH / 2);
          ctx2d.quadraticCurveTo(cx - 5, cy, cx - 14, cy + channelH / 2);
          ctx2d.lineTo(cx - 6, cy + channelH / 2);
          ctx2d.quadraticCurveTo(cx - 3, cy, cx - 6, cy - channelH / 2);
          ctx2d.closePath();
          ctx2d.fill();
          ctx2d.stroke();

          ctx2d.beginPath();
          ctx2d.moveTo(cx + 14, cy - channelH / 2);
          ctx2d.quadraticCurveTo(cx + 5, cy, cx + 14, cy + channelH / 2);
          ctx2d.lineTo(cx + 6, cy + channelH / 2);
          ctx2d.quadraticCurveTo(cx + 3, cy, cx + 6, cy - channelH / 2);
          ctx2d.closePath();
          ctx2d.fill();
          ctx2d.stroke();

          if (ch.isOpen && !isBlocked) {
            ctx2d.fillStyle = 'rgba(190, 18, 60, 0.35)';
            ctx2d.fillRect(cx - 5, cy - channelH / 2 + 2, 10, channelH - 4);
          }

          // 4 visible "n" activation gate subunits at cytoplasmic constriction
          ctx2d.strokeStyle = colors.inh;
          ctx2d.lineWidth = 2.0;

          if (ch.nOpen && !isBlocked) {
            // Swung open outward
            ctx2d.beginPath();
            ctx2d.moveTo(cx - 5, cy + channelH / 2);
            ctx2d.lineTo(cx - 3, cy + channelH / 2 + 7);
            ctx2d.moveTo(cx + 5, cy + channelH / 2);
            ctx2d.lineTo(cx + 3, cy + channelH / 2 + 7);
            ctx2d.stroke();
          } else {
            // Closed: 4 n gates meet in center
            ctx2d.beginPath();
            ctx2d.moveTo(cx - 6, cy + channelH / 2);
            ctx2d.lineTo(cx - 1, cy + channelH / 2);
            ctx2d.moveTo(cx + 1, cy + channelH / 2);
            ctx2d.lineTo(cx + 6, cy + channelH / 2);
            ctx2d.stroke();
          }

          // TEA blocker
          if (isBlocked) {
            ctx2d.fillStyle = colors.mod;
            ctx2d.strokeStyle = colors.paper;
            ctx2d.lineWidth = 1;
            ctx2d.fillRect(cx - 6, cy + channelH / 2 - 3, 12, 7);
            ctx2d.strokeRect(cx - 6, cy + channelH / 2 - 3, 12, 7);
            ctx2d.font = '8px ui-monospace, monospace';
            ctx2d.fillStyle = colors.paper;
            ctx2d.fillText('TEA', cx - 7, cy + channelH / 2 + 2);
          }

        } else if (ch.type === 'leak') {
          // Narrow passive leak pore
          const isHighlighted = s.hoverTerm === 'IL';
          ctx2d.fillStyle = isHighlighted ? 'rgba(180, 83, 9, 0.25)' : colors.paperCard;
          ctx2d.strokeStyle = colors.inkMuted;
          ctx2d.lineWidth = 1.0;

          ctx2d.fillRect(cx - 8, cy - channelH / 2 + 2, 5, channelH - 4);
          ctx2d.strokeRect(cx - 8, cy - channelH / 2 + 2, 5, channelH - 4);
          ctx2d.fillRect(cx + 3, cy - channelH / 2 + 2, 5, channelH - 4);
          ctx2d.strokeRect(cx + 3, cy - channelH / 2 + 2, 5, channelH - 4);

          ctx2d.fillStyle = 'rgba(180, 83, 9, 0.15)';
          ctx2d.fillRect(cx - 3, cy - channelH / 2 + 2, 6, channelH - 4);
        }

        ctx2d.restore();
      }

      // Draw Ion Particles with visible charge symbol
      for (const ion of s.ions) {
        const px = 14 + ion.x * patchW;
        const py = ion.y * H;

        ctx2d.beginPath();
        if (ion.type === 'na') {
          ctx2d.fillStyle = colors.exc;
          ctx2d.arc(px, py, 2.6, 0, Math.PI * 2);
        } else {
          ctx2d.fillStyle = colors.inh;
          ctx2d.arc(px, py, 3.2, 0, Math.PI * 2);
        }
        ctx2d.fill();

        // Small white plus inside ion
        ctx2d.fillStyle = colors.paperElevated;
        ctx2d.font = '8px ui-sans-serif, sans-serif';
        ctx2d.fillText('+', px - 2.5, py + 2.5);
      }

      // Stimulus Electrode Needle in Cytosol
      const isStimulating = s.pulseRemainingMs > 0 || s.isPoking || s.I_ext_slider > 0;
      if (isStimulating) {
        ctx2d.save();
        ctx2d.strokeStyle = colors.volt;
        ctx2d.fillStyle = colors.volt;
        ctx2d.lineWidth = 2.0;

        const tipX = patchW * 0.48;
        const tipY = btmLeafletY + 22;

        ctx2d.beginPath();
        ctx2d.moveTo(tipX, H - 2);
        ctx2d.lineTo(tipX, tipY);
        ctx2d.stroke();

        ctx2d.beginPath();
        ctx2d.arc(tipX, tipY, 6, 0, Math.PI * 2);
        ctx2d.fill();

        ctx2d.font = '10px ui-monospace, monospace';
        ctx2d.fillText('+I_ext', tipX + 8, tipY + 4);
        ctx2d.restore();
      }

      // Fluid labels with clean badge backgrounds drawn after channels & ions so nothing occludes them
      ctx2d.save();
      const outText = isMobile ? 'OUTSIDE (high Na⁺)' : S.outside;
      const inText = isMobile ? 'INSIDE (high K⁺)' : S.inside;
      ctx2d.font = isMobile ? '10px ui-serif, Georgia, serif' : '11px ui-serif, Georgia, serif';
      ctx2d.textBaseline = 'middle';
      ctx2d.textAlign = 'left';

      // Badge for OUTSIDE
      const outW = ctx2d.measureText(outText).width;
      const outX = isMobile ? 12 : 16;
      const outY = isMobile ? 6 : 8;
      const badgeH = 18;
      ctx2d.fillStyle = colors.paperCard;
      ctx2d.strokeStyle = colors.border;
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      if (ctx2d.roundRect) ctx2d.roundRect(outX - 6, outY, outW + 12, badgeH, 3);
      else ctx2d.rect(outX - 6, outY, outW + 12, badgeH);
      ctx2d.fill();
      ctx2d.stroke();
      ctx2d.fillStyle = colors.ink;
      ctx2d.fillText(outText, outX, outY + badgeH / 2);

      // Badge for INSIDE
      const inW = ctx2d.measureText(inText).width;
      const inX = isMobile ? 12 : 16;
      const inY = H - (isMobile ? 24 : 26);
      ctx2d.fillStyle = colors.paperCard;
      ctx2d.strokeStyle = colors.border;
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      if (ctx2d.roundRect) ctx2d.roundRect(inX - 6, inY, inW + 12, badgeH, 3);
      else ctx2d.rect(inX - 6, inY, inW + 12, badgeH);
      ctx2d.fill();
      ctx2d.stroke();
      ctx2d.fillStyle = colors.ink;
      ctx2d.fillText(inText, inX, inY + badgeH / 2);
      ctx2d.restore();

      // B. Glass Thermometer Voltmeter with Nernst Lines & Amber Glow Needle
      renderThermometer(ctx2d, W - thermW + 2, 8, thermW - 8, H - 16, s, colors);

      ctx2d.restore();
    }

    // Thermometer Voltmeter component
    function renderThermometer(ctx2d, tx, ty, tw, th, s, colors) {
      ctx2d.save();

      const isMobile = tw < 80;
      const tubeW = isMobile ? 12 : 15;
      const tubeX = tx + (isMobile ? 22 : 36);
      const bulbRadius = isMobile ? 10 : 13;
      const bulbCy = ty + th - bulbRadius - 4;
      const tubeTopY = ty + 14;
      const tubeBottomY = bulbCy - 6;
      const tubeH = tubeBottomY - tubeTopY;

      // Voltage mapping: EK (-77) at bottom bulb, ENa (+50) at top
      const vToY = (v) => tubeBottomY - ((v - core.E_K) / (core.E_NA - core.E_K)) * tubeH;

      // Outer glass casing and empty tube background
      ctx2d.fillStyle = colors.paper;
      ctx2d.strokeStyle = colors.border;
      ctx2d.lineWidth = 1.2;

      // Bulb glass
      ctx2d.beginPath();
      ctx2d.arc(tubeX + tubeW / 2, bulbCy, bulbRadius, 0, Math.PI * 2);
      ctx2d.fill();
      ctx2d.stroke();

      // Tube glass
      ctx2d.fillRect(tubeX, tubeTopY, tubeW, tubeH);
      ctx2d.strokeRect(tubeX, tubeTopY, tubeW, tubeH);

      // Liquid column filled from bulbCy up to liquidTopY
      const liquidTopY = Math.max(tubeTopY, Math.min(tubeBottomY, vToY(s.sim.V)));
      const isSpike = s.sim.V > 0;

      // Inner liquid column (narrower than tube so glass walls are visible)
      const liquidW = tubeW - 4;
      const liquidX = tubeX + 2;

      ctx2d.fillStyle = isSpike ? colors.volt : colors.inh;
      // Bulb liquid
      ctx2d.beginPath();
      ctx2d.arc(tubeX + tubeW / 2, bulbCy, bulbRadius - 2.5, 0, Math.PI * 2);
      ctx2d.fill();

      // Tube liquid column (rises up to liquidTopY)
      ctx2d.fillRect(liquidX, liquidTopY, liquidW, tubeBottomY - liquidTopY + 2);

      // Etched Nernst Equilibrium Lines
      const drawNernstLine = (v, label, color) => {
        const ly = vToY(v);
        ctx2d.strokeStyle = color;
        ctx2d.lineWidth = 1.0;
        ctx2d.beginPath();
        ctx2d.moveTo(tubeX - 4, ly);
        ctx2d.lineTo(tubeX + tubeW + 4, ly);
        ctx2d.stroke();

        ctx2d.font = '9px ui-monospace, monospace';
        ctx2d.fillStyle = color;
        ctx2d.textAlign = 'left';
        ctx2d.fillText(label, tubeX + tubeW + 5, ly + 3);
      };

      drawNernstLine(core.E_NA, isMobile ? '+50' : '+50 ENa', colors.exc);
      drawNernstLine(0, '0', colors.inkMuted);
      if (!isMobile) drawNernstLine(core.E_L, '-54 EL', colors.inkMuted);
      drawNernstLine(core.REST_V, isMobile ? '-65' : '-65 Vrest', colors.volt);
      drawNernstLine(core.E_K, isMobile ? '-77' : '-77 EK', colors.inh);

      // Active Needle & Meniscus Glow
      const needleY = Math.max(tubeTopY, Math.min(tubeBottomY + 4, vToY(s.sim.V)));

      if (isSpike) {
        ctx2d.shadowColor = colors.volt;
        ctx2d.shadowBlur = 12;
      }

      ctx2d.strokeStyle = colors.volt;
      ctx2d.fillStyle = colors.volt;
      ctx2d.lineWidth = 2.4;

      // Pointer needle
      ctx2d.beginPath();
      ctx2d.moveTo(tubeX - 6, needleY);
      ctx2d.lineTo(tubeX + tubeW + 6, needleY);
      ctx2d.stroke();

      // Indicator triangle pointing into tube
      ctx2d.beginPath();
      ctx2d.moveTo(tubeX - 6, needleY);
      ctx2d.lineTo(tubeX - (isMobile ? 11 : 13), needleY - 4);
      ctx2d.lineTo(tubeX - (isMobile ? 11 : 13), needleY + 4);
      ctx2d.closePath();
      ctx2d.fill();

      // Live reading text (right-aligned before pointer)
      ctx2d.font = (isMobile ? '9px' : '10px') + ' ui-monospace, monospace';
      ctx2d.textAlign = 'right';
      ctx2d.fillText(`${s.sim.V.toFixed(0)}`, tubeX - (isMobile ? 14 : 16), needleY + 3);

      ctx2d.restore();
    }

    // B. Multi-trace Oscilloscope Panes
    function renderOscilloscope(ctx2d, X, Y, W, H, s, colors) {
      ctx2d.save();

      // Separator line
      ctx2d.strokeStyle = colors.border;
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      ctx2d.moveTo(X, Y);
      ctx2d.lineTo(X + W, Y);
      ctx2d.stroke();

      if (s.voltageClamp) {
        // 2 Panes for Voltage Clamp
        const paneW = (W - 28) / 2;

        renderTracePane(ctx2d, X + 8, Y + 8, paneW, H - 16, {
          title: 'Command Potential V_clamp (mV)',
          minY: -90,
          maxY: +60,
          ticks: [-77, -65, 0, 50],
          series: [{ data: s.voltageHistory, color: colors.volt, width: 2.2 }],
        }, colors);

        renderTracePane(ctx2d, X + 16 + paneW, Y + 8, paneW, H - 16, {
          title: 'Clamp Current I_clamp (µA/cm²)',
          minY: -600,
          maxY: +1200,
          ticks: [-500, 0, +500, +1000],
          series: [{ data: s.clampHistory, color: colors.trace, width: 2.2 }],
        }, colors);

      } else {
        // 3 Panes for Current Clamp (V, g, gates)
        const isMobile = W < 520;
        const paneW = (W - 36) / 3;

        // Pane 1: Membrane Potential V(t)
        renderTracePane(ctx2d, X + 8, Y + 8, paneW, H - 16, {
          title: isMobile ? 'V(t) (mV)' : S.traceVoltage,
          minY: -90,
          maxY: +55,
          ticks: [-77, -65, -45, 0, 40],
          series: [{ data: s.voltageHistory, color: colors.volt, width: 2.4 }],
          spikes: s.spikes,
          showCursor: true,
          cursorVal: s.sim.V,
        }, colors);

        // Pane 2: Conductances gNa and gK (auto-scaled so peak fills plot)
        const maxRecentG = Math.max(35, ...s.gNaHistory, ...s.gKHistory);
        renderTracePane(ctx2d, X + 16 + paneW, Y + 8, paneW, H - 16, {
          title: isMobile ? 'g(t) mS/cm²' : S.traceConductance,
          minY: 0,
          maxY: Math.ceil(maxRecentG * 1.05),
          ticks: [0, 15, 30],
          series: [
            { data: s.gNaHistory, color: colors.exc, width: 2.0, label: isMobile ? 'gNa' : 'gNa (Na⁺)' },
            { data: s.gKHistory, color: colors.inh, width: 2.0, label: isMobile ? 'gK' : 'gK (K⁺)' },
          ],
          legend: !isMobile,
        }, colors);

        // Pane 3: Gating Variables m, h, n + Theory Ghost
        renderTracePane(ctx2d, X + 24 + paneW * 2, Y + 8, paneW, H - 16, {
          title: s.theoryActive ? (isMobile ? 'x_inf' : S.steadyStateCurves) : (isMobile ? 'Gates' : S.traceGates),
          minY: 0,
          maxY: 1.02,
          ticks: [0, 0.5, 1.0],
          series: [
            { data: s.mHistory, color: colors.exc, width: 1.8, label: 'm' },
            { data: s.hHistory, color: colors.mod, width: 1.8, label: 'h' },
            { data: s.nHistory, color: colors.inh, width: 1.8, label: 'n' },
          ],
          legend: !isMobile,
          theoryOverlay: s.theoryActive,
          currentV: s.sim.V,
          currentM: s.sim.m,
          currentH: s.sim.h,
          currentN: s.sim.n,
        }, colors);
      }

      ctx2d.restore();
    }

    function renderTracePane(ctx2d, px, py, pw, ph, cfg, colors) {
      ctx2d.save();

      ctx2d.fillStyle = colors.paper;
      ctx2d.strokeStyle = colors.border;
      ctx2d.lineWidth = 1;
      ctx2d.fillRect(px, py, pw, ph);
      ctx2d.strokeRect(px, py, pw, ph);

      // Title
      ctx2d.font = '10px ui-serif, Georgia, serif';
      ctx2d.fillStyle = colors.inkMuted;
      ctx2d.fillText(cfg.title, px + 8, py + 12);

      const toY = (v) => py + ph - 8 - ((v - cfg.minY) / (cfg.maxY - cfg.minY)) * (ph - 24);

      // Grid lines
      if (cfg.ticks) {
        ctx2d.font = '9px ui-monospace, monospace';
        ctx2d.fillStyle = colors.inkMuted;
        ctx2d.lineWidth = 0.6;
        ctx2d.setLineDash([2, 3]);

        for (const tVal of cfg.ticks) {
          const gy = toY(tVal);
          if (gy >= py + 6 && gy <= py + ph - 6) {
            ctx2d.strokeStyle = colors.border;
            ctx2d.beginPath();
            ctx2d.moveTo(px + 24, gy);
            ctx2d.lineTo(px + pw - 6, gy);
            ctx2d.stroke();

            ctx2d.fillText(String(tVal), px + 4, gy + 3);
          }
        }
        ctx2d.setLineDash([]);
      }

      // Legend if requested
      if (cfg.legend && cfg.series) {
        ctx2d.font = '9px ui-monospace, monospace';
        let lx = px + pw - 8;
        for (let i = cfg.series.length - 1; i >= 0; i--) {
          const s = cfg.series[i];
          if (!s.label) continue;
          ctx2d.fillStyle = s.color;
          const textW = ctx2d.measureText(s.label).width;
          lx -= (textW + 14);
          ctx2d.fillRect(lx, py + 5, 8, 8);
          ctx2d.fillText(s.label, lx + 11, py + 12);
        }
      }

      // Theory Ghost steady-state curves
      if (cfg.theoryOverlay) {
        ctx2d.save();
        ctx2d.lineWidth = 1.2;
        ctx2d.setLineDash([3, 2]);

        const drawInfCurve = (fn, color) => {
          ctx2d.strokeStyle = color;
          ctx2d.beginPath();
          for (let ix = 0; ix < pw - 30; ix += 2) {
            const vVal = -90 + (ix / (pw - 30)) * 145;
            const yVal = fn.call(core, vVal);
            const cx = px + 26 + ix;
            const cy = toY(yVal);
            if (ix === 0) ctx2d.moveTo(cx, cy);
            else ctx2d.lineTo(cx, cy);
          }
          ctx2d.stroke();
        };

        drawInfCurve(core.m_inf, colors.exc);
        drawInfCurve(core.h_inf, colors.mod);
        drawInfCurve(core.n_inf, colors.inh);

        // Moving state dots at current V
        const dotX = px + 26 + ((cfg.currentV - (-90)) / 145) * (pw - 30);
        if (dotX >= px + 24 && dotX <= px + pw - 6) {
          ctx2d.setLineDash([]);
          ctx2d.fillStyle = colors.exc;
          ctx2d.beginPath();
          ctx2d.arc(dotX, toY(cfg.currentM), 3.5, 0, Math.PI * 2);
          ctx2d.fill();

          ctx2d.fillStyle = colors.mod;
          ctx2d.beginPath();
          ctx2d.arc(dotX, toY(cfg.currentH), 3.5, 0, Math.PI * 2);
          ctx2d.fill();

          ctx2d.fillStyle = colors.inh;
          ctx2d.beginPath();
          ctx2d.arc(dotX, toY(cfg.currentN), 3.5, 0, Math.PI * 2);
          ctx2d.fill();
        }

        ctx2d.restore();
        ctx2d.restore();
        return;
      }

      // Draw series curves
      for (const s of cfg.series) {
        const data = s.data;
        if (!data || data.length < 2) continue;

        ctx2d.strokeStyle = s.color;
        ctx2d.lineWidth = s.width || 1.8;
        ctx2d.beginPath();

        const stepX = (pw - 32) / (data.length - 1);
        for (let i = 0; i < data.length; i++) {
          const x = px + 26 + i * stepX;
          const y = Math.max(py + 4, Math.min(py + ph - 4, toY(data[i])));
          if (i === 0) ctx2d.moveTo(x, y);
          else ctx2d.lineTo(x, y);
        }
        ctx2d.stroke();
      }

      // Draw Spike Markers
      if (cfg.spikes && cfg.spikes.length > 0) {
        ctx2d.font = '9px ui-monospace, monospace';
        const stepX = (pw - 32) / (state.maxHistory - 1);
        for (const sp of cfg.spikes) {
          const sx = px + 26 + sp.step * stepX;
          const sy = toY(sp.peakV);
          if (sx >= px + 24 && sx <= px + pw - 6) {
            ctx2d.fillStyle = colors.volt;
            ctx2d.beginPath();
            ctx2d.moveTo(sx, sy - 2);
            ctx2d.lineTo(sx - 4, sy - 8);
            ctx2d.lineTo(sx + 4, sy - 8);
            ctx2d.closePath();
            ctx2d.fill();

            ctx2d.fillText(`+${sp.peakV.toFixed(0)}`, sx - 8, sy - 10);
          }
        }
      }

      // Moving Cursor / Playhead indicator at leading edge
      if (cfg.showCursor && cfg.series && cfg.series[0] && cfg.series[0].data.length > 0) {
        const data = cfg.series[0].data;
        const lastVal = data[data.length - 1];
        const lastX = px + pw - 6;
        const lastY = Math.max(py + 4, Math.min(py + ph - 4, toY(lastVal)));

        ctx2d.fillStyle = colors.volt;
        ctx2d.beginPath();
        ctx2d.arc(lastX, lastY, 4.0, 0, Math.PI * 2);
        ctx2d.fill();

        ctx2d.strokeStyle = colors.paper;
        ctx2d.lineWidth = 1.2;
        ctx2d.stroke();
      }

      ctx2d.restore();
    }
  }

  // Export for node tests
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core, S };
  }

  // Register in browser Atlas Shell
  if (typeof root.Atlas !== 'undefined') {
    root.Atlas.Plate.register({
      id: 'N05',
      title: {
        en: S.title,
        fa: S.titleFa,
      },
      source: S.source,
      mount,
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
