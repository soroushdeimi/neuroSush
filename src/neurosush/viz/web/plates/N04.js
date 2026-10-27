/**
 * PLATE N04 · Izhikevich: A Zoo of Firing Patterns
 * Live simulation of the two-variable Izhikevich (2003) neuron model.
 * Mathematical mirror of neurosush.neurons.dynamics.izhikevich_step & izhikevich_reset.
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
    plateId: 'N04',
    title: 'Izhikevich: A Zoo of Firing Patterns',
    source: 'live',
    libraryMirror:
      'neurosush.neurons.dynamics.izhikevich_step · neurosush.neurons.dynamics.izhikevich_reset · neurosush.neurons.models.Izhikevich',

    // Marginalia: Equations & Terms
    eqTitle: 'Governing Equations',
    eqVdot: "v' = 0.04·v² + 5·v + 140 - u + I",
    eqUdot: "u' = a·(b·v - u)",
    eqReset: 'if v ≥ 30 mV:  v ← c,  u ← u + d',

    termQuad: '0.04·v² (spike upstroke)',
    termLin: '5·v (subthreshold leak)',
    termConst: '140 (membrane bias)',
    termU: '- u (outward recovery current)',
    termI: '+ I (injected electrode drive)',
    termA: 'a (time scale of u)',
    termB: 'b (sensitivity of u to v)',
    termC: 'c (after-spike reset voltage)',
    termD: 'd (after-spike recovery jump)',

    // Marginalia: Live readouts
    readoutSection: 'Live Membrane State',
    readoutV: 'Membrane Potential v',
    readoutU: 'Recovery Variable u',
    readoutI: 'Injected Current I',
    readoutDvdt: 'Voltage Drift dv/dt',
    readoutDudt: 'Recovery Drift du/dt',

    // Marginalia: Fingerprint
    fingerprintSection: 'Pattern Fingerprint (Computed)',
    fingerprintLabel: 'Classified Rhythm',
    fingerprintSpikes: 'Spike Count (1 s)',
    fingerprintRate: 'Firing Rate',
    fingerprintMeanIsi: 'Mean ISI',
    fingerprintCvIsi: 'ISI Irregularity (CV)',
    fingerprintAdapt: 'Adaptation Ratio',

    // Marginalia: Theory Ghost
    theorySection: 'Theory Ghost: Nullclines & Fixed Points',
    theoryStatus: 'Theory Status',
    theoryParabola: 'v-nullcline: u = 0.04·v² + 5·v + 140 + I',
    theoryLine: 'u-nullcline: u = b·v',
    theoryRest: 'Stable Rest (v_rest)',
    theoryThresh: 'Threshold Saddle (v_th)',
    theoryDelta: 'Discriminant Δ',
    bifurcationActive: 'Δ < 0: Limit Cycle (Repetitive Spiking)',
    bifurcationRest: 'Δ ≥ 0: Quiescent Equilibrium Exists',

    // Marginalia: Glossary
    glossarySection: 'Glossary & Parameters',
    glossaryU:
      'u (Recovery): Slow outward current combining K⁺ activation and Na⁺ inactivation.',
    glossaryA:
      'a (Time scale): Decay rate of u. Smaller a gives slower recovery, prolonged bursts, and long adaptation.',
    glossaryB:
      'b (Sensitivity): Coupling of u to subthreshold voltage. High b yields resonance and subthreshold oscillations.',
    glossaryC:
      'c (Reset voltage): Post-spike membrane potential set by fast high-threshold K⁺ conductances (-65 to -50 mV).',
    glossaryD:
      'd (Recovery jump): Post-spike increment added to u by slow high-threshold conductances.',
    glossaryNullcline:
      'Nullclines: Curves where dv/dt = 0 (parabola) and du/dt = 0 (line). Their intersection is the equilibrium.',

    // Chapters
    chapterLabel: 'CHAPTER',
    ofChapters: 'OF 5',
    btnPrev: '◀ Back',
    btnNext: 'Next ▶',
    btnChallenge: 'Challenge [P]',

    ch1Title: '1. One Equation, Many Cells',
    ch1Body:
      'With just two coupled equations and four parameters, the Izhikevich model reproduces the entire zoo of cortical firing patterns. Click any cell in the zoo to compare their spiking rhythms and (v,u) phase portraits.',
    ch2Title: '2. The Recovery Variable u',
    ch2Body:
      'Parameter a sets how quickly recovery variable u reacts; b sets its coupling to voltage. In Fast Spiking (FS, a=0.10), u catches up so quickly that the neuron fires at 60+ Hz with virtually zero spike-frequency adaptation.',
    ch3Title: '3. Reset (c, d) Shapes Bursts',
    ch3Body:
      'After each spike, v resets to c and u jumps by d. Setting c high (-50 mV in Chattering) and d small (2.0) leaves the membrane above threshold, triggering rhythmic high-frequency bursts.',
    ch4Title: '4. a, b Make Resonance',
    ch4Body:
      'When b is elevated (0.26) and a is fast (0.10), the subthreshold fixed point becomes a damped focus. The cell exhibits subthreshold oscillations and resonates with high-frequency inputs.',
    ch5Title: '5. Your Own Cell',
    ch5Body:
      'Direct manipulation: drag the handles in the (a-vs-b) and (c-vs-d) parameter maps to sculpt your own custom neuron. Watch the live pattern fingerprint classify its firing mode in real time.',

    // Predict-then-reveal
    predictTitle: 'Predict-Then-Reveal',
    predictQuestion:
      'Which parameter set produces burst firing (Chattering vs Regular Spiking)?',
    predictOptA: 'A) a = 0.02, b = 0.20, c = -65 mV, d = 8.0',
    predictOptB: 'B) a = 0.02, b = 0.20, c = -50 mV, d = 2.0',
    predictSubmit: 'Commit Prediction',
    predictExplanation:
      'Correct! Bursting requires several spikes in rapid succession before recovery variable u can accumulate. Set B has a shallow reset c = -50 mV (near threshold) and a tiny jump d = 2.0, so the membrane immediately re-crosses threshold to burst. Set A resets deeply to -65 mV with d = 8.0, enforcing frequency adaptation.',
    predictWrong:
      'Not quite. Set B has a shallow reset c = -50 mV and a small jump d = 2.0, enabling immediate re-spiking into bursts. Set A resets deeply to -65 mV with a large d = 8.0 jump, producing regular spiking with adaptation.',
    predictBtnTest: 'Test Chattering (Set B)',

    // Controls
    btnStimulate: '⚡ Stimulate',
    btnStimulating: '⚡ Injecting...',
    labelCurrentSlider: 'Drive I',
    toggleTheory: 'Theory [T]',
    theoryOn: 'Theory Ghost: ON',
    theoryOff: 'Theory Ghost: OFF',
    btnReset: '↺ Reset',

    // Units
    unitsMv: 'mV',
    unitsPa: 'pA',
    unitsMs: 'ms',
    unitsHz: 'Hz',
    unitsDvdt: 'mV/ms',
    unitsDudt: 'pA/ms',
  };

  // ---- core: pure math, no DOM, exported for node tests ----
  const core = {
    // 8 canonical neuron presets from Eugene M. Izhikevich (2003)
    CELL_PRESETS: Object.freeze({
      RS: {
        id: 'RS',
        name: 'Regular Spiking',
        tag: 'Tonic adapting',
        a: 0.02,
        b: 0.20,
        c: -65.0,
        d: 8.0,
        I: 10.0,
      },
      IB: {
        id: 'IB',
        name: 'Intrinsically Bursting',
        tag: 'Phasic burst',
        a: 0.02,
        b: 0.20,
        c: -55.0,
        d: 4.0,
        I: 10.0,
      },
      CH: {
        id: 'CH',
        name: 'Chattering',
        tag: 'Rhythmic burst',
        a: 0.02,
        b: 0.20,
        c: -50.0,
        d: 2.0,
        I: 10.0,
      },
      FS: {
        id: 'FS',
        name: 'Fast Spiking',
        tag: 'Tonic high-freq',
        a: 0.10,
        b: 0.20,
        c: -65.0,
        d: 2.0,
        I: 10.0,
      },
      LTS: {
        id: 'LTS',
        name: 'Low-Threshold Spiking',
        tag: 'Tonic low-thresh',
        a: 0.02,
        b: 0.25,
        c: -65.0,
        d: 2.0,
        I: 10.0,
      },
      RZ: {
        id: 'RZ',
        name: 'Resonator',
        tag: 'Subthreshold osc',
        a: 0.10,
        b: 0.26,
        c: -65.0,
        d: 2.0,
        I: 10.0,
      },
      RE: {
        id: 'RE',
        name: 'Rebound Spiker',
        tag: 'Post-inhibitory',
        a: 0.03,
        b: 0.25,
        c: -60.0,
        d: 4.0,
        I: (t) => (t % 500 >= 100 && t % 500 < 300 ? -15.0 : 0.0),
      },
      MM: {
        id: 'MM',
        name: 'Mixed Mode',
        tag: 'Dual rhythm',
        a: 0.02,
        b: 0.20,
        c: -56.0,
        d: 2.0,
        I: 10.0,
      },
    }),

    /**
     * One step of the Izhikevich (2003) neuron without the spike reset.
     * Advances v by substeps Euler steps with u and current held,
     * then advances u by dt * a * (b * v - u).
     */
    izhikevichStep(v, u, current, a, b, dt = 1.0, substeps = 2) {
      if (a <= 0) {
        throw new Error(`a must be positive, got ${a}`);
      }
      if (substeps < 1 || !Number.isInteger(substeps)) {
        throw new Error(`substeps must be a positive integer, got ${substeps}`);
      }
      const h = dt / substeps;
      let vCurr = v;
      for (let s = 0; s < substeps; s++) {
        vCurr += h * (0.04 * vCurr * vCurr + 5.0 * vCurr + 140.0 - u + current);
      }
      const uNext = u + dt * a * (b * vCurr - u);
      return { v: vCurr, u: uNext };
    },

    /**
     * Spike reset step: if v >= vPeak, resets v to c and increments u by d.
     */
    izhikevichReset(v, u, vPeak = 30.0, c = -65.0, d = 8.0) {
      if (c >= vPeak) {
        throw new Error(`c must be less than vPeak (${vPeak}), got ${c}`);
      }
      if (v >= vPeak) {
        return { spike: true, v: c, u: u + d };
      }
      return { spike: false, v, u };
    },

    /**
     * Decompose one Izhikevich step into quadratic, linear, recovery and input increments.
     */
    izhikevichTerms(v, u, current, dt = 1.0, substeps = 2) {
      if (substeps < 1 || !Number.isInteger(substeps)) {
        throw new Error(`substeps must be a positive integer, got ${substeps}`);
      }
      const h = dt / substeps;
      let vCurr = v;
      let quad = 0;
      let lin = 0;
      let rec = 0;
      let inp = 0;
      for (let s = 0; s < substeps; s++) {
        quad += h * (0.04 * vCurr * vCurr);
        lin += h * (5.0 * vCurr + 140.0);
        rec += -h * u;
        inp += h * current;
        vCurr += h * (0.04 * vCurr * vCurr + 5.0 * vCurr + 140.0 - u + current);
      }
      return {
        quadratic: quad,
        linear: lin,
        recovery: rec,
        input: inp,
        total: quad + lin + rec + inp,
      };
    },

    /**
     * v-nullcline parabola: dv/dt = 0 => u = 0.04 v^2 + 5 v + 140 + I.
     */
    vNullcline(v, I) {
      return 0.04 * v * v + 5.0 * v + 140.0 + I;
    },

    /**
     * u-nullcline line: du/dt = 0 => u = b * v.
     */
    uNullcline(v, b) {
      return b * v;
    },

    /**
     * Analytical fixed points: roots of 0.04 v^2 + (5 - b) v + (140 + I) = 0.
     */
    fixedPoints(I, b) {
      const A = 0.04;
      const B = 5.0 - b;
      const C = 140.0 + I;
      const delta = B * B - 4 * A * C;
      if (delta < 0) {
        return {
          delta,
          hasEquilibrium: false,
          vRest: null,
          vThresh: null,
          uRest: null,
          uThresh: null,
        };
      }
      const sqrtD = Math.sqrt(delta);
      const vRest = (-B - sqrtD) / (2 * A);
      const vThresh = (-B + sqrtD) / (2 * A);
      const uRest = b * vRest;
      const uThresh = b * vThresh;
      return {
        delta,
        hasEquilibrium: true,
        vRest,
        vThresh,
        uRest,
        uThresh,
      };
    },

    /**
     * Compute pattern fingerprint from spike arrival times (ISI statistics).
     */
    computeFingerprint(spikes, durationMs = 1000) {
      const count = spikes.length;
      const rateHz = (count / durationMs) * 1000;
      if (count === 0) {
        return {
          count: 0,
          rateHz: 0,
          meanIsi: 0,
          cvIsi: 0,
          adaptationRatio: 1.0,
          burstRatio: 0,
          label: 'Quiescent',
          description: 'Subthreshold, no action potentials fired',
        };
      }
      if (count <= 3 && spikes[0] > 100) {
        const isis = [];
        for (let i = 1; i < count; i++) isis.push(spikes[i] - spikes[i - 1]);
        const meanIsi = isis.length ? isis.reduce((a, b) => a + b, 0) / isis.length : 0;
        return {
          count,
          rateHz,
          meanIsi,
          cvIsi: 0,
          adaptationRatio: 1.0,
          burstRatio: 0,
          label: 'Rebound Spiker (RE)',
          description: 'Fires post-inhibitory action potentials upon release',
        };
      }
      if (count === 1) {
        return {
          count: 1,
          rateHz,
          meanIsi: 0,
          cvIsi: 0,
          adaptationRatio: 1.0,
          burstRatio: 0,
          label: 'Phasic Spike',
          description: 'Single isolated action potential fired',
        };
      }

      const sortedSpikes = [...spikes].sort((a, b) => a - b);
      const isis = [];
      for (let i = 1; i < count; i++) {
        isis.push(Math.max(0.1, sortedSpikes[i] - sortedSpikes[i - 1]));
      }
      const meanIsi = isis.reduce((a, b) => a + b, 0) / isis.length;
      const variance =
        isis.reduce((acc, val) => acc + (val - meanIsi) ** 2, 0) / isis.length;
      const stdIsi = Math.sqrt(variance);
      const cvIsi = meanIsi > 0 ? stdIsi / meanIsi : 0;
      const firstIsi = isis[0];
      const lastIsi = isis[isis.length - 1];
      const adaptationRatio = firstIsi > 0 ? lastIsi / firstIsi : 1.0;

      const shortIsis = isis.filter((x) => x <= 12).length;
      const burstRatio = isis.length > 0 ? shortIsis / isis.length : 0;

      let label = 'Custom Pattern';
      let description = 'Tonic spiking activity';

      if (count >= 75) {
        label = 'Resonator (RZ)';
        description = 'High frequency firing with fast subthreshold recovery';
      } else if (cvIsi >= 0.80 && burstRatio >= 0.45) {
        label = 'Chattering (CH)';
        description = 'Stereotyped high-frequency bursts with quiescent intervals';
      } else if (cvIsi >= 0.45 && burstRatio >= 0.25) {
        label = 'Mixed Mode (MM)';
        description = 'Rhythmic combinations of bursts and regular intervals';
      } else if (
        firstIsi <= 10 &&
        lastIsi >= 30 &&
        burstRatio < 0.20 &&
        count < 35
      ) {
        label = 'Intrinsically Bursting (IB)';
        description = 'Initial high-frequency burst followed by regular single spikes';
      } else if (count >= 55 && meanIsi <= 18) {
        label = 'Fast Spiking (FS)';
        description = 'High-frequency firing with negligible frequency adaptation';
      } else if (count >= 35 && count < 55 && firstIsi <= 10 && meanIsi <= 28) {
        label = 'Low-Threshold Spiking (LTS)';
        description = 'Rapid initial spike pair with sustained moderate rate';
      } else if (count <= 25 && meanIsi >= 40 && burstRatio === 0) {
        label = 'Regular Spiking (RS)';
        description = 'Progressive spike-frequency adaptation (intervals lengthen)';
      } else if (cvIsi < 0.15) {
        label = 'Tonic Spiking';
        description = 'Regular periodic firing with steady inter-spike intervals';
      }

      return {
        count,
        rateHz,
        meanIsi,
        cvIsi,
        adaptationRatio,
        burstRatio,
        label,
        description,
      };
    },

    /**
     * Run a deterministic simulation over durationMs.
     */
    simulate(
      cellParams,
      durationMs = 1000,
      dt = 1.0,
      substeps = 2,
      currentOverride = null,
      vInit = -65.0,
      uInit = null
    ) {
      let v = vInit;
      let u = uInit !== null ? uInit : cellParams.b * vInit;
      const steps = Math.round(durationMs / dt);
      const history = [];
      const spikes = [];

      for (let i = 0; i < steps; i++) {
        const t = i * dt;
        let cur;
        if (currentOverride !== null) {
          cur = typeof currentOverride === 'function' ? currentOverride(t) : currentOverride;
        } else {
          cur = typeof cellParams.I === 'function' ? cellParams.I(t) : cellParams.I;
        }

        const stepRes = this.izhikevichStep(
          v,
          u,
          cur,
          cellParams.a,
          cellParams.b,
          dt,
          substeps
        );
        v = stepRes.v;
        u = stepRes.u;

        const resetRes = this.izhikevichReset(
          v,
          u,
          30.0,
          cellParams.c,
          cellParams.d
        );
        let isSpike = false;
        if (resetRes.spike) {
          spikes.push(t);
          isSpike = true;
          v = resetRes.v;
          u = resetRes.u;
        }

        history.push({ t, v, u, I: cur, isSpike });
      }

      const fingerprint = this.computeFingerprint(spikes, durationMs);
      return { history, spikes, fingerprint };
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
  };

  // ---- Helper to resolve complete design tokens including inset & elevated ----
  function resolveThemeColors(ctxColors) {
    const isDark =
      typeof document !== 'undefined' &&
      (document.documentElement.getAttribute('data-theme') === 'dark' ||
        (!document.documentElement.getAttribute('data-theme') &&
          typeof window !== 'undefined' &&
          window.matchMedia &&
          window.matchMedia('(prefers-color-scheme: dark)').matches));

    return {
      volt: ctxColors.volt || (isDark ? '#f59e0b' : '#b45309'),
      exc: ctxColors.exc || (isDark ? '#14b8a6' : '#0f766e'),
      inh: ctxColors.inh || (isDark ? '#fb7185' : '#be123c'),
      mod: ctxColors.mod || (isDark ? '#c084fc' : '#7e22ce'),
      trace: ctxColors.trace || (isDark ? '#60a5fa' : '#1d4ed8'),
      theory: ctxColors.theory || (isDark ? '#e9e4d8' : '#1b1a17'),
      paper: ctxColors.paper || (isDark ? '#15171a' : '#f6f2e9'),
      paperCard: ctxColors.paperCard || (isDark ? '#1c2024' : '#eeeae0'),
      paperElevated: isDark ? '#24292f' : '#ffffff',
      paperInset: isDark ? '#111316' : '#e7e2d6',
      ink: ctxColors.ink || (isDark ? '#e9e4d8' : '#1b1a17'),
      inkMuted: ctxColors.inkMuted || (isDark ? '#9da3ad' : '#57534e'),
      border: ctxColors.border || (isDark ? '#2e343b' : '#d6cfbf'),
      borderSubtle: isDark ? '#22262c' : '#e5dfd3',
      isDark,
    };
  }

  // ---- UI / Mount (atlas shell integration) ----
  function mount(rootEl, ctx) {
    if (!ctx || !ctx.colors) return;

    // Simulation state for all 8 zoo cells plus selected cell
    const zooKeys = ['RS', 'IB', 'CH', 'FS', 'LTS', 'RZ', 'RE', 'MM'];
    const zooState = {};

    for (const key of zooKeys) {
      const p = core.CELL_PRESETS[key];
      const preSim = core.simulate(p, 400, 1.0, 2);
      const histV = preSim.history.slice(-120).map((h) => h.v);
      const histU = preSim.history.slice(-120).map((h) => h.u);
      const lastState = preSim.history[preSim.history.length - 1];

      zooState[key] = {
        key,
        preset: p,
        v: lastState.v,
        u: lastState.u,
        I: typeof p.I === 'number' ? p.I : 10.0,
        historyV: histV,
        historyU: histU,
        spikes: preSim.spikes.filter((t) => t > 200),
        totalSpikes: preSim.spikes.length,
        lastSpikeTime: preSim.spikes[preSim.spikes.length - 1] || -100,
        fingerprint: preSim.fingerprint,
      };
    }

    const initialSim = core.simulate(core.CELL_PRESETS.RS, 600, 1.0, 2);
    const initHistV = initialSim.history.slice(-240).map((h) => h.v);
    const initHistU = initialSim.history.slice(-240).map((h) => h.u);
    const initHistI = initialSim.history.slice(-240).map((h) => h.I);
    const initLast = initialSim.history[initialSim.history.length - 1];

    const state = {
      selectedKey: 'RS',
      // Active parameters (editable via 2D maps)
      a: 0.02,
      b: 0.20,
      c: -65.0,
      d: 8.0,
      I_slider: 10.0,
      pulseRemainingMs: 0,
      theoryActive: false,
      hoverTerm: null,
      chapter: 1,
      predictAnswer: null,
      predictRevealed: false,
      isPaused: false,

      // Main stage running history
      mainV: initLast.v,
      mainU: initLast.u,
      mainHistoryV: initHistV,
      mainHistoryU: initHistU,
      mainHistoryI: initHistI,
      mainSpikes: initialSim.spikes.filter((t) => t > 300),
      mainTotalSpikes: initialSim.spikes.length,
      fingerprint: initialSim.fingerprint,

      // Interactive drag state for 2D parameter maps
      activeDrag: null, // 'ab' or 'cd'
      dragBoundsAB: null,
      dragBoundsCD: null,
      cardBounds: {},
      traceBounds: null,
    };

    const figureContainer = ctx.figure || rootEl;
    const marginContainer = ctx.margin || document.getElementById('atlas-margin');
    const figureSection =
      document.getElementById('atlas-figure') ||
      figureContainer.parentElement ||
      figureContainer;

    if (ctx.clock) {
      ctx.clock.stepsPerSecond = 60;
    }

    // Set canvas container height to fit in 800px window
    if (figureContainer) {
      figureContainer.style.height = '420px';
      figureContainer.style.maxHeight = '420px';
    }

    // 1. Build Marginalia (Equations, Live Terms, Fingerprint, Theory, Glossary)
    if (marginContainer) {
      marginContainer.innerHTML = `
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.eqTitle}</div>
          <div class="atlas-equation-box" id="n04-eq-box" style="font-size:12px; line-height:1.6;">
            <div>
              <span class="n04-term" data-term="vdot" style="color:var(--volt); cursor:pointer; font-weight:600;">v'</span> =
              <span class="n04-term" data-term="quad" style="color:var(--volt); cursor:pointer;">0.04·v²</span> +
              <span class="n04-term" data-term="lin" style="cursor:pointer;">5·v</span> +
              <span class="n04-term" data-term="const" style="cursor:pointer;">140</span>
              <span class="n04-term" data-term="u" style="color:var(--mod); cursor:pointer; font-weight:600;">- u</span> +
              <span class="n04-term" data-term="I" style="color:var(--exc); cursor:pointer; font-weight:600;">I</span>
            </div>
            <div>
              <span class="n04-term" data-term="udot" style="color:var(--mod); cursor:pointer; font-weight:600;">u'</span> =
              <span class="n04-term" data-term="a" style="cursor:pointer; font-weight:600;">a</span>·(
              <span class="n04-term" data-term="b" style="cursor:pointer; font-weight:600;">b</span>·v - u)
            </div>
            <div style="border-top:1px dashed var(--border); margin-top:3px; padding-top:3px; font-size:11px;">
              if v ≥ 30 mV:
              v ← <span class="n04-term" data-term="c" style="cursor:pointer; font-weight:600;">c</span>,
              u ← u + <span class="n04-term" data-term="d" style="cursor:pointer; font-weight:600;">d</span>
            </div>
          </div>
        </div>

        <div class="atlas-margin-section" id="n04-live-readouts">
          <div class="atlas-margin-label">${S.readoutSection}</div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif">${S.readoutV}:</span>
            <span class="atlas-margin-value atlas-mono" id="val-v" style="color:var(--volt); font-size:15px;">-65.0 mV</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif" style="color:var(--mod); font-weight:600;">${S.readoutU}:</span>
            <span class="atlas-margin-value atlas-mono" id="val-u" style="color:var(--mod);">-13.0 pA</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif" style="color:var(--exc); font-weight:600;">${S.readoutI}:</span>
            <span class="atlas-margin-value atlas-mono" id="val-I" style="color:var(--exc);">10.0 pA</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; border-top:1px dashed var(--border); padding-top:4px;">
            <span class="atlas-serif">${S.readoutDvdt}:</span>
            <span class="atlas-margin-value atlas-mono" id="val-dvdt">0.0 mV/ms</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span class="atlas-serif">${S.readoutDudt}:</span>
            <span class="atlas-margin-value atlas-mono" id="val-dudt">0.0 pA/ms</span>
          </div>
        </div>

        <div class="atlas-margin-section" id="n04-fingerprint-box">
          <div class="atlas-margin-label">${S.fingerprintSection}</div>
          <div style="margin-bottom:4px;">
            <span class="atlas-badge" id="val-fp-badge" style="background:var(--volt); color:var(--paper); font-size:11px; font-weight:600;">Regular Spiking (RS)</span>
          </div>
          <div style="font-size:11px; line-height:1.4; display:grid; grid-template-columns: 1fr 1fr; gap:4px;">
            <div><span class="atlas-serif">Rate:</span> <span class="atlas-mono" id="val-fp-rate">20.0 Hz</span></div>
            <div><span class="atlas-serif">Spikes:</span> <span class="atlas-mono" id="val-fp-spikes">20</span></div>
            <div><span class="atlas-serif">Mean ISI:</span> <span class="atlas-mono" id="val-fp-isi">51.6 ms</span></div>
            <div><span class="atlas-serif">ISI CV:</span> <span class="atlas-mono" id="val-fp-cv">0.15</span></div>
          </div>
          <div style="font-size:10px; color:var(--ink-muted); margin-top:4px;" id="val-fp-desc">
            Progressive spike-frequency adaptation (intervals lengthen)
          </div>
        </div>

        <div class="atlas-margin-section" id="n04-theory-box" style="display:none; font-size:11px;">
          <div class="atlas-margin-label" style="color:var(--theory);">${S.theorySection}</div>
          <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span class="atlas-serif">${S.theoryRest}:</span>
            <span class="atlas-mono" id="val-theory-rest">-70.0 mV</span>
          </div>
          <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span class="atlas-serif">${S.theoryThresh}:</span>
            <span class="atlas-mono" id="val-theory-thresh">-50.0 mV</span>
          </div>
          <div style="display:flex; justify-content:space-between; border-top:1px dashed var(--border); padding-top:3px;">
            <span class="atlas-serif">Δ:</span>
            <span class="atlas-mono" id="val-theory-delta">-0.96</span>
          </div>
          <div id="val-theory-status" style="font-size:10px; font-weight:600; margin-top:3px; color:var(--volt);">
            ${S.bifurcationActive}
          </div>
        </div>

        <div class="atlas-margin-section" style="font-size:11px; line-height:1.4;">
          <div class="atlas-margin-label">${S.glossarySection}</div>
          <p style="margin:2px 0 4px 0;">• ${S.glossaryU}</p>
          <p style="margin:0 0 4px 0;">• ${S.glossaryA}</p>
          <p style="margin:0 0 4px 0;">• ${S.glossaryB}</p>
          <p style="margin:0 0 4px 0;">• ${S.glossaryC}</p>
          <p style="margin:0 0 4px 0;">• ${S.glossaryD}</p>
          <p style="margin:0 0 2px 0;">• ${S.glossaryNullcline}</p>
        </div>
      `;

      marginContainer.querySelectorAll('.n04-term').forEach((el) => {
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
    let chapterBar = document.getElementById('n04-chapter-bar');
    if (!chapterBar) {
      chapterBar = document.createElement('div');
      chapterBar.id = 'n04-chapter-bar';
      chapterBar.style.cssText =
        'background:var(--paper-card); border:1px solid var(--border); border-radius:var(--radius); padding:8px 12px; margin-bottom:6px; display:flex; flex-direction:column; gap:4px;';
      figureSection.insertBefore(chapterBar, figureContainer);
    }

    chapterBar.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
          <span class="atlas-badge" id="n04-ch-badge" style="font-size:10px; padding:1px 5px;">${S.chapterLabel} 1 ${S.ofChapters}</span>
          <span class="atlas-serif" id="n04-ch-title" style="font-size:13px; font-weight:600;">${S.ch1Title}</span>
          <div id="n04-ch-dots" style="display:flex; gap:3px;">
            <span class="n04-dot" data-ch="1" style="cursor:pointer; font-size:11px;">●</span>
            <span class="n04-dot" data-ch="2" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
            <span class="n04-dot" data-ch="3" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
            <span class="n04-dot" data-ch="4" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
            <span class="n04-dot" data-ch="5" style="cursor:pointer; font-size:11px; opacity:0.4;">○</span>
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:4px; flex-shrink:0;">
          <button class="atlas-btn" id="n04-btn-ch-prev" disabled style="padding:2px 6px; font-size:11px;">${S.btnPrev}</button>
          <button class="atlas-btn" id="n04-btn-ch-next" style="padding:2px 6px; font-size:11px;">${S.btnNext}</button>
          <button class="atlas-btn" id="n04-btn-challenge" style="padding:2px 6px; font-size:11px;">${S.btnChallenge}</button>
        </div>
      </div>
      <div class="atlas-serif" id="n04-ch-body" style="font-size:12px; color:var(--ink); line-height:1.4;">
        ${S.ch1Body}
      </div>
      <div id="n04-predict-box" style="display:none; background:var(--paper-inset); border:1px solid var(--border); border-radius:3px; padding:6px 10px; margin-top:4px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; flex-wrap:wrap; gap:4px;">
          <span class="atlas-margin-label" style="font-size:10px; color:var(--volt);">${S.predictTitle}</span>
          <span class="atlas-serif" style="font-size:11px; color:var(--ink); font-weight:600;">${S.predictQuestion}</span>
        </div>
        <div style="display:flex; gap:10px; flex-wrap:wrap; font-size:11px; align-items:center;">
          <label style="cursor:pointer; display:flex; align-items:center; gap:4px;">
            <input type="radio" name="n04-predict" value="A" />
            <span>${S.predictOptA}</span>
          </label>
          <label style="cursor:pointer; display:flex; align-items:center; gap:4px;">
            <input type="radio" name="n04-predict" value="B" />
            <span><b>${S.predictOptB}</b></span>
          </label>
          <button class="atlas-btn" id="n04-btn-predict-submit" disabled style="padding:2px 8px; font-size:11px;">${S.predictSubmit}</button>
          <span id="n04-predict-feedback" style="display:none; font-size:11px; font-weight:600;"></span>
          <button class="atlas-btn" id="n04-btn-test-chattering" style="display:none; padding:2px 8px; font-size:11px; background:var(--volt); color:var(--paper);">${S.predictBtnTest}</button>
        </div>
      </div>
    `;

    // 3. Mount Compact Toolbar (Between chapter banner and canvas)
    let toolbar = document.getElementById('n04-toolbar');
    if (!toolbar) {
      toolbar = document.createElement('div');
      toolbar.id = 'n04-toolbar';
      toolbar.style.cssText =
        'display:flex; align-items:center; justify-content:space-between; gap:6px; margin-bottom:6px; flex-wrap:wrap;';
      figureSection.insertBefore(toolbar, figureContainer);
    }

    toolbar.innerHTML = `
      <div style="display:inline-flex; align-items:center; gap:4px; flex-wrap:wrap;">
        <span class="atlas-margin-label" style="font-size:10px; margin-right:2px;">Cell:</span>
        <button class="atlas-btn active n04-tab-btn" data-key="RS" style="padding:2px 6px; font-size:11px;">RS</button>
        <button class="atlas-btn n04-tab-btn" data-key="IB" style="padding:2px 6px; font-size:11px;">IB</button>
        <button class="atlas-btn n04-tab-btn" data-key="CH" style="padding:2px 6px; font-size:11px;">CH</button>
        <button class="atlas-btn n04-tab-btn" data-key="FS" style="padding:2px 6px; font-size:11px;">FS</button>
        <button class="atlas-btn n04-tab-btn" data-key="LTS" style="padding:2px 6px; font-size:11px;">LTS</button>
        <button class="atlas-btn n04-tab-btn" data-key="RZ" style="padding:2px 6px; font-size:11px;">RZ</button>
        <button class="atlas-btn n04-tab-btn" data-key="RE" style="padding:2px 6px; font-size:11px;">RE</button>
        <button class="atlas-btn n04-tab-btn" data-key="MM" style="padding:2px 6px; font-size:11px;">MM</button>
      </div>

      <div style="display:inline-flex; align-items:center; gap:6px; flex-wrap:wrap;">
        <button class="atlas-btn" id="n04-btn-stim" style="font-weight:600; padding:2px 8px; background:var(--volt); color:var(--paper); border-color:var(--volt); font-size:11px;">
          ${S.btnStimulate}
        </button>
        <div style="display:inline-flex; align-items:center; gap:3px;">
          <label for="n04-input-current" class="atlas-serif" style="font-size:11px; font-weight:600;">${S.labelCurrentSlider}:</label>
          <input type="range" id="n04-input-current" min="0" max="25" step="0.5" value="10" style="width:65px; cursor:pointer;" />
          <span class="atlas-mono" id="n04-txt-current" style="font-size:11px; min-width:38px;">10.0</span>
        </div>
        <button class="atlas-btn" id="n04-toggle-theory" aria-pressed="false" style="font-size:11px; padding:2px 6px;">
          ${S.toggleTheory}
        </button>
        <button class="atlas-btn" id="n04-btn-reset" style="font-size:11px; padding:2px 6px;">
          ${S.btnReset}
        </button>
      </div>
    `;

    // 4. Main Canvas inside figureContainer
    const canvas = figureContainer.querySelector('canvas') || ctx.canvas();
    const cContext = canvas.getContext('2d');

    // DOM References
    const btnStim = document.getElementById('n04-btn-stim');
    const inputCurrent = document.getElementById('n04-input-current');
    const txtCurrent = document.getElementById('n04-txt-current');
    const toggleTheory = document.getElementById('n04-toggle-theory');
    const btnReset = document.getElementById('n04-btn-reset');
    const theoryBox = document.getElementById('n04-theory-box');

    const chBadge = document.getElementById('n04-ch-badge');
    const chTitle = document.getElementById('n04-ch-title');
    const chBody = document.getElementById('n04-ch-body');
    const btnChPrev = document.getElementById('n04-btn-ch-prev');
    const btnChNext = document.getElementById('n04-btn-ch-next');
    const btnChallenge = document.getElementById('n04-btn-challenge');
    const predictBox = document.getElementById('n04-predict-box');

    const predictOptions = document.querySelectorAll('input[name="n04-predict"]');
    const btnPredictSubmit = document.getElementById('n04-btn-predict-submit');
    const txtPredictFeedback = document.getElementById('n04-predict-feedback');
    const btnTestChattering = document.getElementById('n04-btn-test-chattering');

    // Select cell helper
    const selectCell = (key) => {
      if (!core.CELL_PRESETS[key]) return;
      state.selectedKey = key;
      const p = core.CELL_PRESETS[key];
      state.a = p.a;
      state.b = p.b;
      state.c = p.c;
      state.d = p.d;
      if (typeof p.I === 'number') {
        state.I_slider = p.I;
        inputCurrent.value = String(p.I);
        txtCurrent.textContent = p.I.toFixed(1);
      }

      document.querySelectorAll('.n04-tab-btn').forEach((btn) => {
        btn.classList.toggle('active', btn.getAttribute('data-key') === key);
      });

      // Reset main stage history to preset with prefilled simulation
      const cellSim = core.simulate(p, 600, 1.0, 2);
      state.mainV = cellSim.history[cellSim.history.length - 1].v;
      state.mainU = cellSim.history[cellSim.history.length - 1].u;
      state.mainHistoryV = cellSim.history.slice(-240).map((h) => h.v);
      state.mainHistoryU = cellSim.history.slice(-240).map((h) => h.u);
      state.mainHistoryI = cellSim.history.slice(-240).map((h) => h.I);
      const shift = simTimeMs - 600;
      state.mainSpikes = cellSim.spikes
        .map((t) => t + shift)
        .filter((t) => t > simTimeMs - 400);
      state.mainTotalSpikes = cellSim.spikes.length;
    };

    document.querySelectorAll('.n04-tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        selectCell(btn.getAttribute('data-key'));
      });
    });

    // Trigger stimulus pulse helper
    const triggerStimulate = () => {
      state.pulseRemainingMs = 4.0;
      btnStim.textContent = S.btnStimulating;
      setTimeout(() => {
        btnStim.textContent = S.btnStimulate;
      }, 400);
    };

    btnStim.addEventListener('click', triggerStimulate);

    inputCurrent.addEventListener('input', (e) => {
      state.I_slider = parseFloat(e.target.value);
      txtCurrent.textContent = state.I_slider.toFixed(1);
    });

    toggleTheory.addEventListener('click', () => {
      state.theoryActive = !state.theoryActive;
      toggleTheory.classList.toggle('active', state.theoryActive);
      toggleTheory.setAttribute('aria-pressed', String(state.theoryActive));
      if (theoryBox) {
        theoryBox.style.display = state.theoryActive ? 'block' : 'none';
      }
    });

    btnReset.addEventListener('click', () => {
      selectCell(state.selectedKey);
    });

    btnChallenge.addEventListener('click', () => {
      predictBox.style.display =
        predictBox.style.display === 'none' ? 'block' : 'none';
    });

    // Chapters setup
    const chapters = [
      {
        num: 1,
        title: S.ch1Title,
        body: S.ch1Body,
        setup: () => {
          selectCell('RS');
          predictBox.style.display = 'none';
        },
      },
      {
        num: 2,
        title: S.ch2Title,
        body: S.ch2Body,
        setup: () => {
          selectCell('FS');
          predictBox.style.display = 'none';
        },
      },
      {
        num: 3,
        title: S.ch3Title,
        body: S.ch3Body,
        setup: () => {
          selectCell('CH');
          predictBox.style.display = 'block';
        },
      },
      {
        num: 4,
        title: S.ch4Title,
        body: S.ch4Body,
        setup: () => {
          selectCell('RZ');
          predictBox.style.display = 'none';
        },
      },
      {
        num: 5,
        title: S.ch5Title,
        body: S.ch5Body,
        setup: () => {
          predictBox.style.display = 'none';
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

      const dots = document.querySelectorAll('.n04-dot');
      dots.forEach((d, i) => {
        d.textContent = i + 1 === idx ? '●' : '○';
        d.style.opacity = i + 1 === idx ? '1.0' : '0.4';
      });

      ch.setup();
    };

    btnChPrev.addEventListener('click', () => {
      if (state.chapter > 1) applyChapter(state.chapter - 1);
    });

    btnChNext.addEventListener('click', () => {
      if (state.chapter < chapters.length) applyChapter(state.chapter + 1);
    });

    document.querySelectorAll('.n04-dot').forEach((d) => {
      d.addEventListener('click', () => {
        const c = parseInt(d.getAttribute('data-ch'), 10);
        if (c >= 1 && c <= chapters.length) applyChapter(c);
      });
    });

    // Predict-then-reveal logic
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
      if (state.predictAnswer === 'B') {
        txtPredictFeedback.style.color = 'var(--exc)';
        txtPredictFeedback.textContent = S.predictExplanation;
      } else {
        txtPredictFeedback.style.color = 'var(--inh)';
        txtPredictFeedback.textContent = S.predictWrong;
      }
      btnTestChattering.style.display = 'inline-block';
    });

    btnTestChattering.addEventListener('click', () => {
      selectCell('CH');
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'ArrowRight' && !e.shiftKey) {
        if (state.chapter < chapters.length) applyChapter(state.chapter + 1);
      } else if (e.key === 'ArrowLeft' && !e.shiftKey) {
        if (state.chapter > 1) applyChapter(state.chapter - 1);
      } else if (e.key === 't' || e.key === 'T') {
        toggleTheory.click();
      } else if (e.key === 's' || e.key === 'S') {
        triggerStimulate();
      } else if (e.key === 'p' || e.key === 'P') {
        btnChallenge.click();
      } else if (e.key === ' ') {
        e.preventDefault();
        state.isPaused = !state.isPaused;
      } else if (e.key >= '1' && e.key <= '8') {
        const idx = parseInt(e.key, 10) - 1;
        if (zooKeys[idx]) selectCell(zooKeys[idx]);
      }
    });

    // Mouse and touch interaction for 2-D maps and cards on the canvas
    const getCanvasPos = (evt) => {
      const rect = canvas.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const y = evt.clientY - rect.top;
      return { x, y };
    };

    canvas.addEventListener('pointerdown', (e) => {
      const { x, y } = getCanvasPos(e);

      // 1. Check (a vs b) map
      if (state.dragBoundsAB) {
        const b = state.dragBoundsAB;
        if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
          state.activeDrag = 'ab';
          canvas.setPointerCapture(e.pointerId);
          updateDragAB(x, y);
          return;
        }
      }

      // 2. Check (c vs d) map
      if (state.dragBoundsCD) {
        const b = state.dragBoundsCD;
        if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
          state.activeDrag = 'cd';
          canvas.setPointerCapture(e.pointerId);
          updateDragCD(x, y);
          return;
        }
      }

      // 3. Check Zoo cards
      for (const key of zooKeys) {
        const b = state.cardBounds[key];
        if (b && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
          selectCell(key);
          return;
        }
      }

      // 4. Check V(t) trace click -> stimulate
      if (state.traceBounds) {
        const b = state.traceBounds;
        if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
          triggerStimulate();
        }
      }
    });

    const updateDragAB = (x, y) => {
      const b = state.dragBoundsAB;
      if (!b) return;
      const normX = Math.max(0, Math.min(1, (x - b.x) / b.w));
      const normY = Math.max(0, Math.min(1, 1 - (y - b.y) / b.h));
      // a in [0.01, 0.15], b in [0.05, 0.35]
      state.a = 0.01 + normX * (0.15 - 0.01);
      state.b = 0.05 + normY * (0.35 - 0.05);
    };

    const updateDragCD = (x, y) => {
      const b = state.dragBoundsCD;
      if (!b) return;
      const normX = Math.max(0, Math.min(1, (x - b.x) / b.w));
      const normY = Math.max(0, Math.min(1, 1 - (y - b.y) / b.h));
      // c in [-75, -45], d in [0, 10]
      state.c = -75.0 + normX * (-45.0 - -75.0);
      state.d = 0.0 + normY * 10.0;
    };

    canvas.addEventListener('pointermove', (e) => {
      if (!state.activeDrag) {
        const { x, y } = getCanvasPos(e);
        let pointer = false;
        if (
          (state.dragBoundsAB &&
            x >= state.dragBoundsAB.x &&
            x <= state.dragBoundsAB.x + state.dragBoundsAB.w &&
            y >= state.dragBoundsAB.y &&
            y <= state.dragBoundsAB.y + state.dragBoundsAB.h) ||
          (state.dragBoundsCD &&
            x >= state.dragBoundsCD.x &&
            x <= state.dragBoundsCD.x + state.dragBoundsCD.w &&
            y >= state.dragBoundsCD.y &&
            y <= state.dragBoundsCD.y + state.dragBoundsCD.h)
        ) {
          pointer = true;
        } else {
          for (const key of zooKeys) {
            const b = state.cardBounds[key];
            if (b && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
              pointer = true;
              break;
            }
          }
        }
        canvas.style.cursor = pointer ? 'pointer' : 'default';
        return;
      }
      const { x, y } = getCanvasPos(e);
      if (state.activeDrag === 'ab') {
        updateDragAB(x, y);
      } else if (state.activeDrag === 'cd') {
        updateDragCD(x, y);
      }
    });

    const endDrag = (e) => {
      if (state.activeDrag) {
        state.activeDrag = null;
        try {
          canvas.releasePointerCapture(e.pointerId);
        } catch (_) {}
      }
    };

    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    // 5. Simulation & Animation Loop (driven by ctx.onFrame)
    const dt = 1.0; // 1 ms standard timestep
    const substeps = 2; // 2 half-steps per ms as in Izhikevich (2003)
    const stepsPerFrame = 4; // advance 4 ms per animation frame for fluid live spiking
    let simTimeMs = 600; // start aligned with 600 ms prefilled history

    ctx.onFrame(() => {
      if (!state.isPaused) {
        for (let s = 0; s < stepsPerFrame; s++) {
          simTimeMs += dt;

          // 1. Advance the 8 Zoo cells
          for (const key of zooKeys) {
            const cell = zooState[key];
            const p = cell.preset;
            let cur = typeof p.I === 'function' ? p.I(simTimeMs) : p.I;

            const resStep = core.izhikevichStep(
              cell.v,
              cell.u,
              cur,
              p.a,
              p.b,
              dt,
              substeps
            );
            cell.v = resStep.v;
            cell.u = resStep.u;

            const resReset = core.izhikevichReset(
              cell.v,
              cell.u,
              30.0,
              p.c,
              p.d
            );
            if (resReset.spike) {
              cell.spikes.push(simTimeMs);
              cell.totalSpikes += 1;
              cell.lastSpikeTime = simTimeMs;
              cell.historyV.push(30.0);
              cell.historyU.push(cell.u);
              cell.v = resReset.v;
              cell.u = resReset.u;
            }

            // Keep recent history for mini waveforms
            cell.historyV.push(cell.v);
            if (cell.historyV.length > 120) cell.historyV.shift();
            cell.historyU.push(cell.u);
            if (cell.historyU.length > 120) cell.historyU.shift();

            // Keep only recent spikes in window (1000 ms)
            while (cell.spikes.length && simTimeMs - cell.spikes[0] > 1000) {
              cell.spikes.shift();
            }
          }

          // 2. Advance the Main Stage (selected cell with live parameters a, b, c, d)
          let mainCur = state.I_slider;
          if (state.pulseRemainingMs > 0) {
            mainCur += 20.0;
            state.pulseRemainingMs -= dt;
          }
          if (state.selectedKey === 'RE' && typeof core.CELL_PRESETS.RE.I === 'function') {
            mainCur = core.CELL_PRESETS.RE.I(simTimeMs);
            if (state.pulseRemainingMs > 0) mainCur += 20.0;
          }

          const mStep = core.izhikevichStep(
            state.mainV,
            state.mainU,
            mainCur,
            state.a,
            state.b,
            dt,
            substeps
          );
          state.mainV = mStep.v;
          state.mainU = mStep.u;

          const mReset = core.izhikevichReset(
            state.mainV,
            state.mainU,
            30.0,
            state.c,
            state.d
          );
          if (mReset.spike) {
            state.mainSpikes.push(simTimeMs);
            state.mainTotalSpikes += 1;
            state.mainHistoryV.push(30.0);
            state.mainHistoryU.push(state.mainU);
            state.mainHistoryI.push(mainCur);
            state.mainV = mReset.v;
            state.mainU = mReset.u;
          }

          state.mainHistoryV.push(state.mainV);
          if (state.mainHistoryV.length > 240) state.mainHistoryV.shift();
          state.mainHistoryU.push(state.mainU);
          if (state.mainHistoryU.length > 240) state.mainHistoryU.shift();
          state.mainHistoryI.push(mainCur);
          if (state.mainHistoryI.length > 240) state.mainHistoryI.shift();

          while (state.mainSpikes.length && simTimeMs - state.mainSpikes[0] > 1000) {
            state.mainSpikes.shift();
          }
        }
      }

      // Compute live fingerprint from current window
      state.fingerprint = core.computeFingerprint(state.mainSpikes, 1000);

      // Update marginalia DOM readouts
      updateMarginaliaDOM(state, core);

      // Render the entire plate on canvas with resolved theme colors
      const themeColors = resolveThemeColors(ctx.colors());
      renderCanvas(cContext, canvas, state, zooState, zooKeys, themeColors);
    });
  }

  // Update DOM readouts in the marginalia column
  function updateMarginaliaDOM(state, core) {
    const elV = document.getElementById('val-v');
    const elU = document.getElementById('val-u');
    const elI = document.getElementById('val-I');
    const elDvdt = document.getElementById('val-dvdt');
    const elDudt = document.getElementById('val-dudt');

    if (elV) elV.textContent = `${state.mainV.toFixed(1)} mV`;
    if (elU) elU.textContent = `${state.mainU.toFixed(1)} pA`;
    if (elI) elI.textContent = `${state.I_slider.toFixed(1)} pA`;

    const dv =
      0.04 * state.mainV * state.mainV +
      5.0 * state.mainV +
      140.0 -
      state.mainU +
      state.I_slider;
    const du = state.a * (state.b * state.mainV - state.mainU);
    if (elDvdt) elDvdt.textContent = `${dv >= 0 ? '+' : ''}${dv.toFixed(1)} mV/ms`;
    if (elDudt) elDudt.textContent = `${du >= 0 ? '+' : ''}${du.toFixed(2)} pA/ms`;

    const fp = state.fingerprint;
    const elFpBadge = document.getElementById('val-fp-badge');
    const elFpRate = document.getElementById('val-fp-rate');
    const elFpSpikes = document.getElementById('val-fp-spikes');
    const elFpIsi = document.getElementById('val-fp-isi');
    const elFpCv = document.getElementById('val-fp-cv');
    const elFpDesc = document.getElementById('val-fp-desc');

    if (elFpBadge) elFpBadge.textContent = fp.label;
    if (elFpRate) elFpRate.textContent = `${fp.rateHz.toFixed(1)} Hz`;
    if (elFpSpikes) elFpSpikes.textContent = String(fp.count);
    if (elFpIsi) elFpIsi.textContent = `${fp.meanIsi ? fp.meanIsi.toFixed(1) : '—'} ms`;
    if (elFpCv) elFpCv.textContent = fp.meanIsi ? fp.cvIsi.toFixed(2) : '—';
    if (elFpDesc) elFpDesc.textContent = fp.description;

    // Theory readouts
    const fix = core.fixedPoints(state.I_slider, state.b);
    const elTRest = document.getElementById('val-theory-rest');
    const elTThresh = document.getElementById('val-theory-thresh');
    const elTDelta = document.getElementById('val-theory-delta');
    const elTStatus = document.getElementById('val-theory-status');

    if (elTRest) {
      elTRest.textContent = fix.hasEquilibrium ? `${fix.vRest.toFixed(1)} mV` : 'None';
    }
    if (elTThresh) {
      elTThresh.textContent = fix.hasEquilibrium ? `${fix.vThresh.toFixed(1)} mV` : 'None';
    }
    if (elTDelta) elTDelta.textContent = fix.delta.toFixed(2);
    if (elTStatus) {
      elTStatus.textContent = fix.hasEquilibrium
        ? S.bifurcationRest
        : S.bifurcationActive;
      elTStatus.style.color = fix.hasEquilibrium ? 'var(--exc)' : 'var(--volt)';
    }
  }

  // ---- Main Canvas Rendering Function ----
  function renderCanvas(ctx, canvas, state, zooState, zooKeys, colors) {
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width || canvas.width / dpr || 800;
    const H = rect.height || canvas.height / dpr || 420;

    // Clear background
    ctx.fillStyle = colors.paper;
    ctx.fillRect(0, 0, W, H);

    const isStacked = W < 640;

    if (!isStacked) {
      // Desktop: Side-by-side layout (Main Stage on left 58%, Zoo Grid on right 42%)
      const stageW = Math.round(W * 0.58) - 8;
      const zooX = stageW + 16;
      const zooW = W - zooX;

      // 1. Draw Main Stage (Oscilloscope trace, Phase portrait, 2D Maps)
      drawMainStage(ctx, 0, 0, stageW, H, state, colors);

      // 2. Draw Zoo Grid of 8 live cells
      drawZooGrid(ctx, zooX, 0, zooW, H, state, zooState, zooKeys, colors, false);
    } else {
      // Mobile / narrow (< 640px): Full-width Main Stage; cell tabs above handle selection
      drawMainStage(ctx, 0, 0, W, H, state, colors);
    }
  }

  // Draw Main Stage (selected cell enlarged)
  function drawMainStage(ctx, x, y, w, h, state, colors) {
    const gap = 8;
    const traceH = Math.round(h * 0.52);
    const bottomY = y + traceH + gap;
    const bottomH = h - traceH - gap;

    // Top: Big V(t) trace oscilloscope
    drawOscilloscopeTrace(ctx, x, y, w, traceH, state, colors);
    state.traceBounds = { x, y, w, h: traceH };

    // Bottom half: Left side is Phase plane (v,u), Right side is 2-D Parameter maps
    const halfW = Math.round((w - gap) / 2);
    const phaseX = x;
    const mapsX = x + halfW + gap;

    drawPhasePortrait(ctx, phaseX, bottomY, halfW, bottomH, state, colors);
    drawParameterMaps(ctx, mapsX, bottomY, halfW, bottomH, state, colors);
  }

  // Draw Big V(t) Oscilloscope Trace
  function drawOscilloscopeTrace(ctx, x, y, w, h, state, colors) {
    // Card background
    ctx.fillStyle = colors.paperCard;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    // Title & Header badge
    ctx.fillStyle = colors.ink;
    ctx.font = 'bold 12px ui-sans-serif, system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const cellName =
      core.CELL_PRESETS[state.selectedKey] ? core.CELL_PRESETS[state.selectedKey].name : 'Custom';
    ctx.fillText(`${cellName} · V(t) Membrane Trace`, x + 10, y + 8);

    ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = colors.volt;
    ctx.fillText(`${state.mainV.toFixed(1)} mV`, x + w - 10, y + 8);

    // Waveform plot area
    const padL = 44;
    const padR = 10;
    const padT = 26;
    const padB = 22;
    const plotX = x + padL;
    const plotY = y + padT;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    if (plotW <= 10 || plotH <= 10) return;

    // Voltage grid lines: -65 (rest), -50 (threshold), 0, +30 (peak)
    const vMin = -85;
    const vMax = 35;
    const toY = (v) => plotY + plotH - ((v - vMin) / (vMax - vMin)) * plotH;

    const gridVoltages = [
      { v: -65, label: '-65', tag: 'Rest', color: colors.border },
      { v: -50, label: '-50', tag: 'Thresh', color: colors.exc },
      { v: 0, label: '0', tag: '', color: colors.borderSubtle },
      { v: 30, label: '+30', tag: 'Peak', color: colors.volt },
    ];

    ctx.lineWidth = 1;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (const g of gridVoltages) {
      const gy = toY(g.v);
      ctx.strokeStyle = g.color;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(plotX, gy);
      ctx.lineTo(plotX + plotW, gy);
      ctx.stroke();

      ctx.fillStyle = colors.inkMuted;
      ctx.fillText(g.label, plotX - 5, gy);

      if (g.tag) {
        ctx.fillStyle = g.color;
        ctx.textAlign = 'left';
        ctx.font = '9px ui-sans-serif, system-ui';
        ctx.fillText(g.tag, plotX + 6, gy - 6);
        ctx.textAlign = 'right';
        ctx.font = '10px ui-monospace, monospace';
      }
    }
    ctx.setLineDash([]);

    // Reset voltage c line (marker)
    const cy = toY(state.c);
    ctx.strokeStyle = colors.mod;
    ctx.lineWidth = 1;
    ctx.setLineDash([1, 3]);
    ctx.beginPath();
    ctx.moveTo(plotX, cy);
    ctx.lineTo(plotX + plotW, cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Only draw separate label if c is distinct from rest (-65) and threshold (-50)
    if (Math.abs(state.c - -65) > 3 && Math.abs(state.c - -50) > 3) {
      ctx.fillStyle = colors.mod;
      ctx.textAlign = 'right';
      ctx.font = '9px ui-sans-serif, system-ui';
      ctx.fillText(`Reset c = ${state.c.toFixed(0)}`, plotX + plotW - 6, cy - 6);
    }

    // Draw V(t) waveform curve
    const history = state.mainHistoryV;
    const len = history.length;
    if (len > 1) {
      ctx.strokeStyle = colors.volt;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < len; i++) {
        const px = plotX + (i / (len - 1)) * plotW;
        const py = toY(history[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Spark glow on live head
      const headX = plotX + plotW;
      const headY = toY(history[len - 1]);
      ctx.fillStyle = colors.volt;
      ctx.beginPath();
      ctx.arc(headX, headY, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Injected current drive strip at bottom
    const curStripH = 4;
    const curY = plotY + plotH + 4;
    ctx.fillStyle = colors.paperInset;
    ctx.fillRect(plotX, curY, plotW, curStripH);
    const iHist = state.mainHistoryI;
    if (iHist && iHist.length > 1) {
      ctx.fillStyle = colors.exc;
      for (let i = 0; i < iHist.length; i++) {
        if (iHist[i] > 1) {
          const bx = plotX + (i / (iHist.length - 1)) * plotW;
          ctx.fillRect(bx, curY, Math.max(1, plotW / iHist.length), curStripH);
        }
      }
    }
  }

  // Draw Phase Portrait (v, u) plane with theory nullclines and trajectory
  function drawPhasePortrait(ctx, x, y, w, h, state, colors) {
    ctx.fillStyle = colors.paperCard;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    // Title
    ctx.fillStyle = colors.ink;
    ctx.font = 'bold 11px ui-sans-serif, system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('(v, u) Phase Plane', x + 8, y + 6);

    const padL = 28;
    const padR = 8;
    const padT = 20;
    const padB = 20;
    const pw = w - padL - padR;
    const ph = h - padT - padB;
    if (pw <= 10 || ph <= 10) return;

    const px = x + padL;
    const py = y + padT;

    const vMin = -85;
    const vMax = 35;
    const uMin = -25;
    const uMax = 35;

    const mapX = (v) => px + ((v - vMin) / (vMax - vMin)) * pw;
    const mapY = (u) => py + ph - ((u - uMin) / (uMax - uMin)) * ph;

    // Axes
    ctx.strokeStyle = colors.borderSubtle;
    ctx.lineWidth = 1;
    ctx.beginPath();
    // v = 0 axis
    const zX = mapX(0);
    ctx.moveTo(zX, py);
    ctx.lineTo(zX, py + ph);
    // u = 0 axis
    const zY = mapY(0);
    ctx.moveTo(px, zY);
    ctx.lineTo(px + pw, zY);
    ctx.stroke();

    // Axis tick labels
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillStyle = colors.inkMuted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('-65', mapX(-65), py + ph + 3);
    ctx.fillText('0', zX, py + ph + 3);
    ctx.fillText('+30', mapX(30), py + ph + 3);

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('0', px - 3, zY);
    ctx.fillText('20', px - 3, mapY(20));

    // Theory Ghost: nullclines
    if (state.theoryActive) {
      // 1. v-nullcline parabola: u = 0.04 v^2 + 5 v + 140 + I
      ctx.strokeStyle = colors.theory;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      let first = true;
      for (let v = vMin; v <= 30; v += 1) {
        const u = core.vNullcline(v, state.I_slider);
        if (u >= uMin - 10 && u <= uMax + 10) {
          const vx = mapX(v);
          const vy = mapY(u);
          if (first) {
            ctx.moveTo(vx, vy);
            first = false;
          } else {
            ctx.lineTo(vx, vy);
          }
        }
      }
      ctx.stroke();

      // 2. u-nullcline line: u = b * v
      ctx.beginPath();
      const u1 = core.uNullcline(vMin, state.b);
      const u2 = core.uNullcline(vMax, state.b);
      ctx.moveTo(mapX(vMin), mapY(u1));
      ctx.lineTo(mapX(vMax), mapY(u2));
      ctx.stroke();
      ctx.setLineDash([]);

      // 3. Fixed points
      const fix = core.fixedPoints(state.I_slider, state.b);
      if (fix.hasEquilibrium) {
        // Stable node (Rest)
        ctx.fillStyle = colors.exc;
        ctx.beginPath();
        ctx.arc(mapX(fix.vRest), mapY(fix.uRest), 3.5, 0, Math.PI * 2);
        ctx.fill();

        // Saddle point (Threshold)
        ctx.strokeStyle = colors.inh;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(mapX(fix.vThresh), mapY(fix.uThresh), 3.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Trajectory (v(t), u(t))
    const hV = state.mainHistoryV;
    const hU = state.mainHistoryU;
    if (hV.length > 2) {
      ctx.strokeStyle = colors.volt;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let prevV = hV[0];
      let isFirst = true;
      for (let i = 0; i < hV.length; i++) {
        const curV = hV[i];
        const curU = hU[i];
        // If spike reset jump occurred, lift pen cleanly
        if (prevV > 20 && curV < 0) {
          ctx.stroke();
          ctx.beginPath();
          isFirst = true;
        }
        const vx = mapX(curV);
        const vy = mapY(curU);
        if (isFirst) {
          ctx.moveTo(vx, vy);
          isFirst = false;
        } else {
          ctx.lineTo(vx, vy);
        }
        prevV = curV;
      }
      ctx.stroke();

      // Current point (live state)
      const curX = mapX(state.mainV);
      const curY = mapY(state.mainU);
      ctx.fillStyle = colors.volt;
      ctx.beginPath();
      ctx.arc(curX, curY, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Reset Jump Arrow from (30, u) to (c, u+d)
    const resetFromX = mapX(30);
    const resetFromY = mapY(state.mainU);
    const resetToX = mapX(state.c);
    const resetToY = mapY(state.mainU + state.d);

    ctx.strokeStyle = colors.mod;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(resetFromX, resetFromY);
    ctx.lineTo(resetToX, resetToY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Small arrowhead pointing to reset target
    const angle = Math.atan2(resetToY - resetFromY, resetToX - resetFromX);
    const arrowLen = 5;
    ctx.fillStyle = colors.mod;
    ctx.beginPath();
    ctx.moveTo(resetToX, resetToY);
    ctx.lineTo(
      resetToX - arrowLen * Math.cos(angle - Math.PI / 6),
      resetToY - arrowLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      resetToX - arrowLen * Math.cos(angle + Math.PI / 6),
      resetToY - arrowLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
  }

  // Draw 2-D Parameter Maps: (a vs b) square and (c vs d) square
  function drawParameterMaps(ctx, x, y, w, h, state, colors) {
    ctx.fillStyle = colors.paperCard;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    // Title
    ctx.fillStyle = colors.ink;
    ctx.font = 'bold 11px ui-sans-serif, system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('2D Parameter Maps (Drag)', x + 8, y + 6);

    const pad = 8;
    const contentY = y + 20;
    const contentH = h - 24;
    const mapW = Math.round((w - pad * 3) / 2);
    const mapH = contentH - 8;
    if (mapW <= 10 || mapH <= 10) return;

    // Map 1: (a vs b) square
    const m1X = x + pad;
    const m1Y = contentY;
    drawMapSquare(
      ctx,
      m1X,
      m1Y,
      mapW,
      mapH,
      'a-vs-b',
      'a (rate)',
      'b (coupl)',
      state.a,
      0.01,
      0.15,
      state.b,
      0.05,
      0.35,
      colors.exc,
      colors
    );
    state.dragBoundsAB = { x: m1X, y: m1Y, w: mapW, h: mapH };

    // Map 2: (c vs d) square
    const m2X = m1X + mapW + pad;
    const m2Y = contentY;
    drawMapSquare(
      ctx,
      m2X,
      m2Y,
      mapW,
      mapH,
      'c-vs-d',
      'c (reset)',
      'd (jump)',
      state.c,
      -75.0,
      -45.0,
      state.d,
      0.0,
      10.0,
      colors.mod,
      colors
    );
    state.dragBoundsCD = { x: m2X, y: m2Y, w: mapW, h: mapH };
  }

  // Helper to draw a single 2-D parameter box with draggable handle
  function drawMapSquare(
    ctx,
    x,
    y,
    w,
    h,
    label,
    xLabel,
    yLabel,
    valX,
    minX,
    maxX,
    valY,
    minY,
    maxY,
    accentColor,
    colors
  ) {
    // Inset background
    ctx.fillStyle = colors.paperInset;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    // Grid center lines
    ctx.strokeStyle = colors.borderSubtle;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(x + w / 2, y);
    ctx.lineTo(x + w / 2, y + h);
    ctx.moveTo(x, y + h / 2);
    ctx.lineTo(x + w, y + h / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Compute handle position
    const normX = (valX - minX) / (maxX - minX);
    const normY = (valY - minY) / (maxY - minY);
    const hx = x + Math.max(0, Math.min(1, normX)) * w;
    const hy = y + h - Math.max(0, Math.min(1, normY)) * h;

    // Draggable Handle
    ctx.fillStyle = accentColor;
    ctx.beginPath();
    ctx.arc(hx, hy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = colors.paper;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Value label with parameter names
    ctx.fillStyle = colors.ink;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const desc = xLabel.startsWith('a')
      ? `a=${valX.toFixed(2)}, b=${valY.toFixed(2)}`
      : `c=${valX.toFixed(0)}, d=${valY.toFixed(1)}`;
    ctx.fillText(desc, x + w / 2, y + h - 2);

    // Axis tags
    ctx.fillStyle = colors.inkMuted;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(label, x + 3, y + 2);
  }

  // Draw Zoo Grid of 8 live cells (2 columns x 4 rows)
  function drawZooGrid(
    ctx,
    x,
    y,
    w,
    h,
    state,
    zooState,
    zooKeys,
    colors,
    stacked
  ) {
    const cols = 2;
    const rows = 4;
    const gap = 6;
    const cardW = Math.round((w - gap * (cols - 1)) / cols);
    const cardH = Math.round((h - gap * (rows - 1)) / rows);

    state.cardBounds = {};

    for (let i = 0; i < zooKeys.length; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const cx = x + col * (cardW + gap);
      const cy = y + row * (cardH + gap);

      const key = zooKeys[i];
      const cell = zooState[key];
      const isSelected = key === state.selectedKey;

      state.cardBounds[key] = { x: cx, y: cy, w: cardW, h: cardH };

      // Card Box
      ctx.fillStyle = isSelected ? colors.paperElevated : colors.paperCard;
      ctx.fillRect(cx, cy, cardW, cardH);
      ctx.strokeStyle = isSelected ? colors.volt : colors.border;
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.strokeRect(cx, cy, cardW, cardH);

      // Card Header
      ctx.font = 'bold 10px ui-sans-serif, system-ui';
      ctx.fillStyle = isSelected ? colors.volt : colors.ink;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`${cell.preset.id} · ${cell.preset.name}`, cx + 6, cy + 4);

      // Mini wave plot area
      const waveX = cx + 6;
      const waveY = cy + 16;
      const waveW = Math.round(cardW * 0.65);
      const waveH = cardH - 20;

      const hist = cell.historyV;
      if (hist.length > 1 && waveW > 5 && waveH > 5) {
        ctx.strokeStyle = isSelected ? colors.volt : colors.inkMuted;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let j = 0; j < hist.length; j++) {
          const wx = waveX + (j / (hist.length - 1)) * waveW;
          const wy = waveY + waveH - ((hist[j] - -85) / (35 - -85)) * waveH;
          if (j === 0) ctx.moveTo(wx, wy);
          else ctx.lineTo(wx, wy);
        }
        ctx.stroke();
      }

      // Mini phase loop (right side of card)
      const loopX = cx + waveW + 10;
      const loopY = waveY;
      const loopW = cardW - (waveW + 16);
      const loopH = waveH;

      if (loopW > 10 && loopH > 10 && cell.historyU.length > 1) {
        ctx.strokeStyle = isSelected ? colors.volt : colors.inkMuted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        let prevV = hist[0];
        let isFirst = true;
        for (let j = 0; j < hist.length; j++) {
          const cv = hist[j];
          const cu = cell.historyU[j];
          if (prevV > 20 && cv < 0) {
            ctx.stroke();
            ctx.beginPath();
            isFirst = true;
          }
          const lx = loopX + ((cv - -85) / 120) * loopW;
          const ly = loopY + loopH - ((cu - -25) / 60) * loopH;
          if (isFirst) {
            ctx.moveTo(lx, ly);
            isFirst = false;
          } else {
            ctx.lineTo(lx, ly);
          }
          prevV = cv;
        }
        ctx.stroke();
      }

      // Spike counter pill
      ctx.font = '9px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillStyle = colors.inkMuted;
      ctx.fillText(`${cell.spikes.length} spk`, cx + cardW - 5, cy + 4);
    }
  }

  // Export for node tests and Atlas shell registration
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core, S };
  }
  if (typeof Atlas !== 'undefined') {
    Atlas.Plate.register({
      id: 'N04',
      title: { en: S.title },
      source: 'live',
      mount,
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
