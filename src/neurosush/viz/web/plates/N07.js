/**
 * PLATE N07 · Conductance synapse: a tug of war between reversal potentials.
 * Live port of the constant-conductance membrane equation.
 * Equation: C dv/dt = gL(EL − v) + ge(Ee − v) + gi(Ei − v)
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
    id: 'N07',
    title: 'Conductance synapse: tug of war',
    source: 'live',
    mirror: 'neurosush.neurons.dendrite.ConductanceIntegration · conductance_step',

    // Equations & Marginalia
    eqTitle: 'GOVERNING EQUATIONS',
    eqDiff: 'C dv/dt = gL(EL − v) + ge(Ee − v) + gi(Ei − v)',
    eqVinf: 'v∞ = (gL EL + ge Ee + gi Ei) / (gL + ge + gi)',
    eqTau: 'τeff = C / (gL + ge + gi)',
    liveTitle: 'LIVE MEMBRANE STATE',
    readoutV: 'Membrane potential v',
    readoutVinf: 'Equilibrium v∞',
    readoutTau: 'Effective time constant τeff',
    readoutGtot: 'Total conductance gtot',
    readoutIe: 'Exc force Ie = ge(Ee − v)',
    readoutIi: 'Inh force Ii = gi(Ei − v)',
    readoutIL: 'Leak force IL = gL(EL − v)',
    readoutDiff: 'Theory error |v − v∞|',
    readoutModelDiff: 'Contrast |vcond − vcurr|',

    // Parameter Sliders in Margin
    paramsTitle: 'SYNAPTIC CONDUCTANCES',
    labelGe: 'Excitatory ge',
    labelGi: 'Inhibitory gi',
    labelEi: 'Inhibitory Ei',

    // Glossary
    glossaryTitle: 'GLOSSARY & DYNAMICS',
    glossaryConductance: 'Conductance (g): Ease of ion flux across open channels (µS).',
    glossaryReversal: 'Reversal potential (E): Destination potential for an ion pathway (Ee = 0 mV, Ei = −80 mV, EL = −65 mV).',
    glossaryDriving: 'Driving force (E − v): Electrical gradient; force vanishes as v reaches reversal.',
    glossaryShunting: 'Shunting inhibition: High gi increases total conductance, shortening τeff and attenuating subsequent excitatory pulls even at resting voltage.',

    // Controls
    play: 'Play',
    pause: 'Pause',
    reset: 'Reset',
    pulse: '⚡ Pulse excitation',
    theory: 'Theory ghost [T]',
    modeConductance: 'Model: Conductance',
    modeCurrent: 'Model: Current contrast',
    currentNote: 'Current mode: I_inj is constant; driving force does not shrink near 0 mV.',
    conductanceNote: 'Conductance mode: each force is g(E − v); driving force shrinks as v approaches E.',

    // Chapters
    chapter: 'CHAPTER',
    of: 'OF 5',
    back: '← Back',
    next: 'Next →',

    c1t: '1. Leak alone: resting equilibrium',
    c1b: 'With only leak channels open (gL = 0.10 µS), the knot rests at EL = −65 mV. The rope is relaxed: no synapse is pulling. Watch the scripted test pulse show the baseline relaxation time (τeff = 10.0 ms).',

    c2t: '2. Excitation pulls upward',
    c2b: 'Opening excitatory channels (ge > 0) creates an upward pull toward Ee = 0 mV. The team leans back, straining the rope and dragging the knot depolarized.',

    c3t: '3. Inhibition pulls downward',
    c3b: 'Opening inhibitory channels (gi > 0) introduces an opposing team pulling toward Ei = −80 mV. The two teams engage in a true tug of war, settling at a weighted compromise v∞.',

    c4t: '4. Shunting inhibition at rest',
    c4b: 'Set Ei = EL = −65 mV. Because (Ei − v) = 0 at rest, the inhibitory team exerts 0 pA. Yet gi raises total conductance, shortening τeff from 10 ms to 1.3 ms and dividing any excitatory pull!',

    c5t: '5. Current versus conductance',
    c5b: 'A current synapse injects a fixed push regardless of membrane voltage. A conductance synapse weakens as v approaches Ee (driving force shrinks). The dashed ghost shows the runaway linear current model.',

    // Predict-then-reveal
    predictTitle: 'PREDICT, THEN REVEAL',
    predictQuestion: 'If Ei is exactly at resting potential EL (−65 mV), will adding inhibitory conductance gi change the neuron’s behavior?',
    optNo: 'No effect — resting driving force is zero',
    optYes: 'Yes — it shunts and shrinks excitatory responses',
    commit: 'Commit prediction',
    feedbackCorrect: 'Correct! Although resting current is 0 pA, gi adds to total conductance: τeff plunges from 10 ms to 1.3 ms and subsequent excitatory responses are shunted by over 65%.',
    feedbackWrong: 'Not quite. Although the resting current is zero, gi adds to total membrane conductance: τeff becomes much faster and subsequent excitatory pulls are strongly attenuated.',

    // Canvas hero labels
    heroTitle: 'SYNAPTIC TUG OF WAR ON THE MEMBRANE',
    excTeam: 'EXCITATORY TEAM (Ee = 0 mV)',
    inhTeam: 'INHIBITORY TEAM',
    excTeamShort: 'Exc (Ee = 0)',
    inhTeamShort: 'Inh (Ei)',
    leakAnchor: 'LEAK ANCHOR (EL = −65 mV)',
    dragEiHint: '▲ Drag Ei',
    dragHint: 'Drag Ei marker or sliders · Click membrane rope to pulse excitation',
    traceTitle: 'MEMBRANE VOLTAGE OSCILLOSCOPE v(t)',
    timeAxis: 'Time (ms)',
    voltAxis: 'Voltage (mV)',
    ghostLegend: '-- theory v∞',
    ghostCurrLegend: '-- current model ghost',
    unshuntedLegend: '·· unshunted pulse',
    shuntedLegend: '— shunted pulse',
    unitsMv: 'mV',
    unitsUs: 'µS',
    unitsMs: 'ms',
    unitsPa: 'pA',
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

  // ---- core: pure math, no DOM, exported for node tests ----
  const core = {
    G_L: 0.10,
    E_L: -65.0,
    E_E: 0.0,
    E_I: -80.0,
    C: 1.0,
    mulberry32,

    validate(p) {
      for (const key of ['gL', 'ge', 'gi', 'C']) {
        if (!Number.isFinite(p[key]) || p[key] < 0 || (key === 'C' && p[key] === 0)) {
          throw new Error(key + ' must be ' + (key === 'C' ? 'positive' : 'non-negative') + ', got ' + p[key]);
        }
      }
      for (const key of ['EL', 'Ee', 'Ei']) {
        if (!Number.isFinite(p[key])) throw new Error(key + ' must be finite, got ' + p[key]);
      }
    },

    totalConductance(p) {
      this.validate(p);
      return p.gL + p.ge + p.gi;
    },

    vInfinity(p) {
      const g = this.totalConductance(p);
      if (g === 0) throw new Error('total conductance must be positive, got ' + g);
      return (p.gL * p.EL + p.ge * p.Ee + p.gi * p.Ei) / g;
    },

    tauEff(p) {
      const g = this.totalConductance(p);
      if (g === 0) throw new Error('total conductance must be positive, got ' + g);
      return p.C / g;
    },

    currents(v, p) {
      this.validate(p);
      if (!Number.isFinite(v)) throw new Error('v must be finite, got ' + v);
      return {
        leak: p.gL * (p.EL - v),
        exc: p.ge * (p.Ee - v),
        inh: p.gi * (p.Ei - v),
      };
    },

    derivative(v, p) {
      const i = this.currents(v, p);
      return (i.leak + i.exc + i.inh) / p.C;
    },

    exactStep(v, p, dt) {
      if (!Number.isFinite(dt) || dt < 0) throw new Error('dt must be non-negative, got ' + dt);
      const vinf = this.vInfinity(p);
      return vinf + (v - vinf) * Math.exp(-dt / this.tauEff(p));
    },

    currentStep(v, p, input, dt) {
      if (!Number.isFinite(input)) throw new Error('input must be finite, got ' + input);
      if (!Number.isFinite(dt) || dt < 0) throw new Error('dt must be non-negative, got ' + dt);
      const base = { ...p, ge: 0 };
      const g = this.totalConductance(base);
      const vinf = (base.gL * base.EL + base.gi * base.Ei + input) / g;
      return vinf + (v - vinf) * Math.exp(-dt * g / base.C);
    },

    simulate(initialV, p, dt, steps, mode, input) {
      if (!Number.isInteger(steps) || steps < 0) throw new Error('steps must be a non-negative integer, got ' + steps);
      let v = initialV;
      const values = [v];
      for (let n = 0; n < steps; n++) {
        v = mode === 'current' ? this.currentStep(v, p, input, dt) : this.exactStep(v, p, dt);
        values.push(v);
      }
      return values;
    },

    seededPulseTrain(seed, count, chance) {
      const random = mulberry32(seed);
      const out = [];
      for (let n = 0; n < count; n++) out.push(random() < chance ? 1 : 0);
      return out;
    },
  };

  // ---- UI / Mount (atlas shell integration) ----
  function mount(rootEl, ctx) {
    if (!ctx || !ctx.colors) return;

    const doc = rootEl.ownerDocument || document;
    const colors = ctx.colors;
    const marginContainer = ctx.margin || doc.getElementById('atlas-margin');
    const figureContainer = rootEl.classList && rootEl.classList.contains('atlas-canvas-container')
      ? rootEl
      : (rootEl.querySelector('.atlas-canvas-container') || rootEl);
    const figureSection = figureContainer.parentElement || doc.getElementById('atlas-figure') || rootEl;

    // Simulation & UI state
    const HISTORY_LEN = 200;
    const state = {
      v: core.E_L,
      vCurrent: core.E_L,
      ge: 0.0,
      gi: 0.0,
      Ei: core.E_I,
      t: 0.0,
      running: true,
      theory: true,
      mode: 'conductance', // 'conductance' | 'current'
      chapter: 1,
      hoverTerm: null,
      pulse: 0.0, // Remaining pulse time in ms
      pulseAmp: 0.0,
      demoTimer: 0.0,
      prediction: '',
      revealed: false,
      history: new Array(HISTORY_LEN).fill(core.E_L),
      theoryHistory: new Array(HISTORY_LEN).fill(core.E_L),
      currentHistory: new Array(HISTORY_LEN).fill(core.E_L),
      unshuntedHistory: new Array(HISTORY_LEN).fill(core.E_L),
      lastTimestamp: performance.now(),
      dragTarget: null, // 'ei' | 'ge' | 'gi'
    };

    // Prefill waveforms for a live, informative oscilloscope display
    function prefillHistoryForChapter(ch) {
      let vSim = core.E_L;
      let vCurr = core.E_L;
      let vUnshunt = core.E_L;
      const dt = 0.5;

      for (let i = 0; i < HISTORY_LEN; i++) {
        const timeFromEnd = (HISTORY_LEN - 1 - i) * dt;
        let gExt = 0;

        if (ch === 1 || ch === 2 || ch === 3) {
          // Pulse arrived around 35 ms ago, relaxing back to equilibrium
          if (timeFromEnd >= 30 && timeFromEnd <= 34) {
            gExt = 0.40;
          }
        } else if (ch === 4) {
          // Shunted comparison: pulse occurred at 35 ms ago
          if (timeFromEnd >= 30 && timeFromEnd <= 34) {
            gExt = 0.45;
          }
        } else if (ch === 5) {
          // Current vs conductance comparison
          if (timeFromEnd >= 35) {
            gExt = 0.45;
          }
        }

        const pBase = {
          gL: core.G_L,
          EL: core.E_L,
          ge: (ch === 2 ? 0.40 : (ch === 3 ? 0.40 : (ch === 5 ? 0.45 : 0.0))) + gExt,
          Ee: core.E_E,
          gi: ch === 3 ? 0.50 : (ch === 4 ? 0.70 : (ch === 5 ? 0.35 : 0.0)),
          Ei: ch === 4 ? -65 : -80,
          C: core.C,
        };

        vSim = core.exactStep(vSim, pBase, dt);

        // Unshunted reference (gi = 0)
        const pUnshunt = { ...pBase, gi: 0, Ei: -80 };
        vUnshunt = core.exactStep(vUnshunt, pUnshunt, dt);

        // Current reference
        const iInj = pBase.ge * (core.E_E - core.E_L);
        vCurr = core.currentStep(vCurr, pBase, iInj, dt);

        state.history[i] = vSim;
        state.theoryHistory[i] = core.vInfinity(pBase);
        state.currentHistory[i] = vCurr;
        state.unshuntedHistory[i] = vUnshunt;
      }
      state.v = vSim;
      state.vCurrent = vCurr;
    }

    prefillHistoryForChapter(1);

    const params = () => ({
      gL: core.G_L,
      EL: core.E_L,
      ge: state.ge + (state.pulse > 0 ? state.pulseAmp : 0),
      Ee: core.E_E,
      gi: state.gi,
      Ei: state.Ei,
      C: core.C,
    });

    // Style injection for mobile responsiveness
    const styleEl = doc.createElement('style');
    styleEl.textContent = `
      @media (max-width: 720px) {
        .atlas-canvas-container {
          aspect-ratio: auto !important;
          height: 380px !important;
          min-height: 380px !important;
        }
      }
    `;
    (doc.head || doc.body || rootEl).appendChild(styleEl);

    // 1. Marginalia in ctx.margin
    if (marginContainer) {
      marginContainer.innerHTML = `
        <div class="atlas-margin-section" id="n07-margin-eqs">
          <div class="atlas-margin-label">${S.eqTitle}</div>
          <div class="atlas-equation-box atlas-mono" style="font-size:12px; line-height:1.7;">
            <div class="n07-term" data-term="diff" style="cursor:help;">
              C dv/dt = <span style="color:var(--ink-muted);" data-term="leak">gL(EL − v)</span> +
              <span style="color:var(--exc); font-weight:600;" data-term="exc">ge(Ee − v)</span> +
              <span style="color:var(--inh); font-weight:600;" data-term="inh">gi(Ei − v)</span>
            </div>
            <div class="n07-term" data-term="vinf" style="cursor:help; margin-top:4px;">
              ${S.eqVinf}
            </div>
            <div class="n07-term" data-term="tau" style="cursor:help; margin-top:4px; color:var(--mod);">
              ${S.eqTau}
            </div>
          </div>
        </div>

        <div class="atlas-margin-section" id="n07-margin-readouts">
          <div class="atlas-margin-label">${S.liveTitle}</div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
            <span class="atlas-serif" style="font-size:13px;">${S.readoutV}:</span>
            <span class="atlas-mono" id="n07-val-v" style="color:var(--volt); font-size:16px; font-weight:700;">−65.0 mV</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
            <span class="atlas-serif" style="font-size:12px;">${S.readoutVinf}:</span>
            <span class="atlas-mono" id="n07-val-vinf" style="color:var(--theory); font-size:13px; font-weight:600;">−65.0 mV</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
            <span class="atlas-serif" style="font-size:12px;">${S.readoutTau}:</span>
            <span class="atlas-mono" id="n07-val-tau" style="color:var(--mod); font-size:13px; font-weight:600;">10.00 ms</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
            <span class="atlas-serif" style="font-size:12px;">${S.readoutGtot}:</span>
            <span class="atlas-mono" id="n07-val-gtot" style="font-size:13px;">0.10 µS</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
            <span class="atlas-serif" style="font-size:12px; color:var(--exc); font-weight:600;">${S.readoutIe}:</span>
            <span class="atlas-mono" id="n07-val-ie" style="color:var(--exc); font-size:13px; font-weight:600;">0.00 pA</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
            <span class="atlas-serif" style="font-size:12px; color:var(--inh); font-weight:600;">${S.readoutIi}:</span>
            <span class="atlas-mono" id="n07-val-ii" style="color:var(--inh); font-size:13px; font-weight:600;">0.00 pA</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
            <span class="atlas-serif" style="font-size:12px; color:var(--ink-muted);">${S.readoutIL}:</span>
            <span class="atlas-mono" id="n07-val-il" style="font-size:13px;">0.00 pA</span>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:baseline; border-top:1px dashed var(--border); padding-top:4px; margin-top:4px;">
            <span class="atlas-serif" style="font-size:12px;">${S.readoutDiff}:</span>
            <span class="atlas-mono" id="n07-val-diff" style="font-size:13px; color:var(--theory);">0.000 mV</span>
          </div>
          <div id="n07-row-model-diff" style="display:flex; justify-content:space-between; align-items:baseline; margin-top:2px;">
            <span class="atlas-serif" style="font-size:12px; color:var(--trace);">${S.readoutModelDiff}:</span>
            <span class="atlas-mono" id="n07-val-model-diff" style="font-size:13px; color:var(--trace); font-weight:600;">0.000 mV</span>
          </div>
        </div>

        <div class="atlas-margin-section" id="n07-margin-params">
          <div class="atlas-margin-label">${S.paramsTitle}</div>
          <div style="display:flex; flex-direction:column; gap:8px; font-size:12px;">
            <div>
              <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <label for="n07-slider-ge" style="font-weight:600; color:var(--exc);">${S.labelGe}</label>
                <span class="atlas-mono" id="n07-val-slider-ge" style="font-weight:600;">0.00 µS</span>
              </div>
              <input type="range" id="n07-slider-ge" min="0" max="0.90" step="0.01" value="0" style="width:100%; cursor:pointer;" aria-label="Excitatory conductance">
            </div>

            <div>
              <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <label for="n07-slider-gi" style="font-weight:600; color:var(--inh);">${S.labelGi}</label>
                <span class="atlas-mono" id="n07-val-slider-gi" style="font-weight:600;">0.00 µS</span>
              </div>
              <input type="range" id="n07-slider-gi" min="0" max="0.90" step="0.01" value="0" style="width:100%; cursor:pointer;" aria-label="Inhibitory conductance">
            </div>

            <div>
              <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <label for="n07-slider-ei" style="font-weight:600; color:var(--ink);">${S.labelEi}</label>
                <span class="atlas-mono" id="n07-val-slider-ei" style="font-weight:600;">−80 mV</span>
              </div>
              <input type="range" id="n07-slider-ei" min="-90" max="-45" step="1" value="-80" style="width:100%; cursor:pointer;" aria-label="Inhibitory reversal potential">
            </div>
          </div>
        </div>

        <div class="atlas-margin-section" style="font-size:12px; line-height:1.45;">
          <div class="atlas-margin-label">${S.glossaryTitle}</div>
          <p style="margin:0 0 5px 0;">• ${S.glossaryConductance}</p>
          <p style="margin:0 0 5px 0;">• ${S.glossaryReversal}</p>
          <p style="margin:0 0 5px 0;">• ${S.glossaryDriving}</p>
          <p style="margin:0;">• ${S.glossaryShunting}</p>
        </div>

        <div class="atlas-margin-section" style="font-size:12px; color:var(--ink-muted); line-height:1.4;">
          <div class="atlas-mono" style="font-size:11px; text-transform:uppercase;">Source: LIVE</div>
          <div>${S.mirror}</div>
        </div>
      `;

      marginContainer.querySelectorAll('[data-term]').forEach((el) => {
        el.addEventListener('mouseenter', () => {
          state.hoverTerm = el.dataset.term;
          el.style.textDecoration = 'underline';
        });
        el.addEventListener('mouseleave', () => {
          state.hoverTerm = null;
          el.style.textDecoration = 'none';
        });
      });
    }

    const byId = (id) => doc.getElementById(id);

    // 2. Build Chapter Bar AT TOP of Figure Section
    let chapterBar = byId('n07-chapter-bar');
    if (!chapterBar) {
      chapterBar = doc.createElement('div');
      chapterBar.id = 'n07-chapter-bar';
      chapterBar.style.cssText =
        'background:var(--paper-card); border:1px solid var(--border); border-radius:var(--radius); padding:8px 12px; margin-bottom:6px; display:flex; flex-direction:column; gap:4px; font-size:13px;';
      figureSection.insertBefore(chapterBar, figureContainer);
    }

    // 3. Predict-then-reveal Card (for Chapter 4)
    let predictCard = byId('n07-predict-card');
    if (!predictCard) {
      predictCard = doc.createElement('div');
      predictCard.id = 'n07-predict-card';
      predictCard.style.cssText =
        'background:var(--paper-inset); border:1px solid var(--border); border-radius:var(--radius); padding:8px 12px; margin-bottom:6px; display:none; flex-direction:column; gap:6px; font-size:12px;';
      predictCard.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
          <strong style="font-family:var(--font-serif); font-size:13px; color:var(--volt);">${S.predictTitle}</strong>
          <span style="color:var(--ink);">${S.predictQuestion}</span>
        </div>
        <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:2px;">
          <label style="cursor:pointer;"><input type="radio" name="n07-pred" value="no"> ${S.optNo}</label>
          <label style="cursor:pointer;"><input type="radio" name="n07-pred" value="yes"> ${S.optYes}</label>
          <button class="atlas-btn" id="n07-btn-commit" style="font-size:12px; padding:3px 10px; font-weight:600;">${S.commit}</button>
          <span id="n07-predict-feedback" style="font-weight:600; font-size:12px; line-height:1.4;"></span>
        </div>
      `;
      figureSection.insertBefore(predictCard, figureContainer);
    }

    // 4. Interactive Controls Bar
    let controlsBar = byId('n07-controls-bar');
    if (!controlsBar) {
      controlsBar = doc.createElement('div');
      controlsBar.id = 'n07-controls-bar';
      controlsBar.style.cssText =
        'display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; padding:6px 10px; background:var(--paper-card); border:1px solid var(--border); border-radius:var(--radius); font-size:12px; margin-bottom:6px;';
      controlsBar.innerHTML = `
        <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
          <button class="atlas-btn" id="n07-btn-play" style="font-size:12px; padding:4px 9px;">${S.pause}</button>
          <button class="atlas-btn" id="n07-btn-reset" style="font-size:12px; padding:4px 9px;">${S.reset}</button>
          <button class="atlas-btn" id="n07-btn-pulse" style="font-size:12px; padding:4px 9px; font-weight:600; color:var(--exc);">${S.pulse}</button>
          <button class="atlas-btn active" id="n07-btn-theory" style="font-size:12px; padding:4px 9px;">${S.theory}</button>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <button class="atlas-btn" id="n07-btn-mode" style="font-size:12px; padding:4px 9px; font-weight:600;">${S.modeConductance}</button>
          <span class="atlas-mono" id="n07-text-mode-note" style="font-size:12px; color:var(--ink-muted);">${S.conductanceNote}</span>
        </div>
      `;
      figureSection.insertBefore(controlsBar, figureContainer);
    }

    // 5. Setup Canvas inside canvasContainer
    figureContainer.style.aspectRatio = 'auto';
    figureContainer.style.height = '440px';
    figureContainer.style.minHeight = '390px';
    figureContainer.style.maxHeight = '560px';
    figureContainer.innerHTML = '';

    const canvas = doc.createElement('canvas');
    canvas.className = 'atlas-canvas';
    canvas.style.cssText = 'width:100%; height:100%; display:block; cursor:default; touch-action:none;';
    figureContainer.appendChild(canvas);
    const c = canvas.getContext('2d');

    // UI Input References
    const geSlider = byId('n07-slider-ge');
    const giSlider = byId('n07-slider-gi');
    const eiSlider = byId('n07-slider-ei');
    const geVal = byId('n07-val-slider-ge');
    const giVal = byId('n07-val-slider-gi');
    const eiVal = byId('n07-val-slider-ei');

    function syncSliderInputs() {
      if (geSlider) geSlider.value = state.ge;
      if (giSlider) giSlider.value = state.gi;
      if (eiSlider) eiSlider.value = state.Ei;
      if (geVal) geVal.textContent = state.ge.toFixed(2) + ' µS';
      if (giVal) giVal.textContent = state.gi.toFixed(2) + ' µS';
      if (eiVal) eiVal.textContent = state.Ei.toFixed(0) + ' mV';
    }

    function triggerPulse(amp = 0.40, duration = 4.0) {
      state.pulseAmp = amp;
      state.pulse = duration;
    }

    // Chapter definitions
    const chapters = [
      { t: S.c1t, b: S.c1b, ge: 0.0, gi: 0.0, ei: -80, mode: 'conductance' },
      { t: S.c2t, b: S.c2b, ge: 0.40, gi: 0.0, ei: -80, mode: 'conductance' },
      { t: S.c3t, b: S.c3b, ge: 0.40, gi: 0.50, ei: -80, mode: 'conductance' },
      { t: S.c4t, b: S.c4b, ge: 0.0, gi: 0.70, ei: -65, mode: 'conductance' },
      { t: S.c5t, b: S.c5b, ge: 0.45, gi: 0.35, ei: -80, mode: 'current' },
    ];

    function applyChapter(num) {
      state.chapter = Math.max(1, Math.min(5, num));
      const def = chapters[state.chapter - 1];
      state.ge = def.ge;
      state.gi = def.gi;
      state.Ei = def.ei;
      state.mode = def.mode;
      state.pulse = 0;
      state.revealed = false;
      state.prediction = '';

      prefillHistoryForChapter(state.chapter);

      syncSliderInputs();
      renderChapterHeader();
    }

    function renderChapterHeader() {
      const def = chapters[state.chapter - 1];
      chapterBar.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
          <div style="display:flex; align-items:baseline; gap:8px;">
            <span class="atlas-badge">${S.chapter} ${state.chapter} ${S.of}</span>
            <b class="atlas-serif" style="font-size:14px; color:var(--ink);">${def.t}</b>
          </div>
          <div style="display:flex; gap:6px;">
            <button class="atlas-btn" id="n07-btn-back" style="font-size:12px; padding:3px 9px;">${S.back}</button>
            <button class="atlas-btn" id="n07-btn-next" style="font-size:12px; padding:3px 9px;">${S.next}</button>
          </div>
        </div>
        <div class="atlas-serif" style="font-size:13px; line-height:1.45; color:var(--ink);">${def.b}</div>
      `;

      byId('n07-btn-back').disabled = state.chapter === 1;
      byId('n07-btn-next').disabled = state.chapter === 5;
      byId('n07-btn-back').onclick = () => applyChapter(state.chapter - 1);
      byId('n07-btn-next').onclick = () => applyChapter(state.chapter + 1);

      // Chapter 4: show predict card
      predictCard.style.display = state.chapter === 4 ? 'flex' : 'none';
      const predictFeedback = byId('n07-predict-feedback');
      if (predictFeedback) predictFeedback.textContent = '';
      doc.querySelectorAll('input[name="n07-pred"]').forEach((el) => { el.checked = false; });

      // Mode readout
      byId('n07-btn-mode').textContent = state.mode === 'conductance' ? S.modeConductance : S.modeCurrent;
      byId('n07-text-mode-note').textContent = state.mode === 'conductance' ? S.conductanceNote : S.currentNote;
      byId('n07-row-model-diff').style.display = state.mode === 'current' ? 'flex' : 'none';
    }

    // Sliders event listeners
    if (geSlider) {
      geSlider.addEventListener('input', () => {
        state.ge = parseFloat(geSlider.value);
        syncSliderInputs();
      });
    }
    if (giSlider) {
      giSlider.addEventListener('input', () => {
        state.gi = parseFloat(giSlider.value);
        syncSliderInputs();
      });
    }
    if (eiSlider) {
      eiSlider.addEventListener('input', () => {
        state.Ei = parseFloat(eiSlider.value);
        syncSliderInputs();
      });
    }

    // Controls Buttons
    byId('n07-btn-play').onclick = () => {
      state.running = !state.running;
      byId('n07-btn-play').textContent = state.running ? S.pause : S.play;
    };
    byId('n07-btn-reset').onclick = () => {
      applyChapter(state.chapter);
    };
    byId('n07-btn-pulse').onclick = () => {
      triggerPulse(0.45, 4.0);
    };
    byId('n07-btn-theory').onclick = () => {
      state.theory = !state.theory;
      byId('n07-btn-theory').classList.toggle('active', state.theory);
    };
    byId('n07-btn-mode').onclick = () => {
      state.mode = state.mode === 'conductance' ? 'current' : 'conductance';
      byId('n07-btn-mode').textContent = state.mode === 'conductance' ? S.modeConductance : S.modeCurrent;
      byId('n07-text-mode-note').textContent = state.mode === 'conductance' ? S.conductanceNote : S.currentNote;
      byId('n07-row-model-diff').style.display = state.mode === 'current' ? 'flex' : 'none';
    };

    // Predict Card actions
    doc.querySelectorAll('input[name="n07-pred"]').forEach((el) => {
      el.onchange = () => { state.prediction = el.value; };
    });
    byId('n07-btn-commit').onclick = () => {
      if (!state.prediction) return;
      state.revealed = true;
      const fb = byId('n07-predict-feedback');
      const isCorrect = state.prediction === 'yes';
      fb.textContent = isCorrect ? S.feedbackCorrect : S.feedbackWrong;
      fb.style.color = isCorrect ? 'var(--exc)' : 'var(--inh)';
      triggerPulse(0.45, 4.0);
    };

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === ' ') {
        e.preventDefault();
        byId('n07-btn-play').click();
      } else if (e.key === 't' || e.key === 'T') {
        byId('n07-btn-theory').click();
      } else if (e.key === 'p' || e.key === 'P') {
        triggerPulse(0.45, 4.0);
      } else if (e.key === 'ArrowRight' && state.chapter < 5) {
        applyChapter(state.chapter + 1);
      } else if (e.key === 'ArrowLeft' && state.chapter > 1) {
        applyChapter(state.chapter - 1);
      } else if (e.key === 'Home') {
        applyChapter(1);
      } else if (e.key === 'End') {
        applyChapter(5);
      }
    });

    window.addEventListener('atlas:theory', (e) => {
      if (e.detail && typeof e.detail.active === 'boolean') {
        state.theory = e.detail.active;
        byId('n07-btn-theory').classList.toggle('active', state.theory);
      }
    });

    // Pointer Dragging and Clicking on Canvas
    function getCanvasCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const clientX = e.clientX ?? (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
      const clientY = e.clientY ?? (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
      return {
        x: (clientX - rect.left),
        y: (clientY - rect.top),
        w: rect.width,
        h: rect.height,
        dpr,
      };
    }

    canvas.addEventListener('pointerdown', (e) => {
      const { x, y, w, h } = getCanvasCoords(e);
      const pad = Math.max(30, Math.round(w * 0.065));
      const axisW = w - pad * 2;
      const vToX = (v) => pad + ((v - (-90)) / 100) * axisW;
      const ropeY = Math.round(h * 0.25);

      const eiX = vToX(state.Ei);
      // Drag Ei handle (hit box: eiX +/- 40, y in ropeY + 40 .. ropeY + 100)
      if (Math.abs(x - eiX) < 40 && y >= ropeY + 40 && y <= ropeY + 100) {
        state.dragTarget = 'ei';
        canvas.setPointerCapture(e.pointerId);
        return;
      }

      // Drag ge (right half near excitation team)
      if (x > w * 0.65 && y < h * 0.50) {
        state.dragTarget = 'ge';
        canvas.setPointerCapture(e.pointerId);
        return;
      }

      // Drag gi (left half near inhibition team)
      if (x < w * 0.35 && y < h * 0.50) {
        state.dragTarget = 'gi';
        canvas.setPointerCapture(e.pointerId);
        return;
      }

      // Click on rope / membrane area to pulse
      if (y < h * 0.52) {
        triggerPulse(0.45, 4.0);
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!state.dragTarget) return;
      const { x, y, w, h } = getCanvasCoords(e);
      const pad = Math.max(30, Math.round(w * 0.065));
      const axisW = w - pad * 2;

      if (state.dragTarget === 'ei') {
        const ratio = Math.max(0, Math.min(1, (x - pad) / axisW));
        state.Ei = Math.round(-90 + ratio * 100);
        state.Ei = Math.max(-90, Math.min(-45, state.Ei));
        syncSliderInputs();
      } else if (state.dragTarget === 'ge') {
        const val = Math.max(0, Math.min(0.90, (y / (h * 0.45)) * 0.90));
        state.ge = Math.round(val * 100) / 100;
        syncSliderInputs();
      } else if (state.dragTarget === 'gi') {
        const val = Math.max(0, Math.min(0.90, (y / (h * 0.45)) * 0.90));
        state.gi = Math.round(val * 100) / 100;
        syncSliderInputs();
      }
    });

    canvas.addEventListener('pointerup', (e) => {
      state.dragTarget = null;
      if (canvas.hasPointerCapture && canvas.hasPointerCapture(e.pointerId)) {
        canvas.releasePointerCapture(e.pointerId);
      }
    });

    canvas.addEventListener('pointercancel', () => {
      state.dragTarget = null;
    });

    function strokeLine(ctx2d, x1, y1, x2, y2, color, width, dash = []) {
      ctx2d.save();
      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = width;
      ctx2d.setLineDash(dash);
      ctx2d.beginPath();
      ctx2d.moveTo(x1, y1);
      ctx2d.lineTo(x2, y2);
      ctx2d.stroke();
      ctx2d.restore();
    }

    function drawArrow(ctx2d, fromX, fromY, toX, toY, color, width, headLen = 7) {
      const dx = toX - fromX;
      const dy = toY - fromY;
      const angle = Math.atan2(dy, dx);
      ctx2d.save();
      ctx2d.strokeStyle = color;
      ctx2d.fillStyle = color;
      ctx2d.lineWidth = width;
      ctx2d.beginPath();
      ctx2d.moveTo(fromX, fromY);
      ctx2d.lineTo(toX, toY);
      ctx2d.stroke();
      ctx2d.beginPath();
      ctx2d.moveTo(toX, toY);
      ctx2d.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
      ctx2d.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
      ctx2d.closePath();
      ctx2d.fill();
      ctx2d.restore();
    }

    // Draw realistic stick figures pulling the rope
    function drawPullingFigure(ctx2d, x, ropeY, groundY, dir, lean, color) {
      ctx2d.save();
      ctx2d.strokeStyle = color;
      ctx2d.fillStyle = color;
      ctx2d.lineWidth = 2.0;

      const hipX = x;
      const hipY = groundY - 14;
      const headOffset = -dir * (8 + lean * 12);
      const headX = hipX + headOffset;
      const headY = hipY - 16;

      // Head
      ctx2d.beginPath();
      ctx2d.arc(headX, headY, 5, 0, Math.PI * 2);
      ctx2d.fill();

      // Spine / Torso
      ctx2d.beginPath();
      ctx2d.moveTo(hipX, hipY);
      ctx2d.lineTo(headX, headY + 5);
      ctx2d.stroke();

      // Front leg braced
      ctx2d.beginPath();
      ctx2d.moveTo(hipX, hipY);
      ctx2d.lineTo(hipX + dir * 6, groundY - 5);
      ctx2d.lineTo(hipX + dir * 10, groundY);
      ctx2d.stroke();

      // Back leg dug in
      ctx2d.beginPath();
      ctx2d.moveTo(hipX, hipY);
      ctx2d.lineTo(hipX - dir * 10, groundY);
      ctx2d.stroke();

      // Arms gripping rope directly at ropeY
      ctx2d.beginPath();
      ctx2d.moveTo(headX, headY + 8);
      ctx2d.lineTo(headX + dir * 10, (headY + 8 + ropeY) / 2);
      ctx2d.lineTo(x + dir * 6, ropeY);
      ctx2d.stroke();

      ctx2d.restore();
    }

    // Draw stick figure at rest (faint idle crowd waiting, not pulling)
    function drawIdleFigure(ctx2d, x, ropeY, groundY, dir, color) {
      ctx2d.save();
      ctx2d.globalAlpha = 0.30;
      ctx2d.strokeStyle = color;
      ctx2d.fillStyle = color;
      ctx2d.lineWidth = 1.8;

      const hipX = x;
      const hipY = groundY - 14;
      const headX = hipX;
      const headY = hipY - 16;

      // Head
      ctx2d.beginPath();
      ctx2d.arc(headX, headY, 4.5, 0, Math.PI * 2);
      ctx2d.fill();

      // Spine / Torso (standing upright at rest)
      ctx2d.beginPath();
      ctx2d.moveTo(hipX, hipY);
      ctx2d.lineTo(headX, headY + 4.5);
      ctx2d.stroke();

      // Legs standing at rest
      ctx2d.beginPath();
      ctx2d.moveTo(hipX, hipY);
      ctx2d.lineTo(hipX - 3.5, groundY);
      ctx2d.moveTo(hipX, hipY);
      ctx2d.lineTo(hipX + 3.5, groundY);
      ctx2d.stroke();

      // Arms hanging relaxed at sides (not gripping rope)
      ctx2d.beginPath();
      ctx2d.moveTo(headX, headY + 7);
      ctx2d.lineTo(headX - dir * 4, hipY - 1);
      ctx2d.lineTo(headX - dir * 3, groundY - 4);
      ctx2d.moveTo(headX, headY + 7);
      ctx2d.lineTo(headX + dir * 4, hipY - 1);
      ctx2d.lineTo(headX + dir * 3, groundY - 4);
      ctx2d.stroke();

      ctx2d.restore();
    }

    // Draw knot: realistic cord wrap with voltage badge
    function drawKnot(ctx2d, kx, ky, v, col, isGhost = false, label = '') {
      ctx2d.save();
      if (isGhost) {
        ctx2d.strokeStyle = col.theory;
        ctx2d.lineWidth = 1.8;
        ctx2d.setLineDash([4, 3]);
        ctx2d.beginPath();
        ctx2d.arc(kx, ky, 11, 0, Math.PI * 2);
        ctx2d.stroke();

        ctx2d.font = '600 12px ui-monospace';
        ctx2d.fillStyle = col.theory;
        ctx2d.textAlign = 'center';
        ctx2d.textBaseline = 'bottom';
        ctx2d.fillText(label || 'v∞', kx, ky - 14);
      } else {
        // Outer loop coils
        ctx2d.fillStyle = col.volt;
        ctx2d.strokeStyle = col.ink;
        ctx2d.lineWidth = 1.5;

        // Coil 1
        ctx2d.beginPath();
        ctx2d.ellipse(kx - 3, ky, 7, 11, Math.PI / 6, 0, Math.PI * 2);
        ctx2d.fill();
        ctx2d.stroke();

        // Coil 2
        ctx2d.beginPath();
        ctx2d.ellipse(kx + 3, ky, 7, 11, -Math.PI / 6, 0, Math.PI * 2);
        ctx2d.fill();
        ctx2d.stroke();

        // Central bind
        ctx2d.beginPath();
        ctx2d.arc(kx, ky, 5.5, 0, Math.PI * 2);
        ctx2d.fillStyle = col.paperCard || '#ffffff';
        ctx2d.fill();
        ctx2d.stroke();

        // Live Voltage Badge above knot
        const badgeText = `${v.toFixed(1)} mV`;
        ctx2d.font = '700 12px ui-monospace';
        const tw = ctx2d.measureText(badgeText).width;
        const bw = tw + 14;
        const bh = 20;
        const bx = kx - bw / 2;
        const by = ky - 34;

        ctx2d.fillStyle = col.paperCard || '#ffffff';
        ctx2d.strokeStyle = col.volt;
        ctx2d.lineWidth = 1.5;
        ctx2d.beginPath();
        ctx2d.roundRect(bx, by, bw, bh, 3);
        ctx2d.fill();
        ctx2d.stroke();

        ctx2d.fillStyle = col.volt;
        ctx2d.textAlign = 'center';
        ctx2d.textBaseline = 'middle';
        ctx2d.fillText(badgeText, kx, by + bh / 2);
      }
      ctx2d.restore();
    }

    // Main Draw Function
    function draw(col) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = rect.width;
      const h = rect.height;
      if (!w || !h) return;

      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }

      c.resetTransform();
      c.scale(dpr, dpr);

      // Background
      c.fillStyle = col.paper;
      c.fillRect(0, 0, w, h);

      const isNarrow = w < 620;
      const p = params();
      const vinf = core.vInfinity(p);
      const tau = core.tauEff(p);
      const currentValues = core.currents(state.v, p);

      // Layout coordinates
      const pad = Math.max(26, Math.round(w * 0.06));
      const axisW = w - pad * 2;
      const vToX = (v) => pad + ((v - (-90)) / 100) * axisW;

      const ropeY = Math.round(h * 0.23);
      const groundY = ropeY + 34;
      const knotX = vToX(state.v);
      const vinfX = vToX(vinf);
      const elX = vToX(core.E_L);
      const eeX = vToX(core.E_E);
      const eiX = vToX(state.Ei);

      // 1. Hero Title & Team Labels
      if (!isNarrow) {
        c.font = '600 13px ui-serif, Georgia';
        c.fillStyle = col.ink;
        c.textAlign = 'center';
        c.textBaseline = 'top';
        c.fillText(S.heroTitle, w / 2, 8);
      }

      // Team Labels positioned safely at top corners
      c.font = '600 12px ui-sans-serif, system-ui';
      c.textBaseline = 'top';

      // Left team label
      c.fillStyle = col.inh;
      c.textAlign = 'left';
      const inhLabel = isNarrow ? `${S.inhTeamShort} · ${p.gi.toFixed(2)} µS` : `${S.inhTeam} (Ei = ${p.Ei.toFixed(0)} mV) · gi = ${p.gi.toFixed(2)} µS`;
      c.fillText(inhLabel, pad, isNarrow ? 8 : 26);

      // Right team label
      c.fillStyle = col.exc;
      c.textAlign = 'right';
      const excLabel = isNarrow ? `${S.excTeamShort} · ${p.ge.toFixed(2)} µS` : `${S.excTeam} · ge = ${p.ge.toFixed(2)} µS`;
      c.fillText(excLabel, w - pad, isNarrow ? 8 : 26);

      // 2. Ground line under figures
      strokeLine(c, pad, groundY, w - pad, groundY, col.border, 1, [3, 2]);

      // 3. Rope Strain & Sag Computation
      const netTension = Math.abs(currentValues.exc) + Math.abs(currentValues.inh) + Math.abs(currentValues.leak);
      const maxSag = Math.max(1, 8 - Math.min(7, netTension * 0.25));

      // Draw the Rope (thick core with dual interwoven spiral hairlines)
      c.save();
      c.beginPath();
      c.moveTo(pad, ropeY);
      const midLeftX = (pad + knotX) / 2;
      const sagLeftY = ropeY + maxSag * (state.gi > 0 ? 0.3 : 0.8);
      c.quadraticCurveTo(midLeftX, sagLeftY, knotX, ropeY);
      const midRightX = (knotX + (w - pad)) / 2;
      const sagRightY = ropeY + maxSag * (state.ge > 0 ? 0.3 : 0.8);
      c.quadraticCurveTo(midRightX, sagRightY, w - pad, ropeY);

      // Base rope stroke
      c.lineWidth = 7;
      c.strokeStyle = '#c99b66';
      c.stroke();

      // Intertwining twisted rope highlight lines along exact same curve
      c.lineWidth = 2.2;
      c.strokeStyle = '#dfba8a';
      c.setLineDash([5, 5]);
      c.stroke();
      c.restore();

      // Highlight rope parts on term hover
      if (state.hoverTerm === 'exc' || state.hoverTerm === 'diff') {
        strokeLine(c, knotX, ropeY - 6, w - pad, ropeY - 6, col.exc, 3, [3, 2]);
      }
      if (state.hoverTerm === 'inh' || state.hoverTerm === 'diff') {
        strokeLine(c, pad, ropeY - 6, knotX, ropeY - 6, col.inh, 3, [3, 2]);
      }
      if (state.hoverTerm === 'leak' || state.hoverTerm === 'diff') {
        strokeLine(c, elX, ropeY - 10, elX, ropeY + 10, col.ink, 3);
      }

      // 4. Leak Anchor at EL = -65 mV
      c.save();
      c.strokeStyle = col.ink;
      c.fillStyle = col.ink;
      c.lineWidth = 2.0;

      // Anchor stake embedded in ground
      c.beginPath();
      c.moveTo(elX, ropeY);
      c.lineTo(elX, groundY + 10);
      c.stroke();

      // Ground anchor cross-bar and hatch lines
      strokeLine(c, elX - 8, groundY + 10, elX + 8, groundY + 10, col.ink, 2);
      for (let hx = elX - 6; hx <= elX + 6; hx += 3) {
        strokeLine(c, hx, groundY + 10, hx - 2, groundY + 15, col.inkMuted, 1.2);
      }

      // Anchor ring on rope
      c.beginPath();
      c.arc(elX, ropeY, 4.5, 0, Math.PI * 2);
      c.fillStyle = col.paperCard || '#ffffff';
      c.fill();
      c.stroke();

      // Leak Label
      c.font = '600 12px ui-serif, Georgia';
      c.fillStyle = col.inkMuted;
      c.textAlign = 'center';
      c.textBaseline = 'top';
      c.fillText(isNarrow ? 'EL (−65)' : S.leakAnchor, elX, groundY + 18);
      c.restore();

      // 5. Excitatory Team (pulling right toward Ee = 0 mV)
      const numExc = Math.max(0, Math.round(p.ge * 7));
      const excLean = Math.min(1.0, Math.max(0.1, currentValues.exc / 35.0));
      const totalExc = Math.max(3, numExc);
      const spanStartExc = Math.max(knotX + (isNarrow ? 52 : 44), w * (isNarrow ? 0.65 : 0.60));
      const spanEndExc = Math.min(w - pad - 12, w * 0.94);
      const stepXExc = (spanEndExc - spanStartExc) / Math.max(1, totalExc);
      for (let i = 0; i < totalExc; i++) {
        const figX = spanStartExc + i * stepXExc + 8;
        if (i < numExc) {
          drawPullingFigure(c, figX, ropeY, groundY, 1, excLean, col.exc);
        } else {
          drawIdleFigure(c, figX, ropeY, groundY, 1, col.exc);
        }
      }

      // 6. Inhibitory Team (pulling left toward Ei)
      const numInh = Math.max(0, Math.round(p.gi * 7));
      const inhLean = Math.min(1.0, Math.max(0.1, Math.abs(currentValues.inh) / 35.0));
      const totalInh = Math.max(3, numInh);
      const spanStartInh = Math.max(pad + 12, w * 0.08);
      const spanEndInh = Math.min(knotX - (isNarrow ? 52 : 44), w * (isNarrow ? 0.35 : 0.40));
      const stepXInh = (spanEndInh - spanStartInh) / Math.max(1, totalInh);
      for (let i = 0; i < totalInh; i++) {
        const figX = spanStartInh + i * stepXInh + 8;
        if (i < numInh) {
          drawPullingFigure(c, figX, ropeY, groundY, -1, inhLean, col.inh);
        } else {
          drawIdleFigure(c, figX, ropeY, groundY, -1, col.inh);
        }
      }

      // 7. Force Magnitude Arrows at the Knot (placed below rope to never collide with badges)
      if (p.ge > 0.001 && currentValues.exc >= 0.5) {
        const arrowLen = Math.min(isNarrow ? 32 : 55, Math.max(12, currentValues.exc * 1.5));
        drawArrow(c, knotX + 14, ropeY + 11, knotX + 14 + arrowLen, ropeY + 11, col.exc, 1.8);
        c.font = (isNarrow ? '700 10px' : '700 11px') + ' ui-monospace';
        c.fillStyle = col.exc;
        c.textAlign = 'left';
        c.textBaseline = 'top';
        c.fillText(`+${currentValues.exc.toFixed(1)} pA`, knotX + 16, ropeY + 15);
      }

      if (p.gi > 0.001 && Math.abs(currentValues.inh) >= 0.5) {
        const arrowLen = Math.min(isNarrow ? 32 : 55, Math.max(12, Math.abs(currentValues.inh) * 1.5));
        drawArrow(c, knotX - 14, ropeY + 11, knotX - 14 - arrowLen, ropeY + 11, col.inh, 1.8);
        c.font = (isNarrow ? '700 10px' : '700 11px') + ' ui-monospace';
        c.fillStyle = col.inh;
        c.textAlign = 'right';
        c.textBaseline = 'top';
        c.fillText(`${currentValues.inh.toFixed(1)} pA`, knotX - 16, ropeY + 15);
      }

      // 8. Theory Ghost Knot at vInfinity
      if (state.theory) {
        drawKnot(c, vinfX, ropeY, vinf, col, true, 'v∞');
      }

      // 9. Current Synapse Model Ghost Knot (in Chapter 5 or Current Mode)
      if (state.mode === 'current') {
        const currX = vToX(state.vCurrent);
        c.save();
        c.strokeStyle = col.trace;
        c.lineWidth = 1.8;
        c.setLineDash([3, 3]);
        c.beginPath();
        c.arc(currX, ropeY, 11, 0, Math.PI * 2);
        c.stroke();

        const isClose = Math.abs(currX - knotX) < 65;
        const badgeY = isClose ? (ropeY - 48) : (ropeY - 20);
        const currText = `curr: ${state.vCurrent.toFixed(1)}`;
        c.font = '700 11px ui-monospace';
        const ctw = c.measureText(currText).width;
        c.fillStyle = col.paperCard || '#ffffff';
        c.strokeStyle = col.trace;
        c.lineWidth = 1.2;
        c.beginPath();
        if (c.roundRect) c.roundRect(currX - ctw / 2 - 4, badgeY - 14, ctw + 8, 16, 2);
        else c.rect(currX - ctw / 2 - 4, badgeY - 14, ctw + 8, 16);
        c.fill();
        c.stroke();

        c.fillStyle = col.trace;
        c.textAlign = 'center';
        c.textBaseline = 'middle';
        c.fillText(currText, currX, badgeY - 6);
        c.restore();
      }

      // 10. Live Knot (drawn on top of rope)
      drawKnot(c, knotX, ropeY, state.v, col, false);

      // 11. Voltage Axis with clean non-overlapping ticks
      const axisY = groundY + 38;
      strokeLine(c, pad, axisY, w - pad, axisY, col.border, 1.2);

      const ticks = isNarrow ? [-80, -65, 0] : [-80, -65, -50, -30, 0];
      for (const tv of ticks) {
        const tx = vToX(tv);
        strokeLine(c, tx, axisY - 4, tx, axisY + 4, col.inkMuted, 1);
        c.font = '12px ui-monospace';
        c.fillStyle = tv === core.E_E ? col.exc : (tv === core.E_L ? col.ink : col.inkMuted);
        c.textAlign = 'center';
        c.textBaseline = 'top';
        let tLabel = `${tv}`;
        if (tv === core.E_E) tLabel = isNarrow ? '0 (Ee)' : '0 mV (Ee)';
        else if (tv === core.E_L) tLabel = isNarrow ? '−65 (EL)' : '−65 mV (EL)';
        else tLabel = isNarrow ? `${tv}` : `${tv} mV`;
        c.fillText(tLabel, tx, axisY + 6);
      }

      // 12. Draggable Ei Handle Marker
      const eiTagY = axisY + 24;
      c.save();
      const tagText = isNarrow ? `◀ Ei: ${state.Ei.toFixed(0)} ▶` : `◀ Drag Ei (${state.Ei.toFixed(0)} mV) ▶`;
      c.font = (isNarrow ? '700 11px' : '700 12px') + ' ui-monospace';
      const eiTextW = c.measureText(tagText).width;
      const tagW = eiTextW + 14;
      const tagH = isNarrow ? 20 : 22;

      c.fillStyle = col.paperCard || '#ffffff';
      c.strokeStyle = col.inh;
      c.lineWidth = 1.8;
      c.beginPath();
      if (c.roundRect) c.roundRect(eiX - tagW / 2, eiTagY, tagW, tagH, 3);
      else c.rect(eiX - tagW / 2, eiTagY, tagW, tagH);
      c.fill();
      c.stroke();

      // Small notch pointing to the axis
      c.beginPath();
      c.moveTo(eiX - 4, eiTagY);
      c.lineTo(eiX + 4, eiTagY);
      c.lineTo(eiX, axisY + 4);
      c.closePath();
      c.fillStyle = col.inh;
      c.fill();

      c.fillStyle = col.inh;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(tagText, eiX, eiTagY + tagH / 2);
      c.restore();

      // 13. Oscilloscope / Membrane Voltage Trace Plot
      const plotX = pad;
      const plotY = axisY + (isNarrow ? 52 : 56);
      const plotW = axisW;
      const plotH = Math.max(90, h - plotY - 22);

      // Plot card frame
      c.save();
      c.fillStyle = col.paperCard;
      c.strokeStyle = col.border;
      c.lineWidth = 1;
      c.beginPath();
      if (c.roundRect) c.roundRect(plotX, plotY, plotW, plotH, 3);
      else c.rect(plotX, plotY, plotW, plotH);
      c.fill();
      c.stroke();

      // Plot Title & Legend
      c.font = '600 12px ui-serif, Georgia';
      c.fillStyle = col.ink;
      c.textAlign = 'left';
      c.textBaseline = 'top';
      c.fillText(isNarrow ? 'Oscilloscope v(t)' : S.traceTitle, plotX + 8, plotY + 6);

      c.font = (isNarrow ? '10px' : '12px') + ' ui-monospace';
      c.textAlign = 'right';
      let legendStr = '— v(t)';
      if (state.theory) legendStr += isNarrow ? ' · -- v∞' : ' · ' + S.ghostLegend;
      if (state.chapter === 4) legendStr += isNarrow ? ' · ·· unshunted' : ' · ' + S.unshuntedLegend;
      if (state.mode === 'current') legendStr += isNarrow ? ' · -- curr' : ' · ' + S.ghostCurrLegend;
      c.fillStyle = col.inkMuted;
      c.fillText(legendStr, plotX + plotW - 8, plotY + (isNarrow ? 7 : 6));

      // Y-axis gridlines (-80, -65, -50, 0 mV)
      const vToPlotY = (v) => plotY + plotH - 12 - ((v - (-90)) / 100) * (plotH - 24);
      const gridLevels = [-80, -65, -50, 0];
      for (const gv of gridLevels) {
        const gy = vToPlotY(gv);
        strokeLine(c, plotX + 28, gy, plotX + plotW - 8, gy, col.border, 1, [3, 3]);
        c.font = '12px ui-monospace';
        c.fillStyle = gv === core.E_L ? col.ink : col.inkMuted;
        c.textAlign = 'right';
        c.textBaseline = 'middle';
        c.fillText(`${gv}`, plotX + 24, gy);
      }

      // Draw Voltage Traces strictly clipped to oscilloscope screen
      c.save();
      c.beginPath();
      c.rect(plotX + 1, plotY + 1, plotW - 2, plotH - 2);
      c.clip();

      const toPlotX = (idx, len) => plotX + 30 + (idx / Math.max(1, len - 1)) * (plotW - 38);

      // In Chapter 4: draw overlaid unshunted response comparison trace!
      if (state.chapter === 4) {
        c.save();
        c.strokeStyle = col.inkMuted;
        c.lineWidth = 1.8;
        c.setLineDash([2, 2]);
        c.beginPath();
        for (let i = 0; i < state.unshuntedHistory.length; i++) {
          const px = toPlotX(i, state.unshuntedHistory.length);
          const py = vToPlotY(state.unshuntedHistory[i]);
          if (i === 0) c.moveTo(px, py);
          else c.lineTo(px, py);
        }
        c.stroke();
        c.restore();
      }

      // In Chapter 5 or Current mode: draw overlaid current synapse model trace
      if (state.mode === 'current') {
        c.save();
        c.strokeStyle = col.trace;
        c.lineWidth = 1.8;
        c.setLineDash([4, 3]);
        c.beginPath();
        for (let i = 0; i < state.currentHistory.length; i++) {
          const px = toPlotX(i, state.currentHistory.length);
          const py = vToPlotY(state.currentHistory[i]);
          if (i === 0) c.moveTo(px, py);
          else c.lineTo(px, py);
        }
        c.stroke();
        c.restore();
      }

      // Theory ghost trace
      if (state.theory) {
        c.save();
        c.strokeStyle = col.theory;
        c.lineWidth = 1.6;
        c.setLineDash([5, 3]);
        c.beginPath();
        for (let i = 0; i < state.theoryHistory.length; i++) {
          const px = toPlotX(i, state.theoryHistory.length);
          const py = vToPlotY(state.theoryHistory[i]);
          if (i === 0) c.moveTo(px, py);
          else c.lineTo(px, py);
        }
        c.stroke();
        c.restore();
      }

      // Live simulated voltage trace
      c.save();
      c.strokeStyle = col.volt;
      c.lineWidth = 2.4;
      c.beginPath();
      for (let i = 0; i < state.history.length; i++) {
        const px = toPlotX(i, state.history.length);
        const py = vToPlotY(state.history[i]);
        if (i === 0) c.moveTo(px, py);
        else c.lineTo(px, py);
      }
      c.stroke();
      c.restore();

      c.restore(); // End clipping

      // Axis units label
      c.font = '12px ui-monospace';
      c.fillStyle = col.inkMuted;
      c.textAlign = 'right';
      c.textBaseline = 'bottom';
      c.fillText('Time (ms) →', plotX + plotW - 8, plotY + plotH - 3);

      // Footnote hint (hidden on very narrow screens)
      if (!isNarrow) {
        c.font = '12px ui-serif, Georgia';
        c.fillStyle = col.inkMuted;
        c.textAlign = 'center';
        c.textBaseline = 'bottom';
        c.fillText(S.dragHint, w / 2, h - 3);
      }

      c.restore();

      // Update Live DOM Readouts
      if (byId('n07-val-v')) byId('n07-val-v').textContent = state.v.toFixed(1) + ' mV';
      if (byId('n07-val-vinf')) byId('n07-val-vinf').textContent = vinf.toFixed(1) + ' mV';
      if (byId('n07-val-tau')) byId('n07-val-tau').textContent = tau.toFixed(2) + ' ms';
      if (byId('n07-val-gtot')) byId('n07-val-gtot').textContent = core.totalConductance(p).toFixed(2) + ' µS';
      if (byId('n07-val-ie')) byId('n07-val-ie').textContent = currentValues.exc.toFixed(1) + ' pA';
      if (byId('n07-val-ii')) byId('n07-val-ii').textContent = currentValues.inh.toFixed(1) + ' pA';
      if (byId('n07-val-il')) byId('n07-val-il').textContent = currentValues.leak.toFixed(1) + ' pA';
      if (byId('n07-val-diff')) byId('n07-val-diff').textContent = Math.abs(state.v - vinf).toFixed(3) + ' mV';
      if (byId('n07-val-model-diff')) byId('n07-val-model-diff').textContent = Math.abs(state.v - state.vCurrent).toFixed(3) + ' mV';
    }

    // Tick Simulation Step
    function tick(now) {
      const elapsed = Math.min(50, Math.max(1, now - state.lastTimestamp));
      state.lastTimestamp = now;

      if (state.running) {
        const dt = elapsed * 0.12; // Simulation timestep in ms

        // Scripted pulse in Chapter 1: auto-pulse periodically if student is idle
        if (state.chapter === 1) {
          state.demoTimer += dt;
          if (state.demoTimer > 90.0) {
            state.demoTimer = 0;
            triggerPulse(0.40, 4.0);
          }
        }

        const p = params();

        // Advance pulse decay
        if (state.pulse > 0) {
          state.pulse = Math.max(0, state.pulse - dt);
        }

        // Conductance exact step
        state.v = core.exactStep(state.v, p, dt);

        // Current model comparison step
        // In current mode, I_inj equals the excitatory current at rest (-65 mV)
        const iInj = p.ge * (core.E_E - core.E_L);
        state.vCurrent = core.currentStep(state.vCurrent, p, iInj, dt);

        // Unshunted comparison step for Chapter 4
        const pUnshunted = { ...p, gi: 0, Ei: -80 };
        const vUnshunted = core.exactStep(state.unshuntedHistory[state.unshuntedHistory.length - 1], pUnshunted, dt);

        // Theory vInfinity step
        const vinf = core.vInfinity(p);

        // Push histories
        state.history.push(state.v);
        state.theoryHistory.push(vinf);
        state.currentHistory.push(state.vCurrent);
        state.unshuntedHistory.push(vUnshunted);

        if (state.history.length > HISTORY_LEN) {
          state.history.shift();
          state.theoryHistory.shift();
          state.currentHistory.shift();
          state.unshuntedHistory.shift();
        }

        state.t += dt;
      }

      draw(colors());
    }

    // Animation loop: runs both via requestAnimationFrame and ctx.onFrame
    let animId = null;
    function loop(now) {
      tick(now);
      animId = requestAnimationFrame(loop);
    }
    animId = requestAnimationFrame(loop);

    if (typeof ctx.onFrame === 'function') {
      ctx.onFrame(() => tick(performance.now()));
    }

    // Initial setup
    applyChapter(1);
    draw(colors());
  }

  // CommonJS export for Node tests
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core, S };
  }

  // Atlas Plate Registration for browser shell
  const AtlasObj =
    typeof globalThis !== 'undefined' && globalThis.Atlas
      ? globalThis.Atlas
      : typeof Atlas !== 'undefined'
        ? Atlas
        : root && root.Atlas;

  if (AtlasObj && AtlasObj.Plate) {
    AtlasObj.Plate.register({
      id: S.id,
      title: { en: S.title },
      source: S.source,
      mount,
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
