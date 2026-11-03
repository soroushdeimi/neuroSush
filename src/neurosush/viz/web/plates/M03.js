/**
 * Plate M03 — The Nernst equation: why a membrane has a voltage.
 * Two compartments separated by a membrane with one selective channel.
 * Ions random-walk (Brownian) and cross as illustrative markers driven by an imposed
 * RC relaxation toward the Nernst voltage; the escaping charge separates across
 * the bilayer until electric drift cancels diffusion at
 * E = (RT/zF) · ln([out]/[in]). The voltage bookkeeping is the continuum leak ODE
 * C dV/dt = −g (V − E) with E the Nernst potential of the live mM values;
 * the 200 markers are the picture of the flux, not the bookkeeping.
 * Source: live — mirrors the Nernst arithmetic behind the reversal potentials
 * (e_exc / e_inh) that neurosush.neurons.dendrite.conductance_step consumes.
 */
(function (root) {
  'use strict';

  const isNode = typeof module !== 'undefined' && !!module.exports;

  // ---- core: pure math + particle sim, no DOM, exported for node tests -------
  const R_GAS = 8.314462618;   // J/(mol·K)
  const F_FARADAY = 96485.332; // C/mol

  const S = {
    title: 'The Nernst equation: why a membrane has a voltage',
    footer:
      'live · Nernst arithmetic behind the reversal potentials (e_exc / e_inh) of ' +
      'neurosush.neurons.dendrite.conductance_step',
    equations: {
      nernst: 'E = (RT/zF) · ln([out]/[in])',
      ode: 'C·dV/dt = −g · (V − E)'
    },
    terms: {
      RToverF: 'thermal volts RT/zF: 26.7 mV per unit charge at 37 °C',
      ratio: 'concentration ratio [out]/[in], dimensionless',
      nernst: 'Nernst potential E: zero net flux voltage',
      g: 'channel conductance g: how freely the selected ion crosses',
      drive: 'driving force V − E: distance from balance'
    },
    glossary: {
      diffusion: 'Diffusion — ions wander from crowded to empty with no force pushing; a purely statistical walk.',
      anion: 'Fixed anions — large immobile negative proteins stay inside; they cannot follow the cation out, so charge separates.',
      drift: 'Electric drift — a charge in the membrane field feels qE and is steered back or forward.',
      nernst: 'Nernst potential — the voltage at which drift exactly cancels diffusion: zero net flux.',
      decade: 'Decade slope — each tenfold ratio adds RT/zF·ln 10 per unit charge: 61.5 mV at 37 °C, 58.2 mV at 20 °C.'
    },
    chapters: [
      {
        caption: 'A gradient wants to equalize. K⁺ is crowded inside (140 mM) and sparse outside (5 mM). The channel is shut: ions jitter at random and nothing flows.',
        open: false, ion: 'K', temp: 37,
        watch: 'Watch: pure random walks, no net flow; fixed anions (−) sit in the inside pool.'
      },
      {
        caption: 'Ions leave, a voltage appears. The pore opens and K⁺ leaks out down its gradient; every escaping + charge leaves a fixed anion behind, so the inside turns negative.',
        open: true, ion: 'K', temp: 37,
        watch: 'Watch: the voltmeter swings negative and pulses mark each ion that crosses.'
      },
      {
        caption: 'The voltage pushes back. The field now drags K⁺ inward. Diffusion still pushes out; electric drift pulls in. Their toggle arrow shows each force.',
        open: true, ion: 'K', temp: 37,
        watch: 'Watch: the drift arrow grows as |V| rises. Freeze (F) to compare both arrows at one instant.'
      },
      {
        caption: 'Balance — the Nernst potential. Net flux → 0 exactly when drift cancels diffusion, and the amber trace rests on E = (RT/zF)·ln([out]/[in]).',
        open: true, ion: 'K', temp: 37,
        watch: 'Watch: trace lands on the dashed theory line, Δ → 0. Switch the channel to Na⁺ / Cl⁻ (concentrations swap too).'
      },
      {
        caption: 'Predict, then a real neuron. At 37 °C E_K ≈ −89 mV, E_Na ≈ +61 mV; rest sits near E_K because the K⁺ leak dominates. Tenfold [out] moves E by 61.5 mV.',
        open: true, ion: 'K', temp: 37,
        predict: true,
        watch: 'Predict first. Then drag the OUT bar (or press verify) and check the dashed line moved exactly 61.5 mV.'
      }
    ]
  };

  // shared geometry: core sim and ui canvas use the same coordinates
  const GEOM = {
    W: 980, H: 618,
    traceX: 18, traceY: 14, traceW: 944, traceH: 188,
    x0: 18, x1: 962, y0: 266, y1: 604,
    mid: 434, membH: 26,
    ch0: 436, ch1: 560,
    meterX: 912, meterY: 434, meterR: 30
  };

  const core = {
    // Nernst potential in mV. cOut, cIn in the same units; z signed valence;
    // tempC in °C. E = (RT/zF)·ln(cOut/cIn).
    nernst: function (cOut, cIn, z, tempC) {
      if (!isFinite(cOut) || !isFinite(cIn)) {
        throw new Error('concentrations must be finite, got [out]=' + cOut + ' [in]=' + cIn);
      }
      if (!(cOut > 0) || !(cIn > 0)) {
        throw new Error('concentrations must be positive, got [out]=' + cOut + ' [in]=' + cIn);
      }
      if (!isFinite(z) || z === 0) throw new Error('z must be a finite nonzero number, got ' + z);
      return core.rtOverF_mV(tempC) / z * Math.log(cOut / cIn);
    },
    // RT/F for z = 1, in mV, at tempC °C.
    rtOverF_mV: function (tempC) {
      if (!isFinite(tempC) || tempC <= -273.15) {
        throw new Error('temperature must be a finite °C strictly above −273.15, got ' + tempC);
      }
      return 1000 * R_GAS * (273.15 + tempC) / F_FARADAY;
    },
    // Voltage shift per tenfold concentration ratio, mV, per unit |z|.
    perDecade_mV: function (tempC, zAbs) {
      if (!(zAbs > 0)) throw new Error('|z| must be positive, got ' + zAbs);
      return core.rtOverF_mV(tempC) * Math.log(10) / zAbs;
    },
    // One exact analytic step of C·dV/dt = −g·(V−E): dV = (E−V)(1 − e^(−dt/τ)).
    // With C in µF and g in nS the membrane time constant is τ = (C/g)·1000000 ms.
    // Returns {dV}; never mutates inputs.
    odeStep: function (V_mV, E_mV, g_nS, C_uF, dt_ms) {
      if (!(C_uF > 0)) throw new Error('capacitance must be positive, got ' + C_uF);
      if (!(g_nS >= 0)) throw new Error('conductance must be non-negative, got ' + g_nS);
      if (!(dt_ms > 0)) throw new Error('time step must be positive, got ' + dt_ms);
      const tau = (C_uF / g_nS) * 1000000; // ms
      if (g_nS === 0) return { dV: 0 };
      const dV = (E_mV - V_mV) * (1 - Math.exp(-dt_ms / tau));
      return { dV };
    },
    // Seeded PRNG (mulberry32) — determinism under a fixed seed.
    mulberry32: function (seed) {
      if (!isFinite(seed)) throw new Error('seed must be finite, got ' + seed);
      let a = seed >>> 0;
      return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    },
    // One Brownian step plus electrophoretic drift ∝ mV·driftPerMv in y.
    // Returns the new y; pure; never mutates the caller.
    particleStep: function (y, rand, sigma, driftPerMv, V_mV) {
      const jump = sigma * 2 * (rand() - 0.5);
      return y + jump + driftPerMv * V_mV;
    },
    // Crossing demand of the ODE for one tick: the continuum book
    // C·dV/dt = −g(V−E) says how much charge leaked this tick (dV) and which
    // way it flowed (outward for a cation iff z·dV < 0). base is the marker
    // rate from that dV (fraction fate decided by the seeded coin); it is 0
    // exactly when V = E. Zero net flux permits continuing equal inward and
    // outward microscopic fluxes; the markers show net transfer only. Pure.
    fluxDemand: function (V_mV, E_mV, g_nS, C_uF, dt_ms, z) {
      const st = core.odeStep(V_mV, E_mV, g_nS, C_uF, dt_ms);
      const base = Math.abs(st.dV) * 0.3; // picture scale: markers per mV of leak
      if (z === 0) throw new Error('z must be nonzero, got ' + z);
      return { dV: st.dV, base: base, outward: st.dV * z < 0 };
    },
    // Geometric shell of the simulation, in canvas pixels.
    geom: function () {
      return { W: GEOM.W, H: GEOM.H, x0: GEOM.x0, x1: GEOM.x1, y0: GEOM.y0, y1: GEOM.y1,
        mid: GEOM.mid, membH: GEOM.membH, ch0: GEOM.ch0, ch1: GEOM.ch1 };
    },
    // Build the 200-marker pool; the split mirrors the reported mM ratio so the
    // picture matches the numbers at rest. Seeded → reproducible.
    makeSim: function (seed, cIn, cOut, n) {
      const N = n || 200;
      if (!(N > 0)) throw new Error('particle count must be positive, got ' + N);
      const rand = core.mulberry32(seed);
      const g = core.geom();
      const nIn = Math.round(N * cIn / (cIn + cOut));
      const particles = [];
      for (let i = 0; i < N; i++) {
        const inside = i < nIn;
        particles.push({
          inside,
          x: g.x0 + 12 + rand() * (g.x1 - g.x0 - 24),
          y: inside
            ? g.mid + g.membH / 2 + 14 + rand() * (g.y1 - g.mid - g.membH / 2 - 42)
            : g.y0 + 14 + rand() * (g.mid - g.membH / 2 - g.y0 - 42)
        });
      }
      return { N: N, particles: particles, rand: rand, flashes: [] };
    },
    // Advance the pool one picture tick. input: {open, V_mV, E_mV, g_nS, C_uF,
    // z, tempC, dtMs, frozen}. The crossings are the ODE's own flux: the exact
    // implicit step gives dV, and |dV|·scale decides how many markers move.
    // Balance lands exactly at V = E, where the demanded flux is zero.
    // Mutates only sim-owned particles; returns {outN, inN, dV}.
    simStep: function (sim, input) {
      const g = core.geom();
      const p = input || {};
      const dt = p.dtMs || 16;
      if (p.frozen) { core.ageFlashes(sim, dt); return { outN: 0, inN: 0, dV: 0 }; }
      const open = !!p.open && (p.g_nS || 0) > 0;
      const V = p.V_mV || 0;
      const tk = Math.sqrt((273.15 + (p.tempC ?? 37)) / 310.15);
      const sigFree = 1.7 * tk;
      const sigPore = 0.62 * tk;
      const driftPerMv = -0.0008 * (p.z || 1); // px/ms per mV: force of the field
      const inBot = g.y1 - 20;
      const outBot = g.mid - g.membH / 2 - 2;
      // jitter everywhere; inside the pore column the walk is tight and the
      // field nudges the z-ion across the bilayer direction
      for (const q of sim.particles) {
        const nearPore = q.x > g.ch0 && q.x < g.ch1;
        const sigma = nearPore ? sigPore : sigFree;
        const drift = nearPore && open ? driftPerMv : 0;
        q.y = core.particleStep(q.y, sim.rand, sigma, drift, V);
        if (q.inside) {
          if (q.y > inBot) q.y = inBot - (q.y - inBot);
          if (q.y < g.mid + g.membH / 2 + 2) q.y = g.mid + g.membH / 2 + 2;
        } else {
          if (q.y < g.y0 + 10) q.y = g.y0 + 10 + (g.y0 + 10 - q.y);
          if (q.y > outBot) q.y = outBot;
        }
      }
      let outN = 0, inN = 0, dV = 0;
      if (open) {
        const demand = core.fluxDemand(V, p.E_mV || 0, p.g_nS || 0, p.C_uF || 1, dt, p.z || 1);
        dV = demand.dV;
        let quota = Math.floor(demand.base) + (sim.rand() < (demand.base % 1) ? 1 : 0);
        const candidates = sim.particles.filter(q => q.x > g.ch0 + 6 && q.x < g.ch1 - 6 &&
          (demand.outward ? q.inside : !q.inside));
        for (let tries = 0; tries < quota && candidates.length > 0; tries++) {
          const idx = Math.floor(sim.rand() * candidates.length);
          const q = candidates.splice(idx, 1)[0];
          q.inside = !q.inside;
          q.x = g.ch0 + 10 + sim.rand() * (g.ch1 - g.ch0 - 20);
          q.y = q.inside ? g.mid + g.membH / 2 + 10 : g.mid - g.membH / 2 - 10;
          sim.flashes.push({ x: q.x, y: q.y, r: 2, alpha: 1.0, outward: demand.outward });
          if (q.inside) inN++; else outN++;
        }
      }
      core.ageFlashes(sim, dt);
      return { outN, inN, dV };
    },
    ageFlashes: function (sim, dtMs) {
      for (let i = sim.flashes.length - 1; i >= 0; i--) {
        const f = sim.flashes[i];
        f.alpha -= dtMs / 480;
        f.r += dtMs * 0.03;
        if (f.alpha <= 0) sim.flashes.splice(i, 1);
      }
    },
    rebalance: function (sim, cIn, cOut) {
      const wantIn = Math.round(sim.N * cIn / (cIn + cOut));
      let haveIn = 0;
      for (const q of sim.particles) if (q.inside) haveIn++;
      const d = wantIn - haveIn;
      if (d === 0) return false;
      if (Math.abs(d) > sim.N * 0.04 && sim.rand() < 0.5) {
        let k = Math.floor(sim.rand() * sim.N);
        for (let tries = 0; tries < 8; tries++) {
          if (sim.particles[k].inside !== (d > 0)) break;
          k = (k + 1) % sim.N;
        }
        sim.particles[k].inside = (d > 0);
        return true;
      }
      return false;
    }
  };

  core.R_GAS = R_GAS;
  core.F_FARADAY = F_FARADAY;
  core.S = S;
  core.GEOM = GEOM;

  if (isNode) module.exports = { core, S };

  // ---- ui: only runs when an Atlas shell exists -------------------------------
  function mount(rootEl, ctx) {
    if (isNode || typeof document === 'undefined') return;
    const fig = (rootEl && rootEl.figure) || rootEl ||
      document.getElementById('atlas-canvas-container') || document.body;
    const marginEl = ctx.margin || null;
    const createCanvas = ctx.canvas;
    const onFrame = ctx.onFrame || function () {};
    const colors = ctx.colors || function () {
      return { volt: '#b45309', exc: '#0f766e', inh: '#be123c', mod: '#7e22ce',
        trace: '#1d4ed8', theory: '#1b1a17', paper: '#f6f2e9', paperCard: '#eeeae0',
        ink: '#1b1a17', inkMuted: '#57534e', border: '#d6cfbf' };
    };

    const styleEl = document.createElement('style');
    styleEl.textContent = `
      .m03-wrap { font-family: var(--font-sans, system-ui, sans-serif); color: var(--ink, #1b1a17); padding: 4px 2px; }
      .m03-eqlabel { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-muted, #57534e); margin-bottom: 4px; }
      .m03-live { font-weight: 400; text-transform: none; color: var(--volt, #b45309); }
      .m03-eq { font-family: var(--font-mono, monospace); font-size: 12px; background: var(--paper-card, #eeeae0); border: 1px solid var(--border, #d6cfbf); border-radius: 4px; padding: 6px 8px; margin-bottom: 6px; }
      .m03-explain { font-size: 11.5px; line-height: 1.4; color: var(--ink-muted, #57534e); margin-bottom: 8px; }
      .m03-gloss { font-size: 11.5px; line-height: 1.4; background: var(--paper-inset, #f4f0e6); border-left: 3px solid var(--exc, #0f766e); padding: 6px 8px; margin-bottom: 10px; border-radius: 0 4px 4px 0; min-height: 44px; }
      .m03-ctl { display: flex; flex-direction: column; gap: 8px; border-top: 1px solid var(--border, #d6cfbf); padding-top: 8px; }
      .m03-row { display: flex; align-items: center; justify-content: space-between; gap: 6px; font-size: 12px; }
      .m03-lab { font-weight: 600; font-size: 11.5px; min-width: 64px; }
      .m03-btn { font-size: 11.5px; padding: 3px 8px; border: 1px solid var(--border); border-radius: 3px; background: var(--paper-card); cursor: pointer; }
      .m03-btn.active { background: var(--volt); color: #fff; border-color: var(--volt); }
      .m03-range { flex: 1; accent-color: var(--volt); }
      .m03-val { font-family: var(--font-mono); font-size: 11.5px; min-width: 52px; text-align: right; }
      .m03-footer { font-size: 11px; color: var(--ink-muted); margin-top: 6px; padding: 2px 4px; line-height: 1.3; }
      .m03-foot-src { font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }
    `;
    fig.appendChild(styleEl);

    const st = {
      playing: !reducedMotion(),
      chapter: 0,
      ion: 'K',
      inside: 140,
      outside: 5,
      tempC: 37,
      g_nS: 10,
      C_uF: 0.001, // 1 nF = 0.001 µF
      V_mV: 0,
      t_ms: 0,
      frozen: false,
      showTheory: true,
      channelOpen: false,
      gateAnim: 0,           // 0 shut .. 1 open (follows channelOpen in sim time)
      hoverTerm: null,
      dragHandle: null,
      lastX: 0,
      predictChoice: null,
      revealed: false,
      verify: null,          // {out0, E0} active after reveal
      trace: []
    };
    const anions = [];

    function reducedMotion() {
      return typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    function E_of() {
      return core.nernst(st.outside, st.inside, ionData[st.ion].z, st.tempC);
    }

    // -- layout on the canvas ---------------------------------------------------
    function reseedParticles() {
      const d = ionData[st.ion];
      sim.flashes.length = 0;
      const fresh = core.makeSim(0x1EA7, d.in, d.out, 200);
      sim.N = fresh.N;
      sim.rand = fresh.rand;
      sim.particles.length = 0;
      for (const q of fresh.particles) sim.particles.push({ inside: q.inside, x: q.x, y: q.y });
      const aRnd = core.mulberry32(0x8E17);
      anions.length = 0;
      for (let i = 0; i < 16; i++) {
        anions.push({
          x: GEOM.x0 + 40 + aRnd() * 300,
          y: GEOM.mid + GEOM.membH / 2 + 30 + aRnd() * (GEOM.y1 - GEOM.mid - GEOM.membH / 2 - 66)
        });
      }
    }
    const sim = core.makeSim(0x1EA7, 140, 5, 200);

    // -- main canvas ------------------------------------------------------------
    const cvs = createCanvas(GEOM.W, GEOM.H);
    cvs.style.width = '100%';
    cvs.style.height = 'auto';
    cvs.style.touchAction = 'none';
    cvs.setAttribute('role', 'img');
    cvs.setAttribute('aria-label',
      'Two compartments with a selective channel, random-walking ions, the electric field across the ' +
      'membrane, a voltmeter and a voltage trace settling on the dashed Nernst value');

    // honest footer under the figure
    const footerEl = document.createElement('div');
    footerEl.className = 'm03-footer';
    footerEl.innerHTML =
      '<span class="m03-foot-src">source: live</span> — JS port of the Nernst arithmetic, seeded particle ' +
      'picture; mirrors the reversal potentials (e_exc / e_inh) behind ' +
      '<code>neurosush.neurons.dendrite.conductance_step</code> · keys: Space play/pause · T theory · ' +
      '←/→ chapters · F freeze';
    fig.appendChild(footerEl);

    // -- margin (equations, glossary, controls) -----------------------------------
    if (marginEl) {
      marginEl.innerHTML = '';
      const wrap = document.createElement('div');
      wrap.className = 'm03-wrap';
      wrap.innerHTML =
        '<div class="m03-eqlabel">governing equations <span class="m03-live">· live values</span></div>' +
        '<div class="m03-eq" id="m03-eq-nernst"></div>' +
        '<div class="m03-eq" id="m03-eq-ode"></div>' +
        '<div class="m03-explain">The 200 markers are the picture of the flux; the voltage keeps its ' +
        'own clean books in the continuum leak ODE, whose E follows the mM concentrations.</div>' +
        '<div class="m03-gloss" id="m03-gloss" aria-live="polite"></div>' +
        '<details id="m03-learn" ' + (typeof window !== 'undefined' && window.innerWidth >= 620 ? 'open' : '') + ' style="margin-top:10px;border-top:1px solid var(--border);padding-top:8px;">' +
        '<summary style="cursor:pointer;font-weight:600;" class="m03-eqlabel">Learn: the math, the brain, the ML analogy</summary>' +
        '<div style="font-size:11px;line-height:1.4;margin-top:6px;display:flex;flex-direction:column;gap:6px;">' +
        '<div><b>1. What you see:</b> Two compartments separated by a membrane with ion channels and random-walking markers representing ions diffusing down concentration gradients and drifting in the membrane electric field.</div>' +
        '<div><b>2. The math:</b> Nernst equation E = (RT / zF) · ln([out] / [in]), where R = 8.314 J/(mol·K), F = 96485 C/mol, thermal voltage RT / F = 26.7 mV at 37 °C. Membrane voltage relaxes via C · dV/dt = -g · (V - E), with time constant τ = (C / g) · 1000000 = 100 ms (for C = 0.001 µF = 1 nF, g = 10 nS).</div>' +
        '<div><b>3. In the brain:</b> Electrochemical equilibrium for individual ion species (K⁺, Na⁺, Cl⁻). Concentration gradients generate resting and reversal potentials across the neuronal lipid bilayer, establishing the driving force for synaptic transmission.</div>' +
        '<div><b>4. Like classic ML:</b> Nernst equation corresponds to log-odds equilibrium from occupancy ratios (like logit/softmax balance), and RC relaxation is a first-order exponential moving average. However, neither is a learned probability or optimization objective; physical thermodynamics fix the equilibrium.</div>' +
        '</div>' +
        '</details>' +
        controlsHTML();
      marginEl.appendChild(wrap);
      wireControls();
      marginEl.addEventListener('pointerover', onTermOver);
      marginEl.addEventListener('pointerout', onTermOut);
    } else {
      const wrap = document.createElement('div');
      wrap.className = 'm03-wrap';
      wrap.innerHTML = '<div class="m03-eq" id="m03-eq-nernst">margin unavailable</div>';
      fig.appendChild(wrap);
    }

    function controlsHTML() {
      return '<div class="m03-ctl">' +
        '<div class="m03-row"><span class="m03-lab">channel</span>' +
        ['K', 'Na', 'Cl'].map(k =>
          '<button class="m03-btn" data-ion="' + k + '" id="m03-ion-' + k + '">' + k + '</button>'
        ).join('') +
        '<button class="m03-btn" id="m03-freeze">Freeze [F]</button>' +
        '</div>' +
        '<div class="m03-row"><span class="m03-lab">OUT [out]</span>' +
        '<input type="range" class="m03-range" id="m03-out-rng" min="1" max="150" step="1" value="5">' +
        '<span class="m03-val" id="m03-out-val">5 mM</span></div>' +
        '<div class="m03-row"><span class="m03-lab">IN [in]</span>' +
        '<input type="range" class="m03-range" id="m03-in-rng" min="1" max="200" step="1" value="140">' +
        '<span class="m03-val" id="m03-in-val">140 mM</span></div>' +
        '<div class="m03-row"><span class="m03-lab">temp °C</span>' +
        '<input type="range" class="m03-range" id="m03-temp-rng" min="0" max="45" step="1" value="37">' +
        '<span class="m03-val" id="m03-temp-val">37 °C</span></div>' +
        '<div class="m03-row"><span class="m03-lab">pore g</span>' +
        '<input type="range" class="m03-range" id="m03-g-rng" min="0" max="30" step="1" value="10">' +
        '<span class="m03-val" id="m03-g-val">10 nS</span></div>' +
        '<div class="m03-row"><button class="m03-btn" id="m03-play">Pause</button>' +
        '<button class="m03-btn" id="m03-reset">Reset trace</button>' +
        '<button class="m03-btn" id="m03-theory-btn">Theory [T]</button></div>' +
        '</div>';
    }

    const ionData = {
      K:  { z: 1,  out: 5,   in: 140, name: 'Potassium (K⁺)' },
      Na: { z: 1,  out: 145, in: 15,  name: 'Sodium (Na⁺)' },
      Cl: { z: -1, out: 110, in: 10,  name: 'Chloride (Cl⁻)' }
    };

    function setIon(k) {
      if (!ionData[k]) return;
      st.ion = k;
      const d = ionData[k];
      st.outside = d.out;
      st.inside = d.in;
      const oR = document.getElementById('m03-out-rng');
      const iR = document.getElementById('m03-in-rng');
      if (oR) { oR.value = st.outside; oR.max = k === 'Cl' ? 150 : 160; }
      if (iR) { iR.value = st.inside; iR.max = k === 'Cl' ? 140 : 200; }
      reseedParticles();
      ['K', 'Na', 'Cl'].forEach(id => {
        const b = document.getElementById('m03-ion-' + id);
        if (b) b.classList.toggle('active', id === k);
      });
      st.V_mV = E_of();
      st.trace = [{ t: st.t_ms, V: st.V_mV, E: E_of() }];
    }

    function wireControls() {
      ['K', 'Na', 'Cl'].forEach(k => {
        document.getElementById('m03-ion-' + k).onclick = () => setIon(k);
      });
      document.getElementById('m03-freeze').onclick = () => {
        st.frozen = !st.frozen;
        document.getElementById('m03-freeze').classList.toggle('active', st.frozen);
      };
      const oR = document.getElementById('m03-out-rng');
      const iR = document.getElementById('m03-in-rng');
      const tR = document.getElementById('m03-temp-rng');
      const gR = document.getElementById('m03-g-rng');
      oR.oninput = e => { st.outside = parseFloat(e.target.value); };
      iR.oninput = e => { st.inside = parseFloat(e.target.value); };
      tR.oninput = e => { st.tempC = parseFloat(e.target.value); };
      gR.oninput = e => { st.g_nS = parseFloat(e.target.value); };

      document.getElementById('m03-play').onclick = () => {
        st.playing = !st.playing;
        document.getElementById('m03-play').textContent = st.playing ? 'Pause' : 'Play';
      };
      document.getElementById('m03-reset').onclick = () => {
        st.t_ms = 0;
        st.V_mV = 0;
        st.trace = [{ t: 0, V: 0, E: E_of() }];
      };
      document.getElementById('m03-theory-btn').onclick = () => {
        st.showTheory = !st.showTheory;
        document.getElementById('m03-theory-btn').classList.toggle('active', st.showTheory);
      };
      setIon('K');
    }

    function onTermOver(e) {
      const el = e.target.closest('[data-term]');
      if (!el) return;
      st.hoverTerm = el.getAttribute('data-term');
      updateGlossary();
    }
    function onTermOut(e) {
      if (e.target.closest('[data-term]')) {
        st.hoverTerm = null;
        updateGlossary();
      }
    }

    function updateGlossary() {
      const g = document.getElementById('m03-gloss');
      if (!g) return;
      const t = st.hoverTerm;
      if (t && S.glossary[t]) {
        g.innerHTML = '<b>' + t + ':</b> ' + S.glossary[t];
      } else {
        g.innerHTML = '<b>Tip:</b> Hover over terms in the equations above (or symbols on the canvas) to inspect them.';
      }
    }
    updateGlossary();

    // -- physics -----------------------------------------------------------------
    function physicsStep(dtMs) {
      if (!st.playing) return;
      const E = E_of();
      const g = st.channelOpen && !st.frozen ? st.g_nS : 0;
      let dV = 0;
      if (g > 0) {
        dV = core.odeStep(st.V_mV, E, g, st.C_uF, dtMs).dV;
      }
      st.V_mV += dV;
      st.t_ms += dtMs;
      st.gateAnim += ((st.channelOpen ? 1 : 0) - st.gateAnim) * Math.min(1, dtMs / 120);
      if (!st.trace.length || st.t_ms - st.trace[st.trace.length - 1].t >= 40) {
        st.trace.push({ t: st.t_ms, V: st.V_mV, E: E_of() });
        while (st.trace.length > 700) st.trace.shift();
      }
      core.rebalance(sim, st.inside, st.outside);
    }

    function particlesStep(dtMs) {
      if (!st.playing) return;
      core.simStep(sim, {
        open: st.channelOpen && !st.frozen,
        V_mV: st.V_mV,
        E_mV: E_of(),
        g_nS: st.g_nS,
        C_uF: st.C_uF,
        z: ionData[st.ion].z,
        tempC: st.tempC,
        dtMs: dtMs
      });
    }

    // chapter wiring
    const chBar = document.createElement('div');
    chBar.className = 'm03-chapter';
    chBar.style.cssText = 'background:var(--paper-card);border:1px solid var(--border);border-radius:6px;padding:10px 12px;margin-bottom:8px;font-family:var(--font-sans);';
    fig.insertBefore(chBar, cvs);

    function setChapter(n) {
      st.chapter = Math.max(0, Math.min(S.chapters.length - 1, n));
      const c = S.chapters[st.chapter];
      st.channelOpen = !!c.open;
      const d = document.getElementById('m03-chapter-content');
      if (d) d.innerHTML = '<b>Chapter ' + (st.chapter + 1) + ' of ' + S.chapters.length + ':</b> ' + c.caption + '<br><span style="color:var(--volt);font-size:11.5px">' + c.watch + '</span>';
      const prev = document.getElementById('m03-ch-prev');
      const next = document.getElementById('m03-ch-next');
      if (prev) prev.disabled = st.chapter === 0;
      if (next) next.disabled = st.chapter === S.chapters.length - 1;
      buildPredictionBox();
    }

    chBar.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">' +
      '<div style="font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:0.04em">Chapter Journey</div>' +
      '<div><button class="m03-btn" id="m03-ch-prev">← Back</button> <button class="m03-btn" id="m03-ch-next">Next →</button></div></div>' +
      '<div id="m03-chapter-content" style="font-size:12px;line-height:1.4"></div>' +
      '<div id="m03-predict-box"></div>';

    document.getElementById('m03-ch-prev').onclick = () => setChapter(st.chapter - 1);
    document.getElementById('m03-ch-next').onclick = () => setChapter(st.chapter + 1);

    function buildPredictionBox() {
      const box = document.getElementById('m03-predict-box');
      if (!box) return;
      const c = S.chapters[st.chapter];
      if (!c.predict) { box.innerHTML = ''; return; }
      box.innerHTML = '<div style="margin-top:8px;padding:8px;background:var(--paper-inset);border:1px solid var(--border);border-radius:4px;font-size:11.5px">' +
        '<b>Predict-then-reveal:</b> If you increase outside K⁺ from 5 mM to 50 mM (a 10-fold jump), by how many mV does E shift?<br>' +
        '<label><input type="radio" name="m03-p" value="a"> +61.5 mV (more positive)</label><br>' +
        '<label><input type="radio" name="m03-p" value="b"> −61.5 mV (more negative)</label><br>' +
        '<button class="m03-btn" id="m03-commit" style="margin-top:4px">Commit prediction</button>' +
        '<div id="m03-reveal" style="margin-top:4px;display:none;font-weight:600;color:var(--volt)"></div></div>';
      document.getElementById('m03-commit').onclick = () => {
        const sel = box.querySelector('input[name="m03-p"]:checked');
        const rev = document.getElementById('m03-reveal');
        if (!sel || !rev) return;
        rev.style.display = 'block';
        if (sel.value === 'a') {
          rev.textContent = 'Correct! A 10-fold ratio increase adds +61.5 mV, shifting E from −89 mV to −28 mV.';
        } else {
          rev.textContent = 'Not quite! Tenfold out/in ratio increases E by +61.5 mV at 37 °C.';
        }
      };
    }

    setChapter(0);

    // -- render loop ------------------------------------------------------------
    const ctx2d = cvs.getContext('2d');
    let lastT = performance.now();

    function frame(now) {
      const dt = Math.min(50, now - lastT);
      lastT = now;
      physicsStep(dt);
      particlesStep(dt);

      // sync UI inputs
      const oV = document.getElementById('m03-out-rng');
      const iV = document.getElementById('m03-in-rng');
      const tV = document.getElementById('m03-temp-rng');
      const gV = document.getElementById('m03-g-rng');
      if (oV) { oV.value = st.outside; document.getElementById('m03-out-val').textContent = st.outside.toFixed(0) + ' mM'; }
      if (iV) { iV.value = st.inside; document.getElementById('m03-in-val').textContent = st.inside.toFixed(0) + ' mM'; }
      if (tV) { tV.value = st.tempC; document.getElementById('m03-temp-val').textContent = st.tempC.toFixed(0) + ' °C'; }
      if (gV) { gV.value = st.g_nS; document.getElementById('m03-g-val').textContent = st.g_nS.toFixed(0) + ' nS'; }

      const E = E_of();
      const eqN = document.getElementById('m03-eq-nernst');
      const eqO = document.getElementById('m03-eq-ode');
      if (eqN) eqN.innerHTML = '<span data-term="nernst">E = ' + E.toFixed(1) + ' mV</span> = (<span data-term="RToverF">RT/' + Math.abs(ionData[st.ion].z) + 'F</span>) · ln(<span data-term="ratio">' + st.outside + '/' + st.inside + '</span>)';
      if (eqO) eqO.innerHTML = 'C·dV/dt = −<span data-term="g">g</span>(' + st.V_mV.toFixed(1) + ' − ' + E.toFixed(1) + ') · <span data-term="drive">driving force</span>';

      drawCanvas(ctx2d, st, sim, anions, colors());
      onFrame();
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function drawCanvas(c, st, sim, anions, co) {
    const W = GEOM.W, H = GEOM.H;
    c.fillStyle = co.paper;
    c.fillRect(0, 0, W, H);

    // 1. Trace Panel (top)
    c.save();
    c.fillStyle = co.paperCard;
    c.strokeStyle = co.border;
    c.lineWidth = 1;
    c.fillRect(GEOM.traceX, GEOM.traceY, GEOM.traceW, GEOM.traceH);
    c.strokeRect(GEOM.traceX, GEOM.traceY, GEOM.traceW, GEOM.traceH);

    const tMin = 0, tMax = Math.max(5000, st.t_ms);
    const vMin = -120, vMax = 80;
    const tToX = t => GEOM.traceX + 10 + ((t - tMin) / (tMax - tMin || 1)) * (GEOM.traceW - 20);
    const vToY = v => GEOM.traceY + GEOM.traceH - 15 - ((v - vMin) / (vMax - vMin)) * (GEOM.traceH - 30);

    // Grid lines
    for (const gv of [-89, -50, 0, 50]) {
      const gy = vToY(gv);
      c.strokeStyle = co.border;
      c.setLineDash([2, 4]);
      c.beginPath(); c.moveTo(GEOM.traceX + 10, gy); c.lineTo(GEOM.traceX + GEOM.traceW - 10, gy); c.stroke();
      c.setLineDash([]);
      c.fillStyle = co.inkMuted;
      c.font = '10px var(--font-mono, monospace)';
      c.fillText(gv + ' mV', GEOM.traceX + 14, gy - 3);
    }

    // Theory Nernst line
    const E = core.nernst(st.outside, st.inside, ionData[st.ion].z, st.tempC);
    if (st.showTheory && st.trace.length > 0) {
      const ey = vToY(E);
      c.strokeStyle = co.volt;
      c.lineWidth = 1.5;
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(GEOM.traceX + 10, ey);
      c.lineTo(GEOM.traceX + GEOM.traceW - 10, ey);
      c.stroke();
      c.setLineDash([]);
      c.fillStyle = co.volt;
      c.font = '600 10.5px var(--font-sans)';
      c.fillText('Nernst E = ' + E.toFixed(1) + ' mV', GEOM.traceX + GEOM.traceW - 130, ey - 5);
    }

    // Voltage trace polyline
    if (st.trace.length > 1) {
      c.strokeStyle = co.trace;
      c.lineWidth = 2;
      c.beginPath();
      for (let i = 0; i < st.trace.length; i++) {
        const pt = st.trace[i];
        const px = tToX(pt.t);
        const py = vToY(pt.V);
        if (i === 0) c.moveTo(px, py); else c.lineTo(px, py);
      }
      c.stroke();
    }
    c.restore();

    // 2. Compartment chambers (Bottom)
    c.save();
    c.fillStyle = co.paperCard;
    c.strokeStyle = co.border;
    c.lineWidth = 1.5;
    // Outside compartment (top)
    c.fillRect(GEOM.x0, GEOM.y0, GEOM.x1 - GEOM.x0, GEOM.mid - GEOM.membH / 2 - GEOM.y0);
    c.strokeRect(GEOM.x0, GEOM.y0, GEOM.x1 - GEOM.x0, GEOM.mid - GEOM.membH / 2 - GEOM.y0);
    // Inside compartment (bottom)
    c.fillRect(GEOM.x0, GEOM.mid + GEOM.membH / 2, GEOM.x1 - GEOM.x0, GEOM.y1 - (GEOM.mid + GEOM.membH / 2));
    c.strokeRect(GEOM.x0, GEOM.mid + GEOM.membH / 2, GEOM.x1 - GEOM.x0, GEOM.y1 - (GEOM.mid + GEOM.membH / 2));

    // Membrane bilayer bar in center
    c.fillStyle = co.paper;
    c.fillRect(GEOM.x0, GEOM.mid - GEOM.membH / 2, GEOM.x1 - GEOM.x0, GEOM.membH);
    c.strokeRect(GEOM.x0, GEOM.mid - GEOM.membH / 2, GEOM.x1 - GEOM.x0, GEOM.membH);

    // Pore / Channel opening
    const gateW = GEOM.ch1 - GEOM.ch0;
    const openFrac = st.gateAnim;
    c.clearRect(GEOM.ch0 + 2, GEOM.mid - GEOM.membH / 2 - 1, gateW - 4, GEOM.membH + 2);
    c.fillStyle = co.paperCard;
    c.fillRect(GEOM.ch0 + 2, GEOM.mid - GEOM.membH / 2 - 1, gateW - 4, GEOM.membH + 2);
    c.strokeStyle = co.volt;
    c.lineWidth = 2;
    // gate shutter lines
    if (openFrac < 0.99) {
      const sh = (1 - openFrac) * (gateW / 2 - 2);
      c.fillRect(GEOM.ch0 + 2, GEOM.mid - GEOM.membH / 2, sh, GEOM.membH);
      c.fillRect(GEOM.ch1 - 2 - sh, GEOM.mid - GEOM.membH / 2, sh, GEOM.membH);
    }

    // Compartment Labels & Concentrations
    c.fillStyle = co.ink;
    c.font = 'bold 12px var(--font-sans)';
    c.fillText('OUTSIDE (extracellular) — [out] = ' + st.outside.toFixed(0) + ' mM ' + st.ion, GEOM.x0 + 16, GEOM.y0 + 22);
    c.fillText('INSIDE (cytoplasm) — [in] = ' + st.inside.toFixed(0) + ' mM ' + st.ion, GEOM.x0 + 16, GEOM.mid + GEOM.membH / 2 + 22);

    // Draw fixed anions in cytoplasm
    c.fillStyle = co.inh;
    c.font = '12px var(--font-mono)';
    for (const an of anions) {
      c.fillText('−', an.x, an.y);
    }

    // Draw particles (ions)
    const ionColor = st.ion === 'Na' ? co.exc : (st.ion === 'Cl' ? co.inh : co.volt);
    c.fillStyle = ionColor;
    for (const q of sim.particles) {
      c.beginPath();
      c.arc(q.x, q.y, 4.5, 0, Math.PI * 2);
      c.fill();
      c.strokeStyle = co.paper;
      c.lineWidth = 1;
      c.stroke();
    }

    // Draw crossing flashes
    for (const f of sim.flashes) {
      c.strokeStyle = f.outward ? co.exc : co.volt;
      c.globalAlpha = Math.max(0, f.alpha);
      c.beginPath();
      c.arc(f.x, f.y, f.r, 0, Math.PI * 2);
      c.stroke();
      c.globalAlpha = 1.0;
    }

    // Voltmeter badge
    c.save();
    c.translate(GEOM.meterX, GEOM.meterY);
    c.fillStyle = co.paperCard;
    c.strokeStyle = co.volt;
    c.lineWidth = 2;
    c.beginPath(); c.arc(0, 0, GEOM.meterR, 0, Math.PI * 2); c.fill(); c.stroke();
    c.fillStyle = co.ink;
    c.font = 'bold 11px var(--font-mono)';
    c.textAlign = 'center';
    c.fillText('V_m', 0, -10);
    c.fillStyle = co.volt;
    c.font = 'bold 12px var(--font-mono)';
    c.fillText(st.V_mV.toFixed(1) + 'mV', 0, 8);
    c.restore();

    c.restore();
  }

  if (isNode) return;

  if (typeof Atlas !== 'undefined') {
    Atlas.Plate.register({
      id: 'M03',
      title: { en: S.title, fa: 'معادله نرنست: چرا غشاء ولتاژ دارد' },
      source: 'live',
      mount
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
