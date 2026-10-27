/**
 * Plate M03 — The Nernst equation: why a membrane has a voltage.
 * Two compartments separated by a membrane with one selective channel.
 * Ions random-walk (Brownian) and cross the pore with odds derived from the
 * electrochemical potential difference; the escaping charge separates across
 * the bilayer until electric drift cancels diffusion at
 * E = (RT/zF) · ln([out]/[in]). The voltage bookkeeping is the continuum leak
 * ODE  C dV/dt = −g (V − E)  with E the Nernst potential of the live mM values;
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
      if (z === 0) throw new Error('z must be nonzero, got ' + z);
      return core.rtOverF_mV(tempC) / z * Math.log(cOut / cIn);
    },
    // RT/F for z = 1, in mV, at tempC °C.
    rtOverF_mV: function (tempC) {
      if (!isFinite(tempC) || tempC < -273.15) {
        throw new Error('temperature must be a finite °C above −273.15, got ' + tempC);
      }
      return 1000 * R_GAS * (273.15 + tempC) / F_FARADAY;
    },
    // Voltage shift per tenfold concentration ratio, mV, per unit |z|.
    perDecade_mV: function (tempC, zAbs) {
      if (!(zAbs > 0)) throw new Error('|z| must be positive, got ' + zAbs);
      return core.rtOverF_mV(tempC) * Math.log(10) / zAbs;
    },
    // One exact implicit step of C·dV/dt = −g·(V−E): dV = (E−V)(1 − e^(−dt/τ)).
    // With C in µF and g in nS the membrane time constant is τ = (C/g)·1000 ms.
    // Returns {dV}; never mutates inputs.
    odeStep: function (V_mV, E_mV, g_nS, C_uF, dt_ms) {
      if (!(C_uF > 0)) throw new Error('capacitance must be positive, got ' + C_uF);
      if (!(g_nS >= 0)) throw new Error('conductance must be non-negative, got ' + g_nS);
      if (!(dt_ms > 0)) throw new Error('time step must be positive, got ' + dt_ms);
      const tau = (C_uF / g_nS) * 1000; // ms
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
    // exactly when V = E. Pure.
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
      const tk = Math.sqrt((273.15 + (p.tempC || 37)) / 310.15);
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
        q.x += (sim.rand() - 0.5) * 2 * sigma;
        if (q.inside) q.y = Math.max(g.mid + g.membH / 2 + 2, Math.min(inBot, q.y));
        else q.y = Math.max(g.y0 + 20, Math.min(outBot, q.y));
        q.x = Math.max(g.x0 + 10, Math.min(g.x1 - 10, q.x));
      }
      // the ODE's own demand of crossings this tick: the shut gate carries
      // no current, so the book is also silent when open is false
      const fd = open
        ? core.fluxDemand(V, p.E_mV || 0, p.g_nS || 0, p.C_uF || 1, dt, p.z || 1)
        : { dV: 0, base: 0, outward: false };
      let n = Math.min(6, Math.floor(fd.base));
      if (sim.rand() < fd.base - Math.floor(fd.base)) n++;
      let outFlip = 0, inFlip = 0;
      const sourceInside = fd.outward; // outward always exits the inside pool
      for (let k = 0; k < n; k++) {
        // prefer the marker nearest an open pore-mouth on the source side
        let best = -1, bestD = Infinity;
        for (let i = 0; i < sim.particles.length; i++) {
          const qq = sim.particles[i];
          if (qq.inside !== sourceInside) continue;
          if (qq.x < g.ch0 || qq.x > g.ch1) continue;
          const d = Math.abs(qq.y - g.mid);
          if (d < bestD) { bestD = d; best = i; }
        }
        if (best < 0) {
          // fall back to any source-side marker; the picture is a picture
          for (let i = 0; i < sim.particles.length; i++) {
            if (sim.particles[i].inside === sourceInside) { best = i; break; }
          }
        }
        if (best < 0) continue;
        const qq = sim.particles[best];
        qq.inside = !qq.inside;
        qq.x = g.ch0 + 8 + sim.rand() * (g.ch1 - g.ch0 - 16);
        // land just past the membrane on the destination side
        qq.y = g.mid + (qq.inside ? 1 : -1) * (g.membH / 2 + 6 + sim.rand() * 6);
        if (fd.outward) outFlip++; else inFlip++;
        sim.flashes.push({ x: qq.x, y: qq.y, age: 0 });
      }
      core.ageFlashes(sim, dt);
      return { outN: outFlip, inN: inFlip, dV: fd.dV };
    },
    ageFlashes: function (sim, dtMs) {
      for (const f of sim.flashes) f.age += dtMs;
      while (sim.flashes.length && sim.flashes[0].age > 480) sim.flashes.shift();
    },
    // Nudge the marker pool toward the fraction the reported mM imply when the
    // student drags concentration bars: at most one swap per call, seeded.
    rebalance: function (sim, cIn, cOut) {
      const wantIn = Math.round(sim.N * cIn / (cIn + cOut));
      let haveIn = 0;
      for (const q of sim.particles) if (q.inside) haveIn++;
      const d = wantIn - haveIn;
      if (d === 0) return false;
      if (Math.abs(d) > sim.N * 0.04 && sim.rand() < 0.5) {
        let k = Math.floor(sim.rand() * sim.N);
        for (let tries = 0; tries < 8; tries++) {
          if (sim.particles[k].inside !== d > 0) break;
          k = (k + 1) % sim.N;
        }
        sim.particles[k].inside = d > 0;
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
    styleEl.textContent = CSS_TEXT;
    fig.appendChild(styleEl);

    // -- state ---------------------------------------------------------------
    const ionData = {
      K:  { name: 'K⁺',  in: 140, out: 5,   z: 1,  color: 'inh' },
      Na: { name: 'Na⁺', in: 15,  out: 145, z: 1,  color: 'exc' },
      Cl: { name: 'Cl⁻', in: 10,  out: 150, z: -1, color: 'trace' }
    };
    const st = {
      playing: !reducedMotion(),
      chapter: 0,
      ion: 'K',
      inside: 140,
      outside: 5,
      tempC: 37,
      g_nS: 10,
      C_uF: 1,
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
          '<button class="m03-btn m03-ion" data-ion="' + k + '" aria-pressed="false" ' +
          'title="selective channel for ' + ionData[k].name + '">' + ionData[k].name + '</button>').join('') +
        '</div>' +
        '<div class="m03-row"><span class="m03-lab">temp</span>' +
        '<input type="range" id="m03-temp" min="5" max="40" step="1" value="37" aria-label="temperature in degrees celsius">' +
        '<span class="m03-val" id="m03-temp-v">37 °C</span></div>' +
        '<div class="m03-row"><span class="m03-lab">conductance</span>' +
        '<input type="range" id="m03-g" min="0" max="60" step="1" value="10" aria-label="channel conductance in nanosiemens">' +
        '<span class="m03-val" id="m03-g-v">10 nS</span></div>' +
        '<div class="m03-row m03-btns">' +
        '<button class="m03-btn" id="m03-play" aria-label="play or pause">❚❚ pause</button>' +
        '<button class="m03-btn" id="m03-freeze" aria-label="freeze diffusion">freeze diffusion (F)</button>' +
        '<button class="m03-btn" id="m03-reset" aria-label="reset">reset</button>' +
        '</div>' +
        '<div class="m03-note">Drag the thin bars near OUTSIDE / INSIDE to add or remove ions · click the pore to open or shut the gate · click inside a pool to add ions.</div>' +
        '</div>';
    }

    function wireControls() {
      marginEl.querySelectorAll('.m03-ion').forEach(b => {
        b.addEventListener('click', () => switchIon(b.dataset.ion));
      });
      const temp = marginEl.querySelector('#m03-temp');
      temp.addEventListener('input', () => {
        st.tempC = Number(temp.value);
        marginEl.querySelector('#m03-temp-v').textContent = st.tempC + ' °C';
      });
      const gsl = marginEl.querySelector('#m03-g');
      gsl.addEventListener('input', () => {
        st.g_nS = Number(gsl.value);
        marginEl.querySelector('#m03-g-v').textContent = st.g_nS + ' nS';
        if (st.g_nS > 0 && S.chapters[st.chapter].open) st.channelOpen = true;
      });
      marginEl.querySelector('#m03-play').addEventListener('click', () => {
        st.playing = !st.playing; syncPlay();
      });
      marginEl.querySelector('#m03-freeze').addEventListener('click', () => {
        st.frozen = !st.frozen;
        marginEl.querySelector('#m03-freeze').textContent =
          st.frozen ? 'unfreeze diffusion (F)' : 'freeze diffusion (F)';
      });
      marginEl.querySelector('#m03-reset').addEventListener('click', reset);
    }

    function syncPlay() {
      const b = marginEl && marginEl.querySelector('#m03-play');
      if (b) b.textContent = st.playing ? '❚❚ pause' : '▶ play';
    }

    function switchIon(k) {
      st.ion = k;
      const d = ionData[k];
      st.inside = d.in;
      st.outside = d.out;
      st.V_mV = 0;
      st.t_ms = 0;
      st.trace = [];
      reseedParticles();
      marginEl.querySelectorAll('.m03-ion').forEach(b => {
        const on = b.dataset.ion === k;
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    }

    function reset() {
      st.predictChoice = null;
      st.revealed = false;
      st.verify = null;
      const rev = chapterBar.querySelector('.m03-reveal');
      if (rev) rev.remove();
      applyChapter(st.chapter);
      updatePredict();
    }

    // -- chapters ---------------------------------------------------------------
    const chapterBar = document.createElement('div');
    chapterBar.className = 'm03-chapters';
    chapterBar.setAttribute('role', 'group');
    chapterBar.setAttribute('aria-label', 'chapters');
    chapterBar.innerHTML =
      '<div class="m03-ch-top"><span class="m03-ch-code" id="m03-ch-code"></span>' +
      '<span class="m03-ch-hint" id="m03-ch-hint"></span></div>' +
      '<div class="m03-ch-row">' +
      '<button class="m03-btn" id="m03-prev" aria-label="previous chapter">← back</button>' +
      '<div class="m03-ch-cap" id="m03-ch-cap"></div>' +
      '<button class="m03-btn" id="m03-next" aria-label="next chapter">next →</button>' +
      '</div>';
    fig.parentElement.insertBefore(chapterBar, fig);
    chapterBar.querySelector('#m03-prev').addEventListener('click', () => applyChapter(st.chapter - 1));
    chapterBar.querySelector('#m03-next').addEventListener('click', () => applyChapter(st.chapter + 1));

    function applyChapter(i) {
      st.chapter = Math.max(0, Math.min(S.chapters.length - 1, i));
      const ch = S.chapters[st.chapter];
      // the story always restarts from its canonical numbers; play continues freely
      st.ion = ch.ion;
      const d = ionData[st.ion];
      st.inside = d.in;
      st.outside = d.out;
      st.V_mV = 0;
      st.t_ms = 0;
      st.trace = [];
      st.channelOpen = ch.open && st.g_nS > 0;
      st.frozen = false;
      if (marginEl) marginEl.querySelector('#m03-freeze').textContent = 'freeze diffusion (F)';
      reseedParticles();
      syncIonButtons();
      syncPlay();
      renderChapterBar();
      updatePredict();
    }

    function syncIonButtons() {
      if (!marginEl) return;
      marginEl.querySelectorAll('.m03-ion').forEach(b => {
        b.setAttribute('aria-pressed', b.dataset.ion === st.ion ? 'true' : 'false');
      });
    }

    function renderChapterBar() {
      const ch = S.chapters[st.chapter];
      chapterBar.querySelector('#m03-ch-code').textContent =
        'chapter ' + (st.chapter + 1) + '/' + S.chapters.length;
      chapterBar.querySelector('#m03-ch-cap').textContent = ch.caption;
      chapterBar.querySelector('#m03-ch-hint').textContent = ch.watch;
    }

    // -- predict-then-reveal -----------------------------------------------------
    const predictBox = document.createElement('div');
    predictBox.className = 'm03-predict';
    predictBox.innerHTML =
      '<div class="m03-q">PREDICT — raise the outside ' + ionData.K.name +
      ' by ten times. Which way does the equilibrium voltage move, and how far per decade at ' +
      st.tempC + ' °C?</div>' +
      '<div class="m03-ch-row" id="m03-dir-row"><span class="m03-lab">direction</span>' +
      '<button class="m03-btn m03-dir" data-dir="up" aria-label="it goes up">⬆ goes up</button>' +
      '<button class="m03-btn m03-dir" data-dir="down" aria-label="it goes down">⬇ goes down</button>' +
      '</div>' +
      '<div class="m03-ch-row" id="m03-mag-row"><span class="m03-lab">per 10×</span></div>' +
      '<div class="m03-ch-row"><button class="m03-btn" id="m03-commit" disabled ' +
      'aria-label="commit prediction">commit</button>' +
      '<span class="m03-pr-msg" id="m03-pr-msg"></span></div>';
    predictBox.style.display = 'none';
    chapterBar.appendChild(predictBox);

    function magnitudeButtons() {
      const row = predictBox.querySelector('#m03-mag-row');
      while (row.childNodes.length > 1) row.removeChild(row.lastChild);
      const truth = core.perDecade_mV(st.tempC, 1);
      [truth / 2, truth, truth * 2].sort(() => sim.rand() - 0.5).forEach(v => {
        const b = document.createElement('button');
        b.className = 'm03-btn m03-mag';
        b.dataset.mag = v;
        b.textContent = v.toFixed(1) + ' mV';
        b.setAttribute('aria-label', 'magnitude ' + v.toFixed(1) + ' millivolt per decade');
        b.addEventListener('click', () => {
          predictBox.querySelectorAll('.m03-mag').forEach(x =>
            x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
          st.predictSel.mag = v;
          syncCommit();
        });
        row.appendChild(b);
      });
    }

    function syncCommit() {
      const ok = st.predictSel && st.predictSel.dir && st.predictSel.mag !== null;
      const b = marginEl && predictBox.querySelector('#m03-commit');
      if (b) b.disabled = !ok;
      const msg = predictBox.querySelector('#m03-pr-msg');
      if (msg) msg.textContent = st.predictSel && st.predictSel.dir && st.predictSel.mag !== null
        ? 'committed — drag the OUT bar after the reveal to see the dashed line move.'
        : 'pick a direction and a magnitude, then commit.';
    }

    predictBox.querySelectorAll('.m03-dir').forEach(b => {
      b.addEventListener('click', () => {
        predictBox.querySelectorAll('.m03-dir').forEach(x =>
          x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
        st.predictSel.dir = b.dataset.dir;
        syncCommit();
      });
    });

    function updatePredict() {
      const ch = S.chapters[st.chapter];
      if (ch.predict && !st.revealed) {
        st.predictSel = { dir: null, mag: null };
        predictBox.style.display = 'block';
        magnitudeButtons();
        syncCommit();
      } else {
        predictBox.style.display = 'none';
      }
    }

    function commitPredict() {
      const sel = st.predictSel;
      if (st.revealed || !sel || !sel.dir || sel.mag === null) return;
      st.predictChoice = { dir: sel.dir, mag: sel.mag };
      st.revealed = true;
      predictBox.style.display = 'none';
      const slope = core.perDecade_mV(st.tempC, 1);
      const dirTrue = slope > 0 ? 'up' : 'down';
      const magOk = Math.abs(sel.mag - slope) < 0.75;
      const dirOk = sel.dir === dirTrue;
      const good = dirOk && magOk;
      const rev = document.createElement('div');
      rev.className = 'm03-reveal ' + (good ? '' : 'm03-bad');
      rev.setAttribute('role', 'status');
      rev.innerHTML =
        (good ? '✔ correct. ' :
          (dirOk ? '✔ direction right, magnitude off — ' : '✖ not quite — ')) +
        'E = (RT/zF)·ln([out]/[in]): a tenfold [out] adds ln 10 to the log, so E moves <b>' +
        slope.toFixed(1) + ' mV ' + dirTrue + '</b> per decade for a z = 1 ion at ' +
        st.tempC + ' °C — now watch the dashed line. ';
      const demo = document.createElement('button');
      demo.className = 'm03-btn';
      demo.textContent = 'verify: OUT ×10';
      demo.setAttribute('aria-label', 'multiply outside concentration by ten');
      demo.addEventListener('click', () => {
        const out0 = st.outside;
        st.outside = Math.min(300, st.outside * 10);
        st.verify = { out0: out0, E0: core.nernst(out0, st.inside, ionData[st.ion].z, st.tempC) };
        demo.disabled = true;
        st.playing = true;
        syncPlay();
      });
      rev.appendChild(demo);
      chapterBar.insertBefore(rev, predictBox);
    }

    predictBox.querySelector('#m03-commit').addEventListener('click', commitPredict);

    // -- physics -----------------------------------------------------------------
    function physicsStep(dtMs) {
      if (!st.playing) return;
      const E = E_of();
      const g = st.channelOpen && !st.frozen ? st.g_nS : 0;
      let dV = 0;
      if (g > 0) {
        const sub = 4, h = dtMs / sub;
        for (let k = 0; k < sub; k++) dV += core.odeStep(st.V_mV, E, g, st.C_uF, h).dV;
      }
      st.V_mV += dV;
      st.t_ms += dtMs;
      // slow concentration bookkeeping: an escaping + charge drains a cation
      // from [in] into [out]. The mM scale is ~10^11 times coarser than the
      // membrane's charge, so the drift is a crawl — one leak relaxation must
      // barely move it.
      if (g > 0 && dV !== 0) {
        const shift = 0.0005 * dtMs * (g / 10) * -Math.sign(dV) * Math.sign(ionData[st.ion].z);
        st.inside = Math.max(0.5, Math.min(300, st.inside - shift));
        st.outside = Math.max(0.5, Math.min(300, st.outside + shift));
      }
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

    // -- drawing -----------------------------------------------------------------
    function draw() {
      const g2 = cvs.getContext('2d');
      const c = colors();
      const dpr = (typeof devicePixelRatio !== 'undefined' ? devicePixelRatio : 1) || 1;
      const pw = Math.round(GEOM.W * dpr), ph = Math.round(GEOM.H * dpr);
      if (cvs.width !== pw || cvs.height !== ph) { cvs.width = pw; cvs.height = ph; }
      g2.setTransform(dpr, 0, 0, dpr, 0, 0);
      g2.fillStyle = c.paper;
      g2.fillRect(0, 0, GEOM.W, GEOM.H);
      drawTrace(g2, c);
      drawCompartments(g2, c);
      drawAnions(g2, c);
      drawChannel(g2, c);
      drawParticles(g2, c);
      drawElectricArrows(g2, c);
      drawBalanceArrows(g2, c);
      drawDragBars(g2, c);
      drawVoltmeter(g2, c);
      drawReadouts(g2, c);
    }

    // adaptive voltage scale: never clips the theory line for extreme drags
    function vRange() {
      let m = 60;
      const E = E_of();
      m = Math.max(m, Math.abs(E) * 1.2 + 10);
      for (const p of st.trace) m = Math.max(m, Math.abs(p.V) * 1.15, Math.abs(p.E) * 1.15);
      return Math.ceil(m / 10) * 10 + 10;
    }

    function VtoY(V, yMax) {
      const y0 = GEOM.traceY + 26, y1 = GEOM.traceY + GEOM.traceH - 18;
      return y1 - (Math.max(-yMax, Math.min(yMax, V)) + yMax) / (2 * yMax) * (y1 - y0);
    }

    function drawTrace(g2, c) {
      const yMax = vRange();
      g2.strokeStyle = c.border; g2.lineWidth = 1;
      g2.strokeRect(GEOM.traceX, GEOM.traceY, GEOM.traceW, GEOM.traceH);
      g2.fillStyle = c.inkMuted;
      g2.font = '12px ui-monospace, monospace';
      g2.textAlign = 'left';
      g2.fillText('membrane voltage V(t), mV — window ' + (Math.min(12000, Math.max(1, st.t_ms)) / 1000).toFixed(1) + ' s', GEOM.traceX + 8, GEOM.traceY + 15);
      g2.textAlign = 'right';
      g2.fillText('+' + yMax, GEOM.traceX + GEOM.traceW - 6, GEOM.traceY + 34);
      g2.fillText('−' + yMax, GEOM.traceX + GEOM.traceW - 6, GEOM.traceY + GEOM.traceH - 20);
      g2.textAlign = 'left';

      // window grows with the run until the full 12 s is showing, then scrolls
      const winMs = Math.max(1, Math.min(12000, st.t_ms));
      const X = t => GEOM.traceX + 8 +
        Math.max(0, Math.min(1, (t - Math.max(0, st.t_ms - winMs)) / winMs)) * (GEOM.traceW - 16);

      // zero line
      g2.strokeStyle = c.border; g2.setLineDash([2, 4]);
      g2.beginPath(); g2.moveTo(GEOM.traceX + 8, VtoY(0, yMax)); g2.lineTo(GEOM.traceX + GEOM.traceW - 8, VtoY(0, yMax)); g2.stroke();
      g2.setLineDash([]);

      const E = E_of();
      // theory ghost
      if (st.showTheory) {
        g2.save();
        g2.setLineDash([6, 4]); g2.strokeStyle = c.theory; g2.lineWidth = 1.5;
        const hy = VtoY(E, yMax);
        g2.beginPath(); g2.moveTo(GEOM.traceX + 8, hy); g2.lineTo(GEOM.traceX + GEOM.traceW - 8, hy); g2.stroke();
        g2.setLineDash([]);
        g2.fillStyle = c.theory;
        g2.fillText('theory  E = ' + E.toFixed(1) + ' mV', GEOM.traceX + 10, hy - 5);
        g2.restore();
      }

      // sim trace
      g2.strokeStyle = c.volt; g2.lineWidth = 1.8;
      g2.beginPath();
      let started = false;
      for (const p of st.trace) {
        const x = X(p.t), y = VtoY(p.V, yMax);
        if (!started) { g2.moveTo(x, y); started = true; } else g2.lineTo(x, y);
      }
      g2.stroke();

      // honest difference readout
      if (st.trace.length > 3 && st.showTheory) {
        const a = st.trace[st.trace.length - 1];
        const diff = a.V - a.E;
        g2.fillStyle = Math.abs(diff) < 1 ? c.inkMuted : c.volt;
        g2.fillText('Δ sim − theory = ' + diff.toFixed(2) + ' mV', GEOM.traceX + 8, GEOM.traceY + GEOM.traceH - 6);
      }

      // per-decade verification chip
      if (st.verify && st.showTheory) {
        const E1 = E_of();
        const dE = E1 - st.verify.E0;
        const ratio = st.outside / st.verify.out0;
        if (ratio >= 9.5) {
          g2.fillStyle = c.exc;
          g2.fillRect(GEOM.traceX + GEOM.traceW - 320, GEOM.traceY + 22, 312, 20);
          g2.fillStyle = c.paper;
          g2.textAlign = 'right';
          g2.fillText('E moved ' + (dE >= 0 ? '+' : '−') + Math.abs(dE).toFixed(1) +
            ' mV for ×' + ratio.toFixed(1) + ' in [out]', GEOM.traceX + GEOM.traceW - 14, GEOM.traceY + 36);
          g2.textAlign = 'left';
        }
      }
    }

    function drawCompartments(g2, c) {
      const hover = st.hoverTerm;
      const midT = GEOM.mid - GEOM.membH / 2, midB = GEOM.mid + GEOM.membH / 2;
      // pools
      g2.fillStyle = c.paperCard;
      g2.fillRect(GEOM.x0, GEOM.y0, GEOM.x1 - GEOM.x0, midT - GEOM.y0);
      g2.fillRect(GEOM.x0, midB, GEOM.x1 - GEOM.x0, GEOM.y1 - midB);
      // membranes
      g2.fillStyle = c.border;
      g2.fillRect(GEOM.x0, midT, GEOM.x1 - GEOM.x0, GEOM.membH);
      g2.strokeStyle = c.inkMuted; g2.lineWidth = 1;
      g2.strokeRect(GEOM.x0, GEOM.y0, GEOM.x1 - GEOM.x0, midT - GEOM.y0);
      g2.strokeRect(GEOM.x0, midB, GEOM.x1 - GEOM.x0, GEOM.y1 - midB);
      g2.strokeRect(GEOM.x0, midT, GEOM.x1 - GEOM.x0, GEOM.membH);
      // bilayer ticks
      g2.strokeStyle = c.paper; g2.lineWidth = 1;
      for (let x = GEOM.x0 + 10; x < GEOM.x1 - 8; x += 26) {
        g2.beginPath(); g2.moveTo(x, midT + 4); g2.lineTo(x + 12, midT + 4); g2.stroke();
        g2.beginPath(); g2.moveTo(x + 6, midB - 4); g2.lineTo(x + 18, midB - 4); g2.stroke();
      }

      // labels + concentrations
      g2.font = '13px ui-serif, Georgia, serif';
      g2.fillStyle = c.ink;
      g2.textAlign = 'left';
      const labOut = 'OUTSIDE — ' + ionData[st.ion].name + ' ' + st.outside.toFixed(1) + ' mM';
      g2.fillText(labOut, GEOM.x0 + 14, GEOM.y0 + 22);
      g2.font = '12px ui-monospace, monospace';
      g2.fillStyle = c.inkMuted;
      g2.fillText('ratio [out]/[in] = ' + (st.outside / st.inside).toFixed(3), GEOM.x0 + 14, GEOM.y0 + 40);
      g2.font = '13px ui-serif, Georgia, serif';
      g2.fillStyle = c.ink;
      g2.fillText('INSIDE — ' + ionData[st.ion].name + ' ' + st.inside.toFixed(1) +
        ' mM  +  fixed anions (−)', GEOM.x0 + 14, GEOM.y1 - 12);

      // hover highlights per term (never colour alone: dashed outline + label)
      if (hover === 'ratio') {
        g2.save();
        g2.setLineDash([5, 3]); g2.strokeStyle = c.volt; g2.lineWidth = 1.5;
        g2.strokeRect(GEOM.x0 + 2, GEOM.y0 + 2, GEOM.x1 - GEOM.x0 - 4, midT - GEOM.y0 - 4);
        g2.strokeRect(GEOM.x0 + 2, midB + 2, GEOM.x1 - GEOM.x0 - 4, GEOM.y1 - midB - 4);
        g2.setLineDash([]);
        g2.fillStyle = c.volt; g2.font = '12px ui-monospace, monospace';
        g2.fillText('ratio term acts here [out]/[in]', GEOM.x1 - 240, GEOM.y0 + 40);
        g2.restore();
      } else if (hover === 'RToverF') {
        g2.save();
        g2.setLineDash([5, 3]); g2.strokeStyle = c.mod; g2.lineWidth = 1.5;
        g2.strokeRect(GEOM.x0 + 2, GEOM.y0 + 2, GEOM.x1 - GEOM.x0 - 4, midT - GEOM.y0 - 4);
        g2.strokeRect(GEOM.x0 + 2, midB + 2, GEOM.x1 - GEOM.x0 - 4, GEOM.y1 - midB - 4);
        g2.setLineDash([]);
        g2.fillStyle = c.mod; g2.font = '12px ui-monospace, monospace';
        g2.fillText('RT/zF: jitter speed ∝ √T at ' + st.tempC + ' °C', GEOM.x1 - 300, GEOM.y0 + 40);
        g2.restore();
      }
    }

    function drawAnions(g2, c) {
      g2.fillStyle = c.inkMuted;
      for (const a of anions) {
        g2.beginPath(); g2.arc(a.x, a.y, 7, 0, Math.PI * 2); g2.fill();
      }
      g2.strokeStyle = c.paper; g2.lineWidth = 1.4;
      for (const a of anions) {
        g2.beginPath(); g2.moveTo(a.x - 3.5, a.y); g2.lineTo(a.x + 3.5, a.y); g2.stroke();
      }
    }

    function drawChannel(g2, c) {
      const open = st.gateAnim > 0.05;
      const poreH = GEOM.membH + 8 * st.gateAnim;
      g2.fillStyle = c.paper;
      g2.fillRect(GEOM.ch0, GEOM.mid - poreH / 2, GEOM.ch1 - GEOM.ch0, poreH);
      g2.strokeStyle = open ? c.volt : c.inkMuted;
      g2.lineWidth = open ? 1.8 : 1.2;
      g2.strokeRect(GEOM.ch0, GEOM.mid - poreH / 2, GEOM.ch1 - GEOM.ch0, poreH);
      // gate leaves (vanish when open), never colour-alone: shape shows state
      if (!open) {
        g2.fillStyle = c.inkMuted;
        const w = (GEOM.ch1 - GEOM.ch0) / 2 * (1 - st.gateAnim);
        g2.fillRect(GEOM.ch0, GEOM.mid - poreH / 2, w, poreH);
        g2.fillRect(GEOM.ch1 - w, GEOM.mid - poreH / 2, w, poreH);
      }
      if (st.hoverTerm === 'g') {
        g2.save();
        g2.setLineDash([4, 3]); g2.strokeStyle = c.volt; g2.lineWidth = 1.6;
        g2.strokeRect(GEOM.ch0 - 5, GEOM.mid - poreH / 2 - 5, GEOM.ch1 - GEOM.ch0 + 10, poreH + 10);
        g2.restore();
      }
      g2.fillStyle = c.ink;
      g2.font = '12px ui-monospace, monospace';
      g2.textAlign = 'center';
      const cx = (GEOM.ch0 + GEOM.ch1) / 2;
      g2.fillText(ionData[st.ion].name + ' channel · ' + (open ? 'open' : 'shut'),
        cx, GEOM.mid - poreH / 2 - 20);
      g2.fillStyle = c.inkMuted;
      g2.font = '10px ui-monospace, monospace';
      g2.fillText('g = ' + st.g_nS + ' nS — click to toggle', cx, GEOM.mid - poreH / 2 - 7);
      g2.textAlign = 'left';
    }

    function drawParticles(g2, c) {
      const col = c[ionData[st.ion].color];
      const anion = ionData[st.ion].z < 0;
      g2.fillStyle = col;
      for (const q of sim.particles) {
        g2.globalAlpha = q.inside ? 0.85 : 0.6;
        g2.beginPath(); g2.arc(q.x, q.y, 2.6, 0, Math.PI * 2); g2.fill();
        // sign glyph, never colour-alone
        g2.strokeStyle = c.paper; g2.lineWidth = 1;
        g2.globalAlpha = 0.9;
        if (anion) {
          g2.beginPath(); g2.moveTo(q.x - 1.4, q.y); g2.lineTo(q.x + 1.4, q.y); g2.stroke();
        } else {
          g2.beginPath(); g2.moveTo(q.x - 1.4, q.y); g2.lineTo(q.x + 1.4, q.y); g2.stroke();
          g2.beginPath(); g2.moveTo(q.x, q.y - 1.4); g2.lineTo(q.x + 0, q.y + 1.4); g2.stroke();
        }
      }
      g2.globalAlpha = 1;
      // flashes on crossings
      for (const f of sim.flashes) {
        const k = 1 - f.age / 480;
        if (k <= 0) continue;
        g2.globalAlpha = 0.55 * k;
        g2.strokeStyle = c.volt; g2.lineWidth = 1.4;
        g2.beginPath(); g2.arc(f.x, f.y, 3 + 9 * (1 - k), 0, Math.PI * 2); g2.stroke();
      }
      g2.globalAlpha = 1;
    }

    // electric force across the bilayer on the selected ion (force, not field:
    // for anions it flips, and the label says so)
    function drawElectricArrows(g2, c) {
      const open = st.channelOpen;
      if (!open) return;
      const hover = st.hoverTerm === 'drive' || st.hoverTerm === 'nernst';
      const strength = Math.min(1, Math.abs(st.V_mV) / 90);
      if (strength < 0.02 && !hover && !st.frozen) return;
      // inside potential V < 0 pulls +ions into INSIDE (+y), pushes anions out
      const dirDown = (ionData[st.ion].z > 0) === (st.V_mV < 0) ? +1 : -1;
      const skipA = GEOM.ch0 - 58, skipB = GEOM.ch1 + 58;
      const skipM = GEOM.meterX - GEOM.meterR - 18, skipM2 = GEOM.meterX + GEOM.meterR + 18;
      const n = 16;
      for (let i = 0; i < n; i++) {
        const x = GEOM.x0 + 30 + (GEOM.x1 - GEOM.x0 - 60) * (i + 0.5) / n;
        if ((x > skipA && x < skipB) || (x > skipM && x < skipM2)) continue;
        const len = 9 + 14 * strength;
        const yA = GEOM.mid - len * dirDown, yB = GEOM.mid + len * dirDown;
        g2.globalAlpha = hover ? 0.95 : 0.35 + 0.45 * strength;
        g2.strokeStyle = c.mod; g2.fillStyle = c.mod; g2.lineWidth = 1.5;
        g2.beginPath(); g2.moveTo(x, yA); g2.lineTo(x, yB); g2.stroke();
        g2.beginPath();
        g2.moveTo(x, yB);
        g2.lineTo(x - 3.2, yB - dirDown * 5.5);
        g2.lineTo(x + 3.2, yB - dirDown * 5.5);
        g2.closePath(); g2.fill();
        g2.globalAlpha = 1;
      }
      g2.fillStyle = c.mod;
      g2.font = '11.5px ui-monospace, monospace';
      const label = 'electric force on ' + ionData[st.ion].name +
        (ionData[st.ion].z < 0 ? ' (field is opposite)' : '');
      g2.textAlign = 'right';
      g2.globalAlpha = hover ? 1 : 0.8;
      g2.fillText(label, skipA - 10, GEOM.mid - 24);
      g2.globalAlpha = 1;
      g2.textAlign = 'left';
    }

    // the two opposing forces at the pore, lengths from the equations:
    // chemical push (RT·|ln(in/out)| measured in mV) vs electric drift (|V| mV)
    function drawBalanceArrows(g2, c) {
      if (!st.channelOpen) return;
      const RTF = core.rtOverF_mV(st.tempC);
      const chem = Math.min(100, RTF * Math.abs(Math.log(st.outside / st.inside))); // mV-equiv
      const elec = Math.min(100, Math.abs(st.V_mV)); // mV
      const scale = 0.52;
      const dirChem = st.inside > st.outside ? -1 : +1; // −1 = up = out of inside
      const dirElec = (ionData[st.ion].z > 0) === (st.V_mV < 0) ? +1 : -1; // +1 = down = into inside
      arrow(g2, c, GEOM.ch0 - 26, GEOM.mid, dirChem, chem * scale, c.exc, 'diffusion');
      arrow(g2, c, GEOM.ch1 + 26, GEOM.mid, dirElec, elec * scale, c.mod, 'drift');
    }
    function arrow(g2, c, x, yMid, dir, len, color, label) {
      if (len < 3) return;
      const y = yMid + dir * (GEOM.membH / 2 + 6);
      const tip = y + dir * len;
      g2.strokeStyle = color; g2.fillStyle = color; g2.lineWidth = 2.2;
      g2.beginPath(); g2.moveTo(x, y); g2.lineTo(x, tip - dir * 6); g2.stroke();
      g2.beginPath();
      g2.moveTo(x, tip);
      g2.lineTo(x - 4, tip - dir * 8);
      g2.lineTo(x + 4, tip - dir * 8);
      g2.closePath(); g2.fill();
      g2.font = '11px ui-monospace, monospace';
      g2.textAlign = 'center';
      g2.fillText(label, x, y + (dir < 0 ? -len - 6 : len + 14));
      g2.textAlign = 'left';
    }

    function drawDragBars(g2, c) {
      bar(g2, c, GEOM.x1 - 26, GEOM.y0 + 60, 'outside', st.outside);
      bar(g2, c, GEOM.x1 - 26, GEOM.y1 - 60, 'inside', st.inside);
    }
    function bar(g2, c, x, y, which, conc) {
      const w = 10, h = 30;
      const active = st.dragHandle === which;
      g2.fillStyle = c.paper;
      g2.strokeStyle = active ? c.volt : c.inkMuted;
      g2.lineWidth = active ? 2.2 : 1.2;
      g2.fillRect(x - w / 2, y - h / 2, w, h);
      g2.strokeRect(x - w / 2, y - h / 2, w, h);
      g2.beginPath(); g2.moveTo(x - 3, y - 7); g2.lineTo(x + 3, y - 7); g2.stroke();
      g2.beginPath(); g2.moveTo(x - 3, y + 7); g2.lineTo(x + 3, y + 7); g2.stroke();
      g2.fillStyle = c.inkMuted; g2.font = '12px ui-monospace, monospace';
      g2.textAlign = 'right';
      g2.fillText((which === 'outside' ? 'OUT ' : 'IN ') + conc.toFixed(1) + ' mM — drag ↕',
        x - 10, y + 4);
      g2.textAlign = 'left';
    }

    function drawVoltmeter(g2, c) {
      const vx = GEOM.meterX, vy = GEOM.meterY, r = GEOM.meterR;
      g2.fillStyle = c.paper;
      g2.beginPath(); g2.arc(vx, vy, r, 0, Math.PI * 2); g2.fill();
      g2.strokeStyle = c.inkMuted; g2.lineWidth = 1.2;
      g2.beginPath(); g2.arc(vx, vy, r, Math.PI * 0.75, Math.PI * 2.25); g2.stroke();
      for (let i = 0; i <= 12; i++) {
        const a = Math.PI * 0.75 + (i / 12) * Math.PI * 1.5;
        g2.beginPath();
        g2.moveTo(vx + Math.cos(a) * r, vy + Math.sin(a) * r);
        g2.lineTo(vx + Math.cos(a) * (r - 5), vy + Math.sin(a) * (r - 5));
        g2.stroke();
      }
      const E = E_of();
      if (st.showTheory) { // theory tick on the dial
        const aE = Math.PI * 0.75 + (Math.max(-120, Math.min(120, E)) + 120) / 240 * Math.PI * 1.5;
        g2.strokeStyle = c.theory; g2.lineWidth = 2;
        g2.setLineDash([3, 2]);
        g2.beginPath();
        g2.moveTo(vx + Math.cos(aE) * (r - 1), vy + Math.sin(aE) * (r - 1));
        g2.lineTo(vx + Math.cos(aE) * (r - 11), vy + Math.sin(aE) * (r - 11));
        g2.stroke();
        g2.setLineDash([]);
      }
      const cl = Math.max(-120, Math.min(120, st.V_mV)) / 120;
      const ang = Math.PI * 0.75 + (cl + 1) / 2 * Math.PI * 1.5;
      g2.strokeStyle = c.volt; g2.lineWidth = 2;
      g2.beginPath(); g2.moveTo(vx, vy);
      g2.lineTo(vx + Math.cos(ang) * (r - 8), vy + Math.sin(ang) * (r - 8));
      g2.stroke();
      g2.beginPath(); g2.arc(vx, vy, 2.4, 0, Math.PI * 2);
      g2.fillStyle = c.volt; g2.fill();
      // leads to each side of the bilayer
      g2.strokeStyle = c.inkMuted; g2.lineWidth = 1;
      g2.beginPath(); g2.moveTo(vx, vy - r); g2.lineTo(vx, GEOM.mid - GEOM.membH / 2 - 10); g2.stroke();
      g2.beginPath(); g2.moveTo(vx, vy + r); g2.lineTo(vx, GEOM.mid + GEOM.membH / 2 + 10); g2.stroke();
      g2.fillStyle = c.ink; g2.font = '12px ui-monospace, monospace';
      g2.textAlign = 'center';
      g2.fillText('V = ' + st.V_mV.toFixed(1) + ' mV', vx - 6, vy + r + 14);
      g2.textAlign = 'left';
    }

    function drawReadouts(g2, c) {
      g2.fillStyle = c.inkMuted; g2.font = '12px ui-monospace, monospace';
      g2.textAlign = 'right';
      g2.fillText('t = ' + (st.t_ms / 1000).toFixed(1) + ' s', GEOM.x1 - 8, GEOM.y1 - 2);
      if (st.frozen) {
        g2.textAlign = 'center';
        g2.fillStyle = c.mod;
        g2.fillText('diffusion frozen — the two forces at the pore are the balance picture',
          GEOM.W / 2, GEOM.y1 - 90);
      }
      g2.textAlign = 'left';
    }

    // -- margin live values -------------------------------------------------------
    function term(name, html) {
      const on = st.hoverTerm === name;
      return '<span class="m03-term' + (on ? ' m03-term-on' : '') + '" data-term="' + name + '">' + html + '</span>';
    }

    const glossOrder = ['diffusion', 'anion', 'drift', 'nernst', 'decade'];

    function renderMargin() {
      if (!marginEl) return;
      const RTF = core.rtOverF_mV(st.tempC);
      const ratio = st.outside / st.inside;
      const E = E_of();
      const eqN = marginEl.querySelector('#m03-eq-nernst');
      const eqO = marginEl.querySelector('#m03-eq-ode');
      const gl = marginEl.querySelector('#m03-gloss');
      const gEff = st.channelOpen && !st.frozen ? st.g_nS : 0;
      const df = st.V_mV - E;
      if (eqN) eqN.innerHTML =
        '<span class="m03-m">' + S.equations.nernst + '</span>' +
        term('RToverF', 'RT/zF = ' + RTF.toFixed(2) + ' mV') + ' · ' +
        term('ratio', 'ln([out]/[in]) = ' + Math.log(ratio).toFixed(3)) + ' ⇒ ' +
        term('nernst', '<b>E = ' + E.toFixed(2) + ' mV</b>');
      if (eqO) eqO.innerHTML =
        '<span class="m03-m">' + S.equations.ode + '</span>' +
        term('g', 'g = ' + gEff + ' nS') + ' · ' +
        term('drive', 'V − E = ' + df.toFixed(2) + ' mV') + ' ⇒ ' +
        'dV/dt = ' + (-(gEff / st.C_uF) * df).toFixed(3) + ' mV/ms';
      if (gl) {
        const hov = st.hoverTerm && S.terms[st.hoverTerm];
        gl.innerHTML = '<i>' + (hov || S.glossary[glossOrder[st.chapter % glossOrder.length]]) + '</i>';
      }
    }

    function onTermOver(e) {
      const t = e.target && e.target.closest ? e.target.closest('.m03-term') : null;
      if (t) st.hoverTerm = t.dataset.term;
    }
    function onTermOut(e) {
      const t = e.target && e.target.closest ? e.target.closest('.m03-term') : null;
      if (t && st.hoverTerm === t.dataset.term) st.hoverTerm = null;
    }

    // -- direct manipulation ------------------------------------------------------
    function canvasXY(e) {
      const r = cvs.getBoundingClientRect();
      return {
        x: (e.clientX - r.left) * GEOM.W / r.width,
        y: (e.clientY - r.top) * GEOM.H / r.height
      };
    }
    function near(x, y, bx, by, rw, rh) {
      return Math.abs(x - bx) < (rw || 22) && Math.abs(y - by) < (rh || 30);
    }

    cvs.addEventListener('pointerdown', e => {
      const p = canvasXY(e);
      if (near(p.x, p.y, GEOM.x1 - 26, GEOM.y0 + 60)) st.dragHandle = 'outside';
      else if (near(p.x, p.y, GEOM.x1 - 26, GEOM.y1 - 60)) st.dragHandle = 'inside';
      else if (p.x > GEOM.ch0 - 14 && p.x < GEOM.ch1 + 14 &&
               p.y > GEOM.mid - 34 && p.y < GEOM.mid + 34) {
        // click the pore: toggle the gate
        if (st.g_nS > 0) st.channelOpen = !st.channelOpen;
      } else if (p.y > GEOM.mid + GEOM.membH / 2 + 20 && p.y < GEOM.y1 - 20) {
        st.inside = Math.max(0.5, Math.min(300, st.inside + 4));
      } else if (p.y > GEOM.y0 + 20 && p.y < GEOM.mid - GEOM.membH / 2 - 10) {
        st.outside = Math.max(0.5, Math.min(300, st.outside + 4));
      }
      st.lastX = p.x;
      e.preventDefault();
    });
    cvs.addEventListener('pointermove', e => {
      if (!st.dragHandle) return;
      const p = canvasXY(e);
      const dx = p.x - st.lastX;
      st.lastX = p.x;
      const scale = Math.exp(dx * 0.02);
      const key = st.dragHandle === 'outside' ? 'outside' : 'inside';
      st[key] = Math.max(0.5, Math.min(300, st[key] * scale));
    });
    window.addEventListener('pointerup', () => { st.dragHandle = null; });

    // -- keyboard -----------------------------------------------------------------
    document.addEventListener('keydown', e => {
      if (e.target && e.target.tagName === 'INPUT') return;
      if (e.key === ' ') {
        e.preventDefault();
        st.playing = !st.playing; syncPlay();
      } else if (e.key === 't' || e.key === 'T') {
        st.showTheory = !st.showTheory;
      } else if (e.key === 'ArrowRight') {
        e.preventDefault(); applyChapter(st.chapter + 1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault(); applyChapter(st.chapter - 1);
      } else if (e.key === 'Home') {
        e.preventDefault(); applyChapter(0);
      } else if (e.key === 'End') {
        e.preventDefault(); applyChapter(S.chapters.length - 1);
      } else if (e.key === 'f' || e.key === 'F') {
        st.frozen = !st.frozen;
        if (marginEl) marginEl.querySelector('#m03-freeze').textContent =
          st.frozen ? 'unfreeze diffusion (F)' : 'freeze diffusion (F)';
      }
    });

    // -- frame loop ----------------------------------------------------------------
    let lastNow = -1;
    let lastFrameAt = -1e9;
    function frame() {
      const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      if (now - lastFrameAt < 11) return;
      const dt = lastNow < 0 ? 16 : Math.max(0, Math.min(48, now - lastNow));
      lastNow = now;
      lastFrameAt = now;
      physicsStep(dt);
      particlesStep(dt);
      renderMargin();
      draw();
    }

    onFrame(() => frame()); // shell clock pump (30 fps when its clock plays)
    if (typeof requestAnimationFrame === 'function') {
      const raf = () => { frame(); requestAnimationFrame(raf); };
      requestAnimationFrame(raf); // headless/hidden pumps may never tick
    }
    // timer fallback when rAF is throttled or virtual: still honest sim time
    if (typeof setInterval === 'function') setInterval(frame, 16);

    // -- boot ----------------------------------------------------------------------
    applyChapter(0);
    renderMargin();
    draw();
  }

  const CSS_TEXT = [
    '.m03-wrap{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink,#1b1a17);}',
    '.m03-eqlabel{font:11px system-ui;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-muted,#57534e);margin-bottom:4px;}',
    '.m03-eqlabel .m03-live{color:var(--volt,#b45309);}',
    '.m03-eq{margin:8px 0;padding:8px 10px;border:1px solid var(--border,#d6cfbf);border-radius:4px;line-height:1.7;background:var(--paper-card,#eeeae0);font-size:12.5px;}',
    '.m03-eq .m03-m{font-family:ui-monospace,monospace;color:var(--ink-muted,#57534e);display:block;font-size:11.5px;margin-bottom:2px;}',
    '.m03-term{padding:1px 3px;border-radius:3px;cursor:help;transition:background .15s;}',
    '.m03-term-on{background:var(--volt,#b45309);color:var(--paper,#f6f2e9);}',
    '.m03-explain{font:12px ui-serif,Georgia,serif;color:var(--ink-muted,#57534e);margin:10px 0 0;line-height:1.5;}',
    '.m03-gloss{font:italic 12.5px ui-serif,Georgia,serif;color:var(--ink,#1b1a17);border-left:2px solid var(--inh,#be123c);padding-left:9px;margin:10px 0;min-height:38px;}',
    '.m03-ctl{display:flex;flex-direction:column;gap:8px;font:13px system-ui;margin-top:10px;}',
    '.m03-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}',
    '.m03-lab{min-width:86px;font:11px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-muted,#57534e);}',
    '.m03-val{font:12px ui-monospace,monospace;color:var(--ink,#1b1a17);min-width:58px;}',
    '.m03-btn{font:12px system-ui;padding:4px 10px;border:1px solid var(--ink-muted,#57534e);border-radius:4px;background:var(--paper,#f6f2e9);color:var(--ink,#1b1a17);cursor:pointer;}',
    '.m03-btn:hover{background:var(--paper-card,#eeeae0);}',
    '.m03-btn[aria-pressed="true"]{background:var(--volt,#b45309);color:var(--paper,#f6f2e9);border-color:var(--volt,#b45309);}',
    '.m03-btn:disabled{opacity:.55;cursor:default;}',
    '.m03-btn:focus-visible{outline:2px solid var(--volt,#b45309);outline-offset:1px;}',
    '.m03-wrap input[type=range]{accent-color:var(--volt,#b45309);flex:1;min-width:90px;}',
    '.m03-btns{gap:8px;}',
    '.m03-note{font:italic 11.5px ui-serif,Georgia,serif;color:var(--ink-muted,#57534e);}',
    '.m03-footer{font:11.5px ui-monospace,monospace;color:var(--ink-muted,#57534e);margin:2px 4px 8px;line-height:1.5;}',
    '.m03-footer .m03-foot-src{color:var(--volt,#b45309);letter-spacing:.05em;text-transform:uppercase;font-size:10.5px;}',
    '.m03-footer code{font-family:ui-monospace,monospace;color:var(--ink,#1b1a17);}',
    // chapters
    '.m03-chapters{margin:0 4px 12px;border:1px solid var(--border,#d6cfbf);border-radius:6px;padding:10px 14px;background:var(--paper-card,#eeeae0);display:grid;gap:6px;}',
    '.m03-ch-top{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;}',
    '.m03-ch-code{font:11px ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--volt,#b45309);}',
    '.m03-ch-hint{font:12px ui-serif,Georgia,serif;color:var(--ink-muted,#57534e);font-style:italic;}',
    '.m03-ch-cap{font:14px ui-serif,Georgia,serif;color:var(--ink,#1b1a17);line-height:1.45;}',
    '.m03-ch-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}',
    // predict + reveal
    '.m03-predict{border:1px dashed var(--volt,#b45309);border-radius:6px;padding:10px 14px;margin-top:6px;background:var(--paper,#f6f2e9);}',
    '.m03-predict .m03-q{font:13px ui-serif,Georgia,serif;margin-bottom:8px;color:var(--ink,#1b1a17);}',
    '.m03-predict .m03-lab{font:11px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-muted,#57534e);}',
    '.m03-reveal{border:1px solid var(--border,#d6cfbf);border-left:3px solid var(--exc,#0f766e);border-radius:4px;padding:9px 12px;margin-top:8px;font:12.5px ui-serif,Georgia,serif;line-height:1.55;display:flex;gap:10px;align-items:center;flex-wrap:wrap;}',
    '.m03-bad{border-left-color:var(--mod,#7e22ce);}'
  ].join('\n');

  if (typeof module !== 'undefined' && module.exports) module.exports = { core, S };
  if (typeof root.Atlas !== 'undefined' && root.Atlas && root.Atlas.Plate) {
    root.Atlas.Plate.register({
      id: 'M03',
      title: { en: S.title },
      source: 'live',
      mount
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
