/**
 * PLATE Y01 — Spike-timing-dependent plasticity: who fired first?
 * Pair STDP with exponential traces (neurosush.synapses.traces, neurosush.synapses.plasticity).
 */
(function (root) {
  'use strict';

  // ---- String table (English only, no scattered literals) ----
  const S = {
    plate_id: 'PLATE Y01',
    plate_title: 'STDP: Who Fired First?',
    source_label: 'live',
    mirrors_label: 'neurosush.synapses.traces.Traces · neurosush.synapses.plasticity.STDP',

    // Hero anatomical labels
    pre_soma: 'Pre-synaptic Soma',
    post_soma: 'Post-synaptic Soma',
    pre_axon: 'Axon with Myelin Sheaths',
    post_dendrite: 'Dendritic Tree & Spine',
    terminal_bouton: 'Terminal Bouton',
    synaptic_cleft: 'Synaptic Cleft',
    vesicles_label: 'Synaptic Vesicles',
    receptors_label: 'Post-synaptic Density (Receptors)',
    synaptic_weight: 'Synaptic Weight w',
    click_hint: 'Click soma to force spike',

    // Timeline of two spike markers
    timeline_title: 'Spike-Pair Timing: Drag Pre (0 ms) or Post marker',
    marker_pre: 't_pre',
    marker_post: 't_post',
    drag_pre_hint: '◀ Drag t_pre ▶',
    drag_post_hint: '◀ Drag t_post ▶',
    lag_label: 'Timing Difference Δt = t_post - t_pre',
    causal_tag: 'LTP (Causal: Pre before Post)',
    acausal_tag: 'LTD (Anti-causal: Post before Pre)',
    synchronous_tag: 'Synchronous (Δt = 0)',

    // Oscilloscope strip
    strip_title: 'Trace Memory & Spike Sampling Read-Out',
    strip_pre_trace: 'x_pre(t) = e^(-t/τ_+)',
    strip_post_trace: 'x_post(t) = e^(-t/τ_-)',
    strip_sample_ltp: 'Post spike samples Pre trace',
    strip_sample_ltd: 'Pre spike samples Post trace',

    // STDP Window Plot
    plot_window_title: 'STDP Plasticity Window: Δw vs Δt',
    axis_dt: 'Timing Lag Δt = t_post - t_pre (ms)',
    axis_dw: 'Weight change Δw',
    tag_ltp: '▲ LTP (+a_+)',
    tag_ltd: '▼ LTD (-a_-)',
    theory_ghost_legend: 'Theory dw = ±a exp(-|Δt|/τ)',
    sim_points_legend: 'Simulated pairs',
    current_pair_legend: 'Live pair',

    // Running Weight Drift Plot
    plot_drift_title: 'Experiment: Weight Drift w vs Pairing #',
    axis_pairings: 'Pairing iteration (0 to 60)',
    axis_weight: 'Synaptic weight w',
    baseline_label: 'Initial w = 0.50',
    bound_max_label: 'w_max = 1.0',
    bound_min_label: 'w_min = 0.0',

    // Chapters
    chapter_label: 'Chapter',
    of_chapters: 'of',
    ch1_title: '1. Pre before post: potentiation',
    ch1_body: 'When pre fires shortly before post (Δt = +10 ms), the pre-trace is still elevated when the post-spike arrives. The second spike samples the decaying trace: the bouton grows and the synapse strengthens (LTP).',
    ch2_title: '2. Post before pre: depression',
    ch2_body: 'When post fires before pre (Δt = -10 ms), pre arrived too late to have caused post. The pre-spike samples the active post-trace: the bouton shrinks and the synapse weakens (LTD).',
    ch3_title: '3. The window shape: exponential in Δt',
    ch3_body: 'Because traces decay as (1 - dt/τ)^k ≈ exp(-|Δt|/τ), pairings closer in time yield exponentially larger weight changes. Beyond ~60 ms, traces have decayed to near zero.',
    ch4_title: '4. Run the experiment: build the curve',
    ch4_body: 'In classic experiments (Bi & Poo 1998), 60 spike pairs are delivered at 5 Hz for each Δt. Watch the weight drift and map the empirical plasticity window point by point.',
    ch5_title: '5. Why it learns causality (pre helps cause post)',
    ch5_body: 'STDP reinforces causal connections: inputs that consistently fire before post are strengthened, while anti-causal inputs are pruned. Uncorrelated Poisson activity yields balanced LTP/LTD with near-zero net drift.',

    // Controls
    btn_prev: '◀ Prev',
    btn_next: 'Next ▶',
    btn_run_60: '⚡ Run 60 pairings at this Δt',
    btn_running_60: 'Pairing (60 / 60)...',
    btn_mode_pair: 'Mode: Spike-Pair (Fixed Δt)',
    btn_mode_poisson: 'Mode: Uncorrelated Poisson',
    btn_reset_w: 'Reset Weight (w = 0.50)',
    btn_clear_curve: 'Clear Window Sweep',
    lbl_freq: 'Pairing rate:',
    unit_hz: 'Hz',
    unit_ms: 'ms',

    // Predict-then-reveal
    predict_title: 'Predict-then-Reveal',
    predict_question: 'Question: Post fires 5 ms BEFORE pre (Δt = -5 ms). Will the synapse become:',
    opt_stronger: '▲ Stronger (LTP)',
    opt_weaker: '▼ Weaker (LTD)',
    answer_correct: 'Correct! Post fired 5 ms before pre (Δt = -5 ms). Post trace is active when pre fires, triggering depression (dw = -0.0162 < 0).',
    answer_incorrect: 'Not quite. Because post fired before pre (Δt = -5 ms), pre did not cause post. Post trace is active when pre fires, causing depression (dw = -0.0162 < 0).',
    btn_try_again: 'Reset Prediction',

    // Marginalia
    equations_heading: 'Governing Equations',
    eq_pre_trace: 'τ_+ · dx_pre/dt = -x_pre  (decay: ×0.95/ms)',
    eq_post_trace: 'τ_- · dx_post/dt = -x_post  (decay: ×0.95/ms)',
    eq_ltp: 'Δw_+ = +a_+ · x_pre · post_spike',
    eq_ltd: 'Δw_- = -a_- · x_post · pre_spike',
    eq_net: 'Δw = Δw_+ - Δw_-',
    eq_bound: 'w ← clamp(w + Δw, 0, 1)',

    readouts_heading: 'Live Values',
    readout_dt: 'Lag Δt = t_post - t_pre',
    readout_x_pre: 'Pre trace x_pre',
    readout_x_post: 'Post trace x_post',
    readout_dw_pair: 'Pair change Δw',
    readout_weight: 'Synaptic weight w',
    readout_theory: 'Theory ghost Δw',
    readout_diff: 'Theory diff |sim - theo|',

    glossary_heading: 'Glossary',
    gl_stdp_term: 'STDP',
    gl_stdp_def: 'Spike-Timing-Dependent Plasticity: synaptic change dependent on millisecond relative spike timing.',
    gl_trace_term: 'Trace',
    gl_trace_def: 'Exponential memory of recent spikes: x(t) = (1 - dt/τ)^k, bridging the timing gap.',
    gl_ltp_term: 'Potentiation (LTP)',
    gl_ltp_def: 'Strengthening when pre precedes post (Δt > 0); indicates causal contribution.',
    gl_ltd_term: 'Depression (LTD)',
    gl_ltd_def: 'Weakening when post precedes pre (Δt < 0); indicates acausal arrival.',
    gl_causality_term: 'Causality',
    gl_causality_def: 'Inputs that fire before post can contribute to driving it; inputs firing after missed it.'
  };

  // ---- core: pure math, no DOM, exported for node tests ----
  function mulberry32(a) {
    let t = a >>> 0;
    return function () {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), t | 1);
      r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  const core = {
    // Default constants matching neurosush STDP defaults
    DEFAULT_A_PLUS: 0.02,
    DEFAULT_A_MINUS: 0.021,
    DEFAULT_TAU_PLUS: 20.0,
    DEFAULT_TAU_MINUS: 20.0,
    DEFAULT_DT: 1.0,
    DEFAULT_W_MIN: 0.0,
    DEFAULT_W_MAX: 1.0,
    DEFAULT_W_INIT: 0.5,

    mulberry32,

    /**
     * Discrete trace step matching neurosush.synapses.traces.trace_step:
     * decays trace by (1 - dt / tau), then adds spike * scale.
     */
    trace_step(trace, spike = 0, opts = {}) {
      const tau = opts.tau ?? core.DEFAULT_TAU_PLUS;
      const dt = opts.dt ?? core.DEFAULT_DT;
      const scale = opts.scale ?? 1.0;
      const nearest = Boolean(opts.nearest);
      if (!Number.isFinite(tau) || tau <= 0) {
        throw new Error('tau must be positive, got ' + tau);
      }
      if (!Number.isFinite(dt) || dt <= 0) {
        throw new Error('dt must be positive, got ' + dt);
      }
      const decayed = trace * (1.0 - dt / tau);
      if (nearest) {
        return spike > 0 ? scale : decayed;
      }
      return decayed + (spike > 0 ? scale : 0.0);
    },

    /**
     * Closed-form discrete decay after k steps without new spikes:
     * trace_k = trace0 * (1 - dt / tau)^k.
     */
    trace_k_steps(trace0, k, opts = {}) {
      const tau = opts.tau ?? core.DEFAULT_TAU_PLUS;
      const dt = opts.dt ?? core.DEFAULT_DT;
      if (!Number.isFinite(tau) || tau <= 0) {
        throw new Error('tau must be positive, got ' + tau);
      }
      if (!Number.isFinite(dt) || dt <= 0) {
        throw new Error('dt must be positive, got ' + dt);
      }
      if (k < 0) {
        throw new Error('k must be non-negative, got ' + k);
      }
      return trace0 * Math.pow(1.0 - dt / tau, k);
    },

    /**
     * Compute weight change dw for a single pair of spikes with time lag
     * delta_t = t_post - t_pre (ms).
     *
     * Convention at delta_t = 0:
     * Both spikes fire simultaneously at the same discrete step. In discrete STDP
     * where both pre and post traces equal 1, potentiation adds +a_plus and depression
     * subtracts -a_minus, yielding net dw = a_plus - a_minus (or configurable).
     * For delta_t > 0: pre fires first -> potentiation: +a_plus * (1 - dt/tau_plus)^k.
     * For delta_t < 0: post fires first -> depression: -a_minus * (1 - dt/tau_minus)^k.
     */
    stdp_pair_dw(delta_t, opts = {}) {
      const a_plus = opts.a_plus ?? core.DEFAULT_A_PLUS;
      const a_minus = opts.a_minus ?? core.DEFAULT_A_MINUS;
      const tau_plus = opts.tau_plus ?? core.DEFAULT_TAU_PLUS;
      const tau_minus = opts.tau_minus ?? core.DEFAULT_TAU_MINUS;
      const dt = opts.dt ?? core.DEFAULT_DT;
      const at_zero = opts.at_zero ?? 'net';

      if (!Number.isFinite(delta_t)) {
        throw new Error('delta_t must be finite, got ' + delta_t);
      }
      if (tau_plus <= 0 || tau_minus <= 0) {
        throw new Error('time constants must be positive');
      }

      if (delta_t > 0) {
        const k = delta_t / dt;
        return a_plus * Math.pow(1.0 - dt / tau_plus, k);
      } else if (delta_t < 0) {
        const k = -delta_t / dt;
        return -a_minus * Math.pow(1.0 - dt / tau_minus, k);
      } else {
        if (at_zero === 'zero') return 0.0;
        if (at_zero === 'ltp') return a_plus;
        if (at_zero === 'ltd') return -a_minus;
        return a_plus - a_minus;
      }
    },

    /**
     * Continuous closed-form theoretical STDP window:
     * dw = +a_plus * exp(-delta_t / tau_plus) for delta_t > 0
     * dw = -a_minus * exp(delta_t / tau_minus) for delta_t < 0
     */
    theory_dw(delta_t, opts = {}) {
      const a_plus = opts.a_plus ?? core.DEFAULT_A_PLUS;
      const a_minus = opts.a_minus ?? core.DEFAULT_A_MINUS;
      const tau_plus = opts.tau_plus ?? core.DEFAULT_TAU_PLUS;
      const tau_minus = opts.tau_minus ?? core.DEFAULT_TAU_MINUS;

      if (!Number.isFinite(delta_t)) {
        throw new Error('delta_t must be finite, got ' + delta_t);
      }
      if (delta_t > 0) {
        return a_plus * Math.exp(-delta_t / tau_plus);
      } else if (delta_t < 0) {
        return -a_minus * Math.exp(delta_t / tau_minus);
      } else {
        return a_plus - a_minus;
      }
    },

    /**
     * Apply weight bounds (hard clamping, soft boundary gates, or unbounded).
     */
    apply_bounds(w, dw, opts = {}) {
      const w_min = opts.w_min ?? core.DEFAULT_W_MIN;
      const w_max = opts.w_max ?? core.DEFAULT_W_MAX;
      const bound = opts.bound ?? 'hard';

      if (bound === 'none') {
        return w + dw;
      }
      if (bound === 'soft') {
        const gate_plus = Math.max(0.0, w_max - w);
        const gate_minus = Math.max(0.0, w - w_min);
        const effective_dw = dw >= 0 ? dw * gate_plus : dw * gate_minus;
        return Math.min(w_max, Math.max(w_min, w + effective_dw));
      }
      // hard bound
      return Math.min(w_max, Math.max(w_min, w + dw));
    },

    /**
     * Run N repeated pairings at a fixed delta_t and record weight trajectory.
     */
    run_pairings(delta_t, n_pairs = 60, initial_w = core.DEFAULT_W_INIT, opts = {}) {
      const dw_single = core.stdp_pair_dw(delta_t, opts);
      let w = initial_w;
      const history = [w];
      for (let i = 0; i < n_pairs; i++) {
        w = core.apply_bounds(w, dw_single, opts);
        history.push(w);
      }
      return {
        delta_t,
        n_pairs,
        initial_w,
        final_w: w,
        total_dw: w - initial_w,
        single_dw: dw_single,
        history
      };
    },

    /**
     * Sample Poisson spike for interval dt (ms) with rate r (Hz).
     */
    poisson_spike(rate_hz, dt_ms, rng) {
      if (rate_hz <= 0) return 0;
      const p = rate_hz * (dt_ms / 1000.0);
      return rng() < p ? 1 : 0;
    },

    /**
     * Simulate uncorrelated Poisson firing for duration_s and compute drift.
     * Analytical expectation rate: (a_plus * tau_plus/1000 - a_minus * tau_minus/1000) * r1 * r2
     */
    simulate_poisson_drift(r1_hz, r2_hz, duration_s, opts = {}) {
      const dt = opts.dt ?? core.DEFAULT_DT;
      const a_plus = opts.a_plus ?? core.DEFAULT_A_PLUS;
      const a_minus = opts.a_minus ?? core.DEFAULT_A_MINUS;
      const tau_plus = opts.tau_plus ?? core.DEFAULT_TAU_PLUS;
      const tau_minus = opts.tau_minus ?? core.DEFAULT_TAU_MINUS;
      const w_init = opts.w_init ?? core.DEFAULT_W_INIT;
      const seed = opts.seed ?? 12345;
      const bound = opts.bound ?? 'hard';

      const rng = mulberry32(seed);
      const steps = Math.round((duration_s * 1000.0) / dt);

      let pre_trace = 0.0;
      let post_trace = 0.0;
      let w = w_init;
      let pre_spikes = 0;
      let post_spikes = 0;
      let total_ltp = 0.0;
      let total_ltd = 0.0;

      for (let s = 0; s < steps; s++) {
        pre_trace = core.trace_step(pre_trace, 0, { tau: tau_plus, dt });
        post_trace = core.trace_step(post_trace, 0, { tau: tau_minus, dt });

        const s_pre = core.poisson_spike(r1_hz, dt, rng);
        const s_post = core.poisson_spike(r2_hz, dt, rng);

        if (s_pre) {
          pre_trace += 1.0;
          pre_spikes++;
        }
        if (s_post) {
          post_trace += 1.0;
          post_spikes++;
        }

        const ltp = a_plus * pre_trace * s_post;
        const ltd = a_minus * post_trace * s_pre;
        const dw = ltp - ltd;

        total_ltp += ltp;
        total_ltd += ltd;
        w = core.apply_bounds(w, dw, { bound, ...opts });
      }

      // Continuous theoretical expectation of total drift
      const tau_p_sec = tau_plus / 1000.0;
      const tau_m_sec = tau_minus / 1000.0;
      const drift_rate_expected = (a_plus * tau_p_sec - a_minus * tau_m_sec) * r1_hz * r2_hz;
      const expected_dw = drift_rate_expected * duration_s;

      return {
        pre_spikes,
        post_spikes,
        final_w: w,
        total_dw: w - w_init,
        total_ltp,
        total_ltd,
        drift_rate_expected,
        expected_dw,
        duration_s
      };
    }
  };

  // ---- ui: only runs when a browser Atlas shell exists ----
  function mount(rootEl, ctx) {
    const { colors, margin, figure } = ctx;
    const parentContainer = (figure && figure.parentElement) || rootEl;

    // State variables
    let tPreMarker = 0.0; // ms
    let tPostMarker = 10.0; // ms
    let currentDeltaT = 10.0; // ms = tPostMarker - tPreMarker
    let currentWeight = core.DEFAULT_W_INIT;
    let pairingRate = 5.0; // Hz
    let currentChapter = 1; // 1..5
    let showTheory = true;
    let hoveredTerm = null;
    let isPoissonMode = false;
    let isRunning60 = false;
    let run60Progress = 0;
    let run60Timer = 0.0;
    let userPredicted = null;

    // Simulation animation state
    let simTimeMs = 0.0;
    let preTrace = 0.0;
    let postTrace = 0.0;
    let preFlash = 0.0;
    let postFlash = 0.0;
    let axonPulseProgress = -1.0; // [0, 1]
    let backpropPulseProgress = -1.0; // [0, 1]
    let vesiclesReleased = []; // [{ x, y, vx, vy, life }]
    let plasticityPulse = 0.0; // Flash intensity on bouton growth/shrink
    let lastDwApplied = 0.0;
    let weightDriftHistory = [{ step: 0, w: currentWeight }];
    let sweepPoints = []; // Recorded points [{ dt, dw, w }]

    // Seed some initial sweep points for demonstration
    for (const dtVal of [-30, -20, -10, -5, 5, 10, 20, 30]) {
      sweepPoints.push({
        dt: dtVal,
        dw: core.stdp_pair_dw(dtVal),
        w: core.DEFAULT_W_INIT
      });
    }

    // Poisson state
    const poissonRng = core.mulberry32(999);

    // Listen for shell theory ghost event
    window.addEventListener('atlas:theory', (e) => {
      showTheory = e.detail && typeof e.detail.active === 'boolean' ? e.detail.active : !showTheory;
    });

    // Populate Margin
    if (margin) {
      margin.innerHTML = `
        <div class="atlas-margin-section" id="y01-margin-eqs">
          <div class="atlas-margin-label">${S.equations_heading}</div>
          <div class="atlas-equation-box atlas-mono">
            <div class="y01-term" data-term="trace_pre">${S.eq_pre_trace}</div>
            <div class="y01-term" data-term="trace_post">${S.eq_post_trace}</div>
            <div class="y01-term" data-term="ltp">${S.eq_ltp}</div>
            <div class="y01-term" data-term="ltd">${S.eq_ltd}</div>
            <div class="y01-term" data-term="net">${S.eq_net}</div>
            <div class="y01-term" data-term="weight">${S.eq_bound}</div>
          </div>
        </div>

        <div class="atlas-margin-section" id="y01-margin-readouts">
          <div class="atlas-margin-label">${S.readouts_heading}</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div>
              <div style="font-size:11px; color:var(--ink-muted);">${S.readout_dt}</div>
              <div class="atlas-margin-value atlas-mono" id="y01-val-dt">+10.0 ms</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--ink-muted);">${S.readout_dw_pair}</div>
              <div class="atlas-margin-value atlas-mono" id="y01-val-dw">+0.0121</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--ink-muted);">${S.readout_x_pre}</div>
              <div class="atlas-margin-value atlas-mono" id="y01-val-xpre">0.000</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--ink-muted);">${S.readout_x_post}</div>
              <div class="atlas-margin-value atlas-mono" id="y01-val-xpost">0.000</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--ink-muted);">${S.readout_weight}</div>
              <div class="atlas-margin-value atlas-mono" id="y01-val-weight">0.500</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--ink-muted);">${S.readout_diff}</div>
              <div class="atlas-margin-value atlas-mono" id="y01-val-diff">0.0001</div>
            </div>
          </div>
        </div>

        <div class="atlas-margin-section" id="y01-margin-glossary">
          <div class="atlas-margin-label">${S.glossary_heading}</div>
          <div style="display:flex; flex-direction:column; gap:6px; font-size:12px; line-height:1.4;">
            <div><strong>${S.gl_stdp_term}:</strong> ${S.gl_stdp_def}</div>
            <div><strong>${S.gl_trace_term}:</strong> ${S.gl_trace_def}</div>
            <div><strong>${S.gl_ltp_term}:</strong> ${S.gl_ltp_def}</div>
            <div><strong>${S.gl_ltd_term}:</strong> ${S.gl_ltd_def}</div>
            <div><strong>${S.gl_causality_term}:</strong> ${S.gl_causality_def}</div>
          </div>
        </div>
      `;

      // Term hover listeners
      margin.querySelectorAll('.y01-term').forEach((el) => {
        el.style.cursor = 'pointer';
        el.style.padding = '2px 4px';
        el.style.borderRadius = '2px';
        el.addEventListener('mouseenter', () => {
          hoveredTerm = el.getAttribute('data-term');
          el.style.background = 'var(--paper-card)';
        });
        el.addEventListener('mouseleave', () => {
          hoveredTerm = null;
          el.style.background = 'transparent';
        });
      });
    }

    const valDtEl = document.getElementById('y01-val-dt');
    const valDwEl = document.getElementById('y01-val-dw');
    const valXpreEl = document.getElementById('y01-val-xpre');
    const valXpostEl = document.getElementById('y01-val-xpost');
    const valWeightEl = document.getElementById('y01-val-weight');
    const valDiffEl = document.getElementById('y01-val-diff');

    // Create Main Figure Elements
    const storyBanner = document.createElement('div');
    storyBanner.className = 'y01-story-banner';
    storyBanner.style.cssText = `
      background: var(--paper-card);
      border: 1px solid var(--border);
      border-radius: var(--radius, 3px);
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 13px;
    `;

    storyBanner.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-family:var(--font-mono); font-weight:700; color:var(--ink-muted);" id="y01-chapter-idx">
            ${S.chapter_label} 1 ${S.of_chapters} 5
          </span>
          <span style="font-family:var(--font-serif); font-weight:600; font-size:14px;" id="y01-chapter-title">
            ${S.ch1_title}
          </span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <button class="atlas-btn" id="y01-btn-prev">${S.btn_prev}</button>
          <button class="atlas-btn" id="y01-btn-next">${S.btn_next}</button>
        </div>
      </div>
      <div style="color:var(--ink-muted); line-height:1.4;" id="y01-chapter-body">
        ${S.ch1_body}
      </div>
    `;

    // Predict-then-reveal Card
    const predictCard = document.createElement('div');
    predictCard.className = 'y01-predict-card';
    predictCard.style.cssText = `
      background: var(--paper-inset);
      border: 1px solid var(--border);
      border-radius: var(--radius, 3px);
      padding: 6px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
    `;

    predictCard.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
        <strong style="font-family:var(--font-serif); color:var(--ink);">${S.predict_title}</strong>
        <span style="color:var(--ink-muted);">${S.predict_question}</span>
      </div>
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;" id="y01-predict-buttons">
        <button class="atlas-btn" id="y01-opt-stronger" style="border-color:var(--exc); color:var(--exc);">${S.opt_stronger}</button>
        <button class="atlas-btn" id="y01-opt-weaker" style="border-color:var(--inh); color:var(--inh);">${S.opt_weaker}</button>
        <span id="y01-predict-feedback" style="margin-left:8px; font-weight:500;"></span>
      </div>
    `;

    // Interactive Controls Bar
    const controlsBar = document.createElement('div');
    controlsBar.className = 'y01-controls-bar';
    controlsBar.style.cssText = `
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 6px 10px;
      background: var(--paper-card);
      border: 1px solid var(--border);
      border-radius: var(--radius, 3px);
      font-size: 12px;
    `;

    controlsBar.innerHTML = `
      <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
        <button class="atlas-btn" id="y01-btn-run60" style="font-weight:600;">${S.btn_run_60}</button>
        <button class="atlas-btn" id="y01-btn-mode">${S.btn_mode_pair}</button>
        <button class="atlas-btn" id="y01-btn-reset-w">${S.btn_reset_w}</button>
        <button class="atlas-btn" id="y01-btn-clear">${S.btn_clear_curve}</button>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <label for="y01-slider-freq" style="color:var(--ink-muted);">${S.lbl_freq}</label>
        <input type="range" id="y01-slider-freq" min="1" max="25" value="5" step="1" style="width:80px; cursor:pointer;" />
        <span class="atlas-mono" id="y01-text-freq" style="min-width:32px;">5 Hz</span>
      </div>
    `;

    // Canvas container provided by Atlas shell (rootEl is .atlas-canvas-container)
    const canvasContainer = rootEl.classList && rootEl.classList.contains('atlas-canvas-container')
      ? rootEl
      : (rootEl.querySelector('.atlas-canvas-container') || rootEl);
    const figureSection = canvasContainer.parentElement || parentContainer;

    // Optimal canvas height so full plate fits in 800px on desktop without clipping
    canvasContainer.style.aspectRatio = 'auto';
    canvasContainer.style.height = '355px';
    canvasContainer.style.minHeight = '320px';
    canvasContainer.style.maxHeight = '560px';

    // Insert banner, predictCard, controlsBar around the canvasContainer
    if (figureSection && figureSection.contains(canvasContainer)) {
      figureSection.insertBefore(storyBanner, canvasContainer);
      if (canvasContainer.nextSibling) {
        figureSection.insertBefore(predictCard, canvasContainer.nextSibling);
        figureSection.insertBefore(controlsBar, predictCard.nextSibling);
      } else {
        figureSection.appendChild(predictCard);
        figureSection.appendChild(controlsBar);
      }
    }

    // Create Main Canvas inside canvasContainer
    const heroCanvas = document.createElement('canvas');
    heroCanvas.className = 'atlas-canvas';
    heroCanvas.style.cssText = 'width:100%; height:100%; display:block; cursor:default;';
    canvasContainer.innerHTML = '';
    canvasContainer.appendChild(heroCanvas);
    const heroCtx = heroCanvas.getContext('2d');

    // UI Event Handlers
    const chapterTitleEl = document.getElementById('y01-chapter-title');
    const chapterBodyEl = document.getElementById('y01-chapter-body');
    const chapterIdxEl = document.getElementById('y01-chapter-idx');
    const btnPrevChapter = document.getElementById('y01-btn-prev');
    const btnNextChapter = document.getElementById('y01-btn-next');
    const btnRun60 = document.getElementById('y01-btn-run60');
    const btnMode = document.getElementById('y01-btn-mode');
    const btnResetW = document.getElementById('y01-btn-reset-w');
    const btnClearCurve = document.getElementById('y01-btn-clear');
    const sliderFreq = document.getElementById('y01-slider-freq');
    const textFreq = document.getElementById('y01-text-freq');
    const optStronger = document.getElementById('y01-opt-stronger');
    const optWeaker = document.getElementById('y01-opt-weaker');
    const predictFeedback = document.getElementById('y01-predict-feedback');

    function updateTimings(preVal, postVal) {
      tPreMarker = Math.max(-40.0, Math.min(40.0, Math.round(preVal * 10) / 10));
      tPostMarker = Math.max(-40.0, Math.min(40.0, Math.round(postVal * 10) / 10));
      currentDeltaT = Math.round((tPostMarker - tPreMarker) * 10) / 10;
      const dw = core.stdp_pair_dw(currentDeltaT);
      sweepPoints.push({ dt: currentDeltaT, dw, w: currentWeight });
      if (sweepPoints.length > 150) sweepPoints.shift();
    }

    function setDeltaT(val) {
      tPreMarker = 0.0;
      tPostMarker = Math.max(-40.0, Math.min(40.0, Math.round(val * 10) / 10));
      currentDeltaT = tPostMarker;
      const dw = core.stdp_pair_dw(currentDeltaT);
      sweepPoints.push({ dt: currentDeltaT, dw, w: currentWeight });
      if (sweepPoints.length > 150) sweepPoints.shift();
    }

    function applyChapter(num) {
      currentChapter = Math.max(1, Math.min(5, num));
      chapterIdxEl.textContent = `${S.chapter_label} ${currentChapter} ${S.of_chapters} 5`;
      if (currentChapter === 1) {
        chapterTitleEl.textContent = S.ch1_title;
        chapterBodyEl.textContent = S.ch1_body;
        setDeltaT(10.0);
        isPoissonMode = false;
        btnMode.textContent = S.btn_mode_pair;
      } else if (currentChapter === 2) {
        chapterTitleEl.textContent = S.ch2_title;
        chapterBodyEl.textContent = S.ch2_body;
        setDeltaT(-10.0);
        isPoissonMode = false;
        btnMode.textContent = S.btn_mode_pair;
      } else if (currentChapter === 3) {
        chapterTitleEl.textContent = S.ch3_title;
        chapterBodyEl.textContent = S.ch3_body;
        setDeltaT(20.0);
        showTheory = true;
      } else if (currentChapter === 4) {
        chapterTitleEl.textContent = S.ch4_title;
        chapterBodyEl.textContent = S.ch4_body;
        setDeltaT(5.0);
        if (!isRunning60) {
          isRunning60 = true;
          run60Progress = 0;
          run60Timer = 0.0;
          weightDriftHistory = [{ step: 0, w: currentWeight }];
          btnRun60.disabled = true;
          btnRun60.textContent = S.btn_running_60;
        }
      } else if (currentChapter === 5) {
        chapterTitleEl.textContent = S.ch5_title;
        chapterBodyEl.textContent = S.ch5_body;
        isPoissonMode = true;
        btnMode.textContent = S.btn_mode_poisson;
      }
    }

    btnPrevChapter.addEventListener('click', () => applyChapter(currentChapter - 1));
    btnNextChapter.addEventListener('click', () => applyChapter(currentChapter + 1));

    btnRun60.addEventListener('click', () => {
      if (isRunning60) return;
      isRunning60 = true;
      run60Progress = 0;
      run60Timer = 0.0;
      weightDriftHistory = [{ step: 0, w: currentWeight }];
      btnRun60.disabled = true;
      btnRun60.textContent = S.btn_running_60;
    });

    btnMode.addEventListener('click', () => {
      isPoissonMode = !isPoissonMode;
      btnMode.textContent = isPoissonMode ? S.btn_mode_poisson : S.btn_mode_pair;
      if (isPoissonMode) {
        weightDriftHistory = [{ step: 0, w: currentWeight }];
      }
    });

    btnResetW.addEventListener('click', () => {
      currentWeight = core.DEFAULT_W_INIT;
      weightDriftHistory = [{ step: 0, w: currentWeight }];
    });

    btnClearCurve.addEventListener('click', () => {
      sweepPoints = [];
    });

    sliderFreq.addEventListener('input', (e) => {
      pairingRate = parseFloat(e.target.value);
      textFreq.textContent = `${pairingRate} ${S.unit_hz}`;
    });

    optStronger.addEventListener('click', () => {
      userPredicted = 'stronger';
      predictFeedback.textContent = S.answer_incorrect;
      predictFeedback.style.color = colors().inh;
      setDeltaT(-5.0);
      showTheory = true;
    });

    optWeaker.addEventListener('click', () => {
      userPredicted = 'weaker';
      predictFeedback.textContent = S.answer_correct;
      predictFeedback.style.color = colors().exc;
      setDeltaT(-5.0);
      showTheory = true;
    });

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'ArrowLeft') {
        applyChapter(currentChapter - 1);
      } else if (e.key === 'ArrowRight') {
        applyChapter(currentChapter + 1);
      } else if (e.key === 't' || e.key === 'T') {
        showTheory = !showTheory;
      }
    });

    // Dragging state on canvas (Two Timeline Markers, STDP Window Plot)
    let activeDragTarget = null; // 'marker_pre' | 'marker_post' | 'plot'

    function getCanvasCoords(e) {
      const rect = heroCanvas.getBoundingClientRect();
      const clientX = e.clientX ?? (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
      const clientY = e.clientY ?? (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
      const dpr = window.devicePixelRatio || 1;
      const x = (clientX - rect.left) * (heroCanvas.width / (rect.width * dpr));
      const y = (clientY - rect.top) * (heroCanvas.height / (rect.height * dpr));
      return { x, y, width: rect.width, height: rect.height };
    }

    // Geometry layout cache
    let layout = {
      preX: 110,
      postX: 680,
      somaY: 70,
      somaR: 24,
      synX: 395,
      timelineX: 25,
      timelineY: 175,
      timelineW: 395,
      timelineH: 45,
      stripX: 25,
      stripY: 235,
      stripW: 395,
      stripH: 165,
      plotX: 445,
      plotY: 165,
      plotW: 330,
      plotH: 115,
      driftX: 445,
      driftY: 295,
      driftW: 330,
      driftH: 105
    };

    function triggerPreSpike() {
      preFlash = 1.0;
      preTrace = 1.0;
      axonPulseProgress = 0.0;
      // If post was first (dt < 0), depression occurs at pre spike
      if (currentDeltaT < 0 && postTrace > 0.01) {
        const dw = -core.DEFAULT_A_MINUS * postTrace;
        currentWeight = core.apply_bounds(currentWeight, dw);
        lastDwApplied = dw;
        plasticityPulse = 1.0;
        if (weightDriftHistory.length < 60) {
          weightDriftHistory.push({ step: weightDriftHistory.length, w: currentWeight });
        }
      }
    }

    function triggerPostSpike() {
      postFlash = 1.0;
      postTrace = 1.0;
      backpropPulseProgress = 0.0;
      // If pre was first (dt > 0), potentiation occurs at post spike
      if (currentDeltaT > 0 && preTrace > 0.01) {
        const dw = core.DEFAULT_A_PLUS * preTrace;
        currentWeight = core.apply_bounds(currentWeight, dw);
        lastDwApplied = dw;
        plasticityPulse = 1.0;
        if (weightDriftHistory.length < 60) {
          weightDriftHistory.push({ step: weightDriftHistory.length, w: currentWeight });
        }
      }
    }

    heroCanvas.addEventListener('pointerdown', (e) => {
      const { x, y } = getCanvasCoords(e);

      // Check click on Pre soma
      if (Math.hypot(x - layout.preX, y - layout.somaY) <= layout.somaR + 10) {
        triggerPreSpike();
        return;
      }

      // Check click on Post soma
      if (Math.hypot(x - layout.postX, y - layout.somaY) <= layout.somaR + 10) {
        triggerPostSpike();
        return;
      }

      // Marker coordinate helpers on Timeline: range [-50, +50] ms
      const toTimelineX = (tMs) => layout.timelineX + ((tMs + 50.0) / 100.0) * layout.timelineW;
      const preMarkerX = toTimelineX(tPreMarker);
      const postMarkerX = toTimelineX(tPostMarker);
      const tAxisY = layout.timelineY + 24;

      // Hit test Pre marker (above axis)
      if (Math.abs(x - preMarkerX) <= 24 && y >= tAxisY - 20 && y <= tAxisY + 2) {
        activeDragTarget = 'marker_pre';
        heroCanvas.setPointerCapture(e.pointerId);
        return;
      }

      // Hit test Post marker (below axis)
      if (Math.abs(x - postMarkerX) <= 26 && y >= tAxisY - 2 && y <= tAxisY + 22) {
        activeDragTarget = 'marker_post';
        heroCanvas.setPointerCapture(e.pointerId);
        return;
      }

      // Hit test Timeline track
      if (x >= layout.timelineX && x <= layout.timelineX + layout.timelineW && y >= layout.timelineY && y <= layout.timelineY + layout.timelineH) {
        // Drag closest marker to pointer
        const distToPre = Math.abs(x - preMarkerX);
        const distToPost = Math.abs(x - postMarkerX);
        activeDragTarget = distToPost <= distToPre ? 'marker_post' : 'marker_pre';
        heroCanvas.setPointerCapture(e.pointerId);
        updateTimelineMarkerDrag(x);
        return;
      }

      // Hit test STDP Window Plot
      if (x >= layout.plotX && x <= layout.plotX + layout.plotW && y >= layout.plotY && y <= layout.plotY + layout.plotH) {
        activeDragTarget = 'plot';
        heroCanvas.setPointerCapture(e.pointerId);
        updatePlotDrag(x);
        return;
      }
    });

    function updateTimelineMarkerDrag(canvasX) {
      const ratio = (canvasX - layout.timelineX) / layout.timelineW;
      const clampedRatio = Math.max(0.0, Math.min(1.0, ratio));
      const tVal = -50.0 + clampedRatio * 100.0;
      if (activeDragTarget === 'marker_pre') {
        updateTimings(tVal, tPostMarker);
      } else if (activeDragTarget === 'marker_post') {
        updateTimings(tPreMarker, tVal);
      }
    }

    function updatePlotDrag(canvasX) {
      const ratio = (canvasX - layout.plotX) / layout.plotW;
      const clampedRatio = Math.max(0.0, Math.min(1.0, ratio));
      const dt = -50.0 + clampedRatio * 100.0;
      setDeltaT(dt);
    }

    heroCanvas.addEventListener('pointermove', (e) => {
      const { x, y } = getCanvasCoords(e);
      if (activeDragTarget === 'marker_pre' || activeDragTarget === 'marker_post') {
        updateTimelineMarkerDrag(x);
      } else if (activeDragTarget === 'plot') {
        updatePlotDrag(x);
      } else {
        // Change cursor over interactive elements
        const distPre = Math.hypot(x - layout.preX, y - layout.somaY);
        const distPost = Math.hypot(x - layout.postX, y - layout.somaY);
        const toTimelineX = (tMs) => layout.timelineX + ((tMs + 50.0) / 100.0) * layout.timelineW;
        const tAxisY = layout.timelineY + 24;
        const overPreMarker = Math.abs(x - toTimelineX(tPreMarker)) <= 24 && y >= tAxisY - 20 && y <= tAxisY + 2;
        const overPostMarker = Math.abs(x - toTimelineX(tPostMarker)) <= 26 && y >= tAxisY - 2 && y <= tAxisY + 22;
        const overTimeline = x >= layout.timelineX && x <= layout.timelineX + layout.timelineW && y >= layout.timelineY && y <= layout.timelineY + layout.timelineH;
        const overPlot = x >= layout.plotX && x <= layout.plotX + layout.plotW && y >= layout.plotY && y <= layout.plotY + layout.plotH;

        if (distPre <= layout.somaR + 6 || distPost <= layout.somaR + 6) {
          heroCanvas.style.cursor = 'pointer';
        } else if (overPreMarker || overPostMarker || overTimeline || overPlot) {
          heroCanvas.style.cursor = 'ew-resize';
        } else {
          heroCanvas.style.cursor = 'default';
        }
      }
    });

    const stopDrag = (e) => {
      if (activeDragTarget) {
        try {
          heroCanvas.releasePointerCapture(e.pointerId);
        } catch {}
      }
      activeDragTarget = null;
    };
    heroCanvas.addEventListener('pointerup', stopDrag);
    heroCanvas.addEventListener('pointercancel', stopDrag);

    // Animation & Render Loop
    let lastTimestamp = performance.now();

    function render(timestamp) {
      const dtMs = Math.min(60, Math.max(1, timestamp - lastTimestamp));
      lastTimestamp = timestamp;

      const c = colors();
      const rect = heroCanvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = Math.round(rect.width || 800);
      const H = Math.round(rect.height || 415);

      if (heroCanvas.width !== Math.round(W * dpr) || heroCanvas.height !== Math.round(H * dpr)) {
        heroCanvas.width = Math.round(W * dpr);
        heroCanvas.height = Math.round(H * dpr);
      }

      heroCtx.save();
      heroCtx.scale(dpr, dpr);
      heroCtx.clearRect(0, 0, W, H);

      // Dynamic responsive layout
      const isMobile = W < 600;
      const targetH = isMobile ? 510 : 355;
      if (canvasContainer.style.height !== targetH + 'px') {
        canvasContainer.style.height = targetH + 'px';
      }

      const somaY = isMobile ? 60 : 66;
      const preX = isMobile ? Math.round(W * 0.16) : Math.round(W * 0.17);
      const postX = isMobile ? Math.round(W * 0.84) : Math.round(W * 0.83);
      const synX = Math.round(W * 0.5);
      const somaR = isMobile ? 18 : 22;

      let timelineX, timelineY, timelineW, timelineH;
      let stripX, stripY, stripW, stripH;
      let plotX, plotY, plotW, plotH;
      let driftX, driftY, driftW, driftH;

      if (!isMobile) {
        timelineX = Math.round(W * 0.03);
        timelineY = 138;
        timelineW = Math.round(W * 0.45);
        timelineH = 44;

        stripX = timelineX;
        stripY = 190;
        stripW = timelineW;
        stripH = H - stripY - 6;

        plotX = Math.round(W * 0.51);
        plotY = 138;
        plotW = Math.round(W * 0.46);
        plotH = 92;

        driftX = plotX;
        driftY = plotY + plotH + 12;
        driftW = plotW;
        driftH = H - driftY - 6;
      } else {
        timelineX = Math.round(W * 0.04);
        timelineY = 124;
        timelineW = Math.round(W * 0.92);
        timelineH = 44;

        stripX = timelineX;
        stripY = 182;
        stripW = timelineW;
        stripH = 96;

        plotX = timelineX;
        plotY = 292;
        plotW = timelineW;
        plotH = 96;

        driftX = timelineX;
        driftY = 402;
        driftW = timelineW;
        driftH = 96;
      }

      layout = {
        preX,
        postX,
        somaY,
        somaR,
        synX,
        timelineX,
        timelineY,
        timelineW,
        timelineH,
        stripX,
        stripY,
        stripW,
        stripH,
        plotX,
        plotY,
        plotW,
        plotH,
        driftX,
        driftY,
        driftW,
        driftH
      };

      // Update Simulation Physics
      simTimeMs += dtMs;
      const pairPeriodMs = 1000.0 / pairingRate;

      if (isRunning60) {
        run60Timer += dtMs;
        if (run60Timer >= 35.0) {
          run60Timer -= 35.0;
          run60Progress++;
          const dwSingle = core.stdp_pair_dw(currentDeltaT);
          currentWeight = core.apply_bounds(currentWeight, dwSingle);
          lastDwApplied = dwSingle;
          plasticityPulse = 1.0;
          weightDriftHistory.push({ step: run60Progress, w: currentWeight });

          if (currentDeltaT >= 0) {
            triggerPreSpike();
            postFlash = 1.0;
            postTrace = 1.0;
          } else {
            triggerPostSpike();
            preFlash = 1.0;
            preTrace = 1.0;
          }

          if (run60Progress >= 60) {
            isRunning60 = false;
            btnRun60.disabled = false;
            btnRun60.textContent = S.btn_run_60;
            sweepPoints.push({
              dt: currentDeltaT,
              dw: dwSingle,
              w: currentWeight,
              marker: 'exp60'
            });
          }
        }
      } else if (!isPoissonMode) {
        // Periodic spike pair
        const cycleT = simTimeMs % pairPeriodMs;
        const prevCycleT = (simTimeMs - dtMs) % pairPeriodMs;

        let tPreCycle = 25.0;
        let tPostCycle = 25.0 + currentDeltaT;
        if (currentDeltaT < 0) {
          tPostCycle = 25.0;
          tPreCycle = 25.0 - currentDeltaT;
        }

        // Fire Pre
        if (prevCycleT < tPreCycle && cycleT >= tPreCycle) {
          triggerPreSpike();
        }

        // Fire Post
        if (prevCycleT < tPostCycle && cycleT >= tPostCycle) {
          triggerPostSpike();
        }
      } else if (isPoissonMode) {
        // Poisson random firing mode
        if (core.poisson_spike(pairingRate * 2.0, dtMs, poissonRng)) {
          triggerPreSpike();
        }
        if (core.poisson_spike(pairingRate * 2.0, dtMs, poissonRng)) {
          triggerPostSpike();
        }
      }

      // Continuous trace decay: (1 - dt/tau)
      const decayFactor = Math.pow(1.0 - core.DEFAULT_DT / core.DEFAULT_TAU_PLUS, dtMs);
      preTrace *= decayFactor;
      postTrace *= decayFactor;
      preFlash = Math.max(0.0, preFlash - dtMs * 0.004);
      postFlash = Math.max(0.0, postFlash - dtMs * 0.004);
      plasticityPulse = Math.max(0.0, plasticityPulse - dtMs * 0.003);

      // Action potential pulse along axon (1 ms transmission)
      const boutonX = synX - 10;
      if (axonPulseProgress >= 0.0) {
        axonPulseProgress += dtMs * 0.0035;
        if (axonPulseProgress >= 1.0) {
          axonPulseProgress = -1.0;
          // Release neurotransmitter vesicle particles into the cleft!
          for (let k = 0; k < 12; k++) {
            vesiclesReleased.push({
              x: boutonX + 2,
              y: somaY + (Math.random() - 0.5) * 16,
              vx: 0.08 + Math.random() * 0.12,
              vy: (Math.random() - 0.5) * 0.06,
              life: 1.0
            });
          }
        }
      }

      // Backpropagating dendritic spike wave
      if (backpropPulseProgress >= 0.0) {
        backpropPulseProgress += dtMs * 0.004;
        if (backpropPulseProgress >= 1.0) backpropPulseProgress = -1.0;
      }

      // Update neurotransmitter particles in synaptic cleft
      for (let k = vesiclesReleased.length - 1; k >= 0; k--) {
        const v = vesiclesReleased[k];
        v.x += v.vx * dtMs;
        v.y += v.vy * dtMs;
        v.life -= dtMs * 0.0025;
        if (v.life <= 0 || v.x > synX + 16) {
          vesiclesReleased.splice(k, 1);
        }
      }

      // =========================================================================
      // 1. HERO ANIMATION: BIOLOGICAL NEURONS, AXON, BOUTON, CLEFT & PARTICLES
      // =========================================================================

      // --- Pre-synaptic Neuron Halos (Glowing Memory) ---
      const preHaloR = somaR + 10 + preTrace * 42;
      const gradPre = heroCtx.createRadialGradient(preX, somaY, somaR * 0.6, preX, somaY, preHaloR);
      gradPre.addColorStop(0, c.trace);
      gradPre.addColorStop(0.5, c.trace);
      gradPre.addColorStop(1, 'rgba(0,0,0,0)');
      heroCtx.save();
      heroCtx.globalAlpha = Math.min(0.5, 0.05 + preTrace * 0.45);
      heroCtx.fillStyle = gradPre;
      heroCtx.beginPath();
      heroCtx.arc(preX, somaY, preHaloR, 0, Math.PI * 2);
      heroCtx.fill();
      heroCtx.restore();

      // --- Post-synaptic Neuron Halos (Glowing Memory) ---
      const postHaloR = somaR + 10 + postTrace * 42;
      const gradPost = heroCtx.createRadialGradient(postX, somaY, somaR * 0.6, postX, somaY, postHaloR);
      gradPost.addColorStop(0, c.trace);
      gradPost.addColorStop(0.5, c.trace);
      gradPost.addColorStop(1, 'rgba(0,0,0,0)');
      heroCtx.save();
      heroCtx.globalAlpha = Math.min(0.5, 0.05 + postTrace * 0.45);
      heroCtx.fillStyle = gradPost;
      heroCtx.beginPath();
      heroCtx.arc(postX, somaY, postHaloR, 0, Math.PI * 2);
      heroCtx.fill();
      heroCtx.restore();

      // --- Pre-synaptic Dendrites (Branching leftwards from soma) ---
      heroCtx.save();
      heroCtx.strokeStyle = c.inkMuted;
      heroCtx.lineWidth = 1.6;
      heroCtx.lineCap = 'round';
      // Dendrite 1 (up-left)
      heroCtx.beginPath();
      heroCtx.moveTo(preX - somaR + 2, somaY - 8);
      heroCtx.quadraticCurveTo(preX - somaR - 22, somaY - 26, preX - somaR - 44, somaY - 32);
      heroCtx.moveTo(preX - somaR - 22, somaY - 26);
      heroCtx.lineTo(preX - somaR - 36, somaY - 14);
      heroCtx.stroke();
      // Dendrite 2 (mid-left)
      heroCtx.beginPath();
      heroCtx.moveTo(preX - somaR, somaY + 2);
      heroCtx.quadraticCurveTo(preX - somaR - 26, somaY + 4, preX - somaR - 48, somaY + 12);
      heroCtx.stroke();
      // Dendrite 3 (down-left)
      heroCtx.beginPath();
      heroCtx.moveTo(preX - somaR + 4, somaY + 10);
      heroCtx.quadraticCurveTo(preX - somaR - 18, somaY + 28, preX - somaR - 40, somaY + 36);
      heroCtx.moveTo(preX - somaR - 18, somaY + 28);
      heroCtx.lineTo(preX - somaR - 30, somaY + 44);
      heroCtx.stroke();
      heroCtx.restore();

      // --- Axon & Axon Hillock (Pre to Synapse) ---
      // Axon Hillock (smooth funnel from soma to axon)
      heroCtx.beginPath();
      heroCtx.moveTo(preX + somaR - 2, somaY - 7);
      heroCtx.quadraticCurveTo(preX + somaR + 10, somaY - 2, preX + somaR + 18, somaY - 2);
      heroCtx.lineTo(boutonX, somaY - 2);
      heroCtx.lineTo(boutonX, somaY + 2);
      heroCtx.lineTo(preX + somaR + 18, somaY + 2);
      heroCtx.quadraticCurveTo(preX + somaR + 10, somaY + 2, preX + somaR - 2, somaY + 7);
      heroCtx.fillStyle = c.border;
      heroCtx.fill();

      // Axon cable central core line
      heroCtx.beginPath();
      heroCtx.moveTo(preX + somaR, somaY);
      heroCtx.lineTo(boutonX, somaY);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 2;
      heroCtx.stroke();

      // Myelin Sheath Capsules along Axon with Nodes of Ranvier
      const axonStart = preX + somaR + 24;
      const axonLen = boutonX - axonStart - 10;
      const nSheaths = isMobile ? 2 : 3;
      const sheathW = Math.max(16, (axonLen - (nSheaths - 1) * 8) / nSheaths);
      for (let s = 0; s < nSheaths; s++) {
        const sx = axonStart + s * (sheathW + 8);
        heroCtx.save();
        heroCtx.fillStyle = c.paperCard;
        heroCtx.strokeStyle = c.border;
        heroCtx.lineWidth = 1.2;
        heroCtx.beginPath();
        heroCtx.roundRect(sx, somaY - 5.5, sheathW, 11, 4);
        heroCtx.fill();
        heroCtx.stroke();
        heroCtx.restore();
      }

      // Action Potential Travelling Pulse
      if (axonPulseProgress >= 0.0) {
        const pulseX = preX + somaR + (boutonX - (preX + somaR)) * axonPulseProgress;
        heroCtx.save();
        heroCtx.beginPath();
        heroCtx.arc(pulseX, somaY, 6, 0, Math.PI * 2);
        heroCtx.fillStyle = c.volt;
        heroCtx.shadowColor = c.volt;
        heroCtx.shadowBlur = 10;
        heroCtx.fill();
        heroCtx.restore();
      }

      // --- Terminal Bouton (Presynaptic bulb, scales with weight w) ---
      // Base radius scales with synaptic weight w
      const boutonR = 12 + currentWeight * 15;
      const boutonGrowthScale = 1.0 + plasticityPulse * (lastDwApplied >= 0 ? 0.25 : -0.15);
      const effectiveBoutonR = boutonR * boutonGrowthScale;

      heroCtx.save();
      // Bouton outer glow on LTP/LTD
      if (plasticityPulse > 0.05) {
        heroCtx.shadowColor = lastDwApplied >= 0 ? c.exc : c.inh;
        heroCtx.shadowBlur = 14 * plasticityPulse;
      }
      heroCtx.beginPath();
      // Teardrop bulb shape
      heroCtx.moveTo(boutonX - effectiveBoutonR * 0.8, somaY - 3);
      heroCtx.quadraticCurveTo(boutonX - effectiveBoutonR * 0.3, somaY - effectiveBoutonR, boutonX, somaY - effectiveBoutonR);
      heroCtx.arc(boutonX, somaY, effectiveBoutonR, -Math.PI / 2, Math.PI / 2);
      heroCtx.quadraticCurveTo(boutonX - effectiveBoutonR * 0.3, somaY + effectiveBoutonR, boutonX - effectiveBoutonR * 0.8, somaY + 3);
      heroCtx.closePath();
      heroCtx.fillStyle = plasticityPulse > 0.1 ? (lastDwApplied >= 0 ? c.exc : c.inh) : c.paperCard;
      heroCtx.fill();
      heroCtx.strokeStyle = lastDwApplied >= 0 && plasticityPulse > 0.05 ? c.exc : c.ink;
      heroCtx.lineWidth = 2;
      heroCtx.stroke();
      heroCtx.restore();

      // Synaptic Vesicles inside bouton
      heroCtx.save();
      const nVesicles = 12;
      for (let vi = 0; vi < nVesicles; vi++) {
        const angle = (vi / nVesicles) * Math.PI * 1.5 - Math.PI * 0.75;
        const rad = 4 + (vi % 3) * (effectiveBoutonR * 0.3);
        const vx = boutonX - 2 + Math.cos(angle) * rad;
        const vy = somaY + Math.sin(angle) * rad;
        heroCtx.beginPath();
        heroCtx.arc(vx, vy, 2.2, 0, Math.PI * 2);
        heroCtx.fillStyle = axonPulseProgress > 0.8 ? c.volt : c.inkMuted;
        heroCtx.fill();
      }
      heroCtx.restore();

      // --- Postsynaptic Dendritic Spine & Receptor Density ---
      const spineHeadX = synX + 12;
      const spineR = 14;
      heroCtx.save();

      // Dendritic shaft extending from spine to post soma
      heroCtx.beginPath();
      heroCtx.moveTo(spineHeadX + 18, somaY - 3.5);
      heroCtx.lineTo(postX - somaR, somaY - 3.5);
      heroCtx.lineTo(postX - somaR, somaY + 3.5);
      heroCtx.lineTo(spineHeadX + 18, somaY + 3.5);
      heroCtx.closePath();
      heroCtx.fillStyle = c.paperCard;
      heroCtx.fill();
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1.6;
      heroCtx.stroke();

      // Dendritic spines (small thorn-like spines along the shaft)
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1.4;
      for (let spX = spineHeadX + 45; spX < postX - somaR - 25; spX += 35) {
        heroCtx.beginPath();
        heroCtx.moveTo(spX, somaY - 3.5);
        heroCtx.lineTo(spX + 2, somaY - 9);
        heroCtx.moveTo(spX + 18, somaY + 3.5);
        heroCtx.lineTo(spX + 20, somaY + 9);
        heroCtx.stroke();
      }

      // Postsynaptic density (thick dark membrane band facing cleft)
      heroCtx.beginPath();
      heroCtx.arc(spineHeadX, somaY, spineR, Math.PI * 0.65, Math.PI * 1.35);
      heroCtx.strokeStyle = c.ink;
      heroCtx.lineWidth = 4;
      heroCtx.stroke();

      // Spine head mushroom shape
      heroCtx.beginPath();
      heroCtx.arc(spineHeadX, somaY, spineR, Math.PI * 0.55, Math.PI * 1.45);
      heroCtx.quadraticCurveTo(spineHeadX + 10, somaY - 4, spineHeadX + 18, somaY - 3.5);
      heroCtx.lineTo(spineHeadX + 18, somaY + 3.5);
      heroCtx.quadraticCurveTo(spineHeadX + 10, somaY + 4, spineHeadX, somaY + spineR);
      heroCtx.fillStyle = postFlash > 0.1 ? c.volt : c.paperCard;
      heroCtx.fill();
      heroCtx.strokeStyle = c.ink;
      heroCtx.lineWidth = 1.6;
      heroCtx.stroke();

      // Receptor channel teeth on PSD
      heroCtx.strokeStyle = c.exc;
      heroCtx.lineWidth = 1.8;
      for (let r = -8; r <= 8; r += 4) {
        heroCtx.beginPath();
        heroCtx.moveTo(spineHeadX - 2, somaY + r);
        heroCtx.lineTo(spineHeadX + 3, somaY + r);
        heroCtx.stroke();
      }
      heroCtx.restore();

      // Backpropagating Action Potential Glow along Post Dendrite
      if (backpropPulseProgress >= 0.0) {
        const bpX = postX - somaR - (postX - somaR - (spineHeadX + 10)) * backpropPulseProgress;
        heroCtx.save();
        heroCtx.beginPath();
        heroCtx.arc(bpX, somaY, 5, 0, Math.PI * 2);
        heroCtx.fillStyle = c.volt;
        heroCtx.shadowColor = c.volt;
        heroCtx.shadowBlur = 8;
        heroCtx.fill();
        heroCtx.restore();
      }

      // --- Neurotransmitter Particles in Synaptic Cleft ---
      for (const v of vesiclesReleased) {
        heroCtx.save();
        heroCtx.beginPath();
        heroCtx.arc(v.x, v.y, 2.5, 0, Math.PI * 2);
        heroCtx.fillStyle = c.exc;
        heroCtx.globalAlpha = Math.max(0.2, v.life);
        heroCtx.fill();
        heroCtx.restore();
      }

      // --- Pre-synaptic Soma & Nucleus ---
      heroCtx.save();
      heroCtx.beginPath();
      heroCtx.arc(preX, somaY, somaR, 0, Math.PI * 2);
      heroCtx.fillStyle = preFlash > 0.05 ? c.volt : c.paperCard;
      heroCtx.fill();
      heroCtx.strokeStyle = hoveredTerm === 'trace_pre' ? c.trace : c.ink;
      heroCtx.lineWidth = hoveredTerm === 'trace_pre' ? 3 : 2;
      heroCtx.stroke();
      // Pre Nucleus
      heroCtx.beginPath();
      heroCtx.arc(preX - 2, somaY - 1, somaR * 0.38, 0, Math.PI * 2);
      heroCtx.fillStyle = c.border;
      heroCtx.fill();
      heroCtx.beginPath();
      heroCtx.arc(preX - 1, somaY - 1, somaR * 0.16, 0, Math.PI * 2);
      heroCtx.fillStyle = c.inkMuted;
      heroCtx.fill();
      heroCtx.restore();

      heroCtx.font = '11px var(--font-serif)';
      heroCtx.fillStyle = c.ink;
      heroCtx.textAlign = 'center';
      heroCtx.fillText(isMobile ? 'Pre (Click)' : S.pre_soma, preX, somaY - somaR - 8);
      heroCtx.font = '10px var(--font-mono)';
      heroCtx.fillStyle = c.trace;
      heroCtx.fillText(`x_pre = ${preTrace.toFixed(3)}`, preX, somaY + somaR + 14);

      // --- Post-synaptic Soma & Dendritic Arbor ---
      // Post-synaptic radiating dendrites
      heroCtx.save();
      heroCtx.strokeStyle = c.inkMuted;
      heroCtx.lineWidth = 1.6;
      heroCtx.lineCap = 'round';
      // Dendrite up-right
      heroCtx.beginPath();
      heroCtx.moveTo(postX + somaR - 2, somaY - 8);
      heroCtx.quadraticCurveTo(postX + somaR + 24, somaY - 26, postX + somaR + 44, somaY - 32);
      heroCtx.moveTo(postX + somaR + 24, somaY - 26);
      heroCtx.lineTo(postX + somaR + 36, somaY - 14);
      heroCtx.stroke();
      // Dendrite mid-right
      heroCtx.beginPath();
      heroCtx.moveTo(postX + somaR, somaY + 2);
      heroCtx.quadraticCurveTo(postX + somaR + 26, somaY + 4, postX + somaR + 48, somaY + 12);
      heroCtx.stroke();
      // Dendrite down-right
      heroCtx.beginPath();
      heroCtx.moveTo(postX + somaR - 4, somaY + 10);
      heroCtx.quadraticCurveTo(postX + somaR + 20, somaY + 28, postX + somaR + 42, somaY + 36);
      heroCtx.stroke();
      heroCtx.restore();

      // Post Soma Body
      heroCtx.save();
      heroCtx.beginPath();
      heroCtx.arc(postX, somaY, somaR, 0, Math.PI * 2);
      heroCtx.fillStyle = postFlash > 0.05 ? c.volt : c.paperCard;
      heroCtx.fill();
      heroCtx.strokeStyle = hoveredTerm === 'trace_post' ? c.trace : c.ink;
      heroCtx.lineWidth = hoveredTerm === 'trace_post' ? 3 : 2;
      heroCtx.stroke();
      // Post Nucleus
      heroCtx.beginPath();
      heroCtx.arc(postX + 2, somaY - 1, somaR * 0.38, 0, Math.PI * 2);
      heroCtx.fillStyle = c.border;
      heroCtx.fill();
      heroCtx.beginPath();
      heroCtx.arc(postX + 1, somaY - 1, somaR * 0.16, 0, Math.PI * 2);
      heroCtx.fillStyle = c.inkMuted;
      heroCtx.fill();
      heroCtx.restore();

      heroCtx.font = '11px var(--font-serif)';
      heroCtx.fillStyle = c.ink;
      heroCtx.textAlign = 'center';
      heroCtx.fillText(isMobile ? 'Post (Click)' : S.post_soma, postX, somaY - somaR - 8);
      heroCtx.font = '10px var(--font-mono)';
      heroCtx.fillStyle = c.trace;
      heroCtx.fillText(`x_post = ${postTrace.toFixed(3)}`, postX, somaY + somaR + 14);

      // --- Synaptic Weight Readout Above Cleft ---
      heroCtx.save();
      heroCtx.font = 'bold 11px var(--font-mono)';
      heroCtx.fillStyle = plasticityPulse > 0.05 ? (lastDwApplied >= 0 ? c.exc : c.inh) : c.ink;
      heroCtx.textAlign = 'center';
      heroCtx.fillText(`${S.synaptic_weight} = ${currentWeight.toFixed(3)}`, synX, somaY - effectiveBoutonR - 7);

      // Floating Plasticity Change Badge (+Δw or -Δw)
      if (plasticityPulse > 0.05) {
        const floatY = Math.max(12, somaY - effectiveBoutonR - 20);
        heroCtx.fillStyle = lastDwApplied >= 0 ? c.exc : c.inh;
        heroCtx.font = 'bold 10px var(--font-mono)';
        heroCtx.fillText(
          `${lastDwApplied >= 0 ? '+LTP (+' : '-LTD ('}${lastDwApplied.toFixed(4)})`,
          synX,
          floatY
        );
      }
      heroCtx.restore();

      // =========================================================================
      // 2. TIMELINE OF TWO SPIKE MARKERS (DRAG EITHER MARKER)
      // =========================================================================
      heroCtx.save();
      // Timeline Header: Title on left, Tag on right
      heroCtx.font = '11px var(--font-serif)';
      heroCtx.fillStyle = c.ink;
      heroCtx.textAlign = 'left';
      heroCtx.fillText(isMobile ? 'Spike Timing' : 'Spike-Pair Timing (Drag Pre / Post)', layout.timelineX, layout.timelineY - 4);

      const tagText = `Δt = ${currentDeltaT > 0 ? '+' : ''}${currentDeltaT.toFixed(1)} ms (${currentDeltaT > 0 ? 'LTP' : currentDeltaT < 0 ? 'LTD' : 'Sync'})`;
      heroCtx.font = 'bold 10px var(--font-mono)';
      heroCtx.fillStyle = currentDeltaT > 0 ? c.exc : (currentDeltaT < 0 ? c.inh : c.ink);
      heroCtx.textAlign = 'right';
      heroCtx.fillText(tagText, layout.timelineX + layout.timelineW, layout.timelineY - 4);

      // Timeline track box
      heroCtx.fillStyle = c.paperCard;
      heroCtx.fillRect(layout.timelineX, layout.timelineY, layout.timelineW, layout.timelineH);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1;
      heroCtx.strokeRect(layout.timelineX, layout.timelineY, layout.timelineW, layout.timelineH);

      // Timeline axis line
      const tAxisY = layout.timelineY + 22;
      heroCtx.beginPath();
      heroCtx.moveTo(layout.timelineX, tAxisY);
      heroCtx.lineTo(layout.timelineX + layout.timelineW, tAxisY);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1.5;
      heroCtx.stroke();

      // Axis ticks (-40 to +40 ms)
      heroCtx.font = '9px var(--font-mono)';
      heroCtx.textAlign = 'center';
      const toTimelineX = (tMs) => layout.timelineX + ((tMs + 50.0) / 100.0) * layout.timelineW;
      for (let tk = -40; tk <= 40; tk += 20) {
        const tx = toTimelineX(tk);
        heroCtx.beginPath();
        heroCtx.moveTo(tx, tAxisY - 3);
        heroCtx.lineTo(tx, tAxisY + 3);
        heroCtx.strokeStyle = tk === 0 ? c.ink : c.border;
        heroCtx.lineWidth = tk === 0 ? 1.5 : 1;
        heroCtx.stroke();
        heroCtx.fillStyle = c.inkMuted;
        heroCtx.fillText(`${tk > 0 ? '+' : ''}${tk}`, tx, tAxisY + 16);
      }

      // Connecting Span on Axis between Pre and Post
      const pX1 = toTimelineX(tPreMarker);
      const pX2 = toTimelineX(tPostMarker);
      const minX = Math.min(pX1, pX2);
      const maxX = Math.max(pX1, pX2);

      if (Math.abs(currentDeltaT) > 0.1) {
        heroCtx.beginPath();
        heroCtx.moveTo(minX, tAxisY);
        heroCtx.lineTo(maxX, tAxisY);
        heroCtx.strokeStyle = currentDeltaT > 0 ? c.exc : c.inh;
        heroCtx.lineWidth = 2.5;
        heroCtx.stroke();
      }

      // Marker 1: Pre-spike marker (t_pre) - Positioned ABOVE axis
      heroCtx.beginPath();
      heroCtx.roundRect(pX1 - 22, tAxisY - 17, 44, 14, 3);
      heroCtx.fillStyle = c.ink;
      heroCtx.fill();
      heroCtx.strokeStyle = c.paper;
      heroCtx.lineWidth = 1.2;
      heroCtx.stroke();
      // Downward indicator notch to axis
      heroCtx.beginPath();
      heroCtx.moveTo(pX1 - 4, tAxisY - 3);
      heroCtx.lineTo(pX1 + 4, tAxisY - 3);
      heroCtx.lineTo(pX1, tAxisY);
      heroCtx.closePath();
      heroCtx.fillStyle = c.ink;
      heroCtx.fill();
      heroCtx.font = 'bold 9px var(--font-mono)';
      heroCtx.fillStyle = c.paper;
      heroCtx.fillText(`Pre:${tPreMarker > 0 ? '+' : ''}${tPreMarker.toFixed(0)}`, pX1, tAxisY - 7);

      // Marker 2: Post-spike marker (t_post) - Positioned BELOW axis
      heroCtx.beginPath();
      heroCtx.roundRect(pX2 - 24, tAxisY + 3, 48, 14, 3);
      heroCtx.fillStyle = currentDeltaT > 0 ? c.exc : (currentDeltaT < 0 ? c.inh : c.ink);
      heroCtx.fill();
      heroCtx.strokeStyle = c.paper;
      heroCtx.lineWidth = 1.2;
      heroCtx.stroke();
      // Upward indicator notch to axis
      heroCtx.beginPath();
      heroCtx.moveTo(pX2 - 4, tAxisY + 3);
      heroCtx.lineTo(pX2 + 4, tAxisY + 3);
      heroCtx.lineTo(pX2, tAxisY);
      heroCtx.closePath();
      heroCtx.fillStyle = currentDeltaT > 0 ? c.exc : (currentDeltaT < 0 ? c.inh : c.ink);
      heroCtx.fill();
      heroCtx.font = 'bold 9px var(--font-mono)';
      heroCtx.fillStyle = '#ffffff';
      heroCtx.fillText(`Post:${tPostMarker > 0 ? '+' : ''}${tPostMarker.toFixed(0)}`, pX2, tAxisY + 13);
      heroCtx.restore();

      // =========================================================================
      // 3. OSCILLOSCOPE TRACE STRIP & SAMPLING READOUT LINE (Req 2)
      // =========================================================================
      heroCtx.save();
      // Strip Box
      heroCtx.fillStyle = c.paperCard;
      heroCtx.fillRect(layout.stripX, layout.stripY, layout.stripW, layout.stripH);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1;
      heroCtx.strokeRect(layout.stripX, layout.stripY, layout.stripW, layout.stripH);

      // Subtle graticule grid lines
      heroCtx.save();
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 0.5;
      heroCtx.setLineDash([2, 4]);

      // Graph baseline
      const graphZeroY = layout.stripY + layout.stripH - 22;
      const graphTopY = layout.stripY + 22;
      const graphH = graphZeroY - graphTopY;

      for (let gy = graphTopY + graphH * 0.33; gy < graphZeroY; gy += graphH * 0.33) {
        heroCtx.beginPath();
        heroCtx.moveTo(layout.stripX + 6, gy);
        heroCtx.lineTo(layout.stripX + layout.stripW - 6, gy);
        heroCtx.stroke();
      }
      heroCtx.restore();

      // Strip Header Label
      heroCtx.font = '10px var(--font-serif)';
      heroCtx.fillStyle = c.ink;
      heroCtx.textAlign = 'left';
      heroCtx.fillText('Trace Memory & Readout', layout.stripX + 8, layout.stripY + 13);

      heroCtx.beginPath();
      heroCtx.moveTo(layout.stripX + 6, graphZeroY);
      heroCtx.lineTo(layout.stripX + layout.stripW - 6, graphZeroY);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1;
      heroCtx.stroke();

      // Relative time range: window of 100 ms (-50 to +50 ms relative to center)
      const toStripX = (tMs) => layout.stripX + 10 + ((tMs + 50.0) / 100.0) * (layout.stripW - 20);

      // Draw Pre Trace Curve x_pre(t)
      const sxPre = toStripX(tPreMarker);
      heroCtx.save();
      heroCtx.beginPath();
      // Spike tick at tPre
      heroCtx.moveTo(sxPre, graphZeroY);
      heroCtx.lineTo(sxPre, graphTopY);
      // Exponential decay
      for (let tStep = tPreMarker; tStep <= 50.0; tStep += 1.0) {
        const decayVal = Math.pow(0.95, tStep - tPreMarker);
        const px = toStripX(tStep);
        const py = graphZeroY - decayVal * graphH;
        heroCtx.lineTo(px, py);
      }
      heroCtx.strokeStyle = c.trace;
      heroCtx.lineWidth = 1.8;
      heroCtx.stroke();
      heroCtx.restore();

      // Draw Post Trace Curve x_post(t)
      const sxPost = toStripX(tPostMarker);
      heroCtx.save();
      heroCtx.beginPath();
      // Spike tick at tPost
      heroCtx.moveTo(sxPost, graphZeroY);
      heroCtx.lineTo(sxPost, graphTopY);
      // Exponential decay
      for (let tStep = tPostMarker; tStep <= 50.0; tStep += 1.0) {
        const decayVal = Math.pow(0.95, tStep - tPostMarker);
        const px = toStripX(tStep);
        const py = graphZeroY - decayVal * graphH;
        heroCtx.lineTo(px, py);
      }
      heroCtx.strokeStyle = c.volt;
      heroCtx.lineWidth = 1.5;
      heroCtx.setLineDash([3, 2]);
      heroCtx.stroke();
      heroCtx.restore();

      // Spike tick labels
      heroCtx.font = '9px var(--font-mono)';
      heroCtx.fillStyle = c.trace;
      heroCtx.textAlign = 'center';
      heroCtx.fillText('t_pre', sxPre, graphZeroY + 12);
      heroCtx.fillStyle = c.volt;
      heroCtx.fillText('t_post', sxPost, graphZeroY + 12);

      // --- CRITICAL READ-OUT LINE AT SECOND SPIKE (Req 2) ---
      if (currentDeltaT > 0) {
        // Pre fired before Post: Post spike arrives at sxPost and reads decaying Pre trace!
        const sampledTrace = Math.pow(0.95, currentDeltaT);
        const sampleY = graphZeroY - sampledTrace * graphH;
        const dwLtp = core.DEFAULT_A_PLUS * sampledTrace;

        // Vertical read-out line
        heroCtx.save();
        heroCtx.beginPath();
        heroCtx.setLineDash([2, 2]);
        heroCtx.moveTo(sxPost, graphZeroY);
        heroCtx.lineTo(sxPost, sampleY);
        heroCtx.strokeStyle = c.exc;
        heroCtx.lineWidth = 1.8;
        heroCtx.stroke();

        // Sample intercept dot
        heroCtx.beginPath();
        heroCtx.arc(sxPost, sampleY, 5, 0, Math.PI * 2);
        heroCtx.fillStyle = c.exc;
        heroCtx.fill();
        heroCtx.strokeStyle = c.ink;
        heroCtx.lineWidth = 1.5;
        heroCtx.stroke();

        // Header callout inside strip (compact, never wraps or clips)
        heroCtx.font = 'bold 9px var(--font-mono)';
        heroCtx.fillStyle = c.exc;
        heroCtx.textAlign = 'right';
        heroCtx.fillText(
          `Sampled: x_pre = ${sampledTrace.toFixed(3)} → Δw = +${dwLtp.toFixed(4)}`,
          layout.stripX + layout.stripW - 8,
          layout.stripY + 13
        );

        // Near the intercept point: short compact tag
        heroCtx.textAlign = sxPost > layout.stripX + layout.stripW * 0.7 ? 'right' : 'left';
        heroCtx.fillText(
          `x=${sampledTrace.toFixed(2)}`,
          sxPost + (sxPost > layout.stripX + layout.stripW * 0.7 ? -7 : 7),
          sampleY - 3
        );
        heroCtx.restore();
      } else if (currentDeltaT < 0) {
        // Post fired before Pre: Pre spike arrives at sxPre and reads decaying Post trace!
        const sampledTrace = Math.pow(0.95, -currentDeltaT);
        const sampleY = graphZeroY - sampledTrace * graphH;
        const dwLtd = -core.DEFAULT_A_MINUS * sampledTrace;

        // Vertical read-out line
        heroCtx.save();
        heroCtx.beginPath();
        heroCtx.setLineDash([2, 2]);
        heroCtx.moveTo(sxPre, graphZeroY);
        heroCtx.lineTo(sxPre, sampleY);
        heroCtx.strokeStyle = c.inh;
        heroCtx.lineWidth = 1.8;
        heroCtx.stroke();

        // Sample intercept dot
        heroCtx.beginPath();
        heroCtx.arc(sxPre, sampleY, 5, 0, Math.PI * 2);
        heroCtx.fillStyle = c.inh;
        heroCtx.fill();
        heroCtx.strokeStyle = c.ink;
        heroCtx.lineWidth = 1.5;
        heroCtx.stroke();

        // Header callout inside strip (compact, never wraps or clips)
        heroCtx.font = 'bold 9px var(--font-mono)';
        heroCtx.fillStyle = c.inh;
        heroCtx.textAlign = 'right';
        heroCtx.fillText(
          `Sampled: x_post = ${sampledTrace.toFixed(3)} → Δw = ${dwLtd.toFixed(4)}`,
          layout.stripX + layout.stripW - 8,
          layout.stripY + 13
        );

        // Near the intercept point: short compact tag
        heroCtx.textAlign = sxPre > layout.stripX + layout.stripW * 0.7 ? 'right' : 'left';
        heroCtx.fillText(
          `x=${sampledTrace.toFixed(2)}`,
          sxPre + (sxPre > layout.stripX + layout.stripW * 0.7 ? -7 : 7),
          sampleY - 3
        );
        heroCtx.restore();
      }
      heroCtx.restore();

      // =========================================================================
      // 4. STDP WINDOW PLOT (Δw vs Δt) & RUNNING WEIGHT DRIFT MINI PLOT (Req 4, 6)
      // =========================================================================

      // --- Top: STDP Window Plot ---
      heroCtx.save();
      // Plot Box Background
      heroCtx.fillStyle = c.paperCard;
      heroCtx.fillRect(layout.plotX, layout.plotY, layout.plotW, layout.plotH);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1;
      heroCtx.strokeRect(layout.plotX, layout.plotY, layout.plotW, layout.plotH);

      // Plot Header
      heroCtx.font = '10px var(--font-serif)';
      heroCtx.fillStyle = c.ink;
      heroCtx.textAlign = 'left';
      heroCtx.fillText(S.plot_window_title, layout.plotX + 6, layout.plotY + 11);

      // Transforms for plot: X in [-50, +50], Y in [-0.025, +0.025]
      const toPlotX = (dt) => layout.plotX + ((dt + 50.0) / 100.0) * layout.plotW;
      const toPlotY = (dw) => layout.plotY + layout.plotH * (0.5 - dw / 0.05);

      // Zero axes lines
      heroCtx.beginPath();
      heroCtx.moveTo(toPlotX(-50), toPlotY(0));
      heroCtx.lineTo(toPlotX(50), toPlotY(0));
      heroCtx.moveTo(toPlotX(0), layout.plotY);
      heroCtx.lineTo(toPlotX(0), layout.plotY + layout.plotH);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1.2;
      heroCtx.stroke();

      // Region corner labels (placed in corners to never collide with curve, Req 6)
      heroCtx.font = '9px var(--font-sans)';
      heroCtx.fillStyle = c.exc;
      heroCtx.textAlign = 'right';
      heroCtx.fillText(S.tag_ltp, layout.plotX + layout.plotW - 6, layout.plotY + 12);
      heroCtx.fillStyle = c.inh;
      heroCtx.textAlign = 'left';
      heroCtx.fillText(S.tag_ltd, layout.plotX + 6, layout.plotY + layout.plotH - 5);

      // Theory Ghost Curve: dw = a_plus * exp(-dt/tau_plus) for dt>0, -a_minus * exp(dt/tau_minus) for dt<0
      if (showTheory) {
        heroCtx.save();
        heroCtx.beginPath();
        heroCtx.setLineDash([3, 2.5]);
        heroCtx.strokeStyle = c.theory;
        heroCtx.lineWidth = 1.4;

        // Negative branch (dt < 0)
        let first = true;
        for (let dt = -50; dt <= -0.5; dt += 1) {
          const dwTheo = core.theory_dw(dt);
          const px = toPlotX(dt);
          const py = toPlotY(dwTheo);
          if (first) {
            heroCtx.moveTo(px, py);
            first = false;
          } else {
            heroCtx.lineTo(px, py);
          }
        }
        heroCtx.stroke();

        // Positive branch (dt > 0)
        heroCtx.beginPath();
        first = true;
        for (let dt = 0.5; dt <= 50; dt += 1) {
          const dwTheo = core.theory_dw(dt);
          const px = toPlotX(dt);
          const py = toPlotY(dwTheo);
          if (first) {
            heroCtx.moveTo(px, py);
            first = false;
          } else {
            heroCtx.lineTo(px, py);
          }
        }
        heroCtx.stroke();
        heroCtx.restore();
      }

      // Sweep points scatter
      for (const pt of sweepPoints) {
        const px = toPlotX(pt.dt);
        const py = toPlotY(pt.dw);
        heroCtx.beginPath();
        heroCtx.arc(px, py, pt.marker === 'exp60' ? 4 : 2, 0, Math.PI * 2);
        heroCtx.fillStyle = pt.dw >= 0 ? c.exc : c.inh;
        heroCtx.fill();
      }

      // Live Active Pair Point sitting directly on the theory curve
      const currentDw = core.stdp_pair_dw(currentDeltaT);
      const currPtX = toPlotX(currentDeltaT);
      const currPtY = toPlotY(currentDw);

      heroCtx.save();
      // Dashed crosshairs
      heroCtx.beginPath();
      heroCtx.setLineDash([2, 2]);
      heroCtx.moveTo(currPtX, toPlotY(0));
      heroCtx.lineTo(currPtX, currPtY);
      heroCtx.lineTo(toPlotX(0), currPtY);
      heroCtx.strokeStyle = currentDw >= 0 ? c.exc : c.inh;
      heroCtx.lineWidth = 1;
      heroCtx.stroke();

      // Glowing dot
      heroCtx.beginPath();
      heroCtx.arc(currPtX, currPtY, 5, 0, Math.PI * 2);
      heroCtx.fillStyle = currentDw >= 0 ? c.exc : c.inh;
      heroCtx.fill();
      heroCtx.strokeStyle = c.ink;
      heroCtx.lineWidth = 1.5;
      heroCtx.stroke();
      heroCtx.restore();

      // Clean coordinate readout badge in header (no text collision!)
      heroCtx.font = 'bold 9px var(--font-mono)';
      heroCtx.fillStyle = currentDw >= 0 ? c.exc : c.inh;
      heroCtx.textAlign = 'right';
      heroCtx.fillText(
        `Δt = ${currentDeltaT > 0 ? '+' : ''}${currentDeltaT.toFixed(1)} ms, Δw = ${currentDw > 0 ? '+' : ''}${currentDw.toFixed(4)}`,
        layout.plotX + layout.plotW - 6,
        layout.plotY + 23
      );
      heroCtx.restore();

      // --- Bottom: Running Weight Drift Mini Plot (Req 4) ---
      heroCtx.save();
      // Box Background
      heroCtx.fillStyle = c.paperCard;
      heroCtx.fillRect(layout.driftX, layout.driftY, layout.driftW, layout.driftH);
      heroCtx.strokeStyle = c.border;
      heroCtx.lineWidth = 1;
      heroCtx.strokeRect(layout.driftX, layout.driftY, layout.driftW, layout.driftH);

      // Plot Title
      heroCtx.font = '10px var(--font-serif)';
      heroCtx.fillStyle = c.ink;
      heroCtx.textAlign = 'left';
      heroCtx.fillText(S.plot_drift_title, layout.driftX + 6, layout.driftY + 11);

      // Coordinate transforms: X in [0, 60], Y in [0.0, 1.0]
      const toDriftX = (step) => layout.driftX + 26 + (step / 60.0) * (layout.driftW - 36);
      const toDriftY = (w) => layout.driftY + layout.driftH - 14 - w * (layout.driftH - 34);

      // Grid & Bounds lines
      heroCtx.save();
      heroCtx.beginPath();
      heroCtx.setLineDash([2, 2]);
      heroCtx.strokeStyle = c.border;
      // Baseline w = 0.50
      heroCtx.moveTo(layout.driftX + 20, toDriftY(0.5));
      heroCtx.lineTo(layout.driftX + layout.driftW - 8, toDriftY(0.5));
      // Max bound w = 1.0
      heroCtx.moveTo(layout.driftX + 20, toDriftY(1.0));
      heroCtx.lineTo(layout.driftX + layout.driftW - 8, toDriftY(1.0));
      // Min bound w = 0.0
      heroCtx.moveTo(layout.driftX + 20, toDriftY(0.0));
      heroCtx.lineTo(layout.driftX + layout.driftW - 8, toDriftY(0.0));
      heroCtx.stroke();
      heroCtx.restore();

      // Axis ticks and labels
      heroCtx.font = '8px var(--font-mono)';
      heroCtx.fillStyle = c.inkMuted;
      heroCtx.textAlign = 'right';
      heroCtx.fillText('1.0', layout.driftX + 20, toDriftY(1.0) + 3);
      heroCtx.fillText('0.5', layout.driftX + 20, toDriftY(0.5) + 3);
      heroCtx.fillText('0.0', layout.driftX + 20, toDriftY(0.0) + 3);

      heroCtx.textAlign = 'center';
      heroCtx.fillText('0', toDriftX(0), layout.driftY + layout.driftH - 3);
      heroCtx.fillText('30', toDriftX(30), layout.driftY + layout.driftH - 3);
      heroCtx.fillText('60', toDriftX(60), layout.driftY + layout.driftH - 3);

      // Draw Running Drift Line
      if (weightDriftHistory.length > 0) {
        heroCtx.beginPath();
        for (let i = 0; i < weightDriftHistory.length; i++) {
          const pt = weightDriftHistory[i];
          const px = toDriftX(pt.step);
          const py = toDriftY(pt.w);
          if (i === 0) heroCtx.moveTo(px, py);
          else heroCtx.lineTo(px, py);
        }
        heroCtx.strokeStyle = currentWeight > 0.501 ? c.exc : (currentWeight < 0.499 ? c.inh : c.ink);
        heroCtx.lineWidth = 2;
        heroCtx.stroke();

        // Current weight end dot
        const lastPt = weightDriftHistory[weightDriftHistory.length - 1];
        const lx = toDriftX(lastPt.step);
        const ly = toDriftY(lastPt.w);
        heroCtx.beginPath();
        heroCtx.arc(lx, ly, 4, 0, Math.PI * 2);
        heroCtx.fillStyle = currentWeight > 0.501 ? c.exc : (currentWeight < 0.499 ? c.inh : c.ink);
        heroCtx.fill();
        heroCtx.strokeStyle = c.ink;
        heroCtx.lineWidth = 1;
        heroCtx.stroke();

        // Current weight readout in drift header
        heroCtx.font = 'bold 9px var(--font-mono)';
        heroCtx.fillStyle = currentWeight > 0.501 ? c.exc : (currentWeight < 0.499 ? c.inh : c.ink);
        heroCtx.textAlign = 'right';
        heroCtx.fillText(`w = ${currentWeight.toFixed(3)} (${lastPt.step}/60)`, layout.driftX + layout.driftW - 6, layout.driftY + 11);
      }
      heroCtx.restore();

      // =========================================================================
      // 5. UPDATE MARGIN READOUTS
      // =========================================================================
      if (valDtEl) valDtEl.textContent = `${currentDeltaT > 0 ? '+' : ''}${currentDeltaT.toFixed(1)} ms`;
      if (valDwEl) valDwEl.textContent = `${currentDw > 0 ? '+' : ''}${currentDw.toFixed(4)}`;
      if (valXpreEl) valXpreEl.textContent = preTrace.toFixed(3);
      if (valXpostEl) valXpostEl.textContent = postTrace.toFixed(3);
      if (valWeightEl) valWeightEl.textContent = currentWeight.toFixed(3);

      const theoDw = core.theory_dw(currentDeltaT);
      const diffVal = Math.abs(currentDw - theoDw);
      if (valDiffEl) valDiffEl.textContent = diffVal.toFixed(4);

      heroCtx.restore();

      requestAnimationFrame(render);
    }

    if (typeof ctx.onFrame === 'function') {
      ctx.onFrame(() => render(performance.now()));
    }
    render(performance.now());
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
      id: 'Y01',
      title: {
        en: 'Traces and the STDP window, with a spike-pair builder'
      },
      source: 'live',
      mount
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
