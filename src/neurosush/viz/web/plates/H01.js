/**
 * Plate H01 - Sparse distributed representations: why two random patterns almost never match.
 *
 * A 64x32 grid of 2048 cells is an SDR: 40 cells (teal) are active in a stored pattern and 40
 * (amber) in a presented pattern. The plate counts their overlap, streams thousands of random
 * patterns into a histogram and draws the exact hypergeometric pmf as the theory ghost.
 * `live` math mirrors neurosush.htm.sdr.match_probability, overlap_count, union_match_probability
 * and expected_union_size.
 *
 * core is pure (no DOM, seeded mulberry32) and exported for node tests; mount() runs only in a
 * browser with the Atlas shell.
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------ strings
  const S = {
    title: 'Sparse distributed representations',
    ch1_title: 'A pattern is 40 lit cells',
    ch1_cap: '2048 cells, 40 active - 2% sparsity. Click any cell to light your own pattern.',
    ch2_title: 'Two random patterns share almost nothing',
    ch2_cap: 'Drop 40 random cells on the stored pattern. How many land on teal?',
    ch3_title: 'The exact distribution of overlap',
    ch3_cap: 'Thousands of random draws stream in (bars) under the exact hypergeometric pmf (ghost).',
    ch4_title: 'A match threshold of 20 is astronomically unlikely',
    ch4_cap: 'Drag the threshold to shade the tail. The chance of 20 shared bits by accident?',
    ch5_title: 'Noise robustness: lose half the bits',
    ch5_cap: 'Keep 28 of 40 stored bits: the cell still fires (overlap 28 >= 20).',
    ch6_title: 'Unions: how many patterns fit?',
    ch6_cap: 'OR together M stored patterns. The union gets denser and false matches rise.',
    predict_q2: 'How many of the 40 amber cells land on teal?',
    predict_q4: 'Chance a random pattern shares >= 20 bits by accident?',
    stored: 'stored',
    presented: 'presented',
    shared: 'shared',
    overlap: 'Overlap',
    active: 'Active',
    threshold: 'Threshold theta',
    chance: 'Chance overlap',
    tailLabel: 'P(overlap >= theta)',
    difference: 'Ghost difference',
    recognized: 'recognized (>= theta)',
    notRecognized: 'below threshold',
    controls: 'Controls',
    drawRandom: 'Draw random pattern',
    draw100: 'Draw 100 random',
    unionM: 'Union size M',
    unionActivation: 'Union activation p',
    unionSize: 'Union cells',
    unionP: 'P(match union)',
    unionNote: 'union uses the binomial approximation of Ahmad & Hawkins 2016',
    nLabel: 'n cells',
    wLabel: 'w active',
    keepLabel: 'Noise: keep bits',
    glossary: 'Glossary',
    g_sdr: 'SDR: a sparse binary code - few 1s among many 0s.',
    g_sparsity: 'sparsity: active fraction w/n.',
    g_overlap: 'overlap: bits active in both patterns.',
    g_theta: 'theta: dendritic threshold for a match.',
    g_false: 'false match: overlap >= theta by chance.',
    g_union: 'union: OR of many stored patterns.',
    footer: 'live - mirrors neurosush.htm.sdr.match_probability, union_match_probability, expected_union_size',
    stepWord: 'Step'
  };

  // ---------------------------------------------------------------------- core
  /**
   * Deterministic PRNG (mulberry32). Returns a function producing floats in [0,1).
   * @param {number} seed
   * @returns {() => number}
   */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const LOG_FACT = [0];
  /**
   * Natural log of k! by cumulative sums (memoized).
   * @param {number} k
   * @returns {number}
   */
  function logFactorial(k) {
    if (!Number.isInteger(k) || k < 0) {
      throw new Error(`k must be a non-negative integer, got ${k}`);
    }
    for (let i = LOG_FACT.length; i <= k; i++) {
      LOG_FACT[i] = LOG_FACT[i - 1] + Math.log(i);
    }
    return LOG_FACT[k];
  }

  /**
   * Natural log of the binomial coefficient C(n, k); -Infinity outside range.
   * @param {number} n
   * @param {number} k
   * @returns {number}
   */
  function logComb(n, k) {
    if (k < 0 || k > n) return -Infinity;
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
  }

  /**
   * Exact pmf of the overlap between a fixed w_x-bit SDR and a random w-bit SDR.
   * P(B=b) = C(w_x,b) C(n-w_x, w-b) / C(n, w), b in [max(0, w+w_x-n), min(w, w_x)].
   * @param {number} n
   * @param {number} wX
   * @param {number} w
   * @returns {{b:number, p:number}[]}
   */
  function overlapPmf(n, wX, w) {
    if (!Number.isInteger(n) || !Number.isInteger(w) || !Number.isInteger(wX)) {
      throw new Error(`n, w, wX must be integers, got n=${n}, w=${w}, wX=${wX}`);
    }
    if (n < 1 || w < 0 || w > n || wX < 0 || wX > n) {
      throw new Error(`need n >= 1 and 0 <= w,wX <= n, got n=${n}, w=${w}, wX=${wX}`);
    }
    const bMin = Math.max(0, w + wX - n);
    const bMax = Math.min(w, wX);
    const logDen = logComb(n, w);
    const raw = [];
    let sum = 0;
    for (let b = bMin; b <= bMax; b++) {
      const lp = logComb(wX, b) + logComb(n - wX, w - b) - logDen;
      const p = lp === -Infinity ? 0 : Math.exp(lp);
      raw.push({ b, p });
      sum += p;
    }
    return raw.map((o) => ({ b: o.b, p: sum > 0 ? o.p / sum : 0 }));
  }

  /**
   * Exact pmf of the overlap between two random w-bit SDRs (normalized).
   * @param {number} n
   * @param {number} w
   * @returns {{b:number, p:number}[]}
   */
  function sdrPmf(n, w) {
    return overlapPmf(n, w, w);
  }

  /**
   * Exact probability that a random w-bit SDR overlaps a fixed w_x-bit SDR in >= theta bits.
   * Mirrors neurosush.htm.sdr.match_probability.
   * @param {number} n
   * @param {number} wX
   * @param {number} w
   * @param {number} theta
   * @returns {number}
   */
  function matchProbability(n, wX, w, theta) {
    if (!Number.isInteger(theta) || theta < 0) {
      throw new Error(`theta must be a non-negative integer, got ${theta}`);
    }
    let tail = 0;
    for (const { b, p } of overlapPmf(n, wX, w)) {
      if (b >= theta) tail += p;
    }
    return tail;
  }

  /**
   * Exact tail probability for two w-bit SDRs sharing >= theta bits.
   * @param {number} n
   * @param {number} w
   * @param {number} theta
   * @returns {number}
   */
  function sdrTail(n, w, theta) {
    return matchProbability(n, w, w, theta);
  }

  /**
   * Expected overlap of two random w-bit SDRs, w^2 / n.
   * @param {number} n
   * @param {number} w
   * @returns {number}
   */
  function expectedOverlap(n, w) {
    if (n < 1) throw new Error(`n must be positive, got ${n}`);
    return (w * w) / n;
  }

  /**
   * Probability a bit is active in the union of m random w-bit SDRs, 1-(1-w/n)^m.
   * @param {number} n
   * @param {number} w
   * @param {number} m
   * @returns {number}
   */
  function unionActivation(n, w, m) {
    if (!Number.isInteger(m) || m < 0) {
      throw new Error(`m must be a non-negative integer, got ${m}`);
    }
    return 1 - Math.pow(1 - w / n, m);
  }

  /**
   * Expected number of active bits in the union of m random w-bit SDRs, n*(1-(1-w/n)^m).
   * Mirrors neurosush.htm.sdr.expected_union_size.
   * @param {number} n
   * @param {number} w
   * @param {number} m
   * @returns {number}
   */
  function unionExpectedSize(n, w, m) {
    return n * unionActivation(n, w, m);
  }

  /**
   * Approximate probability a random w-bit SDR matches the union of m stored ones.
   * Overlap is Binomial(w, p) with p = 1-(1-w/n)^m (Ahmad & Hawkins 2016).
   * Mirrors neurosush.htm.sdr.union_match_probability.
   * @param {number} n
   * @param {number} w
   * @param {number} m
   * @param {number} theta
   * @returns {number}
   */
  function unionMatchProbability(n, w, m, theta) {
    if (!Number.isInteger(theta) || theta < 0) {
      throw new Error(`theta must be a non-negative integer, got ${theta}`);
    }
    const lo = Math.max(theta, 0);
    const p = unionActivation(n, w, m);
    if (p <= 0) return lo === 0 ? 1 : 0;
    if (p >= 1) return lo <= w ? 1 : 0;
    let total = 0;
    for (let b = lo; b <= w; b++) {
      total += Math.exp(logComb(w, b) + b * Math.log(p) + (w - b) * Math.log(1 - p));
    }
    return total;
  }

  /**
   * Sample the overlap of one random w-bit SDR with a fixed w-bit SDR (hypergeometric).
   * Draws w distinct indices uniformly from [0,n) and counts hits in [0,w).
   * @param {number} n
   * @param {number} w
   * @param {() => number} rng
   * @returns {number}
   */
  function sampleOverlap(n, w, rng) {
    const seen = new Set();
    let hits = 0;
    let drawn = 0;
    while (drawn < w) {
      const x = Math.floor(rng() * n);
      if (seen.has(x)) continue;
      seen.add(x);
      drawn++;
      if (x < w) hits++;
    }
    return hits;
  }

  /**
   * Deterministically sample `trials` overlaps.
   * @param {number} n
   * @param {number} w
   * @param {number} trials
   * @param {number} seed
   * @returns {number[]}
   */
  function sampleOverlaps(n, w, trials, seed) {
    const rng = mulberry32(seed);
    const out = new Array(trials);
    for (let i = 0; i < trials; i++) out[i] = sampleOverlap(n, w, rng);
    return out;
  }

  /**
   * Arithmetic mean of an array.
   * @param {number[]} xs
   * @returns {number}
   */
  function mean(xs) {
    if (xs.length === 0) return 0;
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += xs[i];
    return s / xs.length;
  }

  /**
   * Counts of each value 0..maxB in `samples`.
   * @param {number[]} samples
   * @param {number} maxB
   * @returns {number[]}
   */
  function histogram(samples, maxB) {
    const counts = new Array(maxB + 1).fill(0);
    for (const s of samples) {
      if (s >= 0 && s <= maxB) counts[s]++;
    }
    return counts;
  }

  /**
   * A random pattern with exactly w active bits, drawn without replacement (canonical order).
   * @param {number} n
   * @param {number} w
   * @param {() => number} rng
   * @returns {number[]}
   */
  function randomPattern(n, w, rng) {
    if (!Number.isInteger(w) || w < 0 || w > n) {
      throw new Error(`need 0 <= w <= n, got n=${n}, w=${w}`);
    }
    const set = new Set();
    while (set.size < w) set.add(Math.floor(rng() * n));
    return Array.from(set).sort((a, b) => a - b);
  }

  /**
   * A noisy copy of a stored w-bit pattern: keep the first `keep` stored bits and refill the
   * rest with fresh inactive bits, so the overlap with the stored set is exactly `keep`.
   * @param {number[]} stored
   * @param {number} n
   * @param {number} w
   * @param {number} keep
   * @param {() => number} rng
   * @returns {number[]}
   */
  function noisyPattern(stored, n, w, keep, rng) {
    if (!Array.isArray(stored) || stored.length !== w) {
      throw new Error(`stored must have exactly w=${w} bits, got ${stored && stored.length}`);
    }
    const storedSet = new Set(stored);
    const k = Math.max(0, Math.min(w, Math.floor(keep)));
    const set = new Set();
    for (let i = 0; i < k; i++) set.add(stored[i]);
    while (set.size < w) {
      const x = Math.floor(rng() * n);
      if (storedSet.has(x) || set.has(x)) continue;
      set.add(x);
    }
    return Array.from(set).sort((a, b) => a - b);
  }

  /**
   * Scientific notation with a fixed number of mantissa digits, e.g. 2.49e-26.
   * @param {number} x
   * @param {number} digits
   * @returns {string}
   */
  function formatSci(x, digits = 2) {
    if (typeof x !== 'number' || Number.isNaN(x)) return '\u2014';
    if (!Number.isFinite(x)) return x > 0 ? '\u221e' : '-\u221e';
    if (x === 0) return '0';
    const ax = Math.abs(x);
    if (ax >= 0.01 && ax < 1000) return String(Number(x.toPrecision(3)));
    const e = Math.floor(Math.log10(ax));
    const m = x / Math.pow(10, e);
    return `${m.toFixed(digits)}e${e >= 0 ? '+' : '-'}${Math.abs(e)}`;
  }

  const core = {
    mulberry32,
    logFactorial,
    logComb,
    overlapPmf,
    sdrPmf,
    matchProbability,
    sdrTail,
    expectedOverlap,
    unionActivation,
    unionExpectedSize,
    unionMatchProbability,
    sampleOverlap,
    sampleOverlaps,
    mean,
    histogram,
    randomPattern,
    noisyPattern,
    formatSci
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = { core };

  // ------------------------------------------------------------------------ ui
  const Atlas = typeof globalThis !== 'undefined' && globalThis.Atlas ? globalThis.Atlas : null;
  const N_COLS = 64;
  const N_CHOICES = [512, 1024, 2048, 4096];
  const MC_CAP = 20000;

  const CHAPTERS = [
    { id: 'stored', mode: 'stored', titleKey: 'ch1_title', capKey: 'ch1_cap' },
    { id: 'random', mode: 'random', titleKey: 'ch2_title', capKey: 'ch2_cap', predict: 'overlap' },
    { id: 'dist', mode: 'stream', titleKey: 'ch3_title', capKey: 'ch3_cap' },
    { id: 'threshold', mode: 'stream', titleKey: 'ch4_title', capKey: 'ch4_cap', predict: 'threshold' },
    { id: 'noise', mode: 'noise', titleKey: 'ch5_title', capKey: 'ch5_cap' },
    { id: 'union', mode: 'union', titleKey: 'ch6_title', capKey: 'ch6_cap' }
  ];

  function withAlpha(hex, a) {
    if (typeof hex !== 'string') return hex;
    let h = hex.trim().replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    if (h.length !== 6 || /[^0-9a-fA-F]/.test(h)) return hex;
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  function mount(rootEl, ctx) {
    const doc = rootEl.ownerDocument || document;
    const colors = ctx.colors;

    // The shell's live trace has no steps; keep its clock finite so the timeline stays calm.
    try {
      if (ctx.clock) {
        ctx.clock.steps = 1;
        ctx.clock.t = 0;
      }
    } catch (err) {
      /* clock is shell-owned; ignore */
    }

    const reduced = typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : false;

    const state = {
      n: 2048,
      w: 40,
      theta: 20,
      keep: 40,
      unionM: 1,
      chapter: 0,
      mode: CHAPTERS[0].mode,
      seed: 20260101,
      stored: [],
      presented: null, // Set of active indices, or null (nothing presented)
      storedSet: new Set(),
      unionPatterns: [],
      unionSet: new Set(),
      mcCounts: new Array(41).fill(0),
      mcTotal: 0,
      mcSum: 0,
      mcRng: core.mulberry32(20260418),
      pmfFull: [],
      showTheory: true,
      highlight: null,
      predict: null,
      flash: -1e9,
      lastStream: 0,
      draggingTheta: false
    };

    // ---- style ----
    const style = doc.createElement('style');
    style.textContent = `
.h01-overlay{position:absolute;inset:0;pointer-events:none;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;font-family:var(--font-sans)}
@media (max-width:700px){.h01-overlay{justify-content:center}}
.h01-btn{font-family:var(--font-sans);font-size:12px;color:var(--ink);background:var(--paper-inset);border:1px solid var(--border);border-radius:var(--radius);padding:4px 9px;cursor:pointer}
.h01-btn:hover{background:var(--paper-card)}
.h01-btn:focus-visible{outline:2px solid var(--focus-ring);outline-offset:1px}
.h01-btn[disabled]{opacity:.45;cursor:default}
.h01-step{font-family:var(--font-mono);font-size:11px;color:var(--ink-muted);white-space:nowrap}
.h01-title{font-family:var(--font-serif);font-weight:600;font-size:13.5px;color:var(--ink)}
.h01-cap{font-family:var(--font-serif);font-size:12.5px;color:var(--ink-muted);line-height:1.45}
.h01-predict{pointer-events:auto;max-width:420px;margin:0 12px 10px;padding:12px 14px;background:var(--paper-elevated);border:1px solid var(--border);border-radius:var(--radius);box-shadow:0 4px 16px rgba(0,0,0,.22)}
.h01-predict[hidden]{display:none}
.h01-pq{font-family:var(--font-serif);font-size:13.5px;margin-bottom:8px}
.h01-choices{display:flex;flex-wrap:wrap;gap:6px}
.h01-reveal{margin-top:8px;font-family:var(--font-mono);font-size:12px;color:var(--ink)}
.h01-ghost{color:var(--ink-muted);font-family:var(--font-mono);font-size:12px}
.h01-mini{font-family:var(--font-mono);font-size:11px}
`;
    doc.head.appendChild(style);

    // ---- canvas ----
    const canvas = ctx.canvas();
    canvas.style.touchAction = 'none';
    const g = canvas.getContext('2d');

    // ---- overlay DOM ----
    const overlay = doc.createElement('div');
    overlay.className = 'h01-overlay';
    overlay.innerHTML = `<div class="h01-predict" id="h01-predict" hidden></div>`;
    rootEl.appendChild(overlay);

    const predictEl = overlay.querySelector('#h01-predict');

    // ---- margin ----
    ctx.margin.innerHTML = `
      <div class="atlas-margin-section">
        <div class="atlas-margin-label" id="h01-cstep"></div>
        <div class="h01-title" id="h01-ctitle"></div>
        <div class="h01-cap" id="h01-ccap"></div>
        <div style="display:flex;gap:6px;margin-top:6px;align-items:center">
          <button class="h01-btn" id="h01-back" aria-label="Previous chapter">\u2039 Back</button>
          <button class="h01-btn" id="h01-next" aria-label="Next chapter" style="margin-left:auto">Next \u203a</button>
        </div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.overlap}</div>
        <div class="atlas-margin-value" id="h01-overlap">0 / 40</div>
        <div class="h01-mini" id="h01-status" style="color:var(--ink-muted)"></div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.threshold}</div>
        <input type="range" id="h01-theta" min="0" max="40" step="1" value="20" style="width:100%">
        <div class="atlas-margin-value" id="h01-theta-val">20</div>
        <div class="atlas-margin-label">${S.tailLabel}</div>
        <div class="atlas-margin-value" id="h01-tail">-</div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.chance}</div>
        <div class="atlas-margin-value" id="h01-mean">-</div>
        <div class="h01-mini" id="h01-mcmean" style="color:var(--ink-muted)"></div>
        <div class="atlas-margin-label" style="margin-top:4px">${S.difference}</div>
        <div class="h01-mini" id="h01-diff" style="color:var(--ink-muted)"></div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">Governing equation</div>
        <div class="atlas-equation-box">
P(B \u2265 <span class="h01-term" data-term="theta">\u03b8</span>) =
\u03a3<sub>b\u2265\u03b8</sub>
<span class="h01-term" data-term="w">C(w,b)</span>\u00b7<span class="h01-term" data-term="n">C(n\u2212w, w\u2212b)</span>
/ <span class="h01-term" data-term="n">C(n,w)</span>
<br><span id="h01-eq"></span>
        </div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.controls}</div>
        <label class="h01-mini">${S.nLabel} <span id="h01-n-val" class="atlas-mono">2048</span></label>
        <input type="range" id="h01-n" min="0" max="3" step="1" value="2" style="width:100%">
        <label class="h01-mini">${S.wLabel} <span id="h01-w-val" class="atlas-mono">40</span></label>
        <input type="range" id="h01-w" min="5" max="160" step="5" value="40" style="width:100%">
        <label class="h01-mini">${S.keepLabel} <span id="h01-keep-val" class="atlas-mono">40</span></label>
        <input type="range" id="h01-keep" min="0" max="40" step="1" value="40" style="width:100%">
        <label class="h01-mini">${S.unionM} <span id="h01-m-val" class="atlas-mono">1</span></label>
        <input type="range" id="h01-m" min="1" max="200" step="1" value="1" style="width:100%">
        <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap">
          <button class="h01-btn" id="h01-draw">${S.drawRandom}</button>
          <button class="h01-btn" id="h01-draw100">${S.draw100}</button>
        </div>
        <div class="h01-mini" id="h01-union-readout" style="color:var(--ink-muted);margin-top:6px"></div>
      </div>
      <div class="atlas-margin-section">
        <div class="atlas-margin-label">${S.glossary}</div>
        <div class="h01-mini" style="color:var(--ink-muted);line-height:1.6">
          ${S.g_sdr}<br>${S.g_sparsity}<br>${S.g_overlap}<br>${S.g_theta}<br>${S.g_false}<br>${S.g_union}
        </div>
      </div>
      <div class="atlas-margin-section">
        <div class="h01-mini" style="color:var(--ink-muted)">${S.footer}</div>
      </div>`;

    const $ = (id) => doc.getElementById(id);
    const titleEl = $('h01-ctitle');
    const capEl = $('h01-ccap');
    const stepEl = $('h01-cstep');
    const backBtn = $('h01-back');
    const nextBtn = $('h01-next');
    const overlapEl = $('h01-overlap');
    const statusEl = $('h01-status');
    const thetaEl = $('h01-theta');
    const thetaValEl = $('h01-theta-val');
    const tailEl = $('h01-tail');
    const meanEl = $('h01-mean');
    const mcMeanEl = $('h01-mcmean');
    const diffEl = $('h01-diff');
    const eqEl = $('h01-eq');
    const nValEl = $('h01-n-val');
    const wValEl = $('h01-w-val');
    const keepValEl = $('h01-keep-val');
    const mValEl = $('h01-m-val');
    const unionReadout = $('h01-union-readout');

    // equation terms hover -> figure highlight
    ctx.margin.querySelectorAll('[data-term]').forEach((el) => {
      el.style.cursor = 'help';
      el.addEventListener('mouseenter', () => { state.highlight = el.dataset.term; });
      el.addEventListener('mouseleave', () => { state.highlight = null; });
    });

    // ---- pattern bookkeeping ----
    function rebuildPmf() {
      state.pmfFull = new Array(state.w + 1).fill(0);
      for (const { b, p } of core.sdrPmf(state.n, state.w)) state.pmfFull[b] = p;
      state.theta = Math.min(state.theta, state.w);
      state.keep = Math.min(state.keep, state.w);
    }

    function rebuildPatterns() {
      state.stored = core.randomPattern(state.n, state.w, core.mulberry32(state.seed));
      state.storedSet = new Set(state.stored);
      rebuildPmf();
      state.presented = null;
      state.flash = performance.now();
      regeneratePresented(true);
    }

    function referenceSet() {
      return state.mode === 'union' ? state.unionSet : state.storedSet;
    }

    function overlapCount(set) {
      if (!set) return 0;
      const ref = referenceSet();
      let c = 0;
      for (const i of set) if (ref.has(i)) c++;
      return c;
    }

    function setFlash() {
      state.flash = performance.now();
    }

    function regeneratePresented(resetMc) {
      const mode = state.mode;
      if (resetMc) {
        state.mcCounts = new Array(state.w + 1).fill(0);
        state.mcTotal = 0;
        state.mcSum = 0;
        state.mcRng = core.mulberry32(20260418);
      }
      if (mode === 'stored') {
        state.presented = new Set();
      } else if (mode === 'random') {
        state.presented = new Set(core.randomPattern(state.n, state.w, core.mulberry32(state.seed + 7)));
      } else if (mode === 'stream') {
        state.presented = new Set(core.randomPattern(state.n, state.w, state.mcRng));
      } else if (mode === 'noise') {
        state.presented = new Set(core.noisyPattern(state.stored, state.n, state.w, state.keep, core.mulberry32(state.seed + 11)));
      } else if (mode === 'union') {
        state.presented = new Set(core.randomPattern(state.n, state.w, core.mulberry32(state.seed + 13)));
        rebuildUnion();
      }
      // Seed the histogram so the distribution is readable at once, then keep streaming new draws.
      if (mode === 'stream') addSamples(1500);
      setFlash();
    }

    function rebuildUnion() {
      const need = state.unionM;
      while (state.unionPatterns.length < need) {
        const rng = core.mulberry32(state.seed + 1000 + state.unionPatterns.length);
        state.unionPatterns.push(core.randomPattern(state.n, state.w, rng));
      }
      const u = new Set();
      for (let i = 0; i < need; i++) for (const b of state.unionPatterns[i]) u.add(b);
      state.unionSet = u;
    }

    function addSamples(k) {
      const rng = state.mcRng;
      for (let i = 0; i < k && state.mcTotal < MC_CAP; i++) {
        const b = core.sampleOverlap(state.n, state.w, rng);
        if (b < state.mcCounts.length) state.mcCounts[b]++;
        state.mcTotal++;
        state.mcSum += b;
      }
    }

    // ---- chapter handling ----
    function applyChapter(i) {
      state.chapter = Math.max(0, Math.min(CHAPTERS.length - 1, i));
      const ch = CHAPTERS[state.chapter];
      state.mode = ch.mode;
      if (ch.id === 'threshold') state.theta = 20;
      if (ch.id === 'noise') state.keep = 28;
      if (ch.id === 'union') state.unionM = Math.max(state.unionM, 8);
      syncControls();
      regeneratePresented(true);
      state.predict = ch.predict ? { kind: ch.predict, choice: null } : null;
      renderPredict();
      updateChapterBar();
    }

    function updateChapterBar() {
      const ch = CHAPTERS[state.chapter];
      titleEl.textContent = S[ch.titleKey] || S.title;
      capEl.textContent = S[ch.capKey] || '';
      stepEl.textContent = `${S.stepWord} ${state.chapter + 1}/${CHAPTERS.length}`;
      backBtn.disabled = state.chapter === 0;
      nextBtn.disabled = state.chapter === CHAPTERS.length - 1;
    }

    function renderPredict() {
      const p = state.predict;
      if (!p) {
        predictEl.hidden = true;
        return;
      }
      predictEl.hidden = false;
      const isQ2 = p.kind === 'overlap';
      const q = isQ2 ? S.predict_q2 : S.predict_q4;
      const choices = isQ2 ? ['0', '1', '20', '40'] : ['\u2248 1 in 10', '\u2248 1 in 10\u2076', '\u2248 1 in 10\u00b2\u2076', '\u2248 1 in 10\u2074\u2070'];
      if (p.choice === null || p.choice === undefined) {
        predictEl.innerHTML = `<div class="h01-pq">${q}</div><div class="h01-choices">` +
          choices.map((c, i) => `<button class="h01-btn" data-choice="${i}">${c}</button>`).join('') + `</div>`;
        predictEl.querySelectorAll('[data-choice]').forEach((b) => {
          b.addEventListener('click', () => {
            state.predict.choice = Number(b.dataset.choice);
            renderPredict();
          });
        });
      } else {
        let truth;
        if (isQ2) {
          truth = `Mean overlap = w\u00b2/n = ${core.expectedOverlap(state.n, state.w).toFixed(2)} bits. ` +
            `This draw: ${overlapCount(state.presented)} of ${state.w}.`;
          predictEl.innerHTML = `<div class="h01-pq">${q}</div>` +
            `<div class="h01-reveal">The chance overlap is tiny: <b>${truth}</b></div>` +
            `<div class="h01-ghost" style="margin-top:6px">Your answer: ${choices[p.choice]}</div>`;
        } else {
          const tail = core.sdrTail(state.n, state.w, state.theta);
          const inv = tail > 0 ? (1 / tail) : Infinity;
          truth = `P(overlap \u2265 ${state.theta}) = ${core.formatSci(tail)} \u2248 1 in ${core.formatSci(inv, 1)}.`;
          predictEl.innerHTML = `<div class="h01-pq">${q}</div>` +
            `<div class="h01-reveal">${truth} A 2048-bit population holds nothing like 10\u00b2\u2076 patterns.</div>` +
            `<div class="h01-ghost" style="margin-top:6px">Your answer: ${choices[p.choice]}</div>`;
        }
      }
    }

    function syncControls() {
      thetaEl.value = String(state.theta);
      nValEl.textContent = String(state.n);
      wValEl.textContent = String(state.w);
      keepValEl.textContent = String(state.keep);
      mValEl.textContent = String(state.unionM);
      thetaValEl.textContent = String(state.theta);
    }

    // ---- controls wiring ----
    thetaEl.addEventListener('input', () => {
      state.theta = Number(thetaEl.value);
      thetaValEl.textContent = String(state.theta);
    });
    $('h01-n').addEventListener('input', (e) => {
      state.n = N_CHOICES[Number(e.target.value)];
      if (state.n % N_COLS !== 0) state.n = 2048;
      state.unionPatterns = [];
      rebuildPatterns();
      syncControls();
    });
    $('h01-w').addEventListener('input', (e) => {
      state.w = Number(e.target.value);
      $('h01-m').max = String(Math.min(200, Math.max(1, state.w)));
      if (state.keep > state.w) state.keep = state.w;
      rebuildPatterns();
      syncControls();
    });
    $('h01-keep').addEventListener('input', (e) => {
      state.keep = Number(e.target.value);
      if (state.mode === 'noise') {
        state.presented = new Set(core.noisyPattern(state.stored, state.n, state.w, state.keep, core.mulberry32(state.seed + 11)));
        setFlash();
      }
      keepValEl.textContent = String(state.keep);
    });
    $('h01-m').addEventListener('input', (e) => {
      state.unionM = Number(e.target.value);
      mValEl.textContent = String(state.unionM);
      if (state.mode === 'union') rebuildUnion();
    });
    $('h01-draw').addEventListener('click', () => {
      state.seed = (state.seed * 1103515245 + 12345) >>> 0;
      regeneratePresented(false);
    });
    $('h01-draw100').addEventListener('click', () => {
      addSamples(100);
      if (state.mode === 'random' || state.mode === 'noise') {
        state.seed = (state.seed * 1103515245 + 12345) >>> 0;
        state.presented = new Set(state.mode === 'noise'
          ? core.noisyPattern(state.stored, state.n, state.w, state.keep, core.mulberry32(state.seed + 11))
          : core.randomPattern(state.n, state.w, core.mulberry32(state.seed + 7)));
        setFlash();
      }
    });

    backBtn.addEventListener('click', () => applyChapter(state.chapter - 1));
    nextBtn.addEventListener('click', () => applyChapter(state.chapter + 1));

    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', (e) => {
        const tag = e.target && e.target.tagName;
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
        if (e.key === 'ArrowRight') { applyChapter(state.chapter + 1); }
        else if (e.key === 'ArrowLeft') { applyChapter(state.chapter - 1); }
      });
      window.addEventListener('atlas:theory', (e) => {
        state.showTheory = e.detail && typeof e.detail.active === 'boolean' ? e.detail.active : !state.showTheory;
      });
      // The ghost starts on; align the shell's Theory button with it so the first T press hides it.
      const theoryBtn = doc.getElementById('atlas-btn-theory');
      if (theoryBtn && theoryBtn.getAttribute('aria-pressed') !== 'true') theoryBtn.click();
    }

    // ---- canvas interaction ----
    let geom = null;

    function canvasPoint(e) {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    canvas.addEventListener('pointerdown', (e) => {
      const pt = canvasPoint(e);
      if (geom && geom.hist && pt.y >= geom.hist.y - 14) {
        const barW = geom.hist.w / (geom.hist.bMax + 1);
        const b = Math.round((pt.x - geom.hist.x) / barW);
        state.theta = Math.max(0, Math.min(state.w, b));
        state.theta = Number.isInteger(state.theta) ? state.theta : Math.round(state.theta);
        thetaEl.value = String(state.theta);
        thetaValEl.textContent = String(state.theta);
        state.draggingTheta = true;
        canvas.setPointerCapture && canvas.setPointerCapture(e.pointerId);
        return;
      }
      if (geom && geom.grid) {
        const gd = geom.grid;
        const q = Math.floor((pt.x - gd.x) / gd.cell);
        const r = Math.floor((pt.y - gd.y) / gd.cell);
        if (q >= 0 && q < N_COLS && r >= 0 && r < gd.rows) {
          const idx = r * N_COLS + q;
          if (!state.presented) state.presented = new Set();
          if (state.presented.has(idx)) state.presented.delete(idx);
          else state.presented.add(idx);
          setFlash();
        }
      }
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!state.draggingTheta) return;
      const pt = canvasPoint(e);
      const barW = geom.hist.w / (geom.hist.bMax + 1);
      const b = Math.round((pt.x - geom.hist.x) / barW);
      state.theta = Math.max(0, Math.min(state.w, b));
      thetaEl.value = String(state.theta);
      thetaValEl.textContent = String(state.theta);
    });
    canvas.addEventListener('pointerup', (e) => {
      state.draggingTheta = false;
      try {
        if (canvas.releasePointerCapture && e.pointerId != null) canvas.releasePointerCapture(e.pointerId);
      } catch (err) {
        /* pointer already released */
      }
    });

    // ---- drawing ----
    function cellPos(i, gd) {
      const q = i % N_COLS;
      const r = Math.floor(i / N_COLS);
      return { x: gd.x + q * gd.cell, y: gd.y + r * gd.cell, s: gd.cell };
    }

    function drawGrid(c, now) {
      const pad = 14;
      const W = geom.W;
      const topH = geom.topH;
      const rows = state.n / N_COLS;
      const cell = Math.max(4, Math.floor(Math.min((W - 2 * pad) / N_COLS, (topH - pad - 20) / rows)));
      const gw = cell * N_COLS;
      const gh = cell * rows;
      const gx = Math.round((W - gw) / 2);
      const gy = pad + 18;
      const gd = { x: gx, y: gy, cell, cols: N_COLS, rows };
      geom.grid = gd;

      // frame + caption
      const narrow = W < 560;
      g.fillStyle = c.inkMuted;
      g.font = `${narrow ? 10 : 12}px ui-monospace, monospace`;
      g.textAlign = 'left';
      g.textBaseline = 'top';
      const caption = narrow
        ? `n=${state.n} \u00b7 w=${state.w} (${(100 * state.w / state.n).toFixed(1)}%)`
        : `n = ${state.n} cells  \u00b7  w = ${state.w} active  \u00b7  sparsity ${(100 * state.w / state.n).toFixed(1)}%`;
      g.fillText(caption, pad, pad - 2);

      // base cells
      g.fillStyle = c.paperCard;
      g.fillRect(gx - 1, gy - 1, gw + 2, gh + 2);
      g.fillStyle = withAlpha(c.border, 0.55);
      g.beginPath();
      for (let r = 0; r < rows; r++) {
        for (let q = 0; q < N_COLS; q++) {
          g.rect(gx + q * cell + 0.5, gy + r * cell + 0.5, Math.max(1, cell - 1), Math.max(1, cell - 1));
        }
      }
      g.fill();

      const pulse = reduced ? 1 : 0.55 + 0.45 * Math.sin(now * 0.005);
      const stored = state.storedSet;
      const union = state.mode === 'union' ? state.unionSet : null;
      const presented = state.presented;

      // glow helper
      function glow(i, color, alpha) {
        const p = cellPos(i, gd);
        g.fillStyle = withAlpha(color, alpha * 0.28);
        g.fillRect(p.x - 1, p.y - 1, p.s + 2, p.s + 2);
        g.fillStyle = color;
        g.fillRect(p.x, p.y, p.s, p.s);
      }

      // stored / union cells
      if (union) {
        for (const i of union) glow(i, c.mod, 1);
      } else {
        for (const i of state.stored) glow(i, c.exc, 1);
      }

      // presented cells
      if (presented) {
        for (const i of presented) {
          if (stored.has(i)) continue; // shared drawn on top
          if (union && union.has(i)) continue;
          glow(i, c.volt, 1);
        }
      }

      // shared cells flash
      const flashAge = (now - state.flash) / 700;
      if (presented) {
        for (const i of presented) {
          const inStored = stored.has(i);
          const inUnion = union && union.has(i);
          if (!inStored && !inUnion) continue;
          const p = cellPos(i, gd);
          const base = inStored ? c.exc : c.mod;
          // blend both roles, then a bright white core
          g.fillStyle = c.paperElevated || '#ffffff';
          g.fillRect(p.x, p.y, p.s, p.s);
          g.fillStyle = withAlpha(base, 0.85);
          g.fillRect(p.x + p.s * 0.3, p.y + p.s * 0.3, p.s * 0.4, p.s * 0.4);
          g.strokeStyle = c.ink;
          g.lineWidth = 1;
          g.strokeRect(p.x + 0.5, p.y + 0.5, p.s - 1, p.s - 1);
          if (flashAge >= 0 && flashAge < 1) {
            const rad = p.s * (0.6 + flashAge * 1.8);
            g.strokeStyle = withAlpha(c.ink, 1 - flashAge);
            g.lineWidth = 1.5;
            g.beginPath();
            g.arc(p.x + p.s / 2, p.y + p.s / 2, rad, 0, Math.PI * 2);
            g.stroke();
          }
        }
      }

      // highlight from equation hover
      if (state.highlight) {
        const pulseA = reduced ? 1 : 0.5 + 0.5 * Math.sin(now * 0.008);
        g.strokeStyle = withAlpha(c.trace, 0.5 + 0.5 * pulseA);
        g.lineWidth = 2;
        if (state.highlight === 'w' || state.highlight === 'n') {
          g.strokeRect(gx - 2.5, gy - 2.5, gw + 5, gh + 5);
        }
      }

      g.strokeStyle = c.border;
      g.lineWidth = 1;
      g.strokeRect(gx - 1.5, gy - 1.5, gw + 3, gh + 3);

      // legend
      const ly = gy + gh + 10;
      let lx = gx;
      g.font = '11px ui-sans-serif, system-ui, sans-serif';
      g.textAlign = 'left';
      g.textBaseline = 'middle';
      const legend = [
        [union ? c.mod : c.exc, union ? 'stored union' : S.stored],
        [c.volt, S.presented],
        ['#ffffff', S.shared]
      ];
      for (const [col, label] of legend) {
        g.fillStyle = col;
        g.fillRect(lx, ly - 5, 10, 10);
        g.strokeStyle = c.ink;
        g.lineWidth = 1;
        g.strokeRect(lx + 0.5, ly - 4.5, 9, 9);
        g.fillStyle = c.inkMuted;
        g.fillText(label, lx + 14, ly);
        lx += 14 + g.measureText(label).width + 18;
      }

      // big overlap readout on canvas
      const ov = overlapCount(presented);
      g.textAlign = 'right';
      g.textBaseline = 'top';
      g.fillStyle = c.inkMuted;
      g.font = `${narrow ? 10 : 11}px ui-monospace, monospace`;
      g.fillText('overlap', W - pad, pad - 2);
      g.fillStyle = c.ink;
      g.font = `600 ${narrow ? 22 : 30}px ui-monospace, monospace`;
      g.fillText(String(ov), W - pad, pad + (narrow ? 8 : 12));
      g.font = `${narrow ? 10 : 11}px ui-monospace, monospace`;
      g.fillStyle = c.inkMuted;
      g.fillText(`of ${state.w}`, W - pad, pad + (narrow ? 32 : 46));
    }

    function drawHistogram(c, now) {
      const pad = 14;
      const W = geom.W, H = geom.H;
      const x0 = pad + 46;
      const y0 = geom.topH + 30;
      const w = W - x0 - pad - 8;
      const h = Math.max(60, H - y0 - 34);
      const bMax = Math.min(state.w, Math.max(10, state.theta + 2));
      const barW = w / (bMax + 1);
      geom.hist = { x: x0, y: y0, w, h, bMax };

      // frame
      g.fillStyle = withAlpha(c.border, 0.12);
      g.fillRect(x0, y0, w, h);

      // log-y scale over the shown range
      let minP = Infinity;
      for (let b = 0; b <= bMax; b++) if (state.pmfFull[b] > 0) minP = Math.min(minP, state.pmfFull[b]);
      if (!Number.isFinite(minP) || minP <= 0) minP = 1e-60;
      // log-y floor: keep the Monte Carlo mass and most of the ghost visible, clip the far tail
      const hi = 0.2;
      const lo = Math.min(hi - 1, Math.max(-8, Math.floor(Math.log10(minP)) - 1));
      const yLog = (lg) => y0 + (1 - (lg - lo) / (hi - lo)) * h;
      const xB = (b) => x0 + b * barW;

      // tail shading b >= theta
      const xt = xB(state.theta);
      g.fillStyle = withAlpha(c.volt, 0.13);
      g.fillRect(xt, y0, x0 + w - xt, h);

      // gridlines per decade
      g.strokeStyle = withAlpha(c.border, 0.7);
      g.lineWidth = 0.5;
      g.font = '9px ui-monospace, monospace';
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillStyle = c.inkMuted;
      const decStep = Math.max(5, Math.ceil((hi - lo) / 6 / 5) * 5);
      for (let d = 0; d >= lo; d -= decStep) {
        const y = yLog(d);
        if (y < y0 || y > y0 + h) continue;
        g.beginPath();
        g.moveTo(x0, y);
        g.lineTo(x0 + w, y);
        g.stroke();
        g.fillText(`10^${d}`, x0 - 5, y);
      }

      // Monte Carlo bars
      if (state.mcTotal > 0) {
        g.fillStyle = withAlpha(c.trace, 0.72);
        for (let b = 0; b <= bMax; b++) {
          const p = state.mcCounts[b] / state.mcTotal;
          if (p <= 0) continue;
          const lg = Math.log10(p);
          if (lg < lo) continue;
          const yy = yLog(lg);
          g.fillRect(xB(b) + barW * 0.18, yy, barW * 0.64, y0 + h - yy);
        }
      }

      // theory ghost dashed pmf
      if (state.showTheory) {
        const xs = [];
        const ys = [];
        for (let b = 0; b <= bMax; b++) {
          const p = state.pmfFull[b];
          if (p <= 0 || Math.log10(p) < lo) continue;
          xs.push(xB(b) + barW / 2);
          ys.push(yLog(Math.log10(p)));
        }
        g.save();
        g.setLineDash([5, 4]);
        g.strokeStyle = c.theory;
        g.lineWidth = 1.5;
        g.beginPath();
        for (let i = 0; i < xs.length; i++) (i ? g.lineTo : g.moveTo).call(g, xs[i], ys[i]);
        g.stroke();
        g.restore();
      }

      // mean marker
      const mB = core.expectedOverlap(state.n, state.w);
      if (mB <= bMax) {
        const mx = xB(mB) + barW / 2;
        g.strokeStyle = withAlpha(c.ink, 0.5);
        g.setLineDash([2, 3]);
        g.beginPath();
        g.moveTo(mx, y0);
        g.lineTo(mx, y0 + h);
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = c.inkMuted;
        g.font = '9px ui-monospace, monospace';
        g.textAlign = 'center';
        g.textBaseline = 'bottom';
        g.fillText(`mean ${mB.toFixed(2)}`, mx, y0 - 2);
      }

      // threshold line + knob
      g.strokeStyle = c.ink;
      g.lineWidth = 1.5;
      g.setLineDash([4, 3]);
      g.beginPath();
      g.moveTo(xt, y0 - 6);
      g.lineTo(xt, y0 + h);
      g.stroke();
      g.setLineDash([]);
      g.fillStyle = c.paperCard;
      g.beginPath();
      g.arc(xt, y0 - 6, 7, 0, Math.PI * 2);
      g.fill();
      g.strokeStyle = c.ink;
      g.lineWidth = 1.5;
      g.stroke();
      g.fillStyle = c.ink;
      g.font = '600 10px ui-monospace, monospace';
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText('\u03b8', xt, y0 - 6);
      g.setLineDash([]);

      // x-axis
      g.strokeStyle = c.border;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x0, y0 + h);
      g.lineTo(x0 + w, y0 + h);
      g.stroke();
      g.fillStyle = c.inkMuted;
      g.font = '10px ui-monospace, monospace';
      g.textAlign = 'center';
      g.textBaseline = 'top';
      const xstep = Math.max(1, Math.round(bMax / 8));
      for (let b = 0; b <= bMax; b += xstep) {
        g.fillText(String(b), xB(b) + barW / 2, y0 + h + 4);
      }

      // tail probability label (boxed so the ghost curve never crosses the text)
      const tail = core.sdrTail(state.n, state.w, state.theta);
      const tailText = `P(b \u2265 \u03b8) = ${core.formatSci(tail)}`;
      const capText = state.showTheory ? 'bars Monte Carlo \u00b7 dashed exact pmf' : 'bars Monte Carlo';
      g.font = '12px ui-monospace, monospace';
      const tw = g.measureText(tailText).width;
      const boxX = x0 + w - tw - 22;
      g.fillStyle = c.paperCard;
      g.globalAlpha = 0.94;
      g.fillRect(boxX, y0 + 6, tw + 14, 34);
      g.globalAlpha = 1;
      g.strokeStyle = c.border;
      g.lineWidth = 1;
      g.strokeRect(boxX + 0.5, y0 + 6.5, tw + 13, 33);
      g.textAlign = 'right';
      g.textBaseline = 'top';
      g.fillStyle = c.ink;
      g.fillText(tailText, x0 + w - 14, y0 + 9);
      g.fillStyle = c.inkMuted;
      g.font = '10px ui-monospace, monospace';
      g.fillText(capText, x0 + w - 14, y0 + 25);

      // union readout on canvas (the chapter whose answer lives in the margin below the fold)
      if (state.mode === 'union') {
        const p = core.unionActivation(state.n, state.w, state.unionM);
        const pm = core.unionMatchProbability(state.n, state.w, state.unionM, state.theta);
        const ux = x0 + w * 0.28;
        const ut = `union M=${state.unionM}  p=${p.toFixed(3)}  P(match)=${core.formatSci(pm)}`;
        g.font = '11px ui-monospace, monospace';
        const uw = g.measureText(ut).width;
        g.fillStyle = c.paperCard;
        g.globalAlpha = 0.94;
        g.fillRect(ux - 4, y0 + 5, uw + 8, 30);
        g.globalAlpha = 1;
        g.strokeStyle = c.border;
        g.lineWidth = 1;
        g.strokeRect(ux - 3.5, y0 + 5.5, uw + 7, 29);
        g.textAlign = 'left';
        g.textBaseline = 'top';
        g.fillStyle = c.mod;
        g.fillText(ut, ux, y0 + 7);
        g.fillStyle = c.inkMuted;
        g.font = '9px ui-monospace, monospace';
        g.fillText('binomial approx (Ahmad & Hawkins 2016)', ux, y0 + 22);
      }

      // highlight for theta hover
      if (state.highlight === 'theta') {
        g.strokeStyle = withAlpha(c.trace, 0.9);
        g.lineWidth = 2;
        g.strokeRect(xt - 3, y0 - 14, 6, h + 14);
      }
    }

    function render() {
      const fit = Atlas && Atlas.fitCanvas ? Atlas.fitCanvas(canvas) : { width: canvas.width, height: canvas.height };
      const c = colors();
      const W = fit.width;
      const H = fit.height;
      geom = { W, H, topH: Math.round(H * 0.56), grid: null, hist: null };
      g.clearRect(0, 0, W, H);
      g.fillStyle = c.paperCard;
      g.fillRect(0, 0, W, H);
      const now = performance.now();
      drawGrid(c, now);
      drawHistogram(c, now);
      updateMargin();
    }

    function updateMargin() {
      const ov = overlapCount(state.presented);
      const active = state.presented ? state.presented.size : 0;
      overlapEl.textContent = `${ov} / ${state.w}`;
      const recognized = ov >= state.theta;
      statusEl.textContent = state.presented
        ? `${S.active}: ${active}  \u00b7  ${recognized ? S.recognized : S.notRecognized}`
        : `${S.active}: ${S.stored} ${state.w}`;
      statusEl.style.color = state.presented && recognized ? 'var(--exc)' : 'var(--ink-muted)';

      const tail = core.sdrTail(state.n, state.w, state.theta);
      tailEl.textContent = core.formatSci(tail);
      const m = core.expectedOverlap(state.n, state.w);
      meanEl.textContent = m.toFixed(3);
      const mcMean = state.mcTotal > 0 ? state.mcSum / state.mcTotal : NaN;
      mcMeanEl.textContent = state.mcTotal > 0
        ? `Monte Carlo mean ${mcMean.toFixed(3)} over ${state.mcTotal} draws`
        : `Monte Carlo: press "Draw 100 random"`;
      if (state.mcTotal > 0) {
        let tailN = 0;
        for (let b = state.theta; b < state.mcCounts.length; b++) tailN += state.mcCounts[b];
        const mcTail = tailN / state.mcTotal;
        diffEl.textContent = `mean ${core.formatSci(Math.abs(mcMean - m), 1)}  \u00b7  tail ${core.formatSci(Math.abs(mcTail - tail), 1)}`;
      } else {
        diffEl.textContent = '\u2014';
      }
      eqEl.textContent = `n=${state.n}  w=${state.w}  \u03b8=${state.theta}  \u2192  ${core.formatSci(tail)}`;

      if (state.mode === 'union') {
        const p = core.unionActivation(state.n, state.w, state.unionM);
        const size = core.unionExpectedSize(state.n, state.w, state.unionM);
        const pm = core.unionMatchProbability(state.n, state.w, state.unionM, state.theta);
        unionReadout.textContent = `${S.unionActivation} ${p.toFixed(3)}  \u00b7  ${S.unionSize} ${size.toFixed(0)}  \u00b7  ${S.unionP} ${core.formatSci(pm)}  \u00b7  ${S.unionNote}`;
      } else {
        unionReadout.textContent = '';
      }
    }

    // ---- animation ----
    let raf = 0;
    let lastTs = 0;
    function frame(ts) {
      const dt = lastTs ? Math.min(100, ts - lastTs) : 16;
      lastTs = ts;
      const playing = !reduced && ctx.clock && ctx.clock.isPlaying;
      if (playing && state.mode === 'stream') {
        addSamples(Math.max(1, Math.round((dt / 1000) * 2500)));
        if (state.mcTotal < MC_CAP && ts - state.lastStream > 420) {
          state.presented = new Set(core.randomPattern(state.n, state.w, state.mcRng));
          setFlash();
          state.lastStream = ts;
        }
      }
      render();
      raf = requestAnimationFrame(frame);
    }

    // initial state
    rebuildPatterns();
    applyChapter(0);
    render();
    if (typeof requestAnimationFrame === 'function') raf = requestAnimationFrame(frame);
  }

  if (Atlas && Atlas.Plate) {
    Atlas.Plate.register({ id: 'H01', title: { en: S.title }, source: 'live', mount });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
