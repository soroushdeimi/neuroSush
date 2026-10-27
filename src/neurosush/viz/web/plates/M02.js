/**
 * Plate M02 - Rain on a timeline: Poisson spikes, binomial counts, geometric gaps.
 *
 * The hero draws a discrete-time Bernoulli spike train as rain landing on a timeline;
 * a sliding window counts the spikes per bin, and three histograms compare the simulated
 * counts against the exact binomial, geometric and normal pmfs and their dt -> 0 limits
 * (Poisson, exponential). The live port mirrors neurosush.encoding.rate_poisson, where each
 * step spikes with probability rate * dt.
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Strings (English only; every caption and label lives here)
  // ---------------------------------------------------------------------------
  const STRINGS = {
    title: 'Rain on a timeline: Poisson spikes, binomial counts',
    hero: 'Spike rain',
    countsTitle: 'Counts per window',
    isiTitle: 'Gaps between spikes',
    cltTitle: 'Sums of windows (CLT)',
    axisCount: 'P(count)',
    axisDensity: 'density (per ms)',
    axisSum: 'density',
    noData: 'collecting samples',
    chapterOf: 'Chapter {i} / {n}',
    prev: '\u2039 Back',
    next: 'Next \u203a',
    equations: 'Live equation',
    readoutsTitle: 'Readouts',
    predictTitle: 'Predict, then reveal',
    predictQuestion: 'Rate 10 Hz, window 100 ms: how often do we see 0 spikes?',
    predictTruthBinom: 'binomial (dt = 5 ms)',
    predictTruthPoisson: 'Poisson (dt \u2192 0)',
    glossaryTitle: 'Glossary',
    controls: 'Controls',
    rateLabel: 'rate (Hz)',
    dtLabel: 'step dt (ms)',
    binLabel: 'window W (ms)',
    runMore: 'Run 1000 more windows',
    reseed: 'Reseed',
    theory: 'Theory ghost [T]',
    pause: 'Pause',
    play: 'Play',
    hintHero: 'drag the window edge to change W \u00b7 drag the cloud vertically to change rate \u00b7 click the tape to inject a spike',
    hintClt: 'drag vertically to change how many windows are summed',
    windowLabel: 'window',
    spikes: 'spikes',
    stepRuler: 'coin flips',
    bandNote: 'coin flips too fine to draw',
    footer: 'live \u00b7 mirrors neurosush.encoding.rate_poisson (p = rate \u00b7 dt)',
    chapterCaps: [
      'One neuron, many tiny coin flips. Every dt ms the neuron flips a biased coin: it spikes with probability p = rate\u00b7dt. Make dt smaller and the coin is flipped more often, each time with a smaller chance.',
      'Count spikes in a window. A window of W ms holds n = W/dt coin flips, so its spike count is Binomial(n, p) with mean \u03bb = n\u00b7p = rate\u00b7W, whatever dt is. Watch the bars grow onto the dashed binomial pmf.',
      'Smaller steps: Poisson takes over. As dt \u2192 0 with \u03bb = rate\u00b7W fixed, the binomial pmf slides onto the dots of the Poisson pmf; the total-variation distance shrinks with the step size.',
      'The gaps between spikes. The time from one spike to the next is geometric, becoming exponential as dt \u2192 0. Its coefficient of variation sits near 1 - the fingerprint of a Poisson spike train.',
      "Add many windows: the bell curve. Sum m windows and the histogram of the sums narrows into a bell; the dashed normal ghost is the central limit theorem.",
    ],
    glossary: [
      ['dt', 'Step length in ms; one Bernoulli coin flip per step.'],
      ['rate', 'Mean firing rate in Hz; p = rate\u00b7dt at each step.'],
      ['window', 'A block of W ms whose spike count N we histogram.'],
      ['pmf', 'Probability mass function: the chance of each count.'],
      ['Poisson limit', 'dt\u21920 with rate\u00b7W fixed turns Binomial(n,p) into Poisson(\u03bb).'],
      ['ISI', 'Inter-spike interval: time from one spike to the next.'],
      ['CV', 'Coefficient of variation, sd/mean; near 1 for a Poisson train.'],
      ['TV distance', 'Half the sum of |binomial \u2212 Poisson| over counts; it shrinks with dt.'],
      ['CLT', 'Summing many windows makes the count distribution normal.'],
    ],
  };

  // ---------------------------------------------------------------------------
  // Core: pure math, no DOM. Exported for node tests.
  // ---------------------------------------------------------------------------

  /** Lanczos coefficients for the log-gamma function (g = 7, n = 9). */
  const LG_C = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  /**
   * Natural log of the gamma function via the Lanczos approximation.
   * @param {number} x Argument, x > 0.
   * @returns {number}
   */
  function lgamma(x) {
    if (!(x > 0)) throw new Error(`lgamma needs x > 0, got ${x}`);
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    const z = x - 1;
    let a = LG_C[0];
    const t = z + 7.5;
    for (let i = 1; i < 9; i++) a += LG_C[i] / (z + i);
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
  }

  /** @param {number} p Spike probability; throws unless p is in [0, 1]. */
  function assertProb(p) {
    if (typeof p !== 'number' || !(p >= 0 && p <= 1)) {
      throw new Error(`p must be in [0, 1], got ${p}`);
    }
  }

  /**
   * Binomial pmf P(X = k) = C(n,k) p^k (1-p)^(n-k), computed with lgamma to avoid overflow.
   * @param {number} k
   * @param {number} n
   * @param {number} p
   * @returns {number}
   */
  function binomialPmf(k, n, p) {
    if (!Number.isInteger(n) || n < 0) throw new Error(`n must be a non-negative integer, got ${n}`);
    assertProb(p);
    if (!Number.isInteger(k) || k < 0 || k > n) return 0;
    if (p === 0) return k === 0 ? 1 : 0;
    if (p === 1) return k === n ? 1 : 0;
    const logp =
      lgamma(n + 1) -
      lgamma(k + 1) -
      lgamma(n - k + 1) +
      k * Math.log(p) +
      (n - k) * Math.log1p(-p);
    return Math.exp(logp);
  }

  /**
   * Poisson pmf P(X = k) = e^-lambda lambda^k / k!.
   * @param {number} k
   * @param {number} lambda
   * @returns {number}
   */
  function poissonPmf(k, lambda) {
    if (typeof lambda !== 'number' || !(lambda >= 0)) {
      throw new Error(`lambda must be >= 0, got ${lambda}`);
    }
    if (!Number.isInteger(k) || k < 0) return 0;
    if (lambda === 0) return k === 0 ? 1 : 0;
    return Math.exp(-lambda + k * Math.log(lambda) - lgamma(k + 1));
  }

  /**
   * Geometric pmf P(ISI = k) = p (1-p)^(k-1) on k >= 1 steps.
   * @param {number} k
   * @param {number} p
   * @returns {number}
   */
  function geometricPmf(k, p) {
    assertProb(p);
    if (!Number.isInteger(k) || k < 1) return 0;
    if (p === 0) return 0;
    if (p === 1) return k === 1 ? 1 : 0;
    return p * Math.pow(1 - p, k - 1);
  }

  /**
   * Exponential density f(t) = rate e^(-rate t) for t >= 0.
   * @param {number} t
   * @param {number} rate
   * @returns {number}
   */
  function exponentialPdf(t, rate) {
    if (typeof rate !== 'number' || !(rate >= 0)) {
      throw new Error(`rate must be >= 0, got ${rate}`);
    }
    if (!(t >= 0) || rate === 0) return 0;
    return rate * Math.exp(-rate * t);
  }

  /**
   * Normal density with mean mu and standard deviation sigma.
   * @param {number} x
   * @param {number} mu
   * @param {number} sigma Standard deviation, > 0.
   * @returns {number}
   */
  function normalPdf(x, mu, sigma) {
    if (typeof sigma !== 'number' || !(sigma > 0)) {
      throw new Error(`sigma must be positive, got ${sigma}`);
    }
    const z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  }

  /**
   * mulberry32 PRNG. Returns a function drawing uniform numbers in [0, 1).
   * @param {number} a Integer seed.
   * @returns {function(): number}
   */
  function mulberry32(a) {
    if (!Number.isInteger(a)) throw new Error(`seed must be an integer, got ${a}`);
    let s = a >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Bernoulli spike train: each step is a spike with probability p.
   * @param {number} seed
   * @param {number} p
   * @param {number} steps
   * @returns {Uint8Array} 1 for a spike, 0 otherwise.
   */
  function simulateTrain(seed, p, steps) {
    assertProb(p);
    if (!Number.isInteger(steps) || steps < 0) {
      throw new Error(`steps must be a non-negative integer, got ${steps}`);
    }
    const rng = mulberry32(seed);
    const out = new Uint8Array(steps);
    for (let i = 0; i < steps; i++) out[i] = rng() < p ? 1 : 0;
    return out;
  }

  /**
   * Count spikes per window: bins independent runs of nPerBin Bernoulli(p) steps.
   * @param {number} seed
   * @param {number} p
   * @param {number} nPerBin Positive integer steps per window.
   * @param {number} bins Number of windows.
   * @returns {Int32Array}
   */
  function binCounts(seed, p, nPerBin, bins) {
    assertProb(p);
    if (!Number.isInteger(nPerBin) || nPerBin < 1) {
      throw new Error(`nPerBin must be a positive integer, got ${nPerBin}`);
    }
    if (!Number.isInteger(bins) || bins < 0) {
      throw new Error(`bins must be a non-negative integer, got ${bins}`);
    }
    const rng = mulberry32(seed);
    const out = new Int32Array(bins);
    for (let b = 0; b < bins; b++) {
      let c = 0;
      for (let k = 0; k < nPerBin; k++) if (rng() < p) c++;
      out[b] = c;
    }
    return out;
  }

  /**
   * Inter-spike intervals in steps (differences of spike indices).
   * @param {ArrayLike<number>} train
   * @returns {Float64Array}
   */
  function interSpikeIntervals(train) {
    const idx = [];
    for (let i = 0; i < train.length; i++) if (train[i]) idx.push(i);
    const gaps = new Float64Array(Math.max(0, idx.length - 1));
    for (let i = 0; i + 1 < idx.length; i++) gaps[i] = idx[i + 1] - idx[i];
    return gaps;
  }

  /** @param {ArrayLike<number>} a @returns {number} */
  function sum(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s;
  }

  /**
   * Sum a pmf over an inclusive integer range.
   * @param {function(number): number} f
   * @param {number} lo
   * @param {number} hi
   * @returns {number}
   */
  function pmfSum(f, lo, hi) {
    let s = 0;
    for (let k = lo; k <= hi; k++) s += f(k);
    return s;
  }

  /** @param {ArrayLike<number>} a @returns {number} */
  function mean(a) {
    return a.length ? sum(a) / a.length : NaN;
  }

  /** @param {ArrayLike<number>} a @returns {number} Population variance. */
  function variance(a) {
    if (a.length < 2) return NaN;
    const m = mean(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - m;
      s += d * d;
    }
    return s / a.length;
  }

  /** @param {ArrayLike<number>} a @returns {number} */
  function std(a) {
    return Math.sqrt(variance(a));
  }

  /** @param {ArrayLike<number>} a @returns {number} sd / mean, NaN when mean is 0. */
  function coefficientOfVariation(a) {
    const m = mean(a);
    return m === 0 ? NaN : std(a) / m;
  }

  /** @param {ArrayLike<number>} a @returns {number} Population skewness. */
  function skewness(a) {
    const n = a.length;
    if (n < 3) return NaN;
    const m = mean(a);
    const s = std(a);
    if (s === 0) return NaN;
    let acc = 0;
    for (let i = 0; i < n; i++) {
      const z = (a[i] - m) / s;
      acc += z * z * z;
    }
    return acc / n;
  }

  /**
   * Total-variation distance 0.5 * sum |a - b| between two pmf arrays.
   * @param {ArrayLike<number>} a
   * @param {ArrayLike<number>} b
   * @returns {number}
   */
  function totalVariation(a, b) {
    if (a.length !== b.length) {
      throw new Error(`totalVariation needs equal lengths, got ${a.length} and ${b.length}`);
    }
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return 0.5 * s;
  }

  /**
   * Total-variation distance between Binomial(n, lambda/n) and Poisson(lambda).
   * @param {number} lambda
   * @param {number} n
   * @returns {number}
   */
  function binomialToPoissonTV(lambda, n) {
    if (typeof lambda !== 'number' || !(lambda >= 0)) {
      throw new Error(`lambda must be >= 0, got ${lambda}`);
    }
    if (!Number.isInteger(n) || n < 1) throw new Error(`n must be a positive integer, got ${n}`);
    const p = lambda / n;
    if (p > 1) throw new Error(`lambda must be <= n, got lambda=${lambda}, n=${n}`);
    const K = Math.max(n, Math.ceil(lambda + 10 * Math.sqrt(lambda) + 20));
    let tv = 0;
    for (let k = 0; k <= K; k++) tv += Math.abs(binomialPmf(k, n, p) - poissonPmf(k, lambda));
    return 0.5 * tv;
  }

  /**
   * Group window counts into sums of m consecutive windows.
   * @param {ArrayLike<number>} counts
   * @param {number} m Positive integer windows per sum.
   * @returns {Float64Array}
   */
  function sumGroups(counts, m) {
    if (!Number.isInteger(m) || m < 1) throw new Error(`m must be a positive integer, got ${m}`);
    const out = [];
    for (let i = 0; i + m <= counts.length; i += m) {
      let s = 0;
      for (let j = 0; j < m; j++) s += counts[i + j];
      out.push(s);
    }
    return new Float64Array(out);
  }

  const core = {
    lgamma,
    binomialPmf,
    poissonPmf,
    geometricPmf,
    exponentialPdf,
    normalPdf,
    mulberry32,
    simulateTrain,
    binCounts,
    interSpikeIntervals,
    sum,
    pmfSum,
    mean,
    variance,
    std,
    coefficientOfVariation,
    skewness,
    totalVariation,
    binomialToPoissonTV,
    sumGroups,
  };

  // ---------------------------------------------------------------------------
  // UI: only runs when a browser Atlas shell exists.
  // ---------------------------------------------------------------------------

  /** @param {number} x @param {number} lo @param {number} hi @returns {number} */
  function clamp(x, lo, hi) {
    return x < lo ? lo : x > hi ? hi : x;
  }

  /** @param {number} x @param {number} d @returns {string} */
  function fmt(x, d) {
    if (!Number.isFinite(x)) return '-';
    return x.toFixed(d);
  }

  // History simulated before the first frame so the hero is already raining at t=0.
  const PREFILL_MS = 3200;

  const CHAPTERS = [
    { rate: 10, dt: 5, binMs: 100, m: 10, hl: 'hero' },
    { rate: 10, dt: 5, binMs: 100, m: 10, hl: 'counts' },
    { rate: 10, dt: 0.5, binMs: 100, m: 10, hl: 'counts' },
    { rate: 10, dt: 5, binMs: 100, m: 10, hl: 'isi' },
    { rate: 10, dt: 5, binMs: 100, m: 10, hl: 'clt' },
  ];

  const S = STRINGS;

  /**
   * Mount the plate: build the margin, wire interactions and drive the animation.
   * @param {HTMLElement} rootEl
   * @param {object} ctx Atlas shell context.
   */
  function mount(rootEl, ctx) {
    const fig = ctx.figure || rootEl;
    const marginEl = ctx.margin || (fig && fig.margin) || null;
    const colors = ctx.colors || function () {
      return {};
    };
    const canvas = ctx.canvas();
    const g = canvas.getContext('2d');

    // Fonts pulled from the atlas tokens so type matches the shell.
    const rootStyle = document.documentElement ? getComputedStyle(document.documentElement) : null;
    const fontVar = (name, fallback) =>
      (rootStyle && rootStyle.getPropertyValue(name).trim()) || fallback;
    const FONT_MONO = fontVar('--font-mono', 'ui-monospace, monospace');
    const FONT_SERIF = fontVar('--font-serif', 'ui-serif, Georgia, serif');
    const FONT_SANS = fontVar('--font-sans', 'system-ui, sans-serif');

    const state = {
      seed: 20261017,
      rate: 10,
      dt: 5,
      binMs: 100,
      m: 10,
      nBinsExtra: 0,
      theory: true,
      chapter: 0,
      highlight: CHAPTERS[0].hl,
      hover: null,
      predictChoice: null,
      paused: false,
      // derived
      p: 0.05,
      nPerBin: 20,
      lambda: 1,
      tv: 0,
      counts: null,
      countsBins: 0,
      isi: null,
      isiN: 0,
      isiMeanMs: 0,
      isiCV: 0,
      sums: null,
      sumsM: 10,
      // stream
      rng: null,
      simT: 0,
      nextStep: 0,
      drops: [],
      winCount: 0,
      viewSpan: 3000,
      // interaction
      hit: null,
      drag: null,
      dirty: true,
    };

    const refs = {};

    // ------------------------------------------------------------------ style
    const style = document.createElement('style');
    style.id = 'm02-style';
    style.textContent = `
      .m02-caption { font-family: ${FONT_SERIF}; font-size: 13px; line-height: 1.45; color: var(--ink); }
      .m02-nav { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }
      .m02-btn { background: var(--paper-card); color: var(--ink); border:1px solid var(--border);
        border-radius:3px; padding:3px 8px; font:inherit; font-size:13px; cursor:pointer; min-height:28px; }
      .m02-btn:hover { background: var(--paper-elevated); border-color: var(--ink-muted); }
      .m02-btn:focus-visible { outline:2px solid var(--focus-ring); outline-offset:2px; }
      .m02-btn.active, .m02-btn[aria-pressed="true"] { background: var(--ink); color: var(--paper); border-color: var(--ink); }
      .m02-eq { font-family:${FONT_MONO}; font-size:13px; line-height:1.7; direction:ltr; text-align:left; }
      .m02-term { cursor:help; border-bottom:1px dotted var(--ink-muted); }
      .m02-term:hover, .m02-term:focus { background: var(--paper-inset); outline:none; }
      .m02-term.active { background: var(--exc); color: var(--paper); }
      .m02-val { float:right; font-family:${FONT_MONO}; font-variant-numeric:tabular-nums; }
      .m02-row { display:flex; align-items:center; gap:8px; margin:6px 0 2px; }
      .m02-row label { font-family:${FONT_SANS}; font-size:13px; color:var(--ink-muted); min-width:88px; }
      .m02-row input[type=range] { flex:1; accent-color: var(--exc); }
      .m02-row .m02-val { float:none; min-width:54px; text-align:right; }
      .m02-choice { display:flex; gap:6px; margin-top:6px; flex-wrap:wrap; }
      .m02-gl { font-size:13px; line-height:1.5; }
      .m02-gl b { font-family:${FONT_MONO}; color:var(--ink); }
      .m02-note { font-size:13px; color:var(--ink-muted); }
      .m02-hint { font-family:${FONT_SERIF}; font-size:13px; color:var(--ink-muted); margin-top:6px; }
    `;
    document.head.appendChild(style);

    // ------------------------------------------------------------------ margin
    if (marginEl) {
      const glossary = S.glossary
        .map(([w, d]) => `<div class="m02-gl"><b>${w}</b> \u2014 ${d}</div>`)
        .join('');
      marginEl.innerHTML = `
        <div class="atlas-margin-section">
          <div class="atlas-margin-label" id="m02-chapter-label">Chapter 1 / 5</div>
          <div class="m02-caption" id="m02-caption"></div>
          <div class="m02-nav">
            <button class="m02-btn" id="m02-prev" aria-label="Previous chapter">${S.prev}</button>
            <button class="m02-btn" id="m02-next" aria-label="Next chapter">${S.next}</button>
          </div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.equations}</div>
          <div class="m02-eq">
            <div><span class="m02-term" data-term="rate" tabindex="0" role="button" aria-label="rate">rate</span> \u00b7
              <span class="m02-term" data-term="dt" tabindex="0" role="button" aria-label="dt">dt</span>
              <span class="m02-val" id="m02-eq-p">p = 0.050</span></div>
            <div><span class="m02-term" data-term="W" tabindex="0" role="button" aria-label="window width">W</span> /
              <span class="m02-term" data-term="dt" tabindex="0" role="button" aria-label="dt">dt</span>
              <span class="m02-val" id="m02-eq-n">n = 20</span></div>
            <div><span class="m02-term" data-term="lambda" tabindex="0" role="button" aria-label="lambda">\u03bb</span> = n\u00b7p
              <span class="m02-val" id="m02-eq-lambda">\u03bb = 1.00</span></div>
            <div>Var = n\u00b7p\u00b7(1\u2212p)
              <span class="m02-val" id="m02-eq-var">0.95</span></div>
            <div>TV(binomial, Poisson)
              <span class="m02-val" id="m02-eq-tv">0.0179</span></div>
          </div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.readoutsTitle}</div>
          <div class="m02-eq">
            <div>counts mean (th) <span class="m02-val" id="m02-ro-cmean">1.00 (1.00)</span></div>
            <div>counts var (th) <span class="m02-val" id="m02-ro-cvar">0.95 (0.95)</span></div>
            <div>ISI mean <span class="m02-val" id="m02-ro-imean">100.0 ms</span></div>
            <div>ISI CV (th 1) <span class="m02-val" id="m02-ro-icv">0.964</span></div>
            <div>sums mean (th) <span class="m02-val" id="m02-ro-smean">10.0 (10.0)</span></div>
            <div>sums sd (th) <span class="m02-val" id="m02-ro-ssd">3.08 (3.08)</span></div>
            <div>sums skew <span class="m02-val" id="m02-ro-sskew">0.29</span></div>
          </div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.predictTitle}</div>
          <div class="m02-caption" id="m02-predict-q">${S.predictQuestion}</div>
          <div class="m02-choice" id="m02-choices"></div>
          <div class="m02-note" id="m02-predict-a"></div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.controls}</div>
          <div class="m02-row"><label for="m02-rate">${S.rateLabel}</label>
            <input type="range" id="m02-rate" min="1" max="60" step="1" value="10" aria-label="rate in Hz">
            <span class="m02-val" id="m02-rate-val">10</span></div>
          <div class="m02-row"><label for="m02-dt">${S.dtLabel}</label>
            <input type="range" id="m02-dt" min="-200" max="200" step="1" value="0" aria-label="step size in ms">
            <span class="m02-val" id="m02-dt-val">5.0</span></div>
          <div class="m02-row"><label for="m02-bin">${S.binLabel}</label>
            <input type="range" id="m02-bin" min="10" max="1500" step="10" value="100" aria-label="window width in ms">
            <span class="m02-val" id="m02-bin-val">100</span></div>
          <div class="m02-nav">
            <button class="m02-btn" id="m02-more">${S.runMore}</button>
            <button class="m02-btn" id="m02-reseed">${S.reseed}</button>
          </div>
          <div class="m02-nav">
            <button class="m02-btn" id="m02-theory" aria-pressed="true">${S.theory}</button>
            <button class="m02-btn" id="m02-pause">${S.pause}</button>
          </div>
          <div class="m02-hint">${S.hintHero}</div>
          <div class="m02-hint">${S.hintClt}</div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.glossaryTitle}</div>
          ${glossary}
        </div>
        <div class="atlas-margin-section">
          <div class="m02-note">${S.footer}</div>
        </div>
      `;

      const id = (x) => document.getElementById(x);
      refs.chapterLabel = id('m02-chapter-label');
      refs.caption = id('m02-caption');
      refs.prev = id('m02-prev');
      refs.next = id('m02-next');
      refs.eqP = id('m02-eq-p');
      refs.eqN = id('m02-eq-n');
      refs.eqLambda = id('m02-eq-lambda');
      refs.eqVar = id('m02-eq-var');
      refs.eqTv = id('m02-eq-tv');
      refs.roCmean = id('m02-ro-cmean');
      refs.roCvar = id('m02-ro-cvar');
      refs.roImean = id('m02-ro-imean');
      refs.roIcv = id('m02-ro-icv');
      refs.roSmean = id('m02-ro-smean');
      refs.roSsd = id('m02-ro-ssd');
      refs.roSskew = id('m02-ro-sskew');
      refs.choices = id('m02-choices');
      refs.predictA = id('m02-predict-a');
      refs.rate = id('m02-rate');
      refs.dt = id('m02-dt');
      refs.bin = id('m02-bin');
      refs.rateVal = id('m02-rate-val');
      refs.dtVal = id('m02-dt-val');
      refs.binVal = id('m02-bin-val');
      refs.more = id('m02-more');
      refs.reseed = id('m02-reseed');
      refs.theory = id('m02-theory');
      refs.pause = id('m02-pause');

      // dt slider: 5 ms at the centre, down to 0.1 ms, up to 100 ms.
      refs.dt.addEventListener('input', () => {
        const v = Number(refs.dt.value);
        state.dt = v >= 0 ? 5 - (v / 200) * 4.9 : 5 + (-v / 200) * 95;
        refs.dtVal.textContent = state.dt.toFixed(state.dt < 1 ? 2 : 1);
        state.dirty = true;
      });
      refs.rate.addEventListener('input', () => {
        state.rate = Number(refs.rate.value);
        refs.rateVal.textContent = String(Math.round(state.rate));
        state.dirty = true;
      });
      refs.bin.addEventListener('input', () => {
        state.binMs = Number(refs.bin.value);
        refs.binVal.textContent = String(Math.round(state.binMs));
        state.dirty = true;
      });
      refs.more.addEventListener('click', () => {
        state.nBinsExtra = Math.min(state.nBinsExtra + 1, 20);
        state.dirty = true;
      });
      refs.reseed.addEventListener('click', () => {
        state.seed = (state.seed * 1103515245 + 12345) >>> 0;
        reset();
      });
      refs.theory.addEventListener('click', () => {
        state.theory = !state.theory;
        refs.theory.classList.toggle('active', state.theory);
        refs.theory.setAttribute('aria-pressed', String(state.theory));
      });
      refs.pause.addEventListener('click', () => togglePause());
      refs.prev.addEventListener('click', () => goChapter(state.chapter - 1));
      refs.next.addEventListener('click', () => goChapter(state.chapter + 1));

      document.querySelectorAll('.m02-term').forEach((el) => {
        const term = el.getAttribute('data-term');
        el.addEventListener('mouseenter', () => setHover(term));
        el.addEventListener('mouseleave', () => setHover(null));
        el.addEventListener('focus', () => setHover(term));
        el.addEventListener('blur', () => setHover(null));
      });

      buildChoices();
    }

    function setHover(term) {
      state.hover = term;
      if (marginEl) {
        document.querySelectorAll('.m02-term').forEach((el) => {
          el.classList.toggle('active', el.getAttribute('data-term') === term);
        });
      }
    }

    function togglePause() {
      state.paused = !state.paused;
      if (refs.pause) refs.pause.textContent = state.paused ? S.play : S.pause;
    }

    function buildChoices() {
      if (!refs.choices) return;
      refs.choices.innerHTML = '';
      [0.14, 0.37, 0.61].forEach((v) => {
        const b = document.createElement('button');
        b.className = 'm02-btn';
        b.textContent = v.toFixed(2);
        b.setAttribute('aria-label', `choose probability ${v.toFixed(2)}`);
        b.addEventListener('click', () => {
          if (state.predictChoice !== null) return;
          state.predictChoice = v;
          revealPredict();
        });
        refs.choices.appendChild(b);
      });
    }

    function revealPredict() {
      const p0p = poissonPmf(0, 1);
      const p0b = binomialPmf(0, 20, 0.05);
      if (refs.predictA) {
        const tag = state.predictChoice === 0.37 ? 'correct' : 'not it';
        refs.predictA.textContent =
          `you chose ${state.predictChoice.toFixed(2)} (${tag}); truth: ` +
          `${S.predictTruthBinom} ${p0b.toFixed(3)}, ` +
          `${S.predictTruthPoisson} ${p0p.toFixed(3)} = e\u207b\u00b9`;
      }
      document.querySelectorAll('#m02-choices .m02-btn').forEach((b, i) => {
        b.classList.toggle('active', [0.14, 0.37, 0.61][i] === state.predictChoice);
        b.disabled = true;
        b.setAttribute('aria-disabled', 'true');
      });
    }

    function goChapter(i) {
      const n = CHAPTERS.length;
      state.chapter = clamp(i, 0, n - 1);
      const ch = CHAPTERS[state.chapter];
      state.rate = ch.rate;
      state.dt = ch.dt;
      state.binMs = ch.binMs;
      state.m = ch.m;
      state.highlight = ch.hl;
      state.theory = true;
      if (refs.rate) {
        refs.rate.value = String(ch.rate);
        refs.rateVal.textContent = String(ch.rate);
      }
      if (refs.bin) {
        refs.bin.value = String(ch.binMs);
        refs.binVal.textContent = String(ch.binMs);
      }
      if (refs.dt) {
        refs.dt.value = String(ch.dt >= 5 ? 0 : Math.round(((5 - ch.dt) / 4.9) * 200));
        refs.dtVal.textContent = ch.dt.toFixed(ch.dt < 1 ? 2 : 1);
      }
      if (refs.caption) refs.caption.textContent = S.chapterCaps[state.chapter];
      if (refs.chapterLabel) {
        refs.chapterLabel.textContent = S.chapterOf
          .replace('{i}', String(state.chapter + 1))
          .replace('{n}', String(n));
      }
      if (refs.theory) {
        refs.theory.classList.add('active');
        refs.theory.setAttribute('aria-pressed', 'true');
      }
      state.dirty = true;
    }

    // ------------------------------------------------------------ derived data
    function recomputeSums() {
      if (!state.counts || state.countsBins === 0) {
        state.sums = null;
        return;
      }
      const maxM = Math.max(1, Math.min(60, Math.floor(state.countsBins / 60)));
      state.sumsM = clamp(Math.round(state.m), 1, Math.max(1, maxM));
      state.sums = sumGroups(state.counts, state.sumsM);
    }

    function rebuild() {
      const p = clamp((state.rate * state.dt) / 1000, 0, 1);
      const nPerBin = Math.max(1, Math.round(state.binMs / state.dt));
      state.p = p;
      state.nPerBin = nPerBin;
      state.lambda = nPerBin * p;

      const targetBins = 2000 + 1000 * state.nBinsExtra;
      const budget = state.drag ? 600000 : 1400000;
      const bins = Math.max(80, Math.min(targetBins, Math.floor(budget / nPerBin)));
      state.countsBins = bins;
      state.counts = binCounts(state.seed, p, nPerBin, bins);

      const maxSteps = 1500000;
      const steps = Math.min(maxSteps, Math.ceil(20020 / Math.max(p, 1e-9)));
      const train = simulateTrain((state.seed ^ 0x9e3779b9) >>> 0, p, steps);
      const gaps = interSpikeIntervals(train);
      const isiMs = new Float64Array(gaps.length);
      for (let i = 0; i < gaps.length; i++) isiMs[i] = gaps[i] * state.dt;
      state.isi = isiMs;
      state.isiN = isiMs.length;
      state.isiMeanMs = mean(isiMs);
      state.isiCV = coefficientOfVariation(isiMs);

      try {
        state.tv = binomialToPoissonTV(state.lambda, nPerBin);
      } catch (err) {
        state.tv = NaN;
      }

      recomputeSums();
      state.dirty = false;
      updateMarginValues();
    }

    // Simulate one full view of history so the very first frame is already raining.
    function prefill() {
      state.drops = [];
      const dt = state.dt;
      const n = Math.max(0, Math.floor(state.simT / dt));
      for (let i = 0; i <= n; i++) {
        if (state.rng() < state.p) state.drops.push(i * dt);
      }
      state.nextStep = (n + 1) * dt;
      updateWin();
    }

    function reset() {
      state.rng = mulberry32(state.seed);
      state.simT = PREFILL_MS;
      state.nBinsExtra = 0;
      rebuild();
      prefill();
    }

    // -------------------------------------------------------------- hero motion
    function updateWin() {
      const lo = state.simT - state.binMs;
      let c = 0;
      for (let d = state.drops.length - 1; d >= 0; d--) {
        const t = state.drops[d];
        if (t > state.simT) continue;
        if (t >= lo) c++;
        else break;
      }
      state.winCount = c;
    }

    function advance(msReal) {
      if (state.paused || !state.rng) return;
      const dtReal = Math.min(64, msReal);
      state.simT += dtReal;
      let guard = 0;
      while (state.nextStep <= state.simT && guard < 500000) {
        if (state.rng() < state.p) state.drops.push(state.nextStep);
        state.nextStep += state.dt;
        guard++;
      }
      const cutoff = state.simT - state.viewSpan - 400;
      let i = 0;
      while (i < state.drops.length && state.drops[i] < cutoff) i++;
      if (i > 512) state.drops = state.drops.slice(i);
      updateWin();
    }

    // ------------------------------------------------------------- draw helpers
    function roundRect(x, y, w, h, r) {
      const rr = Math.min(r, w / 2, h / 2);
      g.beginPath();
      g.moveTo(x + rr, y);
      g.arcTo(x + w, y, x + w, y + h, rr);
      g.arcTo(x + w, y + h, x, y + h, rr);
      g.arcTo(x, y + h, x, y, rr);
      g.arcTo(x, y, x + w, y, rr);
      g.closePath();
    }

    function label(txt, x, y, color, opts) {
      const o = opts || {};
      g.save();
      g.font = `${o.size || 13}px ${o.font || FONT_MONO}`;
      g.fillStyle = color;
      g.textAlign = o.align || 'left';
      g.textBaseline = o.baseline || 'alphabetic';
      g.fillText(txt, x, y);
      g.restore();
    }

    function strokeLine(x0, y0, x1, y1, color, width, dash) {
      g.save();
      g.strokeStyle = color;
      g.lineWidth = width || 1;
      if (dash) g.setLineDash(dash);
      g.beginPath();
      g.moveTo(x0, y0);
      g.lineTo(x1, y1);
      g.stroke();
      g.restore();
    }

    function panelFrame(box, c, title, subtitle, highlighted) {
      g.save();
      g.fillStyle = c.paperCard || c.paper;
      g.strokeStyle = c.border;
      g.lineWidth = 1;
      roundRect(box.x, box.y, box.w, box.h, 4);
      g.fill();
      g.stroke();
      if (highlighted) {
        g.strokeStyle = c.exc || c.ink;
        g.lineWidth = 2;
        roundRect(box.x + 1, box.y + 1, box.w - 2, box.h - 2, 4);
        g.stroke();
      }
      g.restore();
      label(title, box.x + 10, box.y + 16, c.ink, { size: 13, font: FONT_SANS });
      if (subtitle) label(subtitle, box.x + 10, box.y + 31, c.inkMuted, { size: box.w < 240 ? 11 : 12 });
      return {
        x: box.x + 10,
        y: box.y + 38,
        w: box.w - 20,
        h: box.h - 46,
      };
    }

    function panelAxes(inner, c, yLabel, xLabel, showTicks) {
      const padL = 32;
      const padB = showTicks ? 32 : 16;
      const px = inner.x + padL;
      const py = inner.y;
      const pw = inner.w - padL;
      const ph = Math.max(8, inner.h - padB);
      strokeLine(px, py, px, py + ph, c.border, 1);
      strokeLine(px, py + ph, px + pw, py + ph, c.border, 1);
      label(yLabel, inner.x, py + 4, c.inkMuted, { size: 13 });
      // Axis title on its own row below the tick labels, so they never collide.
      if (xLabel) {
        label(xLabel, px + pw / 2, py + ph + (showTicks ? 30 : 13), c.inkMuted, {
          size: showTicks ? 13 : 12,
          align: 'center',
        });
      }
      return { px, py, pw, ph };
    }

    function drawHero(box, c) {
      const highlighted =
        state.highlight === 'hero' ||
        state.hover === 'rate' ||
        state.hover === 'dt' ||
        state.hover === 'W';
      g.save();
      g.fillStyle = c.paperCard || c.paper;
      g.strokeStyle = c.border;
      roundRect(box.x, box.y, box.w, box.h, 4);
      g.fill();
      g.stroke();
      if (highlighted) {
        g.strokeStyle = c.exc || c.ink;
        g.lineWidth = 2;
        roundRect(box.x + 1, box.y + 1, box.w - 2, box.h - 2, 4);
        g.stroke();
      }
      g.restore();

      label(S.hero, box.x + 10, box.y + 18, c.ink, { size: 14, font: FONT_SANS });
      label(
        `${fmt(state.rate, 0)} Hz \u00b7 dt ${fmt(state.dt, state.dt < 1 ? 2 : 1)} ms \u00b7 W ${fmt(state.binMs, 0)} ms`,
        box.x + box.w - 10,
        box.y + 18,
        c.inkMuted,
        { size: 13, align: 'right' }
      );

      const x0 = box.x + 16;
      const x1 = box.x + box.w - 16;
      const compact = box.h < 150;
      const subY = box.y + (compact ? 32 : 37);
      const cloudY = box.y + (compact ? 48 : 64);
      const axisY = box.y + box.h - (compact ? 30 : 40);
      const span = state.viewSpan;
      const xOf = (t) => x1 - ((state.simT - t) / span) * (x1 - x0);

      const stepsInView = span / state.dt;
      // Coin-flip ruler note on its own line under the title (never on the axis).
      label(
        stepsInView <= 800
          ? `one coin flip every ${fmt(state.dt, state.dt < 1 ? 2 : 1)} ms`
          : `${S.bandNote} (${Math.round(stepsInView)} in view)`,
        x0,
        subY,
        c.inkMuted,
        { size: 13 }
      );

      strokeLine(x0, axisY, x1, axisY, c.border, 1.5);

      if (stepsInView <= 800) {
        g.save();
        g.strokeStyle = c.border;
        g.lineWidth = 1;
        g.globalAlpha = 0.6;
        g.beginPath();
        const t0 = Math.ceil((state.simT - span) / state.dt) * state.dt;
        for (let t = t0; t <= state.simT; t += state.dt) {
          const x = xOf(t);
          g.moveTo(x, axisY - 6);
          g.lineTo(x, axisY);
        }
        g.stroke();
        g.restore();
      } else {
        g.save();
        g.fillStyle = c.border;
        g.globalAlpha = 0.5;
        g.fillRect(x0, axisY - 5, x1 - x0, 5);
        g.restore();
      }

      // The cloud that emits the drops (drawn first so the rain falls in front of it).
      g.save();
      g.beginPath();
      g.moveTo(x0, cloudY);
      for (let cx = x0 + 12; cx < x1 - 6; cx += 22) {
        g.arc(cx, cloudY, 11 + 3 * Math.sin(cx * 0.09), Math.PI, 0);
      }
      g.lineTo(x1, cloudY);
      g.closePath();
      g.fillStyle = c.paperCard || c.paper;
      g.fill();
      g.strokeStyle = c.inkMuted;
      g.lineWidth = 1.2;
      g.stroke();
      g.restore();

      g.save();
      g.strokeStyle = c.border;
      g.lineWidth = 1;
      g.setLineDash([3, 4]);
      g.globalAlpha = 0.8;
      g.beginPath();
      const firstBin = Math.ceil((state.simT - span) / state.binMs) * state.binMs;
      for (let t = firstBin; t <= state.simT; t += state.binMs) {
        const x = xOf(t);
        g.moveTo(x, cloudY);
        g.lineTo(x, axisY);
      }
      g.stroke();
      g.restore();

      const wl = Math.max(x0, xOf(state.simT - state.binMs));
      const wr = xOf(state.simT);
      g.save();
      g.fillStyle = c.volt;
      g.globalAlpha = 0.1;
      g.fillRect(wl, cloudY, wr - wl, axisY - cloudY);
      g.restore();
      strokeLine(wl, cloudY, wl, axisY, c.volt, 2);
      strokeLine(wr, cloudY, wr, axisY, c.volt, 2);

      // Rain: a landed drop is a dash on the timeline, a young drop is a falling streak.
      const fall = 850;
      g.save();
      g.lineWidth = 2;
      for (let i = 0; i < state.drops.length; i++) {
        const t = state.drops[i];
        const age = state.simT - t;
        if (age < 0 || age > span + fall) continue;
        const x = xOf(t);
        g.strokeStyle = c.volt;
        if (age >= fall) {
          g.globalAlpha = 0.85;
          g.beginPath();
          g.moveTo(x, axisY - 6);
          g.lineTo(x, axisY + 2);
          g.stroke();
        } else {
          const frac = age / fall;
          const y = cloudY + (axisY - cloudY) * frac;
          g.globalAlpha = 0.95;
          g.beginPath();
          g.moveTo(x + 3, y - 15);
          g.lineTo(x, y);
          g.stroke();
          g.beginPath();
          g.arc(x, y, 2.2, 0, Math.PI * 2);
          g.fillStyle = c.volt;
          g.fill();
        }
      }
      g.restore();

      // Draggable handle on the window's left edge.
      const midY = (cloudY + axisY) / 2;
      g.save();
      g.fillStyle = c.volt;
      roundRect(wl - 3, midY - 10, 6, 20, 3);
      g.fill();
      g.restore();

      // Window readout, right-aligned so it never clips at the figure edge.
      label(
        `${S.windowLabel}: N = ${state.winCount} ${S.spikes}`,
        x1,
        cloudY + 16,
        c.volt,
        { size: 13, align: 'right' }
      );

      label('now', x1, axisY + 17, c.inkMuted, { size: 13, align: 'right' });
      label(`${fmt(span / 1000, 1)} s shown`, x0, axisY + 17, c.inkMuted, { size: 13 });

      state.heroGeom = { x0, x1, axisY, cloudY, xOf, wl };
    }

    function drawCounts(box, c) {
      const hl =
        state.highlight === 'counts' ||
        state.hover === 'counts' ||
        state.hover === 'n' ||
        state.hover === 'p' ||
        state.hover === 'lambda';
      const inner = panelFrame(
        box,
        c,
        S.countsTitle,
        box.w < 240
          ? `n ${state.nPerBin}  p ${fmt(state.p, 3)}  \u03bb ${fmt(state.lambda, 2)}`
          : `n=${state.nPerBin}  p=${fmt(state.p, 3)}  \u03bb=${fmt(state.lambda, 2)}  TV=${fmt(state.tv, 4)}`,
        hl
      );
      const counts = state.counts;
      const N = state.countsBins;
      if (!counts || N === 0) {
        label(S.noData, box.x + box.w / 2, box.y + box.h / 2, c.inkMuted, { align: 'center' });
        return;
      }
      let maxObs = 0;
      for (let i = 0; i < N; i++) if (counts[i] > maxObs) maxObs = counts[i];
      const Kmax = Math.min(
        60,
        Math.max(4, maxObs, Math.ceil(state.lambda + 8 * Math.sqrt(state.lambda)) + 1)
      );
      const freq = new Float64Array(Kmax + 1);
      for (let i = 0; i < N; i++) freq[Math.min(counts[i], Kmax)]++;
      for (let k = 0; k <= Kmax; k++) freq[k] /= N;
      const bp = (k) => binomialPmf(k, state.nPerBin, state.p);
      const pp = (k) => poissonPmf(k, state.lambda);
      let yMax = 0;
      for (let k = 0; k <= Kmax; k++) yMax = Math.max(yMax, freq[k], bp(k), pp(k));
      yMax = yMax > 0 ? yMax * 1.2 : 1;

      const showTicks = box.h >= 96;
      const a = panelAxes(inner, c, 'P(count)', 'count k', showTicks);
      const slot = a.pw / (Kmax + 1);
      const xk = (k) => a.px + (k + 0.5) * slot;
      const yv = (v) => a.py + a.ph - (v / yMax) * a.ph;

      g.save();
      g.fillStyle = c.volt;
      g.globalAlpha = 0.55;
      for (let k = 0; k <= Kmax; k++) {
        g.fillRect(xk(k) - slot * 0.35, yv(freq[k]), slot * 0.7, (freq[k] / yMax) * a.ph);
      }
      g.restore();

      if (state.theory) {
        g.save();
        g.strokeStyle = c.theory;
        g.lineWidth = 1.5;
        g.setLineDash([5, 4]);
        g.beginPath();
        for (let k = 0; k <= Kmax; k++) {
          const X = xk(k);
          const Y = yv(bp(k));
          if (k === 0) g.moveTo(X, Y);
          else g.lineTo(X, Y);
        }
        g.stroke();
        g.restore();
      }

      g.save();
      g.fillStyle = c.exc;
      for (let k = 0; k <= Kmax; k++) {
        g.beginPath();
        g.arc(xk(k), yv(pp(k)), 2.4, 0, Math.PI * 2);
        g.fill();
      }
      g.restore();

      if (state.lambda <= Kmax) {
        strokeLine(xk(state.lambda), a.py, xk(state.lambda), a.py + a.ph, c.inkMuted, 1, [2, 3]);
        label('\u03bb', xk(state.lambda), a.py + 13, c.inkMuted, { size: 13, align: 'center' });
      }

      if (showTicks) {
        const stride = Kmax > 14 ? 5 : Kmax > 8 ? 2 : 1;
        for (let k = 0; k <= Kmax; k += stride) {
          label(String(k), xk(k), a.py + a.ph + 16, c.inkMuted, { size: 13, align: 'center' });
        }
      }
    }

    function drawIsi(box, c) {
      const hl = state.highlight === 'isi' || state.hover === 'isi';
      const inner = panelFrame(
        box,
        c,
        S.isiTitle,
        box.w < 240
          ? `\u03bc ${fmt(state.isiMeanMs, 1)} ms  CV ${fmt(state.isiCV, 3)}`
          : `mean ${fmt(state.isiMeanMs, 1)} ms (th ${fmt(1000 / state.rate, 1)})  CV ${fmt(state.isiCV, 3)}`,
        hl
      );
      const isi = state.isi;
      const N = state.isiN;
      if (!isi || N < 2) {
        label(S.noData, box.x + box.w / 2, box.y + box.h / 2, c.inkMuted, { align: 'center' });
        return;
      }
      const rateHz = state.rate;
      const meanMsTheory = rateHz > 0 ? 1000 / rateHz : state.isiMeanMs;
      const Tmax = Math.max(1, meanMsTheory * 5);
      const nb = 24;
      const bw = Tmax / nb;
      const hist = new Float64Array(nb);
      for (let i = 0; i < N; i++) hist[Math.min(nb - 1, Math.floor(isi[i] / bw))]++;
      const dens = new Float64Array(nb);
      for (let b = 0; b < nb; b++) dens[b] = hist[b] / (N * bw);
      const geoDens = new Float64Array(nb);
      const expDens = new Float64Array(nb);
      const ratePerMs = rateHz / 1000;
      for (let b = 0; b < nb; b++) {
        const loMs = b * bw;
        const hiMs = (b + 1) * bw;
        let mass = 0;
        const kLo = Math.max(1, Math.ceil(loMs / state.dt));
        const kHi = Math.floor(hiMs / state.dt);
        for (let k = kLo; k <= kHi && mass < 1; k++) mass += geometricPmf(k, state.p);
        geoDens[b] = mass / bw;
        expDens[b] = exponentialPdf(loMs + bw / 2, ratePerMs);
      }
      let yMax = 0;
      for (let b = 0; b < nb; b++) yMax = Math.max(yMax, dens[b], geoDens[b], expDens[b]);
      yMax = yMax > 0 ? yMax * 1.2 : 1;

      const showTicks = box.h >= 96;
      const a = panelAxes(inner, c, 'density /ms', 'gap (ms)', showTicks);
      const slot = a.pw / nb;
      const xb = (b) => a.px + (b + 0.5) * slot;
      const yv = (v) => a.py + a.ph - (v / yMax) * a.ph;

      g.save();
      g.fillStyle = c.trace;
      g.globalAlpha = 0.5;
      for (let b = 0; b < nb; b++) {
        g.fillRect(xb(b) - slot * 0.4, yv(dens[b]), slot * 0.8, (dens[b] / yMax) * a.ph);
      }
      g.restore();

      if (state.theory) {
        g.save();
        g.strokeStyle = c.theory;
        g.lineWidth = 1.5;
        g.setLineDash([5, 4]);
        g.beginPath();
        for (let b = 0; b < nb; b++) {
          const X = xb(b);
          const Y = yv(geoDens[b]);
          if (b === 0) g.moveTo(X, Y);
          else g.lineTo(X, Y);
        }
        g.stroke();
        g.restore();
      }
      g.save();
      g.fillStyle = c.exc;
      for (let b = 0; b < nb; b++) {
        g.beginPath();
        g.arc(xb(b), yv(expDens[b]), 2.2, 0, Math.PI * 2);
        g.fill();
      }
      g.restore();

      if (showTicks) {
        for (const tv of [0, Tmax / 2, Tmax]) {
          const x = a.px + (tv / Tmax) * a.pw;
          label(fmt(tv, 0), x, a.py + a.ph + 16, c.inkMuted, { size: 13, align: 'center' });
        }
      }
    }

    function drawClt(box, c) {
      const hl = state.highlight === 'clt' || state.hover === 'clt';
      const sums = state.sums;
      const m = state.sumsM;
      const meanT = m * state.lambda;
      const sdT = Math.sqrt(Math.max(m * state.nPerBin * state.p * (1 - state.p), 1e-9));
      const inner = panelFrame(
        box,
        c,
        S.cltTitle,
        box.w < 240
          ? `m ${m} \u00b7 \u03bc ${fmt(meanT, 1)} \u00b7 \u03c3 ${fmt(sdT, 2)}`
          : `m=${m}  mean ${fmt(meanT, 1)}  sd ${fmt(sdT, 2)}`,
        hl
      );
      if (!sums || sums.length < 20) {
        label(S.noData, box.x + box.w / 2, box.y + box.h / 2, c.inkMuted, { align: 'center' });
        return;
      }
      const lo = Math.max(-0.5, meanT - 4.2 * sdT);
      const hi = meanT + 4.2 * sdT;
      const nb = 26;
      const bw = (hi - lo) / nb;
      const hist = new Float64Array(nb + 1);
      for (let i = 0; i < sums.length; i++) hist[clamp(Math.floor((sums[i] - lo) / bw), 0, nb)]++;
      const dens = new Float64Array(nb + 1);
      for (let b = 0; b <= nb; b++) dens[b] = hist[b] / (sums.length * bw);
      let yMax = 0;
      for (let b = 0; b <= nb; b++) {
        yMax = Math.max(yMax, dens[b], normalPdf(lo + (b + 0.5) * bw, meanT, sdT));
      }
      yMax = yMax > 0 ? yMax * 1.2 : 1;

      const showTicks = box.h >= 96;
      const a = panelAxes(inner, c, 'density', 'sum of m windows', showTicks);
      const slot = a.pw / (nb + 1);
      const xb = (b) => a.px + (b + 0.5) * slot;
      const yv = (v) => a.py + a.ph - (v / yMax) * a.ph;

      g.save();
      g.fillStyle = c.mod;
      g.globalAlpha = 0.5;
      for (let b = 0; b <= nb; b++) {
        g.fillRect(xb(b) - slot * 0.4, yv(dens[b]), slot * 0.8, (dens[b] / yMax) * a.ph);
      }
      g.restore();

      if (state.theory) {
        g.save();
        g.strokeStyle = c.theory;
        g.lineWidth = 1.5;
        g.setLineDash([5, 4]);
        g.beginPath();
        for (let b = 0; b <= nb; b++) {
          const X = xb(b);
          const Y = yv(normalPdf(lo + (b + 0.5) * bw, meanT, sdT));
          if (b === 0) g.moveTo(X, Y);
          else g.lineTo(X, Y);
        }
        g.stroke();
        g.restore();
      }

      if (showTicks) {
        for (const tv of [lo, (lo + hi) / 2, hi]) {
          const x = a.px + ((tv - lo) / (hi - lo)) * a.pw;
          label(fmt(tv, 1), x, a.py + a.ph + 16, c.inkMuted, { size: 13, align: 'center' });
        }
      }
    }

    // ------------------------------------------------------------------- render
    function fitCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(rect.width));
      const h = Math.max(1, Math.round(rect.height));
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { w, h };
    }

    function render() {
      const { w, h } = fitCanvas();
      if (w <= 1 || h <= 1) return;
      const c = colors();
      g.clearRect(0, 0, w, h);
      const pad = 14;
      const x0 = pad;
      const y0 = pad;
      const W = w - 2 * pad;
      const H = h - 2 * pad;
      const narrow = w < 620;
      let hero;
      let boxes;
      if (narrow) {
        // Phone: hero on top, then counts + ISI side by side and the CLT full width.
        const heroH = H * 0.42;
        hero = { x: x0, y: y0, w: W, h: heroH };
        const gap = 8;
        const remY = y0 + heroH + 8;
        const remH = H - heroH - 8;
        const r1 = remH * 0.5 - 4;
        const r2 = remH - r1 - 8;
        const bw = (W - gap) / 2;
        boxes = [
          { x: x0, y: remY, w: bw, h: r1 },
          { x: x0 + bw + gap, y: remY, w: bw, h: r1 },
          { x: x0, y: remY + r1 + 8, w: W, h: r2 },
        ];
      } else {
        const heroH = H * 0.55;
        hero = { x: x0, y: y0, w: W, h: heroH };
        const gap = 10;
        const bw = (W - 2 * gap) / 3;
        const by = y0 + heroH + 8;
        const bh = H - heroH - 8;
        boxes = [
          { x: x0, y: by, w: bw, h: bh },
          { x: x0 + bw + gap, y: by, w: bw, h: bh },
          { x: x0 + 2 * (bw + gap), y: by, w: bw, h: bh },
        ];
      }
      state.hit = { hero, boxes };
      drawHero(hero, c);
      drawCounts(boxes[0], c);
      drawIsi(boxes[1], c);
      drawClt(boxes[2], c);
      updateMarginValues();
    }

    function updateMarginValues() {
      if (!refs.eqP) return;
      refs.eqP.textContent = `p = ${fmt(state.p, 3)}`;
      refs.eqN.textContent = `n = ${state.nPerBin}`;
      refs.eqLambda.textContent = `\u03bb = ${fmt(state.lambda, 2)}`;
      refs.eqVar.textContent = fmt(state.nPerBin * state.p * (1 - state.p), 3);
      refs.eqTv.textContent = fmt(state.tv, 4);

      if (state.counts && state.countsBins > 0) {
        refs.roCmean.textContent = `${fmt(mean(state.counts), 3)} (${fmt(state.lambda, 3)})`;
        refs.roCvar.textContent = `${fmt(variance(state.counts), 3)} (${fmt(
          state.nPerBin * state.p * (1 - state.p),
          3
        )})`;
      }
      refs.roImean.textContent = `${fmt(state.isiMeanMs, 1)} ms`;
      refs.roIcv.textContent = fmt(state.isiCV, 3);
      if (state.sums && state.sums.length > 0) {
        const meanT = state.sumsM * state.lambda;
        const sdT = Math.sqrt(state.sumsM * state.nPerBin * state.p * (1 - state.p));
        refs.roSmean.textContent = `${fmt(mean(state.sums), 2)} (${fmt(meanT, 2)})`;
        refs.roSsd.textContent = `${fmt(std(state.sums), 2)} (${fmt(sdT, 2)})`;
        refs.roSskew.textContent = fmt(skewness(state.sums), 2);
      }
    }

    // -------------------------------------------------------------- interaction
    function toLocal(e) {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function inBox(p, b) {
      return p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h;
    }

    // On phones give the figure enough height for a hero plus three legible panels.
    function applyResponsive() {
      if (!fig || !fig.style) return;
      if (window.innerWidth <= 720) {
        fig.style.aspectRatio = 'auto';
        fig.style.height = '600px';
      } else {
        fig.style.aspectRatio = '';
        fig.style.height = '';
      }
    }
    window.addEventListener('resize', applyResponsive);

    // Pointer capture fails for synthetic events without a pointerId; ignore it there.
    function capturePointer(e) {
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch (err) {
        /* no active pointer */
      }
    }

    canvas.addEventListener('pointerdown', (e) => {
      const p = toLocal(e);
      const hit = state.hit;
      if (!hit) return;
      if (inBox(p, hit.hero)) {
        const geo = state.heroGeom;
        if (geo && Math.abs(p.x - Math.max(geo.x0, geo.xOf(state.simT - state.binMs))) < 12) {
          state.drag = { kind: 'bin' };
          capturePointer(e);
          return;
        }
        // Grabbing the cloud changes the rate; the tape below injects a spike.
        if (geo && p.y < geo.cloudY + 10) {
          state.drag = { kind: 'rate', start: p.y, rate0: state.rate };
          capturePointer(e);
          return;
        }
        if (geo) {
          const t = state.simT - ((geo.x1 - p.x) / (geo.x1 - geo.x0)) * state.viewSpan;
          if (t <= state.simT && t >= state.simT - state.viewSpan && state.rng) {
            state.drops.push(t);
            state.drops.sort((a, b) => a - b);
            updateWin();
          }
        }
        return;
      }
      if (inBox(p, hit.boxes[2])) {
        state.drag = { kind: 'm', start: p.y, m0: state.m };
        capturePointer(e);
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      const p = toLocal(e);
      const geo = state.heroGeom;
      if (!state.drag) {
        // Cursor hint for the draggable parts.
        const hit = state.hit;
        let cursor = 'crosshair';
        if (hit && geo && inBox(p, hit.hero)) {
          if (Math.abs(p.x - Math.max(geo.x0, geo.xOf(state.simT - state.binMs))) < 12) {
            cursor = 'ew-resize';
          } else if (p.y < geo.cloudY + 10) {
            cursor = 'ns-resize';
          }
        } else if (hit && inBox(p, hit.boxes[2])) {
          cursor = 'ns-resize';
        }
        canvas.style.cursor = cursor;
        return;
      }
      if (state.drag.kind === 'bin' && geo) {
        const binMs = ((geo.x1 - p.x) / (geo.x1 - geo.x0)) * state.viewSpan;
        state.binMs = clamp(binMs, 10, 1500);
        if (refs.bin) {
          refs.bin.value = String(Math.round(state.binMs));
          refs.binVal.textContent = String(Math.round(state.binMs));
        }
        rebuild();
      } else if (state.drag.kind === 'rate') {
        state.rate = clamp(state.drag.rate0 + (state.drag.start - p.y) * 0.35, 1, 60);
        if (refs.rate) {
          refs.rate.value = String(Math.round(state.rate));
          refs.rateVal.textContent = String(Math.round(state.rate));
        }
        rebuild();
      } else if (state.drag.kind === 'm') {
        state.m = clamp(Math.round(state.drag.m0 + (state.drag.start - p.y) / 6), 1, 60);
        recomputeSums();
      }
    });

    const endDrag = () => {
      if (state.drag) {
        state.drag = null;
        rebuild();
      }
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', 'Spike rain timeline with count, interval and sum histograms');

    window.addEventListener('keydown', (e) => {
      const tag = e.target && e.target.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goChapter(state.chapter - 1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        goChapter(state.chapter + 1);
      } else if (e.code === 'Space') {
        togglePause();
      }
    });

    window.addEventListener('atlas:theory', (e) => {
      state.theory = !!(e.detail && e.detail.active);
      if (refs.theory) {
        refs.theory.classList.toggle('active', state.theory);
        refs.theory.setAttribute('aria-pressed', String(state.theory));
      }
    });

    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      state.paused = true;
      if (refs.pause) refs.pause.textContent = S.play;
    }

    // ------------------------------------------------------------------- loop
    let lastTs = null;
    function frame(ts) {
      if (state.dirty) rebuild();
      if (lastTs === null) lastTs = ts;
      const dtMs = ts - lastTs;
      lastTs = ts;
      advance(dtMs);
      render();
      requestAnimationFrame(frame);
    }

    goChapter(0);
    reset();
    applyResponsive();
    render();
    requestAnimationFrame(frame);

    if (ctx.onFrame) ctx.onFrame(() => render());
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core: core, S: S };
  }
  if (typeof Atlas !== 'undefined' && Atlas.Plate) {
    Atlas.Plate.register({ id: 'M02', title: { en: S.title }, source: 'live', mount: mount });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
