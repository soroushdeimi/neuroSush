/**
 * Core tests for plate M02: the Bernoulli spike train, its binomial/geometric statistics
 * and the dt -> 0 Poisson/exponential limits. Pure math only; no DOM.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core, S } = require('../../../src/neurosush/viz/web/plates/M02.js');

// ---------------------------------------------------------------- pmf values

test('binomialPmf matches hand-computed values', () => {
  // C(4,1) * 0.5^4 = 4 / 16
  assert.ok(Math.abs(core.binomialPmf(1, 4, 0.5) - 0.25) < 1e-12);
  // C(4,0) * 0.2^0 * 0.8^4 = 0.4096
  assert.ok(Math.abs(core.binomialPmf(0, 4, 0.2) - 0.4096) < 1e-12);
  // C(5,2) * 0.3^2 * 0.7^3 = 10 * 0.09 * 0.343
  assert.ok(Math.abs(core.binomialPmf(2, 5, 0.3) - 10 * 0.09 * 0.343) < 1e-12);
});

test('poissonPmf matches hand-computed values', () => {
  assert.ok(Math.abs(core.poissonPmf(0, 1) - Math.exp(-1)) < 1e-14);
  assert.ok(Math.abs(core.poissonPmf(2, 1) - Math.exp(-1) / 2) < 1e-14);
  assert.ok(Math.abs(core.poissonPmf(3, 2) - (Math.exp(-2) * 8) / 6) < 1e-14);
});

test('geometricPmf and exponentialPdf match hand-computed values', () => {
  // P(k=3) = p (1-p)^2 = 0.1 * 0.81
  assert.ok(Math.abs(core.geometricPmf(3, 0.1) - 0.081) < 1e-12);
  assert.equal(core.geometricPmf(0, 0.1), 0);
  // Exponential density f(t) = rate e^(-rate t)
  assert.ok(Math.abs(core.exponentialPdf(0, 2) - 2) < 1e-14);
  assert.ok(Math.abs(core.exponentialPdf(1, 2) - 2 * Math.exp(-2)) < 1e-14);
});

test('normalPdf matches the standard normal peak', () => {
  assert.ok(Math.abs(core.normalPdf(0, 0, 1) - 1 / Math.sqrt(2 * Math.PI)) < 1e-14);
  assert.equal(core.normalPdf(1, 1, 3), core.normalPdf(0, 0, 3) * 1);
});

// ------------------------------------------------------------ normalization

test('binomial pmf sums to one within 1e-12', () => {
  for (const [n, p] of [
    [4, 0.5],
    [40, 0.05],
    [1000, 0.001],
    [200, 0.9],
  ]) {
    const s = core.pmfSum((k) => core.binomialPmf(k, n, p), 0, n);
    assert.ok(Math.abs(s - 1) < 1e-12, `binomial(${n}, ${p}) sum = ${s}`);
  }
  // degenerate cases
  assert.equal(core.pmfSum((k) => core.binomialPmf(k, 5, 0), 0, 5), 1);
  assert.equal(core.pmfSum((k) => core.binomialPmf(k, 5, 1), 0, 5), 1);
});

test('poisson pmf sums to one within 1e-12', () => {
  for (const lambda of [0.5, 1, 4, 25]) {
    const hi = Math.ceil(lambda + 12 * Math.sqrt(lambda) + 30);
    const s = core.pmfSum((k) => core.poissonPmf(k, lambda), 0, hi);
    assert.ok(Math.abs(s - 1) < 1e-12, `poisson(${lambda}) sum = ${s}`);
  }
  assert.equal(core.poissonPmf(0, 0), 1);
});

test('geometric pmf sums to one within 1e-12', () => {
  for (const p of [0.01, 0.1, 0.5]) {
    const hi = Math.ceil(Math.log(1e-15) / Math.log(1 - p));
    const s = core.pmfSum((k) => core.geometricPmf(k, p), 1, hi);
    assert.ok(Math.abs(s - 1) < 1e-12, `geometric(${p}) sum = ${s}`);
  }
});

test('normal density integrates to one', () => {
  const dx = 0.005;
  let s = 0;
  for (let x = -60; x <= 60; x += dx) s += core.normalPdf(x, 0, 1) * dx;
  assert.ok(Math.abs(s - 1) < 1e-6, `normal integral = ${s}`);
});

// ----------------------------------------------------------- Poisson moments

test('the Poisson pmf has mean equal to variance equal to lambda', () => {
  for (const lambda of [0.5, 1, 4, 25]) {
    const hi = Math.ceil(lambda + 12 * Math.sqrt(lambda) + 30);
    let m = 0;
    let m2 = 0;
    for (let k = 0; k <= hi; k++) {
      const q = core.poissonPmf(k, lambda);
      m += k * q;
      m2 += k * k * q;
    }
    assert.ok(Math.abs(m - lambda) < 1e-9, `mean ${m} vs ${lambda}`);
    assert.ok(Math.abs(m2 - m * m - lambda) < 1e-9, `var ${m2 - m * m} vs ${lambda}`);
  }
});

// ------------------------------------------------------- binomial -> Poisson

test('binomial-to-Poisson TV distance decreases monotonically as dt halves', () => {
  // For fixed lambda = rate * W, halving dt doubles n at p = lambda / n.
  const lambda = 1;
  const ns = [2, 4, 8, 16, 32, 64, 128, 256, 512];
  let prev = Infinity;
  for (const n of ns) {
    const d = core.binomialToPoissonTV(lambda, n);
    assert.ok(d < prev, `TV did not decrease: n=${n}, d=${d}, prev=${prev}`);
    prev = d;
  }
  assert.ok(prev < 1e-3, `TV at n=512 should be small, got ${prev}`);
});

test('binomial-to-Poisson TV is zero for lambda = 0', () => {
  assert.equal(core.binomialToPoissonTV(0, 8), 0);
});

// ------------------------------------------------------- simulated histogram

test('simulated window counts match the binomial pmf by chi-square', () => {
  // rate 10 Hz, dt 5 ms, window 100 ms -> n = 20 steps, p = 0.05, lambda = 1.
  const p = 0.05;
  const nPerBin = 20;
  const bins = 4000;
  const counts = core.binCounts(12345, p, nPerBin, bins);
  assert.equal(counts.length, bins);

  const cells = 6; // k = 0, 1, 2, 3, 4, >= 5
  const observed = new Float64Array(cells);
  for (let i = 0; i < bins; i++) observed[Math.min(counts[i], cells - 1)]++;

  let chi2 = 0;
  for (let k = 0; k < cells; k++) {
    const pm =
      k < cells - 1
        ? core.binomialPmf(k, nPerBin, p)
        : 1 - core.pmfSum((j) => core.binomialPmf(j, nPerBin, p), 0, cells - 2);
    const expected = pm * bins;
    assert.ok(expected > 5, `cell ${k} expected ${expected} too small to pool`);
    chi2 += ((observed[k] - expected) ** 2) / expected;
  }
  // chi-square with df = cells - 1 has mean df and sd sqrt(2 df); 5 sd is a 1e-6 tail.
  const df = cells - 1;
  const threshold = df + 5 * Math.sqrt(2 * df);
  assert.ok(Math.abs(chi2 - df) < 5 * Math.sqrt(2 * df), `chi2 = ${chi2}, threshold ${threshold}`);
});

// ------------------------------------------------------------------- ISI

test('inter-spike intervals have mean 1/rate and CV near 1', () => {
  // rate 50 Hz, dt 1 ms -> p = 0.05, about 20000 spikes over 400000 steps.
  const rate = 50;
  const dtMs = 1;
  const p = (rate * dtMs) / 1000;
  const steps = 400000;
  const train = core.simulateTrain(999, p, steps);
  const gaps = core.interSpikeIntervals(train);
  assert.ok(gaps.length > 19000, `expected ~20000 spikes, got ${gaps.length}`);

  const isiMs = Array.from(gaps, (g) => g * dtMs);
  const observedMean = core.mean(isiMs);
  const expectedMean = 1000 / rate; // ms
  assert.ok(
    Math.abs(observedMean - expectedMean) / expectedMean < 0.03,
    `mean ISI ${observedMean} vs ${expectedMean}`
  );

  const cv = core.coefficientOfVariation(isiMs);
  assert.ok(Math.abs(cv - 1) < 0.05, `CV ${cv} should be near 1`);
});

test('interSpikeIntervals handles empty and single-spike trains', () => {
  assert.equal(core.interSpikeIntervals([0, 0, 0]).length, 0);
  assert.equal(core.interSpikeIntervals([1, 0, 0, 0]).length, 0);
  const gaps = core.interSpikeIntervals([1, 0, 1, 0, 0, 1]);
  assert.deepEqual(Array.from(gaps), [2, 3]);
});

// ------------------------------------------------------------------- CLT

test('sums of windows converge to the normal moments as m grows', () => {
  const p = 0.05;
  const nPerBin = 20;
  const counts = core.binCounts(7, p, nPerBin, 20000);
  const lambda = nPerBin * p;
  for (const m of [1, 5, 25]) {
    const sums = core.sumGroups(counts, m);
    const meanT = m * lambda;
    const varT = m * nPerBin * p * (1 - p);
    assert.ok(Math.abs(core.mean(sums) - meanT) < 0.05 * meanT, `mean m=${m}`);
    assert.ok(Math.abs(core.variance(sums) - varT) < 0.08 * varT, `var m=${m}`);
  }
  // Skewness falls towards the normal's 0 as more windows are added.
  const sk1 = Math.abs(core.skewness(core.sumGroups(counts, 1)));
  const sk25 = Math.abs(core.skewness(core.sumGroups(counts, 25)));
  assert.ok(sk25 < sk1, `skew did not fall: ${sk25} vs ${sk1}`);
  assert.ok(sk1 > 0.5, `raw counts should be visibly skewed, got ${sk1}`);
});

// ------------------------------------------------------------ determinism

test('simulation is deterministic under the seed', () => {
  const a = core.binCounts(4242, 0.05, 20, 500);
  const b = core.binCounts(4242, 0.05, 20, 500);
  const c = core.binCounts(4243, 0.05, 20, 500);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.notDeepEqual(Array.from(a), Array.from(c));

  const t1 = core.simulateTrain(11, 0.3, 200);
  const t2 = core.simulateTrain(11, 0.3, 200);
  assert.deepEqual(Array.from(t1), Array.from(t2));
});

test('mulberry32 pins the first draws', () => {
  const rng = core.mulberry32(1);
  const first = [rng(), rng(), rng()];
  assert.ok(Math.abs(first[0] - 0.6270739405881613) < 1e-12, `r0 = ${first[0]}`);
  assert.ok(Math.abs(first[1] - 0.002735721180215478) < 1e-12, `r1 = ${first[1]}`);
  assert.ok(Math.abs(first[2] - 0.5274470399599527) < 1e-12, `r2 = ${first[2]}`);
});

// -------------------------------------------------------------- edge cases

test('invalid arguments raise errors that name the offending value', () => {
  assert.throws(() => core.binomialPmf(1, -1, 0.5), /n must be/);
  assert.throws(() => core.binomialPmf(1, 4, 1.5), /p must be in \[0, 1\], got 1\.5/);
  assert.throws(() => core.poissonPmf(0, -1), /lambda must be >= 0, got -1/);
  assert.throws(() => core.geometricPmf(1, 2), /p must be in \[0, 1\], got 2/);
  assert.throws(() => core.exponentialPdf(1, -1), /rate must be >= 0/);
  assert.throws(() => core.normalPdf(0, 0, 0), /sigma must be positive, got 0/);
  assert.throws(() => core.simulateTrain(1, 0.5, -1), /steps must be a non-negative integer/);
  assert.throws(() => core.binCounts(1, 0.5, 0, 5), /nPerBin must be a positive integer/);
  assert.throws(() => core.binomialToPoissonTV(5, 2), /lambda must be <= n/);
  assert.throws(() => core.mulberry32(1.5), /seed must be an integer/);
});

test('out-of-range counts contribute zero probability', () => {
  assert.equal(core.binomialPmf(-1, 4, 0.5), 0);
  assert.equal(core.binomialPmf(5, 4, 0.5), 0);
  assert.equal(core.poissonPmf(-1, 1), 0);
  assert.equal(core.poissonPmf(1.5, 1), 0);
  assert.equal(core.geometricPmf(2.5, 0.5), 0);
  assert.equal(core.exponentialPdf(-1, 1), 0);
});

test('totalVariation needs equal lengths and obeys its bounds', () => {
  assert.throws(() => core.totalVariation([1, 0], [1, 0, 0]), /equal lengths/);
  assert.equal(core.totalVariation([0.5, 0.5], [0.5, 0.5]), 0);
  assert.equal(core.totalVariation([1, 0], [0, 1]), 1);
});

test('the string table is English and complete', () => {
  assert.equal(typeof S.title, 'string');
  assert.equal(S.chapterCaps.length, 5);
  assert.ok(S.glossary.length >= 8);
  for (const cap of S.chapterCaps) assert.ok(cap.length > 20);
});
