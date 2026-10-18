// Tests for plate H01 core (SDR overlap statistics). Pure math, no DOM.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const platePath = path.resolve(here, '../../../src/neurosush/viz/web/plates/H01.js');
const { core } = require(platePath);

const relErr = (a, b) => Math.abs(a - b) / Math.abs(b);

test('overlapPmf for n=10, w=2 matches hand-computed hypergeometric', () => {
  const pmf = core.sdrPmf(10, 2);
  const byB = new Map(pmf.map((o) => [o.b, o.p]));
  assert.equal(byB.size, 3);
  assert.ok(relErr(byB.get(0), 28 / 45) < 1e-12, `b=0 got ${byB.get(0)}`);
  assert.ok(relErr(byB.get(1), 16 / 45) < 1e-12, `b=1 got ${byB.get(1)}`);
  assert.ok(relErr(byB.get(2), 1 / 45) < 1e-12, `b=2 got ${byB.get(2)}`);
});

test('sdrPmf is a probability distribution (sums to 1) for several (n, w)', () => {
  for (const [n, w] of [[2048, 40], [512, 20], [4096, 160], [100, 90]]) {
    const pmf = core.sdrPmf(n, w);
    const sum = pmf.reduce((s, o) => s + o.p, 0);
    assert.ok(relErr(sum, 1) < 1e-12, `n=${n} w=${w} sum=${sum}`);
    for (const o of pmf) assert.ok(o.p >= 0, `negative p at b=${o.b}`);
  }
});

test('edge case n === w collapses to a single sure overlap', () => {
  const pmf = core.sdrPmf(5, 5);
  assert.deepEqual(pmf, [{ b: 5, p: 1 }]);
});

test('overlapPmf rejects invalid arguments naming the value', () => {
  assert.throws(() => core.overlapPmf(0, 1, 1), /n=0/);
  assert.throws(() => core.overlapPmf(10, 11, 2), /wX=11/);
});

test('expected overlap equals w^2/n exactly', () => {
  assert.equal(core.expectedOverlap(2048, 40), 1600 / 2048);
  assert.equal(core.expectedOverlap(2048, 40), 0.78125);
});

test('Monte Carlo mean overlap is within 3 percent of w^2/n over 20000 trials', () => {
  const samples = core.sampleOverlaps(2048, 40, 20000, 12345);
  assert.equal(samples.length, 20000);
  const m = core.mean(samples);
  const theory = core.expectedOverlap(2048, 40);
  assert.ok(relErr(m, theory) < 0.03, `Monte Carlo mean ${m} vs ${theory}`);
});

test('sampling is deterministic per seed and varies across seeds', () => {
  const a = core.sampleOverlaps(2048, 40, 500, 7);
  const b = core.sampleOverlaps(2048, 40, 500, 7);
  const c = core.sampleOverlaps(2048, 40, 500, 8);
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
  for (const s of a) assert.ok(s >= 0 && s <= 40, `overlap out of range: ${s}`);
});

test('exact tail probability matches the library value to 1e-9 relative', () => {
  // Computed offline with Python fractions for the library call
  // neurosush.htm.sdr.match_probability(2048, 40, 40, 20):
  //   sum_{b>=20} C(40,b) C(2008, 40-b) / C(2048, 40)
  //   = 538122681134297238645281138786252429635555463870139647 /
  //     21600503739871387467380035333608342908960862412202880928513217588413996848604032
  const expected = 2.4912506097762944e-26;
  const tail = core.matchProbability(2048, 40, 40, 20);
  assert.ok(relErr(tail, expected) < 1e-9, `tail=${tail} expected=${expected}`);
  assert.ok(tail > 0 && tail < 1e-25);
});

test('matchProbability boundary cases', () => {
  assert.ok(relErr(core.matchProbability(2048, 40, 40, 0), 1) < 1e-12);
  assert.equal(core.matchProbability(2048, 40, 40, 41), 0);
  assert.ok(relErr(core.matchProbability(2048, 40, 40, 40), core.sdrPmf(2048, 40).find((o) => o.b === 40).p) < 1e-15);
  assert.throws(() => core.matchProbability(2048, 40, 40, -1), /theta/);
});

test('sdrTail equals matchProbability with w_x = w', () => {
  for (const theta of [0, 1, 5, 20, 40]) {
    assert.equal(core.sdrTail(2048, 40, theta), core.matchProbability(2048, 40, 40, theta));
  }
});

test('noisy pattern keeps the requested number of stored bits', () => {
  const n = 2048;
  const w = 40;
  const storedArr = core.randomPattern(n, w, core.mulberry32(3));
  const stored = new Set(storedArr);
  for (const keep of [0, 7, 28, 40]) {
    const noisy = core.noisyPattern(storedArr, n, w, keep, core.mulberry32(99));
    assert.equal(noisy.length, w, `noisy must still have w active bits, keep=${keep}`);
    let shared = 0;
    for (const i of noisy) if (stored.has(i)) shared++;
    assert.equal(shared, keep, `overlap must equal keep=${keep}, got ${shared}`);
  }
});

test('noisy pattern rejects a stored pattern of the wrong length', () => {
  assert.throws(() => core.noisyPattern([1, 2, 3], 100, 40, 10, core.mulberry32(1)), /w=40/);
});

test('union activation and expected size reduce to a single pattern at m = 1', () => {
  assert.ok(relErr(core.unionActivation(2048, 40, 1), 40 / 2048) < 1e-15);
  assert.ok(relErr(core.unionExpectedSize(2048, 40, 1), 40) < 1e-12);
  assert.equal(core.unionActivation(2048, 40, 0), 0);
  assert.throws(() => core.unionActivation(2048, 40, -1), /m/);
});

test('union match probability is monotone in M and stays a probability', () => {
  // The union uses the binomial approximation (Ahmad & Hawkins 2016), so even at M=1 it is
  // not the exact hypergeometric; it must still be a valid probability that grows with M.
  const p1 = core.unionMatchProbability(2048, 40, 1, 20);
  const p2 = core.unionMatchProbability(2048, 40, 2, 20);
  const p10 = core.unionMatchProbability(2048, 40, 10, 20);
  assert.ok(p1 > 0 && p1 < 1);
  assert.ok(p1 < p2 && p2 < p10, `${p1} ${p2} ${p10}`);
  assert.ok(core.unionMatchProbability(2048, 40, 100, 20) > 0.9);
});

test('histogram counts each overlap bucket', () => {
  const counts = core.histogram([0, 1, 1, 3, 40], 40);
  assert.equal(counts[0], 1);
  assert.equal(counts[1], 2);
  assert.equal(counts[3], 1);
  assert.equal(counts[40], 1);
  assert.equal(counts.reduce((a, b) => a + b, 0), 5);
});

test('logComb and formatSci behave', () => {
  assert.ok(relErr(core.logComb(10, 2), Math.log(45)) < 1e-12);
  assert.equal(core.logComb(5, 6), -Infinity);
  assert.equal(core.formatSci(2.4912506097762944e-26), '2.49e-26');
  assert.equal(core.formatSci(0), '0');
  assert.equal(core.formatSci(0.5), '0.5');
});
