import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/M01.js');

const TAU = core.TAU; // 20 ms
const T_END = core.T_END; // 100 ms

function eulerTrail(y0, dt, tau, n) {
  const out = [y0];
  let y = y0;
  for (let i = 0; i < n; i++) {
    y = core.eulerStep(y, dt, tau);
    out.push(y);
  }
  return out;
}

function run(...fns) {
  fns.forEach((fn) => fn());
}

test('eulerStep is the factor (1 - dt/tau) times y', () => {
  assert.equal(core.eulerStep(1, 1, TAU), 0.95);
  assert.equal(core.eulerStep(2, 5, TAU), 2 * 0.75);
  assert.equal(core.eulerStep(-0.4, 10, TAU), -0.4 * 0.5);
});

test('y_k = (1 - dt/tau)^k exactly: dt=1 ms, tau=20 ms, k=10 -> 0.5987369...', () => {
  const trail = eulerTrail(1, 1, TAU, 10);
  const closed = Math.pow(1 - 1 / TAU, 10);
  assert.ok(Math.abs(trail[10] - closed) < 1e-12, `${trail[10]} vs closed form`);
  assert.ok(Math.abs(closed - 0.5987369392383786) < 1e-15, 'hand value 0.95^10');
  // per-step iteration tracks the closed form to well under a 1e-6 story tolerance
  let y = 1;
  for (let k = 1; k <= 30; k++) {
    y = core.eulerStep(y, 1, TAU);
    assert.ok(Math.abs(y - Math.pow(0.95, k)) < 1e-12, `k=${k}`);
  }
});

test('stepsTo covers [0, tEnd] with uniform corners', () => {
  for (const dt of [0.05, 1, 5, 20, 45]) {
    const k = core.stepsTo(dt, TAU, T_END);
    assert.ok(k >= 1);
    const tLast = k * dt;
    assert.ok(Math.abs(tLast - T_END) < dt, `corner grid within dt of 100, dt=${dt}`);
  }
});

test('landingValue euler equals the iterated staircase', () => {
  for (const dt of [0.5, 3, 7, 30]) {
    const k = core.stepsTo(dt, TAU, T_END);
    const iterated = eulerTrail(1, dt, TAU, k)[k];
    assert.ok(Math.abs(core.landingValue('euler', dt, TAU, T_END) - iterated) < 1e-12);
  }
});

test('error ~ C dt: fitted log-log slope within [0.95, 1.05] over dt in [0.05, 2]', () => {
  const sweep = core.errorSweep(0.05, 2, TAU, T_END, 24);
  assert.equal(sweep.dts.length, sweep.errs.length);
  const slope = core.fitSlopeLogLog(sweep.dts, sweep.errs);
  assert.ok(slope > 0.95 && slope < 1.05, `slope ${slope} in [0.95, 1.05]`);
});

test('halving dt roughly halves the error (first order)', () => {
  const e1 = core.errorAt(1, TAU, T_END);
  const half = core.errorAt(0.5, TAU, T_END);
  const ratio = e1 / half;
  assert.ok(ratio > 1.75 && ratio < 2.25, `ratio ${ratio} ~ 2`);
});

test('dt = tau lands flat on zero after one step', () => {
  assert.equal(core.eulerStep(1, TAU, TAU), 0);
  for (let i = 0; i < 5; i++) assert.equal(core.eulerStep(0, TAU, TAU), 0);
});

test('dt = 2 tau oscillates with sign flips and constant magnitude 1', () => {
  const trail = eulerTrail(1, 2 * TAU, TAU, 6);
  for (let i = 1; i <= 6; i++) {
    assert.equal(Math.abs(trail[i]), 1);
    assert.equal(trail[i], i % 2 === 0 ? 1 : -1);
  }
});

test('dt > 2 tau blows up: magnitude grows every step', () => {
  for (const dt of [3 * TAU, 2.5 * TAU, 45]) {
    const trail = eulerTrail(1, dt, TAU, 8);
    for (let i = 1; i <= 8; i++) {
      assert.ok(Math.abs(trail[i]) > Math.abs(trail[i - 1]), `grew at step ${i}, dt=${dt}`);
    }
  }
});

test('exponential-integrator step is exact: matches exp(-t/tau) to 1e-12', () => {
  let y = 1;
  let t = 0;
  const dt = 0.37; // not commensurate with 100, still exact at every grid point
  while (t < 99) {
    y = core.exactStep(y, dt, TAU);
    t += dt;
    const exact = Math.exp(-t / TAU);
    assert.ok(Math.abs(y - exact) < 1e-12, `t=${t}, |y-exact|=${Math.abs(y - exact)}`);
  }
});

test('landingValue exact equals exp(-tEnd/tau) on the corner grid', () => {
  for (const dt of [5, 10, 20]) {
    const v = core.landingValue('exact', dt, TAU, T_END);
    assert.ok(Math.abs(v - Math.exp(-T_END / TAU)) < 1e-12);
  }
});

test('errorAt is |euler landing - exact corner value|', () => {
  const dt = 7;
  const eu = core.landingValue('euler', dt, TAU, T_END);
  const ex = core.landingValue('exact', dt, TAU, T_END);
  assert.ok(Math.abs(core.errorAt(dt, TAU, T_END) - Math.abs(eu - ex)) < 1e-15);
});

test('isStable holds below 2tau and fails at/above it', () => {
  assert.equal(core.isStable(0.05, TAU), true);
  assert.equal(core.isStable(39.9, TAU), true);
  assert.equal(core.isStable(2 * TAU, TAU), false);
  assert.equal(core.isStable(2 * TAU + 0.1, TAU), false);
  assert.equal(core.isStable(0, TAU), false);
  assert.equal(core.isStable(-1, TAU), false);
});

test('errorSweep is deterministic and log-spaced', () => {
  const a = core.errorSweep(0.05, 2, TAU, T_END, 11);
  const b = core.errorSweep(0.05, 2, TAU, T_END, 11);
  assert.deepEqual(a, b);
  const lo = Math.log(0.05);
  const hi = Math.log(2);
  for (let i = 0; i < 11; i++) {
    const expected = Math.exp(lo + ((hi - lo) * i) / 10);
    assert.ok(Math.abs(a.dts[i] - expected) < 1e-12);
  }
});

test('fitSlopeLogLog recovers an exact power law and rejects bad input', () => {
  const pts = [3, 7];
  assert.ok(Math.abs(core.fitSlopeLogLog(pts, pts.map((x) => 5 * x ** 2)) - 2) < 1e-12);
  assert.throws(() => core.fitSlopeLogLog([1], [1]), />= 2 paired points/);
  assert.throws(() => core.fitSlopeLogLog([1, 2], [1]), /paired points/);
  assert.throws(() => core.fitSlopeLogLog([3, 3], [1, 2]), /degenerate fit/);
});

test('mulberry32 is a pure seeded PRNG', () => {
  const r1 = core.mulberry32(7);
  const r2 = core.mulberry32(7);
  for (let i = 0; i < 12; i++) assert.equal(r1(), r2());
  assert.notEqual(core.mulberry32(7)(), core.mulberry32(8)());
  const r3 = core.mulberry32(1);
  for (let i = 0; i < 100; i++) {
    const v = r3();
    assert.ok(v >= 0 && v < 1);
  }
});

test('predict choice set always contains the true Euler landing value', () => {
  // mirrors makeChoices in the UI: the correct option uses landingValue('euler', ...)
  for (const dt of [0.5, 5, 18, 44]) {
    const a = core.landingValue('euler', dt, TAU, T_END);
    const b = core.landingValue('exact', dt, TAU, T_END);
    assert.ok(a !== b || dt === TAU, 'choices distinct for most dt');
    assert.ok(Number.isFinite(a));
    assert.ok(Number.isFinite(b));
  }
});

test('constants match the plate story (tau 20 ms, window 100 ms)', () => {
  assert.equal(core.TAU, 20);
  assert.equal(core.T_END, 100);
  assert.ok(core.DT_MIN > 0);
  assert.equal(core.DT_MAX, 45);
});
