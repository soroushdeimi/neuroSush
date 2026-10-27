import test from 'node:test';
import assert from 'node:assert/strict';
import plate from '../../../src/neurosush/viz/web/plates/Y02.js';

const { core } = plate;
const close = (actual, expected, eps = 1e-12) => assert.ok(Math.abs(actual - expected) < eps, `${actual} != ${expected}`);

test('Euler trace decay has the hand-computed value', () => {
  close(core.traceDecay(1, 20, 1), 0.95);
  close(core.traceStep(1, false, { tau: 20, dt: 1 }), 0.95);
  close(core.traceStep(0.95, true, { tau: 20, dt: 1 }), 1.9025);
});

test('triplet update samples slow traces before their increment', () => {
  const p = { ...core.DEFAULT, a2_plus: 0.005, a3_plus: 0.006, a2_minus: 0.007, a3_minus: 0.0023 };
  const before = { r1: 0.5, o1: 0.4, r2: 0.3, o2: 0.8, w: 0.5 };
  const got = core.step(before, true, true, p);
  const r1 = .5 * (1 - 1 / p.tau_plus) + 1;
  const o1 = .4 * (1 - 1 / p.tau_minus) + 1;
  const r2Before = .3 * (1 - 1 / p.tau_x);
  const o2Before = .8 * (1 - 1 / p.tau_y);
  close(got.ltp, r1 * (p.a2_plus + p.a3_plus * o2Before));
  close(got.ltd, o1 * (p.a2_minus + p.a3_minus * r2Before));
  close(got.dw, got.ltp - got.ltd);
});

test('zero triplet amplitudes equal the pair-only calculation', () => {
  const p = { ...core.DEFAULT, a3_plus: 0, a3_minus: 0 };
  const state = { r1: .75, o1: .5, r2: 7, o2: 8, w: .5 };
  const got = core.step(state, true, true, p);
  close(got.dw, got.r1 * p.a2_plus - got.o1 * p.a2_minus);
});

test('default triplet LTP increases monotonically with pairing frequency', () => {
  const freqs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
  const sweep = core.sweep(freqs, 24, { a2_minus: 0, a3_minus: 0 });
  const values = sweep.map((x) => x.delta);
  for (let i = 1; i < values.length; i += 1) {
    assert.ok(values[i] > values[i - 1], `delta at ${freqs[i]} Hz (${values[i]}) must exceed ${freqs[i - 1]} Hz (${values[i - 1]})`);
  }
});

test('full STDP with depression produces high-frequency LTD hump from inter-pair intervals', () => {
  // With depression enabled, consecutive pairs at 50 Hz (period 20 ms, lag 10 ms)
  // place post(j) only 10 ms before pre(j+1), causing net depression (the hump)
  const sweepFull = core.sweep([1, 10, 20, 50], 24, {});
  const delta20 = sweepFull.find((x) => x.frequency === 20).delta;
  const delta50 = sweepFull.find((x) => x.frequency === 50).delta;
  assert.ok(delta20 > delta50, `full STDP 20 Hz delta (${delta20}) exceeds 50 Hz (${delta50}) due to inter-pair LTD`);
});

test('the pair reference disables both triplet amplitudes', () => {
  const reference = core.pairTrain(20, 4);
  const direct = core.train(20, 4, { a3_plus: 0, a3_minus: 0 });
  close(reference.delta, direct.delta);
});

test('seeded demonstration is deterministic and validation rejects invalid rates', () => {
  assert.deepEqual(core.deterministicDemo(42), core.deterministicDemo(42));
  assert.throws(() => core.train(0), /frequency must be positive, got 0/);
  assert.throws(() => core.traceDecay(1, -1, 1), /tau must be positive, got -1/);
});
