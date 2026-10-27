import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N03.js');

test('AdEx voltage terms match the library Euler decomposition by hand', () => {
  const p = { v_rest: -65, resistance: 1, delta: 2, theta_rh: -50, tau: 20, dt: 0.5 };
  const terms = core.adex_terms(-65, 2, 1.5, p);
  assert.equal(terms.leak, 0);
  assert.equal(terms.input, 0.05);
  assert.ok(Math.abs(terms.adaptation - (-0.0375)) < 1e-15);
  assert.ok(Math.abs(terms.spike_initiation - 0.00002765421850739168) < 1e-16);
  const d = core.derivative(-65, 1.5, 2, { ...p, tau_w: 100, a: 2 });
  assert.ok(Math.abs(d.dv - 0.025055308437014784) < 1e-14);
  assert.equal(d.dw, -0.015);
});

test('a spike resets voltage and adds exactly b to adaptation after its decay step', () => {
  const p = { ...core.DEFAULTS, dt: 0.1, threshold: -60, v_reset: -58, b: 40, current: 0, a: 0 };
  const state = { v: -59, omega: 10, t: 0, spikes: [], history: [], isis: [], lastSpikeTime: -Infinity };
  const expectedPreJump = state.omega + ((p.a * (state.v - p.v_rest) - state.omega) / p.tau_w) * p.dt;
  const out = core.step(state, p);
  assert.equal(out.spiked, true);
  assert.equal(state.v, -58);
  assert.ok(Math.abs(state.omega - (expectedPreJump + 40)) < 1e-12);
  assert.deepEqual(state.spikes, [0.1]);
});

test('adapting preset ISIs are non-decreasing after the initial transient', () => {
  const p = { ...core.DEFAULTS, ...core.presets('adapting') };
  const run = core.simulate(p, 1000);
  assert.ok(run.isi.length >= 5, 'expected repeated spikes');
  for (let i = 1; i < run.isi.length; i++) {
    assert.ok(run.isi[i] + 1e-9 >= run.isi[i - 1], `${run.isi[i]} should not precede ${run.isi[i - 1]}`);
  }
});

test('tonic preset produces constant inter-spike intervals without adaptation', () => {
  const p = { ...core.DEFAULTS, ...core.presets('tonic') };
  const run = core.simulate(p, 1000);
  assert.ok(run.isi.length >= 10, 'expected multiple tonic spikes');
  const firstISI = run.isi[0];
  for (let i = 1; i < run.isi.length; i++) {
    assert.ok(Math.abs(run.isi[i] - firstISI) < 0.2, `expected constant ISI ${firstISI}, got ${run.isi[i]}`);
  }
});

test('bursting preset produces fast intra-burst spikes and a long recovery pause', () => {
  const p = { ...core.DEFAULTS, ...core.presets('bursting') };
  const run = core.simulate(p, 1000);
  assert.ok(run.isi.length >= 6, 'expected multiple spikes in bursting mode');
  const hasShort = run.isi.some(val => val < 20.0);
  const hasLong = run.isi.some(val => val > 100.0);
  assert.ok(hasShort, 'expected fast intra-burst spikes');
  assert.ok(hasLong, 'expected long inter-burst pause');
});

test('simulation is deterministic and input validation names offending value', () => {
  const p = { ...core.DEFAULTS, ...core.presets('bursting'), seed: 777 };
  const a = core.simulate(p, 300);
  const b = core.simulate(p, 300);
  assert.deepEqual(a.spikes, b.spikes);
  assert.deepEqual(a.history.map(q => [q.v, q.omega]), b.history.map(q => [q.v, q.omega]));
  assert.throws(() => core.adex_terms(-65, 0, 0, { tau: 0 }), /tau must be positive, got 0/);
  assert.throws(() => core.step(core.createState(), { ...core.DEFAULTS, tau_w: -1 }), /tau_w must be positive, got -1/);
  assert.throws(() => core.adex_terms(NaN, 0, 0, {}), /v must be finite, got NaN/);
});

test('nullclines follow their closed-form definitions and f-I is nonnegative', () => {
  const p = { ...core.DEFAULTS, current: 500, a: 2 };
  assert.equal(core.nullclineW(-55, p), 20);
  const v = -60;
  const expected = (p.v_rest - v + p.delta * Math.exp((v - p.theta_rh) / p.delta) + p.resistance * p.current) / p.resistance;
  assert.equal(core.nullclineV(v, p), expected);
  assert.ok(core.theoryFI(500, p) >= 0);
});

test('theory steady-state omega calculation returns expected values', () => {
  const p = { ...core.DEFAULTS, a: 0.02, b: 0.5, tau_w: 120, v_rest: -65 };
  const omegaEst = core.theorySteadyStateOmega(20, -60, p);
  // tau_w * (a * (vMean - v_rest) + b * (f / 1000))
  // 120 * (0.02 * (-60 - (-65)) + 0.5 * (20 / 1000))
  // 120 * (0.02 * 5 + 0.01) = 120 * (0.1 + 0.01) = 120 * 0.11 = 13.2
  assert.ok(Math.abs(omegaEst - 13.2) < 1e-10);
});
