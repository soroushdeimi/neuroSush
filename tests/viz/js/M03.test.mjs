/**
 * Tests for plate M03 core (Nernst equation + particle picture) — node --test.
 * Run: node --test tests/viz/js
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/M03.js');

test('nernst: K+ at 37 °C, 5/140 mM → −89.0 mV (within 0.5)', () => {
  const E = core.nernst(5, 140, 1, 37);
  assert.ok(Math.abs(E - (-89.0)) < 0.5, 'expected −89.0±0.5 mV, got ' + E);
});

test('nernst: squid 6.3 °C RT/F = 24.10 mV (within 0.05)', () => {
  const rtf = core.rtOverF_mV(6.3);
  assert.ok(Math.abs(rtf - 24.10) < 0.05, 'expected 24.10 mV, got ' + rtf);
});

test('per-decade slope: 61.5 mV at 37 °C, 58.2 mV at 20 °C (z=1)', () => {
  const a = core.perDecade_mV(37, 1);
  assert.ok(Math.abs(a - 61.5) < 0.5, 'expected 61.5±0.5 mV/decade at 37°C, got ' + a);
  const b = core.perDecade_mV(20, 1);
  assert.ok(Math.abs(b - 58.2) < 0.5, 'expected 58.2±0.5 mV/decade at 20°C, got ' + b);
});

test('nernst: divalent cation halves the slope (z=2)', () => {
  const one = core.nernst(10, 1, 1, 37);
  const two = core.nernst(10, 1, 2, 37);
  assert.ok(Math.abs(two - one / 2) < 1e-9, 'z=2 must halve E, got ' + two + ' vs ' + one / 2);
});

test('sign flips for anions: Cl− 150/10 gives negative E, exact mirror of z=1', () => {
  const cation = core.nernst(150, 10, 1, 37);
  const anion = core.nernst(150, 10, -1, 37);
  assert.ok(Math.abs(anion + cation) < 1e-9, 'anion must flip sign: ' + anion + ' vs ' + cation);
  assert.ok(anion < 0, 'Cl− with out>in must sit negative, got ' + anion);
});

test('rejected inputs: zero/negative concentration and z=0 throw', () => {
  assert.throws(() => core.nernst(0, 140, 1, 37), /positive/);
  assert.throws(() => core.nernst(5, -1, 1, 37), /positive/);
  assert.throws(() => core.nernst(5, 140, 0, 37), /nonzero/);
  assert.throws(() => core.rtOverF_mV(-400), /−273\.15|temperature/);
});

test('odeStep: exact implicit solution settles within 1 % of Nernst', () => {
  let V = 0;
  const E = core.nernst(5, 140, 1, 37);
  for (let k = 0; k < 2000; k++) {
    V += core.odeStep(V, E, 10, 0.001, 1).dV; // 1 ms per step
  }
  const gap = Math.abs(V - E);
  assert.ok(gap / Math.abs(E) < 0.01, 'V must settle within 1 % of E, gap=' + gap);
});

test('odeStep rejects nonphysical arguments', () => {
  assert.throws(() => core.odeStep(0, 0, 10, 0, 1), /capacitance/);
  assert.throws(() => core.odeStep(0, 0, -1, 1, 1), /conductance/);
  assert.throws(() => core.odeStep(0, 0, 1, 1, 0), /time step/);
});

test('mulberry32: deterministic and reproducible under a seed', () => {
  const a = core.mulberry32(1234);
  const b = core.mulberry32(1234);
  for (let i = 0; i < 1000; i++) assert.equal(a(), b());
});

test('particleStep: pure function, drift moves the ion without mutating caller', () => {
  const r = core.mulberry32(7);
  const y0 = 100;
  const y1 = core.particleStep(y0, r, 1.5, 0.3, -80);
  assert.ok(Math.abs(y1 - y0) > 0, 'step must move the particle');
  const fresh = core.mulberry32(7);
  const yDrift = core.particleStep(y0, fresh, 1.5, 0.3, -80);
  assert.equal(yDrift, y1, 'same seed → same value');
});

test('fluxDemand: crossing demand vanishes exactly when V sits on the Nernst value', () => {
  for (const [o, i, z, T] of [[5, 140, 1, 37], [145, 15, 1, 37], [150, 10, -1, 20], [0.5, 300, 1, 6.3]]) {
    const E = core.nernst(o, i, z, T);
    const d = core.fluxDemand(E, E, 10, 0.001, 16, z);
    assert.ok(
      d.base === 0 && d.dV === 0,
      'demand at V=E must be 0 for ' + [o, i, z, T].join('/') + ', got ' + d.base
    );
  }
});

// Failure demonstration tests
test('FAILURE DEMO 1: Incorrect Nernst potential value assertion fails', () => {
  const E = core.nernst(5, 140, 1, 37);
  assert.throws(() => {
    assert.equal(E, 50.0, 'Incorrect Nernst potential value assertion should fail');
  }, /AssertionError/);
});

test('FAILURE DEMO 2: Incorrect temperature RT/F assertion fails', () => {
  const rtf = core.rtOverF_mV(37);
  assert.throws(() => {
    assert.equal(rtf, 100.0, 'Incorrect temperature RT/F assertion should fail');
  }, /AssertionError/);
});
