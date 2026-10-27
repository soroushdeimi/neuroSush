import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N02.js');

test('ELIF derivative is the hand-computed sum of leak, exponential boost, and input', () => {
  const p = { tau: 20, v_rest: -70, theta_rh: -50, delta: 2, resistance: 10 };
  // At -70 mV and I=1 nA: numerator = 0 + 2e^-10 + 10.
  const expected = (2 * Math.exp(-10) + 10) / 20;
  assert.ok(Math.abs(core.derivative(-70, 1, p) - expected) < 1e-14);
  assert.ok(Math.abs(core.boost(-50, p) - 2) < 1e-14);
});

test('bisection fixed points are real zeroes and have stable then unstable order', () => {
  const p = { theta_rh: -50, delta: 2, v_rest: -70, resistance: 10 };
  const points = core.fixedPoints(1.0, p);
  assert.equal(points.length, 2);
  assert.equal(points[0].stable, true); assert.equal(points[1].stable, false);
  for (const point of points) assert.ok(Math.abs(core.numerator(point.v, 1, p)) < 1e-8);
});

test('analytic rheobase merges fixed points at theta_rh', () => {
  const p = { theta_rh: -50, delta: 2, v_rest: -70, resistance: 10 };
  // f'(v) = -1 + exp((v-theta)/delta), so merge occurs v=theta;
  // f(theta)=v_rest-theta+delta+R I = 0.
  const irh = core.rheobase(p);
  assert.equal(irh, 1.8);
  assert.ok(Math.abs(core.numerator(p.theta_rh, irh, p)) < 1e-12);
  const merged = core.fixedPoints(irh, p); assert.equal(merged.length, 1); assert.equal(merged[0].v, -50);
  assert.equal(core.fixedPoints(irh + 0.001, p).length, 0);
});

test('potential slope is negative ELIF numerator and Euler is deterministic', () => {
  const p = { dt: 0.04 }; const v = -61.2; const I = 1.0; const eps = 1e-5;
  const slope = (core.potential(v + eps, I, p) - core.potential(v - eps, I, p)) / (2 * eps);
  assert.ok(Math.abs(slope + core.numerator(v, I, p)) < 1e-7);
  assert.deepEqual(core.simulate(-69, 1, 10, p), core.simulate(-69, 1, 10, p));
  const a = core.mulberry32(91), b = core.mulberry32(91); assert.deepEqual([a(),a(),a()], [b(),b(),b()]);
});
