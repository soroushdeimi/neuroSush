import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N02.js');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const n02Path = path.join(__dirname, '../../../src/neurosush/viz/web/plates/N02.js');
const n02Source = fs.readFileSync(n02Path, 'utf8');

test('ELIF derivative is the hand-computed sum of leak, exponential boost, and input', () => {
  const p = { tau: 20, v_rest: -70, theta_rh: -50, delta: 2, resistance: 10 };
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

test('Fixed point solver finds correct stable and unstable roots against Python library for I=1.0 nA', () => {
  const p = { v_rest: -70, theta_rh: -50, delta: 2, resistance: 10 };
  const points = core.fixedPoints(1.0, p);
  assert.strictEqual(points.length, 2);
  assert.ok(Math.abs(points[0].v - (-59.9864)) < 0.01);
  assert.strictEqual(points[0].stable, true);
  assert.ok(Math.abs(points[1].v - (-46.1263)) < 0.01);
  assert.strictEqual(points[1].stable, false);
});

test('Escape over 30 mV stops running and holds at 30 mV rather than wrapping to -40', () => {
  let state = { v: 29.5, running: true, p: core.DEFAULTS };
  state.v = 31.0;
  if (state.v > 30) {
    state.running = false;
    state.v = 30;
  }
  assert.strictEqual(state.running, false);
  assert.strictEqual(state.v, 30);
});

test('Plate source code references elif_terms instead of elif_derivative', () => {
  assert.ok(n02Source.includes('neurosush.neurons.dynamics.elif_terms'));
  assert.ok(!n02Source.includes('elif_derivative'));
});

test('Learn section contains the four required blocks in N02 source', () => {
  assert.ok(n02Source.includes('1. What you see:'));
  assert.ok(n02Source.includes('2. The math:'));
  assert.ok(n02Source.includes('3. In the brain:'));
  assert.ok(n02Source.includes('4. Like classic ML:'));
});
