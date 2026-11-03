import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N05.js');

test('m_inf(-65) matches theory and sub-terms match expectations', () => {
  const a_m = core.alpha_m(-65);
  const b_m = core.beta_m(-65);
  assert.ok(Math.abs(a_m - 0.223547) < 1e-4, `alpha_m(-65) was ${a_m}`);
  assert.equal(b_m, 4.0);
  const m_val = core.m_inf(-65);
  assert.ok(Math.abs(m_val - 0.0529) < 1e-4, `m_inf(-65) was ${m_val}`);
});

test('h_inf(-65) matches theory', () => {
  const a_h = core.alpha_h(-65);
  const b_h = core.beta_h(-65);
  assert.equal(a_h, 0.07);
  assert.ok(Math.abs(b_h - 0.047426) < 1e-4, `beta_h(-65) was ${b_h}`);
  const h_val = core.h_inf(-65);
  assert.ok(Math.abs(h_val - 0.5961) < 1e-4, `h_inf(-65) was ${h_val}`);
});

test('n_inf(-65) matches theory', () => {
  const a_n = core.alpha_n(-65);
  const b_n = core.beta_n(-65);
  assert.ok(Math.abs(a_n - 0.058198) < 1e-4, `alpha_n(-65) was ${a_n}`);
  assert.equal(b_n, 0.125);
  const n_val = core.n_inf(-65);
  assert.ok(Math.abs(n_val - 0.3177) < 1e-4, `n_inf(-65) was ${n_val}`);
});

test('singular points are finite and continuous', () => {
  const a_m_sing = core.alpha_m(-40);
  assert.equal(a_m_sing, 1.0);
  assert.ok(Math.abs(core.alpha_m(-40 + 1e-5) - 1.0) < 1e-4);
  assert.ok(Math.abs(core.alpha_m(-40 - 1e-5) - 1.0) < 1e-4);

  const a_n_sing = core.alpha_n(-55);
  assert.equal(a_n_sing, 0.1);
  assert.ok(Math.abs(core.alpha_n(-55 + 1e-5) - 0.1) < 1e-4);
  assert.ok(Math.abs(core.alpha_n(-55 - 1e-5) - 0.1) < 1e-4);
});

test('rest stays within 0.5 mV for 100 ms with I=0', () => {
  const res = core.simulate(100, 0.025, 0, {
    V: -65,
    m: 0.0529,
    h: 0.5961,
    n: 0.3177,
  });
  let maxDev = 0;
  for (const step of res.history) {
    const dev = Math.abs(step.V - (-65));
    if (dev > maxDev) maxDev = dev;
  }
  assert.ok(maxDev < 0.5, `max deviation was ${maxDev} mV (must be < 0.5 mV)`);
});

test('I=10 gives repeated spikes peaking between +20 and +60 mV', () => {
  const res = core.simulate(100, 0.025, 10, {
    V: -65,
    m: 0.0529,
    h: 0.5961,
    n: 0.3177,
  });
  const peaks = [];
  const h = res.history;
  for (let i = 1; i < h.length - 1; i++) {
    if (h[i].V > h[i - 1].V && h[i].V > h[i + 1].V && h[i].V > 0) {
      peaks.push(h[i]);
    }
  }
  assert.ok(peaks.length >= 5, `expected at least 5 spikes, got ${peaks.length}`);
  for (const p of peaks) {
    assert.ok(
      p.V >= 20 && p.V <= 60,
      `spike peak voltage ${p.V} was not between +20 and +60 mV`
    );
  }
});

test('I=2 does not fire', () => {
  const res = core.simulate(50, 0.025, 2, {
    V: -65,
    m: 0.0529,
    h: 0.5961,
    n: 0.3177,
  });
  let maxV = -100;
  for (const step of res.history) {
    if (step.V > maxV) maxV = step.V;
  }
  assert.ok(maxV < -50, `I=2 reached max V = ${maxV} mV (should not fire)`);
});

test('gates stay in [0, 1]', () => {
  const res = core.simulate(80, 0.025, 15, {
    V: -65,
    m: 0.0529,
    h: 0.5961,
    n: 0.3177,
  });
  for (const step of res.history) {
    assert.ok(step.m >= 0 && step.m <= 1, `m out of bounds: ${step.m}`);
    assert.ok(step.h >= 0 && step.h <= 1, `h out of bounds: ${step.h}`);
    assert.ok(step.n >= 0 && step.n <= 1, `n out of bounds: ${step.n}`);
  }
});

test('TTX removes spiking', () => {
  const res = core.simulate(100, 0.025, 10, {
    V: -65,
    m: 0.0529,
    h: 0.5961,
    n: 0.3177,
  }, { ttx: true });
  let maxV = -100;
  for (const step of res.history) {
    if (step.V > maxV) maxV = step.V;
  }
  assert.ok(maxV < -50, `with TTX, max V was ${maxV} mV (spikes should be abolished)`);
});

test('deterministic under the seed', () => {
  const rng1 = core.createRng(12345);
  const rng2 = core.createRng(12345);
  for (let i = 0; i < 50; i++) {
    assert.equal(rng1(), rng2());
  }

  const channels1 = core.initChannels(10, 6, 2, rng1);
  const channels2 = core.initChannels(10, 6, 2, rng2);
  assert.deepEqual(channels1, channels2);
});

// Explicit failure proof tests (prove that false claims fail)
test('FAILURE_PROOF: expecting resting potential to be +40 mV fails as expected', () => {
  let failed = false;
  try {
    const res = core.simulate(10, 0.025, 0);
    assert.equal(res.state.V, 40, 'Expecting resting V to be +40 mV');
  } catch (err) {
    failed = true;
  }
  assert.ok(failed, 'Resting potential assertion correctly failed');
});

test('FAILURE_PROOF: expecting spike peak to exceed +120 mV fails as expected', () => {
  let failed = false;
  try {
    const res = core.simulate(30, 0.025, 15);
    const maxV = Math.max(...res.history.map(s => s.V));
    assert.ok(maxV > 120, `Max V ${maxV} does not exceed +120 mV`);
  } catch (err) {
    failed = true;
  }
  assert.ok(failed, 'Excessive spike peak assertion correctly failed');
});
