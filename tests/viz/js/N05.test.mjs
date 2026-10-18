import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N05.js');

test('m_inf(-65) matches theory and sub-terms match expectations', () => {
  const a_m = core.alpha_m(-65);
  const b_m = core.beta_m(-65);
  // alpha_m(-65) = 0.1 * (-25) / (1 - exp(2.5)) ~ 0.22355
  assert.ok(Math.abs(a_m - 0.223547) < 1e-4, `alpha_m(-65) was ${a_m}`);
  // beta_m(-65) = 4 * exp(0) = 4.0
  assert.equal(b_m, 4.0);
  const m_val = core.m_inf(-65);
  assert.ok(Math.abs(m_val - 0.0529) < 1e-4, `m_inf(-65) was ${m_val}`);
});

test('h_inf(-65) matches theory', () => {
  const a_h = core.alpha_h(-65);
  const b_h = core.beta_h(-65);
  assert.equal(a_h, 0.07);
  // beta_h(-65) = 1 / (1 + exp(3)) ~ 0.047426
  assert.ok(Math.abs(b_h - 0.047426) < 1e-4, `beta_h(-65) was ${b_h}`);
  const h_val = core.h_inf(-65);
  assert.ok(Math.abs(h_val - 0.5961) < 1e-4, `h_inf(-65) was ${h_val}`);
});

test('n_inf(-65) matches theory', () => {
  const a_n = core.alpha_n(-65);
  const b_n = core.beta_n(-65);
  // alpha_n(-65) = 0.01 * (-10) / (1 - exp(1)) ~ 0.058198
  assert.ok(Math.abs(a_n - 0.058198) < 1e-4, `alpha_n(-65) was ${a_n}`);
  assert.equal(b_n, 0.125);
  const n_val = core.n_inf(-65);
  assert.ok(Math.abs(n_val - 0.3177) < 1e-4, `n_inf(-65) was ${n_val}`);
});

test('singular points are finite and continuous', () => {
  // Singularity at V = -40 for alpha_m (limit is 1.0)
  const a_m_sing = core.alpha_m(-40);
  assert.equal(a_m_sing, 1.0);
  assert.ok(Math.abs(core.alpha_m(-40 + 1e-5) - 1.0) < 1e-4);
  assert.ok(Math.abs(core.alpha_m(-40 - 1e-5) - 1.0) < 1e-4);

  // Singularity at V = -55 for alpha_n (limit is 0.1)
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

test('RK4 convergence: halving dt changes spike time by less than 0.05 ms', () => {
  function getFirstSpikeTime(dt) {
    const res = core.simulate(10, dt, 10, {
      V: -65,
      m: 0.0529,
      h: 0.5961,
      n: 0.3177,
    });
    const h = res.history;
    for (let i = 1; i < h.length - 1; i++) {
      if (h[i].V > h[i - 1].V && h[i].V > h[i + 1].V && h[i].V > 0) {
        return h[i].t;
      }
    }
    return null;
  }

  const t1 = getFirstSpikeTime(0.025);
  const t2 = getFirstSpikeTime(0.0125);
  assert.ok(t1 !== null && t2 !== null, 'spike was detected at both resolutions');
  const diff = Math.abs(t1 - t2);
  assert.ok(diff < 0.05, `spike time diff was ${diff} ms (must be < 0.05 ms)`);
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

test('voltage clamp reproduces classic inward-then-outward current', () => {
  // Step from holding -65 mV to 0 mV
  const dt = 0.025;
  const steps = Math.round(15 / dt);
  let state = { V: -65, m: 0.0529, h: 0.5961, n: 0.3177 };
  let minCurrent = 1000; // inward current is negative
  let minTime = -1;
  let finalCurrent = 0;

  for (let i = 0; i < steps; i++) {
    const t = i * dt;
    // Step to 0 mV after 1 ms
    const clampV = t >= 1.0 ? 0 : -65;
    state = core.rk4Step(state, 0, dt, { clamp: true, clampV });
    const totalI = state.I_Na + state.I_K + state.I_L;
    if (t > 1.0) {
      if (totalI < minCurrent) {
        minCurrent = totalI;
        minTime = t;
      }
      finalCurrent = totalI;
    }
  }

  // Classic clamp response:
  // 1. Transient inward current (minCurrent strongly negative, e.g. < -400 uA/cm^2)
  assert.ok(minCurrent < -200, `expected inward Na current, got ${minCurrent}`);
  assert.ok(minTime < 3.0, `inward peak occurred at ${minTime} ms (should be rapid)`);
  // 2. Delayed outward current (finalCurrent strongly positive, e.g. > 1000 uA/cm^2)
  assert.ok(finalCurrent > 500, `expected outward K current, got ${finalCurrent}`);
});
