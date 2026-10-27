import { test } from 'node:test';
import assert from 'node:assert/strict';
import { core } from '../../../src/neurosush/viz/web/plates/Y01.js';

test('trace after k steps = (1-dt/tau)^k: tau=20, dt=1, k=10 gives 0.95^10 = 0.5987', () => {
  const tau = 20.0;
  const dt = 1.0;
  const k = 10;
  // Hand-computed value: (1 - 1/20)^10 = 0.95^10 = 0.5987369392383789
  const expected = Math.pow(0.95, 10);
  assert.ok(Math.abs(expected - 0.5987369) < 1e-6);

  // Closed form helper
  const closedForm = core.trace_k_steps(1.0, k, { tau, dt });
  assert.equal(closedForm, expected);

  // Step-by-step decay using trace_step
  let stepTrace = 1.0;
  for (let i = 0; i < k; i++) {
    stepTrace = core.trace_step(stepTrace, 0, { tau, dt });
  }
  assert.ok(
    Math.abs(stepTrace - expected) < 1e-15,
    `expected stepTrace ${stepTrace} to equal ${expected}`
  );
});

test('dw for dt=+5 equals a_plus*0.95^5 within 1e-12', () => {
  const a_plus = 0.02;
  const tau_plus = 20.0;
  const dt = 1.0;
  const delta_t = 5.0;

  // Discrete trace after 5 ms decay from pre spike: (1 - 1/20)^5 = 0.95^5 = 0.7737809375
  // Potentiation dw = a_plus * 0.95^5 = 0.02 * 0.7737809375 = 0.01547561875
  const expected = a_plus * Math.pow(0.95, 5);
  const dw = core.stdp_pair_dw(delta_t, { a_plus, tau_plus, dt });

  assert.ok(
    Math.abs(dw - expected) < 1e-12,
    `expected dw ${dw} to equal ${expected} within 1e-12`
  );
});

test('dw for dt=-5 equals -a_minus*0.95^5 within 1e-12', () => {
  const a_minus = 0.021;
  const tau_minus = 20.0;
  const dt = 1.0;
  const delta_t = -5.0;

  // Discrete trace after 5 ms decay from post spike: (1 - 1/20)^5 = 0.95^5
  // Depression dw = -a_minus * 0.95^5 = -0.021 * 0.7737809375 = -0.0162493996875
  const expected = -a_minus * Math.pow(0.95, 5);
  const dw = core.stdp_pair_dw(delta_t, { a_minus, tau_minus, dt });

  assert.ok(
    Math.abs(dw - expected) < 1e-12,
    `expected dw ${dw} to equal ${expected} within 1e-12`
  );
});

test('sign flips at dt=0 convention stated', () => {
  // STDP Convention:
  // Pre before post (dt > 0) -> potentiation (dw > 0).
  // Post before pre (dt < 0) -> depression (dw < 0).
  // At dt = 0: simultaneous discrete spikes; in neurosush both pre and post traces
  // are evaluated with both spikes present, giving net a_plus - a_minus = 0.02 - 0.021 = -0.001.
  const dw_pos = core.stdp_pair_dw(+1.0);
  const dw_neg = core.stdp_pair_dw(-1.0);
  const dw_zero = core.stdp_pair_dw(0.0);

  assert.ok(dw_pos > 0, `dw_pos must be positive, got ${dw_pos}`);
  assert.ok(dw_neg < 0, `dw_neg must be negative, got ${dw_neg}`);
  assert.equal(dw_zero, core.DEFAULT_A_PLUS - core.DEFAULT_A_MINUS);
});

test('weight bounds respected under repeated potentiation and depression', () => {
  const w_min = 0.0;
  const w_max = 1.0;

  // Hard bounds clamp to [0, 1]
  let w = 0.95;
  const big_ltp = 0.1;
  w = core.apply_bounds(w, big_ltp, { w_min, w_max, bound: 'hard' });
  assert.equal(w, 1.0, `w should be clamped at w_max 1.0, got ${w}`);

  let w_low = 0.05;
  const big_ltd = -0.1;
  w_low = core.apply_bounds(w_low, big_ltd, { w_min, w_max, bound: 'hard' });
  assert.equal(w_low, 0.0, `w should be clamped at w_min 0.0, got ${w_low}`);

  // Repeated 60 pairings from run_pairings cannot exceed [0, 1]
  const run_ltp = core.run_pairings(+5.0, 60, 0.5);
  assert.ok(run_ltp.final_w <= 1.0 && run_ltp.final_w > 0.5);

  const run_ltd = core.run_pairings(-5.0, 60, 0.5);
  assert.ok(run_ltd.final_w >= 0.0 && run_ltd.final_w < 0.5);
});

test('uncorrelated Poisson pairs drift by about (a_plus*tau_plus - a_minus*tau_minus)*r1*r2', () => {
  // Theoretical expectation:
  // For independent Poisson processes at rates r1 and r2 (Hz), the expected pre-trace is
  // E[x_pre] = r1 * (tau_plus / 1000).
  // The expected potentiation rate is a_plus * E[x_pre] * r2 = a_plus * (tau_plus/1000) * r1 * r2.
  // The expected depression rate is a_minus * E[x_post] * r1 = a_minus * (tau_minus/1000) * r1 * r2.
  // Net expected drift rate (dw/sec) = (a_plus * tau_plus/1000 - a_minus * tau_minus/1000) * r1 * r2.
  const r1 = 15.0; // Hz
  const r2 = 15.0; // Hz
  const duration = 50.0; // seconds
  const a_plus = 0.02;
  const a_minus = 0.021;
  const tau_plus = 20.0; // ms
  const tau_minus = 20.0; // ms

  const tau_p_s = tau_plus / 1000.0;
  const tau_m_s = tau_minus / 1000.0;
  const expected_drift_rate = (a_plus * tau_p_s - a_minus * tau_m_s) * r1 * r2;
  const expected_total_dw = expected_drift_rate * duration;
  // With parameters: (0.0004 - 0.00042) * 225 * 50 = -0.00002 * 11250 = -0.225

  const sim = core.simulate_poisson_drift(r1, r2, duration, {
    seed: 42,
    a_plus,
    a_minus,
    tau_plus,
    tau_minus,
    w_init: 0.5,
    bound: 'none' // test unbounded linear drift to compare against expectation
  });

  const sim_drift_rate = sim.total_dw / duration;
  assert.ok(
    Math.abs(sim_drift_rate - expected_drift_rate) < 0.001,
    `simulated drift rate ${sim_drift_rate} should be close to expected ${expected_drift_rate}`
  );

  // When a_plus * tau_plus == a_minus * tau_minus, expected drift is exactly zero
  const balanced = core.simulate_poisson_drift(r1, r2, duration, {
    seed: 42,
    a_plus: 0.02,
    a_minus: 0.02,
    tau_plus: 20.0,
    tau_minus: 20.0,
    w_init: 0.5,
    bound: 'none'
  });
  const balanced_rate = balanced.total_dw / duration;
  assert.ok(
    Math.abs(balanced_rate) < 0.001,
    `balanced STDP should have near zero drift rate, got ${balanced_rate}`
  );
});

test('determinism under seed for PRNG and Poisson simulation', () => {
  const sim1 = core.simulate_poisson_drift(10, 10, 5, { seed: 98765 });
  const sim2 = core.simulate_poisson_drift(10, 10, 5, { seed: 98765 });
  assert.equal(sim1.final_w, sim2.final_w);
  assert.equal(sim1.pre_spikes, sim2.pre_spikes);
  assert.equal(sim1.post_spikes, sim2.post_spikes);
  assert.equal(sim1.total_ltp, sim2.total_ltp);
  assert.equal(sim1.total_ltd, sim2.total_ltd);

  // Different seed produces different results
  const sim3 = core.simulate_poisson_drift(10, 10, 5, { seed: 54321 });
  assert.notEqual(sim1.final_w, sim3.final_w);
});

test('theory_dw matches continuous exponential and converges to discrete trace as dt -> 0', () => {
  const dt = 1.0;
  const tau = 20.0;
  const delta_t = 10.0;
  const dw_discrete = core.stdp_pair_dw(delta_t, { tau_plus: tau, dt });
  const dw_theory = core.theory_dw(delta_t, { tau_plus: tau });

  // With dt=1ms, tau=20ms: discrete is 0.02*0.95^10 = 0.0119747, theory is 0.02*exp(-0.5) = 0.0121306
  const diff = Math.abs(dw_discrete - dw_theory);
  assert.ok(diff < 0.0005, `expected close match, diff was ${diff}`);

  // For very fine dt = 0.01 ms:
  const dw_fine = core.stdp_pair_dw(delta_t, { tau_plus: tau, dt: 0.01 });
  assert.ok(
    Math.abs(dw_fine - dw_theory) < 1e-5,
    `as dt -> 0, discrete converges to continuous theory: got ${dw_fine} vs ${dw_theory}`
  );
});

test('parameter validation throws on non-positive time constants and step', () => {
  assert.throws(() => core.trace_step(1.0, 0, { tau: -5 }));
  assert.throws(() => core.trace_step(1.0, 0, { dt: 0 }));
  assert.throws(() => core.trace_k_steps(1.0, -1));
  assert.throws(() => core.stdp_pair_dw(5, { tau_plus: 0 }));
});
