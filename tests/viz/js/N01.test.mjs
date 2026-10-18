import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N01.js');

test('Euler step by hand matches spec', () => {
  // Spec: v=-70, I=5, dt=0.1: dv = 0.1 * (0 + 50) / 20 = 0.25, v_next = -69.75
  const params = { tau: 20.0, v_rest: -70.0, resistance: 10.0, dt: 0.1 };
  const tau_dv_dt = core.lif_derivative(-70.0, 5.0, params);
  assert.strictEqual(tau_dv_dt, 50.0);

  const v_next = core.euler_step(-70.0, tau_dv_dt, params);
  assert.strictEqual(Math.round(v_next * 1e6) / 1e6, -69.75);

  const dv = v_next - (-70.0);
  assert.strictEqual(Math.round(dv * 1e6) / 1e6, 0.25);

  const state = core.createState({
    tau: 20.0,
    v_rest: -70.0,
    threshold: -40.0,
    v_reset: -75.0,
    resistance: 10.0,
    current: 5.0,
    dt: 0.1,
    t_ref: 2.0,
  });
  const out = core.step(state, 0.1);
  assert.strictEqual(Math.round(out.v * 1e6) / 1e6, -69.75);
  assert.strictEqual(out.spiked, false);
});

test('no spike for R * I <= v_th - v_rest (rheobase threshold)', () => {
  // v_th = -40, v_rest = -70 -> v_th - v_rest = 30 mV. R = 10 -> I_rh = 3.0 nA.
  for (const I of [0.0, 1.0, 2.0, 2.9, 3.0]) {
    const res = core.simulateRun({ current: I }, 5000);
    assert.strictEqual(res.spikes, 0, `Expected 0 spikes for I=${I}, got ${res.spikes}`);
    const theoryRate = core.theoryRate(I);
    assert.strictEqual(theoryRate, 0, `Expected theory rate 0 for I=${I}`);
  }

  // At I=2.0 nA, membrane asymptotically settles at v_rest + R*I = -70 + 20 = -50 mV
  const res2 = core.simulateRun({ current: 2.0 }, 5000);
  assert.ok(
    Math.abs(res2.finalV - (-50.0)) < 1e-4,
    `Final voltage ${res2.finalV} did not converge to -50.0 mV`
  );
});

test('simulated rate within 3 percent of closed-form 1000/T for I=5, 8, 12 (5000 ms)', () => {
  for (const I of [5.0, 8.0, 12.0]) {
    const T = core.theoryISI(I);
    assert.ok(Number.isFinite(T) && T > 0, `Invalid theory ISI for I=${I}`);
    const theoryRate = 1000.0 / T;

    const sim = core.simulateRun({ current: I }, 5000);
    const diffPct = (Math.abs(sim.rate - theoryRate) / theoryRate) * 100.0;
    assert.ok(
      diffPct <= 3.0,
      `I=${I}: sim rate ${sim.rate.toFixed(2)} Hz differs from theory ${theoryRate.toFixed(2)} Hz by ${diffPct.toFixed(2)}% (limit 3%)`
    );
  }
});

test('no two spikes closer than refractory time t_ref', () => {
  const t_ref = 2.0;
  for (const I of [5.0, 10.0, 20.0]) {
    const sim = core.simulateRun({ current: I, t_ref }, 5000);
    assert.ok(sim.spikes > 10, `Expected spikes for I=${I}`);
    for (const isi of sim.isiList) {
      assert.ok(
        isi >= t_ref - 1e-9,
        `ISI ${isi} ms is smaller than refractory period ${t_ref} ms for I=${I}`
      );
    }
    assert.ok(
      sim.minISI >= t_ref - 1e-9,
      `Min ISI ${sim.minISI} is less than refractory time ${t_ref}`
    );
  }
});

test('seed determinism with noise', () => {
  const params = {
    current: 4.0,
    noise_enabled: true,
    noise_sigma: 0.8,
    seed: 77777,
  };

  const simA = core.simulateRun(params, 1000);
  const simB = core.simulateRun(params, 1000);

  assert.strictEqual(simA.spikes, simB.spikes);
  assert.strictEqual(simA.finalV, simB.finalV);
  assert.deepStrictEqual(simA.spikeTimes, simB.spikeTimes);

  const simDiffSeed = core.simulateRun({ ...params, seed: 88888 }, 1000);
  assert.notStrictEqual(
    simA.finalV,
    simDiffSeed.finalV,
    'Different seeds must produce different trajectories'
  );
});

test('closed-form voltage trajectory matches Euler integration', () => {
  const I = 2.0;
  const params = { tau: 20.0, v_rest: -70.0, resistance: 10.0, current: I, dt: 0.05 };
  const state = core.createState(params);

  // Check at 10 ms, 20 ms, 40 ms
  for (const targetT of [10.0, 20.0, 40.0]) {
    while (state.t < targetT - 1e-9) {
      core.step(state, params.dt);
    }
    const v_theory = core.theoryVoltage(state.t, I, params);
    assert.ok(
      Math.abs(state.v - v_theory) < 0.1,
      `At t=${state.t.toFixed(1)} ms, sim V=${state.v.toFixed(3)} differs from theory ${v_theory.toFixed(3)}`
    );
  }
});

test('droplets simulation is deterministic and rate matches physical drive', () => {
  const d1 = core.createDroplets(42);
  const d2 = core.createDroplets(42);

  // At zero current and v = v_rest, no droplets should spawn
  d1.update(100, 0.0, -70.0, -70.0, 100, 50, 200, 150, 300, 350);
  assert.strictEqual(d1.inflow.length, 0);
  assert.strictEqual(d1.leak.length, 0);

  // At high current (5 nA) and depolarized membrane (v = -50 mV), droplets spawn
  for (let s = 0; s < 30; s++) {
    d1.update(16.6, 5.0, -50.0, -70.0, 100, 50, 200, 150, 300, 350);
    d2.update(16.6, 5.0, -50.0, -70.0, 100, 50, 200, 150, 300, 350);
  }
  assert.ok(d1.inflow.length > 0, 'Inflow droplets should spawn when current is active');
  assert.ok(d1.leak.length > 0, 'Leak droplets should spawn when membrane is depolarized');

  // Bit-for-bit determinism under same seed
  assert.strictEqual(d1.inflow.length, d2.inflow.length);
  assert.strictEqual(d1.leak.length, d2.leak.length);
  for (let i = 0; i < d1.inflow.length; i++) {
    assert.strictEqual(d1.inflow[i].x, d2.inflow[i].x);
    assert.strictEqual(d1.inflow[i].y, d2.inflow[i].y);
    assert.strictEqual(d1.inflow[i].vy, d2.inflow[i].vy);
  }
});

