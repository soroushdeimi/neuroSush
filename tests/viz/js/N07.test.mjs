import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N07.js');
const p = { gL: 0.1, EL: -65, ge: 0.4, Ee: 0, gi: 0.5, Ei: -80, C: 1 };

test('1. vInfinity is the hand-computed conductance weighted mean', () => {
  // (-6.5 + 0 - 40) / 1.0 = -46.5 mV
  assert.equal(core.vInfinity(p), -46.5);
  assert.equal(core.totalConductance(p), 1.0);
});

test('2. effective time constant is capacitance divided by total conductance', () => {
  assert.equal(core.tauEff(p), 1);
  assert.equal(core.tauEff({ ...p, ge: 0, gi: 0 }), 10);
  assert.equal(core.tauEff({ ...p, C: 2.0 }), 2.0);
});

test('3. exact conductance step follows the closed form and lands at equilibrium', () => {
  const oneTau = core.exactStep(-65, p, 1);
  assert.ok(Math.abs(oneTau - (-46.5 + (-18.5) * Math.exp(-1))) < 1e-12);
  assert.ok(Math.abs(core.exactStep(-65, p, 100) - -46.5) < 1e-12);
});

test('4. currents calculation matches Ohm law I = g(E - v) for each pathway', () => {
  const iRest = core.currents(-65, p);
  assert.equal(iRest.leak, 0);
  assert.equal(iRest.exc, 0.4 * (0 - (-65))); // 26.0 pA
  assert.equal(iRest.inh, 0.5 * (-80 - (-65))); // -7.5 pA
  const sum = iRest.leak + iRest.exc + iRest.inh;
  assert.equal(sum, 18.5);
});

test('5. derivative equals sum of currents divided by capacitance', () => {
  const dV = core.derivative(-65, p);
  assert.equal(dV, 18.5 / p.C);
  const dVEquil = core.derivative(-46.5, p);
  assert.ok(Math.abs(dVEquil) < 1e-12, 'derivative at vInfinity must be 0');
});

test('6. shunting inhibition at rest has zero resting current but reduces an excitatory response', () => {
  const shunt = { gL: 0.1, EL: -65, ge: 0.4, Ee: 0, gi: 0.8, Ei: -65, C: 1 };
  assert.equal(core.currents(-65, shunt).inh, 0);
  const without = core.exactStep(-65, { ...shunt, gi: 0 }, 5);
  const withShunt = core.exactStep(-65, shunt, 5);
  assert.ok(withShunt < without, `shunted ${withShunt}, unshunted ${without}`);
  assert.ok(core.tauEff(shunt) < core.tauEff({ ...shunt, gi: 0 }));
});

test('7. current mode adds a voltage-independent input and exhibits no reversal ceiling', () => {
  const a = core.currentStep(-65, { ...p, ge: 0, gi: 0 }, 0.75, 2);
  const b = core.currentStep(-25, { ...p, ge: 0, gi: 0 }, 0.75, 2);
  assert.ok(a > core.currentStep(-65, { ...p, ge: 0, gi: 0 }, 0, 2));
  assert.ok(b > core.currentStep(-25, { ...p, ge: 0, gi: 0 }, 0, 2));
});

test('8. simulate produces multi-step trajectories matching sequential exact steps', () => {
  const steps = 10;
  const dt = 0.5;
  const traj = core.simulate(-65, p, dt, steps, 'conductance', 0);
  assert.equal(traj.length, steps + 1);
  let v = -65;
  for (let i = 1; i <= steps; i++) {
    v = core.exactStep(v, p, dt);
    assert.ok(Math.abs(traj[i] - v) < 1e-12);
  }
});

test('9. seeded helpers are deterministic and reproducible', () => {
  const train1 = core.seededPulseTrain(17, 30, 0.35);
  const train2 = core.seededPulseTrain(17, 30, 0.35);
  const train3 = core.seededPulseTrain(42, 30, 0.35);
  assert.deepEqual(train1, train2);
  assert.notDeepEqual(train1, train3);
});

test('10. invalid physical parameters fail early with descriptive error messages', () => {
  assert.throws(() => core.vInfinity({ ...p, gi: -1 }), /gi must be non-negative/);
  assert.throws(() => core.vInfinity({ ...p, ge: -0.5 }), /ge must be non-negative/);
  assert.throws(() => core.vInfinity({ ...p, gL: -0.1 }), /gL must be non-negative/);
  assert.throws(() => core.exactStep(-65, p, -1), /dt must be non-negative/);
  assert.throws(() => core.tauEff({ ...p, C: 0 }), /C must be positive/);
  assert.throws(() => core.currents(NaN, p), /v must be finite/);
});
