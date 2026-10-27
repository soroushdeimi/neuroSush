import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core } = require('../../../src/neurosush/viz/web/plates/N08.js');

function relativeError(actual, expected) {
  return Math.abs(actual - expected) / expected;
}

test('LIF closed form has the hand-computed rheobase and ISI', () => {
  assert.equal(core.lifRheobase(), 1.5);
  assert.equal(core.lifRate(1.5), 0);
  // At 3 nA, T = 20 ln(3 / (3 - 1.5)) = 20 ln(2) ms.
  const expected = 1000 / (20 * Math.log(2));
  assert.ok(Math.abs(core.lifRate(3) - expected) < 1e-12);
});

test('closed-form LIF and ELIF rates match their simulated steady rates within 3 percent', () => {
  for (const [model, currents] of [
    ['lif', [1.6, 2.0, 3.0, 4.0]],
    ['elif', [1.31, 1.6, 2.0, 3.0]],
  ]) {
    for (const I of currents) {
      const theory = core.theoryRate(model, I);
      const simulated = core.simulateSteady(model, I, { warmup: 200, duration: 2400 });
      assert.ok(theory > 0, `${model} I=${I} should fire in theory`);
      assert.ok(simulated.n >= 2, `${model} I=${I} produced too few spikes`);
      assert.ok(
        relativeError(simulated.rate, theory) <= 0.03,
        `${model} I=${I}: sim=${simulated.rate}, theory=${theory}`
      );
    }
  }
});

test('both models stay silent at and below their rheobase', () => {
  for (const model of ['lif', 'elif']) {
    const Irh = core.rheobase(model);
    for (const I of [0, Irh - 0.2, Irh]) {
      assert.equal(core.theoryRate(model, I), 0, `${model} theory at ${I}`);
      const simulated = core.simulateSteady(model, I, { warmup: 100, duration: 800 });
      assert.equal(simulated.n, 0, `${model} simulation at ${I}`);
      assert.equal(simulated.rate, 0, `${model} rate at ${I}`);
    }
  }
});

test('f-I relations are monotonically increasing above rheobase', () => {
  for (const model of ['lif', 'elif']) {
    const Irh = core.rheobase(model);
    const rates = [0.02, 0.08, 0.25, 0.7, 1.5].map((dI) => core.theoryRate(model, Irh + dI));
    for (let i = 1; i < rates.length; i++) {
      assert.ok(rates[i] > rates[i - 1], `${model}: ${rates[i - 1]} !< ${rates[i]}`);
    }
  }
});

test('ELIF onset has the type-I square-root exponent', () => {
  const Irh = core.elifRheobase();
  const low = 0.002;
  const high = 0.032;
  const fLow = core.elifRate(Irh + low);
  const fHigh = core.elifRate(Irh + high);
  const exponent = Math.log(fHigh / fLow) / Math.log(high / low);
  assert.ok(Math.abs(exponent - 0.5) < 0.04, `onset exponent was ${exponent}`);
});

test('seeded current noise is exactly reproducible', () => {
  const options = { warmup: 100, duration: 900, sigma: 0.16, seed: 418 };
  const first = core.simulateSteady('elif', 1.31, options);
  const second = core.simulateSteady('elif', 1.31, options);
  assert.deepEqual(second, first);

  const changed = core.simulateSteady('elif', 1.31, { ...options, seed: 419 });
  assert.notDeepEqual(changed.spikes, first.spikes, 'a different seed must change the path');
});
