import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { core, S } = require('../../../src/neurosush/viz/web/plates/N04.js');

test('izhikevichStep by hand matches exact analytical value and library', () => {
  // Hand-computed step:
  // Given: v = -65.0 mV, u = -13.0 pA, I = 10.0 pA, a = 0.02, b = 0.2, dt = 1.0 ms, substeps = 2.
  // h = dt / 2 = 0.5 ms.
  // Substep 1:
  // dv1 = 0.04 * (-65)^2 + 5 * (-65) + 140 - (-13) + 10 = 169 - 325 + 140 + 23 = 7.0 mV/ms.
  // v_mid = -65.0 + 0.5 * 7.0 = -61.5 mV.
  // Substep 2:
  // dv2 = 0.04 * (-61.5)^2 + 5 * (-61.5) + 140 + 23 = 151.29 - 307.5 + 140 + 23 = 6.79 mV/ms.
  // v_next = -61.5 + 0.5 * 6.79 = -58.105 mV.
  // u update:
  // du = 0.02 * (0.2 * (-58.105) - (-13.0)) = 0.02 * (-11.621 + 13.0) = 0.02 * 1.379 = 0.02758 pA.
  // u_next = -13.0 + 1.0 * 0.02758 = -12.97242 pA.
  const res = core.izhikevichStep(-65.0, -13.0, 10.0, 0.02, 0.2, 1.0, 2);
  assert.ok(
    Math.abs(res.v - -58.105) < 1e-12,
    `v was expected -58.105, got ${res.v}`
  );
  assert.ok(
    Math.abs(res.u - -12.97242) < 1e-12,
    `u was expected -12.97242, got ${res.u}`
  );
});

test('izhikevichTerms decomposes step increments additively', () => {
  // Hand-computed increments:
  // h = 0.5:
  // Substep 1: quad = 0.5 * 0.04 * 4225 = 84.5; lin = 0.5 * (-325 + 140) = -92.5; rec = 0.5 * 13 = 6.5; inp = 5.0
  // Substep 2: quad = 0.5 * 0.04 * 3782.25 = 75.645; lin = 0.5 * (-307.5 + 140) = -83.75; rec = 6.5; inp = 5.0
  // Total quad = 160.145, lin = -176.25, rec = 13.0, inp = 10.0 -> total = 6.895
  const terms = core.izhikevichTerms(-65.0, -13.0, 10.0, 1.0, 2);
  assert.ok(Math.abs(terms.quadratic - 160.145) < 1e-12);
  assert.ok(Math.abs(terms.linear - -176.25) < 1e-12);
  assert.ok(Math.abs(terms.recovery - 13.0) < 1e-12);
  assert.ok(Math.abs(terms.input - 10.0) < 1e-12);
  assert.ok(Math.abs(terms.total - 6.895) < 1e-12);
  assert.ok(Math.abs(-65.0 + terms.total - -58.105) < 1e-12);
});

test('izhikevichReset handles above-threshold and subthreshold transitions', () => {
  // Spike event: v >= 30 mV -> v = c, u = u + d
  const spikeRes = core.izhikevichReset(32.0, -5.0, 30.0, -65.0, 8.0);
  assert.equal(spikeRes.spike, true);
  assert.equal(spikeRes.v, -65.0);
  assert.equal(spikeRes.u, 3.0);

  // Subthreshold: v < 30 mV -> no change
  const subRes = core.izhikevichReset(25.0, -5.0, 30.0, -65.0, 8.0);
  assert.equal(subRes.spike, false);
  assert.equal(subRes.v, 25.0);
  assert.equal(subRes.u, -5.0);
});

test('parameter validation throws on invalid input naming the offender', () => {
  assert.throws(
    () => core.izhikevichStep(-65.0, -13.0, 10.0, -0.01, 0.2),
    /a must be positive, got -0.01/
  );
  assert.throws(
    () => core.izhikevichStep(-65.0, -13.0, 10.0, 0.02, 0.2, 1.0, 0),
    /substeps must be a positive integer, got 0/
  );
  assert.throws(
    () => core.izhikevichReset(20.0, 0.0, 30.0, 35.0, 8.0),
    /c must be less than vPeak \(30\), got 35/
  );
});

test('published patterns reproduce their exact known spike counts in 1000 ms', () => {
  // Tolerance statement: under standard Izhikevich 2003 formulation (dt=1.0 ms, 2 substeps/step),
  // each canonical pattern produces a deterministic integer spike count in 1000 ms.
  // Tolerance is 0 spikes for the deterministic benchmark.
  const expectedCounts = {
    RS: 20, // Regular Spiking: spike-frequency adaptation (mean ISI 51.6 ms)
    IB: 28, // Intrinsically Bursting: initial burst followed by single spikes
    CH: 43, // Chattering: rhythmic bursts of 2-5 spikes
    FS: 63, // Fast Spiking: high-frequency tonic firing (mean ISI 16.0 ms)
    LTS: 44, // Low-Threshold Spiking: rapid onset with sustained rate
    RZ: 80, // Resonator: high-frequency oscillations under DC drive
    RE: 2, // Rebound Spiker: 2 spikes triggered post-inhibition
    MM: 40, // Mixed Mode: dual bursting and regular rhythm
  };

  for (const [key, expected] of Object.entries(expectedCounts)) {
    const preset = core.CELL_PRESETS[key];
    const res = core.simulate(preset, 1000, 1.0, 2);
    assert.equal(
      res.spikes.length,
      expected,
      `${key} expected ${expected} spikes in 1000 ms, got ${res.spikes.length}`
    );
  }
});

test('simulation is deterministic bit for bit across runs', () => {
  const p = core.CELL_PRESETS.RS;
  const run1 = core.simulate(p, 500, 1.0, 2);
  const run2 = core.simulate(p, 500, 1.0, 2);

  assert.equal(run1.spikes.length, run2.spikes.length);
  for (let i = 0; i < run1.spikes.length; i++) {
    assert.equal(run1.spikes[i], run2.spikes[i]);
  }
  for (let i = 0; i < run1.history.length; i++) {
    assert.equal(run1.history[i].v, run2.history[i].v);
    assert.equal(run1.history[i].u, run2.history[i].u);
  }
});

test('theory ghost nullclines and fixed points match analytical closed form', () => {
  // For I = 0, b = 0.2:
  // 0.04 v^2 + (5 - 0.2) v + 140 = 0
  // 0.04 v^2 + 4.8 v + 140 = 0
  // v^2 + 120 v + 3500 = 0 => (v + 70)(v + 50) = 0
  // Roots: v_rest = -70.0 mV, v_thresh = -50.0 mV.
  const fix = core.fixedPoints(0, 0.2);
  assert.equal(fix.hasEquilibrium, true);
  assert.ok(Math.abs(fix.vRest - -70.0) < 1e-12, `vRest was ${fix.vRest}`);
  assert.ok(Math.abs(fix.vThresh - -50.0) < 1e-12, `vThresh was ${fix.vThresh}`);
  assert.ok(Math.abs(fix.uRest - -14.0) < 1e-12, `uRest was ${fix.uRest}`);
  assert.ok(Math.abs(fix.uThresh - -10.0) < 1e-12, `uThresh was ${fix.uThresh}`);

  // Nullclines at fixed points evaluate to identical values
  const vNullVal = core.vNullcline(-70.0, 0);
  const uNullVal = core.uNullcline(-70.0, 0.2);
  assert.ok(Math.abs(vNullVal - uNullVal) < 1e-12);

  // Bifurcation: with I = 10, discriminant becomes negative (limit cycle active)
  const fixSpiking = core.fixedPoints(10.0, 0.2);
  assert.equal(fixSpiking.hasEquilibrium, false);
  assert.ok(fixSpiking.delta < 0, `delta was ${fixSpiking.delta} (must be < 0)`);
});

test('pattern fingerprint classifies all 8 canonical patterns correctly from ISI stats', () => {
  const expectedClasses = {
    RS: 'Regular Spiking (RS)',
    IB: 'Intrinsically Bursting (IB)',
    CH: 'Chattering (CH)',
    FS: 'Fast Spiking (FS)',
    LTS: 'Low-Threshold Spiking (LTS)',
    RZ: 'Resonator (RZ)',
    RE: 'Rebound Spiker (RE)',
    MM: 'Mixed Mode (MM)',
  };

  for (const [key, label] of Object.entries(expectedClasses)) {
    const preset = core.CELL_PRESETS[key];
    const res = core.simulate(preset, 1000, 1.0, 2);
    assert.equal(
      res.fingerprint.label,
      label,
      `${key} was classified as '${res.fingerprint.label}', expected '${label}'`
    );
  }

  // Edge cases
  const quiescent = core.computeFingerprint([], 1000);
  assert.equal(quiescent.label, 'Quiescent');
  const phasic = core.computeFingerprint([50.0], 1000);
  assert.equal(phasic.label, 'Phasic Spike');
});

test('the string table is complete and in English', () => {
  assert.equal(S.plateId, 'N04');
  assert.ok(S.title.includes('Izhikevich'));
  assert.ok(S.ch1Title.length > 0);
  assert.ok(S.predictExplanation.length > 0);
  assert.ok(S.glossaryU.length > 0);
});
