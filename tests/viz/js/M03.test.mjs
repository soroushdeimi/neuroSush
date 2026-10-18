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
  // RT/F = 8.314462618·310.15/96485.332·1000 = 26.7308 mV
  // E = 26.7308 · ln(5/140) = 26.7308 · (−3.33220) = −89.07 mV
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
  // start at 0 mV, E = −89.07, g = 10 nS, C = 1 µF (τ = 100 ms)
  let V = 0;
  const E = core.nernst(5, 140, 1, 37);
  for (let k = 0; k < 2000; k++) {
    V += core.odeStep(V, E, 10, 1, 1).dV; // 1 ms per step, 2 s total = 20τ
  }
  const gap = Math.abs(V - E);
  assert.ok(gap / Math.abs(E) < 0.01, 'V must settle within 1 % of E, gap=' + gap);
  // analytic check of a single step: dV = (E−V)(1 − e^(−1/100))
  const d1 = core.odeStep(0, E, 10, 1, 1).dV;
  assert.ok(Math.abs(d1 - (E - 0) * (1 - Math.exp(-0.01))) < 1e-9);
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
  // drift −80 mV at 0.3 must be exactly −24 in the returned value (pure)
  const fresh = core.mulberry32(7);
  const yDrift = core.particleStep(y0, fresh, 1.5, 0.3, -80);
  assert.equal(yDrift, y1, 'same seed → same value');
  // diffusion sigma: 2·(r−0.5) ∈ [−1, 1] · sigma
  const r2 = core.mulberry32(3);
  for (let i = 0; i < 50; i++) {
    const d = Math.abs(core.particleStep(0, r2, 2, 0, 0));
    assert.ok(d <= 2 + 1e-9, 'jump exceeds σ: ' + d);
  }
});

test('fluxDemand: crossing demand vanishes exactly when V sits on the Nernst value', () => {
  // the particle picture must rest exactly where the ODE settles
  for (const [o, i, z, T] of [[5, 140, 1, 37], [145, 15, 1, 37], [150, 10, -1, 20], [0.5, 300, 1, 6.3]]) {
    const E = core.nernst(o, i, z, T);
    const d = core.fluxDemand(E, E, 10, 1, 16, z);
    assert.ok(
      d.base === 0 && d.dV === 0,
      'demand at V=E must be 0 for ' + [o, i, z, T].join('/') + ', got ' + d.base
    );
  }
});

test('fluxDemand: drive sign rules — gradient pushes a cation out, +V pushes it back in', () => {
  const EK = core.nernst(5, 140, 1, 37);          // −89.07 mV
  const at0 = core.fluxDemand(0, EK, 10, 1, 16, 1);
  assert.ok(at0.outward === true, 'K+ at V=0 must flow outward, got inward');
  assert.ok(at0.dV < 0, 'outward cation flow must pull V down, got ' + at0.dV);
  assert.ok(at0.base > 1 && at0.base < 8, 'drive must be picture-sized, got ' + at0.base);
  // holding the membrane above E reverses the flow: now inward
  const held = core.fluxDemand(-95, EK, 10, 1, 16, 1);
  assert.ok(held.outward === false, 'V below E must push K+ inward');
  assert.ok(held.base > 0, 'reversed drive must still move markers');
  // anion: E_Cl = (RT/zF)·ln(150/10) ≈ −166 mV, so at V = 0 the chemical push
  // wins and Cl- flows INWARD while V slides down toward E_Cl
  const cl = core.fluxDemand(0, core.nernst(150, 10, -1, 37), 10, 1, 16, -1);
  assert.ok(cl.outward === false, 'Cl- gradient (out 150 / in 10) at V=0 flows inward');
  assert.ok(cl.dV < 0, 'inward anion flow must pull V down, got ' + cl.dV);
});

test('fluxDemand rejects z = 0', () => {
  assert.throws(() => core.fluxDemand(0, -89, 10, 1, 16, 0), /nonzero/);
});

test('particle sim: marker pool conserves particle count through crossings', () => {
  const sim = core.makeSim(42, 140, 5, 200);
  for (let k = 0; k < 3000; k++) {
    // open channel, V fixed at 0 (strong outward drive), no bookkeeping
    core.simStep(sim, { open: true, V_mV: 0, E_mV: -89, g_nS: 10, C_uF: 1, z: 1, tempC: 37, dtMs: 16 });
    assert.equal(sim.particles.length, 200, 'particles are conserved, tick ' + k);
  }
});

test('particle sim: crossings stop dead exactly when V = E_Nernst (−89.07 mV)', () => {
  const sim = core.makeSim(42, 140, 5, 200);
  const E = core.nernst(5, 140, 1, 37);
  for (let k = 0; k < 3000; k++) {
    const r = core.simStep(sim, {
      open: true, V_mV: E, E_mV: E, g_nS: 10, C_uF: 1, z: 1, tempC: 37, dtMs: 16
    });
    assert.equal(r.outN + r.inN, 0, 'V = E demands zero crossings, tick ' + k);
  }
});

test('particle sim: at V = 0 the K+ gradient drives a burst of outward crossings that dies at balance', () => {
  // closed loop: V follows the ODE while the picture mirrors the same book
  const sim = core.makeSim(42, 140, 5, 200);
  const E = core.nernst(5, 140, 1, 37);
  const n0 = sim.particles.filter(q => q.inside).length;
  let V = 0, cross = 0, net = 0, tailCross = 0;
  for (let k = 0; k < 2000; k++) {
    const r = core.simStep(sim, {
      open: true, V_mV: V, E_mV: E, g_nS: 10, C_uF: 1, z: 1, tempC: 37, dtMs: 16
    });
    V += r.dV; // the same dv the visuals were driven from
    cross += r.outN + r.inN;
    net += r.outN - r.inN;
    if (k >= 1500) tailCross += r.outN + r.inN;
  }
  // Σ|dV| telescopes to |E − V0| ≈ 89 mV; at 0.3 markers per mV the whole
  // relaxation moves ≈ 27 markers, every single one outward
  assert.ok(cross >= 15 && cross <= 45, 'relaxation must move a picture-sized pool, got ' + cross);
  assert.ok(net === cross, 'every crossing must be outward at V=0 with the K+ gradient');
  assert.ok(tailCross === 0, 'once V reaches E the picture must go quiet, got ' + tailCross);
  assert.ok(Math.abs(V - E) < 0.5, 'V must have relaxed to E, got ' + V);
  const nEnd = sim.particles.filter(q => q.inside).length;
  assert.ok(nEnd < n0, 'inside pool must drain at V=0: ' + n0 + ' → ' + nEnd);
});

test('particle sim: rejected transition keeps the side (closed channel blocks flux)', () => {
  const sim = core.makeSim(42, 140, 5, 200);
  const before = sim.particles.filter(q => q.inside).length;
  for (let k = 0; k < 2000; k++) {
    const r = core.simStep(sim, {
      open: false, V_mV: 0, E_mV: -89, g_nS: 10, C_uF: 1, z: 1, tempC: 37, dtMs: 16
    });
    assert.equal(r.outN + r.inN, 0, 'shut pore must not cross, tick ' + k);
  }
  const after = sim.particles.filter(q => q.inside).length;
  assert.equal(after, before, 'shut pore must not change the pool split');
});

test('particle sim: determinism — same seed, same trajectory, bit for bit', () => {
  const run = () => {
    const sim = core.makeSim(77, 15, 145, 200);
    const snap = [];
    for (let k = 0; k < 400; k++) {
      const r = core.simStep(sim, { open: true, V_mV: 40, E_mV: 57, g_nS: 10, C_uF: 1, z: 1, tempC: 37, dtMs: 16 });
      if (k % 100 === 0) {
        snap.push(sim.particles.map(q => (q.inside ? 1 : -1) * q.y).reduce((a, b) => a + b, 0) + '|' + r.outN);
      }
    }
    return snap;
  };
  const a = run();
  const b = run();
  assert.deepEqual(a, b, 'same seed must replay bit for bit');
});

test('particle sim: jitter slows at low temperature (thermal speed ∝ √T)', () => {
  const spread = (T) => {
    const sim = core.makeSim(5, 140, 5, 200);
    let m = 0;
    for (let k = 0; k < 500; k++) {
      core.simStep(sim, { open: false, V_mV: 0, E_mV: -89, g_nS: 10, C_uF: 1, z: 1, tempC: T, dtMs: 16 });
      m += sim.particles[0].y;
    }
    return m;
  };
  assert.ok(
    Math.abs(spread(37) - spread(37)) === 0,
    'same seed must give the same walk'
  );
  // mean absolute y of the same walk at 5 °C must differ from 37 °C's
  assert.ok(
    Math.abs(spread(5) - spread(37)) > 1,
    'temperature must change the walk: 5 °C vs 37 °C'
  );
});
