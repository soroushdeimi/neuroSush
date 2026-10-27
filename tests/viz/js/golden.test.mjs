import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const plateDirectory = new URL('../../../src/neurosush/viz/web/plates/', import.meta.url);
const golden = JSON.parse(readFileSync(new URL('../../golden/equations.json', import.meta.url), 'utf8'));
const plates = Object.fromEntries(readdirSync(plateDirectory)
  .filter(name => /^[A-Z][0-9]{2}\.js$/.test(name))
  .map(name => [name.slice(0, -3), require(fileURLToPath(new URL(name, plateDirectory))).core])
  .filter(([, core]) => core));

const NOT_PORTED = [];

// These float64 vectors use the generator’s per-case budgets for rounding and libm differences.
// Match Python math.isclose: the larger relative or absolute error budget applies.
function close(actual, expected, label, tolerance) {
  const { abs, rel } = tolerance;
  assert.ok(Number.isFinite(abs) && abs > 0 && Number.isFinite(rel) && rel > 0,
    `${label}: invalid tolerance ${JSON.stringify(tolerance)}`);
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${label}: nonfinite ${actual}, ${expected}`);
  const limit = Math.max(abs, rel * Math.max(Math.abs(actual), Math.abs(expected)));
  assert.ok(Math.abs(actual - expected) <= limit,
    `${label}: got ${actual}, expected ${expected}, difference ${Math.abs(actual - expected)}, tolerance ${limit}`);
}

function vector(p, expected, evaluate, label, tolerance) {
  assert.equal(expected.length, p.v.length, `${label}: vector length`);
  p.v.forEach((v, i) => close(evaluate(v, i), expected[i], `${label}, v=${v}`, tolerance));
}

// N08 exposes its equation constants through P; restore them after each case.
function withParameters(p, evaluate) {
  const core = plates.N08;
  const saved = { ...core.P };
  Object.assign(core.P, { V_REST: p.v_rest, R: p.resistance, TAU: p.tau,
    DELTA: p.delta, THETA_RH: p.theta_rh, V_RESET: p.v_reset, LIF_THRESHOLD: p.threshold });
  try { return evaluate(core); } finally { Object.assign(core.P, saved); }
}

const cases = [];
function port(group, plate, name, check, applies = () => true) {
  assert.equal(typeof plates[plate]?.[name], 'function', `${plate}.${name} missing`);
  assert.ok(golden[group].some(applies), `${plate}.${name}: no applicable ${group} vectors`);
  cases.push({ group, plate, name, check, applies });
}

port('izhikevich_terms', 'N04', 'izhikevichTerms', ({ inputs: p, outputs, total, tolerance }) => {
  const evaluate = v => plates.N04.izhikevichTerms(v, p.u, p.current, p.dt, p.substeps);
  for (const term of Object.keys(outputs)) {
    vector(p, outputs[term], v => evaluate(v)[term], term, tolerance);
  }
  vector(p, total, v => evaluate(v).total, 'Izhikevich increment', tolerance);
});

port('adex_terms', 'N03', 'adex_terms', ({ inputs: p, outputs, total, tolerance }) => {
  for (const term of Object.keys(outputs)) {
    vector(p, outputs[term], v => plates.N03.adex_terms(v, p.current, p.omega, p)[term], term, tolerance);
  }
  vector(p, total, v => Object.values(plates.N03.adex_terms(v, p.current, p.omega, p))
    .reduce((sum, value) => sum + value, 0), 'AdEx increment', tolerance);
});

port('lif_derivative', 'N01', 'lif_derivative', ({ inputs: p, output, tolerance }) =>
  vector(p, output, v => plates.N01.lif_derivative(v, p.current, p), 'LIF drive', tolerance));
port('lif_derivative', 'N08', 'lifDrive', ({ inputs: p, output, tolerance }) =>
  withParameters(p, core => vector(p, output, v => core.lifDrive(v, p.current), 'LIF drive', tolerance)));
port('euler_step', 'N01', 'euler_step', ({ inputs: p, output, tolerance }) =>
  vector(p, output, (v, i) => plates.N01.euler_step(v, p.tau_dv_dt[i], p), 'Euler step', tolerance));
port('exponential_boost', 'N02', 'boost', ({ inputs: p, output, tolerance }) =>
  vector(p, output, v => plates.N02.boost(v, p), 'ELIF boost', tolerance));
port('lif_terms', 'N01', 'euler_step', ({ inputs: p, total, tolerance }) =>
  vector(p, total, v => plates.N01.euler_step(v, plates.N01.lif_derivative(v, p.current, p), p) - v, 'LIF increment', tolerance));
port('lif_terms', 'N08', 'lifDrive', ({ inputs: p, total, tolerance }) =>
  withParameters(p, core => vector(p, total, v => core.lifDrive(v, p.current) * p.dt / p.tau, 'LIF increment', tolerance)));
for (const name of ['numerator', 'derivative', 'euler']) {
  port('elif_terms', 'N02', name, ({ inputs: p, total, tolerance }) => {
    const scale = name === 'numerator' ? p.dt / p.tau : name === 'derivative' ? p.dt : 1;
    vector(p, total, v => (plates.N02[name](v, p.current, p) - (name === 'euler' ? v : 0)) * scale, name, tolerance);
  });
}
port('elif_terms', 'N08', 'elifDrive', ({ inputs: p, total, tolerance }) =>
  withParameters(p, core => vector(p, total, v => core.elifDrive(v, p.current) * p.dt / p.tau, 'ELIF increment', tolerance)));
for (const [group, names] of [
  ['gate_rates', ['alpha_m', 'beta_m', 'alpha_h', 'beta_h', 'alpha_n', 'beta_n']],
  ['gate_steady_state', ['m_inf', 'h_inf', 'n_inf']],
  ['gate_time_constant', ['tau_m', 'tau_h', 'tau_n']],
]) {
  for (const name of names) {
    port(group, 'N05', name, ({ inputs: p, outputs, tolerance }) =>
      vector(p, outputs[name], v => plates.N05[name](v), name, tolerance));
  }
}
port('lif_rate', 'N01', 'theoryRate', ({ inputs: p, output, tolerance }) =>
  close(plates.N01.theoryRate(p.current, p), output, 'LIF rate', tolerance));
port('lif_rate', 'N08', 'lifRate', ({ inputs: p, output, tolerance }) => {
  // N08's closed form specializes to reset=rest and no refractory period.
  withParameters(p, core => close(core.lifRate(p.current), output, 'LIF rate', tolerance));
}, ({ inputs: p }) => p.v_reset === p.v_rest && p.t_ref === 0);
for (const [plate, name] of [['M01', 'eulerStep'], ['M01', 'landingValue'],
  ['Y01', 'trace_step'], ['Y01', 'trace_k_steps'], ['Y02', 'traceDecay']]) {
  port('trace_decay', plate, name, ({ inputs: p, output, tolerance }) => {
    let value = 1;
    output.forEach((expected, k) => {
      if (name === 'landingValue') value = k === 0 ? 1 : plates.M01.landingValue('euler', p.dt, p.tau, k * p.dt);
      else if (name === 'trace_k_steps') value = plates.Y01.trace_k_steps(1, k, p);
      else if (name === 'traceDecay') value = k === 0 ? 1 : plates.Y02.traceDecay(value, p.tau, p.dt);
      else if (k > 0) value = name === 'eulerStep' ? plates.M01.eulerStep(value, p.dt, p.tau) : plates.Y01.trace_step(value, 0, p);
      close(value, expected, `${plate}.${name}, k=${k}`, tolerance);
    });
  });
}

const groups = Object.keys(golden).filter(key => Array.isArray(golden[key]));
const normalize = name => name.replaceAll('_', '').toLowerCase();
function checkCoverage(availablePlates, registeredCases) {
  for (const [plate, core] of Object.entries(availablePlates)) {
    for (const [name, fn] of Object.entries(core)) {
      if (typeof fn !== 'function') continue;
      const group = groups.find(group => normalize(group) === normalize(name));
      if (group) assert.ok(registeredCases.some(c => c.plate === plate && c.name === name),
        `${plate}.${name} matches golden group ${group} but has no case`);
    }
  }
}

test('every golden group is exercised or explicitly not ported', () => {
  assert.deepEqual([...new Set([...cases.map(c => c.group), ...NOT_PORTED.map(([group]) => group)])].sort(), groups.toSorted());
  for (const [group, reason] of NOT_PORTED) {
    assert.ok(reason.length > 0);
    assert.ok(!cases.some(c => c.group === group), `${group} is both covered and NOT_PORTED`);
  }
  checkCoverage(plates, cases);
});
test('coverage rejects an exported golden function without cases', () => {
  assert.throws(() => checkCoverage({ extra: { lif_derivative() {} } }, cases), /extra.lif_derivative.*no case/);
});
for (const { group, plate, name, check, applies } of cases) {
  golden[group].forEach((entry, i) => {
    if (applies(entry)) test(`${plate}.${name}: ${group}[${i}]`, () => check(entry));
  });
}

for (const rel of [1e-12, 1e-11]) {
  test(`numerical comparison uses per-case relative tolerance ${rel}`, () => {
    const tolerance = { rel, abs: 1e-14 };
    close(1e9 * (1 + rel / 2), 1e9, 'inside', tolerance);
    assert.throws(() => close(1e9 * (1 + 2 * rel), 1e9, 'outside', tolerance), /outside/);
    close(5e-15, 0, 'absolute floor', tolerance);
    assert.throws(() => close(2e-14, 0, 'outside floor', tolerance), /outside floor/);
    assert.throws(() => close(NaN, 0, 'nonfinite', tolerance), /nonfinite/);
  });
}
