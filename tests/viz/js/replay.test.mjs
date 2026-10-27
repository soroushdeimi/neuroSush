import test from 'node:test';
import assert from 'node:assert/strict';
import Atlas from '../../../src/neurosush/viz/web/atlas.js';
import '../../../src/neurosush/viz/web/plates/demo.js';

function packed(values, shape, dtype = 'float32') {
  const data = dtype === 'uint8' ? new Uint8Array(values) : new Float32Array(values);
  return { dtype, shape, data: Buffer.from(data.buffer).toString('base64') };
}

for (const indices of [Array.from({ length: 500 }, (_, i) => i + 1), [], [3, 6, 9, 12]]) {
  test(`replay preserves ${indices.length} step indices and uses sample positions`, () => {
    const spikes = indices.flatMap((_, i) => [0, Number(i % 2 === 1)]);
    const trace = new Atlas.Trace({ steps: indices, series: {
      spikes: packed(spikes, [indices.length, 2], 'uint8')
    } });
    assert.equal(trace.steps, indices.length);
    assert.deepEqual(trace.indices, indices);
    assert.notEqual(trace.indices, indices);
    const events = indices.flatMap((_, i) => i % 2 === 1 ? [i] : []);
    assert.deepEqual(trace.events, events);
    const clock = new Atlas.Clock({ steps: trace.steps, events: trace.events });
    clock.seek(Infinity);
    assert.equal(clock.t, Math.max(0, indices.length - 1));
    clock.seek(-100);
    assert.equal(clock.t, 0);
    if (events.length) {
      clock.jumpEvent(1);
      assert.equal(clock.t, 1);
      assert.deepEqual(Array.from(trace.at('spikes', clock.t)), [0, 1]);
    } else {
      assert.throws(() => trace.at('spikes', 0), /out of bounds/);
    }
  });
}

test('shell uses library metadata and has no fallback version', () => {
  const oldDocument = globalThis.document;
  const oldWindow = globalThis.window;
  try {
    globalThis.window = { matchMedia: () => ({ matches: true }) };
    const stop = new Error('stop after shell decoding');
    for (const [meta, expected] of [[{ library: '1.2.3', version: 'wrong' }, '1.2.3'], [{ version: 'wrong' }, '']]) {
      const root = {};
      globalThis.document = { documentElement: { removeAttribute() {} }, getElementById: () => { throw stop; } };
      assert.throws(() => Atlas.start({ root, trace: { steps: [], meta } }), (err) => err === stop);
      assert.ok(root.innerHTML.includes(`<span class="atlas-mono">${expected}</span>`));
      assert.ok(!root.innerHTML.includes('wrong'));
      assert.ok(!root.innerHTML.includes('0.6.0'));
    }
  } finally {
    globalThis.document = oldDocument;
    globalThis.window = oldWindow;
  }
});

test('demo draws neuron zero with the same stride as its margin', () => {
  const oldDocument = globalThis.document;
  const oldWindow = globalThis.window;
  const oldFit = Atlas.fitCanvas;
  const oldLine = Atlas.lineSeries;
  try {
    const elements = new Map();
    globalThis.document = { getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, { style: {} });
      return elements.get(id);
    } };
    globalThis.window = { addEventListener() {} };
    Atlas.fitCanvas = () => ({ width: 800, height: 500 });
    const lines = [];
    Atlas.lineSeries = (_, xs, ys, style) => lines.push({ xs, ys, style });
    const arcs = [];
    const drawing = new Proxy({}, { get: (_, key) => key === 'arc' ? (...args) => arcs.push(args) : () => {} });
    const trace = new Atlas.Trace({ steps: [1, 2, 3], series: {
      v: packed([-60, 100, 200, 300, 400, -50, 101, 201, 301, 401, -40, 102, 202, 302, 402], [3, 5]),
      v_theory: packed([-59, 0, 0, 0, 0, -49, 0, 0, 0, 0, -39, 0, 0, 0, 0], [3, 5]),
      spikes: packed([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], [3, 5], 'uint8')
    } });
    Atlas.Plate.get('demo').mount({}, {
      trace, clock: { steps: 3, t: 1 }, colors: () => ({}),
      canvas: () => ({ getContext: () => drawing }), onFrame() {}, margin: {}
    });
    const y = v => 30 + (1 - (v + 62) / 24) * 430;
    assert.deepEqual(lines[0].ys, [-59, -49, -39].map(y));
    assert.deepEqual(lines[1].ys, [-50, -40].map(y));
    assert.deepEqual(lines[2].ys, [-60, -50].map(y));
    assert.equal(elements.get('demo-v-val').textContent, '-50.00 mV');
    assert.equal(elements.get('demo-v-theory').textContent, '-49.00 mV');
    assert.equal(arcs.filter(args => args[2] === 3).length, 1);
    assert.equal(arcs[0][1], y(-50));
    assert.equal(arcs.find(args => args[2] === 6)[1], y(-50));
  } finally {
    globalThis.document = oldDocument;
    globalThis.window = oldWindow;
    Atlas.fitCanvas = oldFit;
    Atlas.lineSeries = oldLine;
  }
});
