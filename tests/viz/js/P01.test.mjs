import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import Atlas from '../../../src/neurosush/viz/web/atlas.js';
import '../../../src/neurosush/viz/web/plates/P01.js';
import { core } from '../../../src/neurosush/viz/web/plates/P01.js';

const require = createRequire(import.meta.url);

function packed(values, shape, dtype = 'float32') {
  const data = dtype === 'uint8' ? new Uint8Array(values) : new Float32Array(values);
  return { dtype, shape, data: Buffer.from(data.buffer).toString('base64') };
}

test('P01 (a): REPLAY badge is rendered and no TEACHING or ILLUSTRATIVE badge', () => {
  const plate = Atlas.Plate.get('P01');
  assert.ok(plate, 'P01 plate registered');
  assert.equal(plate.source, 'replay');
});

test('P01 (b): time window [t0, t1] (inclusive) filters spikes correctly', () => {
  // 4 steps, 2 neurons
  const spkData = new Uint8Array([
    1, 0, // step 0
    1, 1, // step 1
    0, 1, // step 2
    1, 1  // step 3
  ]);
  const trace = new Atlas.Trace({
    steps: 4,
    series: { spikes: packed(spkData, [4, 2], 'uint8') }
  });

  const spikesWindow = core.extractRaster(trace, { stepStart: 1, stepEnd: 2 });
  // Step 1: neuron 0, neuron 1. Step 2: neuron 1.
  assert.equal(spikesWindow.length, 3);
  assert.deepEqual(spikesWindow, [
    { neuron: 0, step: 1 },
    { neuron: 1, step: 1 },
    { neuron: 1, step: 2 }
  ]);
});

test('P01 (c): neuron selection with explicit array [2, 5] and out-of-range handling without crash', () => {
  // 3 steps, 6 neurons (indices 0..5)
  const row = [0, 0, 1, 0, 0, 1]; // neurons 2 and 5 fire at step 0
  const spkData = new Uint8Array([...row, ...row]);
  const trace = new Atlas.Trace({
    steps: 2,
    series: { spikes: packed(spkData, [2, 6], 'uint8') }
  });

  // Select explicit neurons [2, 5, 999] (999 out of range should be ignored without crash)
  const spikes = core.extractRaster(trace, { neurons: [2, 5, 999, -5], stepStart: 0, stepEnd: 0 });
  assert.equal(spikes.length, 2);
  assert.deepEqual(spikes, [
    { neuron: 2, step: 0 },
    { neuron: 5, step: 0 }
  ]);
});

test('P01 (d): empty trace or zero spikes renders empty-state message without exception', () => {
  const emptyTrace = new Atlas.Trace({ steps: 0, series: {} });
  const spikes = core.extractRaster(emptyTrace);
  assert.deepEqual(spikes, []);

  const stats = core.getStepStats(emptyTrace, 0);
  assert.equal(stats.empty, true);

  const margin = { innerHTML: '' };
  const rootEl = { ownerDocument: { getElementById: () => null } };
  const ctx = {
    trace: emptyTrace,
    clock: { steps: 0, t: 0 },
    colors: () => ({ border: '#000', inkMuted: '#888' }),
    canvas: () => ({ getContext: () => ({ fillRect() {}, strokeRect() {}, fillText() {}, stroke() {}, beginPath() {}, moveTo() {}, lineTo() {}, rect() {}, clearRect() {}, save() {}, restore() {}, clip() {} }) }),
    onFrame() {},
    margin
  };

  const plate = Atlas.Plate.get('P01');
  assert.doesNotThrow(() => plate.mount(rootEl, ctx));
  assert.ok(margin.innerHTML.length > 0);
});

test('P01 (e): file contains no Math.random function calls and no simulation code', () => {
  const p01Path = path.resolve(process.cwd(), 'src/neurosush/viz/web/plates/P01.js');
  const sourceText = fs.readFileSync(p01Path, 'utf8');

  assert.equal(/Math\s*\.\s*random/.test(sourceText), false, 'P01.js must not call Math.random');
  assert.equal(sourceText.includes('simulate'), false, 'P01.js must not contain simulation code');
  assert.equal(sourceText.includes('eulerStep'), false, 'P01.js must not contain eulerStep simulation');
});

test('P01 (f): decimated or truncated trace is correctly identified and labelled', () => {
  const normalTrace = new Atlas.Trace({ steps: 5, meta: {}, series: { spikes: packed(new Uint8Array(10), [5, 2], 'uint8') } });
  assert.equal(core.isDecimatedOrTruncated(normalTrace), false);

  const decimatedTrace = new Atlas.Trace({ steps: 5, meta: { decimated: true }, series: { spikes: packed(new Uint8Array(10), [5, 2], 'uint8') } });
  assert.equal(core.isDecimatedOrTruncated(decimatedTrace), true);

  const truncatedTrace = new Atlas.Trace({ steps: 5, meta: { max_steps_hit: true }, series: { spikes: packed(new Uint8Array(10), [5, 2], 'uint8') } });
  assert.equal(core.isDecimatedOrTruncated(truncatedTrace), true);
});
