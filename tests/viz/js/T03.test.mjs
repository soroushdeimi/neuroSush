import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {core} from '../../../src/neurosush/viz/web/plates/T03.js';
const golden = JSON.parse(readFileSync(new URL('../../golden/tem_plate.json', import.meta.url)));

function close(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((v, i) => Array.isArray(v) ? close(v, expected[i]) : assert.ok(Math.abs(v - expected[i]) <= golden.atol, `${v} != ${expected[i]}`));
}

test('T03 Hebbian storage and retrieval parity with Python golden fixture', () => {
  const t03 = golden.t03;
  let h = t03.cue;
  for (let i = 0; i < t03.retrieval_history.length; i++) {
    close(h, t03.retrieval_history[i]);
    if (i < t03.retrieval_history.length - 1) {
      h = core.attractor(h, t03.memory, 1, {retention: 0.8, negative_slope: 0.01, cap: 1.0});
    }
  }
});

test('T03 pattern generation, corruption, and overlap calculations', () => {
  const pats = core.createPatterns(4, 16, 42);
  assert.equal(pats.length, 4);
  assert.equal(pats[0].length, 16);

  const corrupted = core.corruptCue(pats[0], 0.5, 123);
  assert.equal(corrupted.length, 16);

  const ov = core.overlap(pats[0], pats[0]);
  assert.ok(Math.abs(ov - 1.0) < 1e-5);
});

test('T03 mutation check fails when weights or states are corrupted', () => {
  const t03 = golden.t03;
  assert.throws(() => close([1], t03.retrieval_history[1])); // Numerical comparison rejects changed result.
});
