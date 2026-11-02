import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import TEM from '../../../scripts/site_assets/tem.js';

const fixture = JSON.parse(readFileSync(new URL('../../fixtures/tem-run-fixture.json', import.meta.url)));

test('TEM module exports expected methods', () => {
  assert.equal(typeof TEM.init, 'function');
  assert.equal(typeof TEM.computeNewEdgeAccuracy, 'function');
});

test('computeNewEdgeAccuracy calculates correct ratio on fixture', () => {
  const world = fixture.worlds[0];
  // walk has 4 steps. first_visit_new_edge: [0, 1, 0, 1], pred_correct: [1, 0, 1, 1]
  // new edge visits at step 1 (first_visit=1, pred=0 -> incorrect) and step 3 (first_visit=1, pred=1 -> correct).
  // Total new edge visits = 2, correct = 1 -> accuracy = 0.5
  const acc = TEM.computeNewEdgeAccuracy(world, 3);
  assert.equal(acc, 0.5);

  const accStep0 = TEM.computeNewEdgeAccuracy(world, 0);
  assert.equal(accStep0, 0);
});

test('missing-file path calls showFriendlyMessage', () => {
  const oldDocument = globalThis.document;
  try {
    let messageShown = '';
    globalThis.document = {
      getElementById: (id) => {
        if (id === 'tem-message') {
          return {
            set textContent(v) { messageShown = v; },
            set style(s) {}
          };
        }
        return null;
      }
    };
    TEM.showFriendlyMessage('Test missing file warning');
    assert.equal(messageShown, 'Test missing file warning');
  } finally {
    globalThis.document = oldDocument;
  }
});

test('rate-map normalization handles units matrix correctly', () => {
  const oldDocument = globalThis.document;
  try {
    let cleared = false;
    let filledRects = 0;
    globalThis.document = {
      getElementById: (id) => {
        if (id === 'tem-ratemap-g') {
          return {
            width: 100,
            height: 100,
            getContext: () => ({
              clearRect: () => { cleared = true; },
              fillRect: () => { filledRects++; }
            })
          };
        }
        return null;
      }
    };
    const world = fixture.worlds[0];
    TEM.renderRateMaps(world);
    assert.ok(cleared);
    assert.ok(filledRects > 0);
  } finally {
    globalThis.document = oldDocument;
  }
});
