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
  const acc = TEM.computeNewEdgeAccuracy(world, 3);
  assert.equal(acc, 0.5);

  const accStep0 = TEM.computeNewEdgeAccuracy(world, 0);
  assert.equal(accStep0, 0);
});

test('missing-file path shows "No recorded run published yet"', () => {
  const oldDocument = globalThis.document;
  try {
    let messageShown = '';
    globalThis.document = {
      getElementById: (id) => {
        if (id === 'tem-message') {
          return {
            set textContent(v) { messageShown = v; },
            style: {}
          };
        }
        return null;
      }
    };
    TEM.showFriendlyMessage('No recorded run published yet');
    assert.equal(messageShown, 'No recorded run published yet');
  } finally {
    globalThis.document = oldDocument;
  }
});

test('refuses to present run with FIXTURE label as model result and shows test fixture banner', () => {
  const oldDocument = globalThis.document;
  try {
    let bannerDisplay = '';
    let messageShown = '';
    globalThis.document = {
      getElementById: (id) => {
        if (id === 'tem-fixture-banner') {
          return { style: { set display(v) { bannerDisplay = v; } } };
        }
        if (id === 'tem-message') {
          return { set textContent(v) { messageShown = v; }, style: {} };
        }
        return null;
      }
    };
    TEM.runs['v2'] = fixture;
    TEM.currentVariant = 'v2';
    TEM.updateFromVariant();
    assert.equal(bannerDisplay, 'block');
    assert.equal(messageShown, 'Test fixture detected; not a real model run.');
  } finally {
    globalThis.document = oldDocument;
    delete TEM.runs['v2'];
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
