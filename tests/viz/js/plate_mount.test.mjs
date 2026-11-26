import test from 'node:test';
import assert from 'node:assert/strict';
import { loadPlate } from './fake_dom.mjs';

// Every plate must load, mount and survive a few frames in a fake DOM, so a
// free variable that was never declared (ReferenceError) fails here, not in
// the reader's browser.
const PLATES = ['N01', 'N02', 'N03', 'N04', 'N05', 'N07', 'N08', 'M01', 'M02', 'M03', 'T01', 'T02', 'T03', 'Y01', 'Y02', 'H01'];

for (const id of PLATES) {
  for (const [label, width] of [['desktop', 1200], ['mobile', 390]]) {
    test(`${id} mounts and runs frames without throwing (${label})`, () => {
      const h = loadPlate(id, { width });
      assert.doesNotThrow(() => h.mount());
      assert.doesNotThrow(() => h.tick(3));
    });
  }
}

test('M01 draws its hero canvas (render sets its layout, no ReferenceError on L)', () => {
  const h = loadPlate('M01');
  const { figure } = h.mount();
  const canvases = figure.querySelectorAll('canvas');
  assert.ok(canvases.length > 0, 'M01 created a canvas');
  assert.ok(canvases.some((c) => c._ctx && c._ctx.__calls.count > 50), 'M01 canvas received drawing calls');
});

test('N02 draws its potential well (ctx.colors() and state.p are defined)', () => {
  const h = loadPlate('N02');
  const { figure } = h.mount();
  const canvas = figure.querySelector('canvas');
  assert.ok(canvas && canvas._ctx && canvas._ctx.__calls.count > 50, 'N02 canvas received drawing calls');
  assert.doesNotThrow(() => h.tick(5));
});
