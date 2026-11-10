import test from 'node:test';
import assert from 'node:assert/strict';
import {clamp, linearScale, niceTicks, formatValue, errorBarEnds, validateEntry} from '../../scripts/site_assets/story-tem/native-charts.mjs';

test('clamp keeps values inside the domain and rejects non-finite input', () => {
  assert.equal(clamp(5, 0, 10), 5);
  assert.equal(clamp(-3, 0, 10), 0);
  assert.equal(clamp(99, 0, 10), 10);
  for (const bad of [NaN, Infinity, -Infinity]) assert.throws(() => clamp(bad, 0, 1), RangeError);
});

test('linearScale maps endpoints exactly and inverts direction', () => {
  const s = linearScale(0, 10, 100, 0);
  assert.equal(s(0), 100);
  assert.equal(s(10), 0);
  assert.equal(s(5), 50);
  for (const bad of [NaN, Infinity]) assert.throws(() => linearScale(bad, 1, 0, 1), RangeError);
});

test('linearScale degenerates to the range midpoint on an all-zero domain', () => {
  assert.equal(linearScale(0, 0, 40, 80)(0), 60);
});

test('niceTicks stays inside the domain and never invents endpoints', () => {
  const t = niceTicks(0, 1, 5);
  assert.ok(t.length >= 2);
  assert.ok(t.every(v => v >= 0 && v <= 1));
  assert.equal(t[0], 0);
  assert.equal(t[t.length - 1], 1);
  assert.deepEqual(niceTicks(0, 0), [0]);
  assert.throws(() => niceTicks(NaN, 1), RangeError);
});

test('formatValue keeps units attached and handles zero and tiny values', () => {
  assert.equal(formatValue(0), '0');
  assert.equal(formatValue(0, 'Hz'), '0 Hz');
  assert.equal(formatValue(1.5, 'ms'), '1.5 ms');
  assert.equal(formatValue(0.001), '0.001');
  assert.throws(() => formatValue(NaN), RangeError);
});

test('errorBarEnds uses sd or low/high and clips to the domain', () => {
  assert.deepEqual(errorBarEnds(10, {sd: 2}, 0, 20), {top: 12, bottom: 8});
  assert.deepEqual(errorBarEnds(10, {low: 4, high: 16}, 0, 20), {top: 16, bottom: 4});
  // Negative lower errorbar below the domain floor is clipped, not drawn outside.
  assert.deepEqual(errorBarEnds(1, {sd: 4}, 0, 10), {top: 5, bottom: 0});
  assert.deepEqual(errorBarEnds(9, {sd: 4}, 0, 10), {top: 10, bottom: 5});
  assert.deepEqual(errorBarEnds(0, {}, 0, 10), {top: 0, bottom: 0});
  assert.throws(() => errorBarEnds(NaN, {}, 0, 1), RangeError);
});

test('errorBarEnds accepts a high-only bound without inventing a lower bound', () => {
  // high alone: cap above the point, lower bound stays at the point itself.
  assert.deepEqual(errorBarEnds(15.7, {high: 33.3}, 0, 100), {top: 33.3, bottom: 15.7});
  // low alone: floor below the point, upper bound stays at the point itself.
  assert.deepEqual(errorBarEnds(15.7, {low: 2.0}, 0, 100), {top: 15.7, bottom: 2.0});
  // A high-only bound that clips at the domain ceiling still keeps the point as the floor.
  assert.deepEqual(errorBarEnds(90, {high: 120}, 0, 100), {top: 100, bottom: 90});
  // A low-only bound that clips at the domain floor still keeps the point as the ceiling.
  assert.deepEqual(errorBarEnds(10, {low: -5}, 0, 100), {top: 10, bottom: 0});
});

test('errorBarEnds returns a zero-length bar when no bound is present', () => {
  assert.deepEqual(errorBarEnds(42, {}, 0, 100), {top: 42, bottom: 42});
  assert.deepEqual(errorBarEnds(42, {sd: 0}, 0, 100), {top: 42, bottom: 42});
});

test('validateEntry accepts a minimal line entry and returns it unchanged', () => {
  const entry = {
    kind: 'line', title: 't', source: 's', provenance: 'p', xLabel: 'x', yLabel: 'y',
    series: [{name: 'a', color: 'accent', points: [{x: 0, y: 1}, {x: 1, y: 2, sd: .5}]}],
  };
  assert.equal(validateEntry(entry), entry);
});

test('validateEntry rejects unknown kinds and malformed data clearly', () => {
  assert.throws(() => validateEntry({kind: 'scatter', title: 't'}), /unknown kind/);
  assert.throws(() => validateEntry({kind: 'line', title: 't', series: []}), /series/);
  assert.throws(() => validateEntry({kind: 'line', title: 't', series: [{name: 'a', color: 'red', points: [{x: 0, y: 1}]}]}), /color/);
  assert.throws(() => validateEntry({kind: 'line', title: 't', series: [{name: 'a', color: 'blue', points: [{x: 0, y: NaN}]}]}), /point\.y/);
  assert.throws(() => validateEntry({kind: 'line', title: 't', series: [{name: 'a', color: 'blue', points: [{x: 0, y: 1, sd: Infinity}]}]}), /point\.sd/);
  assert.throws(() => validateEntry(null), /object/);
});

test('validateEntry rejects heatmap values outside the normalized 0..1 range', () => {
  const base = {kind: 'heatmaps', title: 't', groups: [{name: 'g', maps: [{label: 'm', values: [[0, .5], [1, 0]]}]}]};
  assert.equal(validateEntry(base), base);
  const over = structuredClone(base); over.groups[0].maps[0].values[0][1] = 1.2;
  assert.throws(() => validateEntry(over), /0\.\.1/);
  const neg = structuredClone(base); neg.groups[0].maps[0].values[1][1] = -0.1;
  assert.throws(() => validateEntry(neg), /0\.\.1/);
  const ragged = structuredClone(base); ragged.groups[0].maps[0].values[1].pop();
  assert.throws(() => validateEntry(ragged), /width/);
  assert.throws(() => validateEntry({kind: 'heatmaps', title: 't', groups: []}), /groups/);
});
