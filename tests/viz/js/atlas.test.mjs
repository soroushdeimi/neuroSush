/**
 * Unit tests for Atlas client library.
 * Run with: node --test tests/viz/js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import Atlas from '../../../src/neurosush/viz/web/atlas.js';

function toBase64(typedArray) {
  const bytes = new Uint8Array(
    typedArray.buffer,
    typedArray.byteOffset,
    typedArray.byteLength
  );
  return Buffer.from(bytes).toString('base64');
}

test('unpack: float32, uint8, int32 and shape', () => {
  // float32
  const f32Original = new Float32Array([1.5, -2.25, 100.125]);
  const f32Packed = {
    dtype: 'float32',
    shape: [3],
    data: toBase64(f32Original)
  };
  const f32Unpacked = Atlas.unpack(f32Packed);
  assert.ok(f32Unpacked instanceof Float32Array);
  assert.strictEqual(f32Unpacked.length, 3);
  assert.strictEqual(f32Unpacked[0], 1.5);
  assert.strictEqual(f32Unpacked[1], -2.25);
  assert.strictEqual(f32Unpacked[2], 100.125);

  // uint8
  const u8Original = new Uint8Array([0, 1, 127, 255]);
  const u8Packed = {
    dtype: 'uint8',
    shape: [4],
    data: toBase64(u8Original)
  };
  const u8Unpacked = Atlas.unpack(u8Packed);
  assert.ok(u8Unpacked instanceof Uint8Array);
  assert.strictEqual(u8Unpacked.length, 4);
  assert.strictEqual(u8Unpacked[0], 0);
  assert.strictEqual(u8Unpacked[3], 255);

  // int32
  const i32Original = new Int32Array([-1000, 0, 42, 999999]);
  const i32Packed = {
    dtype: 'int32',
    shape: [4],
    data: toBase64(i32Original)
  };
  const i32Unpacked = Atlas.unpack(i32Packed);
  assert.ok(i32Unpacked instanceof Int32Array);
  assert.strictEqual(i32Unpacked.length, 4);
  assert.strictEqual(i32Unpacked[0], -1000);
  assert.strictEqual(i32Unpacked[3], 999999);
});

test('unpack: empty arrays for all dtypes', () => {
  for (const dtype of ['float32', 'uint8', 'int32']) {
    const unpacked = Atlas.unpack({ dtype, shape: [0], data: '' });
    assert.strictEqual(unpacked.length, 0);
  }
});

test('unpack: unsupported dtype throws descriptive error', () => {
  assert.throws(
    () => Atlas.unpack({ dtype: 'float64', data: '' }),
    /unsupported dtype: float64/
  );
  assert.throws(
    () => Atlas.unpack({ dtype: 'int64', data: '' }),
    /unsupported dtype: int64/
  );
  assert.throws(
    () => Atlas.unpack(null),
    /packed must be an object/
  );
});

test('Trace: steps, series, slice view, and events', () => {
  const vData = new Float32Array([-65.0, -60.0, -55.0, -50.0, -65.0]);
  const spkData = new Uint8Array([0, 0, 0, 1, 0]);

  const trace = new Atlas.Trace({
    steps: 5,
    meta: { threshold: -50.0 },
    series: {
      v: { dtype: 'float32', shape: [5], data: toBase64(vData) },
      spikes: { dtype: 'uint8', shape: [5], data: toBase64(spkData) }
    }
  });

  assert.strictEqual(trace.steps, 5);
  assert.strictEqual(trace.meta.threshold, -50.0);
  assert.deepStrictEqual(trace.events, [3]);

  // Test .at(name, i) slice view
  const slice = trace.at('v', 1);
  assert.ok(slice instanceof Float32Array);
  assert.strictEqual(slice.length, 1);
  assert.strictEqual(slice[0], -60.0);

  // Modifying slice view modifies underlying series buffer
  slice[0] = -62.5;
  assert.strictEqual(trace.series.v.data[1], -62.5);

  assert.throws(() => trace.at('unknown', 0), /unknown series: unknown/);
  assert.throws(() => trace.at('v', 10), /out of bounds/);
});

test('Clock: tick advances exactly speed*dt steps', () => {
  const clock = new Atlas.Clock({
    steps: 100,
    speed: 1,
    stepsPerSecond: 10,
    loop: false
  });
  clock.play();

  // 100ms at 10 steps/sec = 1 step
  clock.tick(100);
  assert.strictEqual(clock.t, 1);

  // 500ms at 10 steps/sec = 5 steps
  clock.tick(500);
  assert.strictEqual(clock.t, 6);

  // Change speed to 2x: 200ms at 10 sps * 2 = 4 steps
  clock.setSpeed(2);
  clock.tick(200);
  assert.strictEqual(clock.t, 10);
});

test('Clock: looping behavior', () => {
  const clock = new Atlas.Clock({
    steps: 20,
    speed: 1,
    stepsPerSecond: 10,
    loop: true
  });
  clock.play();
  clock.seek(18);

  // 400ms = 4 steps -> 18 + 4 = 22 -> loops to 2
  clock.tick(400);
  assert.strictEqual(clock.t, 2);
});

test('Clock: clamping behavior and pausing at end', () => {
  const clock = new Atlas.Clock({
    steps: 20,
    speed: 1,
    stepsPerSecond: 10,
    loop: false
  });
  clock.play();
  clock.seek(18);

  // 500ms = 5 steps -> 18 + 5 = 23 -> clamps to 19 and pauses
  clock.tick(500);
  assert.strictEqual(clock.t, 19);
  assert.strictEqual(clock.isPlaying, false);
});

test('Clock: jumpEvent lands exactly on event steps', () => {
  const clock = new Atlas.Clock({
    steps: 50,
    events: [5, 15, 30]
  });

  assert.strictEqual(clock.t, 0);

  // Jump forward
  clock.jumpEvent(1);
  assert.strictEqual(clock.t, 5);

  clock.jumpEvent(1);
  assert.strictEqual(clock.t, 15);

  // Jump backward
  clock.seek(22);
  clock.jumpEvent(-1);
  assert.strictEqual(clock.t, 15);

  clock.jumpEvent(-1);
  assert.strictEqual(clock.t, 5);

  // Wrap or clamp when at bounds
  clock.seek(35);
  clock.jumpEvent(1); // loop wraps to first event
  assert.strictEqual(clock.t, 5);
});

test('Clock: seek clamps to valid bounds', () => {
  const clock = new Atlas.Clock({ steps: 20 });

  clock.seek(-10);
  assert.strictEqual(clock.t, 0);

  clock.seek(100);
  assert.strictEqual(clock.t, 19);

  clock.seek(7.8);
  assert.strictEqual(clock.t, 8);
});

test('niceTicks: produces expected intervals and covers range', () => {
  const t1 = Atlas.niceTicks(0, 10, 5);
  assert.deepStrictEqual(t1, [0, 2, 4, 6, 8, 10]);

  const t2 = Atlas.niceTicks(0, 100, 5);
  assert.deepStrictEqual(t2, [0, 20, 40, 60, 80, 100]);

  const t3 = Atlas.niceTicks(5, 5, 5);
  assert.deepStrictEqual(t3, [5]);

  const t4 = Atlas.niceTicks(10, 0, 5);
  assert.deepStrictEqual(t4, [0, 2, 4, 6, 8, 10]);
});

test('strings: single-table lookup (missing key returns the key, vars substitution)', () => {
  Atlas.setStrings({
    greeting: 'Hello',
    farewell: 'Goodbye',
    welcome: 'Welcome, {name}!'
  });

  assert.strictEqual(Atlas.t('greeting'), 'Hello');
  assert.strictEqual(Atlas.t('farewell'), 'Goodbye');
  assert.strictEqual(Atlas.t('welcome', { name: 'Soroush' }), 'Welcome, Soroush!');
  assert.strictEqual(Atlas.t('non_existent_key'), 'non_existent_key');
});

test('formatNumber: handles decimals, negatives, NaN, and zeros', () => {
  assert.strictEqual(Atlas.formatNumber(1.23456, 2), '1.23');
  assert.strictEqual(Atlas.formatNumber(42, 1), '42.0');
  assert.strictEqual(Atlas.formatNumber(-0.00001, 2), '0.00');
  assert.strictEqual(Atlas.formatNumber(-65.123, 2), '-65.12');
  assert.strictEqual(Atlas.formatNumber(NaN, 2), '-');
  assert.strictEqual(Atlas.formatNumber(null, 2), '-');
  assert.strictEqual(Atlas.formatNumber(undefined, 2), '-');
});
