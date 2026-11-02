import test from 'node:test';
import assert from 'node:assert/strict';
import { core } from './plate.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('template lif_derivative matches parity.json', () => {
  const parityPath = path.join(__dirname, 'parity.json');
  const parityData = JSON.parse(fs.readFileSync(parityPath, 'utf8'));

  const eq = parityData.find(e => e.js === 'core.lif_derivative');
  assert.ok(eq, 'parity entry found for core.lif_derivative');

  const [v, current] = [-70.0, 5.0];
  const result = core.lif_derivative(v, current, { v_rest: -70.0, resistance: 10.0 });
  assert.strictEqual(result, eq.results[0]);
});
