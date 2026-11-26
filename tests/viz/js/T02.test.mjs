import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {core} from '../../../src/neurosush/viz/web/plates/T02.js';
const golden = JSON.parse(readFileSync(new URL('../../golden/tem_t02.json',import.meta.url)));
function close(actual, expected) {
  assert.equal(actual.length,expected.length);
  actual.forEach((v,i)=>Array.isArray(v)?close(v,expected[i]):assert.ok(Math.abs(v-expected[i])<=golden.atol,`${v} != ${expected[i]}`));
}
test('Hebbian update matches Python including clipping and unbounded weights',()=>{
  for(const c of golden.updates) close(core.hebbianUpdate(golden.memory,golden.pattern,golden.predicted,c.opts),c.expected);
});
test('attractor matches Python including signed weights, cap, masks and zero iterations',()=>{
  for(const c of golden.retrievals) close(core.attractor(golden.cue,golden.memory,c.iterations,c.opts),c.expected);
  assert.notDeepEqual(core.attractor(golden.cue,golden.memory,3),golden.cue);
  assert.throws(()=>close([1],[0])); // Rejects changed result.
});
test('shape and parameter errors fail early; inputs remain unchanged',()=>{
  const before=JSON.stringify(golden);
  core.hebbianUpdate(golden.memory,golden.pattern,golden.predicted);
  core.attractor(golden.cue,golden.memory);
  assert.equal(JSON.stringify(golden),before);
  for(const opts of [{decay:-1},{rate:-1},{clip:0}]) assert.throws(()=>core.hebbianUpdate(golden.memory,golden.pattern,golden.predicted,opts));
  assert.throws(()=>core.hebbianUpdate([], [1], []));
  assert.throws(()=>core.attractor([1],[],1));
  assert.throws(()=>core.attractor([1],[[1]],-1));
  assert.throws(()=>core.attractor([1],[[1]],1,{mask:[]}));
});
test('6x6 grid neighbors, world switching and remapping behaviour',()=>{
  assert.deepEqual(core.neighbors(0),[6,1]);
  assert.deepEqual(core.neighbors(35),[29,34]);
  const w = core.createWorld(()=>.5);
  assert.equal(w.worldType, 'A');
  core.switchWorld(w, 'B');
  assert.equal(w.worldType, 'B');
  assert.equal(w.pos, 0);
  assert.equal(w.trials, 0);
});

test('Learn section is present with four blocks',()=>{
  const plateSource = readFileSync(new URL('../../../src/neurosush/viz/web/plates/T02.js', import.meta.url), 'utf8');
  assert.ok(plateSource.includes('Learn: the math, the brain, the ML analogy'));
  assert.ok(plateSource.includes('What you see:'));
  assert.ok(plateSource.includes('The math:'));
  assert.ok(plateSource.includes('In the brain:'));
  assert.ok(plateSource.includes('Like classic ML:'));
});
