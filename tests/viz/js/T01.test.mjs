import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {core} from '../../../src/neurosush/viz/web/plates/T01.js';
const golden = JSON.parse(readFileSync(new URL('../../golden/tem_plate.json',import.meta.url)));
function close(actual, expected) {
  assert.equal(actual.length,expected.length);
  actual.forEach((v,i)=>Array.isArray(v)?close(v,expected[i]):assert.ok(Math.abs(v-expected[i])<=golden.atol,`${v} != ${expected[i]}`));
}
test('Hebbian update matches Python including clipping and unbounded weights',()=>{
  for(const c of golden.updates) close(core.hebbianUpdate(golden.memory,golden.pattern,golden.predicted,c.opts),c.expected);
});
test('attractor matches Python including signed weights, cap, masks and zero iterations',()=>{
  for(const c of golden.retrievals) close(core.attractor(golden.cue,golden.memory,c.iterations,c.opts),c.expected);
  assert.notDeepEqual(core.attractor(golden.cue,golden.memory,5),golden.cue);
  assert.throws(()=>close([1],[0])); // The numerical comparison rejects a changed result.
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
test('no wrapping, prediction precedes observation, eligibility uses directed edges',()=>{
  assert.deepEqual(core.neighbors(0),[4,1]);
  assert.deepEqual(core.neighbors(15),[11,14]);
  const w=core.createWorld(()=>.6);
  assert.throws(()=>core.step(w,{target:15}));
  assert.equal(core.step(w,{target:1}).eligible,false);
  assert.equal(w.trials,0);
  assert.equal(core.step(w,{target:0}).eligible,true);
  assert.equal(w.trials,1);
  assert.equal(core.step(w,{target:1}).eligible,false);
  assert.equal(core.step(w,{target:5}).eligible,false);
  assert.equal(core.step(w,{target:4}).eligible,false);
  assert.equal(core.step(w,{target:0}).eligible,true);
  const a=core.createWorld(()=>.1),b=core.createWorld(()=>.1);
  b.symbols[1]=3;
  assert.deepEqual(core.step(a,{target:1}).guesses,core.step(b,{target:1}).guesses);
});
test('memory ablation freezes M; a new world resets bindings and retains structure',()=>{
  const w=core.createWorld(()=>.6),before=JSON.stringify(w.m);
  core.step(w,{target:1,memory:false});
  assert.equal(JSON.stringify(w.m),before);
  const fresh=core.createWorld(()=>.1);
  assert.equal(fresh.trials,0); assert.equal(fresh.edges.size,0);
  assert.equal(fresh.seen.size,1); assert.notDeepEqual(fresh.m,w.m);
  assert.deepEqual(core.neighbors(fresh.pos),core.neighbors(0));
});
