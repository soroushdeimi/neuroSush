import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {cameraAt,fieldAt,validateReplay} from '../../scripts/site_assets/story-tem/engine.mjs';
const keys=[{t:0,x:0,y:1,log:0},{t:1,x:1,y:0,log:-1}];
test('TEM camera clamps endpoints and preserves its output object',()=>{
 const out={};assert.equal(cameraAt(.5,keys,out),out);assert.equal(out.x,.5);assert.equal(out.log,-.5);
 assert.equal(cameraAt(-1,keys,out).x,0);assert.equal(cameraAt(2,keys,out).x,1);
});
test('TEM camera rejects nonfinite scroll and supports a single opening panel',()=>{
 for(const t of [NaN,Infinity,-Infinity])assert.throws(()=>cameraAt(t,keys,{}),RangeError);
 assert.equal(cameraAt(0,[keys[0]],{}).x,0);
});

test('Teaching fields distinguish one local peak from repeated peaks',()=>{
 assert.equal(fieldAt(.48,.52),1);
 assert.ok(fieldAt(0,0)<.001);
 assert.equal(fieldAt(0,0,true),1);
 assert.equal(fieldAt(.34,0,true),1);
 assert.ok(fieldAt(.17,0,true)<.01);
 for(const bad of [NaN,Infinity,-Infinity])assert.throws(()=>fieldAt(bad,0),RangeError);
});

test('Published replays have bounded paths and finite signed activity maps',()=>{
 for(const variant of ['v1','v2']){
  const run=JSON.parse(readFileSync(new URL(`../../scripts/site_assets/data/tem-run-${variant}-s0.json`,import.meta.url)));
  assert.equal(validateReplay(run),run);
  const broken=structuredClone(run);broken.worlds[0].walk[0][0]=broken.worlds[0].size;
  assert.throws(()=>validateReplay(broken),/position/);
  const truncated=structuredClone(run);truncated.worlds[0].rate_maps.g[0].pop();
  assert.throws(()=>validateReplay(truncated),/activity/);
 }
 assert.throws(()=>validateReplay({worlds:[]}),/worlds/);
});
