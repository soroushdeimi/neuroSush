import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {lifRate,lifStep,nernst,stdp,gateRates,gates,hhRest,hhTrace,hhStep,ionBudget,weightedRest,conductanceStep} from '../../scripts/site_assets/story/math.mjs';
const ledger=JSON.parse(await readFile(new URL('../../scripts/site_assets/story/facts.json',import.meta.url)));
const f=Object.fromEntries(ledger.facts.filter(x=>x.status==='VERIFIED').map(x=>[x.id,x.value]));
const p=prefix=>Object.fromEntries(Object.entries(f).filter(([k])=>k.startsWith(prefix+'_')).map(([k,v])=>[k.slice(prefix.length+1),v]));
const near=(a,b,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol,`${a} differs from ${b}`);
test('browser derived values match the fact ledger',()=>{
 const values={thermal:nernst(1,Math.E,1,f.temperature,f),lif_rheobase:(f.lif_threshold-f.lif_rest)/f.lif_resistance,lif_rate:lifRate(f.lif_input,p('lif')),weighted_rest:weightedRest(f),hh_rest:hhRest(p('hh')),hh_detection:p('hh').v_spike,stdp_positive:stdp(f.stdp_lag,p('stdp')),stdp_negative:stdp(-f.stdp_lag,p('stdp'))};
 const response=hhTrace(p('hh'));values.hh_charge_density=response.at(-1).charge;values.hh_ions_entered=values.hh_charge_density*1e-9*f.budget_area*1e4/f.elementary_charge;
 for(const ion of ['k','na']){values[ion+'_ratio']=f[ion+'_out']/f[ion+'_in'];values['e_'+ion]=nernst(f[ion+'_in'],f[ion+'_out'],1,f.temperature,f);}
 for(const [k,v]of Object.entries(ionBudget(f)))values['budget_'+k]=v;
 for(const [k,v]of Object.entries(gates(values.hh_rest)))values['hh_rest_'+k]=v;
 const derived=ledger.facts.filter(x=>x.formula);assert.deepEqual(Object.keys(values).sort(),derived.map(x=>x.id).sort());
 for(const row of derived)near(values[row.id],row.value,row.tolerance);
});
test('LIF rheobase has no finite-time spike and zero input stays at rest',()=>{near(lifRate(f.lif_rheobase,p('lif')),0);near(lifStep(f.lif_rest,0,p('lif')),f.lif_rest);assert.ok(lifRate(f.lif_rheobase+f.lif_dt,p('lif'))>0);});
test('Nernst handles reversal, divalent ions, zero Celsius and invalid inputs',()=>{near(nernst(5,5,1,0,f),0);near(nernst(f.k_in,f.k_out,-1,f.temperature,f),-f.e_k);near(nernst(f.k_in,f.k_out,2,f.temperature,f),f.e_k/2);assert.notEqual(nernst(f.k_in,f.k_out,1,0,f),f.e_k);for(const args of [[0,1,1,37],[1,-1,1,37],[1,2,0,37],[1,2,1,-f.kelvin_offset],[NaN,2,1,37]])assert.throws(()=>nernst(...args,f),RangeError);});
test('HH remains finite at rate singularities and gates remain probabilities',()=>{near(gateRates(-40).m[0],1);near(gateRates(-55).n[0],.1);const trace=hhTrace(p('hh'));for(const row of trace){assert.ok(Number.isFinite(row.v));for(const k of ['m','h','n'])assert.ok(row[k]>=0&&row[k]<=1);}assert.ok(trace.some(x=>x.v>f.hh_detection));assert.throws(()=>hhStep(trace[0],0,{...p('hh'),dt:1}),RangeError);});
test('STDP update ordering and non-integer or unstable steps',()=>{near(stdp(0,p('stdp')),f.stdp_a_plus-f.stdp_a_minus);assert.throws(()=>stdp(.5,p('stdp')),RangeError);assert.throws(()=>stdp(1,{...p('stdp'),dt:21}),RangeError);assert.ok(stdp(1,p('stdp'))>0&&stdp(-1,p('stdp'))<0);});
test('conductance balance stays between voltage and reversal',()=>{const q={rest:f.lif_rest,resistance:f.lif_resistance,tau:f.lif_tau,dt:1,e_exc:f.hh_v_spike,e_inh:f.lif_reset};for(const g of [0,1,1e6]){const v=conductanceStep(q.rest,g,0,q);assert.ok(v>=q.rest&&v<=q.e_exc);}});
import {cameraAt,bandOpacity} from '../../scripts/site_assets/story/engine.mjs';
import {makePanels} from '../../scripts/site_assets/story/panels.mjs';
test('camera keyframes and crossfades stay continuous and bounded',()=>{const keys=ledger.camera.map(row=>({t:f[row.t],log:f[row.log],x:f[row.x],y:f[row.y]})),out={};for(const key of keys){cameraAt(key.t,keys,out);near(out.log,key.log);near(out.x,key.x);near(out.y,key.y);}near(cameraAt(-1,keys,out).log,keys[0].log);near(cameraAt(2,keys,out).log,keys.at(-1).log);assert.throws(()=>cameraAt(NaN,keys,out),RangeError);for(let i=0;i<keys.length-1;i++){const mid=(keys[i].t+keys[i+1].t)/2;near(cameraAt(mid,keys,out).log,(keys[i].log+keys[i+1].log)/2);}near(bandOpacity(-5,-6,-4),1);near(bandOpacity(-6.4,-6,-4),0);near(bandOpacity(-6,-6,-4),.5);for(let x=-10;x<0;x+=.01){const opacity=bandOpacity(x,-6,-4);assert.ok(opacity>=0&&opacity<=1);}});
test('panels stay concise and every displayed fact ID exists',()=>{for(const panel of makePanels(f)){const text=(panel.title+' '+panel.copy+' '+panel.note).replace(/<[^>]+>/g,' ');assert.ok(text.trim().split(/\s+/).length<=70,`${text.trim().split(/\s+/).length}: ${panel.title}`);for(const match of panel.copy.matchAll(/data-fact-id="([^"]+)"/g))assert.ok(Object.hasOwn(f,match[1]),match[1]);assert.equal((panel.copy.match(/class="surprise"/g)||[]).length,1);}});
