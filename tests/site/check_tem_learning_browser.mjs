import {spawn,execFileSync} from 'node:child_process';
import {mkdir,writeFile,rm,access,readFile} from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const directory=process.argv[2]||'/private/tmp/tem-native-review';await mkdir(directory,{recursive:true});
const root=new URL('../..',import.meta.url).pathname;
const siteDir=process.env.TEM_BROWSER_SITE_DIR;
assert.ok(siteDir, 'Set TEM_BROWSER_SITE_DIR to the built site directory');
const server=http.createServer((req,res)=>{
 const p=path.join(siteDir,req.url.split('?')[0]);
 fs.readFile(p,(err,data)=>{
  if(err){res.writeHead(404);res.end('Not found');return;}
  const map={'.html':'text/html','.css':'text/css','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.png':'image/png'};
  res.writeHead(200,{'Content-Type':map[path.extname(p)]||'text/plain'});res.end(data);
 });
});
await new Promise(r=>server.listen(8777,'127.0.0.1',r));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions','--remote-debugging-port=9348','--user-data-dir=/tmp/tem-native-chrome','about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));let ws,next=0;const pending=new Map(),errors=[],pngRequests=[];
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function snap(name){const r=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${directory}/${name}.png`,Buffer.from(r.data,'base64'));assertVisible(name);}
// Objective pixel guard: a screenshot is blank if every pixel is (near) the
// same colour, and two captures are duplicates if their pixels are identical.
// Runs PIL in a subprocess so the guard is independent of the page's DOM.
function pixelStats(file){
 const script=`
import sys
from PIL import Image
im=Image.open(sys.argv[1]).convert('RGB')
px=list(im.getdata())
n=len(px)
uniq=len(set(px))
r=sum(p[0] for p in px)//n;g=sum(p[1] for p in px)//n;b=sum(p[2] for p in px)//n
print(f'{uniq} {r} {g} {b}')
`;
 const out=execFileSync('python3',['-c',script,file],{encoding:'utf8'}).trim();
 const [uniq,avgR,avgG,avgB]=out.split(' ').map(Number);
 return {uniq,avg:[avgR,avgG,avgB]};
}
const shotStats=new Map();
function assertVisible(name){
 const s=pixelStats(`${directory}/${name}.png`);
 shotStats.set(name,s);
 assert.ok(s.uniq>50,`${name} looks blank: only ${s.uniq} distinct pixels, avg rgb ${s.avg}`);
 for(const [other,o]of shotStats){
  if(other!==name&&o.uniq===s.uniq&&o.avg.every((v,i)=>v===s.avg[i]))
   assert.fail(`${name} is pixel-identical to ${other}`);
 }
}
async function settle(){await evaluate('new Promise(resolve=>{const check=()=>{if(Math.abs(storyEngine.t-storyEngine.target)<.000001)resolve();else requestAnimationFrame(check)};requestAnimationFrame(check)})');}
// The chart is created while the engine plays, so its CSS entrance animation
// runs; a paused chart would freeze the marks at opacity 0. Wait until every
// mark has faded in, then assert the actual marks are visible (not just axes).
async function waitChartSettled(){
 for(let i=0;i<100;i++){
  const pending=await evaluate(`[...document.querySelectorAll('#native-figure-chart .nc-fade,#native-figure-chart .nc-grow,#native-figure-chart .nc-draw')].some(n=>parseFloat(getComputedStyle(n).opacity)<.99)`);
  if(!pending)return;
  await delay(100);
 }
 assert.fail('chart entrance animation did not finish');
}
async function assertMarksVisible(kind){
 const sel={line:'.nc-line, .nc-dot',bars:'.nc-bar',histograms:'.nc-bar',heatmaps:'.nc-cell'}[kind];
 const r=await evaluate(`(()=>{const els=[...document.querySelectorAll('#native-figure-chart ${sel}')];if(!els.length)return{count:0,visible:0};return{count:els.length,visible:els.filter(n=>parseFloat(getComputedStyle(n).opacity)>.99).length}})()`);
 assert.ok(r.count>0,`no ${kind} marks found`);
 assert.equal(r.visible,r.count,`${kind} marks not fully visible: ${r.visible}/${r.count}`);
}
try{
 let targets;for(let i=0;i<50;i++){try{targets=await(await fetch('http://127.0.0.1:9348/json')).json();break;}catch{await delay(100);}}
 ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
 ws.addEventListener('message',({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push(m.params);else if(m.method==='Network.responseReceived'&&m.params.response.status>=400)errors.push(m.params.response);else if(m.method==='Network.requestWillBeSent'&&/\.png($|\?)/i.test(m.params.request.url))pngRequests.push(m.params.request.url);});
 await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:'http://127.0.0.1:8777/story-tem/story.html'});
 for(let i=0;i<100;i++){if(await evaluate('document.documentElement.dataset.ready==="true"'))break;await delay(100);}
 assert.equal(await evaluate('document.documentElement.dataset.ready'),'true');await evaluate('document.fonts.ready');
 const count=await evaluate('storyEngine.panels.length');
 // Panel 9: STDP pointer lag sign (negative/zero/positive) and pause-phase stability.
 const stdp=await evaluate('storyEngine.panels.findIndex(p=>p.scene==="stdp-demo")');
 assert.equal(stdp,9);
 await evaluate(`storyEngine.setTarget(storyEngine.keys[${stdp}].t);storyEngine.setPlaying(false)`);await settle();
 for(const [name,frac]of [['neg',-0.5],['zero',0],['pos',0.5]]){
  const [x,y]=await evaluate(`(()=>{const e=storyEngine,key=e.keys[9],scale=10**((key.log-e.camera.log)*.24);const localX=${frac}*e.size*.36;return[Math.round(e.cx+(key.x-e.camera.x)*e.size+localX*scale),Math.round(e.cy-e.size*.15*scale)]})()`);
  await send('Input.dispatchMouseEvent',{type:'mouseMoved',x,y});
  const lag=await evaluate('storyEngine.stdpPointerLag');
  assert.ok(Number.isFinite(lag),`STDP lag ${name} not finite`);
  if(frac===0){assert.ok(Math.abs(lag)<.05,`STDP zero lag ${lag} not near zero`);}
  else{assert.ok(Math.abs(lag-frac)<.05,`STDP lag ${name} ${lag} != ${frac}`);assert.equal(Math.sign(lag),Math.sign(frac),`STDP lag ${name} sign ${Math.sign(lag)} != ${Math.sign(frac)}`);}
 }
 const phaseBefore=await evaluate('storyEngine.phase');await delay(400);
 assert.equal(await evaluate('storyEngine.phase'),phaseBefore,'STDP pause phase not stable');
 await snap('stdp-pointer-lag');
 // Panel 21: plasticity scene renders without error.
 const plasticity=await evaluate('storyEngine.panels.findIndex(p=>p.scene==="plasticity")');
 assert.equal(plasticity,21);
 await evaluate(`storyEngine.setTarget(storyEngine.keys[${plasticity}].t);storyEngine.setPlaying(true)`);await settle();await delay(300);
 assert.ok(await evaluate('storyEngine.drawCount')>0,'plasticity scene did not draw');
 await snap('plasticity-scene');
 // Native results: present mode, panels 11..17 render native SVG, no PNG.
 const nativeAvailable=await access(path.join(siteDir,'story-tem','paper-figures','native-data.json')).then(()=>true,()=>false);
 assert.ok(nativeAvailable, 'Native dataset must exist before verifying result charts');
 if(nativeAvailable){
  const nativeData=JSON.parse(await readFile(path.join(siteDir,'story-tem','paper-figures','native-data.json'),'utf8'));
  await evaluate('document.querySelector("#presenter-toggle").click()');await delay(100);
  assert.equal(await evaluate('document.querySelector("#presenter-toggle").getAttribute("aria-pressed")'),'true');
  const pngBefore=pngRequests.length;
  for(let i=11;i<=17;i++){
   // Render while the engine plays so the chart's CSS entrance animation runs;
   // a chart created while paused freezes its marks at opacity 0.
   await evaluate(`storyEngine.setPlaying(true);storyEngine.setTarget(storyEngine.keys[${i}].t)`);await settle();
   assert.equal(await evaluate('document.querySelector("#paper-figure").hidden'),false,`figure hidden at ${i}`);
   assert.equal(await evaluate('document.querySelector("#native-figure-chart svg")!==null'),true,`no native SVG at ${i}`);
   assert.equal(await evaluate('document.querySelector("#native-figure-unavailable").hidden'),true,`unavailable shown at ${i}`);
   const caption=await evaluate('document.querySelector("#paper-caption").textContent');
   assert.ok(caption.length>0,`no caption at ${i}`);
   if(i===11)assert.ok(!/lattice/i.test(caption),`panel 11 caption claims a lattice: ${caption}`);
   const figureShort={11:'Fig 3C',12:'Fig 4C',13:'Figs 5–6',14:'Table 2',15:'Fig 7',16:'Fig 8A',17:'Fig 13C'}[i];
   const sourceLabel=figureShort+(nativeData[i].approx?' · Digitized approximation':'');
   assert.equal(await evaluate('document.querySelector("#paper-source").textContent'),sourceLabel,`source label at ${i}`);
   assert.equal(await evaluate('document.querySelector("#paper-audit-link").hidden'),false,`no audit link at ${i}`);
   await waitChartSettled();
   await assertMarksVisible(nativeData[i].kind);
   if(nativeData[i].kind==='heatmaps'){
    const hm=await evaluate(`(()=>{const cells=[...document.querySelectorAll('#native-figure-chart .nc-cell')];if(!cells.length)return null;const r=cells[0].getBoundingClientRect();const maxB=Math.max(...cells.map(c=>c.getBoundingClientRect().bottom));const maxR=Math.max(...cells.map(c=>c.getBoundingClientRect().right));return{count:cells.length,w:r.width,h:r.height,maxBottom:maxB,maxRight:maxR}})()`);
    assert.ok(hm&&hm.count>0,`no heatmap cells at ${i}`);
    assert.ok(Math.abs(hm.w-hm.h)<1,`heatmap cells not square at ${i}: ${hm.w}x${hm.h}`);
    assert.ok(hm.maxBottom<900,`heatmap clipped below viewport at ${i}: ${hm.maxBottom}`);
    assert.ok(hm.maxRight<1440,`heatmap clipped right at ${i}: ${hm.maxRight}`);
   }
   if(nativeData[i].kind==='histograms'){
    const rot=await evaluate(`(()=>{const t=[...document.querySelectorAll('#native-figure-chart text')].find(n=>/rotate\\(-45\\)/.test(n.getAttribute('transform')||''));return t?t.getAttribute('transform'):null})()`);
    assert.ok(rot&&/rotate\(-45\)/.test(rot),`histogram bin labels not rotated at ${i}: ${rot}`);
   }
   if(/percent/i.test(nativeData[i].yLabel||'')){
    const minTick=await evaluate(`Math.min(...[...document.querySelectorAll('#native-figure-chart .nc-grid text')].map(t=>parseFloat(t.textContent)))`);
    assert.ok(minTick>=0,`percentage y-axis has negative tick at ${i}: ${minTick}`);
   }
   // Pause now that the entrance has finished, for a stable capture.
   await evaluate('storyEngine.setPlaying(false)');await delay(150);
   await snap(`native-panel-${i}`);
  }
  assert.equal(pngRequests.length,pngBefore,'PNG requested during native results');
  // H highlight forwards to the controller.
  await evaluate(`storyEngine.setTarget(storyEngine.keys[11].t);storyEngine.setPlaying(false)`);await settle();
  assert.equal(await evaluate('document.querySelector("#highlight-next").hidden'),false,'highlight button hidden');
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'h',code:'KeyH'});await delay(100);
  assert.equal(await evaluate('document.querySelector("#paper-highlight-text").hidden'),false,'highlight text not shown');
  await snap('native-highlight');
  // Cycle through every callout then one more to clear; count is data-driven, not assumed.
  for(let h=0;h<10 && !(await evaluate('document.querySelector("#paper-highlight-text").hidden'));h++){
   await send('Input.dispatchKeyEvent',{type:'keyDown',key:'h',code:'KeyH'});await delay(100);
  }
  assert.equal(await evaluate('document.querySelector("#paper-highlight-text").hidden'),true,'highlight not cleared');
  // Semantic highlight: the ring must land on the point the callout text
  // describes, not a generic index. Panel 11 callout 0 -> last point (24k).
  await evaluate(`storyEngine.setTarget(storyEngine.keys[11].t);storyEngine.setPlaying(false)`);await settle();
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'h',code:'KeyH'});await delay(100);
  const ring11=await evaluate(`(()=>{const r=document.querySelector('#native-figure-chart .nc-hl-ring');const d=[...document.querySelectorAll('#native-figure-chart .nc-dot')].pop();return r&&d?{ring:+r.getAttribute('cx'),dot:+d.getAttribute('cx')}:null})()`);
  assert.ok(ring11,`no highlight ring on panel 11`);
  assert.ok(Math.abs(ring11.ring-ring11.dot)<1,`panel 11 ring not on last point: ring ${ring11.ring} vs dot ${ring11.dot}`);
  // Panel 17 callout 0 -> MEC II precession bar (first bar of first series).
  await evaluate(`storyEngine.setTarget(storyEngine.keys[17].t);storyEngine.setPlaying(false)`);await settle();
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'h',code:'KeyH'});await delay(100);
  const hl17=await evaluate(`(()=>{const bars=[...document.querySelectorAll('#native-figure-chart .nc-bar')];const b=bars.find(x=>x.classList.contains('nc-hl'));const band=document.querySelector('#native-figure-chart .nc-hl-band');return b&&band?{bar:+b.getAttribute('x'),w:+b.getAttribute('width'),band:+band.getAttribute('x'),isFirst:b===bars[0]}:null})()`);
  assert.ok(hl17,`no highlighted bar/band on panel 17`);
  assert.ok(hl17.isFirst,`panel 17 highlight not on MEC II precession (first bar)`);
  assert.ok(Math.abs(hl17.bar+hl17.w/2-10-hl17.band)<2,`panel 17 band not over highlighted bar: bar ${hl17.bar} w ${hl17.w} band ${hl17.band}`);
  // Theme toggle updates without error.
  await evaluate('document.querySelector("#theme").click()');await delay(100);
  assert.equal(await evaluate('document.documentElement.dataset.theme'),'light');
  await evaluate('document.querySelector("#theme").click()');await delay(100);
  assert.equal(await evaluate('document.documentElement.dataset.theme'),'dark');
  // Pause/Animate drives the native controller.
  await evaluate(`storyEngine.setTarget(storyEngine.keys[11].t);storyEngine.setPlaying(true)`);await settle();
  await evaluate('document.querySelector("#play").click()');await delay(100);
  assert.equal(await evaluate('storyEngine.playing'),false,'engine not paused');
  assert.equal(await evaluate('document.querySelector("#native-figure-chart .nc-root").classList.contains("nc-paused")'),true,'chart not paused');
  await evaluate('document.querySelector("#play").click()');await delay(100);
  assert.equal(await evaluate('storyEngine.playing'),true,'engine not resumed');
  // Paused entry: a chart created while the engine is paused must rest fully
  // visible (no entrance animation to freeze at opacity 0).
  await evaluate(`storyEngine.setTarget(storyEngine.keys[11].t);storyEngine.setPlaying(true)`);await settle();
  await evaluate('storyEngine.setPlaying(false)');await delay(100);
  assert.equal(await evaluate('storyEngine.playing'),false,'engine not paused before paused-entry');
  await evaluate(`storyEngine.setTarget(storyEngine.keys[12].t)`);await settle();await delay(150);
  assert.equal(await evaluate('document.querySelector("#native-figure-chart svg")!==null'),true,'no chart on paused entry');
  const pausedMarks=await evaluate(`(()=>{const els=[...document.querySelectorAll('#native-figure-chart .nc-bar')];return els.length?{count:els.length,visible:els.filter(n=>parseFloat(getComputedStyle(n).opacity)>.99).length}:null})()`);
  assert.ok(pausedMarks&&pausedMarks.count>0,'no marks on paused-entry chart');
  assert.equal(pausedMarks.visible,pausedMarks.count,`paused-entry marks not visible: ${pausedMarks.visible}/${pausedMarks.count}`);
  await snap('native-paused-entry');
  // Figure toggle relabels Results/Explainer.
  assert.equal(await evaluate('document.querySelector("#figure-toggle").textContent'),'Explainer');
  await evaluate('document.querySelector("#figure-toggle").click()');await delay(100);
  assert.equal(await evaluate('document.querySelector("#paper-figure").hidden'),true,'figure not hidden on toggle');
  assert.equal(await evaluate('document.querySelector("#figure-toggle").textContent'),'Results');
  await evaluate('document.querySelector("#figure-toggle").click()');await delay(100);
  assert.equal(await evaluate('document.querySelector("#paper-figure").hidden'),false,'figure not shown on toggle');
  await evaluate('document.querySelector("#presenter-toggle").click()');await delay(100);
 }
 // Mobile: no overflow, no nav/panel overlap.
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await send('Page.navigate',{url:'http://127.0.0.1:8777/story-tem/story.html'});
 for(let i=0;i<100;i++){if(await evaluate('document.documentElement.dataset.ready==="true"'))break;await delay(100);}
 assert.equal(await evaluate('document.documentElement.dataset.ready'),'true');
 for(const i of [0,9,11,21]){
  await evaluate(`storyEngine.setTarget(storyEngine.keys[${i}].t);storyEngine.setPlaying(false)`);await settle();
  assert.ok(await evaluate('document.documentElement.scrollWidth<=innerWidth'),`mobile overflow at ${i}`);
  assert.ok(await evaluate(`(()=>{const p=document.querySelector('.floating').getBoundingClientRect(),n=document.querySelector('.navigation').getBoundingClientRect();return p.bottom<n.top&&p.left>=0&&p.right<=innerWidth})()`),`mobile panel/nav overlap at ${i}`);
 }
 // Capture panel 11 specifically: enter presenting mode, wait for the correct
 // active panel, and verify its title is actually visible before the shot.
 await evaluate('document.querySelector("#presenter-toggle").click()');await delay(100);
 await evaluate(`storyEngine.setTarget(storyEngine.keys[11].t);storyEngine.setPlaying(false)`);await settle();
 for(let i=0;i<50;i++){if(await evaluate('storyEngine.active===11'))break;await delay(100);}
 assert.equal(await evaluate('storyEngine.active'),11,'mobile capture not on panel 11');
 const mTitle=await evaluate(`(()=>{const t=document.querySelector('#panel-title');const r=t.getBoundingClientRect();return{txt:t.textContent.trim(),visible:r.width>0&&r.height>0&&r.top>=0&&r.bottom<=innerHeight}})()`);
 assert.ok(mTitle.visible,`panel 11 title not visible on mobile: ${JSON.stringify(mTitle)}`);
 assert.ok(mTitle.txt.length>0,'panel 11 title empty on mobile');
 assert.ok(await evaluate(`(()=>{const p=document.querySelector('.floating').getBoundingClientRect(),n=document.querySelector('.navigation').getBoundingClientRect();return p.bottom<n.top})()`),'mobile panel 11 covered by nav');
 await snap('mobile-panel-11');
 await evaluate('document.querySelector("#presenter-toggle").click()');await delay(100);
 // Reduced motion: world hidden, stills built, renderer idle.
 await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});await delay(300);
 assert.equal(await evaluate('document.querySelectorAll("#reduced-story section").length'),count);
 assert.equal(await evaluate('getComputedStyle(document.querySelector("#world")).display'),'none');
 const stillDraws=await evaluate('storyEngine.drawCount');await delay(200);
 assert.equal(await evaluate('storyEngine.drawCount'),stillDraws,'reduced-motion renderer active');
 await snap('reduced-motion');
 await send('Emulation.setEmulatedMedia',{features:[]});
 // The site is served from the built output: every asset must resolve, so
 // there are no 404s to ignore.
 assert.equal(errors.length,0,'built site must serve every asset (no 404s): '+JSON.stringify(errors));
 await writeFile(`${directory}/report.json`,JSON.stringify({nativeAvailable,pngRequests,errors},null,2));
 console.log(JSON.stringify({directory,nativeAvailable,pngRequests,errors}));
}finally{ws?.close();chrome.kill();server.close();await delay(500);await rm('/tmp/tem-native-chrome',{recursive:true,force:true}).catch(()=>{});}
