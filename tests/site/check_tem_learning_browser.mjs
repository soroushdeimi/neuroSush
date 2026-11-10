import {spawn} from 'node:child_process';
import {mkdir,writeFile,rm,access} from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const directory=process.argv[2]||'/private/tmp/tem-native-review';await mkdir(directory,{recursive:true});
const root=new URL('../..',import.meta.url).pathname;
const siteDir=path.join(root,'scripts','site_assets');
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
async function snap(name){const r=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${directory}/${name}.png`,Buffer.from(r.data,'base64'));}
async function settle(){await evaluate('new Promise(resolve=>{const check=()=>{if(Math.abs(storyEngine.t-storyEngine.target)<.000001)resolve();else requestAnimationFrame(check)};requestAnimationFrame(check)})');}
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
 const nativeAvailable=await access(path.join(siteDir,'paper-figures','native-data.json')).then(()=>true,()=>false);
 if(nativeAvailable){
  await evaluate('document.querySelector("#presenter-toggle").click()');await delay(100);
  assert.equal(await evaluate('document.querySelector("#presenter-toggle").getAttribute("aria-pressed")'),'true');
  const pngBefore=pngRequests.length;
  for(let i=11;i<=17;i++){
   await evaluate(`storyEngine.setTarget(storyEngine.keys[${i}].t);storyEngine.setPlaying(false)`);await settle();await delay(150);
   assert.equal(await evaluate('document.querySelector("#paper-figure").hidden'),false,`figure hidden at ${i}`);
   assert.equal(await evaluate('document.querySelector("#native-figure-chart svg")!==null'),true,`no native SVG at ${i}`);
   assert.equal(await evaluate('document.querySelector("#native-figure-unavailable").hidden'),true,`unavailable shown at ${i}`);
   assert.equal(await evaluate('document.querySelector("#paper-caption").textContent.length>0'),true,`no caption at ${i}`);
   assert.equal(await evaluate('document.querySelector("#paper-source").textContent'),`Fig ${i-10}`);
   assert.equal(await evaluate('document.querySelector("#paper-audit-link").hidden'),false,`no audit link at ${i}`);
   await snap(`native-panel-${i}`);
  }
  assert.equal(pngRequests.length,pngBefore,'PNG requested during native results');
  // H highlight forwards to the controller.
  await evaluate(`storyEngine.setTarget(storyEngine.keys[11].t);storyEngine.setPlaying(false)`);await settle();
  assert.equal(await evaluate('document.querySelector("#highlight-next").hidden'),false,'highlight button hidden');
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'h',code:'KeyH'});await delay(100);
  assert.equal(await evaluate('document.querySelector("#paper-highlight-text").hidden'),false,'highlight text not shown');
  await snap('native-highlight');
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'h',code:'KeyH'});await delay(100);
  assert.equal(await evaluate('document.querySelector("#paper-highlight-text").hidden'),true,'highlight not cleared');
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
 await snap('mobile-panel-11');
 // Reduced motion: world hidden, stills built, renderer idle.
 await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});await delay(300);
 assert.equal(await evaluate('document.querySelectorAll("#reduced-story section").length'),count);
 assert.equal(await evaluate('getComputedStyle(document.querySelector("#world")).display'),'none');
 const stillDraws=await evaluate('storyEngine.drawCount');await delay(200);
 assert.equal(await evaluate('storyEngine.drawCount'),stillDraws,'reduced-motion renderer active');
 await snap('reduced-motion');
 await send('Emulation.setEmulatedMedia',{features:[]});
 // Peer files 404 until merged; tokens.css/favicon 404 are dev-server artifacts (source served unbuilt).
 const ignorePaths=['native-data.json','native-charts.mjs','native-charts.css','tokens.css','favicon.ico'];
 const realErrors=errors.filter(e=>!(e.url&&ignorePaths.some(p=>e.url.includes(p))));
 assert.equal(realErrors.length,0,JSON.stringify(realErrors));
 await writeFile(`${directory}/report.json`,JSON.stringify({nativeAvailable,pngRequests,errors,realErrors},null,2));
 console.log(JSON.stringify({directory,nativeAvailable,pngRequests,errors}));
}finally{ws?.close();chrome.kill();server.close();await delay(500);await rm('/tmp/tem-native-chrome',{recursive:true,force:true}).catch(()=>{});}
