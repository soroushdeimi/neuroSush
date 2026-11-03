import {spawn} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const directory=process.argv[2]||'/tmp/story-browser';await mkdir(directory,{recursive:true});
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions','--remote-debugging-port=9337','--user-data-dir=/tmp/story-chrome','about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));let ws,next=0;const pending=new Map(),errors=[],external=[],report=[];
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function snap(name){const r=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${directory}/${name}.png`,Buffer.from(r.data,'base64'));}
try{
 let targets;for(let i=0;i<50;i++){try{targets=await(await fetch('http://127.0.0.1:9337/json')).json();break;}catch{await delay(100);}}
 ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
 ws.addEventListener('message',({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Network.requestWillBeSent'&&/^https?:/.test(m.params.request.url)&&!m.params.request.url.startsWith('http://127.0.0.1:8776/'))external.push(m.params.request.url);});
 await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
 for(const [label,width,height]of [['desktop',1440,900],['mobile',390,844]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:label==='mobile'});
  await send('Page.navigate',{url:'http://127.0.0.1:8776/story/story.html'});
  for(let i=0;i<100;i++){if(await evaluate('document.documentElement.dataset.ready==="true"'))break;await delay(100);}
  assert.equal(await evaluate('document.documentElement.dataset.ready'),'true');await evaluate('document.fonts.ready');
  for(let i=0;i<=10;i++){
   await evaluate(`storyEngine.setTarget(${i/10})`);await delay(100);
   const wordCount=await evaluate(`document.querySelector('#panel-copy').textContent.trim().split(/\\s+/).length`);
   assert.ok(wordCount>=25&&wordCount<=70,`Chapter ${i} word count ${wordCount} outside [25, 70]`);
   const panelBox=await evaluate(`document.querySelector('.floating').getBoundingClientRect().toJSON()`);
   if(width>=701)assert.ok(panelBox.right<width*0.6,`Panel box right ${panelBox.right} intersects safe area on desktop`);
  }
  for(const theme of ['dark','light']){
   await evaluate(`document.documentElement.dataset.theme='${theme}';storyEngine.updateTheme()`);
   for(let i=0;i<=10;i++){
    await evaluate(`storyEngine.setTarget(${i/10})`);await delay(150);
    await snap(`${label}-${theme}-chapter-${i}`);
   }
   for(const t of [0,.1,.15,.2,.275,.35,.42,.46,.5,.575,.65,.72,.8,.86,1]){
    await evaluate(`storyEngine.setTarget(${t})`);await evaluate('new Promise(resolve=>{let count=0;const check=()=>{if(Math.abs(storyEngine.t-storyEngine.target)<.000001||count++>300)resolve();else requestAnimationFrame(check)};requestAnimationFrame(check)})');
    assert.ok(await evaluate('document.documentElement.scrollWidth<=innerWidth'),`overflow ${label} t=${t}`);
    const actual=await evaluate('({t:storyEngine.t,log:storyEngine.camera.log,active:storyEngine.active})');
    assert.ok(Math.abs(actual.t-t)<.00001,JSON.stringify({target:t,...actual}));await snap(`${label}-${theme}-t-${t}`);report.push({label,theme,t,...actual});
   }
   for(const [time,id]of [[.2,'current'],[.42,'sodium'],[.65,'stimulus'],[.86,'stdp-tau']]){
    await evaluate(`storyEngine.setTarget(${time})`);await evaluate('new Promise(resolve=>{let count=0;const check=()=>{if(Math.abs(storyEngine.t-storyEngine.target)<.000001||count++>300)resolve();else requestAnimationFrame(check)};requestAnimationFrame(check)})');
    await evaluate('document.querySelector("#play").click()');
    assert.equal(await evaluate('storyEngine.playing'),true);const frozen=await evaluate('storyEngine.t');
    const before=await evaluate('document.querySelector("#scene").toDataURL()');await snap(`${label}-${theme}-live-${id}-0`);await delay(700);const after=await evaluate('document.querySelector("#scene").toDataURL()');await snap(`${label}-${theme}-live-${id}-1`);assert.ok(before!==after,`Frozen live ${id}`);
    await evaluate(`(()=>{const slider=document.querySelector('#${id}');slider.value=Number(slider.value)+Number(slider.step);slider.dispatchEvent(new Event('input'));})()`);
    assert.equal(await evaluate('storyEngine.t'),frozen);await evaluate('document.querySelector("#play").click()');
   }
  }
  const performance=await evaluate(`new Promise(resolve=>{let frames=0,first=0,last=0;const renderStart=storyEngine.drawCount;const tick=now=>{if(!first)first=now;frames++;last=now;storyEngine.setTarget(.5+.16*Math.sin(now/700));if(now-first<2500)requestAnimationFrame(tick);else resolve({fps:(frames-1)*1000/(last-first),frames,elapsed:last-first,renderedFrames:storyEngine.drawCount-renderStart,renderFps:(storyEngine.drawCount-renderStart)*1000/(last-first)});};requestAnimationFrame(tick);})`);
  report.push({label,performance});assert.ok(performance.renderFps>=55,`${label} FPS ${performance.fps}`);
  assert.equal(await evaluate('document.querySelectorAll("canvas").length'),1);
  await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});await delay(300);
  assert.equal(await evaluate('document.querySelectorAll("#reduced-story section").length'),11);assert.equal(await evaluate('getComputedStyle(document.querySelector("#world")).display'),'none');
  await snap(`${label}-reduced-motion`);await send('Emulation.setEmulatedMedia',{features:[]});

 }
 assert.equal(errors.length,0,JSON.stringify(errors));assert.equal(external.length,0,JSON.stringify(external));await writeFile(`${directory}/report.json`,JSON.stringify({report,errors,external},null,2));console.log(JSON.stringify({directory,report,errors,external}));
}finally{ws?.close();chrome.kill();}
