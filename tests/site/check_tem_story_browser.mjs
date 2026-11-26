import {spawn} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const directory=process.argv[2]||'/tmp/tem-story-browser';await mkdir(directory,{recursive:true});
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions','--remote-debugging-port=9338','--user-data-dir=/tmp/tem-story-chrome','about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));let ws,next=0;const pending=new Map(),errors=[],external=[],report=[];
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function snap(name){const r=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${directory}/${name}.png`,Buffer.from(r.data,'base64'));}
try{
 let targets;for(let i=0;i<50;i++){try{targets=await(await fetch('http://127.0.0.1:9338/json')).json();break;}catch{await delay(100);}}
 ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
 ws.addEventListener('message',({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push(m.params);else if(m.method==='Network.responseReceived'&&m.params.response.status>=400)errors.push(m.params.response);else if(m.method==='Network.requestWillBeSent'&&/^https?:/.test(m.params.request.url)&&!m.params.request.url.startsWith('http://127.0.0.1:8777/'))external.push(m.params.request.url);});
 await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
 for(const [label,width,height]of [['desktop',1440,900],['mobile',390,844]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:label==='mobile'});
  await send('Page.navigate',{url:'http://127.0.0.1:8777/story-tem/story.html'});
  for(let i=0;i<100;i++){if(await evaluate('document.documentElement.dataset.ready==="true"'))break;await delay(100);}
  assert.equal(await evaluate('document.documentElement.dataset.ready'),'true');await evaluate('document.fonts.ready');
  const count=await evaluate('storyEngine.panels.length');
  for(const theme of ['dark','light']){
   await evaluate(`document.documentElement.dataset.theme='${theme}';storyEngine.updateTheme()`);
   for(let i=0;i<count;i++){
    await evaluate(`storyEngine.setTarget(storyEngine.keys[${i}].t);storyEngine.setPlaying(false);storyEngine.setPlaying(true)`);
    await evaluate('new Promise(resolve=>{const check=()=>{if(Math.abs(storyEngine.t-storyEngine.target)<.000001)resolve();else requestAnimationFrame(check)};requestAnimationFrame(check)})');
    const words=await evaluate(`document.querySelector('#panel-copy').textContent.trim().split(/\\s+/).length`);
    assert.ok(words>=45&&words<=60,`Panel ${i}: ${words} words`);
    assert.ok(await evaluate('document.documentElement.scrollWidth<=innerWidth'),`Overflow ${label}, ${i}`);
    assert.ok(await evaluate(`(()=>{const p=document.querySelector('.floating').getBoundingClientRect(),n=document.querySelector('.navigation').getBoundingClientRect();return p.bottom<n.top&&p.left>=0&&p.right<=innerWidth})()`),`Panel/nav overlap ${label}, ${i}`);
    await snap(`${label}-${theme}-panel-${i}`);
   }
  }
  await evaluate('storyEngine.setTarget(0)');await delay(500);
  const before=await evaluate('document.querySelector("#scene").toDataURL()');await delay(300);
  assert.notEqual(await evaluate('document.querySelector("#scene").toDataURL()'),before,'Animation frozen');
  await evaluate('document.querySelector("#play").click()');await delay(300);
  const paused=await evaluate('storyEngine.drawCount');await delay(300);
  assert.equal(await evaluate('storyEngine.drawCount'),paused,'Paused renderer still running');
  await evaluate('document.querySelector("#play").click()');
  const performance=await evaluate(`new Promise(resolve=>{let frames=0,first=0,last=0;const renderStart=storyEngine.drawCount;const tick=now=>{if(!first)first=now;frames++;last=now;storyEngine.setTarget(.5+.16*Math.sin(now/700));if(now-first<2500)requestAnimationFrame(tick);else resolve({fps:(frames-1)*1000/(last-first),frames,elapsed:last-first,renderedFrames:storyEngine.drawCount-renderStart,renderFps:(storyEngine.drawCount-renderStart)*1000/(last-first)});};requestAnimationFrame(tick);})`);
  report.push({label,performance});assert.ok(performance.renderFps>=55,`${label} FPS ${performance.fps}`);
  await evaluate('document.activeElement.blur()');
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'End',code:'End'});await delay(1200);
  assert.equal(await evaluate('storyEngine.active'),count-1,'End navigation');
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Home',code:'Home'});await delay(1200);
  assert.equal(await evaluate('storyEngine.active'),0,'Home navigation');
  await evaluate('document.querySelector("#next").click()');await delay(1200);
  assert.equal(await evaluate('storyEngine.active'),1,'Button navigation');
  await evaluate('document.querySelector("#references-toggle").click()');
  assert.equal(await evaluate('document.querySelector("#references").open'),true);
  await send('Input.dispatchKeyEvent',{type:'rawKeyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});await delay(100);
  assert.equal(await evaluate('document.querySelector("#references").open'),false);
  const replay=await evaluate('storyEngine.panels.findIndex(p=>p.scene==="replay")');
  await evaluate(`storyEngine.setTarget(storyEngine.keys[${replay}].t)`);await delay(1200);
  assert.equal(await evaluate('document.querySelector("#replay-controls").hidden'),false);
  await evaluate('document.querySelector("#recording").value="1";document.querySelector("#recording").dispatchEvent(new Event("change"));document.querySelector("#recorded-world").value="2";document.querySelector("#recorded-world").dispatchEvent(new Event("change"));document.querySelector("#recorded-step").value="1";document.querySelector("#recorded-step").dispatchEvent(new Event("input"))');await delay(100);
  assert.equal(await evaluate('storyEngine.replayVariant'),1);assert.equal(await evaluate('storyEngine.replayWorld'),2);
  assert.equal(await evaluate('storyEngine.playing'),false);
  assert.equal(await evaluate('Math.round(storyEngine.phase*4)'),await evaluate('storyEngine.replays[1].worlds[2].walk.length-1'));
  await snap(`${label}-replay-controls`);
  assert.equal(await evaluate('document.querySelectorAll("canvas").length'),1);
  await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});await delay(300);
  assert.equal(await evaluate('document.querySelectorAll("#reduced-story section").length'),count);assert.equal(await evaluate('getComputedStyle(document.querySelector("#world")).display'),'none');
  assert.equal(await evaluate('document.querySelectorAll("#reduced-story footer a").length'),4);
  await evaluate('document.activeElement.blur()');
  await send('Input.dispatchKeyEvent',{type:'rawKeyDown',key:'End',code:'End',windowsVirtualKeyCode:35});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'End',code:'End',windowsVirtualKeyCode:35});await delay(300);
  assert.ok(await evaluate('scrollY>innerHeight'),'Reduced-motion keyboard scrolling blocked');
  const stillDraws=await evaluate('storyEngine.drawCount');await delay(200);assert.equal(await evaluate('storyEngine.drawCount'),stillDraws,'Reduced-motion renderer active');
  await evaluate('scrollTo(0,0)');await snap(`${label}-reduced-motion`);await send('Emulation.setEmulatedMedia',{features:[]});

 }
 assert.equal(errors.length,0,JSON.stringify(errors));assert.equal(external.length,0,JSON.stringify(external));await writeFile(`${directory}/report.json`,JSON.stringify({report,errors,external},null,2));console.log(JSON.stringify({directory,report,errors,external}));
}finally{ws?.close();chrome.kill();}
