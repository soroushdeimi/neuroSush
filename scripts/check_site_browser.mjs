import {spawn} from 'node:child_process';
import {writeFile,mkdir} from 'node:fs/promises';
// Run after serving a built site on 127.0.0.1:8765. No npm packages required.
const shots=process.argv[2];
if(!shots) throw Error('Usage: node scripts/check_site_browser.mjs SCREENSHOT_DIRECTORY');
await mkdir(shots,{recursive:true});
const chrome=spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--enable-unsafe-swiftshader','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions','--metrics-recording-only','--disable-features=MediaRouter,OptimizationHints','--remote-debugging-port=9229','--user-data-dir=/tmp/neurosush-chrome-check','about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
let ws;
try{
let targets;
for(let i=0;i<50;i++){try{targets=await (await fetch('http://127.0.0.1:9229/json')).json();break;}catch{await delay(100);}}
ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let next=0;const pending=new Map(),errors=[],requests=[];
ws.addEventListener('message',({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(Error(JSON.stringify(m.error)));else p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Log.entryAdded'&&m.params.entry.level==='error')errors.push(m.params.entry);else if(m.method==='Fetch.requestPaused'){const p=m.params;if(p.request.url.startsWith('http://127.0.0.1:8765/'))send('Fetch.continueRequest',{requestId:p.requestId});else{requests.push(p.request.url);send('Fetch.failRequest',{requestId:p.requestId,errorReason:'BlockedByClient'});}}});
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function ready(path='/'){for(let i=0;i<80;i++){if(await evaluate(`document.readyState==='complete' && location.pathname==='${path}' && !!document.querySelector('#theme')`))return;await delay(100);}throw Error('Page did not become ready');}
async function evaluate(expression){const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
await send('Page.enable');await send('Page.addScriptToEvaluateOnNewDocument',{source:'try { localStorage.removeItem("neurosush-theme"); } catch {}'});await send('Runtime.enable');await send('Log.enable');await send('Fetch.enable',{patterns:[{urlPattern:'*'}]});
for(const [label,width,height] of [['desktop',1440,900],['tablet',768,1024],['mobile',390,844]]){
 await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:'http://127.0.0.1:8765/'});await ready();await delay(300);
 for(const theme of ['dark','light']){
  await evaluate(`document.documentElement.dataset.theme==='${theme}' || document.querySelector('#theme').click()`);
  for(const progress of [0,.15,.2,.3,.4,.45,.6,.75,.8,.9,1]){
   await evaluate(`scrollTo({top:document.querySelector('#story').offsetTop+${progress}*(document.querySelector('#story').offsetHeight-innerHeight),behavior:'instant'})`);await delay(700);
   if(await evaluate('document.documentElement.scrollWidth')>width)throw Error('Horizontal overflow index '+label+' progress '+progress);
   const snap=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${shots}/${label}-${theme}-${progress}.png`,Buffer.from(snap.data,'base64'));
  }
 }
 console.log(label,'scene',await evaluate('JSON.stringify(neurosushScene.stats)'));
 for(const theme of ['dark','light']){
  await evaluate(`document.documentElement.dataset.theme==='${theme}' || document.querySelector('#theme').click()`);
  for(const section of ['workflow','evidence','studio']){
   await evaluate(`document.querySelector('#${section}').scrollIntoView({behavior:'instant'})`);await delay(200);
   if(await evaluate('document.documentElement.scrollWidth')>width)throw Error('Section overflow '+section+' '+label);
   const snap=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${shots}/${label}-${theme}-${section}.png`,Buffer.from(snap.data,'base64'));
  }
 }

 await evaluate('document.querySelector("#plate-search").value="N05";document.querySelector("#plate-search").dispatchEvent(new Event("input"))');
 if(await evaluate('document.querySelectorAll(".plate-list li:not([hidden])").length')!==1)throw Error('Search failed');
 for(const page of ['gallery.html','results.html','research.html','studio-live.html','plates/N01.html','plates/Y01.html']){
  await send('Page.navigate',{url:'http://127.0.0.1:8765/'+page});await ready('/'+page);await delay(300);
  if(await evaluate('document.documentElement.scrollWidth')>width)throw Error('Overflow '+page+' '+label);
  for(const theme of ['dark','light']){
   await evaluate(`document.documentElement.dataset.theme==='${theme}' || document.querySelector('#theme').click()`);await delay(150);
   const snap=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${shots}/${label}-${theme}-${page.replaceAll('/','-')}.png`,Buffer.from(snap.data,'base64'));
  }
 }
}
await send('Page.navigate',{url:'http://127.0.0.1:8765/'});await ready();await delay(300);
await send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
await delay(150);
if(await evaluate('document.documentElement.scrollWidth')>1920)throw Error('1920px overflow');
console.log('1920px layout passed');
await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
await evaluate('scrollTo({top:0,behavior:"instant"})');await delay(200);
console.log('reduced motion',await evaluate('matchMedia("(prefers-reduced-motion: reduce)").matches'));
console.log('console errors',errors.length,'external page requests',requests.length);
if(errors.length||requests.length)throw Error(JSON.stringify({errors,requests}));
}catch(e){console.error(e);process.exitCode=1;}finally{ws?.close();chrome.kill();}
