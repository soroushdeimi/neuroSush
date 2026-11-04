import {panels as templates} from './panels.mjs';
import {ZoomEngine,cameraAt,validateReplay} from './engine.mjs';
const response=await fetch(new URL('./facts.json',import.meta.url));
if(!response.ok)throw Error('Fact ledger unavailable');
const ledger=await response.json();
const facts=Object.fromEntries(ledger.filter(f=>f.status==='VERIFIED').map(f=>[f.id,f.value]));
const panels=templates.map(p=>({...p,copy:p.copy.replace(/\{([a-z0-9_]+)\}/g,(_,id)=>{if(!(id in facts))throw Error('Missing fact: '+id);return (id.startsWith('ideal_')&&id.endsWith('_score'))||id.endsWith('_margin')?facts[id].toFixed(3):facts[id];})}));
const replays=await Promise.all(['v1','v2'].map(async variant=>{
  const response=await fetch(new URL(`../data/tem-run-${variant}-s0.json`,import.meta.url));
  if(!response.ok)throw Error('Recorded run unavailable');return validateReplay(await response.json());
}));
let chapter=-1;
const q=s=>document.querySelector(s);
function onState(active){if(panels[active].scene==='replay'){const world=engine.replays[engine.replayVariant].worlds[engine.replayWorld];q('#recorded-step').value=(Math.floor(engine.phase*4)%world.walk.length)/(world.walk.length-1);}if(chapter===active)return;chapter=active;q('#story-panel').scrollTop=0;const p=panels[active];q('#panel-title').innerHTML=p.title;q('#panel-label').textContent='THE SPIKING TEM · '+p.label.toUpperCase();q('#panel-copy').textContent=p.copy;q('#replay-controls').hidden=p.scene!=='replay';q('#panel-copy').dataset.factIds=p.facts.join(' ');q('#panel-note').textContent=p.note;q('#citation').textContent=p.source;q('#camera-badge').textContent=p.badge;q('#scene-caption').textContent=p.caption||'Illustrated mechanisms · not a trained network';q('#position').textContent=p.label;q('#previous').disabled=active===0;q('#next').disabled=active===panels.length-1;}
const engine=new ZoomEngine(q('#scene'),q('#overlay'),panels,facts,onState);
engine.replays=replays;engine.replayVariant=0;engine.replayWorld=0;
for(const [id,key]of [['recording','replayVariant'],['recorded-world','replayWorld']])q('#'+id).addEventListener('change',event=>{engine[key]=Number(event.target.value);engine.phase=0;engine.wake();});
q('#recorded-step').addEventListener('input',event=>{engine.setPlaying(false);q('#play').textContent='Animate';q('#play').setAttribute('aria-pressed','false');const world=engine.replays[engine.replayVariant].worlds[engine.replayWorld];engine.phase=Number(event.target.value)*(world.walk.length-1)/4;engine.wake();});
q('#scroll-track').style.height=panels.length*100+'svh';
let range=1;
function resize(){range=q('#scroll-track').offsetHeight-innerHeight;}resize();addEventListener('resize',resize);
addEventListener('scroll',()=>{if(!engine.reduced)engine.setTarget(scrollY/range);},{passive:true});
function navigate(i){const index=Math.max(0,Math.min(panels.length-1,i));scrollTo({top:engine.keys[index].t*range,behavior:'instant'});engine.setTarget(engine.keys[index].t);}
q('#previous').addEventListener('click',()=>navigate(chapter-1));q('#next').addEventListener('click',()=>navigate(chapter+1));
addEventListener('keydown',event=>{if(engine.reduced||q('dialog').open||event.target.closest('button,a,input,select'))return;if(['ArrowDown','PageDown','ArrowUp','PageUp','Home','End'].includes(event.key)){event.preventDefault();navigate(event.key==='Home'?0:event.key==='End'?panels.length-1:chapter+(['ArrowDown','PageDown'].includes(event.key)?1:-1));}});
q('#play').addEventListener('click',()=>{engine.setPlaying(!engine.playing);q('#play').setAttribute('aria-pressed',String(engine.playing));q('#play').textContent=engine.playing?'Pause':'Animate';});
q('#theme').addEventListener('click',()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';engine.updateTheme();});
q('#references-toggle').addEventListener('click',()=>q('#references').showModal());q('#close-references').addEventListener('click',()=>q('#references').close());
for(const p of panels){const section=document.createElement('p');section.textContent=p.label+': '+p.facts.map(id=>{const f=ledger.find(row=>row.id===id);if(!f||f.status!=='VERIFIED')throw Error('Unverified fact: '+id);return f.source;}).join('; ');q('#panel-sources').append(section);}
const motion=matchMedia('(prefers-reduced-motion: reduce)');let stillsBuilt=false;
function reducedMotion(){engine.reduced=motion.matches;resize();if(motion.matches){cancelAnimationFrame(engine.raf);engine.raf=0;if(!stillsBuilt){const saved=engine.t,savedPhase=engine.phase,savedX=engine.cx,savedY=engine.cy;engine.phase=3;engine.cx=engine.w/2;engine.cy=engine.h/2;for(let i=0;i<panels.length;i++){const p=panels[i];engine.t=engine.keys[i].t;cameraAt(engine.t,engine.keys,engine.camera);engine.draw();const section=document.createElement('section'),heading=document.createElement('h2'),img=document.createElement('img'),copy=document.createElement('p'),note=document.createElement('p');heading.innerHTML=p.title;img.src=engine.canvas.toDataURL();img.alt=p.label+' — '+p.badge;img.width=engine.w;img.height=engine.h;copy.textContent=p.copy;note.textContent=p.badge+' · '+p.note+' '+p.source;section.append(heading,img,copy,note);q('#reduced-story').append(section);}const footer=document.createElement('footer'),heading=document.createElement('h2');heading.textContent='Sources';const ledgerLink=document.createElement('a');ledgerLink.href='facts.json';ledgerLink.textContent='Complete fact ledger ↗';footer.append(heading,q('#references ol').cloneNode(true),ledgerLink);q('#reduced-story').append(footer);engine.t=saved;engine.phase=savedPhase;engine.cx=savedX;engine.cy=savedY;cameraAt(saved,engine.keys,engine.camera);stillsBuilt=true;}}else engine.setTarget(scrollY/range);}
motion.addEventListener('change',reducedMotion);addEventListener('visibilitychange',()=>{engine.lastTime=0;if(!document.hidden)engine.wake();});
window.storyEngine=engine;onState(0);reducedMotion();engine.wake();document.documentElement.dataset.ready='true';
