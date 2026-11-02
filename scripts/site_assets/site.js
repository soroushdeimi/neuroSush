'use strict';
(() => {
 const $=s=>document.querySelector(s), motion=matchMedia('(prefers-reduced-motion: reduce)');
 let userReduced=false;try{userReduced=localStorage.getItem('neurosush-motion')==='reduce';}catch{}
 const reduced=()=>motion.matches||userReduced;
 function motionState(){document.documentElement.dataset.motion=reduced()?'reduce':'full';$('#motion')?.setAttribute('aria-pressed',String(reduced()));}
 motionState();
 $('#motion')?.addEventListener('click',()=>{userReduced=!reduced();try{localStorage.setItem('neurosush-motion',userReduced?'reduce':'full');}catch{}motionState();document.dispatchEvent(new Event('scene:motion'));});
 document.querySelectorAll('#main-nav a').forEach(a=>a.addEventListener('click',()=>{$('#main-nav').classList.remove('open');$('#menu').setAttribute('aria-expanded','false');}));
 addEventListener('keydown',e=>{if(e.key==='Escape'){$('#main-nav')?.classList.remove('open');$('#menu')?.setAttribute('aria-expanded','false');}});
 let light=false;
 function theme(value){light=value==='light';document.documentElement.dataset.theme=value;if(window.Atlas)window.Atlas.Theme.set(value);$('#theme')?.setAttribute('aria-label',`Switch to ${light?'dark':'light'} theme`);$('#theme')?.setAttribute('aria-pressed',String(light));try{localStorage.setItem('neurosush-theme',value);}catch{} }
 let saved='dark';try{saved=localStorage.getItem('neurosush-theme')||'dark';}catch{}theme(saved==='light'?'light':'dark');
 $('#theme')?.addEventListener('click',()=>{theme(light?'dark':'light');if($('#neural'))drawStatic();});
 $('#menu')?.addEventListener('click',()=>{const open=$('#main-nav').classList.toggle('open');$('#menu').setAttribute('aria-expanded',String(open));});
 document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(button.dataset.copy);button.setAttribute('aria-label','Copied');}catch{button.setAttribute('aria-label','Select and copy the command');}}));
 $('#plate-search')?.addEventListener('input',e=>{let count=0;document.querySelectorAll('.plate-list li').forEach(li=>{li.hidden=!li.textContent.toLowerCase().includes(e.target.value.toLowerCase().trim());if(!li.hidden)count++;});$('#search-status').textContent=`${count} plates found`;});
 const css=()=>getComputedStyle(document.documentElement);
 const colors={};const color=name=>colors[String(light)+name]||(colors[String(light)+name]=css().getPropertyValue(name).trim());
 const rgb=name=>{const hex=color(name).slice(1);return [0,2,4].map(i=>parseInt(hex.slice(i,i+2),16)/255);};
 const canvas=$('#neural');if(!canvas)return;
 let seed=812;const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;};
 const count=innerWidth<650?9000:32000, shapes=Array.from({length:6},()=>new Float32Array(count*3));
 const tau=Math.PI*2;
 shapes[0]=window.NeuroShapes.neuron(count);
 shapes[1]=window.NeuroShapes.synapse(count);
 shapes[2]=window.NeuroShapes.microcircuit(count);
 shapes[3]=window.NeuroShapes.sparse(count);
 shapes[4]=window.NeuroShapes.hierarchy(count);
 shapes[5]=window.NeuroShapes.brain(count);
 shapes[6]=window.NeuroShapes.grid(count);
 function mobileFigures(ms=0){
  if(innerWidth>=1024)return;
  const time=ms/1000,step=Math.floor(time*.65)%3;
  if(!$('.mobile-grid')){const extra=document.createElement('div');extra.className='mobile-grid';$('#space .chapter-copy').before(extra);}
  [...document.querySelectorAll('.chapter'),$('.mobile-grid')].forEach((chapter,index)=>{
   let figure=chapter.querySelector('.chapter-figure');
   if(!figure){figure=document.createElement('canvas');figure.className='chapter-figure';figure.setAttribute('aria-hidden','true');chapter.prepend(figure);}
   if(ms&&figure.dataset.visible!=='true')return;
   if(!figure.width||!ms){const d=Math.min(devicePixelRatio,1.5);figure.width=figure.clientWidth*d;figure.height=figure.clientHeight*d;}
   const w=figure.width,h=figure.height,context=figure.getContext('2d'),data=shapes[index];context.clearRect(0,0,w,h);
   const project=p=>[w*.5+p[0]*w*.43,h*.48-p[1]*w*.43];
   context.strokeStyle=color('--text-3');context.lineWidth=1.5;
   if(data.lines){context.beginPath();for(let i=0;i<Math.min(data.lines.length,count*6);i+=6){const a=project(data.lines.slice(i,i+3)),b=project(data.lines.slice(i+3,i+6));context.moveTo(...a);context.lineTo(...b);}context.stroke();}
   for(let i=0;i<count;i++){
    const tag=data.tags?.[i]||0;let ink='--text-2',alpha=.45;
    if(tag<0)alpha=.15+Math.abs(tag)*.7;if(tag===3)ink='--trace';
    if(i%13===0){if(tag===1)ink='--exc';if(tag===2)ink='--inh';if(tag===3||tag===4)ink='--trace';}
    if(tag>=100&&tag<1124&&(Math.round(tag-100)*37+step*211)%1024<20){ink='--spike';alpha=1;}
    if(tag>=1200&&tag<2224){if((Math.round(tag-1200)*37+((step+1)%3)*211)%1024<20){ink='--trace';alpha=1;}else continue;}
    if(tag>=3000){if(tag-3000>8-(time*.6%3)*3)alpha=.06;if(tag===3000)ink='--trace';}
    if(tag===7||tag===8)ink='--spike';if(tag===9)ink='--trace';if(tag===10){ink='--exc';alpha=Math.exp(-(time%4));}
    let p=Array.from(data.slice(i*3,i*3+3));if(index===1&&tag===7)p[1]=.1-((time*1.4+i*.13)%1)*.16;
    if(index===4&&tag===9)p[1]=.74-((time*.38+i*.017)%1.3);if(index===4&&tag===10)p[1]=-.56+((time*.38+i*.017)%1.3);
    if(index===0&&Math.abs(Math.abs(p[1]+.0864)-window.NeuroInstruments.spikeAge*25)<.04)ink='--spike';
    const [x,y]=project(p);context.globalAlpha=alpha;context.fillStyle=color(ink);context.fillRect(x,y,2,2);
   }
   context.globalAlpha=1;context.font=(13*Math.min(devicePixelRatio,1.5))+'px '+css().getPropertyValue('--font-sans');context.fillStyle=color('--text');(data.labels||[]).forEach(([label,x,y])=>context.fillText(label,...project([x,y,0])));
  });
 }
 let gl=null,ctx=null,program,uniform={},buffers=[],quality=count,progress=0,target=0,px=0,py=0,frame=0,visible=true,last=0,slow=0;
 if(!reduced())gl=canvas.getContext('webgl2',{alpha:true,premultipliedAlpha:false,antialias:false,powerPreference:'low-power'});
 if(gl){
 const vertex=`#version 300 es
 precision highp float;
 in vec3 a;in vec3 b;in vec3 c;in vec3 d;in vec3 e;in vec3 f;in vec3 g;
 in float ta,tb,tc,td,te,tf,tg;
 uniform float progress,time,aspect,mobile,spikeAge,lineMode,zoom;uniform vec2 pointer;out float glow;out float region;out float strength;
 void main(){float q=clamp(progress,0.,1.)*5.;int k=min(4,int(floor(q)));float u=q>=5.?1.:smoothstep(.05,.25,fract(q));q=float(k)+u;vec3 p=k==0?mix(a,b,u):k==1?mix(b,c,u):k==2?mix(c,d,u):k==3?mix(d,e,u):mix(e,f,u);
 if(q>4.8)p=mix(p,g,zoom);
 float tag=k==0?mix(ta,tb,u):k==1?mix(tb,tc,u):k==2?mix(tc,td,u):k==3?mix(td,te,u):mix(te,tf,u);
 if(q>.8&&q<1.2&&lineMode<.5){if(abs(tag-6.)<.1){float weight=.5+.2*sin(time*.7);p.x*=.8+weight*.4;p.y=(p.y+.4)*(.8+weight*.4)-.4;}if(abs(tag-7.)<.1)p.y=.1-fract(time*1.4+float(gl_VertexID)*.13)*.16;}
 if(q>3.8&&q<4.2&&lineMode<.5){if(abs(tag-9.)<.1){p.y=.74-mod(time*.38+float(gl_VertexID)*.017,1.3);tag=3.;}if(abs(tag-10.)<.1){p.y=-.56+mod(time*.38+float(gl_VertexID)*.017,1.3);tag=1.;}} if(q>4.8)tag=zoom>.5?tg:tag;
 float turn=.18*sin(time*.12)+0.;mat2 rot=mat2(cos(turn),-sin(turn),sin(turn),cos(turn));p.xz=rot*p.xz;float perspective=3.8/(3.8+p.z);vec2 xy=p.xy*perspective*mix(1.,.55,mobile);xy.x=xy.x/aspect*.9+mix(.43*cos(q*3.14159),.02,mobile);xy.y=xy.y*.7-mix(0.,.44,mobile);gl_Position=vec4(xy,0.,1.);gl_PointSize=2.;glow=.25;float travel=abs(a.y+.0864);float wave=exp(-pow((travel-spikeAge*25.)*18.,2.));glow+=wave*(a.y<-.144?1.8:.55)*(1.-smoothstep(0.,.15,progress));if(q>1.8&&q<2.2){float wave=abs(p.x-sin(time*1.1));glow+=step(wave,.05)*.9;if(abs(tag-8.)<.1)glow=1.;}strength=abs(tag-1.)<.1&&q>3.8&&q<4.2?exp(-mod(time,4.)):1.;
 if(q>2.8&&q<3.2&&lineMode<.5){float stepIndex=mod(floor(time*.65),3.);if(tag>=100.&&tag<1124.){float id=tag-100.;if(mod(id*37.+stepIndex*211.,1024.)<20.)glow=1.;}if(tag>=1200.&&tag<2224.){float id=tag-1200.;if(mod(id*37.+mod(stepIndex+1.,3.)*211.,1024.)<20.)tag=3.;else strength=.1;}if(tag>=3000.){float candidate=tag-3000.;strength=candidate<=8.-mod(time*.6,3.)*3.?1.:.08;tag=candidate==0.?3.:0.;}}

 if(q>4.8&&zoom>.05&&zoom<.5&&tag<0.)strength*=.2;
 if(tag<0.)strength*=.15+abs(tag)*.85;region=tag;}`;
 const fragment=`#version 300 es
 precision highp float;uniform float light,lineMode;uniform vec3 inkColor,spikeColor,excColor,inhColor,traceColor;uniform float time;in float glow;in float region;in float strength;out vec4 color;
 void main(){if(lineMode>.5){color=vec4(inkColor,.6);return;}float r=length(gl_PointCoord-.5);if(r>.5)discard;vec3 ink=inkColor;
 if(mod(gl_FragCoord.x+gl_FragCoord.y,13.)<1.){if(abs(region-1.)<.1)ink=excColor;if(abs(region-2.)<.1)ink=inhColor;if(abs(region-3.)<.1||abs(region-4.)<.1||abs(region-5.)<.1)ink=traceColor;if(abs(region-6.)<.1)ink=sin(time*.7)>0.?excColor:inhColor;}
 if(abs(region-3.)<.1)ink=traceColor;if(abs(region-7.)<.1||glow>.7)ink=spikeColor;color=vec4(ink,(glow>.7?1.:.6)*strength);}`;
 function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
 program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);gl.useProgram(program);
 shapes.forEach((data,i)=>{const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);buffers.push(buffer);const loc=gl.getAttribLocation(program,'abcdefg'[i]);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0);const tags=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,tags);gl.bufferData(gl.ARRAY_BUFFER,data.tags||new Float32Array(count),gl.STATIC_DRAW);const tagLoc=gl.getAttribLocation(program,'t'+'abcdefg'[i]);gl.enableVertexAttribArray(tagLoc);gl.vertexAttribPointer(tagLoc,1,gl.FLOAT,false,0,0);});
 for(const key of ['excColor','inhColor','traceColor','inkColor','spikeColor','lineMode','progress','time','aspect','mobile','pointer','light','spikeAge','zoom'])uniform[key]=gl.getUniformLocation(program,key);
 buffers.lines=shapes.map(shape=>{const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,shape.lines||new Float32Array(0),gl.STATIC_DRAW);return buffer;});
 gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
 }else{ctx=canvas.getContext('2d');document.documentElement.classList.add('static-scene');}
 function advanceLIF(dt){window.NeuroInstruments.advance(dt);}
 function trace(ms=0){window.NeuroInstruments.draw(ms/1000);}
 function size(){const d=Math.min(devicePixelRatio,2),w=Math.round(canvas.clientWidth*d),h=Math.round(canvas.clientHeight*d);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return [w,h];}
 const annotation=document.createElement('canvas');annotation.className='scene-annotations';canvas.after(annotation);
 function labels(w,h){annotation.width=w;annotation.height=h;const c=annotation.getContext('2d'),d=Math.min(devicePixelRatio,2);c.font=(13*d)+'px '+css().getPropertyValue('--font-sans');c.fillStyle=color('--text-2');if(progress>0.92)return;const raw=progress*5,base=Math.floor(raw),u=Math.max(0,Math.min(1,(raw-base-.05)/.2)),q=base+u*u*(3-2*u),k=progress>.97?6:Math.min(5,Math.round(q)),offset=.43*Math.cos(q*Math.PI);(shapes[k].labels||[]).forEach(([label,x,y])=>c.fillText(label,(x/(w/h)*.9+offset+1)*w/2,(1-y*.7)*h/2));}
 function render(ms){if(innerWidth<1024){mobileFigures(ms);return;}const [w,h]=size();labels(w,h);if(gl){gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.uniform1f(uniform.progress,progress);gl.uniform1f(uniform.zoom,Math.max(0,Math.min(1,(progress-.94)/.06)));gl.uniform1f(uniform.spikeAge,window.NeuroInstruments.spikeAge);gl.uniform1f(uniform.time,reduced()?0:ms/1000);gl.uniform1f(uniform.aspect,w/h);gl.uniform1f(uniform.mobile,innerWidth<650?1:0);gl.uniform1f(uniform.light,light?1:0);gl.uniform3fv(uniform.inkColor,rgb('--text'));gl.uniform3fv(uniform.spikeColor,rgb('--spike'));gl.uniform3fv(uniform.excColor,rgb('--exc-mark'));gl.uniform3fv(uniform.inhColor,rgb('--inh-mark'));gl.uniform3fv(uniform.traceColor,rgb('--trace-mark'));gl.uniform2f(uniform.pointer,px,py);gl.drawArrays(gl.POINTS,0,quality);
 const raw=progress*5,k=Math.floor(raw),u=Math.max(0,Math.min(1,(raw-k-.05)/.2)),scene=progress>.97?6:Math.min(5,Math.round(k+u*u*(3-2*u)));
 if(shapes[scene].lines?.length){gl.uniform1f(uniform.lineMode,1);for(let i=0;i<7;i++){gl.bindBuffer(gl.ARRAY_BUFFER,buffers.lines[scene]);gl.vertexAttribPointer(gl.getAttribLocation(program,'abcdefg'[i]),3,gl.FLOAT,false,0,0);}gl.drawArrays(gl.LINES,0,Math.min(count,shapes[scene].lines.length/3));gl.uniform1f(uniform.lineMode,0);for(let i=0;i<7;i++){gl.bindBuffer(gl.ARRAY_BUFFER,buffers[i]);gl.vertexAttribPointer(gl.getAttribLocation(program,'abcdefg'[i]),3,gl.FLOAT,false,0,0);}}


 }else if(ctx){ctx.clearRect(0,0,w,h);ctx.fillStyle=color('--text');for(let i=0;i<5000;i++){const p=shapes[0],x=w*(innerWidth<650?.55:.73)+p[i*3]*h*.31,y=h*(innerWidth<650?.73:.5)-p[i*3+1]*h*.31;ctx.globalAlpha=.3+(.7*i/5000);ctx.fillRect(x,y,1.5,1.5);}}
 }
 function drawStatic(){trace();render(0);mobileFigures();}
 const chapters=[...document.querySelectorAll('.chapter')];
 function scroll(){const rect=$('#story').getBoundingClientRect();target=Math.max(0,Math.min(1,-rect.top/($('#story').offsetHeight-innerHeight)));chapters.forEach(chapter=>{const copy=chapter.querySelector('.chapter-copy'),r=copy.getBoundingClientRect();copy.style.opacity=reduced()||innerWidth<1024?'1':String(Math.max(0,Math.min(1,(innerHeight-r.top)/180,(r.bottom-65)/150)));});document.querySelectorAll('.chapter-figure').forEach(f=>{const r=f.getBoundingClientRect();f.dataset.visible=String(r.bottom>64&&r.top<innerHeight);});const active=Math.round(target*5);$('.rail').hidden=rect.bottom<innerHeight*.3;document.querySelectorAll('.rail a').forEach((a,i)=>a.setAttribute('aria-current',String(i===active)));const stage=$('.stage');if(stage){stage.style.visibility=(target>=1||rect.bottom<=0)?'hidden':'visible';}}
 addEventListener('scroll',()=>{scroll();if(reduced()){progress=target;render(0);}},{passive:true});addEventListener('resize',()=>{scroll();drawStatic();},{passive:true});

 document.documentElement.classList.add('has-js');
 const wfTabs=document.querySelectorAll('.workflow-tabs [role="tab"]');
 const wfPanels=document.querySelectorAll('.workflow-panel');
 wfTabs.forEach((tab,i)=>{
  tab.addEventListener('click',()=>{
   wfTabs.forEach((t,j)=>{t.setAttribute('aria-selected',String(j===i));t.setAttribute('tabindex',j===i?'0':'-1');});
   wfPanels.forEach((p,j)=>{if(j===i)p.removeAttribute('hidden');else p.setAttribute('hidden','');});
  });
  tab.addEventListener('keydown',e=>{
   let targetIdx=i;
   if(e.key==='ArrowRight'||e.key==='ArrowDown'){targetIdx=(i+1)%wfTabs.length;e.preventDefault();}
   else if(e.key==='ArrowLeft'||e.key==='ArrowUp'){targetIdx=(i-1+wfTabs.length)%wfTabs.length;e.preventDefault();}
   if(targetIdx!==i){wfTabs[targetIdx].focus();wfTabs[targetIdx].click();}
  });
 });

 const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)start();else{cancelAnimationFrame(frame);frame=0;}});observer.observe($('#story'));
 const timing=[];window.neurosushScene={get stats(){return {points:quality,frames:timing.length,fps:timing.length?1000/(timing.reduce((a,b)=>a+b,0)/timing.length):0,webgl:!!gl};}};
 function loop(ms){frame=0;if(document.hidden||!visible||reduced())return;const dt=last?ms-last:16.7;last=ms;if(dt<100){timing.push(dt);if(timing.length>180)timing.shift();slow=dt>25?slow+1:Math.max(0,slow-1);if(slow>40&&quality>6000){quality=Math.floor(quality*.75);slow=0;}}advanceLIF(dt);trace(ms);progress+=(target-progress)*(1-Math.exp(-dt/100));render(ms);frame=requestAnimationFrame(loop);}
 function start(){last=0;if(gl&&!frame&&!document.hidden&&visible&&!reduced())frame=requestAnimationFrame(loop);}
 document.addEventListener('visibilitychange',()=>{cancelAnimationFrame(frame);frame=0;if(!document.hidden)start();});function changeMotion(){motionState();cancelAnimationFrame(frame);frame=0;progress=target;drawStatic();start();}
 motion.addEventListener('change',changeMotion);document.addEventListener('scene:motion',changeMotion);
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();cancelAnimationFrame(frame);frame=0;});scroll();drawStatic();start();
})();
