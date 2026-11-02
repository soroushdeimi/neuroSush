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
 let saved='light';try{saved=localStorage.getItem('neurosush-theme')||'light';}catch{}theme(saved==='light'?'light':'dark');
 $('#theme')?.addEventListener('click',()=>{theme(light?'dark':'light');if($('#neural'))drawStatic();});
 $('#menu')?.addEventListener('click',()=>{const open=$('#main-nav').classList.toggle('open');$('#menu').setAttribute('aria-expanded',String(open));});
 document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(button.dataset.copy);button.setAttribute('aria-label','Copied');}catch{button.setAttribute('aria-label','Select and copy the command');}}));
 $('#plate-search')?.addEventListener('input',e=>{let count=0;document.querySelectorAll('.plate-list li').forEach(li=>{li.hidden=!li.textContent.toLowerCase().includes(e.target.value.toLowerCase().trim());if(!li.hidden)count++;});$('#search-status').textContent=`${count} plates found`;});
 const css=()=>getComputedStyle(document.documentElement);
 const color=name=>css().getPropertyValue(name).trim();
 const rgb=name=>{const hex=color(name).slice(1);return [0,2,4].map(i=>parseInt(hex.slice(i,i+2),16)/255);};
 const canvas=$('#neural');if(!canvas)return;
 let seed=812;const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;};
 const count=innerWidth<650?9000:32000, shapes=Array.from({length:6},()=>new Float32Array(count*3));
 const tau=Math.PI*2;
 for(let i=0;i<6;i++)shapes[i]=window.NeuroShapes.neuron(count);
 function mobileFigures(){
  if(innerWidth>=1024)return;
  document.querySelectorAll('.chapter').forEach((chapter,index)=>{
   let figure=chapter.querySelector('.chapter-figure');
   if(!figure){figure=document.createElement('canvas');figure.className='chapter-figure';figure.setAttribute('aria-hidden','true');chapter.prepend(figure);}
   const w=figure.clientWidth,h=figure.clientHeight,d=Math.min(devicePixelRatio,1.5);figure.width=w*d;figure.height=h*d;
   const context=figure.getContext('2d');context.scale(d,d);context.fillStyle=color('--text-2');context.globalAlpha=.6;
   const data=shapes[index];for(let i=0;i<count;i++){context.fillRect(w*.5+data[i*3]*w*.4,h*.48-data[i*3+1]*w*.4,1.5,1.5);}
  });
 }
 let gl=null,ctx=null,program,uniform={},buffers=[],quality=count,progress=0,target=0,px=0,py=0,frame=0,visible=true,last=0,slow=0;
 if(!reduced())gl=canvas.getContext('webgl2',{alpha:true,premultipliedAlpha:false,antialias:false,powerPreference:'low-power'});
 if(gl){
 const vertex=`#version 300 es
 precision highp float;
 in vec3 a;in vec3 b;in vec3 c;in vec3 d;in vec3 e;in vec3 f;
 uniform float progress,time,aspect,mobile,spikeAge;uniform vec2 pointer;out float glow;out float region;
 void main(){float q=clamp(progress,0.,1.)*5.;int k=int(floor(q));float u=fract(q);u=u*u*(3.-2.*u);vec3 p=k==0?mix(a,b,u):k==1?mix(b,c,u):k==2?mix(c,d,u):k==3?mix(d,e,u):mix(e,f,u);
 float turn=.18*sin(time*.12)+0.;mat2 rot=mat2(cos(turn),-sin(turn),sin(turn),cos(turn));p.xz=rot*p.xz;float perspective=3.8/(3.8+p.z);vec2 xy=p.xy*perspective*mix(1.,.55,mobile);xy.x=xy.x/aspect*.9+mix(.43*cos(progress*5.*3.14159),.02,mobile);xy.y=xy.y*.7-mix(0.,.44,mobile);gl_Position=vec4(xy,0.,1.);gl_PointSize=(1.6+1.1*sin(float(gl_VertexID)*1.8+time*2.-length(p)*8.))*min(2.,mobile+1.);glow=.25;float travel=abs(a.y+.0864);float wave=exp(-pow((travel-spikeAge*25.)*18.,2.));glow+=wave*(a.y<-.144?1.8:.55)*(1.-smoothstep(0.,.15,progress));region=step(.4,progress)*step(progress,.78)*step(d.y,-.15)*step(abs(d.x),.65);}`;
 const fragment=`#version 300 es
 precision highp float;uniform float light,lineMode;uniform vec3 inkColor,spikeColor;in float glow;in float region;out vec4 color;
 void main(){if(lineMode>.5){color=vec4(inkColor,.6);return;}float r=length(gl_PointCoord-.5);if(r>.5)discard;vec3 ink=mix(inkColor,spikeColor,step(.7,glow));color=vec4(ink,(1.-r*2.)*(.6+glow*.3));}`;
 function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
 program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);gl.useProgram(program);
 shapes.forEach((data,i)=>{const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);buffers.push(buffer);const loc=gl.getAttribLocation(program,'abcdef'[i]);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0);});
 for(const key of ['inkColor','spikeColor','lineMode','progress','time','aspect','mobile','pointer','light','spikeAge'])uniform[key]=gl.getUniformLocation(program,key);
 const lines=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,lines);gl.bufferData(gl.ARRAY_BUFFER,shapes[0].lines,gl.STATIC_DRAW);buffers.lines=lines;
 gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
 }else{ctx=canvas.getContext('2d');document.documentElement.classList.add('static-scene');}
 function advanceLIF(dt){window.NeuroInstruments.advance(dt);}
 function trace(ms=0){window.NeuroInstruments.draw(ms/1000);}
 function size(){const d=Math.min(devicePixelRatio,2),w=Math.round(canvas.clientWidth*d),h=Math.round(canvas.clientHeight*d);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return [w,h];}
 function render(ms){const [w,h]=size();if(gl){gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.uniform1f(uniform.progress,progress);gl.uniform1f(uniform.spikeAge,window.NeuroInstruments.spikeAge);gl.uniform1f(uniform.time,reduced()?0:ms/1000);gl.uniform1f(uniform.aspect,w/h);gl.uniform1f(uniform.mobile,innerWidth<650?1:0);gl.uniform1f(uniform.light,light?1:0);gl.uniform3fv(uniform.inkColor,rgb('--text'));gl.uniform3fv(uniform.spikeColor,rgb('--spike'));gl.uniform2f(uniform.pointer,px,py);gl.drawArrays(gl.POINTS,0,quality);
 if(progress<.005){const loc=gl.getAttribLocation(program,'a');gl.bindBuffer(gl.ARRAY_BUFFER,buffers.lines);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0);gl.uniform1f(uniform.lineMode,1);gl.drawArrays(gl.LINES,0,shapes[0].lines.length/3);gl.uniform1f(uniform.lineMode,0);gl.bindBuffer(gl.ARRAY_BUFFER,buffers[0]);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0);}

 }else if(ctx){ctx.clearRect(0,0,w,h);ctx.fillStyle=color('--text');for(let i=0;i<5000;i++){const p=shapes[0],x=w*(innerWidth<650?.55:.73)+p[i*3]*h*.31,y=h*(innerWidth<650?.73:.5)-p[i*3+1]*h*.31;ctx.globalAlpha=.3+(.7*i/5000);ctx.fillRect(x,y,1.5,1.5);}}
 }
 function drawStatic(){trace();render(0);mobileFigures();}
 const chapters=[...document.querySelectorAll('.chapter')];
 function scroll(){const rect=$('#story').getBoundingClientRect();target=Math.max(0,Math.min(1,-rect.top/($('#story').offsetHeight-innerHeight)));chapters.forEach(chapter=>{const copy=chapter.querySelector('.chapter-copy'),r=copy.getBoundingClientRect();copy.style.opacity=reduced()||innerWidth<1024?'1':String(Math.max(0,Math.min(1,(innerHeight-r.top)/180,(r.bottom-65)/150)));});const active=Math.round(target*5);$('.rail').hidden=rect.bottom<innerHeight*.3;document.querySelectorAll('.rail a').forEach((a,i)=>a.setAttribute('aria-current',String(i===active)));}
 addEventListener('scroll',()=>{scroll();if(reduced()){progress=target;render(0);}},{passive:true});addEventListener('resize',()=>{scroll();drawStatic();},{passive:true});

 const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)start();else{cancelAnimationFrame(frame);frame=0;}});observer.observe($('#story'));
 const timing=[];window.neurosushScene={get stats(){return {points:quality,frames:timing.length,fps:timing.length?1000/(timing.reduce((a,b)=>a+b,0)/timing.length):0,webgl:!!gl};}};
 function loop(ms){frame=0;if(document.hidden||!visible||reduced())return;const dt=last?ms-last:16.7;last=ms;if(dt<100){timing.push(dt);if(timing.length>180)timing.shift();slow=dt>25?slow+1:Math.max(0,slow-1);if(slow>40&&quality>6000){quality=Math.floor(quality*.75);slow=0;}}advanceLIF(dt);trace(ms);progress+=(target-progress)*(1-Math.exp(-dt/100));render(ms);frame=requestAnimationFrame(loop);}
 function start(){last=0;if(gl&&!frame&&!document.hidden&&visible&&!reduced())frame=requestAnimationFrame(loop);}
 document.addEventListener('visibilitychange',()=>{cancelAnimationFrame(frame);frame=0;if(!document.hidden)start();});function changeMotion(){motionState();cancelAnimationFrame(frame);frame=0;progress=target;drawStatic();start();}
 motion.addEventListener('change',changeMotion);document.addEventListener('scene:motion',changeMotion);
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();cancelAnimationFrame(frame);frame=0;});scroll();drawStatic();start();
})();
