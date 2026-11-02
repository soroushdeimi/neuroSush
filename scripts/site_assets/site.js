'use strict';
(() => {
 const $=s=>document.querySelector(s), motion=matchMedia('(prefers-reduced-motion: reduce)');
 let light=false;
 function theme(value){light=value==='light';document.documentElement.dataset.theme=value;if(window.Atlas)window.Atlas.Theme.set(value);$('#theme')?.setAttribute('aria-label',`Switch to ${light?'dark':'light'} theme`);$('#theme')?.setAttribute('aria-pressed',String(light));try{localStorage.setItem('neurosush-theme',value);}catch{} }
 let saved='dark';try{saved=localStorage.getItem('neurosush-theme')||'dark';}catch{}theme(saved==='light'?'light':'dark');
 $('#theme')?.addEventListener('click',()=>{theme(light?'dark':'light');if($('#neural'))drawStatic();});
 $('#plate-search')?.addEventListener('input',e=>{let count=0;document.querySelectorAll('.plate-list li').forEach(li=>{li.hidden=!li.textContent.toLowerCase().includes(e.target.value.toLowerCase().trim());if(!li.hidden)count++;});$('#search-status').textContent=`${count} plates found`;});
 const canvas=$('#neural');if(!canvas)return;
 let seed=812;const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;};
 const count=innerWidth<650?9000:32000, shapes=Array.from({length:6},()=>new Float32Array(count*3));
 const tau=Math.PI*2;
 for(let i=0;i<count;i++){
  const u=rnd(),v=rnd(),a=u*tau,z=v*2-1,r=Math.sqrt(1-z*z),depth=rnd(),side=i%2?1:-1;
  const orb=[r*Math.cos(a)*.85,z*.85,r*Math.sin(a)*.85];
  const radius=Math.sqrt(depth)*1.15,arm=i%4,angle=radius*4.5+arm*tau/4+(rnd()-.5)*.32;
  const galaxy=[Math.cos(angle)*radius,Math.sin(angle)*radius*.6+(rnd()-.5)*.09,Math.sin(angle)*radius*.55];
  const column=i%6,layer=Math.floor(i/6)%4;
  const columns=[(column-2.5)*.44+(rnd()-.5)*.16,(layer-1.5)*.45+(rnd()-.5)*.13,(rnd()-.5)*.65];
  const fold=1+.065*Math.sin(a*17+z*9)*Math.cos(z*19),bx=side*(.13+Math.abs(r*Math.cos(a))*.69)*fold;
  let brain=[bx,z*.91*fold+.12,r*Math.sin(a)*.72*fold];
  if(i%17===0)brain=[(rnd()-.5)*.2,-.7-depth*.65,(rnd()-.5)*.24];
  const predictive=[(Math.floor(i/4)%4-1.5)*.58+(rnd()-.5)*.25,(i%4-1.5)*.54,(rnd()-.5)*.8];
  const tile=i%12,grid=[(tile%4-1.5)*.65+(rnd()-.5)*.5,(Math.floor(tile/4)-1)*.55+(rnd()-.5)*.36,(rnd()-.5)*.04];
  [orb,galaxy,columns,brain,predictive,grid].forEach((p,j)=>shapes[j].set(p,i*3));
 }
 let gl=null,ctx=null,program,uniform={},quality=count,progress=0,target=0,px=0,py=0,frame=0,visible=true,last=0,slow=0;
 if(!motion.matches)gl=canvas.getContext('webgl2',{alpha:true,antialias:false,powerPreference:'low-power'});
 if(gl){
 const vertex=`#version 300 es
 precision highp float;
 in vec3 a;in vec3 b;in vec3 c;in vec3 d;in vec3 e;in vec3 f;
 uniform float progress,time,aspect,mobile;uniform vec2 pointer;out float glow;out float region;
 void main(){float q=clamp(progress,0.,1.)*5.;int k=int(floor(q));float u=fract(q);u=u*u*(3.-2.*u);vec3 p=k==0?mix(a,b,u):k==1?mix(b,c,u):k==2?mix(c,d,u):k==3?mix(d,e,u):mix(e,f,u);
 float turn=.18*sin(time*.12)+pointer.x*.14;mat2 rot=mat2(cos(turn),-sin(turn),sin(turn),cos(turn));p.xz=rot*p.xz;p.y+=pointer.y*.07;p*=1.+.015*sin(time*1.8)*(1.-smoothstep(0.,.2,progress));float perspective=3.8/(3.8+p.z);vec2 xy=p.xy*perspective*mix(1.,.55,mobile);xy.x=xy.x/aspect*.9+mix(.43,.02,mobile);xy.y=xy.y*.7-mix(0.,.44,mobile);gl_Position=vec4(xy,0.,1.);gl_PointSize=(1.6+1.1*sin(float(gl_VertexID)*1.8+time*2.-length(p)*8.))*min(2.,mobile+1.);glow=.5+.5*sin(float(gl_VertexID)*.01-time*3.);region=step(.4,progress)*step(progress,.78)*step(d.y,-.15)*step(abs(d.x),.65);}`;
 const fragment=`#version 300 es
 precision highp float;uniform float light;in float glow;in float region;out vec4 color;
 void main(){float r=length(gl_PointCoord-.5);if(r>.5)discard;vec3 ink=mix(vec3(.65,.52,1.),vec3(.27,.12,.52),light);ink=mix(ink,vec3(.12,.8,.68),region);color=vec4(ink,(1.-r*2.)*(.35+glow*.65));}`;
 function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
 program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);gl.useProgram(program);
 shapes.forEach((data,i)=>{const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);const loc=gl.getAttribLocation(program,'abcdef'[i]);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0);});
 for(const key of ['progress','time','aspect','mobile','pointer','light'])uniform[key]=gl.getUniformLocation(program,key);
 gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
 }else{ctx=canvas.getContext('2d');document.documentElement.classList.add('static-scene');}
 const lif=$('#lif'),lc=lif.getContext('2d');let voltage=-65;const samples=[];
 for(let i=0;i<2000;i++){voltage+=.5/20*(-(voltage+65)+20);if(voltage>=-50){samples.push(0);voltage=-70;}else samples.push(voltage);}
 function trace(){const d=Math.min(devicePixelRatio,2);lif.width=lif.clientWidth*d;lif.height=lif.clientHeight*d;lc.strokeStyle=light?'#654096':'#c4adff';lc.lineWidth=d;lc.beginPath();samples.forEach((v,i)=>{const x=i/1999*lif.width,y=lif.height*(1-(v+75)/80);i?lc.lineTo(x,y):lc.moveTo(x,y);});lc.stroke();}
 function size(){const d=Math.min(devicePixelRatio,2),w=Math.round(canvas.clientWidth*d),h=Math.round(canvas.clientHeight*d);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return [w,h];}
 function render(ms){const [w,h]=size();if(gl){gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.uniform1f(uniform.progress,progress);gl.uniform1f(uniform.time,motion.matches?0:ms/1000);gl.uniform1f(uniform.aspect,w/h);gl.uniform1f(uniform.mobile,innerWidth<650?1:0);gl.uniform1f(uniform.light,light?1:0);gl.uniform2f(uniform.pointer,px,py);gl.drawArrays(gl.POINTS,0,quality);
 // The same interpolated vertices connect sparse neighbouring population samples.
 if(progress>.31&&progress<.43||progress>.74&&progress<.89){gl.drawArrays(gl.LINES,0,Math.min(quality,1200));}
 }else if(ctx){ctx.clearRect(0,0,w,h);ctx.fillStyle=light?'#654096':'#c4adff';for(let i=0;i<5000;i++){const p=shapes[0],x=w*(innerWidth<650?.55:.73)+p[i*3]*h*.31,y=h*(innerWidth<650?.73:.5)+p[i*3+1]*h*.31;ctx.globalAlpha=.3+(.7*i/5000);ctx.fillRect(x,y,1.5,1.5);}}
 }
 function drawStatic(){trace();render(0);}
 const chapters=[...document.querySelectorAll('.chapter')];
 function scroll(){const rect=$('#story').getBoundingClientRect();target=Math.max(0,Math.min(1,-rect.top/($('#story').offsetHeight-innerHeight)));chapters.forEach(chapter=>{const copy=chapter.querySelector('.chapter-copy'),r=copy.getBoundingClientRect();copy.style.opacity=motion.matches?'1':String(Math.max(0,Math.min(1,(innerHeight-r.top)/180,(r.bottom-65)/150)));});const active=Math.round(target*5);document.querySelectorAll('.rail a').forEach((a,i)=>a.setAttribute('aria-current',String(i===active)));}
 addEventListener('scroll',scroll,{passive:true});addEventListener('resize',()=>{scroll();drawStatic();},{passive:true});
 addEventListener('pointermove',e=>{if(!motion.matches){px=e.clientX/innerWidth-.5;py=e.clientY/innerHeight-.5;}},{passive:true});
 const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)start();else{cancelAnimationFrame(frame);frame=0;}});observer.observe($('#story'));
 const timing=[];window.neurosushScene={get stats(){return {points:quality,frames:timing.length,fps:timing.length?1000/(timing.reduce((a,b)=>a+b,0)/timing.length):0,webgl:!!gl};}};
 function loop(ms){frame=0;if(document.hidden||!visible||motion.matches)return;const dt=last?ms-last:16.7;last=ms;if(dt<100){timing.push(dt);if(timing.length>180)timing.shift();slow=dt>25?slow+1:Math.max(0,slow-1);if(slow>40&&quality>6000){quality=Math.floor(quality*.75);slow=0;}}progress+=(target-progress)*(1-Math.exp(-dt/100));render(ms);frame=requestAnimationFrame(loop);}
 function start(){last=0;if(gl&&!frame&&!document.hidden&&visible&&!motion.matches)frame=requestAnimationFrame(loop);}
 document.addEventListener('visibilitychange',()=>{cancelAnimationFrame(frame);frame=0;if(!document.hidden)start();});motion.addEventListener('change',()=>{cancelAnimationFrame(frame);frame=0;progress=0;drawStatic();start();});
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();cancelAnimationFrame(frame);frame=0;});scroll();drawStatic();start();
})();
