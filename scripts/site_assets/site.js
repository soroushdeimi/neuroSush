'use strict';
(() => {
  const $ = s => document.querySelector(s);
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let reduced = motion.matches;
  let light = false;
  const theme = $('#theme');
  function setTheme(value) {
    light = value === 'light';
    document.documentElement.dataset.theme = value;
    theme.setAttribute('aria-label', `Switch to ${light ? 'dark' : 'light'} theme`);
    theme.setAttribute('aria-pressed', String(light));
    try { localStorage.setItem('neurosush-theme', value); } catch { /* Storage is optional. */ }
    redraw();
  }
  theme.addEventListener('click', () => {
    const change = () => setTheme(light ? 'dark' : 'light');
    if (document.startViewTransition && !reduced) document.startViewTransition(change);
    else change();
  });
  $('#copy').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText('pip install neurosush');
      $('#copy-status').textContent = 'Copied. Your next experiment is one command away.';
    } catch {
      $('#copy-status').textContent = 'Select and copy: pip install neurosush';
      const range = document.createRange(); range.selectNodeContents($('.install code'));
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    }
  });
  if (!CSS.supports('animation-timeline: view()') && !reduced && 'IntersectionObserver' in window) {
    document.documentElement.classList.add('observer-ready');
    const reveals = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.remove('pending'); reveals.unobserve(entry.target); }
    }), {threshold: .08});
    document.querySelectorAll('.reveal').forEach(el => { el.classList.add('pending'); reveals.observe(el); });
  }
  const search = $('#plate-search');
  $('#open-search').addEventListener('click', () => {
    $('.plate-list').open = true;
    search.scrollIntoView({behavior: reduced ? 'instant' : 'smooth', block:'center'});
    search.focus({preventScroll:true});
  });
  search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase(); let count = 0;
    document.querySelectorAll('.plate-list li').forEach(item => {
      item.hidden = !item.textContent.toLowerCase().includes(query);
      if (!item.hidden) count++;
    });
    $('#search-status').textContent = `${count} ${count === 1 ? 'plate' : 'plates'} found`;
  });
  const canvases = [...document.querySelectorAll('canvas')];
  const visible = new Set();
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => entries.forEach(e => {
      if (e.isIntersecting) visible.add(e.target); else visible.delete(e.target);
    }), {rootMargin:'80px'});
    canvases.forEach(c => observer.observe(c));
  } else canvases.forEach(c => visible.add(c));
  const contexts = new Map(canvases.map(c => [c, c.getContext('2d')]));
  function size(c) {
    const w = Math.round(c.clientWidth), h = Math.round(c.clientHeight), d = Math.min(devicePixelRatio || 1, 1.5);
    if (c.width !== Math.round(w*d) || c.height !== Math.round(h*d)) {
      c.width = Math.round(w*d); c.height = Math.round(h*d);
    }
    const ctx = contexts.get(c); ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0,0,w,h);
    return [ctx,w,h];
  }
  // Fixed pseudo-random seed keeps the original artwork stable across reloads.
  let seed = 7241;
  function random() { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
  const neurons = Array.from({length:94}, () => {
    const angle=random()*Math.PI*2, radius=Math.sqrt(random());
    return {x:Math.cos(angle)*radius, y:Math.sin(angle)*radius, z:random(), phase:random()*6.28};
  });
  const edges=[];
  neurons.forEach((a,i) => neurons.forEach((b,j) => {
    const distance=Math.hypot(a.x-b.x,a.y-b.y);
    if (j>i && distance<.36 && distance>.08) edges.push([i,j,random()]);
  }));
  let pointer={x:0,y:0}, scroll=0;
  const hero=$('#neural');
  hero.parentElement.addEventListener('pointermove',e => {
    const r=hero.getBoundingClientRect(); pointer={x:(e.clientX/r.width-.5)*2,y:((e.clientY-r.top)/r.height-.5)*2};
    if(reduced) drawHero(0);
  },{passive:true});
  hero.parentElement.addEventListener('pointerleave',()=> {pointer={x:0,y:0};});
  addEventListener('scroll',()=>{scroll=Math.min(1,scrollY/800);},{passive:true});
  function drawHero(t) {
    const [ctx,w,h]=size(hero), mobile=w<650;
    const cx=w*(mobile?.65:.76),cy=h*(mobile?.71:.48),scale=Math.min(w*(mobile?.62:.37),h*.65);
    const color=light?'103,65,153':'187,151,255';
    const haze=ctx.createRadialGradient(cx,cy,0,cx,cy,scale*1.3);
    haze.addColorStop(0,`rgba(${color},${light?.08:.12})`);haze.addColorStop(.6,`rgba(${color},.025)`);haze.addColorStop(1,`rgba(${color},0)`);
    ctx.fillStyle=haze;ctx.fillRect(0,0,w,h);
    const pts=neurons.map(n=>{
      const turn=t*.035+scroll*.35, x=n.x*Math.cos(turn)-n.y*.18*Math.sin(turn);
      return {x:cx+x*scale+Math.sin(t*.4+n.phase)*6+pointer.x*n.z*18,y:cy+n.y*scale*.8+Math.cos(t*.3+n.phase)*5+pointer.y*n.z*14,z:n.z};
    });
    ctx.lineWidth=.6;
    edges.forEach(([i,j,phase])=>{
      const a=pts[i],b=pts[j], bend=(neurons[i].z-.5)*45;
      ctx.strokeStyle=`rgba(${color},${.10+a.z*.16})`;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo((a.x+b.x)/2+bend,(a.y+b.y)/2-bend,b.x,b.y);ctx.stroke();
      const u=(t*.21+phase+scroll*.15)%1,v=1-u;
      if(phase>.38){const x=v*v*a.x+2*v*u*((a.x+b.x)/2+bend)+u*u*b.x,y=v*v*a.y+2*v*u*((a.y+b.y)/2-bend)+u*u*b.y;
        ctx.fillStyle=`rgba(${color},${.4+Math.sin(u*Math.PI)*.5})`;ctx.shadowColor=`rgb(${color})`;ctx.shadowBlur=8;ctx.beginPath();ctx.arc(x,y,1.3,0,6.29);ctx.fill();ctx.shadowBlur=0;}
    });
    pts.forEach((p,i)=>{
      const pulse=.5+.5*Math.sin(t*1.5+neurons[i].phase), r=1.5+p.z*3;
      const glow=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,r*9);
      glow.addColorStop(0,`rgba(${color},${.2+pulse*.15})`);glow.addColorStop(1,`rgba(${color},0)`);
      ctx.fillStyle=glow;ctx.beginPath();ctx.arc(p.x,p.y,r*9,0,6.29);ctx.fill();
      ctx.fillStyle=light?`rgba(100,60,150,${.4+p.z*.5})`:`rgba(225,207,255,${.45+p.z*.5})`;ctx.beginPath();ctx.arc(p.x,p.y,r,0,6.29);ctx.fill();
      if(p.z>.75){ctx.strokeStyle=`rgba(${color},.3)`;ctx.beginPath();ctx.arc(p.x,p.y,r+5+pulse*3,0,6.29);ctx.stroke();}
    });
  }
  function drawArt(c,t) {
    const [x,w,h]=size(c), kind=c.dataset.art, accent=light?'#76509e':'#c3a7ff';
    const colors={stdp:light?'#a45e50':'#e8aa9b',sdr:light?'#337b68':'#8bd4b5',predict:light?'#436caa':'#8fbafa',grid:light?'#967030':'#e6c288',wave:light?'#965783':'#df9dc9'};
    const ink=colors[kind]||accent; x.strokeStyle=ink;x.fillStyle=ink;x.lineWidth=1.2;
    const line=(points)=>{x.beginPath();points.forEach(([a,b],i)=>i?x.lineTo(a,b):x.moveTo(a,b));x.stroke();};
    if(kind==='stdp'||kind==='wave') {
      x.globalAlpha=.12;for(let i=1;i<6;i++)line([[0,h*i/6],[w,h*i/6]]);x.globalAlpha=1;
      for(let row=0;row<3;row++) {
        x.globalAlpha=1-row*.23;const points=[];
        for(let a=0;a<w;a+=2){const u=(a/w*4+t*.25+row*.22)%1;
          const v=kind==='stdp'?Math.exp(-u*8)*Math.sin(u*9):Math.exp(-Math.pow((u-.7)*19,2))*1.5-.18*Math.sin(u*6);
          points.push([a,h*(.3+row*.24)-v*h*.2]);}line(points);
      }x.globalAlpha=1;
    } else if(kind==='sdr'||kind==='grid') {
      const gap=Math.min(w/15,h/9);for(let r=0;r<8;r++)for(let col=0;col<13;col++){
        const on=kind==='grid'?(r+col)%3===0:Math.sin(r*31+col*17+Math.floor(t*.5))>.68;
        const a=w/2+(col-6)*gap+(r%2)*gap*.5,b=h/2+(r-3.5)*gap*.86;
        x.globalAlpha=on?.9:.12;x.beginPath();x.arc(a,b,on?3.6:1.7,0,6.29);x.fill();
        if(on&&kind==='grid'){x.globalAlpha=.15;x.beginPath();x.arc(a,b,9+Math.sin(t+r)*2,0,6.29);x.stroke();}
      }x.globalAlpha=1;
    } else if(kind==='predict') {
      for(let row=0;row<3;row++){const count=3+row*2,y=h*(.25+row*.25);
        for(let i=0;i<count;i++){const a=w/2+(i-(count-1)/2)*28;
          if(row<2){x.globalAlpha=.2;line([[a,y],[w/2+(i-(count-1)/2)*36,h*(.5+row*.25)]]);x.globalAlpha=1;}
          x.beginPath();x.arc(a,y,5+Math.sin(t+row)*1.5,0,6.29);x.fill();}
      }
      x.globalAlpha=.6;line([[w*.18,h*.28],[w*.18,h*.73]]);line([[w*.82,h*.73],[w*.82,h*.28]]);x.globalAlpha=1;
    } else {
      const cx=w*.5,cy=h*.52;for(let i=0;i<12;i++){
        const angle=i*Math.PI/6,ex=cx+Math.cos(angle)*w*.32,ey=cy+Math.sin(angle)*h*.30;
        x.globalAlpha=.55;line([[cx,cy],[(cx+ex)/2+Math.sin(i)*14,(cy+ey)/2],[ex,ey]]);
        for(let j=0;j<2;j++){const u=.6+j*.3;line([[cx+(ex-cx)*u,cy+(ey-cy)*u],[cx+(ex-cx)*u+Math.cos(angle+1)*20,cy+(ey-cy)*u+Math.sin(angle+1)*20]]);}
        const u=(t*.25+i*.11)%1;x.globalAlpha=1;x.beginPath();x.arc(cx+(ex-cx)*u,cy+(ey-cy)*u,2,0,6.29);x.fill();
      }x.globalAlpha=1;const g=x.createRadialGradient(cx,cy,0,cx,cy,40);g.addColorStop(0,light?'#a28bc7':'#dfcfff');g.addColorStop(.25,accent);g.addColorStop(1,'transparent');x.fillStyle=g;x.beginPath();x.arc(cx,cy,40,0,6.29);x.fill();
    }
  }
  const drive=$('#drive'), lif=$('#lif'); let samples=[];
  function simulate() {
    const current=Number(drive.value);$('#drive-value').textContent=current.toFixed(1)+' nA';
    let v=-65,spikes=0;samples=[];
    // README.md single-neuron equations: Euler membrane update, threshold and reset.
    for(let i=0;i<2000;i++){v+=.5/20*(-(v+65)+10*current);if(v>=-50){samples.push(0);v=-70;spikes++;}else samples.push(v);}
    $('#rate').textContent=`${spikes} SPIKES / 1 s`; drawLif(0);
  }
  function drawLif(t) {
    const [x,w,h]=size(lif), left=35, right=w-8;
    const py=v=>h-20-(v+75)/80*(h-38);
    x.font='8px monospace';x.fillStyle=light?'#625b70':'#a3a0b1';
    [-70,-50,0].forEach(v=>{x.fillText(String(v),0,py(v)+3);x.strokeStyle=v===-50?'#ad88e755':light?'#30204a15':'#ffffff10';x.setLineDash(v===-50?[3,5]:[]);x.beginPath();x.moveTo(left,py(v));x.lineTo(right,py(v));x.stroke();});x.setLineDash([]);
    x.strokeStyle=light?'#674199':'#c6acff';x.lineWidth=1.3;x.beginPath();
    samples.forEach((v,i)=>{const a=left+i/samples.length*(right-left),b=py(v);i?x.lineTo(a,b):x.moveTo(a,b);});x.stroke();
    if(!reduced){const a=left+((t*.12)%1)*(right-left);x.fillStyle='#b69aff0c';x.fillRect(left,10,a-left,h-25);x.strokeStyle='#b69aff55';x.beginPath();x.moveTo(a,10);x.lineTo(a,h-15);x.stroke();}
  }
  drive.addEventListener('input',simulate);
  function redraw(){drawHero(0);canvases.filter(c=>c.dataset.art).forEach(c=>drawArt(c,0));drawLif(0);}
  let frame=0,last=0;
  function loop(ms){frame=0;if(document.hidden||reduced)return;if(ms-last>32){last=ms;const t=ms/1000;visible.forEach(c=>{if(c===hero)drawHero(t);else if(c===lif)drawLif(t);else drawArt(c,t);});}frame=requestAnimationFrame(loop);}
  function start(){if(!frame&&!reduced&&!document.hidden)frame=requestAnimationFrame(loop);}
  document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=0;}else start();});
  motion.addEventListener('change',e=>{reduced=e.matches;if(reduced){cancelAnimationFrame(frame);frame=0;document.querySelectorAll('.pending').forEach(el=>el.classList.remove('pending'));redraw();}else start();});
  let resizeFrame;addEventListener('resize',()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(redraw);},{passive:true});
  simulate();let saved='dark';try{saved=localStorage.getItem('neurosush-theme')||'dark';}catch{/* Dark is the default. */}setTheme(saved==='light'?'light':'dark');
  // SVG motion follows the same preference as CSS and canvas.
  function svgMotion(){document.querySelectorAll('.gui-canvas svg').forEach(svg=>{if(reduced)svg.pauseAnimations();else svg.unpauseAnimations();});}
  motion.addEventListener('change',svgMotion);svgMotion();start();
})();
