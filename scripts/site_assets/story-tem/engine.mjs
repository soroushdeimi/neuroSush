const TAU = Math.PI * 2;
const clamp = x => Math.max(0, Math.min(1, x));
export function cameraAt(t, keys, out) {
  if (!Number.isFinite(t)) throw RangeError('Camera time must be finite.');
  t = clamp(t);
  let i = 0;
  while (i < keys.length - 2 && t > keys[i + 1].t) i++;
  const a = keys[i], b = keys[Math.min(i + 1, keys.length - 1)];
  const u = a === b ? 0 : clamp((t - a.t) / (b.t - a.t));
  for (const key of ['x', 'y', 'log']) out[key] = a[key] + (b[key] - a[key]) * u;
  out.index = i; out.u = u;
  return out;
}
// Teaching geometry only; these fields are not fitted model activity.
export function fieldAt(x,y,grid=false){
  if(!Number.isFinite(x)||!Number.isFinite(y))throw RangeError('Field coordinates must be finite.');
  if(!grid)return Math.exp(-((x-.48)**2+(y-.52)**2)/.018);
  let value=0;
  for(let row=-1;row<5;row++)for(let col=-1;col<5;col++){
    const dx=x-(col*.34+(row%2)*.17),dy=y-row*.294;
    value=Math.max(value,Math.exp(-(dx*dx+dy*dy)/.005));
  }
  return value;
}
export function validateReplay(run){
  if(!Array.isArray(run?.worlds)||!run.worlds.length)throw Error('Replay has no worlds');
  for(const world of run.worlds){
    const n=world.size;
    if(!Number.isInteger(n)||n<2||world.sensory?.length!==n*n)throw Error('Invalid replay arena');
    if(!Array.isArray(world.walk)||!world.walk.length||world.walk.some(p=>p.length!==2||p.some(v=>!Number.isInteger(v)||v<0||v>=n)))throw Error('Invalid recorded position');
    for(const stream of ['g','p'])if(!world.rate_maps?.[stream]?.length||world.rate_maps[stream].some(row=>row.length!==n*n||row.some(v=>!Number.isFinite(v))))throw Error('Invalid recorded activity');
  }
  return run;
}
export class ZoomEngine {
  constructor(canvas, overlay, panels, facts, onState) {
    Object.assign(this, {canvas, overlay, panels, facts, onState});
    this.ctx = canvas.getContext('2d', {alpha:false});
    this.keys = panels.map((p,i)=>({t:panels.length===1?0:i/(panels.length-1),x:.5+(p.chapter%2)*.025,y:.5,log:[0,-.15,-.5,-.3,-.5,-.8,0,-.1,-.25,0,.1][p.chapter]}));
    this.target=0; this.t=0; this.active=0; this.playing=true; this.phase=0; this.raf=0; this.drawCount=0; this.camera={}; this.reduced=false;
    this.step=this.step.bind(this); this.resize=this.resize.bind(this);
    this.fieldPointer=null;
    canvas.addEventListener('pointerdown',event=>this.setFieldPointer(event));
    canvas.addEventListener('pointermove',event=>this.setFieldPointer(event));
    canvas.addEventListener('pointerleave',event=>{if(event.pointerType==='mouse'){this.fieldPointer=null;this.wake();}});
    this.walk=[[3,3]];let seed=17;
    for(let i=0;i<300;i++){seed=(seed*1664525+1013904223)>>>0;const [x,y]=this.walk.at(-1),d=seed%5;this.walk.push([Math.max(0,Math.min(facts.arena_side-1,x+(d===0?1:d===1?-1:0))),Math.max(0,Math.min(facts.arena_side-1,y+(d===2?1:d===3?-1:0)))]);}
    this.resize(); addEventListener('resize',this.resize);
  }
  resize(){this.w=innerWidth;this.h=innerHeight;this.mobile=this.w<700;this.dpr=Math.min(devicePixelRatio||1,2);this.canvas.width=Math.round(this.w*this.dpr);this.canvas.height=Math.round(this.h*this.dpr);this.overlay.setAttribute('viewBox',`0 0 ${this.w} ${this.h}`);this.cx=this.w*(this.mobile?.5:.7);this.cy=this.h*(this.mobile?.22:.46);this.size=Math.min(this.w*(this.mobile?.82:.46),this.h*(this.mobile?.29:.58));this.overlay.querySelector('#caption').setAttribute('transform',`translate(${this.mobile?22:this.w*.49} ${this.mobile?this.h*.375:this.h*.84})`);const y=this.mobile?62:100;this.overlay.querySelector('#journey-line').setAttribute('d',`M${this.mobile?22:this.w*.49} ${y}H${this.w-22}`);this.updateTheme();}
  updateTheme(){const s=getComputedStyle(document.documentElement);this.colors=['--bg','--text','--text-3','--accent-text','--inh','--trace'].map(k=>s.getPropertyValue(k).trim());this.wake();}
  setTarget(t){if(!Number.isFinite(t))throw RangeError('Scroll must be finite.');this.target=clamp(t);this.wake();}
  setPlaying(value){this.playing=value;if(!value)this.t=this.target;this.lastTime=0;this.wake();}
  setFieldPointer(event){
    if(this.active!==1||this.reduced||this.panels[1]?.scene!=='fields')return;
    const rect=this.canvas.getBoundingClientRect();
    const x=(event.clientX-rect.left)*this.w/rect.width,y=(event.clientY-rect.top)*this.h/rect.height;
    const key=this.keys[1],scale=10**((key.log-this.camera.log)*.24);
    const localX=(x-this.cx-(key.x-this.camera.x)*this.size)/scale,localY=(y-this.cy)/scale;
    const width=this.size*.42,top=-this.size*.23;
    for(const left of [-this.size*.48,this.size*.05]){
      if(localX>=left&&localX<=left+width&&localY>=top&&localY<=top+width){
        this.fieldPointer={x:(localX-left)/width,y:(localY-top)/width};
        this.wake();return;
      }
    }
  }
  wake(){if(!this.raf&&!this.reduced&&!document.hidden)this.raf=requestAnimationFrame(this.step);}
  step(now){this.raf=0;if(this.reduced||document.hidden)return;const delta=this.target-this.t;this.t=Math.abs(delta)<.000001?this.target:this.t+delta*.16;if(this.playing&&this.lastTime)this.phase+=(now-this.lastTime)/1000;this.lastTime=now;cameraAt(this.t,this.keys,this.camera);this.draw();this.onState(this.active);if(this.playing||Math.abs(this.target-this.t)>.000001)this.wake();}
  line(x,y,xx,yy,color,width=1){const c=this.ctx;c.beginPath();c.moveTo(x,y);c.lineTo(xx,yy);c.strokeStyle=color;c.lineWidth=width;c.stroke();}
  circle(x,y,r,color,fill=true){const c=this.ctx;c.beginPath();c.arc(x,y,r,0,TAU);if(fill){c.fillStyle=color;c.fill();}else{c.strokeStyle=color;c.stroke();}}
  text(s,x,y,color,size=11){this.ctx.fillStyle=color;this.ctx.font=`${size}px IBM Plex Mono,monospace`;this.ctx.fillText(s,x,y);}
  draw(){this.drawCount++;const c=this.ctx;c.setTransform(this.dpr,0,0,this.dpr,0,0);c.globalAlpha=1;c.fillStyle=this.colors[0];c.fillRect(0,0,this.w,this.h);const {index,u}=this.camera,mix=u*u*(3-2*u);this.active=mix<.5?index:Math.min(index+1,this.panels.length-1);this.drawLayer(index,1-mix);if(mix)this.drawLayer(Math.min(index+1,this.panels.length-1),mix);c.globalAlpha=1;const start=this.mobile?22:this.w*.49;this.overlay.querySelector('#journey-dot').setAttribute('cx',start+(this.w-22-start)*this.t);this.overlay.querySelector('#journey-dot').setAttribute('cy',this.mobile?62:100);this.overlay.querySelector('#journey-label').setAttribute('x',start);this.overlay.querySelector('#journey-label').setAttribute('y',this.mobile?84:119);this.overlay.querySelector('#journey-label').textContent=this.panels[this.active].label;}
  drawLayer(index,opacity){if(!opacity)return;const c=this.ctx;c.save();c.globalAlpha=opacity;const key=this.keys[index];c.translate(this.cx+(key.x-this.camera.x)*this.size,this.cy);const scale=10**((key.log-this.camera.log)*.24);c.scale(scale,scale);const scene=this.panels[index].scene;if(scene==='prediction')this.prediction();else if(scene==='predictive-grid')this.predictiveGrid();else if(scene==='ambiguity')this.ambiguity();else if(scene==='phase-result')this.phaseResult();else if(scene==='fields')this.fields(this.panels[index].single);else if(scene==='circuit')this.circuit();else if(scene==='timing')this.timing();else if(scene==='mechanisms')this.mechanisms();else if(scene==='replay')this.replay();else if(scene==='margins')this.margins();else if(scene==='limits')this.limits();else if(scene==='discrepancies')this.discrepancies();else if(scene==='standardscore')this.standardscore();else if(scene==='results')this.results(this.panels[index]);else if(scene==='arena')this.arena();else if(scene==='neurons')this.neurons();else if(scene==='plasticity')this.plasticity();else if(scene==='binding')this.binding();else if(scene==='remapping')this.remapping(this.panels[index].square);else this.ingredients();c.restore();}
  prediction(){const s=this.size,c=this.ctx,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4];
    const paths=[['MEC II + action','MEC III','CA1 + memory','MEC V','senses'],['senses','MEC II / LEC','DG','CA3','CA1']];
    for(let path=0;path<2;path++){
      const x=(path?.25:-.25)*s,color=path?gold:blue;
      this.text(path?'INFERENCE':'PREDICTION',x-38,-s*.46,color,this.mobile?8:10);
      for(let i=0;i<5;i++){
        const y=(-.32+i*.17)*s;
        if(i<4){this.line(x,y,x,y+s*.17,muted);const q=(this.phase*.5)%1;this.circle(x,y+q*s*.17,3,color);}
        c.fillStyle=this.colors[0];c.fillRect(x-42,y-12,84,24);c.strokeStyle=muted;c.strokeRect(x-42,y-12,84,24);this.text(paths[path][i],x-38,y+4,color,this.mobile?8:10);
      }
    }
  }
  replay(){const run=this.replays[this.replayVariant],world=run.worlds[this.replayWorld],s=this.size,c=this.ctx,n=world.size,w=s*.4,top=-s*.22,at=Math.floor(this.phase*4)%world.walk.length;
    for(let kind=0;kind<2;kind++){
      const left=kind?s*.06:-s*.47,map=world.rate_maps.g[0],max=Math.max(...map.map(Math.abs))||1;
      this.text(kind?'g · SIGNED ACTIVITY':'RECORDED PATH',left,top-20,this.colors[2],this.mobile?8:10);
      for(let y=0;y<n;y++)for(let x=0;x<n;x++){
        const value=kind?map[y*n+x]/max:world.sensory[y*n+x]/this.facts.sensory_neurons,alpha=c.globalAlpha;
        c.globalAlpha=alpha*(.12+.8*Math.abs(value));c.fillStyle=kind&&value<0?this.colors[3]:this.colors[4];c.fillRect(left+x/n*w,top+y/n*w,w/n-1,w/n-1);c.globalAlpha=alpha;
      }
      if(!kind){c.beginPath();for(let i=Math.max(0,at-25);i<=at;i++){const [x,y]=world.walk[i],xx=left+(x+.5)/n*w,yy=top+(y+.5)/n*w;i===Math.max(0,at-25)?c.moveTo(xx,yy):c.lineTo(xx,yy);}c.strokeStyle=this.colors[1];c.stroke();const [x,y]=world.walk[at];this.circle(left+(x+.5)/n*w,top+(y+.5)/n*w,4,this.colors[1]);}
    }
    this.text('WHOLE-WALK MAP · NOT INSTANTANEOUS',-s*.45,s*.35,this.colors[2],this.mobile?8:10);
  }
  margins(){const s=this.size,c=this.ctx,f=this.facts,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4],left=s*.06,scale=s*.9;
    this.text('MARGIN OVER CONTROL UPPER TAIL (RSA)',-s*.45,-s*.39,muted,this.mobile?8:10);
    this.line(left,-s*.3,left,s*.3,muted);const threshold=left+f.margin_threshold*scale;this.line(threshold,-s*.3,threshold,s*.3,gold);
    this.text('RULE',threshold-12,-s*.32,gold,8);
    const ids=['v2_s0','v2_s1','v2_s2','v1_s0','v1_s1','v1_s2'];
    for(let i=0;i<ids.length;i++){const y=(-.23+i*.09)*s,v=f[ids[i]+'_margin'];this.text(ids[i].replaceAll('_',' '),-s*.46,y+4,muted,this.mobile?8:10);this.line(left,y,left+v*scale,y,blue,4);this.circle(left+v*scale,y,3,blue);this.text((v>=0?'+':'')+v.toFixed(3),s*.25,y+4,muted,this.mobile?8:10);}
    this.text('RULE NOT MET',-s*.18,s*.42,gold,11);
  }
  limits(){const s=this.size,muted=this.colors[2],gold=this.colors[3];
    for(const [i,label] of ['GEOMETRY','PHASE DIVERSITY','TIME RESOLUTION','LEARNING RULE'].entries()){
      const y=(-.3+i*.2)*s;this.circle(-s*.38,y,7,gold,false);this.line(-s*.3,y,s*.4,y,muted);this.text(label,-s*.28,y-12,muted,10);
    }
  }
  discrepancies(){const s=this.size,f=this.facts,muted=this.colors[2],gold=this.colors[3];
    for(const [i,title,a,b] of [[0,'GRID CELLS (%)',`${f.grid_mean} · Table 2`,`≈ ${f.fig7_approx} · Fig. 7`],[1,'DENTATE NEURONS',`${f.dg_table} · Table 1`,`${f.dg_discussion} · Discussion`]]){
      const y=(-.3+i*.45)*s;this.text(title,-s*.43,y,muted,10);this.text(a,-s*.43,y+s*.1,gold,this.mobile?12:18);this.text(b,-s*.43,y+s*.2,this.colors[4],this.mobile?12:18);
    }
  }
  standardscore(){const s=this.size,c=this.ctx,f=this.facts,w=s*.4,top=-s*.25;
    for(const [i,name] of ['square','hex'].entries()){
      const left=i?s*.06:-s*.47,map=f.ideal_maps[name],n=map.length;
      this.text(name==='hex'?'IDEAL HEXAGONAL':'IDEAL SQUARE',left,top-20,this.colors[2],this.mobile?8:10);
      for(let y=0;y<n;y++)for(let x=0;x<n;x++){
        const v=map[y][x]/(name==='square'?4:1),alpha=c.globalAlpha;c.globalAlpha=alpha*(.07+.9*v);c.fillStyle=this.colors[4];c.fillRect(left+x/n*w,top+y/n*w,w/n+.2,w/n+.2);c.globalAlpha=alpha;
      }
      this.text(f[name==='hex'?'ideal_hex_score':'ideal_square_score'].toFixed(3),left,top+w+28,this.colors[3],this.mobile?13:20);
    }
    this.text('STANDARD HEX SCORE · OUR COMPUTATION',-s*.45,s*.4,this.colors[2],this.mobile?8:10);
  }
  results(panel){const c=this.ctx,s=this.size,f=this.facts,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4];
    if(panel.kind==='grids'){
      const w=s*.43,left=-w/2,top=-s*.33,mix=(1-Math.cos(this.phase*.7))/2;
      this.text('FIELD FORMATION · SCHEMATIC',-s*.35,top-20,muted,this.mobile?8:10);
      for(let y=0;y<32;y++)for(let x=0;x<32;x++){
        const v=Math.exp(-((Math.sin(x/32*Math.PI*3))**2+(Math.sin(y/32*Math.PI*3))**2)*5),alpha=c.globalAlpha;
        c.globalAlpha=alpha*(.08+mix*v*.9);c.fillStyle=blue;c.fillRect(left+x/32*w,top+y/32*w,w/32+.3,w/32+.3);c.globalAlpha=alpha;
      }
      this.text(`${f.grid_mean} ± ${f.grid_sd}% · ${f.paper_runs} RUNS`,-s*.3,s*.28,gold,this.mobile?11:16);
      this.text('MEAN ± SD · REPORTED',-s*.3,s*.36,muted,9);
      return;
    }
    const rows=[['Baseline','grid_mean','grid_sd'],['No feedback','no_feedback','no_feedback_sd'],['No theta','no_theta','no_theta_sd'],['No STDP','no_stdp','no_stdp_sd'],['No sparsity','no_sparsity','no_sparsity_sd'],['MSE loss','mse','mse_sd'],['No gain','no_gain','no_gain_sd']];
    const left=-s*.1,width=s*.5,top=-s*.33,spacing=s*.1;
    this.text('GRID CELLS (%) · MEAN ± SD',-s*.45,top-25,muted,this.mobile?8:10);
    for(let i=0;i<rows.length;i++){
      const [label,id,sd]=rows[i],y=top+i*spacing,value=f[id],spread=f[sd];
      this.text(label,-s*.46,y+4,muted,this.mobile?8:10);this.line(left,y,left+width*value/100,y,i?blue:gold,7);
      this.line(left+width*Math.max(0,value-spread)/100,y,left+width*(value+spread)/100,y,this.colors[1]);
      for(const v of [Math.max(0,value-spread),value+spread])this.line(left+width*v/100,y-4,left+width*v/100,y+4,this.colors[1]);
      this.text(String(value),left+width*.86,y+4,muted,this.mobile?8:10);
    }
  }
  arena(){const s=this.size,c=this.ctx,n=this.facts.arena_side,w=s*.7,left=-w/2,top=-w/2,cell=w/n,muted=this.colors[2];
    for(let y=0;y<n;y++)for(let x=0;x<n;x++){
      const identity=(x*7+y*11)%this.facts.sensory_neurons,alpha=c.globalAlpha;
      c.globalAlpha=alpha*(.15+.6*identity/this.facts.sensory_neurons);c.fillStyle=identity%2?this.colors[3]:this.colors[4];c.fillRect(left+x*cell,top+y*cell,cell-1,cell-1);c.globalAlpha=alpha;
    }
    const at=Math.floor(this.phase*4)%(this.walk.length-1),u=(this.phase*4)%1;
    c.beginPath();for(let i=Math.max(0,at-20);i<=at;i++){const [x,y]=this.walk[i],xx=left+(x+.5)*cell,yy=top+(y+.5)*cell;i===Math.max(0,at-20)?c.moveTo(xx,yy):c.lineTo(xx,yy);}c.strokeStyle=this.colors[1];c.stroke();
    const a=this.walk[at],b=this.walk[at+1];this.circle(left+(a[0]+(b[0]-a[0])*u+.5)*cell,top+(a[1]+(b[1]-a[1])*u+.5)*cell,5,this.colors[1]);
    this.text(`${n} × ${n} POSITIONS · ${this.facts.sensory_neurons} SENSORY LABELS`,left,top+w+25,muted,this.mobile?8:10);
  }
  ambiguity(){const s=this.size,c=this.ctx,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4],ink=this.colors[1];
    const width=s*.27,top=-s*.15,cell=width/6,step=Math.floor(this.phase*1.2)%6;
    this.text('SAME VIEW CAN MEAN DIFFERENT PLACES',-s*.45,-s*.39,muted,this.mobile?9:12);
    for(let mode=0;mode<3;mode++){
      const left=(-.45+mode*.315)*s;
      this.text(['UNIQUE CUES','SOME REPEATS','MANY REPEATS'][mode],left,-s*.21,mode===1?gold:muted,this.mobile?8:10);
      for(let y=0;y<6;y++)for(let x=0;x<6;x++){
        const identity=mode===0?y*6+x:mode===1?(x*3+y*5)%12:(x+y)%3;
        const match=identity===(mode===0?14:1);
        c.fillStyle=match?gold:blue;c.globalAlpha*=match?.9:.12+((identity%5)*.045);
        c.fillRect(left+x*cell,top+y*cell,cell-2,cell-2);
        c.globalAlpha/=match?.9:.12+((identity%5)*.045);
      }
      c.strokeStyle=muted;c.strokeRect(left,top,width,width);
      const row=mode===0?2:mode===1?2:1,col=mode===0?2:mode===1?1:0;
      this.circle(left+(col+.5)*cell,top+(row+.5)*cell,4,ink);
      if(mode===1){
        const trail=[[0,5],[1,5],[2,5],[2,4],[2,3],[3,3],[4,3]];
        c.beginPath();for(let i=0;i<=step;i++){const [x,y]=trail[i],xx=left+(x+.5)*cell,yy=top+(y+.5)*cell;i?c.lineTo(xx,yy):c.moveTo(xx,yy);}c.strokeStyle=ink;c.lineWidth=2;c.stroke();
        const [x,y]=trail[step];this.circle(left+(x+.5)*cell,top+(y+.5)*cell,5,ink);
      }
    }
    this.text('MOVEMENT HISTORY HELPS DISAMBIGUATE',-s*.43,s*.34,gold,this.mobile?8:11);
    this.text('ILLUSTRATION · NOT MEASURED GRIDNESS',-s*.43,s*.43,muted,this.mobile?8:10);
  }
  predictiveGrid(){const s=this.size,c=this.ctx,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4],ink=this.colors[1];
    const path=[[1,5],[2,5],[3,5],[3,4],[4,4],[5,4],[5,3],[5,2],[4,2],[4,1],[3,1],[2,1],[1,1],[1,2],[1,3],[1,4]];
    const at=Math.floor(this.phase*1.5)%path.length,from=path[at],to=path[(at+1)%path.length],w=s*.36,top=-s*.2,n=this.facts.arena_side,cell=w/n;
    this.text('CURRENT POSITION',-s*.44,-s*.34,blue,this.mobile?9:12);
    this.text('ONE STEP AHEAD',s*.05,-s*.34,gold,this.mobile?9:12);
    for(let side=0;side<2;side++){
      const left=(side?.05:-.44)*s;
      for(let y=0;y<n;y++)for(let x=0;x<n;x++){
        const active=(side?to:from),distance=(x-active[0])**2+(y-active[1])**2;
        c.fillStyle=side?gold:blue;c.globalAlpha*=.12+.7*Math.exp(-distance/1.2);
        c.fillRect(left+x*cell,top+y*cell,cell-2,cell-2);
        c.globalAlpha/=.12+.7*Math.exp(-distance/1.2);
      }
      c.strokeStyle=muted;c.strokeRect(left,top,w,w);
      const [x,y]=side?to:from;this.circle(left+(x+.5)*cell,top+(y+.5)*cell,5,ink);
    }
    this.line(-s*.035,0,s*.015,0,muted,2);this.line(s*.015,0,s*.001,-s*.014,muted,2);this.line(s*.015,0,s*.001,s*.014,muted,2);
    this.text('MEC II · PRESENT',-s*.44,s*.28,muted,this.mobile?9:11);
    this.text('MEC III · PREDICTED',s*.05,s*.28,muted,this.mobile?9:11);
    this.text('PREDICTION TARGET IS SUPPLIED IN TRAINING',-s*.44,s*.42,muted,this.mobile?8:10);
  }
  phaseResult(){const s=this.size,c=this.ctx,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4],left=-s*.43,width=s*.86,cycles=6,active=Math.floor(this.phase*.9)%cycles;
    this.text('GAIN + IMPOSED THETA',left,-s*.4,muted,this.mobile?9:12);
    for(let k=0;k<cycles;k++){
      const x=left+k*width/cycles,opacity=k===active?.1:.025;c.fillStyle=gold;c.globalAlpha*=opacity;c.fillRect(x,-s*.31,width/cycles,s*.65);c.globalAlpha/=opacity;
      this.line(x,-s*.31,x,s*.34,muted,.7);
    }
    this.line(left+width,-s*.31,left+width,s*.34,muted,.7);
    c.beginPath();for(let i=0;i<=240;i++){const q=i/240,y=-s*.22+Math.sin(q*TAU*cycles)*s*.055;i?c.lineTo(left+q*width,y):c.moveTo(left+q*width,y);}c.strokeStyle=muted;c.lineWidth=1.5;c.stroke();
    this.text('MEC II · PRECESSION',left,-s*.05,gold,this.mobile?9:12);
    this.text('MEC III · LOCKING',left,s*.2,blue,this.mobile?9:12);
    for(let k=0;k<cycles;k++){
      const pre=(k+.83-k*.1)/cycles,lock=(k+.58)/cycles;
      this.line(left+pre*width,s*.02,left+pre*width,s*.12,gold,k===active?4:2);
      this.line(left+lock*width,s*.25,left+lock*width,s*.35,blue,k===active?4:2);
    }
    this.text(`${this.facts.precession_mean}% OF II CELLS`,left,s*.44,gold,this.mobile?9:11);
    this.text(`${this.facts.locking_mean}% OF III CELLS`,s*.05,s*.44,blue,this.mobile?9:11);
  }
  neurons(){const s=this.size,c=this.ctx,left=-s*.44,w=s*.88,gold=this.colors[3],blue=this.colors[4],muted=this.colors[2];
    this.text('ACCUMULATE → THRESHOLD → RESET',left,-s*.35,gold,this.mobile?8:10);
    this.line(left,-s*.19,left+w,-s*.19,muted);
    c.beginPath();for(let i=0;i<=240;i++){const x=i/240,u=(x*4+this.phase*.2)%1,v=(1-Math.exp(-u*2))/(1-Math.exp(-2)),y=s*.08-v*s*.27;i?c.lineTo(left+x*w,y):c.moveTo(left+x*w,y);}c.strokeStyle=gold;c.stroke();
    this.text('THETA INHIBITION · IMPOSED',left,s*.25,blue,10);
    c.beginPath();for(let i=0;i<=180;i++){const q=i/180,y=s*.38+(1+Math.sin(q*TAU*3-this.phase))*s*.045;i?c.lineTo(left+q*w,y):c.moveTo(left+q*w,y);}c.strokeStyle=blue;c.stroke();
  }
  plasticity(){const s=this.size,c=this.ctx,left=-s*.43,w=s*.86,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4],lag=Math.sin(this.phase*.65);
    this.text('POST FIRST ← TIMING → PRE FIRST',left,-s*.34,muted,this.mobile?8:10);
    this.line(left,0,left+w,0,muted);this.line(0,-s*.25,0,s*.25,muted);
    for(const sign of [-1,1]){c.beginPath();for(let i=0;i<=80;i++){const q=i/80,x=sign*q*s*.43,y=-sign*Math.exp(-q*4)*s*.24;i?c.lineTo(x,y):c.moveTo(x,y);}c.strokeStyle=sign>0?gold:blue;c.stroke();}
    const value=Math.sign(lag)*Math.exp(-Math.abs(lag)*4);this.circle(lag*s*.43,-value*s*.24,4,gold);
    this.text(value>0?'PRE BEFORE POST · STRENGTHEN':'POST BEFORE PRE · WEAKEN',left,s*.34,muted,this.mobile?8:10);
    this.line(left,s*.44,left+w,s*.44,value>0?gold:blue,2+Math.abs(value)*7);
  }
  binding(){const s=this.size,gold=this.colors[3],blue=this.colors[4],muted=this.colors[2],q=(this.phase*.25)%1;
    const nodes=[['g · STRUCTURE',-.4,-.25],['x · CONTENT',-.4,.22],['p · CONJUNCTION',.02,0],['M · MEMORY',.36,0]];
    for(const [a,b] of [[0,2],[1,2],[2,3]]){const n=nodes[a],m=nodes[b];this.line(n[1]*s,n[2]*s,m[1]*s,m[2]*s,muted);this.circle((n[1]+(m[1]-n[1])*q)*s,(n[2]+(m[2]-n[2])*q)*s,4,a===1?blue:gold);}
    for(const [label,x,y] of nodes){this.circle(x*s,y*s,12,gold,false);this.text(label,x*s-40,y*s+32,muted,this.mobile?8:10);}
    this.text('REUSABLE RULES + CURRENT EXPERIENCE',-s*.44,s*.43,muted,this.mobile?8:10);
  }
  remapping(square=false){const s=this.size,c=this.ctx,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4];
    const mix=(1-Math.cos(this.phase*.7))/2,w=s*.4,top=-s*.22;
    for(let kind=0;kind<2;kind++){
      const left=kind?s*.06:-s*.47;this.text(kind?'PLACE · REMAPS':'GRID · REALIGNS',left,top-20,kind?gold:blue,this.mobile?8:10);
      for(let row=0;row<32;row++)for(let col=0;col<32;col++){
        const x=(col+.5)/32,y=(row+.5)/32,v=kind?Math.exp(-((x-(.25+mix*.5))**2+(y-(.7-mix*.4))**2)/.018):square?Math.exp(-((Math.sin((x+mix*.17)*Math.PI*3))**2+(Math.sin((y+mix*.12)*Math.PI*3))**2)*5):fieldAt((x+mix*.17)%1,(y+mix*.12)%1,true);
        const alpha=c.globalAlpha;c.globalAlpha=alpha*(.08+v*.85);c.fillStyle=kind?gold:blue;c.fillRect(left+col/32*w,top+row/32*w,w/32+.3,w/32+.3);c.globalAlpha=alpha;
      }
      c.strokeStyle=muted;c.strokeRect(left,top,w,w);
    }
    this.text(mix<.5?'ROOM A → ROOM B':'ROOM B → ROOM A',-s*.25,s*.35,muted,10);
  }
  mechanisms(){const s=this.size,c=this.ctx,left=-s*.44,width=s*.88,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4];
    this.text('OSCILLATORY INTERFERENCE',left,-s*.36,gold,10);
    for(let kind=0;kind<3;kind++){
      c.beginPath();for(let i=0;i<=240;i++){const q=i/240,a=Math.sin(q*TAU*8-this.phase),b=Math.sin(q*TAU*9-this.phase),yy=-s*.17+(kind===2?(a+b)*.5:kind===0?a:b)*s*.08; i?c.lineTo(left+q*width,yy):c.moveTo(left+q*width,yy);}c.strokeStyle=kind===2?gold:kind?blue:muted;c.lineWidth=kind===2?2:1;c.stroke();
    }
    this.text('CONTINUOUS ATTRACTOR · MOVING BUMP',left,s*.12,blue,10);
    for(let row=0;row<5;row++)for(let col=0;col<18;col++){
      const center=8+5*Math.sin(this.phase*.6),v=Math.exp(-((col-center)**2+(row-2)**2)/5);
      this.circle(left+col/17*width,s*.23+row*s*.035,1.3+v*4,v>.3?blue:muted);
    }
  }
  circuit(){const s=this.size,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4];
    const nodes=[['MEC II',-.4,-.26],['MEC III',-.4,0],['LEC',-.4,.26],['DG',-.08,-.13],['CA3',.18,-.13],['CA1',.43,-.13]];
    for(const [a,b] of [[0,3],[0,4],[2,3],[2,4],[2,5],[3,4],[4,5],[1,5]]){
      const from=nodes[a],to=nodes[b],x=from[1]*s,y=from[2]*s,xx=to[1]*s,yy=to[2]*s,q=(this.phase*.4+a*.13)%1;
      this.line(x,y,xx,yy,a===1?blue:muted);this.circle(x+(xx-x)*q,y+(yy-y)*q,3,a===1?blue:gold);
    }
    for(const [label,x,y] of nodes){this.ctx.fillStyle=this.colors[0];this.ctx.fillRect(x*s-28,y*s-13,56,26);this.ctx.strokeStyle=muted;this.ctx.strokeRect(x*s-28,y*s-13,56,26);this.text(label,x*s-23,y*s+4,gold,10);}
    this.text('ENTORHINAL → HIPPOCAMPAL',-s*.4,s*.43,muted,10);
  }
  timing(){const s=this.size,muted=this.colors[2],gold=this.colors[3],blue=this.colors[4],left=-s*.44,width=s*.88;
    for(let row=0;row<2;row++){
      const base=(row? .24:-.2)*s;
      this.text(row?'MEC III · PHASE LOCKING':'MEC II · PHASE PRECESSION',left,base-s*.14, row?blue:gold,10);
      this.ctx.beginPath();for(let i=0;i<=240;i++){const q=i/240,xx=left+q*width,yy=base+Math.sin(q*TAU*6-this.phase*TAU*.3)*s*.055;i?this.ctx.lineTo(xx,yy):this.ctx.moveTo(xx,yy);}this.ctx.strokeStyle=muted;this.ctx.stroke();
      for(let k=0;k<6;k++){
        const phase=row?.55:.85-k*.11,q=((k+phase)/6+this.phase*.05)%1,x=left+q*width;
        this.line(x,base-s*.07,x,base+s*.07,row?blue:gold,2);
      }
    }
  }
  fields(single=false){const c=this.ctx,s=this.size,gold=this.colors[3],blue=this.colors[4],muted=this.colors[2];
    const width=s*(single?.58:.42),top=-s*.23;
    const teaching=!single&&this.active===1;
    const q=this.phase*.3,auto={x:.5+.4*Math.sin(q*1.7),y:.5+.4*Math.sin(q*2.3)};
    const position=teaching&&this.fieldPointer?this.fieldPointer:auto;
    if(teaching)this.text(this.reduced?'SCHEMATIC · IDEALIZED FIELDS':'SCHEMATIC · MOVE POINTER OR TAP',-s*.48,-s*.37,muted,this.mobile?9:12);
    for(let kind=0;kind<(single?1:2);kind++){
      const left=single?-width/2:kind?s*.05:-s*.48;
      const color=kind?blue:gold;
      this.text(kind?'GRID CELL':'PLACE CELL',left,top-18,color,this.mobile?9:12);
      for(let row=0;row<40;row++)for(let col=0;col<40;col++){
        const v=fieldAt((col+.5)/40,(row+.5)/40,!!kind);
        c.globalAlpha*=.08+.85*v;c.fillStyle=kind?blue:gold;c.fillRect(left+col/40*width,top+row/40*width,width/40+.3,width/40+.3);c.globalAlpha/=.08+.85*v;
      }
      c.strokeStyle=muted;c.strokeRect(left,top,width,width);
      const {x,y}=position,value=fieldAt(x,y,!!kind);
      this.circle(left+x*width,top+y*width,4,this.colors[1]);
      this.circle(left+x*width,top+y*width,7+value*9,color,false);
      if(teaching){
        const base=top+width;
        this.text(`RATE ${Math.round(value*100)}%`,left,base+22,color,this.mobile?9:11);
        this.line(left,base+32,left+width,base+32,muted,4);
        this.line(left,base+32,left+width*value,base+32,color,5);
        const spike=value>.3&&Math.sin(this.phase*12+kind*2)>.65;
        this.circle(left+5,base+53,spike?6:4,color,spike);
        this.text('EXAMPLE SPIKE',left+16,base+57,muted,this.mobile?8:10);
      }else this.text(kind?'REPEATED FIELDS':'LOCAL FIELD',left,top+width+24,muted,9);
    }
    if(teaching)this.text(this.mobile?'SCHEMATIC · NOT MODEL DATA':'ILLUSTRATIVE RATES & SPIKES · NOT MODEL DATA',-s*.48,this.mobile?s*.48:s*.42,muted,this.mobile?9:10);
  }
  ingredients(){const c=this.ctx,s=this.size,ink=this.colors[1],muted=this.colors[2],gold=this.colors[3],blue=this.colors[4];const left=-s*.45,w=s*.9;
    this.text('INPUT → LEAK → SPIKE',left,-s*.39,muted,10);
    this.circle(-s*.28,-s*.13,s*.1,gold,false);
    for(let k=0;k<5;k++){const a=k/5*TAU;this.line(-s*.28+Math.cos(a)*s*.1,-s*.13+Math.sin(a)*s*.1,-s*.28+Math.cos(a)*s*.17,-s*.13+Math.sin(a)*s*.17,muted);}
    this.line(-s*.18,-s*.13,s*.4,-s*.13,gold);
    for(let k=0;k<7;k++){const q=(this.phase*.32+k/7)%1;this.circle(-s*.18+q*s*.58,-s*.13,2+2*Math.sin(q*Math.PI),gold);}
    const top=s*.11;this.text('THETA: FIRING WINDOWS',left,top,muted,10);c.beginPath();for(let i=0;i<=150;i++){const q=i/150,yy=top+s*.13+Math.sin(q*TAU*3-this.phase*TAU)*s*.05;i?c.lineTo(left+q*w,yy):c.moveTo(left+q*w,yy);}c.strokeStyle=blue;c.stroke();
    this.text('TIMING CHANGES CONNECTIONS',left,s*.39,muted,10);
    const beat=(Math.sin(this.phase*2)+1)/2;this.line(left,s*.48,left+w,s*.48,gold,1+beat*5);this.circle(left,s*.48,4,ink);this.circle(left+w,s*.48,4,ink);
  }
}
