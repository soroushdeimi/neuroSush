'use strict';
window.NeuroInstruments = (() => {
  let voltage=-65, accumulator=0, age=1;
  const history=[];
  const style=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function step(){
    voltage+=.5/20*(-(voltage+65)+20);
    const spike=voltage>=-50;
    history.push(spike?0:voltage);
    if(spike){voltage=-70;age=0;}
    if(history.length>401)history.shift();
  }
  for(let i=0;i<401;i++)step();
  age=1;
  function advance(dt){accumulator+=Math.min(dt,50);age+=dt/1000;while(accumulator>=.5){accumulator-=.5;step();}}
  function plot(canvas, type, time=0){
    if(!canvas)return;
    const d=Math.min(devicePixelRatio,2),w=canvas.clientWidth,h=canvas.clientHeight;
    canvas.width=w*d;canvas.height=h*d;
    const c=canvas.getContext('2d');c.scale(d,d);
    const left=44,right=w-16,top=16,bottom=h-28;
    const x=t=>left+t*(right-left), y=t=>bottom-t*(bottom-top);
    c.font='12px '+getComputedStyle(document.documentElement).getPropertyValue('--font-sans');
    c.fillStyle=style('--text-3');c.strokeStyle=style('--rule-strong');c.lineWidth=1;
    c.beginPath();c.moveTo(left,top);c.lineTo(left,bottom);c.lineTo(right,bottom);c.stroke();
    const line=(values,color,dashed=false)=>{
      c.strokeStyle=style(color);c.lineWidth=1.5;c.setLineDash(dashed?[4,3]:[]);c.beginPath();
      values.forEach(([a,b],i)=>{i?c.lineTo(x(a),y(b)):c.moveTo(x(a),y(b));});c.stroke();c.setLineDash([]);
    };
    if(type==='lif'){
      for(const v of [-75,-60,-45])c.fillText(String(v),4,y((v+75)/30)+4);
      c.fillText('mV',4,12);c.fillText('0',left,bottom+20);c.fillText('200 ms',right-44,bottom+20);
      line([[0,25/30],[1,25/30]],'--text-3',true);
      c.fillText('−50',right-30,y(25/30)-4);
      line(history.map((v,i)=>[i/400,Math.min(1,(v+75)/30)]),'--text');
      c.fillStyle=style('--spike');history.forEach((v,i)=>{if(v===0)c.fillRect(x(i/400)-1,top,2,4);});
    }else if(type==='stdp'){
      c.fillText('Δw',4,12);c.fillText('−50',left,bottom+20);c.fillText('Δt (ms)',right-48,bottom+20);
      line([[0,1/3],[1,1/3]],'--rule-strong');
      for(const sign of [-1,1]){
        const values=[];
        for(let ms=1;ms<=50;ms++)values.push([.5+sign*ms/100,(sign>0?.01:-.005)*Math.exp(-ms/20)/.015+1/3]);
        line(values,sign>0?'--exc':'--inh',sign<0);
      }
      c.fillText('0',x(.5)-3,bottom+20);c.fillText('+50',right-24,bottom+5);
      const dt=24*Math.sin(time*.7),dw=(dt>0?.01:-.005)*Math.exp(-Math.abs(dt)/20);
      c.fillStyle=style('--spike');c.beginPath();c.arc(x(.5+dt/100),y(dw/.015+1/3),3,0,Math.PI*2);c.fill();
    }else if(type==='error'){
      c.fillText('error',4,12);c.fillText('0',left,bottom+20);c.fillText('iteration',right-48,bottom+20);
      line(Array.from({length:80},(_,i)=>[i/79,.85*Math.exp(-i/15)+.03]),'--exc');
    }else{
      c.fillText('cell',4,12);c.fillText('time (ms)',right-60,bottom+20);
      c.fillStyle=style('--spike');let seed=771;
      for(let row=0;row<12;row++)for(let t=0;t<100;t++){
        seed=seed*16807%2147483647;
        if(seed/2147483647<.025)c.fillRect(x(t/100),y((row+.5)/12),2,2);
      }
    }
  }
  function draw(time=0){plot(document.querySelector('#lif'),'lif',time);document.querySelectorAll('.miniature').forEach(c=>plot(c,c.dataset.model,time));}
  return {advance,draw,get spikeAge(){return age;}};
})();
