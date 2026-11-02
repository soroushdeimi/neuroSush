'use strict';
window.NeuroShapes.brain = function(count) {
  const b=window.NeuroShapes.builder(1749);
  function cortex(u,v,side){
    const r=Math.sin(v),x=Math.cos(u)*r*1.02;
    let y=Math.cos(v)*.71+.21;
    const z=side*(.045+Math.abs(Math.sin(u))*r*.43);
    if(y<-.17)y=-.17+(y+.17)*.55;
    y-=.13*Math.exp(-Math.pow((x+.08)/.48,2))*Math.max(0,-Math.cos(v));
    const wx=x+.085*Math.sin(y*7+z*5),wy=y+.07*Math.sin(x*8-z*6);
    const folds=Math.sin(wx*23+2.5*Math.sin(wy*17)+1.2*Math.sin(wx*7+wy*9))+.7*Math.sin(wy*31+2*Math.sin(wx*19))+.4*Math.sin(wx*47+wy*36);
    const central=Math.exp(-Math.pow((x-.08-.07*Math.sin(y*6))/.025,2));
    const sylvian=Math.exp(-Math.pow((y+.08-.16*x)/.022,2))*Math.exp(-Math.pow(x/.85,6));
    const sulcus=Math.exp(-Math.abs(folds)*14);
    const displacement=.025-.055*sulcus-.035*central-.045*sylvian;
    const normal=[x/1.02,(y-.2)/.71,z/.47],length=Math.hypot(...normal);
    const p=[x,y,z].map((a,k)=>a+normal[k]/length*displacement);
    return {p,shade:.18+.82*Math.max(sulcus,central,sylvian)};
  }
  // A paired parametric mesh with anatomical poles and warped gyral ridges.
  for(const side of [-1,1]){
    for(let i=0;i<14500;i++){
      const u=b.random()*Math.PI*2,v=Math.acos(b.random()*2-1),sample=cortex(u,v,side);
      // Most visible samples belong to the near hemisphere in a lateral view.
      if(side<0&&sample.p[1]<.58&&b.random()>.18)continue;
      b.add(sample.p,-sample.shade);
    }
  }
  // Trace warped sulcal level sets across the lateral cortical surface.
  const field=(x,y)=>{
    const wx=x+.085*Math.sin(y*7+2.1),wy=y+.07*Math.sin(x*8-2.5);
    return Math.sin(wx*23+2.5*Math.sin(wy*17)+1.2*Math.sin(wx*7+wy*9))+.7*Math.sin(wy*31+2*Math.sin(wx*19))+.4*Math.sin(wx*47+wy*36);
  };
  const inside=(x,y)=>Math.pow(x/1.02,2)+Math.pow((y-.21)/.71,2)<.96&&y>-.27;
  for(let ix=0;ix<160;ix++)for(let iy=0;iy<110;iy++){
    const x=-1.02+ix*.01275,y=-.27+iy*.011;
    const corners=[[x,y],[x+.01275,y],[x+.01275,y+.011],[x,y+.011]];
    if(!corners.every(p=>inside(...p)))continue;
    const values=corners.map(p=>field(...p)),crossings=[];
    for(let edge=0;edge<4;edge++){
      const next=(edge+1)%4;
      if(values[edge]*values[next]>=0)continue;
      const t=values[edge]/(values[edge]-values[next]);
      const p=corners[edge].map((v,k)=>v+(corners[next][k]-v)*t);
      const z=.045+.43*Math.sqrt(Math.max(0,1-Math.pow(p[0]/1.02,2)-Math.pow((p[1]-.21)/.71,2)));
      crossings.push([p[0],p[1],z]);
    }
    for(let i=0;i+1<crossings.length;i+=2)b.line(crossings[i],crossings[i+1],.001,0,500);
  }
  // Cerebellar folia are finer and approximately parallel.
  for(let i=0;i<5000;i++){
    const u=b.random()*Math.PI*2,z=b.random()*2-1,r=Math.sqrt(1-z*z);
    const folia=.012*Math.sin(z*95);
    b.add([.64+Math.cos(u)*r*(.34+folia),-.47+z*.23,Math.sin(u)*r*.28],-(.25+.7*Math.pow(Math.sin(z*95),8)));
  }
  for(let ring=0;ring<17;ring++){
    const y=-.67+ring*.025,r=Math.sqrt(Math.max(0,1-Math.pow((y+.47)/.23,2)));
    let previous=null;
    for(let i=0;i<70;i++){
      const a=i/69*Math.PI,p=[.64+Math.cos(a)*r*.34,y,Math.sin(a)*r*.29];
      if(previous)b.line(previous,p,.001,0,150);previous=p;
    }
  }
  b.line([.22,-.4,.02],[.16,-.78,.04],.075,0,6000);
  b.line([.16,-.78,.04],[.23,-1.01,.04],.036,0,4500);
  // Medial temporal formation: curved body, curled head and tapered tail.
  for(let j=0;j<110;j++){
    const t=j/109,a=-.3+t*Math.PI*1.5;
    const p=[-.12+.36*Math.cos(a),-.23+.15*Math.sin(a),.43];
    b.ellipsoid(p,[.024*(1-t*.55),.023,.018],3,14);
  }
  b.ellipsoid([-.46,-.28,.4],[.1,.085,.015],3,700);
  for(let j=0;j<80;j++){
    const a=.1+j/79*Math.PI;
    b.add([-.08+.39*Math.cos(a),.03+.19*Math.sin(a),.04],0);
  }
  const shape=b.finish(count);
  shape.labels=[['Cerebral cortex',-.89,1.02],['Hippocampal formation',-.92,-.5],['Cerebellum',.53,-.83],['Brainstem',-.09,-1.08]];
  return shape;
};
