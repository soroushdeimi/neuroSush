'use strict';
window.NeuroShapes.microcircuit = function(count) {
  const b=window.NeuroShapes.builder(423),cells=count<12000?1200:3200;
  const boundaries=[1,.82,.55,.28,-.02,-.35,-.64];
  for(let layer=0;layer<6;layer++){
    const y=boundaries[layer];
    b.line([-.92,y,-.25],[.92,y,-.25],.001,0,200);
    b.line([.92,y,-.25],[.92,y,.25],.001,0,200);
  }
  for(let i=0;i<cells;i++){
    const layer=i<cells*.035?0:1+Math.floor(b.random()*5);
    const p=[(b.random()-.5)*1.7,boundaries[layer+1]+b.random()*(boundaries[layer]-boundaries[layer+1])*.83,(b.random()-.5)*.5];
    const inhibitory=i%5===0,tag=inhibitory?2:1,r=inhibitory?.012:.014;
    if(inhibitory){
      for(let j=0;j<8;j++){
        const a=j*Math.PI/4,next=(j+1)*Math.PI/4;
        b.line([p[0]+Math.cos(a)*r,p[1]+Math.sin(a)*r,p[2]],[p[0]+Math.cos(next)*r,p[1]+Math.sin(next)*r,p[2]],.001,tag,30);
      }
    }else{
      const triangle=[[p[0],p[1]+r,p[2]],[p[0]-r,p[1]-r,p[2]],[p[0]+r,p[1]-r,p[2]]];
      for(let j=0;j<3;j++)b.line(triangle[j],triangle[(j+1)%3],.001,tag,40);
      b.line(p,[p[0],p[1]+.065,p[2]],.001,tag,40);
      for(const side of [-1,1])b.line(p,[p[0]+side*.025,p[1]-.025,p[2]],.001,tag,40);
    }
    if(i%16===0){
      const end=[p[0]+(b.random()-.5)*.14,p[1]-.12,p[2]+.025];
      b.line(p,end,.001,0,120);
    }
  }
  // Generated spike raster: no network dynamics are claimed for these events.
  for(let row=0;row<50;row++)for(let t=0;t<140;t++){
    if(b.random()<.028)b.add([-.85+t/140*1.7,-.77-row*.005,.05],8);
  }
  b.line([-.88,-1.06,0],[.9,-1.06,0],.001,0,200);
  const shape=b.finish(count);
  shape.labels=[['L1',.84,.92],['L2',.84,.68],['L3',.84,.4],['L4',.84,.12],['L5',.84,-.19],['L6',.84,-.5],['Generated spike raster',.64,-1.14]];
  return shape;
};
