'use strict';
window.NeuroShapes = window.NeuroShapes || {};
window.NeuroShapes.builder = function(seed=812) {
  const points=[],tags=[],lines=[];
  const random=()=>((seed=seed*16807%2147483647)/2147483647);
  const add=(p,tag=0)=>{points.push(...p);tags.push(tag);};
  function line(a,b,radius=.004,tag=0,density=1200){
    lines.push(...a,...b);
    const length=Math.hypot(...a.map((v,k)=>v-b[k]));
    for(let i=0;i<Math.max(2,length*density);i++){
      const t=random(),angle=random()*Math.PI*2;
      add(a.map((v,k)=>v+(b[k]-v)*t+(k===2?Math.sin(angle):Math.cos(angle))*radius),tag);
    }
  }
  function ellipsoid(center,radii,tag=0,n=1000,clip=null){
    for(let i=0;i<n;i++){
      const a=random()*Math.PI*2,z=random()*2-1,r=Math.sqrt(1-z*z);
      let p=[Math.cos(a)*r*radii[0],z*radii[1],Math.sin(a)*r*radii[2]].map((v,k)=>v+center[k]);
      if(clip)p=clip(p);
      add(p,tag);
    }
  }
  function finish(count){
    const result=new Float32Array(count*3),data=new Float32Array(count);
    for(let i=0;i<count;i++){
      const j=Math.floor(random()*tags.length);
      result.set(points.slice(j*3,j*3+3),i*3);data[i]=tags[j];
    }
    result.tags=data;result.lines=new Float32Array(lines);return result;
  }
  return {random,add,line,ellipsoid,finish};
};
