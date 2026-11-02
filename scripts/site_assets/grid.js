'use strict';
window.NeuroShapes.grid = function(count) {
  const b=window.NeuroShapes.builder(1601);
  const panels=[[-.39,.15,1.02],[.63,.51,.5],[.63,-.3,.5]];
  for(let module=0;module<3;module++){
    const [cx,cy,width]=panels[module],half=width/2,spacing=.2*Math.pow(1.4,module)*width;
    const corners=[[cx-half,cy-half,0],[cx+half,cy-half,0],[cx+half,cy+half,0],[cx-half,cy+half,0]];
    for(let i=0;i<4;i++)b.line(corners[i],corners[(i+1)%4],.001,0,2800);
    // A hexagonal lattice is imposed explicitly, never fitted or learned.
    for(let row=-8;row<=8;row++)for(let col=-8;col<=8;col++){
      const x=cx+(col+(Math.abs(row)%2)*.5)*spacing,y=cy+row*spacing*Math.sqrt(3)/2;
      if(Math.abs(x-cx)>half*.89||Math.abs(y-cy)>half*.89)continue;
      for(let sample=0;sample<32;sample++){
        const angle=b.random()*Math.PI*2,r=Math.sqrt(-2*Math.log(Math.max(.001,b.random())))*spacing*.1;
        b.add([x+Math.cos(angle)*r,y+Math.sin(angle)*r,0],3);
      }
    }
    for(let sample=0;sample<3500;sample++)b.add([cx+(b.random()-.5)*width,cy+(b.random()-.5)*width,-.01],-.16);
    let previous=null;
    for(let i=0;i<100;i++){
      const t=i/99,p=[cx+Math.sin(t*9)*half*.83,cy+Math.sin(t*13+.5)*half*.83,.03];
      if(previous)b.line(previous,p,.001,0,350);previous=p;
      if(i%9===0)b.ellipsoid(p,[.004,.004,.004],7,8);
    }
  }
  const shape=b.finish(count);
  shape.labels=[['Imposed hexagonal firing fields',-.92,1.03],['Module 1',-.89,.79],['Module 2',-.5,.92],['Module 3',-.5,.08],['Spacing ratio 1.4 · trajectory and field hits',-.92,-.76],['Illustration. This lattice is imposed by the scene code, not learned.',-.92,-.91]];
  return shape;
};
