'use strict';
window.NeuroShapes.hierarchy = function(count) {
  const b=window.NeuroShapes.builder(574);
  for(let layer=0;layer<3;layer++){
    const y=-.62+layer*.65;
    const corners=[[-.8,y,-.25],[.6,y,-.25],[.85,y+.14,.25],[-.55,y+.14,.25]];
    for(let j=0;j<4;j++)b.line(corners[j],corners[(j+1)%4],.002,0,5000);
    for(let cell=0;cell<9;cell++){
      const x=-.58+cell*.145;
      b.ellipsoid([x,y+.065,0],[.014,.014,.014],0,140);
    }
  }
  for(const x of [-.44,-.18,.2,.46]){
    const down=x<0,tag=down?9:10;
    for(let level=0;level<2;level++){
      const lower=-.55+level*.65,upper=lower+.53;
      b.line([x,lower,0],[x,upper,0],.002,0,1800);
      const tip=down?lower:upper,direction=down?1:-1;
      b.line([x,tip,0],[x-.035,tip+direction*.065,0],.002,0,1200);
      b.line([x,tip,0],[x+.035,tip+direction*.065,0],.002,0,1200);
    }
    for(let i=0;i<500;i++)b.add([x+(b.random()-.5)*.012,-.56+b.random()*1.3,.02],tag);
  }
  const shape=b.finish(count);
  shape.labels=[['Higher area',.5,.94],['Intermediate',.52,.27],['V1',.6,-.44],['Predictions ↓',-.82,-.94],['Errors ↑',.24,-.94]];
  return shape;
};
