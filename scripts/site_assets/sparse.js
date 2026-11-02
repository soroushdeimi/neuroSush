'use strict';
window.NeuroShapes.sparse = function(count) {
  const b=window.NeuroShapes.builder(913);
  for(let row=0;row<32;row++)for(let col=0;col<32;col++){
    const id=row*32+col,x=(col-15.5)*.053+(row%2)*.026;
    const y=(row-15.5)*.032+.25,z=(row-15.5)*.018;
    for(let cell=0;cell<4;cell++){
      const p=[x,y+cell*.025,z];
      b.ellipsoid(p,[.007,.009,.007],100+id,12);
      if(cell===3){
        for(let j=0;j<8;j++){
          const a=j*Math.PI/4;
          b.add([x+Math.cos(a)*.014,p[1]+Math.sin(a)*.014,z],1200+id);
        }
      }
    }
    if(id%51===0)b.line([x,y+.075,z],[x+.16,y+.12,z],.001,3,120);
  }
  // Three columns exchange votes over shrinking candidate sets.
  for(let group=0;group<3;group++){
    const cx=(group-1)*.62;
    for(let candidate=0;candidate<9;candidate++){
      const x=cx+(candidate%3-1)*.11,y=-.66+Math.floor(candidate/3)*.1;
      b.ellipsoid([x,y,0],[.025,.025,.025],3000+candidate,160);
    }
    b.line([cx-.19,-.79,0],[cx+.19,-.79,0],.002,0,400);
    if(group<2){
      b.line([cx+.18,-.58,0],[cx+.42,-.58,0],.003,3,1500);
      b.line([cx+.42,-.58,0],[cx+.38,-.54,0],.003,3,1000);
    }
  }
  const shape=b.finish(count);
  shape.labels=[['32 × 32 minicolumns',-.87,.98],['A → B → C · 20 / 1,024 active',-.87,.84],['Distal prediction rings',-.87,-.36],['Lateral voting · shared object',-.87,-.95]];
  return shape;
};
