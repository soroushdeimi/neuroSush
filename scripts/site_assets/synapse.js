'use strict';
window.NeuroShapes.synapse = function(count) {
  const b=window.NeuroShapes.builder(231);
  // An axon widens into a bouton; the active zone faces the cleft.
  b.line([0,.69,0],[0,1.06,0],.09,0,6000);
  b.ellipsoid([0,.47,0],[.58,.43,.3],0,9000,p=>[p[0],Math.max(.1,p[1]),p[2]]);
  // Vesicle membranes are distinct spheres inside the presynaptic terminal.
  for(let row=0;row<3;row++)for(let col=0;col<5;col++){
    const center=[(col-2)*.18+(row%2)*.035,.23+row*.17,.05];
    b.ellipsoid(center,[.055,.055,.055],4,180);
  }
  // Flattened spine head and narrow neck, attached to the dendritic shaft.
  b.ellipsoid([0,-.24,0],[.46,.19,.3],6,6500);
  b.line([0,-.39,0],[.02,-.77,0],.065,6,6500);
  b.line([-.92,-.82,0],[.92,-.82,0],.095,0,5000);
  for(let i=0;i<13;i++){
    const x=(i-6)*.052;
    b.line([x,-.065,.2],[x,-.15,.2],.009,5,1800);
    b.line([x-.016,-.065,.2],[x+.016,-.065,.2],.006,5,1800);
  }
  // Released transmitter travels across the cleft after the pre-spike.
  for(let i=0;i<700;i++)b.add([(b.random()-.5)*.55,b.random()*.15-.06,(b.random()-.5)*.25],7);
  function outline(center,rx,ry){
    for(let i=0;i<80;i++){
      const a=i*Math.PI/40,next=(i+1)*Math.PI/40;
      b.line([center[0]+Math.cos(a)*rx,center[1]+Math.sin(a)*ry,.34],[center[0]+Math.cos(next)*rx,center[1]+Math.sin(next)*ry,.34],.002,0,600);
    }
  }
  outline([0,.47],.58,.37);outline([0,-.24],.46,.19);
  for(let row=0;row<3;row++)for(let col=0;col<5;col++)outline([(col-2)*.18+(row%2)*.035,.23+row*.17],.055,.055);
  b.line([-.065,-.42,.3],[-.065,-.74,.3]);b.line([.065,-.42,.3],[.065,-.74,.3]);
  const shape=b.finish(count);
  shape.labels=[['Presynaptic bouton',-.57,.97],['Vesicles',.35,.55],['Cleft',.52,.04],['Spine head',.5,-.27],['Dendrite',.52,-.9]];
  return shape;
};
