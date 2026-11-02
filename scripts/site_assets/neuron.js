'use strict';
window.NeuroShapes = window.NeuroShapes || {};
window.NeuroShapes.neuron = function(count) {
  let seed = 812;
  const random = () => ((seed = seed * 16807 % 2147483647) / 2147483647);
  const points = [];
  const add = (x, y, z) => points.push(x, y, z);
  // Sample tapered tube surfaces, including occasional dendritic spines.
  function tube(start, end, radius, density = 1600, spines = false) {
    const delta = end.map((v, i) => v - start[i]);
    const length = Math.hypot(...delta);
    const axis = delta.map(v => v / length);
    const normal = Math.abs(axis[1]) < .9 ? [-axis[2], 0, axis[0]] : [1, 0, 0];
    const norm = Math.hypot(...normal);
    const n = normal.map(v => v / norm);
    const b = [axis[1]*n[2]-axis[2]*n[1], axis[2]*n[0]-axis[0]*n[2], axis[0]*n[1]-axis[1]*n[0]];
    for (let i = 0; i < Math.ceil(length * density); i++) {
      const t = random(), angle = random() * Math.PI * 2;
      const r = radius * (1 - .45*t);
      const p = start.map((v, k) => v + delta[k]*t + r*(n[k]*Math.cos(angle)+b[k]*Math.sin(angle)));
      add(...p);
      if (spines && i % 31 === 0) {
        const direction = n.map((v,k) => v*Math.cos(angle)+b[k]*Math.sin(angle));
        for (let s = 1; s <= 5; s++) add(...p.map((v,k) => v+direction[k]*s*.003));
      }
    }
  }
  function branch(start, angle, length, depth, radius, spread = .62) {
    const end = [start[0]+Math.sin(angle)*length, start[1]+Math.cos(angle)*length, start[2]+(random()-.5)*length*.5];
    tube(start, end, radius, 1800, true);
    if (depth === 0) return;
    for (const side of [-1, 1]) branch(end, angle+side*(spread+random()*.25), length*(.63+random()*.14), depth-1, radius*.66, spread);
  }
  // Pyramidal soma: rounded triangular cross-sections rather than a sphere.
  const soma = [0, -.12, 0];
  for (let i = 0; i < 2600; i++) {
    const t = random(), a = random()*Math.PI*2;
    const r = .115*(1-t)+.018;
    add(Math.cos(a)*r, -.23+t*.27, Math.sin(a)*r*.72);
  }
  tube(soma, [0,.59,0], .026, 4000, true);
  for (const side of [-1,1]) {
    branch([0,.56,0], side*.48, .29, 4, .014, .43);
    branch([0,.27,0], side*1.02, .22, 3, .012);
    for (let j = 0; j < 3; j++) branch(soma, side*(1.8+j*.32), .28+random()*.12, 4, .017);
  }
  // Hillock, thin axon and separated thicker internodes.
  tube([0,-.2,0], [.04,-.35,0], .018, 2500);
  const axon = [[.04,-.35,0],[.07,-.53,.02],[.15,-.74,.02],[.12,-.97,.01],[.2,-1.2,0]];
  for (let j = 0; j < axon.length-1; j++) {
    const start = axon[j], end = axon[j+1];
    tube(start,end,.006,3500);
    tube(start.map((v,k)=>v+(end[k]-v)*.12),start.map((v,k)=>v+(end[k]-v)*.87),.016,2800);
  }
  for (let j = 1; j < 4; j++) {
    const start = axon[j], side = j%2 ? 1 : -1;
    const end = [start[0]+side*.28,start[1]-.15,.02];
    tube(start,end,.006,2000);
    for (let k = 0; k < 3; k++) {
      const tip = [end[0]+side*(.06+k*.035),end[1]-.04-k*.04,.02];
      tube(end,tip,.004,1800);
      for (let p = 0; p < 70; p++) {
        const a = random()*Math.PI*2, z = random()*2-1, r = Math.sqrt(1-z*z)*.016;
        add(tip[0]+Math.cos(a)*r,tip[1]+z*.016,tip[2]+Math.sin(a)*r);
      }
    }
  }
  const result = new Float32Array(count*3);
  // Shuffle sampling so adaptive draw counts retain the whole morphology.
  for (let i = 0; i < count; i++) {
    const source = Math.floor(random()*(points.length/3))*3;
    result.set(points.slice(source,source+3).map(v => v*.72),i*3);
  }
  return result;
};
