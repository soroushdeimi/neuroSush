export function makePanels(f){
 const n=(id,unit='',digits)=>`<span data-fact-id="${id}">${digits===undefined?f[id]:f[id].toFixed(digits)}${unit?' '+unit:''}</span>`;
 return [
 {
   title:'Receive.<br>Decide.<br><em>Send.</em>',
   copy:`<p>What is a <b>neuron</b>? A biological cell that receives electrical signals, integrates incoming currents, and fires action potential outputs. Vast interconnected networks coordinate all thought and perception.</p><strong class="surprise">${n('neurons')} billion neurons</strong><p>Like an artificial neural unit weighting multiple inputs before applying an activation threshold.</p>`,
   note:'Mean and reported spread in Azevedo sample. Brain length unverified.'
 },
 {
   title:'A cell.<br>A branching story.',
   copy:`<p>Dendritic trees catch incoming signals, funneling currents into the central soma cell body. The axon carries output across meters of length. Despite trillions of operations per second, brain power consumption is remarkably low.</p><strong class="surprise">~${n('power','W')} for the brain</strong><p>Like a fan-in layer aggregating distributed features.</p>`,
   note:'Nominal whole-brain glucose budget. Soma size calibrated; axon schematic.'
 },
 {
   title:'Fill. Leak.<br>Fire.',
   copy:`<p>A <b>leaky integrate-and-fire</b> model remembers recent inputs while losing charge over time. Voltage rises toward a threshold, emitting a spike event upon crossing the boundary.</p><strong class="surprise">${n('lif_tau','ms')} memory time</strong><p>Like an exponential moving average unit paired with a step threshold.</p>`,
   note:'Illustrative LIF parameters. Spike marks are events, not waveforms.'
 },
 {
   title:'An electrical<br>boundary.',
   copy:`<p>The biological <b>lipid bilayer</b> separates internal and external conducting fluids. Electric charge collects tightly on its opposing surfaces, creating a natural electrical membrane capacitor.</p><strong class="surprise">~${n('bilayer','nm')} thick</strong><p>Like a microscopic dielectric barrier separating parallel conductor plates in circuits.</p>`,
   note:'Bilayer thickness calibrated from Boal. Protein packing schematic.'
 },
 {
   title:'A gradient<br>becomes a battery.',
   copy:`<p>Ion concentration gradients create chemical driving forces across the membrane. The Nernst equation translates concentration ratios into precise electrical potentials.</p><strong class="surprise">Baseline Na⁺: +${n('e_na','mV',1)}</strong><p>Like an engineered bias term setting resting and operational activation potentials.</p>`,
   note:'Values vary by cell type. Resting balance uses simplified weighted conductances.'
 },
 {
   title:'Shed water.<br>Find a fit.',
   copy:`<p>The biological <b>selectivity filter</b> strips away an ion’s hydration shell using precision protein carbonyl oxygens. Shedding hydration water requires substantial metabolic energy repaid by precise molecular coordination.</p><strong class="surprise">K⁺ radius: ${n('k_radius','nm')}</strong><p>Like a strict feature filter rejecting mismatched molecular inputs while allowing target ions through.</p>`,
   note:'Shannon CR column. Hydrated radii approximate. Width schematic.'
 },
 {
   title:'A tiny fraction<br>carries the signal.',
   copy:`<p>Hodgkin and Huxley mapped the squid giant axon in 1952 at 6.3 °C. Voltage gates open sequentially: activation <em>m</em> and potassium <em>n</em> depolarize, while inactivation <em>h</em> closes.</p><strong class="surprise">~${n('budget_percent','%',3)} of internal ions</strong><p>Like a dynamic gating circuit controlling signal propagation.</p>`,
   note:'Squid HH trace. Gates end spike. Concentrations barely change.'
 },
 {
   title:'Recovery<br>takes time.',
   copy:`<p>Following a spike, sodium inactivation drops availability near zero, enforcing a refractory period. Potassium outflow maintains hyperpolarization, preventing immediate re-excitation.</p><strong class="surprise">h gate rest: ${n('hh_rest_h','',2)}</strong><p>Like a temporal dropout state preventing burst feedback.</p>`,
   note:'Availability is model probability. Blue h shows recovery.'
 },
 {
   title:'Across<br>the gap.',
   copy:`<p>An action potential triggers calcium influx, causing vesicles to fuse and release neurotransmitters across a microscopic cleft.</p><strong class="surprise">~${n('synapse_gap','nm')} cleft</strong><p>Receptors convert chemical messages to currents. Like a weighted directed edge.</p>`,
   note:'Cleft calibrated. Marker motion follows scroll, not release kinetics.'
 },
 {
   title:'Timing<br>leaves a trace.',
   copy:`<p><b>STDP</b> adjusts connection weights based on precise millisecond firing order. Pre-before-post timing strengthens the synapse; reverse timing weakens it.</p><strong class="surprise">${n('stdp_tau','ms')} trace time</strong><p>Like a Hebbian update rule driven by temporal correlation.</p>`,
   note:'Illustrative discrete library parameters. Windows vary by synapse type.'
 },
 {
   title:'One mechanism.<br>Many open questions.',
   copy:`<p>Neocortex organizes information across six laminar layers. Grid cells in medial entorhinal cortex fire in periodic triangular patterns, forming internal coordinate systems for navigation.</p><strong class="surprise">${n('layers')} cortical layers</strong><p>Like hierarchical feature extraction layers mapping abstract space.</p>`,
   note:'Generalized grid coding hypothesis. No TEM grid-cell claim.'
 }
 ];
}
