export function makePanels(f){
 const n=(id,unit='',digits)=>`<span data-fact-id="${id}">${digits===undefined?f[id]:f[id].toFixed(digits)}${unit?' '+unit:''}</span>`;
 return [
 {
   title:'Receive.<br>Decide.<br><em>Send.</em>',
   copy:`<p>What is a <b>neuron</b>? A cell receives signals, integrates them, and sends its own.</p><strong class="surprise">${n('neurons')} billion neurons</strong><p>Like an artificial neural unit, it weights inputs before passing them on.</p>`,
   note:'Mean and reported spread in Azevedo’s adult male sample. Connectivity schematic; brain length unverified.'
 },
 {
   title:'A cell.<br>A branching story.',
   copy:`<p><b>Dendrites</b> receive inputs. The soma integrates signals. The axon sends output. Whole-brain energy is astonishingly low for this massive network.</p><strong class="surprise">~${n('power','W')} for the brain</strong><p>Like a fan-in layer aggregating distributed features.</p>`,
   note:'Nominal whole-brain glucose budget. Soma size calibrated; branches and axon length schematic.'
 },
 {
   title:'Fill. Leak.<br>Fire.',
   copy:`<p>A <b>leaky integrator</b> remembers past inputs while losing charge over time. Voltage rises toward a threshold. Crossing it emits a spike.</p><strong class="surprise">${n('lif_tau','ms')} memory time</strong><p>Like an exponential moving average with a step activation threshold.</p>`,
   note:'Illustrative LIF parameters. Play changes current inside the scene. Spike marks are events, not waveforms.'
 },
 {
   title:'An electrical<br>boundary.',
   copy:`<p>The <b>lipid bilayer</b> keeps conducting fluids apart while storing charge like a tiny capacitor. Pumps and channels regulate internal composition.</p><strong class="surprise">~${n('bilayer','nm')} thick</strong><p>Like a dielectric barrier separating parallel conductor plates.</p>`,
   note:'Bilayer thickness calibrated from Boal. Protein shapes and packing are schematic.'
 },
 {
   title:'A gradient<br>becomes a battery.',
   copy:`<p>The <b>Nernst potential</b> balances chemical and electrical forces. Concentration ratios create specific reversal voltages across the membrane.</p><strong class="surprise">Baseline Na⁺: +${n('e_na','mV',1)}</strong><p>Like a bias term setting baseline activation potentials.</p>`,
   note:'Values vary by cell type. Concentrations approximate activities; resting balance uses simplified weighted conductances.'
 },
 {
   title:'Shed water.<br>Find a fit.',
   copy:`<p>The <b>selectivity filter</b> strips an ion's water shell using carbonyl oxygens. Re-coordination energetics differentiate potassium from sodium.</p><strong class="surprise">K⁺ radius: ${n('k_radius','nm')}</strong><p>Like a hard feature filter rejecting mismatched inputs.</p>`,
   note:'Shannon CR column, six-coordinate. Hydrated radii are approximate, model dependent. Inner rings mark oxygen centers. Width schematic. KcsA, not voltage-gated. No molecular dynamics.'
 },
 {
   title:'A tiny fraction<br>carries the signal.',
   copy:`<p>A <b>capacitive minimum</b> estimates ion transfer during a voltage excursion. Real spikes move more charge than this baseline.</p><strong class="surprise">~${n('budget_percent','%',3)} of internal ions</strong><p>Like a sparse activation firing packet.</p>`,
   note:'Squid HH trace. Gates end the spike. Concentrations barely change.'
 },
 {
   title:'Recovery<br>takes time.',
   copy:`<p><b>Inactivation</b> shuts sodium channels after a spike, while potassium channels stay open. This prevents immediate re-excitation.</p><strong class="surprise">h gate rest: ${n('hh_rest_h','',2)}</strong><p>Like a refractory dropout state preventing burst feedback.</p>`,
   note:'Availability is a model probability. Blue h shows recovery. No universal maximum firing rate is implied.'
 },
 {
   title:'Across<br>the gap.',
   copy:`<p>A <b>chemical synapse</b> converts electrical spikes into neurotransmitter release across a narrow cleft. Calcium influx triggers vesicle fusion.</p><strong class="surprise">~${n('synapse_gap','nm')} cleft</strong><p>Like a weighted directed edge transmitting signals.</p>`,
   note:'Cleft calibrated. Marker motion follows scroll, not diffusion or release kinetics.'
 },
 {
   title:'Timing<br>leaves a trace.',
   copy:`<p><b>STDP</b> adjusts connection weights based on exact spike timing differences. Pre-before-post timing strengthens the synapse.</p><strong class="surprise">${n('stdp_tau','ms')} trace time</strong><p>Like a Hebbian update rule driven by temporal correlation.</p>`,
   note:'Illustrative discrete library parameters. Windows vary by synapse type. Timing alone does not establish causality.'
 },
 {
   title:'One mechanism.<br>Many open questions.',
   copy:`<p><b>Neocortex</b> organizes inputs across cortical layers. Grid cells exhibit periodic spatial firing fields, suggesting abstract coordinate geometry.</p><strong class="surprise">${n('layers')} cortical layers</strong><p>Like hierarchical feature extraction layers.</p>`,
   note:'Generalized grid coding is a hypothesis. This does not claim our TEM reproduced grid cells.'
 }
 ];
}
