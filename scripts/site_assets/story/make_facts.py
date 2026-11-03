"""Regenerate the sourced teaching parameters and numerical references."""
import json
import math
from pathlib import Path

facts = []
sources = {
 'deck': {'citation': 'Deimi. What is a neuron? The_Neuron_v2, slides 11–12, 18, 20, 26–28.', 'url': '', 'opened': 'Local rendered slides p01–p30'},
 'anatomy': {'citation': 'National Research Council (1992). Environmental Neurotoxicology, chapter 2, p. 22. National Academies Press.', 'url': 'https://www.nationalacademies.org/read/1801/chapter/4', 'opened': 'Structure of the Neuron'},
 'azevedo': {'citation': 'Azevedo et al. (2009). J Comp Neurol 513:532–541, abstract.', 'url': 'https://gwern.net/doc/psychology/neuroscience/2009-azevedo.pdf', 'opened': 'PDF p. 532'},
 'energy': {'citation': 'Levy & Calvert (2021). PNAS 118:e2008173118, Energy Audit.', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8106317/', 'opened': 'Abstract and Results / Energy Audit'},
 'lif': {'citation': 'neuroSush, neurons/dynamics.py, lif_terms and lif_rate; N01 audit §§2–5.', 'url': '', 'opened': 'Local code and audit'},
 'hh': {'citation': 'neuroSush, neurons/hodgkin_huxley.py, gate_rates, hh_currents, resting_potential and HodgkinHuxley; N05 audit §§2–5.', 'url': '', 'opened': 'Local code and audit'},
 'stdp': {'citation': 'neuroSush, synapses/traces.py trace_step and synapses/plasticity.py stdp_dense; Y01 audit §§2–5.', 'url': '', 'opened': 'Local code and audit; illustrative Y01 parameters'},
 'nernst': {'citation': 'M03 audit §§2–5: Nernst equation and physical constants; deck p26–28: concentrations.', 'url': '', 'opened': 'Local audit and deck; no library Nernst function exists'},
 'synapse': {'citation': 'neuroSush, neurons/dendrite.py, conductance_step; deck p18.', 'url': '', 'opened': 'Local code and rendered slide'},
}
def add(id, value, unit, source, **kw):
 facts.append(dict(id=id,value=value,unit=unit,source=source,status='VERIFIED',**kw))
add('power',20,'W','energy',note='Nominal whole-brain glucose energy budget, not ATP watts exclusively available for computation.')
add('neurons',86.1,'billion neurons','azevedo',note='Mean NeuN-positive cell estimate in the adult male sample; not an exact count for every brain.')
add('soma_min',10,'µm','anatomy');add('soma_max',100,'µm','anatomy');add('soma',20,'µm','deck',note='Illustrative spherical cell from slide 11; branching geometry is schematic.')
for id,v,u in [('tau',20,'ms'),('rest',-70,'mV'),('threshold',-40,'mV'),('reset',-75,'mV'),('resistance',10,'MΩ'),('refractory',2,'ms'),('dt',.1,'ms'),('input',3.5,'nA'),('input_max',12,'nA'),('duration',400,'ms')]:add('lif_'+id,v,u,'lif',note='Illustrative N01 teaching parameter; not cell-specific calibration.')
add('lif_rheobase',3,'nA','lif',formula='(lif_threshold-lif_rest)/lif_resistance',tolerance=1e-12)
add('lif_rate',1000/(2+20*math.log(40/5)),'Hz','lif',formula='lif_rate(current=lif_input)',tolerance=1e-9)
add('temperature',37,'°C','nernst')
add('gas_constant',8.314462618,'J mol⁻¹ K⁻¹','nernst');add('faraday',96485.332,'C mol⁻¹','nernst');add('kelvin_offset',273.15,'K','nernst')
add('k_in',140,'mM','deck');add('k_out',5,'mM','deck');add('na_in',12,'mM','deck',note='Cell-type dependent; M03/N05 plates instead illustrate 15 mM and approximately +60.6 mV. Neither concentration is universal.');add('na_out',145,'mM','deck')
thermal=1000*8.314462618*(37+273.15)/96485.332
add('thermal',thermal,'mV','nernst',formula='1000*gas_constant*(temperature+kelvin_offset)/faraday',tolerance=1e-10)
for ion,inside,outside in [('k',140,5),('na',12,145)]:
 add(ion+'_ratio',outside/inside,'out/in','nernst',formula=f'{ion}_out/{ion}_in',tolerance=1e-12)
 add('e_'+ion,thermal*math.log(outside/inside),'mV','nernst',formula=f'thermal*ln({ion}_ratio)',tolerance=1e-9)
for id,v,u in [('c',1,'µF/cm²'),('g_na',120,'mS/cm²'),('g_k',36,'mS/cm²'),('g_leak',.3,'mS/cm²'),('e_na',50,'mV'),('e_k',-77,'mV'),('e_leak',-54.387,'mV'),('temperature',6.3,'°C'),('v_spike',0,'mV'),('dt',.025,'ms'),('duration',30,'ms'),('pulse',15,'µA/cm²'),('pulse_start',2,'ms'),('pulse_end',4,'ms')]:add('hh_'+id,v,u,'hh')
for id,v,u in [('a_plus',.02,'weight units'),('a_minus',.021,'weight units'),('tau',20,'ms'),('dt',1,'ms'),('lag',20,'ms')]:add('stdp_'+id,v,u,'stdp',note='Illustrative Y01 parameter, explicitly supplied to library; not a library default.')
add('stdp_positive',.02*.95**20,'weight units','stdp',formula='stdp_a_plus*(1-stdp_dt/stdp_tau)^(stdp_lag/stdp_dt)',tolerance=1e-12)
add('stdp_negative',-.021*.95**20,'weight units','stdp',formula='-stdp_a_minus*(1-stdp_dt/stdp_tau)^(stdp_lag/stdp_dt)',tolerance=1e-12)
add('synapse_gap',20,'nm','deck',note='Approximate chemical synapse cleft; slide 18.')
add('layers',6,'layers','deck',note='Neocortical layer scheme, slide 20; not every cortex region has an identical laminar structure.')
for id in ['cl_in','cl_out','ca_in','ca_out','hydrated_na','hydrated_k','filter_width','channel_rate','channel_conductance','bilayer','na_radius','k_radius']:
 facts.append(dict(id=id,value=None,unit='',source=None,status='NOT VERIFIED',note='A primary or textbook source must be opened before showing this number.'))
Path(__file__).with_name('facts.json').write_text(json.dumps(dict(sources=sources,facts=facts),indent=2,ensure_ascii=False)+'\n')
