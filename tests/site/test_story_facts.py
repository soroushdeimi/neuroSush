"""Check the actual story ledger and browser mathematics against the library."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from neurosush.neurons.dynamics import lif_rate, lif_terms
from neurosush.neurons.hodgkin_huxley import (
    HodgkinHuxley, gate_rates, gate_steady_state, gating_step, hh_currents,
    membrane_step, resting_potential,
)
from neurosush.synapses.traces import trace_step
from neurosush.synapses.plasticity import stdp_dense

STORY = ROOT / 'scripts/site_assets/story'
LEDGER = json.loads((STORY / 'facts.json').read_text())
F = {row['id']: row['value'] for row in LEDGER['facts'] if row['status'] == 'VERIFIED'}

def tensor(value):
    return torch.tensor(value, dtype=torch.float64)

def pair(lag):
    pre, post = tensor([0.]), tensor([0.])
    total = 0.
    for t in range(abs(lag) + 1):
        sp = tensor([float(t == (0 if lag >= 0 else -lag))])
        so = tensor([float(t == (lag if lag >= 0 else 0))])
        pre = trace_step(pre, sp, tau=F['stdp_tau'], dt=F['stdp_dt'])
        post = trace_step(post, so, tau=F['stdp_tau'], dt=F['stdp_dt'])
        total += float(stdp_dense(pre_spike=sp, pre_trace=pre, post_spike=so,
                                  post_trace=post, a_plus=F['stdp_a_plus'], a_minus=F['stdp_a_minus']))
    return total

def recompute():
    thermal = 1000 * F['gas_constant'] * (F['temperature'] + F['kelvin_offset']) / F['faraday']
    out = {'thermal': thermal, 'lif_rheobase': (F['lif_threshold']-F['lif_rest'])/F['lif_resistance']}
    out['lif_rate'] = lif_rate(tau=F['lif_tau'],v_rest=F['lif_rest'],v_reset=F['lif_reset'],threshold=F['lif_threshold'],resistance=F['lif_resistance'],current=F['lif_input'],t_ref=F['lif_refractory'])
    for ion in ['na', 'k']:
        out[ion+'_ratio'] = F[ion+'_out']/F[ion+'_in']
        out['e_'+ion] = thermal * math.log(out[ion+'_ratio'])
    out['weighted_rest'] = (out['e_k']+F['rest_na_weight']*out['e_na'])/(1+F['rest_na_weight'])
    out['hh_rest'] = resting_potential()
    out['hh_detection'] = HodgkinHuxley().v_spike
    for key, value in zip(['m','h','n'], gate_steady_state(tensor(out['hh_rest']))):
        out['hh_rest_'+key] = float(value)
    out['stdp_positive'] = pair(int(F['stdp_lag']))
    out['stdp_negative'] = pair(-int(F['stdp_lag']))
    d=F['soma']*1e-6
    out['budget_area']=math.pi*d*d
    out['budget_volume']=math.pi*d**3/6
    out['budget_capacitance']=out['budget_area']*F['hh_c']*.01
    out['budget_charge']=out['budget_capacitance']*F['voltage_swing']*.001
    out['budget_crossing']=out['budget_charge']/F['elementary_charge']
    out['budget_inside']=F['na_in']*out['budget_volume']*F['avogadro']
    out['budget_percent']=100*out['budget_crossing']/out['budget_inside']
    v=tensor(out['hh_rest']);m,h,n=gate_steady_state(v);charge=0.
    for i in range(round(F['hh_duration']/F['hh_dt'])):
        rates=gate_rates(v)
        m,h,n=[gating_step(x,r.alpha,r.beta,dt=F['hh_dt']) for x,r in zip([m,h,n],rates)]
        currents=hh_currents(v,m,h,n,g_na=F['hh_g_na'],g_k=F['hh_g_k'],g_leak=F['hh_g_leak'],e_na=F['hh_e_na'],e_k=F['hh_e_k'],e_leak=F['hh_e_leak'])
        charge+=max(0,-float(currents[0]))*F['hh_dt']
        t=i*F['hh_dt'];drive=F['hh_pulse'] if F['hh_pulse_start']<=t<F['hh_pulse_end'] else 0
        v=membrane_step(v,tensor(drive),*currents,c=F['hh_c'],dt=F['hh_dt'])
    out['hh_charge_density']=charge
    out['hh_ions_entered']=charge*1e-9*out['budget_area']*1e4/F['elementary_charge']
    return out

def test_every_derived_fact():
    computed = recompute()
    derived = {row['id']: row for row in LEDGER['facts'] if 'formula' in row}
    assert set(computed) == set(derived), 'Every derived fact needs an independent recomputation'
    for key, row in derived.items():
        assert computed[key] == pytest.approx(row['value'], rel=0, abs=row['tolerance']), key

def test_fact_provenance_and_omissions():
    ids = [row['id'] for row in LEDGER['facts']]
    assert len(ids) == len(set(ids))
    for row in LEDGER['facts']:
        if row['status']=='VERIFIED':
            assert math.isfinite(row['value']) and row['unit']
            source=LEDGER['sources'][row['source']]
            assert source['citation'] and source['opened']
        else:
            assert row['status']=='NOT VERIFIED' and row['value'] is None

def test_browser_port_against_library():
    program = """
import fs from 'node:fs';
import {gateRates,hhTrace,lifStep,stdp} from './scripts/site_assets/story/math.mjs';
const f=Object.fromEntries(JSON.parse(fs.readFileSync('./scripts/site_assets/story/facts.json')).facts.filter(r=>r.status==='VERIFIED').map(r=>[r.id,r.value]));
const p=prefix=>Object.fromEntries(Object.entries(f).filter(([k])=>k.startsWith(prefix+'_')).map(([k,v])=>[k.slice(prefix.length+1),v]));
console.log(JSON.stringify({rates:[-100,-65,-55,-40,40].map(gateRates),hh:hhTrace(p('hh')),lif:[-100,-70,-40,0].map(v=>lifStep(v,f.lif_input,p('lif'))),pairs:[-40,-1,0,1,40].map(l=>stdp(l,p('stdp')))}));
"""
    result=json.loads(subprocess.check_output(['node','--input-type=module','-e',program],cwd=ROOT,text=True))
    for voltage, actual in zip([-100,-65,-55,-40,40],result['rates']):
        expected=gate_rates(tensor(voltage))
        for name, rates in zip(['m','h','n'],expected):
            assert actual[name] == pytest.approx([float(rates.alpha),float(rates.beta)],rel=1e-12)
    for voltage,actual in zip([-100,-70,-40,0],result['lif']):
        terms=lif_terms(tensor(voltage),F['lif_input'],v_rest=F['lif_rest'],resistance=F['lif_resistance'],tau=F['lif_tau'],dt=F['lif_dt'])
        assert actual==pytest.approx(voltage+float(terms['leak']+terms['input']),abs=1e-12)
    assert result['pairs']==pytest.approx([pair(t) for t in [-40,-1,0,1,40]],abs=1e-12)
    v=tensor(resting_potential());m,h,n=gate_steady_state(v)
    for i,row in enumerate(result['hh']):
        assert [row['v'],row['m'],row['h'],row['n']] == pytest.approx([float(v),float(m),float(h),float(n)],rel=1e-10,abs=1e-9)
        if i==len(result['hh'])-1: break
        rates=gate_rates(v)
        m,h,n=[gating_step(x,r.alpha,r.beta,dt=F['hh_dt']) for x,r in zip([m,h,n],rates)]
        currents=hh_currents(v,m,h,n,g_na=F['hh_g_na'],g_k=F['hh_g_k'],g_leak=F['hh_g_leak'],e_na=F['hh_e_na'],e_k=F['hh_e_k'],e_leak=F['hh_e_leak'])
        t=i*F['hh_dt'];drive=F['hh_pulse'] if F['hh_pulse_start']<=t<F['hh_pulse_end'] else 0
        v=membrane_step(v,tensor(drive),*currents,c=F['hh_c'],dt=F['hh_dt'])
