"""Regenerate the sourced teaching parameters and numerical references."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch

from neurosush.neurons.hodgkin_huxley import (
    HodgkinHuxley,
    gate_rates,
    gate_steady_state,
    gating_step,
    hh_currents,
    membrane_step,
    resting_potential,
)

facts = []
sources = {
    "deck": {
        "citation": (
            "Deimi. What is a neuron? The_Neuron_v2, slides 11\u201312, 18, 20, 26\u201328."
        ),
        "url": "",
        "opened": "Local rendered slides p01\u2013p30",
    },
    "anatomy": {
        "citation": (
            "National Research Council (1992). Environmental Neurotoxicology, "
            "chapter 2, p. 22. National Academies Press."
        ),
        "url": "https://www.nationalacademies.org/read/1801/chapter/4",
        "opened": "Structure of the Neuron",
    },
    "azevedo": {
        "citation": ("Azevedo et al. (2009). J Comp Neurol 513:532\u2013541, abstract."),
        "url": "https://gwern.net/doc/psychology/neuroscience/2009-azevedo.pdf",
        "opened": "PDF p. 532",
    },
    "energy": {
        "citation": "Levy & Calvert (2021). PNAS 118:e2008173118, Energy Audit.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8106317/",
        "opened": "Abstract and Results / Energy Audit",
    },
    "lif": {
        "citation": (
            "neuroSush, neurons/dynamics.py, lif_terms and lif_rate; N01 audit §§2\u20135."
        ),
        "url": "",
        "opened": "Local code and audit",
    },
    "hh": {
        "citation": (
            "Hodgkin & Huxley (1952). J Physiol 117:500\u2013544. "
            "Numerical implementation: neuroSush neurons/hodgkin_huxley.py, "
            "gate_rates, hh_currents, resting_potential and "
            "HodgkinHuxley; N05 audit §§2\u20135."
        ),
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1392413/",
        "opened": (
            "Local code and audit; original article landing-page title "
            "and author metadata checked; scanned original pages not read"
        ),
    },
    "stdp": {
        "citation": (
            "neuroSush, synapses/traces.py trace_step and "
            "synapses/plasticity.py stdp_dense; Y01 audit §§2\u20135."
        ),
        "url": "",
        "opened": "Local code and audit; illustrative Y01 parameters",
    },
    "nernst": {
        "citation": (
            "M03 audit §§2\u20135: Nernst equation and physical constants; "
            "deck p26\u201328: concentrations."
        ),
        "url": "",
        "opened": "Local audit and deck; no library Nernst function exists",
    },
    "synapse": {
        "citation": "neuroSush, neurons/dendrite.py, conductance_step; deck p18.",
        "url": "",
        "opened": "Local code and rendered slide",
    },
}


def add(id, value, unit, source, **kw):
    facts.append(
        {
            "id": id,
            "value": value,
            "unit": unit,
            "source": source,
            "status": "VERIFIED",
            **kw,
        }
    )


add(
    "power",
    20,
    "W",
    "energy",
    note=(
        "Nominal whole-brain glucose energy budget, not ATP watts "
        "exclusively available for computation."
    ),
)
add(
    "neurons",
    86.1,
    "billion neurons",
    "azevedo",
    note=(
        "Mean NeuN-positive cell estimate in the adult male sample; "
        "not an exact count for every brain."
    ),
)
add("soma_min", 10, "µm", "anatomy")
add("soma_max", 100, "µm", "anatomy")
add(
    "soma",
    20,
    "µm",
    "deck",
    note=("Illustrative spherical cell from slide 11; branching geometry is schematic."),
)
for id, v, u in [
    ("tau", 20, "ms"),
    ("rest", -70, "mV"),
    ("threshold", -40, "mV"),
    ("reset", -75, "mV"),
    ("resistance", 10, "MΩ"),
    ("refractory", 2, "ms"),
    ("dt", 0.1, "ms"),
    ("input", 3.5, "nA"),
    ("input_max", 12, "nA"),
    ("duration", 400, "ms"),
]:
    add(
        "lif_" + id,
        v,
        u,
        "lif",
        note="Illustrative N01 teaching parameter; not cell-specific calibration.",
    )
add(
    "lif_rheobase",
    3,
    "nA",
    "lif",
    formula="(lif_threshold-lif_rest)/lif_resistance",
    tolerance=1e-12,
)
add(
    "lif_rate",
    1000 / (2 + 20 * math.log(40 / 5)),
    "Hz",
    "lif",
    formula="lif_rate(current=lif_input)",
    tolerance=1e-9,
)
add(
    "nernst_initial",
    0,
    "mV",
    "nernst",
    note=("M03 illustrative initial voltage; relaxation starts after a voltage perturbation."),
)
add("temperature", 37, "°C", "nernst")
add("gas_constant", 8.314462618, "J mol⁻¹ K⁻¹", "nernst")
add("faraday", 96485.332, "C mol⁻¹", "nernst")
add("kelvin_offset", 273.15, "K", "nernst")
add("k_in", 140, "mM", "deck")
add("k_out", 5, "mM", "deck")
add(
    "na_in",
    12,
    "mM",
    "deck",
    note=(
        "Cell-type dependent; M03/N05 plates instead illustrate 15 mM "
        "and approximately +60.6 mV. Neither concentration is universal."
    ),
)
add("na_out", 145, "mM", "deck")
thermal = 1000 * 8.314462618 * (37 + 273.15) / 96485.332
add(
    "thermal",
    thermal,
    "mV",
    "nernst",
    formula="1000*gas_constant*(temperature+kelvin_offset)/faraday",
    tolerance=1e-10,
)
for ion, inside, outside in [("k", 140, 5), ("na", 12, 145)]:
    add(
        ion + "_ratio",
        outside / inside,
        "out/in",
        "nernst",
        formula=f"{ion}_out/{ion}_in",
        tolerance=1e-12,
    )
    add(
        "e_" + ion,
        thermal * math.log(outside / inside),
        "mV",
        "nernst",
        formula=f"thermal*ln({ion}_ratio)",
        tolerance=1e-9,
    )
for id, v, u in [
    ("c", 1, "µF/cm²"),
    ("g_na", 120, "mS/cm²"),
    ("g_k", 36, "mS/cm²"),
    ("g_leak", 0.3, "mS/cm²"),
    ("e_na", 50, "mV"),
    ("e_k", -77, "mV"),
    ("e_leak", -54.387, "mV"),
    ("temperature", 6.3, "°C"),
    ("v_spike", 0, "mV"),
    ("dt", 0.025, "ms"),
    ("duration", 30, "ms"),
    ("pulse", 15, "µA/cm²"),
    ("pulse_start", 2, "ms"),
    ("pulse_end", 4, "ms"),
]:
    add("hh_" + id, v, u, "hh")
for id, v, u in [
    ("a_plus", 0.02, "weight units"),
    ("a_minus", 0.021, "weight units"),
    ("tau", 20, "ms"),
    ("dt", 1, "ms"),
    ("lag", 20, "ms"),
]:
    add(
        "stdp_" + id,
        v,
        u,
        "stdp",
        note=("Illustrative Y01 parameter, explicitly supplied to library; not a library default."),
    )
add(
    "stdp_positive",
    0.02 * 0.95**20,
    "weight units",
    "stdp",
    formula="stdp_a_plus*(1-stdp_dt/stdp_tau)^(stdp_lag/stdp_dt)",
    tolerance=1e-12,
)
add(
    "stdp_negative",
    -0.021 * 0.95**20,
    "weight units",
    "stdp",
    formula="-stdp_a_minus*(1-stdp_dt/stdp_tau)^(stdp_lag/stdp_dt)",
    tolerance=1e-12,
)
add(
    "synapse_gap",
    20,
    "nm",
    "deck",
    note="Approximate chemical synapse cleft; slide 18.",
)
add(
    "layers",
    6,
    "layers",
    "deck",
    note=(
        "Neocortical layer scheme, slide 20; "
        "not every cortex region has an identical laminar structure."
    ),
)
sources.update(
    {
        "boal": {
            "citation": (
                "Boal (2012). Mechanics of the Cell, 2nd ed., "
                "§1.2.1, p. 10. Cambridge University Press."
            ),
            "url": (
                "https://assets.cambridge.org/97805211/13762/excerpt/9780521113762_excerpt.pdf"
            ),
            "opened": "PDF downloaded and read with PDFKit, §1.2.1",
        },
        "shannon": {
            "citation": (
                "Shannon (1976). Acta Crystallogr A32:751\u2013767, Table 1, pp. 752\u2013753."
            ),
            "url": ("https://nilab.physics.ucla.edu/sites/default/files/shannon-1976-radius.pdf"),
            "opened": "Table 1 Na and K, six-coordinate CR and IR columns",
        },
        "hydration": {
            "citation": ("Mähler & Persson (2012). Inorganic Chemistry 51:425\u2013438, abstract."),
            "url": (
                "https://research.slu.se/en/publications/"
                "a-study-of-the-hydration-of-the-alkali-metal-ions-in-aqueous-solu/"
            ),
            "opened": ("Author institution abstract: mean ion\u2013water oxygen distances"),
        },
        "zhou": {
            "citation": (
                "Zhou et al. (2001). Nature 414:43\u201348, abstract and figure captions."
            ),
            "url": "https://www.nature.com/articles/35102009",
            "opened": (
                "Publisher abstract, displacement of water and carbonyl "
                "coordination; full paper inaccessible"
            ),
        },
        "doyle": {
            "citation": ("Doyle et al. (1998). Science 280:69\u201377, pp. 69\u201371 and 75."),
            "url": ("https://lab.rockefeller.edu/chait/pdf/98/98_doyle_science.pdf"),
            "opened": (
                "Original PDF: throughput, pore length, filter length, carbonyl coordination"
            ),
        },
        "si": {
            "citation": (
                "NIST (2019/2022). SI Brochure §2 and CODATA, "
                "elementary charge and Avogadro constant."
            ),
            "url": ("https://www.nist.gov/pml/special-publication-330/sp-330-section-2"),
            "opened": ("SI defining constants and NIST elementary-charge value"),
        },
    }
)
add(
    "bilayer",
    5,
    "nm",
    "boal",
    note=(
        "Upper end of the textbook 4\u20135 nm range; lipid bilayer, "
        "not a universal full membrane-protein thickness."
    ),
)
add(
    "na_radius",
    0.116,
    "nm",
    "shannon",
    note=(
        "Six-coordinate Shannon crystal radius, CR column, p. 753. "
        "A radius convention, not a rigid sphere in solution."
    ),
)
add(
    "k_radius",
    0.152,
    "nm",
    "shannon",
    note="Six-coordinate Shannon crystal radius, CR column, p. 752.",
)
add(
    "shell_na",
    0.243,
    "nm",
    "hydration",
    note=(
        "Mean Na\u2013water oxygen distance; shell ring marks "
        "oxygen centers, not outer hydrated or hydrodynamic radius."
    ),
)
add(
    "shell_k",
    0.281,
    "nm",
    "hydration",
    note=(
        "Mean K\u2013water oxygen distance; shell ring marks "
        "oxygen centers, not outer hydrated or hydrodynamic radius."
    ),
)
add(
    "coordination",
    6,
    "oxygen neighbors",
    "shannon",
    note="Coordination number for the chosen crystal radii.",
)
add(
    "na_effective_radius",
    0.102,
    "nm",
    "shannon",
    note=("Six-coordinate IR column; distinct from CR values used for the circles."),
)
add(
    "k_effective_radius",
    0.138,
    "nm",
    "shannon",
    note=("Six-coordinate IR column; distinct from CR values used for the circles."),
)
add(
    "filter_length",
    1.2,
    "nm",
    "doyle",
    note=("KcsA structural filter, not a full voltage-gated channel structure."),
)
add("pore_length", 4.5, "nm", "doyle")
add(
    "filter_spacing",
    0.75,
    "nm",
    "doyle",
    note=(
        "Approximate center separation of two occupied ions in Doyle 1998; "
        "not all adjacent binding sites simultaneously occupied."
    ),
)
add(
    "grid_angle",
    60,
    "degrees",
    "deck",
    note=(
        "Three-wave idealized grid illustration, slide 21; "
        "not an experimentally measured firing map."
    ),
)
add(
    "filter_coordination",
    8,
    "oxygen ligands",
    "zhou",
    note=("Square antiprism around a K binding site; diagram projects coordination schematically."),
)
add(
    "channel_rate",
    1e8,
    "ions/s",
    "doyle",
    note=("Throughput up to this order; not a universal rate or the playback speed."),
)
add("elementary_charge", 1.602176634e-19, "C", "si")
add("avogadro", 6.02214076e23, "mol⁻¹", "si")
add(
    "voltage_swing",
    100,
    "mV",
    "deck",
    note=("Illustrative capacitive voltage excursion, not the integrated HH sodium charge."),
)
add(
    "rest_na_weight",
    0.15,
    "g_Na/g_K",
    "nernst",
    note=(
        "Chosen illustrative conductance ratio; ignores Cl and "
        "voltage dependence. Conductance-weighted simplification of "
        "the multi-ion resting-potential idea, not GHK."
    ),
)
add(
    "weighted_rest",
    (thermal * math.log(5 / 140) + 0.15 * thermal * math.log(145 / 12)) / 1.15,
    "mV",
    "nernst",
    formula="(e_k+rest_na_weight*e_na)/(1+rest_na_weight)",
    tolerance=1e-9,
)
d = 20e-6
area = math.pi * d * d
volume = math.pi * d**3 / 6
cap = area * 0.01
charge = cap * 0.1
crossing = charge / 1.602176634e-19
inside = 12 * volume * 6.02214076e23
for id, value, unit, formula in [
    ("area", area, "m²", "pi*(soma*1e-6)^2"),
    ("volume", volume, "m³", "pi*(soma*1e-6)^3/6"),
    ("capacitance", cap, "F", "area*hh_c*.01"),
    ("charge", charge, "C", "capacitance*voltage_swing*.001"),
    ("crossing", crossing, "ions", "charge/elementary_charge"),
    ("inside", inside, "ions", "na_in*volume*avogadro"),
    ("percent", 100 * crossing / inside, "%", "100*crossing/inside"),
]:
    add(
        "budget_" + id,
        value,
        unit,
        "deck",
        formula=formula,
        tolerance=max(abs(value) * 1e-10, 1e-25),
        note=(
            "Spherical cell estimate; capacitive lower bound, "
            "not total actual Na entry. Deck warns real entry "
            "is several times greater."
        ),
    )

rest = resting_potential()
add("hh_rest", rest, "mV", "hh", formula="resting_potential()", tolerance=1e-9)
for key, value in zip(
    ["m", "h", "n"],
    gate_steady_state(torch.tensor(rest, dtype=torch.float64)),
    strict=False,
):
    add(
        "hh_rest_" + key,
        float(value),
        "probability",
        "hh",
        formula="gate_steady_state(hh_rest)." + key,
        tolerance=1e-12,
    )
add(
    "hh_detection",
    HodgkinHuxley().v_spike,
    "mV",
    "hh",
    formula="HodgkinHuxley().v_spike",
    tolerance=1e-12,
    note=("Event detection crossing; not a fixed biological excitability threshold."),
)
v = torch.tensor(rest, dtype=torch.float64)
m, h, n = gate_steady_state(v)
integrated = 0.0
for i in range(round(30 / 0.025)):
    rates = gate_rates(v)
    m, h, n = [
        gating_step(x, r.alpha, r.beta, dt=0.025) for x, r in zip([m, h, n], rates, strict=False)
    ]
    currents = hh_currents(
        v,
        m,
        h,
        n,
        g_na=120,
        g_k=36,
        g_leak=0.3,
        e_na=50,
        e_k=-77,
        e_leak=-54.387,
    )
    integrated += max(0, -float(currents[0])) * 0.025
    t = i * 0.025
    v = membrane_step(
        v,
        torch.tensor(15.0 if 2 <= t < 4 else 0.0, dtype=torch.float64),
        *currents,
        c=1,
        dt=0.025,
    )
add(
    "hh_charge_density",
    integrated,
    "µA ms/cm²",
    "hh",
    formula="sum(max(0,-I_Na)*hh_dt) over hh_duration",
    tolerance=1e-8,
)
add(
    "hh_ions_entered",
    integrated * 1e-9 * area * 1e4 / 1.602176634e-19,
    "ions",
    "hh",
    formula=("hh_charge_density*1e-9*budget_area*1e4/elementary_charge"),
    tolerance=1e-4,
    note=(
        "Integrated sodium entry for the stated HH pulse; "
        "hybrid scaling to the illustrative spherical cell area, "
        "not an observed mammalian or squid count."
    ),
)
for id in ["cl_in", "cl_out", "ca_in", "ca_out"]:
    facts.append(
        {
            "id": id,
            "value": None,
            "unit": "",
            "source": None,
            "status": "NOT VERIFIED",
            "note": (
                "Audit gives illustrative concentrations but no named primary "
                "or textbook concentration source was opened. Omitted; "
                "free Ca and Cl are cell-type dependent."
            ),
        }
    )
sources["fusion"] = {
    "citation": (
        "Siepi et al. (2011). Biophysical Journal 100:2412\u20132421, "
        "Table 2, adopting Nightingale (1959) hydrated radii."
    ),
    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3093559/",
    "opened": ("Table 2 HTML read directly; original Nightingale full text inaccessible"),
}
sources["single_channel"] = {
    "citation": ("Llano, Webb & Bezanilla (1988). J Gen Physiol 92:179\u2013196, abstract."),
    "url": "https://pubmed.ncbi.nlm.nih.gov/3171538/",
    "opened": ("PubMed abstract HTML read directly; reported single-channel conductances"),
}
add(
    "hydrated_na",
    0.358,
    "nm",
    "fusion",
    note=(
        "Approximate effective hydrated radius, model dependent; "
        "Nightingale values reproduced in Table 2. Not a hard outer "
        "molecular boundary or the oxygen-center radius."
    ),
)
add(
    "hydrated_k",
    0.331,
    "nm",
    "fusion",
    note=(
        "Approximate effective hydrated radius, model dependent; "
        "Nightingale values reproduced in Table 2."
    ),
)
add(
    "channel_conductance",
    20,
    "pS",
    "single_channel",
    note=(
        "Most frequently observed conductance in this cut-open "
        "squid-axon study; not a universal channel conductance."
    ),
)
sources["hille_pore"] = {
    "citation": ("Hille (1973). J Gen Physiol 61:669\u2013686, abstract and p. 683."),
    "url": ("https://pdfs.semanticscholar.org/0408/bf0bdee90a70935b8c898e0404c9f7f18013.pdf"),
    "opened": (
        "Original PDF: hypothesized oxygen-lined narrow pore of about 3 Å; "
        "p. 683 gives 3.0\u20133.3 Å"
    ),
}
add(
    "filter_width",
    0.3,
    "nm",
    "hille_pore",
    note=(
        "Approximate historical functional pore estimate; "
        "not a precise KcsA structural measurement. Diagram aperture "
        "schematic; CR ionic radii are a different radius convention."
    ),
)
sources["design"] = {
    "citation": (
        "Story camera and interaction specification (manager, "
        "November 2026). Authored visualization parameters; "
        "not biological measurements."
    ),
    "url": "",
    "opened": "Manager messages in this session",
}
for id, value, unit in [("neurons_error", 8.1, "billion neurons")]:
    add(id, value, unit, "azevedo")
for id, value, unit in [
    ("camera_smoothing", 0.12, "fraction/frame"),
    ("slider_min", 0, "nA"),
    ("concentration_min", 1, "mM"),
    ("concentration_max", 200, "mM"),
    ("stimulus_min", 0, "µA/cm²"),
    ("stimulus_max", 30, "µA/cm²"),
    ("timing_min", 1, "ms"),
    ("timing_max", 15, "ms"),
    ("ruler_soma", 20e-6, "m"),
    ("ruler_cleft", 20e-9, "m"),
    ("ruler_membrane", 5e-9, "m"),
    ("ruler_k", 0.152e-9, "m"),
]:
    add(
        id,
        value,
        unit,
        "design",
        note=(
            "Camera/ruler configuration; physical values refer to "
            "separately sourced biological facts."
        ),
    )
for id in ["brain_length", "axon_diameter"]:
    facts.append(
        {
            "id": id,
            "value": None,
            "unit": "m",
            "source": None,
            "status": "NOT VERIFIED",
            "note": ("Ruler shows name only until a primary or textbook size source is opened."),
        }
    )
camera = []
for i, (t, log, x, y) in enumerate(
    [
        (0, -1, 0.5, 0.5),
        (0.1, -4, 0.48, 0.48),
        (0.2, -5, 0.5, 0.46),
        (0.35, -8, 0.52, 0.48),
        (0.42, -8.5, 0.51, 0.49),
        (0.5, -10, 0.5, 0.5),
        (0.65, -6, 0.52, 0.48),
        (0.72, -6, 0.54, 0.48),
        (0.8, -7, 0.5, 0.5),
        (0.86, -6, 0.49, 0.49),
        (1, -2, 0.5, 0.5),
    ]
):
    for suffix, value, unit in [
        ("t", t, "fraction"),
        ("log", log, "log10(m)"),
        ("x", x, "fraction"),
        ("y", y, "fraction"),
    ]:
        add(f"camera_{i}_{suffix}", value, unit, "design")
    camera.append(
        {
            "t": f"camera_{i}_t",
            "log": f"camera_{i}_log",
            "x": f"camera_{i}_x",
            "y": f"camera_{i}_y",
        }
    )
Path(__file__).with_name("facts.json").write_text(
    json.dumps(
        {"sources": sources, "facts": facts, "camera": camera},
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)
