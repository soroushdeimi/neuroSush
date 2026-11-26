"""Generate the sourced Spiking TEM fact ledger; model units are not SI units."""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch

from neurosush.htm.grid_cells import hexagonal_rate
from neurosush.spatial import gridness


def ideal_lattices():
    """Use the existing 49-bin library reference-test geometry."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        axis = torch.arange(49, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        maps = {
            "square": 2 + torch.cos(2 * math.pi * x / 10) + torch.cos(2 * math.pi * y / 10),
            "hex": hexagonal_rate(torch.stack((x, y), -1), scale=10, orientation=0),
        }
        scores = {name: gridness(rate, annulus=(5, 15)) for name, rate in maps.items()}
        return maps, scores
    finally:
        torch.set_num_threads(previous_threads)


FACTS = [
    {
        "id": "baseline_accuracy",
        "value": 0.1027,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "training_updates",
        "value": 16000,
        "unit": "updates",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v0_mean",
        "value": 0.236,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v0_sd",
        "value": 0.15,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_mean",
        "value": 0.721,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_sd",
        "value": 0.038,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_mean",
        "value": 0.769,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_sd",
        "value": 0.007,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v3_v0_mean",
        "value": 0.832,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v3_v0_sd",
        "value": 0.014,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v3_v1_mean",
        "value": 0.783,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v3_v1_sd",
        "value": 0.048,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_s0_rsa",
        "value": 0.779,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_s1_rsa",
        "value": 0.642,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_s2_rsa",
        "value": 0.807,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_s0_rsa",
        "value": 0.85,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_s1_rsa",
        "value": 0.794,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_s2_rsa",
        "value": 0.93,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "control_p95_min",
        "value": 0.84,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "control_p95_max",
        "value": 0.954,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_s0_margin",
        "value": -0.103,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_s1_margin",
        "value": -0.244,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v2_s2_margin",
        "value": -0.087,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_s0_margin",
        "value": 0.01,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_s1_margin",
        "value": -0.054,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "v1_s2_margin",
        "value": -0.024,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "margin_threshold",
        "value": 0.15,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "world_size_side",
        "value": 16,
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md, section 6.2",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "whittington_citation",
        "value": "Whittington et al. 2020, Cell 183, 1249-1263",
        "unit": "",
        "source": "Whittington et al. 2020, Cell 183, p. 6",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "g_definition",
        "value": (
            "Abstract relational structure / position (Medial Entorhinal Cortex interpretation)"
        ),
        "unit": "",
        "source": "Whittington et al. 2020, Cell 183, p. 4, Fig. 2",
        "type": "DEFINITION",
        "status": "VERIFIED",
    },
    {
        "id": "x_definition",
        "value": "Sensory identity (Lateral Entorhinal Cortex interpretation)",
        "unit": "",
        "source": "Whittington et al. 2020, Cell 183, p. 4, Fig. 2",
        "type": "DEFINITION",
        "status": "VERIFIED",
    },
    {
        "id": "p_definition",
        "value": "Conjunction of g and x (Hippocampal interpretation / place cells)",
        "unit": "",
        "source": "Whittington et al. 2020, Cell 183, p. 4, Fig. 2",
        "type": "DEFINITION",
        "status": "VERIFIED",
    },
    {
        "id": "hebbian_memory_definition",
        "value": "Fast associative memory M binding p at location g",
        "unit": "",
        "source": "Whittington et al. 2020, Cell 183, p. 23",
        "type": "DEFINITION",
        "status": "VERIFIED",
    },
    {
        "id": "generative_factorisation_eq",
        "value": (
            "p(x_{1:T}, p_{1:T}, g_{1:T}) = \\prod_{t=1}^T "
            "p(x_t | p_t) p(p_t | M_{t-1}, g_t) p(g_t | g_{t-1}, a_t)"
        ),
        "unit": "",
        "source": "Whittington et al. 2020, Cell 183, STAR Methods e5 (PDF p. 21)",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "hebbian_memory_update_eq",
        "value": "M_t = \\lambda M_{t-1} + \\eta (p_t - \\hat{p}_t)(p_t + \\hat{p}_t)^T",
        "unit": "",
        "source": "Whittington et al. 2020, Cell 183, STAR Methods e7 (PDF p. 23)",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "theta_hz",
        "value": 8,
        "unit": "Hz",
        "source": "Kawahara & Fujisawa 2025, PDF p. 3, Eq. 5",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "opening_ingredients",
        "value": (
            "LIF threshold/reset, STDP associative memory, "
            "imposed theta inhibition; most weights use backpropagation"
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF pp. 3-4, Eqs. 1-5 and 9; p. 8, section 2.3.2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "place_grid_definition",
        "value": (
            "Place cells encode location-specific activity; "
            "MEC grid cells exhibit periodic spatial firing."
        ),
        "unit": "",
        "source": (
            "Kawahara & Fujisawa 2025, PDF p. 1, Author Summary; p. 2, Introduction lines 35-38"
        ),
        "type": "DEFINITION",
        "status": "VERIFIED",
    },
    {
        "id": "entorhinal_routes",
        "value": (
            "DG receives MECII and LECII; CA3 combines DG, MECII, LECII and recurrent CA3; "
            "CA1 combines CA3 and LECIII."
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 4, section 2.2.2; p. 5, Fig. 1",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "phase_definition",
        "value": (
            "Precession shifts spikes earlier in theta; "
            "phase locking concentrates spikes at a phase. "
            "MECII precession and MECIII locking are compared with experimental findings."
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 2, lines 55-59; p. 17, lines 337-352",
        "type": "DEFINITION",
        "status": "VERIFIED",
    },
    {
        "id": "earlier_mechanisms",
        "value": (
            "Oscillatory interference combines theta-modulated inputs; continuous attractors "
            "maintain a moving activity bump. "
            "Many rely on handcrafted architecture or idealized input."
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 2, Introduction lines 40-50",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "original_remapping",
        "value": (
            "Original TEM structural grid-like codes transfer; place-like codes remap while "
            "preserving relationships to structural codes."
        ),
        "unit": "",
        "source": "Whittington et al. 2020, PDF pp. 8-10, Figs. 4 and 6",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "spiking_architecture",
        "value": (
            "Prediction: MECII plus action to MECIII, memory-assisted CA1 to MECV to senses. "
            "Inference: senses and previous MECIII to MECII, DG/CA3/CA1 circuit. "
            "Original multiplicative binding omitted after failure to produce grids."
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF pp. 4-5, Fig. 1; p. 8, lines 173-177",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "spiking_training",
        "value": (
            "Associative memory uses STDP; other weights use backpropagation through time "
            "with surrogate spike gradients."
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 4, Eq. 9; p. 8, section 2.3.2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "arena_side",
        "value": 8,
        "unit": "positions per side",
        "source": "Kawahara & Fujisawa 2025, PDF p. 10, section 3.1.1",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "arena_positions",
        "value": 8**2,
        "unit": "positions",
        "formula": "arena_side ** 2",
        "source": "Kawahara & Fujisawa 2025, PDF p. 15, lines 295-298",
        "type": "DERIVED",
        "status": "VERIFIED",
    },
    {
        "id": "sensory_neurons",
        "value": 20,
        "unit": "sensory neurons",
        "source": "Kawahara & Fujisawa 2025, PDF p. 9, Table 1, 2D setting",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "task_definition",
        "value": (
            "Random exploration with up/down/left/right/stay actions; arbitrary one-hot "
            "sensory labels repeat across locations. No rewards."
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF pp. 7-10, Algorithm 1 and section 3.1.1",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "grid_mean",
        "value": 59.6,
        "unit": "percent of MECII units",
        "source": "Kawahara & Fujisawa 2025, PDF p. 10, section 3.1.1; p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "grid_sd",
        "value": 18.0,
        "unit": "percentage points, SD across runs",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "paper_runs",
        "value": 4,
        "unit": "simulation runs per condition",
        "source": "Kawahara & Fujisawa 2025, PDF p. 11, Fig. 3C; p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "grid_cutoff",
        "value": 0.8,
        "unit": "gridness score",
        "source": "Kawahara & Fujisawa 2025, PDF p. 10, lines 219-220; S1 definition unavailable",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "ca1_theta",
        "value": "CA1 spatial-information distribution changed little with training; "
        "removing theta at the beginning of training lowered spatial information.",
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 10, lines 225-233; p. 12, Fig. 4",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "spiking_remapping",
        "value": "Changing one-hot to two-hot observations with fixed geometry and "
        "parameters realigns MECII fields and remaps CA1 fields. Two model "
        "initializations.",
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF pp. 12-15, section 3.1.2, Figs. 5-6",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_feedback",
        "value": 43.4,
        "unit": "percent of MECII units",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_theta",
        "value": 25.6,
        "unit": "percent of MECII units",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_stdp",
        "value": 10.1,
        "unit": "percent of MECII units",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_sparsity",
        "value": 0.875,
        "unit": "percent of MECII units",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "mse",
        "value": 0.25,
        "unit": "percent of MECII units",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_gain",
        "value": 0.0,
        "unit": "percent of MECII units",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_feedback_sd",
        "value": 17.5,
        "unit": "percentage points, SD across runs",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_theta_sd",
        "value": 20.6,
        "unit": "percentage points, SD across runs",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_stdp_sd",
        "value": 16.6,
        "unit": "percentage points, SD across runs",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_sparsity_sd",
        "value": 1.75,
        "unit": "percentage points, SD across runs",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "mse_sd",
        "value": 0.353,
        "unit": "percentage points, SD across runs",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "no_gain_sd",
        "value": 0.0,
        "unit": "percentage points, SD across runs",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, Table 2",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "feedback_p",
        "value": 0.244,
        "unit": "reported p value",
        "source": "Kawahara & Fujisawa 2025, PDF p. 13, line 288",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "ambiguity",
        "value": "Moderate sensory ambiguity, around 20 sensory units, favors grid "
        "classification in this task.",
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF pp. 15-16, section 3.1.4 and Fig. 7",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "predictive_grid",
        "value": "MECIII is trained to predict MECII one step ahead; next-position "
        "alignment increases gridness.",
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 16, section 3.1.5; p. 17, Fig. 8",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "precession_mean",
        "value": 80.2,
        "unit": "percent of MECII grid cells",
        "source": "Kawahara & Fujisawa 2025, PDF p. 17, section 3.2.1; p. 22, Fig. 13C",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "locking_mean",
        "value": 86.7,
        "unit": "percent of MECIII grid cells",
        "source": "Kawahara & Fujisawa 2025, PDF p. 17, section 3.2.1; p. 22, Fig. 13C",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "learned_signs",
        "value": "Mostly negative effective MECII to MECIII and MECIII to CA1 weights are "
        "interpreted as inhibition.",
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF pp. 18-19, section 3.3; pp. 23-24, Figs. 14-15",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "paper_geometry",
        "value": (
            "Authors report square or diamond grids, no canonical hexagons; "
            "limited phase diversity and no reproduced toroidal population structure."
        ),
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 25, lines 428-442",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "paper_limitations",
        "value": "Coarse time resolution; BPTT computational cost and disputed biological "
        "plausibility; temporal replay remains future work.",
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 25, lines 416-448",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "fig7_approx",
        "value": 26,
        "unit": "percent, approximate visual reading",
        "source": "Kawahara & Fujisawa 2025, PDF p. 16, Fig. 7, bar at 20 sensory neurons; "
        "deck author approximate reading, independently inspected",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "dg_table",
        "value": 400,
        "unit": "DG neurons",
        "source": "Kawahara & Fujisawa 2025, PDF p. 9, Table 1, Number of dentate gyrus neurons",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "dg_discussion",
        "value": 200,
        "unit": "DG neurons used as denominator",
        "source": "Kawahara & Fujisawa 2025, PDF p. 22, Discussion, lines 387-389",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "equation_ambiguities",
        "value": "Eq. 9 places minus before A-minus; Table 1 lists negative A-minus. Eq. 17 "
        "repeats the DG activity norm. Intended implementation not established.",
        "unit": "",
        "source": "Kawahara & Fujisawa 2025, PDF p. 4, Eq. 9; p. 8, Eq. 17; p. 9, Table 1",
        "type": "PAPER-CLAIM",
        "status": "VERIFIED",
    },
    {
        "id": "square_grid_score_critique",
        "value": None,
        "unit": "gridness score",
        "source": "Preprint p. 10 defers definition to S1, absent from supplied PDF",
        "type": "UNRESOLVED",
        "status": "NOT VERIFIED",
    },
    {
        "id": "our_replay",
        "value": (
            "Frozen TEM models in new worlds with empty memory; "
            "recorded paths and whole-walk activity."
        ),
        "unit": "",
        "source": (
            "RESULTS_AND_EXPLANATIONS.md sections 1, 6.1-6.2; "
            "tem-run-v1-s0.json and tem-run-v2-s0.json metadata"
        ),
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
    {
        "id": "our_control_caveats",
        "value": (
            "Control p95 has nine random-weight values and is near their maximum; "
            "coherent path integration explains cross-world g RSA. Larger-world probes not run."
        ),
        "unit": "",
        "source": "RESULTS_AND_EXPLANATIONS.md sections 6.1-6.2, Reading and Caveats",
        "type": "OUR-RESULT",
        "status": "VERIFIED",
    },
]


for fact in FACTS:
    if fact["type"] == "OUR-RESULT" and isinstance(fact["value"], (int, float)):
        if fact["id"] == "world_size_side":
            fact["unit"] = "positions per side, proposed unrun world"
        elif not fact["unit"]:
            fact["unit"] = (
                "fraction correct"
                if fact["id"] == "baseline_accuracy" or fact["id"].endswith(("_mean", "_sd"))
                else "dimensionless RSA value"
            )

ideal_maps, ideal_scores = ideal_lattices()
for fact_id, value, unit in [
    (
        "ideal_square_score",
        round(ideal_scores["square"].hexagonal, 6),
        "dimensionless standard hex score",
    ),
    (
        "ideal_square_score_display",
        f"{ideal_scores['square'].hexagonal:.3f}",
        "dimensionless, rounded",
    ),
    (
        "ideal_hex_score",
        round(ideal_scores["hex"].hexagonal, 6),
        "dimensionless standard hex score",
    ),
    ("ideal_hex_score_display", f"{ideal_scores['hex'].hexagonal:.3f}", "dimensionless, rounded"),
    (
        "ideal_maps",
        {
            name: [[round(v, 8) for v in row] for row in rate.tolist()]
            for name, rate in ideal_maps.items()
        },
        "ideal activity",
    ),
    ("ideal_map_bins", 49, "bins per side"),
    ("ideal_scale", 10, "bins per period"),
    ("ideal_annulus_inner", 5, "autocorrelation lag bins"),
    ("ideal_annulus_outer", 15, "autocorrelation lag bins"),
]:
    FACTS.append(
        {
            "id": fact_id,
            "value": value,
            "unit": unit,
            "source": (
                "OUR COMPUTATION: spatial/gridness.py gridness; htm/grid_cells.py hexagonal_rate; "
                "geometry from tests/spatial/test_gridness.py reference tests. "
                "Score=min(r60,r120)-max(r30,r90,r150); float64 ideal maps, orientation zero."
            ),
            "type": "OUR-COMPUTATION",
            "status": "VERIFIED",
        }
    )


def main():
    path = Path(__file__).with_name("facts.json")
    text = json.dumps(FACTS, indent=2, ensure_ascii=False)
    # Keep sampled field rows together for review.
    text = re.sub(
        r"\[\n(?:\s+[-+0-9.eE]+,?\n)+\s*\]",
        lambda match: json.dumps(json.loads(match.group())),
        text,
    )
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
