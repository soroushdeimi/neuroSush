# Same map, new world: the Spiking Tolman–Eichenbaum Machine

Reader transcript. Each panel’s numerical substitutions come from the generated fact ledger. The diagrams are schematics unless labelled as recorded replays.

## Chapter 0: Can spikes learn a map?

Imagine finding your way through a room using brief electrical pulses. A leaky integrate-and-fire neuron collects input, loses charge, then fires at a threshold. Connections can change with spike timing. An imposed theta rhythm opens and closes windows for firing. Can these ingredients support a map learned from experience?

What this does not show: These ingredients do not establish that a map will emerge. Illustration, not trained activity.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: opening_ingredients, theta_hz.

## Chapter 1: One place. Many places.

Watch the same journey through a room. A place cell fires most strongly near a particular location. A grid cell fires at repeated locations, forming a spatial pattern. These are firing fields: maps of where a neuron tends to be active. The pattern belongs to a cell, not to the animal’s path.

What this does not show: These idealized fields explain the terms. They are neither recordings nor learned outputs from the spiking network.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: place_grid_definition.

## Chapter 2: Follow the two routes.

The entorhinal cortex is a gateway to hippocampal memory. Its medial part, MEC, carries spatial signals; its lateral part, LEC, carries sensory information. One route passes through the dentate gyrus, DG, then hippocampal regions CA3 and CA1. Another reaches CA1 more directly. The spiking model borrows this circuit organization.

What this does not show: This is an anatomically inspired computational diagram. It leaves out many biological cell types and connections.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: entorhinal_routes.

## Chapter 2: Earlier. Or on the beat.

Theta is a repeating rhythm of neural activity. Phase means a moment within each cycle. During phase precession, a cell fires earlier in successive cycles as the animal crosses its field. Phase locking keeps spikes near the same phase. The authors compare these patterns in the second and third MEC layers.

What this does not show: The animation explains timing. It is not a recording or evidence that the network generates its own theta rhythm.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: phase_definition.

## Chapter 3: A pattern by design.

Earlier models offer useful mechanisms. Oscillatory interference combines rhythms: their overlap creates repeating peaks. A continuous attractor holds a bump of activity that moves with self-motion. Many such models build key structure into their connections or inputs. TEM asks a different question: can useful spatial codes develop while learning to remember experience?

What this does not show: This comparison concerns the model families described by the authors. It does not establish that every earlier grid model lacks learning.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: earlier_mechanisms.

## Chapter 4: Same streets. Different shops.

The original Tolman–Eichenbaum Machine, TEM, separates structure from content. Its structural code, g, tracks abstract location. Its sensory code, x, identifies what is there. Their conjunction, p, binds the two. Fast Hebbian memory, M, links patterns active together and stores their associations. Think of reusable street coordinates linked to a changing directory of shops.

What this does not show: This is the original rate model’s design. Separating these variables is an architectural choice, not a discovery made by the network.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: g_definition, x_definition, p_definition, hebbian_memory_definition.

## Chapter 4: Reuse the map. Rebind the places.

When the sensory scenery changes, a useful map need not start again. Whittington and colleagues report reusable grid-like structural codes alongside place-like codes that remap. A place field can move while its relationship to the grid remains organized. Slow learning supplies reusable structure; fast memory links that structure to the current world.

What this does not show: This illustration explains the original TEM result. It does not demonstrate that the spiking version inherits every result of that model.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: original_remapping.

## Chapter 5: A circuit that predicts.

The spiking version has a prediction route and an inference route. Prediction uses the previous spatial state and movement to anticipate activity, then sensory input. Inference combines current senses with internal predictions. The hippocampal circuit binds them. Importantly, the authors changed the original TEM’s multiplicative binding because it failed to produce grids here.

What this does not show: This is a related architecture with changes. It does not establish that the original TEM’s full range of relational abilities survives.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: spiking_architecture.

## Chapter 5: Gain turns the volume.

Each spiking unit accumulates input and leaks toward rest. Crossing a threshold emits a binary spike and resets the voltage. A learned gain scales incoming current, like turning a volume knob. Imposed theta inhibition periodically suppresses firing. These choices let the authors study when spikes occur as well as where cells fire.

What this does not show: Gain is a trained scalar, not a simulated chemical pathway. Theta is supplied to the network; it is not an emergent clock.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: opening_ingredients, theta_hz.

## Chapter 5: Timing leaves a memory.

Spike-timing-dependent plasticity, STDP, changes a connection according to the order of nearby spikes. It supplies the fast associative memory in this model. Like a recurrent neural network, the other weights learn by backpropagation through time. A surrogate gradient supplies a usable training signal near the firing threshold, even though the forward computation still emits discrete spikes.

What this does not show: Local timing plasticity is only part of the learning system. This schematic window does not resolve the paper’s depression-sign convention.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: spiking_training.

## Chapter 6: Many places. Repeated clues.

The task is a random walk through a small square arena. Its 64 positions share only 20 sensory neurons, so the same observation can occur in different places. Movement supplies another clue. The network learns to predict observations from this history. Like remembering identical hotel doors by tracking the turns between them.

What this does not show: Sensory labels are arbitrary. This schematic walk is illustrative; it is not a training run, an animal trajectory, or continuous locomotion.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: arena_side, arena_positions, sensory_neurons, task_definition.

## Chapter 7: Learning changes the fields.

The authors report that 59.6 ± 18.0% of MEC cells in layer II passed their gridness cutoff of 0.8 after training. That is a mean and standard deviation across 4 simulation runs. The patterns developed with learning. Crucially, the authors describe square or diamond patterns, not the canonical hexagonal grids seen in animals.

What this does not show: This does not establish biological hexagonal grids. The forming field is a schematic; the percentage and spread are reported measurements.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: grid_mean, grid_sd, paper_runs, grid_cutoff, paper_geometry.

## Chapter 7: Place-like firing needs context.

The authors show localized CA1 firing fields. But spatial selectivity changed little between the beginning and end of training. Removing theta inhibition at the beginning of training reduced spatial information. That makes timing important within this setup, while weakening the simple story that learning alone created all the place-like selectivity.

What this does not show: This does not test switching theta off after training. The authors’ comparison begins without theta; the illustrated field is not recorded activity.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: ca1_theta.

## Chapter 7: New clues. New places.

Keeping the arena geometry and model weights fixed, the authors changed sensory inputs. Grid-like patterns realigned, while CA1 fields moved: remapping. They also compared each cell’s gridness or spatial information across the environments. Similar scores can coexist with different field locations. Preserving selectivity is not the same as preserving the map.

What this does not show: This does not measure identical field locations across worlds. The animation is schematic; the paper shows examples and score correlations from two model initializations.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: spiking_remapping.

## Chapter 7: Remove a piece. Watch what changes.

An ablation removes one mechanism. In the authors’ tests, removing gain, STDP, theta inhibition, or the sparsity constraint reduced grid classification. Removing feedback produced a smaller, statistically unresolved difference. These comparisons identify dependencies of the implemented system. They do not show that every corresponding mechanism is necessary for grid cells in the brain.

What this does not show: Four runs per condition leave substantial uncertainty. A nonsignificant feedback comparison does not establish that feedback has no effect.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: grid_mean, grid_sd, no_feedback, no_feedback_sd, no_theta, no_theta_sd, no_stdp, no_stdp_sd, no_sparsity, no_sparsity_sd, mse, mse_sd, no_gain, no_gain_sd, feedback_p, paper_runs.

## Chapter 7: Ambiguity gives movement a job.

If every location looks unique, an observation can identify a position. If many places look alike, movement history becomes useful. The authors varied the sensory vocabulary and found the most grid-like classifications at moderate ambiguity. This links the representation to the prediction task, rather than treating grid patterns as an end in themselves.

What this does not show: This does not establish an optimal sensory vocabulary for animals. Figure 7’s percentages also differ from the baseline table; those protocols should not be silently equated.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: ambiguity, arena_side, sensory_neurons.

## Chapter 7: A code for the next place.

MEC layer III is trained to predict the next activity of layer II. The authors find its spatial pattern looks more grid-like when spikes are assigned to the next position rather than the current one. This is evidence consistent with anticipation, interpreted in light of a circuit explicitly trained to predict ahead.

What this does not show: This does not show prediction arising without predictive supervision. The moving marker is a schematic, not the authors’ spike data.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: predictive_grid, arena_side, sensory_neurons.

## Chapter 7: The rhythm shapes the sequence.

On a linear track, gain and theta inhibition together produced different timing in the two MEC layers. The authors report 80.2% phase precession in layer II and 86.7% phase locking in layer III. Changing the modulation changed this balance. The result ties spike timing to specific supplied and learned mechanisms.

What this does not show: These percentages describe model grid cells. This does not demonstrate a spontaneously generated theta rhythm, or timing independent of gain and inhibition settings.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: precession_mean, locking_mean.

## Chapter 7: A familiar sign. A limited analogy.

The authors find predominantly negative learned weights between some entorhinal and hippocampal modules and interpret them as inhibition. That offers a mechanistic clue about how this network works. Yet a negative effective weight is a mathematical influence. It need not correspond to a direct inhibitory projection made by a particular biological cell type.

What this does not show: This does not establish a complete biological wiring diagram. Effective signed weights can summarize pathways that would require additional cells in a physiological circuit.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: learned_signs.

## Chapter 8: What remains unfinished?

The authors name important limits. Their spatial patterns are square or diamond shaped. Grid phases lack biological diversity, and the population does not reproduce a toroidal organization: a doughnut-shaped arrangement of activity states. Temporal bins are coarse. Training depends on backpropagation through time, whose biological implementation remains disputed. These boundaries matter when interpreting emergence.

What this does not show: This does not establish a complete biological map-making mechanism. A useful computational demonstration can still leave geometry, population organization, and learning mechanisms unresolved.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: paper_geometry, paper_limitations.

## Chapter 8: Read the numbers together.

The paper reports 59.6% grid classification in its baseline table. At the same sensory population size, the deck author reads approximately 26% from Figure 7. The protocols are not fully reconciled. Table 1 lists 400 dentate neurons; the discussion instead counts silent cells out of 200. These are reporting discrepancies worth clarifying.

What this does not show: This does not identify which setting or count is correct. The Figure 7 percentage is approximate, read from the figure, not a tabulated measurement.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: grid_mean, sensory_neurons, fig7_approx, dg_table, dg_discussion.

## Chapter 8: A formula also needs a check.

Two implementation details deserve care. The STDP equation places a minus sign before the depression amplitude, while the parameter table gives that amplitude a negative value. The training loss also prints the dentate activity penalty twice. These observations identify ambiguities in the text. They do not determine what the released code actually executes.

What this does not show: This does not justify silently correcting the model. The intended sign and duplicated term require checking the matching implementation before reproduction.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: equation_ambiguities.

## Chapter 8: Built in, learned, or measured?

The architecture supplies named regions, prediction targets, theta inhibition, and a sparsity objective. Training adjusts weights and gain. Spatial fields and timing are then measured. Keeping these categories separate makes emergence a precise claim. The strongest next tests would vary assumptions, repeat across more runs, and clarify the scoring and reporting details.

What this does not show: This does not reduce the result to a hand-drawn pattern. It asks which observations depend on which supplied mechanisms and how reliably they recur.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: spiking_architecture, spiking_training, paper_runs, paper_limitations.

## Chapter 8: What does a score reward?

Our separate calculation applies a standard hexagonal gridness score to ideal lattices. It compares rotational similarity at selected angles. Our square example scores -1.215; our hexagonal example scores 1.310. These values depend on sampling and the scoring ring. They illustrate this metric, not the score used for the paper’s reported cells.

What this does not show: OUR COMPUTATION on ideal patterns, not trained models. The paper defines its score in supplementary S1, which we could not read; we cannot equate implementations.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: ideal_square_score, ideal_square_score_display, ideal_hex_score, ideal_hex_score_display, ideal_maps, ideal_map_bins, ideal_scale, ideal_annulus_inner, ideal_annulus_outer.

## Chapter 9: Now follow our recording.

In neuroSush, we built and tested a TEM implementation. Here you can replay frozen models exploring new worlds with initially empty memory. The path and activity maps come from recorded runs. They show what those models did in those worlds. They do not establish what caused the activity or whether it represents learned geometry.

What this does not show: REPLAY, not a live model. These small worlds cannot resolve gridness. The spatial activity maps summarize the whole recorded walk; they are not instantaneous firing.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: REPLAY. Fact IDs: our_replay.

## Chapter 9: A strong-looking code needs a control.

Our structural code showed similarities across worlds. But path integrators with random transition weights reached the same range. Across six runs, margins over the control’s upper tail ranged from -0.244 to +0.010. Our preset rule required at least 0.15 in every seed. It was not met. The comparison limits the learning claim.

What this does not show: This is not evidence for grid-cell emergence in our models. Theta is imposed, gridness is unresolved in these worlds, and this is not a ranking against the paper.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: v2_s0_margin, v2_s1_margin, v2_s2_margin, v1_s0_margin, v1_s1_margin, v1_s2_margin, margin_threshold.

## Chapter 9: What the control does mean.

The control comparison uses representational similarity: whether pairs of positions have similar relationships across worlds. A coherent path integrator can create that structure without learning it from sensory experience. Our control percentile comes from a small sample and lies near its maximum. Falling below this upper tail does not mean falling below typical controls.

What this does not show: This does not establish that memory is useless. It says this measure failed to separate learned structure from the selected path-integration controls; larger-world probes remain unrun.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: our_control_caveats.

## Chapter 10: A map is more than a pattern.

The spiking TEM connects spatial prediction, associative memory, and spike timing in one computational system. Its authors report learned spatial patterns and timing shaped by gain and supplied theta. The next question is sharper than whether a picture resembles a grid: which mechanisms explain reusable knowledge, under which controls, and with what biological limits?

What this does not show: A schematic explains an idea. A replay shows a recorded run. A convincing scientific claim also needs a defined measure, appropriate controls, and reproducible evidence. Open Sources to follow the evidence.

Evidence: [verified source ledger](facts.json), using the fact IDs below.
Provenance: SCHEMATIC. Fact IDs: spiking_architecture, spiking_training, paper_geometry, paper_limitations, our_control_caveats, arena_side, sensory_neurons.
