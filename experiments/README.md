# Experiments

Scripts that produce the numbers behind the validation claims, with every parameter and
seed written next to the results. Each runs on a CPU in a few minutes.

| script | what it measures |
|---|---|
| `sequence_learning.py` | learning curves (bursting columns per element and repetition) of the spiking sequence memory and of the temporal memory on sequences that share their middle |

Run a script from the repository root with the package installed (`pip install -e .`), for
example `python experiments/sequence_learning.py --out results.json`; `--help` lists its
parameters.
