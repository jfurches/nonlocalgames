This file provides details on data inside this folder.

`seeds.txt`: List of random seeds to use, generated from random.org

### N-partite symmetric data

We only give data for $N=6$, but the code can be run for other player counts.

```
nps/
 |
 +--- nps_6.csv: Table of energy per iteration per trial for N=6
 |
 +--- nps_dpo.json: More detailed version, contains ansatz, phi, and energy
```

### G14 data

```
g14_<mode>/
 |
 +--- g14_state.json: Contains best ansatz, measurement settings, and metrics
 |
 +--- g14_trials.csv: Contains table of metrics for each trial, some are incomplete
 |
 +--- g14_responses[_ibm_device].csv: Breakdown of the answer counts per
        question [on a particular quantum device, if not specified, it's AerSimulator]
 |
 +--- g14_winrate[_ibm_device].csv: Aggregation of g14_responses.csv to
        the winrate per question
```

The possible modes are as follows:
- `constrained_ry`: Nonviolation hamiltonian, fixing Alice and Bob's measurement parameters
    to Bob = Alice*, with Ry measurement layer

These modes are not described in the paper:
- `balanced`: Balanced violation hamiltonian, independent measurement parameters
- `balanced_adapt_1e-6`: Balanced violation hamiltonian, independent measurement
    parameters, tighter adapt pool convergence criterion
- `imbalanced`: Violation hamiltonian without correcting for question imbalance or
    without the factor $p(q)$ in the sum.