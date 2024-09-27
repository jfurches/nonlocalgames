# Non-Local Games

![G14 strategy circuit](https://github.com/jfurches/nonlocalgames/assets/38408451/633f61dd-b0e0-41e3-b726-aa40c58fb9c6)

Code for _Variational Methods for Computing Non-Local Quantum Strategies_ ([arXiv](https://arxiv.org/abs/2311.01363)).

### Circuits

If you're interested in the strategy circuits for benchmarking hardware or any other applications, they can be found in the `circuits/` folder. There's 3 subfolders:

- `4q/`: The original perfect strategy with the large 4-qubit unitary operators.
- `bell_pair/`: The same strategy as `4q`, except that the state preparation circuit has been greatly reduced to 2 Bell pairs.
- `ghz/`: **(Don't use this one)** ~~An imperfect strategy (V(G) = 0.9921) with the GHZ state shared instead.~~

Each folder contains an OpenQASM file for each possible question pair, with the filenames taking the form `circuit_{va}_{vb}.qasm`. `va` and `vb` correspond to vertex indices on the G14 graph.

### Examples

The `examples/` folder contains different scripts that use the nonlocal games library.

- `dpo/`: Script for executing multiple trials of DPO on the NPS game in parallel
- `g14/`: Program that runs multiple trials of DPO on G14 in parallel
- `g14_on_ibm/`: Dispatches a strategy for G14 onto multiple IBM quantum devices. See this if you're interested in loading the G14 circuit. Also contains examples for zero-noise extrapolation (ZNE), but this is not in the paper.

### Data

Look at `data/metadata.md` for details.

### Installation

You can install this package with `$ pip install .`

This package depends on a custom implementation of ADAPT-VQE called `adaptgym` that's tailored to another project (currently private code). You can find an implementation of ADAPT [here](https://github.com/nmayhall-vt/adapt-vqe) or roll your own. When you see code like

```python
env = AdaptGame(...)
env.reset()
while not done:
    env.step(0)
```

this is equivalent to running ADAPT, with each `step()` call corresponding to adding the operator with the highest gradient.

### Testing

This package has a few tests that rely on the pytest framework. You can run them with `$ pytest`.
