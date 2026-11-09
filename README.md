# Quantum Circuit Compilation: Optimal Qubit Routing & SWAP Minimization

Course optimization project. We pick where each logical qubit starts on the chip and which SWAP gates to insert, so that a quantum circuit can run on IBM hardware (FakeBrisbane: 127 qubits, 144 coupling edges) with as few SWAPs as possible.

**Team:** Harit Goyal (BC2025037), Daksh Bansal (BC2025025), Tanishq Bansal (BC2025100), Kavya Singla (BC2025042)

## Data
- **Hardware:** IBM `FakeBrisbane` coupling map (`qiskit_ibm_runtime.fake_provider`)
- **Circuits:** [QASMBench](https://github.com/pnnl/QASMBench) OpenQASM benchmarks (small/medium)

## Formulations
- **Easy:** spectral / quadratic placement (continuous positions), snapped to hardware via linear assignment
- **Hard:** time-expanded MILP: binary placement `x[i,p,t]` and SWAP `s[e,t]` variables, minimizing total SWAPs under adjacency and gate-order constraints

## Layout
```
data/         chosen circuit subset, cached coupling map
src/          data loading, easy QP, hard MILP, verification, SABRE baseline
experiments/  run scripts and results
analysis/     plots and notebooks
report/       final report
docs/         guidelines and presentations
```
