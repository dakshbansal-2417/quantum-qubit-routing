# Quantum Circuit Compilation: Optimal Qubit Routing & SWAP Minimization

Course optimization project. We pick where each logical qubit starts on the chip and which SWAP gates to insert, so that a quantum circuit can run on IBM hardware (FakeBrisbane: 127 qubits, 144 coupling edges) with as few SWAPs as possible.

**Team:** Harit Goyal (BC2025037), Daksh Bansal (BC2025025), Tanishq Bansal (BC2025100), Kavya Singla (BC2025042)

## Data
- **Hardware:** IBM `FakeBrisbane` coupling map (`qiskit_ibm_runtime.fake_provider`)
- **Circuits:** [QASMBench](https://github.com/pnnl/QASMBench) OpenQASM benchmarks (small/medium)

## Formulations
- **Easy:** spectral / quadratic placement (continuous positions), snapped to hardware via linear assignment
- **Hard:** time-expanded MILP: binary placement `x[i,p,t]` and SWAP `s[e,t]` variables, minimizing total SWAPs under adjacency and gate-order constraints

## Setup
```bash
python -m venv .venv            # needs Python 3.10–3.14
.venv/Scripts/activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/pnnl/QASMBench.git data/QASMBench
git -C data/QASMBench sparse-checkout set small
```

## Run
```bash
python src/load_data.py                 # list circuits and their sizes
python experiments/run_all.py           # easy + hard + SABRE on all small circuits
```
Results go to `experiments/results/results.csv` (one row per circuit) and `solutions.json` (layouts and SWAP sequences).

| Module | What it does |
|---|---|
| `src/load_data.py` | FakeBrisbane coupling graph, hop distances, QASMBench parsing into ordered CNOT lists |
| `src/easy_qp.py` | Spectral placement (global optimum via Laplacian eigenvectors, Ky Fan certificate) + LP snapping to the chip |
| `src/hard_milp.py` | Time-expanded MILP over placement `x[i,p,t]` and SWAP `s[e,t]` variables, solved with HiGHS |
| `src/exact_search.py` | Exact breadth-first search of the same hard problem: true optimum over all slot structures, proven lower bound, warm start for the MILP |
| `src/verify.py` | Replay check, Qiskit `CheckMap`, unitary equivalence, SABRE baseline |

## Layout
```
data/         chosen circuit subset, cached coupling map
src/          data loading, easy QP, hard MILP, verification, SABRE baseline
experiments/  run scripts and results
analysis/     plots and notebooks
report/       final report
docs/         guidelines and presentations
```
