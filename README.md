# Quantum Circuit Compilation: Optimal Qubit Routing & SWAP Minimization

Course optimization project. A quantum computer can only run a two-qubit gate on qubits that are physically wired together. We decide **where each logical qubit starts on the chip** and **which SWAP gates to insert**, so a circuit runs on IBM hardware (FakeBrisbane: 127 qubits, 144 couplings) with as few SWAPs as possible.

**Team:** Harit Goyal (BC2025037), Daksh Bansal (BC2025025), Tanishq Bansal (BC2025100), Kavya Singla (BC2025042)

## Quick start

You need **Python 3.12 or newer** (tested on 3.14) and **git**.

**Windows (PowerShell)**
```powershell
git clone https://github.com/dakshbansal-2417/quantum-qubit-routing.git
cd quantum-qubit-routing
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/get_data.py
python route.py data/QASMBench/small/adder_n4/adder_n4.qasm
```

**macOS / Linux**
```bash
git clone https://github.com/dakshbansal-2417/quantum-qubit-routing.git
cd quantum-qubit-routing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_data.py
python route.py data/QASMBench/small/adder_n4/adder_n4.qasm
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` from Command Prompt.

The last command routes one circuit and prints the easy solution, the SABRE baseline, the hard model's result with its proof of optimality, and the three verification checks. It takes a few seconds.

## Using the model

```bash
python route.py <circuit.qasm>                     # easy + SABRE + hard + verification
python route.py <circuit.qasm> --plot out.png      # also draw easy vs hard layouts
python route.py <circuit.qasm> --json out.json     # save all results
python route.py <circuit.qasm> --block 20          # size of the chip block (default max(n+4, 2n))
python route.py <circuit.qasm> --full-chip         # all 127 qubits (exact methods only if small)
```
Any OpenQASM 2 file works, not only QASMBench.

## Reproducing all results

```bash
python experiments/run_all.py      # 26 small circuits, chip blocks, exact methods (~1.5 h)
python experiments/run_scale.py    # small + medium circuits on the full 127-qubit chip (~1 h)
python analysis/plots.py           # figures in analysis/figures/, tables in analysis/summary.md
```
The committed `experiments/results/` already contains the outputs, so `analysis/plots.py` runs without the long experiments. Solver times differ between machines; with the default time limits, SWAP counts and proofs match the committed results.

## The two models

**Easy (convex QP, closed form).** Logical qubit `i` gets a continuous position `x_i` in the plane:
`min tr(XᵀLX)` subject to `1ᵀX = 0`, `XᵀX = I`, where `L` is the Laplacian of the interaction counts `w_ij`. The constraints stop all qubits collapsing onto one point. The global optimum is given by the 2nd and 3rd eigenvectors of `L` (certificate: objective = λ₂ + λ₃). Positions are snapped onto physical qubits with a linear assignment. It ignores the wiring and the gate order.

**Hard (MILP).** Binary `x[i,p,t]` (qubit `i` on physical qubit `p` at time `t`) and `s[e,t]` (SWAP on wire `e` at time `t`). Minimize `Σ s[e,t]` subject to: one place per qubit, at most one qubit per place, every gate on a wired pair when it runs, SWAPs in a round disjoint, SWAPs move qubits and otherwise qubits stay, gate dependencies respected (gates on different qubits may run in parallel). Solved with HiGHS (branch and bound), warm-started from a greedy router or SABRE.

**Exact search.** The same hard problem solved by breadth-first search over (placement, gate progress), with lower-bound pruning. It gives the true optimum when the circuit is small enough and certifies the MILP's answer.

**Verification.** Every solution is replayed step by step, checked with Qiskit's `CheckMap`, and for blocks up to 11 qubits proven to be the same unitary as the original circuit.

## Results

See [analysis/summary.md](analysis/summary.md) and [analysis/figures/](analysis/figures/). In short: on 26 small circuits the hard model is proven optimal on 22. The easy model's layout needs on average about twice as many SWAPs as the hard optimum (up to 4×). Exact methods stop scaling at about 6–8 logical qubits; on the full 127-qubit chip only heuristics remain practical.

## Repository layout

| Path | What it is |
|---|---|
| `route.py` | Run the whole model on one circuit |
| `src/load_data.py` | FakeBrisbane graph, hop distances, QASMBench parsing into ordered CNOT lists, chip blocks |
| `src/easy_qp.py` | Easy model: spectral placement, Ky Fan certificate, LP snapping |
| `src/hard_milp.py` | Hard model: time-expanded MILP (HiGHS), greedy router, warm starts |
| `src/exact_search.py` | Exact search: true optimum and lower bounds for small circuits |
| `src/verify.py` | Replay, Qiskit `CheckMap`, unitary equivalence, SABRE baseline |
| `experiments/` | Experiment scripts and their results (`results/`) |
| `analysis/` | Plots and the results summary |
| `scripts/get_data.py` | Downloads the pinned QASMBench circuits |
| `docs/` | Course guidelines and the proposal presentation |
| `ROADMAP.md` | Project steps and status |

## Data
- **Hardware:** IBM `FakeBrisbane` from `qiskit-ibm-runtime` (a snapshot of the real device's coupling map; no IBM account needed).
- **Circuits:** [QASMBench](https://github.com/pnnl/QASMBench) small and medium suites, pinned to commit `357b942`.
