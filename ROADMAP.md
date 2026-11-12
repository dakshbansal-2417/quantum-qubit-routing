# Project Roadmap

Each step ends with a commit + push to GitHub, so the repo always shows a working state.

**Deadlines (from course guidelines)**
- Milestone 1, Proposal & Motivation (10%): ✅ presented
- Milestones 2–3, Formulation (10%) + Solution (5%) + Analysis (5%): 10-min presentation, early November
- Final report + code on LMS: early December

| # | Step | Output | Grading link | Status |
|---|---|---|---|---|
| 1 | Project setup | repo, README, folder layout, proposal deck in `docs/` | — | ✅ pushed |
| 2 | Data pipeline | `src/load_data.py`: FakeBrisbane graph, QASMBench parsing, chip regions | Dataset description | ✅ pushed |
| 3 | Easy model | `src/easy_qp.py`: spectral placement + Ky Fan certificate + LP snapping | Formulation (easy) | ✅ pushed |
| 4 | Hard model | `src/hard_milp.py`: time-expanded SWAP MILP (HiGHS) | Formulation (hard) | ✅ pushed |
| 5 | Verification + baseline | `src/verify.py`: replay, CheckMap, unitary check, SABRE | Solution: "verification needed" | ✅ pushed |
| 6 | Experiments | `experiments/run_all.py` → `results.csv`, `solutions.json` | Solution: objective + key variables | ✅ pushed |
| 7 | Analysis & plots | `analysis/`: layouts on chip, easy vs hard vs SABRE bars, solve time vs size, price of realism | Analysis | ⬜ |
| 8 | Formulation + results slides | `docs/` deck for the 10-min presentation | Milestones 2–3 | ⬜ |
| 9 | Reproducibility | pinned `requirements.txt`, data download script, step-by-step README for Windows/macOS/Linux, tested on a fresh clone | Code submission | ⬜ |
| 10 | Final report | `report/`: formulations, results, verification, discussion | Final report | ⬜ |

## Step details

**2. Data pipeline.** Load the 127-qubit / 144-edge coupling graph and all-pairs hop distances. Parse each OpenQASM circuit into an ordered list of CNOT pairs and interaction counts `w_ij`. Choose a connected region of size `max(n+4, 2n)` around the chip centre, shared by both models.

**3. Easy model.** `min tr(XᵀLX)` s.t. `1ᵀX = 0`, `XᵀX = I`. Exact optimum from Laplacian eigenvectors. Certificate: objective = λ₂ + λ₃. Snap to physical qubits with linear assignment.

**4. Hard model.** Binary `x[i,p,t]` and `s[e,t]`. Minimize total SWAPs subject to placement, adjacency, disjoint-SWAP, movement and dependency-order constraints. Gates on disjoint qubits may share a time slot. Strengthening cuts: two-way adjacency, no SWAP of two empty spots. Warm start from the better of a lookahead greedy router and SABRE, so the MILP is never worse than SABRE. Optional fixed initial layout (easy layout + optimal routing).

**4b. Exact search (certifier).** `src/exact_search.py` solves the same hard problem by breadth-first search over (placement, gate progress) states, one SWAP per move, with admissible pruning. It needs no slot structure and no round limit, so its answer is the true optimum. The MILP constraints are unchanged: the search only supplies a starting solution and a proven lower bound. Feasible when the number of initial placements m!/(m-n)! is at most 1.5 million (n ≤ 6 here); always feasible for a fixed layout.

**5. Verification.** Every solution is replayed by hand, checked with Qiskit `CheckMap`, and (for regions ≤ 11 qubits) proven unitarily equivalent to the original circuit. SABRE (best of 20 seeds) is the industry baseline.

**6. Experiments.** All QASMBench small circuits with ≥ 3 qubits and ≤ 45 distinct two-qubit gates, 120 s solver limit. Also re-solve with R+1 SWAP rounds per gap to show R is large enough.
Result: 26 circuits, all verified. Hard model proven optimal on 22 (17 by the MILP alone, 5 more by the exact search); easy layout + optimal routing proven optimal on 24. Still open: dnn_n8, ising_n10, qpe_n9 (too many initial placements for the search) and qaoa_n6 (best 8, proven ≥ 4). The easy layout costs up to ~3× the hard optimum (e.g. error_correctiond3_n5 9 vs 3, qec_en_n5 4 vs 1, adder_n4 4 vs 2).

**7. Analysis.** Price of realism = easy-ideal (0 SWAPs) vs hard optimum. Value of joint optimization = easy layout + optimal routing vs hard optimum. Success-probability estimate `(1-ε)^(3·SWAPs)`. Scaling: MILP size and time vs circuit size, and where it stops being tractable.

**9. Reproducibility.** Goal: a teammate clones the repo, follows the README, and gets the same `results.csv`.
