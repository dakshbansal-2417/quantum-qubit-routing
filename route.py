"""Run the whole model on one circuit.

    python route.py data/QASMBench/small/adder_n4/adder_n4.qasm
    python route.py my_circuit.qasm --block 20 --time-limit 60 --plot adder.png
    python route.py my_circuit.qasm --full-chip            (easy + SABRE only)

Steps
  1. load the circuit (OpenQASM 2) and IBM FakeBrisbane (127 qubits)
  2. EASY model: spectral placement, certified optimal, snapped to the chip
  3. SABRE baseline (Qiskit)
  4. HARD model: exact search (when small enough) + MILP, with proven lower bound
  5. verification: replay, Qiskit CheckMap, unitary equivalence (small blocks)
Results are printed; --json saves them, --plot draws easy vs hard layouts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from easy_qp import solve_easy  # noqa: E402
from exact_search import exact_route, to_warm_start  # noqa: E402
from hard_milp import greedy_warm_start, model_size, solve_hard  # noqa: E402
from load_data import load_circuit, load_hardware, region, subgraph  # noqa: E402
from verify import check_map, equivalent, replay, routed_circuit, sabre  # noqa: E402


def route(path, block=None, full_chip=False, time_limit=120.0, search_limit=300.0,
          seeds=20, quiet=False):
    say = (lambda *a: None) if quiet else print
    work = load_circuit(path)
    hw = load_hardware()
    if full_chip:
        nodes, sub = list(range(hw.num_qubits)), hw
    else:
        size = block or max(work.n + 4, 2 * work.n)
        nodes = region(hw, min(size, hw.num_qubits))
        sub = subgraph(hw, nodes)
    say(f"Circuit {work.name}: {work.n} logical qubits, {len(work.gates)} CNOTs")
    say(f"Hardware: FakeBrisbane, using {len(nodes)} of {hw.num_qubits} physical qubits")

    out = {"circuit": work.name, "n": work.n, "cnots": len(work.gates),
           "physical_qubits": nodes}

    # ---- easy
    easy = solve_easy(work, sub)
    phys = lambda L: [nodes[p] for p in L]  # noqa: E731  (block index -> chip qubit)
    say("\n[EASY] spectral placement")
    say(f"  objective tr(X'LX) = {easy.objective:.4f}  (Ky Fan bound "
        f"{easy.certificate['kyfan_lower_bound']:.4f}, certified optimal: "
        f"{easy.certificate['optimal']})")
    say(f"  layout (logical -> physical qubit): {phys(easy.layout)}")
    say(f"  SWAP proxy if each gate is routed on its own: {easy.static_cost}")
    out["easy"] = {"objective": easy.objective, "certified": easy.certificate["optimal"],
                   "layout": phys(easy.layout), "static_cost": easy.static_cost}

    # ---- SABRE
    sab = sabre(work, sub, seeds=range(seeds))
    sab_easy = sabre(work, sub, initial_layout=easy.layout, seeds=range(seeds))
    say("\n[SABRE] Qiskit heuristic")
    say(f"  SABRE layout + routing: {sab.swaps} SWAPs")
    say(f"  easy layout + SABRE routing: {sab_easy.swaps} SWAPs")
    out["sabre"] = {"swaps": sab.swaps, "easy_layout_swaps": sab_easy.swaps}

    # ---- hard
    warm = [greedy_warm_start(work.gates, easy.layout, sub),
            greedy_warm_start(work.gates, sab.layout, sub), sab]
    size = model_size(work.gates, work.n, sub, sab)
    say("\n[HARD] SWAP-minimization MILP")
    if size["vars"] > 400_000:
        say(f"  model size: {size['vars']:,} binary variables, {size['constraints']:,} constraints")
        say("  too large to build on this machine; best known solution = SABRE")
        out["hard"] = {"status": "too_large", "swaps": sab.swaps, **size}
        return out, None
    ex = exact_route(work.gates, work.n, sub, ub=min(w.swaps for w in warm),
                     time_limit=search_limit)
    if ex.improved:
        warm.append(to_warm_start(ex, work.gates))
    t = min(time_limit, 60) if ex.status == "optimal" else time_limit
    hard = solve_hard(work.gates, work.n, sub, warm, time_limit=t)
    lb = max(math.ceil(hard.dual_bound - 1e-6), ex.lower_bound)
    status = "PROVEN OPTIMAL" if hard.objective <= lb else f"best found (optimum >= {lb})"
    size = {"vars": hard.num_vars, "constraints": hard.num_constraints}
    say(f"  model size: {size['vars']:,} binary variables, {size['constraints']:,} constraints")
    say(f"  exact search: {ex.status}, lower bound {ex.lower_bound}"
        + (f", {ex.states:,} states" if ex.states else ""))
    say(f"  result: {int(hard.objective)} SWAPs, {status}")
    say(f"  initial layout (logical -> physical qubit): {phys(hard.layout)}")
    n_sw = 0
    for k, gap in enumerate(hard.swaps):
        for rnd in gap:
            for p, q in rnd:
                n_sw += 1
                say(f"    SWAP {n_sw}: physical qubits {nodes[p]} <-> {nodes[q]}  (before slot {k})")

    # ---- verification
    rp = replay(work.gates, hard.slot_of, hard.layout, hard.swaps, sub)
    qc = routed_circuit(work, hard.slot_of, hard.layout, hard.swaps, sub)
    cm = check_map(qc, sub)
    eq = equivalent(work, hard.layout, qc, sub)
    say("\n[VERIFY]")
    say(f"  replay (every gate adjacent, SWAPs valid): {rp['ok']}")
    say(f"  Qiskit CheckMap (circuit fits the chip):   {cm}")
    say(f"  same unitary as the original circuit:     "
        f"{eq if eq is not None else 'skipped (block > 11 qubits)'}")
    say(f"\nPrice of realism: easy layout needs {sab_easy.swaps} SWAPs (SABRE) vs "
        f"{int(hard.objective)} with the hard model.")
    out["hard"] = {"swaps": hard.objective, "lower_bound": lb,
                   "status": "optimal" if hard.objective <= lb else "time_limit",
                   "layout": phys(hard.layout),
                   "swaps_sequence": [[nodes[p], nodes[q]] for gap in hard.swaps
                                      for rnd in gap for p, q in rnd], **size}
    out["verified"] = {"replay": rp["ok"], "checkmap": cm, "unitary": eq}
    return out, (sub, nodes, easy, hard)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("qasm", help="OpenQASM 2 circuit file")
    ap.add_argument("--block", type=int, help="size of the chip block (default max(n+4, 2n))")
    ap.add_argument("--full-chip", action="store_true", help="use all 127 qubits")
    ap.add_argument("--time-limit", type=float, default=120.0, help="MILP seconds")
    ap.add_argument("--search-limit", type=float, default=300.0, help="exact search seconds")
    ap.add_argument("--json", help="save results to this JSON file")
    ap.add_argument("--plot", help="save an easy-vs-hard layout picture (PNG)")
    a = ap.parse_args()
    res, detail = route(a.qasm, a.block, a.full_chip, a.time_limit, a.search_limit)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))
        print(f"\nSaved {a.json}")
    if a.plot and detail:
        from plots import plot_layouts
        sub, nodes, easy, hard = detail
        plot_layouts(sub, nodes, load_circuit(a.qasm), easy.layout, hard, a.plot)
        print(f"Saved {a.plot}")
