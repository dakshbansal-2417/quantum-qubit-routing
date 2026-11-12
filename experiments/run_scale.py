"""Scale experiment: the full 127-qubit FakeBrisbane chip, small + medium circuits.

The exact methods (MILP, exact search) were run on a chip block in run_all.py.
Here every circuit uses the WHOLE chip, to answer two questions:

1. Does the price of realism hold at real size?
     easy_static   easy layout, each gate routed on its own (SWAP proxy)
     easy_sabre    easy layout + SABRE routing
     sabre         SABRE layout + SABRE routing (best known solution)
2. Where do the exact methods stop?
     milp_vars / milp_constraints   size of the hard MILP on the full chip
     hard_full     for small circuits (n <= --milp-max-n): the MILP actually solved on
                   the full chip, with its proven lower bound, to compare with the
                   chip-block optimum from run_all.py

Circuits with more than --max-cx CNOTs are listed as skipped (e.g. vqe_n24: 1.5M).

Usage:  python experiments/run_scale.py [--max-cx 5000] [--milp-max-n 5] [--time-limit 120]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from easy_qp import solve_easy, static_cost  # noqa: E402
from hard_milp import greedy_warm_start, model_size, solve_hard  # noqa: E402
from load_data import QASMBENCH_SMALL, load_circuit, load_hardware, region, subgraph  # noqa: E402
from verify import replay, sabre  # noqa: E402

OUT = ROOT / "experiments" / "results"
MEDIUM = QASMBENCH_SMALL.parent / "medium"


def circuits():
    for suite, folder in (("small", QASMBENCH_SMALL), ("medium", MEDIUM)):
        for p in sorted(folder.glob("*/*.qasm")):
            if not p.stem.endswith("_transpiled"):
                yield suite, p


def run(max_cx: int, milp_max_n: int, time_limit: float, seeds: int):
    hw = load_hardware()
    block = pd.read_csv(OUT / "results.csv").set_index("circuit") \
        if (OUT / "results.csv").exists() else None
    rows = []
    for suite, path in circuits():
        try:
            w = load_circuit(path)
        except Exception as e:
            rows.append({"suite": suite, "circuit": path.stem, "skipped": f"parse error ({type(e).__name__})"})
            continue
        if w.n < 3:
            continue
        if len(w.gates) > max_cx:
            rows.append({"suite": suite, "circuit": w.name, "n": w.n, "gates": len(w.gates),
                         "skipped": f"> {max_cx} CNOTs"})
            print(f"{w.name}: skipped ({len(w.gates)} CNOTs)", flush=True)
            continue
        # Easy model solved and snapped on the same centred block as in run_all.py, then
        # mapped to chip qubit numbers (snapping straight onto all 127 qubits rounds
        # poorly and would penalise the easy model for a rounding artefact).
        t0 = time.time()
        nodes = region(hw, max(w.n + 4, 2 * w.n))
        easy = solve_easy(w, subgraph(hw, nodes))
        easy.layout = [nodes[p] for p in easy.layout]
        easy.static_cost = static_cost(easy.layout, w.gates, hw.dist)
        t_easy = time.time() - t0
        t0 = time.time()
        sab = sabre(w, hw, seeds=range(seeds))
        t_sabre = time.time() - t0
        sab_easy = sabre(w, hw, initial_layout=easy.layout, seeds=range(seeds))
        size = model_size(w.gates, w.n, hw, sab)
        row = {
            "suite": suite, "circuit": w.name, "n": w.n, "gates": len(w.gates),
            "easy_certified": easy.certificate["optimal"], "easy_seconds": round(t_easy, 2),
            "easy_static": easy.static_cost, "easy_sabre": sab_easy.swaps,
            "sabre": sab.swaps, "sabre_seconds": round(t_sabre, 2),
            "easy_over_sabre": round(sab_easy.swaps / sab.swaps, 2) if sab.swaps else None,
            "milp_vars": size["vars"], "milp_constraints": size["constraints"],
            "search_placements_log10": round(math.log10(math.perm(hw.num_qubits, w.n)), 1),
            "verified": replay(w.gates, sab.slot_of, sab.layout, sab.gaps, hw)["ok"]
            and replay(w.gates, sab_easy.slot_of, sab_easy.layout, sab_easy.gaps, hw)["ok"],
        }
        if suite == "small" and w.n <= milp_max_n:
            hard = solve_hard(w.gates, w.n, hw,
                              [greedy_warm_start(w.gates, easy.layout, hw), sab],
                              time_limit=time_limit)
            lb = math.ceil(hard.dual_bound - 1e-6)
            row.update({"hard_full": hard.objective, "hard_full_bound": lb,
                        "hard_full_status": "optimal" if hard.objective <= lb else "time_limit",
                        "hard_full_seconds": round(hard.seconds, 1)})
            if block is not None and w.name in block.index:
                row["hard_block"] = block.loc[w.name, "hard"]
                row["block_status"] = block.loc[w.name, "hard_status"]
        rows.append(row)
        print(f"{w.name}: n={w.n} cx={len(w.gates)} easy_sabre={row['easy_sabre']} "
              f"sabre={row['sabre']} milp_vars={row['milp_vars']:,}"
              + (f" hard_full={row['hard_full']} (lb {row['hard_full_bound']})"
                 if "hard_full" in row else ""), flush=True)
        OUT.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / "scale_results.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "scale_results.csv", index=False)
    print(f"\nSaved {len(rows)} rows to {OUT / 'scale_results.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cx", type=int, default=5000)
    ap.add_argument("--milp-max-n", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    run(a.max_cx, a.milp_max_n, a.time_limit, a.seeds)
