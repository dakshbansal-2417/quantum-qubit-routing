"""Run easy + hard models and baselines on the QASMBench small circuits.

For each circuit (on a region of FakeBrisbane of size max(n+4, 2n)):
  easy_ideal      SWAPs the easy model believes it needs (0: no wiring limits)
  easy_static     easy layout snapped to chip, each gate routed on its own
  easy_sabre      easy layout + SABRE routing
  easy_milp       easy layout + optimal routing (MILP with fixed initial layout)
  hard            full MILP: optimal placement + routing (with proven lower bound)
  search          exact state-space search of the same problem: gives the true optimum
                  (any slot structure) as MILP warm start, and a proven lower bound
  sabre           Qiskit SABRE layout + routing (best of 20 seeds)
  hard_R+1        MILP with one more SWAP round per gap (checks R is large enough)

Usage:  python experiments/run_all.py [--time-limit 120] [--max-gates 45]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from easy_qp import solve_easy  # noqa: E402
from exact_search import exact_route, to_warm_start  # noqa: E402
from hard_milp import dag_merge, greedy_warm_start, solve_hard  # noqa: E402
from load_data import list_small_circuits, load_circuit, load_hardware, region, subgraph  # noqa: E402
from verify import check_map, equivalent, replay, routed_circuit, sabre  # noqa: E402

R = 2


def check(w, res, sub):
    rp = replay(w.gates, res.slot_of, res.layout, res.swaps, sub)
    qc = routed_circuit(w, res.slot_of, res.layout, res.swaps, sub)
    return {"replay": rp["ok"], "swaps_replayed": rp["swaps"], "checkmap": check_map(qc, sub),
            "equivalent": equivalent(w, res.layout, qc, sub)}


def run(time_limit: float, max_gates: int, min_qubits: int, search_limit: float):
    hw = load_hardware()
    rows, sols = [], {}
    for path in list_small_circuits():
        try:
            w = load_circuit(path)
        except Exception:
            continue
        K = len(dag_merge(w.gates)[0])
        if w.n < min_qubits or K == 0 or K > max_gates:
            continue
        nodes = region(hw, max(w.n + 4, 2 * w.n))
        sub = subgraph(hw, nodes)
        print(f"{w.name}: n={w.n} gates={len(w.gates)} (distinct {K}) region={len(nodes)}",
              flush=True)

        easy = solve_easy(w, sub)
        sab = sabre(w, sub)                                   # SABRE layout + routing
        sab_easy = sabre(w, sub, initial_layout=easy.layout)  # easy layout + SABRE routing
        warm_free = [greedy_warm_start(w.gates, easy.layout, sub),
                     greedy_warm_start(w.gates, sab.layout, sub), sab]
        warm_fixed = [greedy_warm_start(w.gates, easy.layout, sub), sab_easy]

        # Exact search: true optimum over all slot structures (MILP constraints untouched).
        # Its solution, when better than the heuristics, becomes the MILP's warm start.
        ex = exact_route(w.gates, w.n, sub, ub=min(x.swaps for x in warm_free),
                         time_limit=search_limit)
        ex_fixed = exact_route(w.gates, w.n, sub, ub=min(x.swaps for x in warm_fixed),
                               roots=[easy.layout], time_limit=search_limit)
        for res, warm in ((ex, warm_free), (ex_fixed, warm_fixed)):
            if res.improved:
                ws = to_warm_start(res, w.gates)
                rp = replay(w.gates, ws.slot_of, ws.layout, ws.gaps, sub)
                assert rp["ok"] and ws.swaps == res.value, rp["errors"]
                warm.append(ws)

        # once the search has certified the optimum the MILP only has to reproduce it
        t_free = min(time_limit, 60) if ex.status == "optimal" else time_limit
        t_fix = min(time_limit, 60) if ex_fixed.status == "optimal" else time_limit
        hard = solve_hard(w.gates, w.n, sub, warm_free, R, time_limit=t_free)
        fixed = solve_hard(w.gates, w.n, sub, warm_fixed, R, fix_layout=True, time_limit=t_fix)
        small = len(hard.swaps) <= 15
        hard_r1 = solve_hard(w.gates, w.n, sub, warm_free, R + 1,
                             time_limit=t_free) if small else None

        def certify(milp, search):
            """Combine the MILP's bound with the search's. Returns (bound, status, proof)."""
            milp_lb = math.ceil(milp.dual_bound - 1e-6)
            lb = max(milp_lb, search.lower_bound)
            if milp.objective <= lb:
                proof = "milp" if milp.objective <= milp_lb else "exact_search"
                return lb, "optimal", proof
            return lb, "time_limit", "none"

        hard_lb, hard_status, hard_proof = certify(hard, ex)
        fix_lb, fix_status, fix_proof = certify(fixed, ex_fixed)

        checks = {"hard": check(w, hard, sub), "easy_milp": check(w, fixed, sub)}
        row = {
            "circuit": w.name, "n": w.n, "gates": len(w.gates), "distinct_gates": K,
            "region": len(nodes),
            "easy_objective": round(easy.objective, 4),
            "easy_kyfan_bound": round(easy.certificate["kyfan_lower_bound"], 4),
            "easy_certified": easy.certificate["optimal"],
            "easy_ideal": 0,
            "easy_static": easy.static_cost,
            "easy_sabre": sab_easy.swaps,
            "easy_milp": fixed.objective, "easy_milp_bound": fix_lb,
            "easy_milp_status": fix_status, "easy_milp_proof": fix_proof,
            "hard": hard.objective, "hard_bound": hard_lb,
            "hard_status": hard_status, "hard_proof": hard_proof,
            "hard_milp_bound": round(hard.dual_bound, 2), "hard_milp_status": hard.status,
            "search_status": ex.status, "search_bound": ex.lower_bound,
            "search_states": ex.states, "search_seconds": round(ex.seconds, 2),
            "hard_warm_start": hard.warm_start, "hard_warm_source": hard.warm_source,
            "hard_seconds": round(hard.seconds, 2), "hard_slots": len(hard.swaps),
            "hard_vars": hard.num_vars, "hard_constraints": hard.num_constraints,
            "hard_R+1": hard_r1.objective if hard_r1 else None,
            "sabre": sab.swaps,
            "verified": all(c["replay"] and c["checkmap"] and c["equivalent"] is not False
                            for c in checks.values()),
            "unitary_checked": all(c["equivalent"] is True for c in checks.values()),
        }
        rows.append(row)
        sols[w.name] = {
            "region_nodes": nodes,
            "easy": {"X": easy.X.tolist(), "layout": easy.layout,
                     "certificate": easy.certificate},
            "hard": {"layout": hard.layout, "slot_of": hard.slot_of, "swaps": hard.swaps,
                     "status": hard.status},
            "easy_milp": {"layout": fixed.layout, "slot_of": fixed.slot_of,
                          "swaps": fixed.swaps},
            "sabre": {"layout": sab.layout, "swaps": sab.swaps},
            "exact_search": {"status": ex.status, "lower_bound": ex.lower_bound,
                             "improved_on_heuristics": ex.improved, "states": ex.states},
            "checks": checks,
        }
        print("   ", {k: row[k] for k in ("easy_static", "easy_sabre", "easy_milp", "hard",
                                         "hard_bound", "hard_status", "hard_proof", "sabre", "hard_R+1",
                                         "verified")}, flush=True)
        save(rows, sols)  # after every circuit, so a stopped run keeps its results

    print(f"\nSaved {len(rows)} circuits to {OUT}")


OUT = ROOT / "experiments" / "results"


def save(rows, sols):
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    (OUT / "solutions.json").write_text(json.dumps(sols, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--max-gates", type=int, default=45)
    ap.add_argument("--search-limit", type=float, default=600.0)
    ap.add_argument("--min-qubits", type=int, default=3)
    a = ap.parse_args()
    run(a.time_limit, a.max_gates, a.min_qubits, a.search_limit)
