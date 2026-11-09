"""Independent verification of routing solutions, plus the Qiskit SABRE baseline.

1. replay():      re-simulate the placement step by step; check every SWAP is on a
                  real edge, SWAPs in a round are disjoint, every gate is adjacent.
2. check_map():   build the routed circuit in Qiskit and run its CheckMap pass.
3. equivalent():  for small regions, check the routed circuit is the same unitary
                  as the original (undoing the SWAP permutation at the end).
"""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import CheckMap

from hard_milp import merge_repeats
from load_data import Hardware, Workload


def coupling_map(hw: Hardware) -> CouplingMap:
    return CouplingMap([[u, v] for u, v in hw.edges] + [[v, u] for u, v in hw.edges])


def replay(gates, layout, swaps, R, hw: Hardware) -> dict:
    """Re-simulate a MILP solution and check every constraint by hand."""
    G = merge_repeats(gates)
    edges = set(hw.edges)
    where = list(layout)                   # logical -> physical
    errors = []
    if len(set(where)) != len(where):
        errors.append("initial layout not injective")
    for k, (a, b) in enumerate(G):
        if k > 0:
            for t in range((k - 1) * R, k * R):
                used = set()
                for p, q in swaps[t]:
                    if (min(p, q), max(p, q)) not in edges:
                        errors.append(f"round {t}: ({p},{q}) is not an edge")
                    if p in used or q in used:
                        errors.append(f"round {t}: overlapping SWAPs")
                    used |= {p, q}
                    where = [q if x == p else p if x == q else x for x in where]
        if hw.dist[where[a], where[b]] != 1:
            errors.append(f"gate {k} ({a},{b}) not adjacent")
    return {"ok": not errors, "errors": errors[:5],
            "swaps": sum(len(s) for s in swaps)}


def routed_circuit(work: Workload, layout, swaps, R, hw: Hardware) -> QuantumCircuit:
    """Physical circuit: original CNOTs (all of them) + inserted SWAPs."""
    G = merge_repeats(work.gates)
    qc = QuantumCircuit(hw.num_qubits)
    where = list(layout)
    gi = 0
    for k, g in enumerate(G):
        if k > 0:
            for t in range((k - 1) * R, k * R):
                for p, q in swaps[t]:
                    qc.swap(p, q)
                    where = [q if x == p else p if x == q else x for x in where]
        # emit this gate and its merged repeats in original orientation
        while gi < len(work.gates) and set(work.gates[gi]) == set(g):
            a, b = work.gates[gi]
            qc.cx(where[a], where[b])
            gi += 1
    return qc


def check_map(qc: QuantumCircuit, hw: Hardware) -> bool:
    p = CheckMap(coupling_map(hw))
    p.run(circuit_to_dag(qc))
    return bool(p.property_set["is_swap_mapped"])


def equivalent(work: Workload, layout, qc_routed: QuantumCircuit, hw: Hardware,
               max_qubits: int = 11) -> bool | None:
    """Unitary equivalence check (only for small regions; None if too large)."""
    if hw.num_qubits > max_qubits:
        return None
    orig = QuantumCircuit(hw.num_qubits)
    for a, b in work.gates:
        orig.cx(layout[a], layout[b])
    undo = qc_routed.copy()
    for inst in reversed([i for i in qc_routed.data if i.operation.name == "swap"]):
        undo.swap(*[qc_routed.find_bit(q).index for q in inst.qubits])
    return Operator(undo).equiv(Operator(orig))


# --------------------------------------------------------------------------- baseline
def sabre(work: Workload, hw: Hardware, initial_layout=None, seeds=range(20)) -> int:
    """Best SWAP count from Qiskit's SABRE router over several seeds."""
    cm = coupling_map(hw)
    best = None
    for seed in seeds:
        kw = dict(initial_layout=initial_layout) if initial_layout else dict(layout_method="sabre")
        out = transpile(work.circuit, coupling_map=cm, basis_gates=["cx", "swap"],
                        routing_method="sabre", optimization_level=0,
                        seed_transpiler=seed, **kw)
        n_swaps = out.count_ops().get("swap", 0)
        best = n_swaps if best is None else min(best, n_swaps)
    return int(best)
