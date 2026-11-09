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

from hard_milp import WarmStart, apply_swap
from load_data import Hardware, Workload


def coupling_map(hw: Hardware) -> CouplingMap:
    return CouplingMap([[u, v] for u, v in hw.edges] + [[v, u] for u, v in hw.edges])


def respects_dependencies(gates, slot_of) -> bool:
    """Every gate's slot is >= the slot of each earlier gate sharing a qubit."""
    last = {}
    for (a, b), s in zip(gates, slot_of):
        if any(last.get(q, -1) > s for q in (a, b)):
            return False
        last[a] = last[b] = s
    return True


def replay(gates, slot_of, layout, swaps, hw: Hardware) -> dict:
    """Re-simulate a MILP solution and check every constraint by hand.

    swaps[k] = SWAP rounds performed before slot k; slot_of[j] = slot of gate j.
    """
    edges = set(hw.edges)
    where = list(layout)                   # logical -> physical
    errors = []
    if len(set(where)) != len(where):
        errors.append("initial layout not injective")
    if not respects_dependencies(gates, slot_of):
        errors.append("slot order violates gate dependencies")
    for k, gap in enumerate(swaps):
        for r, rnd in enumerate(gap):
            used = set()
            for p, q in rnd:
                if (min(p, q), max(p, q)) not in edges:
                    errors.append(f"gap {k} round {r}: ({p},{q}) is not an edge")
                if p in used or q in used:
                    errors.append(f"gap {k} round {r}: overlapping SWAPs")
                used |= {p, q}
                where = apply_swap(where, p, q)
        for j, (a, b) in enumerate(gates):
            if slot_of[j] == k and hw.dist[where[a], where[b]] != 1:
                errors.append(f"gate {j} ({a},{b}) in slot {k} not adjacent")
    return {"ok": not errors, "errors": errors[:5],
            "swaps": sum(len(rnd) for gap in swaps for rnd in gap)}


def routed_circuit(work: Workload, slot_of, layout, swaps, hw: Hardware) -> QuantumCircuit:
    """Physical circuit: original CNOTs (all of them, slot by slot) + inserted SWAPs."""
    qc = QuantumCircuit(hw.num_qubits)
    where = list(layout)
    for k, gap in enumerate(swaps):
        for rnd in gap:
            for p, q in rnd:
                qc.swap(p, q)
                where = apply_swap(where, p, q)
        for j, (a, b) in enumerate(work.gates):
            if slot_of[j] == k:
                qc.cx(where[a], where[b])
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
def sabre(work: Workload, hw: Hardware, initial_layout=None, seeds=range(20)) -> WarmStart:
    """Qiskit's SABRE router (best of several seeds), returned in MILP slot form.

    The returned WarmStart is checked with replay(), so a conversion error can
    never slip into the MILP.
    """
    cm = coupling_map(hw)
    best = None
    for seed in seeds:
        kw = dict(initial_layout=initial_layout) if initial_layout else dict(layout_method="sabre")
        out = transpile(work.circuit, coupling_map=cm, basis_gates=["cx", "swap"],
                        routing_method="sabre", optimization_level=0,
                        seed_transpiler=seed, **kw)
        n_swaps = out.count_ops().get("swap", 0)
        if best is None or n_swaps < best[0]:
            best = (n_swaps, out)
    ws = sabre_to_slots(work, best[1])
    assert ws.swaps == best[0]
    rp = replay(work.gates, ws.slot_of, ws.layout, ws.gaps, hw)
    assert rp["ok"], rp["errors"]
    return ws


def sabre_to_slots(work: Workload, out: QuantumCircuit) -> WarmStart:
    """Convert a SABRE-routed circuit to (layout, slot of each gate, SWAP rounds).

    Each run of CNOTs between SWAPs becomes one slot; the SWAPs before it become its
    gap, packed into rounds as early as their order allows.
    """
    layout = [int(p) for p in out.layout.initial_index_layout(filter_ancillas=True)[: work.n]]
    owner = {p: i for i, p in enumerate(layout)}           # physical -> logical
    queue = {q: [] for q in range(work.n)}                  # pending gates per qubit
    for j, (a, b) in enumerate(work.gates):
        queue[a].append(j); queue[b].append(j)
    slot_of = [None] * len(work.gates)
    gaps, cur, slot, open_slot = [], [], 0, False

    def pack(swaps):
        rounds, when = [], {}
        for p, q in swaps:
            r = 1 + max(when.get(p, -1), when.get(q, -1))
            while len(rounds) <= r:
                rounds.append([])
            rounds[r].append((min(p, q), max(p, q)))
            when[p] = when[q] = r
        return rounds

    for inst in out.data:
        qs = [out.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "swap":
            if open_slot:
                slot += 1; open_slot = False
            cur.append(tuple(qs))
            pa, pb = owner.get(qs[0]), owner.get(qs[1])
            owner.pop(qs[0], None); owner.pop(qs[1], None)
            if pa is not None: owner[qs[1]] = pa
            if pb is not None: owner[qs[0]] = pb
        elif inst.operation.name == "cx":
            if not open_slot:
                gaps.append(pack(cur)); cur = []; open_slot = True
            a, b = owner[qs[0]], owner[qs[1]]
            j = queue[a][0]
            assert queue[b][0] == j and work.gates[j] == (a, b)
            queue[a].pop(0); queue[b].pop(0)
            slot_of[j] = slot
    return WarmStart("sabre", layout, slot_of, gaps)
