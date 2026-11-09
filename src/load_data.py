"""Load the hardware graph (IBM FakeBrisbane) and circuit workloads (QASMBench).

A routing instance is:
  * a hardware graph H = (P, E): physical qubits and coupling edges
  * an ordered list of two-qubit gates G = [(a_1, b_1), ..., (a_K, b_K)] on n logical qubits
  * interaction weights w[i, j] = number of gates between i and j (order ignored)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

ROOT = Path(__file__).resolve().parents[1]
QASMBENCH_SMALL = ROOT / "data" / "QASMBench" / "small"


# --------------------------------------------------------------------------- hardware
@dataclass
class Hardware:
    graph: nx.Graph                 # undirected coupling graph
    dist: np.ndarray                # all-pairs hop distance D[p, q]
    num_qubits: int
    pos: dict = field(default_factory=dict)  # 2D drawing coordinates per physical qubit

    @property
    def edges(self) -> list[tuple[int, int]]:
        return sorted(tuple(sorted(e)) for e in self.graph.edges())


def load_hardware() -> Hardware:
    """IBM FakeBrisbane: 127 qubits, 144 coupling edges (heavy-hex lattice)."""
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    backend = FakeBrisbane()
    g = nx.Graph()
    g.add_nodes_from(range(backend.num_qubits))
    g.add_edges_from(tuple(e) for e in backend.coupling_map.get_edges())
    n = backend.num_qubits
    dist = np.zeros((n, n), dtype=int)
    for p, lengths in nx.all_pairs_shortest_path_length(g):
        for q, d in lengths.items():
            dist[p, q] = d
    pos = nx.kamada_kawai_layout(g)
    return Hardware(graph=g, dist=dist, num_qubits=n, pos=pos)


def region(hw: Hardware, size: int, center: int | None = None) -> list[int]:
    """Connected block of `size` physical qubits grown by BFS from a central qubit.

    The exact MILP cannot handle all 127 qubits, so both models are solved on the
    same region; this keeps the easy-vs-hard comparison fair.
    """
    if center is None:
        center = int(np.argmin(hw.dist.sum(axis=1)))  # most central qubit
    order = [center] + [v for _, v in nx.bfs_edges(hw.graph, center)]
    return sorted(order[:size])


def subgraph(hw: Hardware, nodes: list[int]) -> Hardware:
    """Hardware restricted to `nodes`, relabelled 0..m-1 (in the order of `nodes`)."""
    idx = {p: k for k, p in enumerate(nodes)}
    g = nx.relabel_nodes(hw.graph.subgraph(nodes).copy(), idx)
    m = len(nodes)
    dist = np.zeros((m, m), dtype=int)
    for p, lengths in nx.all_pairs_shortest_path_length(g):
        for q, d in lengths.items():
            dist[p, q] = d
    pos = {idx[p]: hw.pos[p] for p in nodes} if hw.pos else {}
    return Hardware(graph=g, dist=dist, num_qubits=m, pos=pos)


# --------------------------------------------------------------------------- circuits
@dataclass
class Workload:
    name: str
    n: int                              # logical qubits that take part in 2Q gates
    gates: list[tuple[int, int]]        # ordered 2Q interactions (CNOTs)
    circuit: QuantumCircuit             # 2Q skeleton circuit (cx only), n qubits

    @property
    def weights(self) -> np.ndarray:
        w = np.zeros((self.n, self.n))
        for a, b in self.gates:
            w[a, b] += 1
            w[b, a] += 1
        return w


def load_circuit(path: str | Path) -> Workload:
    """Parse an OpenQASM 2 file and extract its ordered two-qubit interactions.

    Multi-qubit gates (Toffoli, Fredkin, ...) are decomposed into CNOTs first.
    Single-qubit gates are dropped: they never need routing.
    """
    path = Path(path)
    qc = qasm2.load(
        str(path),
        include_path=[str(path.parent), str(QASMBENCH_SMALL)],
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    qc = qc.remove_final_measurements(inplace=False) or qc
    qc = transpile(qc, basis_gates=["cx", "u"], optimization_level=0)

    raw = []
    for inst in qc.data:
        if inst.operation.name == "cx":
            a, b = (qc.find_bit(q).index for q in inst.qubits)
            raw.append((a, b))

    # Relabel so only qubits that take part in 2Q gates count as logical qubits.
    used = sorted({q for g in raw for q in g})
    relabel = {q: k for k, q in enumerate(used)}
    gates = [(relabel[a], relabel[b]) for a, b in raw]

    skeleton = QuantumCircuit(len(used), name=path.stem)
    for a, b in gates:
        skeleton.cx(a, b)
    return Workload(name=path.stem, n=len(used), gates=gates, circuit=skeleton)


def list_small_circuits() -> list[Path]:
    return sorted(p for p in QASMBENCH_SMALL.glob("*/*.qasm")
                  if not p.stem.endswith("_transpiled"))


if __name__ == "__main__":
    hw = load_hardware()
    print(f"FakeBrisbane: {hw.num_qubits} qubits, {len(hw.edges)} edges, "
          f"diameter {hw.dist.max()}")
    print(f"{'circuit':28s} {'n':>3s} {'#2Q':>5s}")
    for p in list_small_circuits():
        try:
            w = load_circuit(p)
            print(f"{w.name:28s} {w.n:3d} {len(w.gates):5d}")
        except Exception as e:  # some benchmark files use non-standard syntax
            print(f"{p.stem:28s}  skipped ({type(e).__name__}: {str(e)[:60]})")
