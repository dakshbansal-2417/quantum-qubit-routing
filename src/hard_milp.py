"""HARD model: time-expanded integer routing (MILP).

Time slots. Two-qubit gates are grouped into slots 0..L-1 that respect circuit
dependencies: a gate's slot is later than the slot of every earlier gate sharing
a qubit with it (gates on disjoint qubits may share a slot, as on real hardware).
All gates in a slot run at the same placement. Before slot k there is a gap of R_k
SWAP rounds; in each round any set of disjoint SWAPs may happen. Placement states
are t = 0..S-1, and slot k runs at state tau_k = R_0 + ... + R_k.

Variables
    x[i,p,t] in {0,1}   logical qubit i sits on physical qubit p at state t
    s[e,t]   in {0,1}   SWAP on hardware edge e = (p,q) between state t and t+1

Objective
    min  sum_{e,t} s[e,t]                                   (total SWAPs)

Constraints
    (1) sum_p x[i,p,t] = 1                     each logical qubit in one place
    (2) sum_i x[i,p,t] <= 1                    each physical qubit holds <= 1
    (3) x[a,p,tau_k] <= sum_{q in N(p)} x[b,q,tau_k]   every gate (a,b) in slot k
        is on a hardware edge (also with a and b exchanged)
    (4) sum_{e ni p} s[e,t] <= 1               SWAPs in one round are disjoint
    (5) x[i,q,t+1] >= x[i,p,t] + s[pq,t] - 1   a SWAP moves the qubit (both ways)
    (6) x[i,p,t+1] >= x[i,p,t] - sum_{e ni p} s[e,t]   no SWAP at p: it stays
    (7) s[pq,t] <= sum_i x[i,p,t] + sum_i x[i,q,t]     never SWAP two empty spots
(3, both directions) and (7) do not change the optimum; they tighten the LP
relaxation so the solver prunes faster.

Slots and gap sizes come from a warm-start routing: our greedy router (ASAP
layers; a gate that cannot be routed without disturbing the slot's other gates
moves to a new slot) or Qiskit's SABRE, whichever uses fewer SWAPs, with
R_k = max(R, rounds the warm start used). The warm start therefore fits the model
and is given to HiGHS: the MILP always has a feasible answer and can only improve
on it. "Optimal" means optimal for that slot structure and R; experiments re-solve
with R+1 to check R is large enough.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import highspy
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from load_data import Hardware

Rounds = list[list[tuple[int, int]]]   # round -> SWAP edges


@dataclass
class HardResult:
    status: str                 # optimal | time_limit | ...
    objective: float | None     # SWAPs in best solution found
    dual_bound: float | None    # proven lower bound on the optimum
    gap: float | None           # relative gap between the two
    layout: list[int]           # initial placement (logical -> region index)
    swaps: list[Rounds]         # swaps[k] = SWAP rounds before slot k
    slot_of: list[int]          # slot of every original gate
    warm_start: int             # SWAPs in the starting solution
    warm_source: str            # which heuristic it came from
    rounds: list[int]           # R_k per gap
    num_vars: int
    num_constraints: int
    seconds: float


def apply_swap(where: list[int], p: int, q: int) -> list[int]:
    return [q if x == p else p if x == q else x for x in where]


def dag_merge(gates):
    """Deduplicate gates and compute ASAP layers on the dependency DAG.

    A gate that repeats the immediately preceding gate on both its qubits (same
    pair) can run at the same placement, so it is folded into that gate.
    Returns uniq (list of pairs), rep (orig gate -> uniq index), layer (per uniq).
    """
    last = {}                      # qubit -> uniq index of last gate on it
    uniq, rep, layer = [], [], []
    for a, b in gates:
        pa, pb = last.get(a), last.get(b)
        if pa is not None and pa == pb and set(uniq[pa]) == {a, b}:
            rep.append(pa)
            continue
        lv = 1 + max(layer[pa] if pa is not None else -1, layer[pb] if pb is not None else -1)
        uniq.append((a, b)); layer.append(lv); rep.append(len(uniq) - 1)
        last[a] = last[b] = len(uniq) - 1
    return uniq, rep, layer


def greedy_route(gates, layout, hw: Hardware, lookahead: int = 8):
    """Layer-by-layer lookahead greedy router.

    Returns (slots: list of uniq-gate lists, gaps: SWAP rounds before each slot, rep).
    Within a slot, gates are routed one at a time along shortest paths that avoid
    the qubits already fixed for that slot; a gate with no such path goes to the
    next slot. Which end moves (a, b, or both to the middle) is chosen by lookahead.
    """
    uniq, rep, layer = dag_merge(gates)
    order = sorted(range(len(uniq)), key=lambda g: (layer[g], g))
    future_of = {g: [uniq[h] for h in order[pos + 1: pos + 1 + lookahead]]
                 for pos, g in enumerate(order)}
    where = list(layout)
    slots, gaps = [], []
    for lv in sorted(set(layer)):
        pending = [g for g in order if layer[g] == lv]
        while pending:
            locked, gap, placed, rest = set(), [], [], []
            for g in pending:                                # already adjacent
                a, b = uniq[g]
                if hw.dist[where[a], where[b]] == 1:
                    placed.append(g); locked |= {where[a], where[b]}
                else:
                    rest.append(g)
            left = []
            for g in sorted(rest, key=lambda g: hw.dist[where[uniq[g][0]], where[uniq[g][1]]]):
                a, b = uniq[g]
                free = hw.graph.subgraph([p for p in hw.graph if p not in locked
                                          or p in (where[a], where[b])])
                try:
                    path = nx.shortest_path(free, where[a], where[b])
                except nx.NetworkXNoPath:
                    left.append(g)
                    continue
                d, best = len(path) - 1, None
                for split in sorted({0, d - 1, (d - 1) // 2}):
                    w, steps = list(where), []
                    for j in range(split):                   # a forward
                        p, q = path[j], path[j + 1]
                        steps.append((min(p, q), max(p, q))); w = apply_swap(w, p, q)
                    for j in range(d, split + 1, -1):        # b backward
                        p, q = path[j], path[j - 1]
                        steps.append((min(p, q), max(p, q))); w = apply_swap(w, p, q)
                    fut = sum(0.8 ** f * hw.dist[w[x], w[y]]
                              for f, (x, y) in enumerate(future_of[g]))
                    if best is None or fut < best[0]:
                        best = (fut, steps, w)
                _, steps, where = best
                gap += [[e] for e in steps]
                placed.append(g); locked |= {where[a], where[b]}
            slots.append(placed); gaps.append(gap)
            pending = left
    return slots, gaps, rep


@dataclass
class WarmStart:
    """A feasible routing in slot form, used to start the MILP."""
    source: str                 # "greedy" or "sabre"
    layout: list[int]           # initial placement
    slot_of: list[int]          # slot of every original gate
    gaps: list[Rounds]          # SWAP rounds before each slot

    @property
    def swaps(self) -> int:
        return sum(len(rnd) for gap in self.gaps for rnd in gap)


def greedy_warm_start(gates, layout, hw: Hardware) -> WarmStart:
    slots, gaps, rep = greedy_route(gates, layout, hw)
    slot_of_uniq = {g: k for k, sl in enumerate(slots) for g in sl}
    return WarmStart("greedy", list(layout), [slot_of_uniq[rep[j]] for j in range(len(gates))],
                     gaps)


def solve_hard(gates, n: int, hw: Hardware, warm_starts: list[WarmStart],
               rounds_per_gate: int = 2, fix_layout: bool = False,
               time_limit: float = 120.0, verbose: bool = False) -> HardResult:
    """Solve the MILP.

    warm_starts: feasible routings (greedy, SABRE, ...). The one with the fewest SWAPs
                 is given to HiGHS and also fixes the slot structure.
    fix_layout:  if True the initial placement is fixed to the warm start's layout,
                 so only the routing is optimized ("easy layout + optimal routing");
                 all warm starts must then share that layout.
    """
    t0 = time.time()
    m = hw.num_qubits
    E = hw.edges
    eidx = {e: k for k, e in enumerate(E)}
    nbr = {p: list(hw.graph.neighbors(p)) for p in range(m)}
    edges_at = {p: [k for k, (u, v) in enumerate(E) if p in (u, v)] for p in range(m)}

    ws = min(warm_starts, key=lambda w: w.swaps)
    warm_swaps, start_layout, warm, slot_of = ws.swaps, ws.layout, ws.gaps, ws.slot_of
    K = len(warm)
    # distinct gate pairs per slot (repeats at the same placement need one constraint)
    slots = [sorted({tuple(sorted(gates[j])) for j in range(len(gates)) if slot_of[j] == k})
             for k in range(K)]

    # gap 0 (before the first slot) only needs rounds when the layout is fixed
    rounds = [max(rounds_per_gate if (k > 0 or fix_layout) else 0, len(warm[k]))
              for k in range(K)]
    tau = list(np.cumsum(rounds))
    S = int(tau[-1]) + 1 if K else 1
    T = S - 1

    nx_ = n * m * S
    X = lambda i, p, t: (t * n + i) * m + p          # noqa: E731
    Sv = lambda e, t: nx_ + t * len(E) + e           # noqa: E731
    nvar = nx_ + len(E) * T

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(coefs, lb, ub):
        nonlocal r
        for c, v in coefs:
            rows.append(r); cols.append(c); vals.append(v)
        lo.append(lb); hi.append(ub)
        r += 1

    for t in range(S):
        for i in range(n):                                   # (1)
            add([(X(i, p, t), 1) for p in range(m)], 1, 1)
        for p in range(m):                                   # (2)
            add([(X(i, p, t), 1) for i in range(n)], -np.inf, 1)
    for k, sl in enumerate(slots):                           # (3)
        t = int(tau[k])
        for a, b in sl:
            for p in range(m):
                add([(X(a, p, t), 1)] + [(X(b, q, t), -1) for q in nbr[p]], -np.inf, 0)
                add([(X(b, p, t), 1)] + [(X(a, q, t), -1) for q in nbr[p]], -np.inf, 0)
    for t in range(T):
        for p in range(m):                                   # (4)
            if len(edges_at[p]) > 1:
                add([(Sv(e, t), 1) for e in edges_at[p]], -np.inf, 1)
        for e, (p, q) in enumerate(E):
            for i in range(n):                               # (5)
                add([(X(i, q, t + 1), 1), (X(i, p, t), -1), (Sv(e, t), -1)], -1, np.inf)
                add([(X(i, p, t + 1), 1), (X(i, q, t), -1), (Sv(e, t), -1)], -1, np.inf)
            add([(Sv(e, t), 1)] + [(X(i, p, t), -1) for i in range(n)]   # (7)
                + [(X(i, q, t), -1) for i in range(n)], -np.inf, 0)
        for p in range(m):                                   # (6)
            for i in range(n):
                add([(X(i, p, t + 1), 1), (X(i, p, t), -1)]
                    + [(Sv(e, t), 1) for e in edges_at[p]], 0, np.inf)

    A = csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    lb, ub = np.zeros(nvar), np.ones(nvar)
    if fix_layout:
        for i, p in enumerate(start_layout):
            lb[X(i, p, 0)] = 1

    # ---- warm start vector from the greedy schedule (padded to R_k rounds per gap)
    x0 = np.zeros(nvar)
    where, t = list(start_layout), 0
    for i, p in enumerate(where):
        x0[X(i, p, 0)] = 1
    for k in range(K):
        for rnd in warm[k] + [[]] * (rounds[k] - len(warm[k])):
            for p, q in rnd:
                x0[Sv(eidx[(p, q)], t)] = 1
                where = apply_swap(where, p, q)
            t += 1
            for i, p in enumerate(where):
                x0[X(i, p, t)] = 1

    # ---- HiGHS
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.addVars(nvar, lb, ub)
    h.changeColsCost(nvar - nx_, np.arange(nx_, nvar, dtype=np.int32),
                     np.ones(nvar - nx_))
    h.changeColsIntegrality(nvar, np.arange(nvar, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * nvar))
    h.addRows(r, np.array(lo), np.array(hi), A.nnz,
              A.indptr[:-1].astype(np.int32), A.indices.astype(np.int32),
              A.data.astype(float))
    sol = highspy.HighsSolution()
    sol.col_value = list(x0)
    sol.value_valid = True
    h.setSolution(sol)
    h.run()
    secs = time.time() - t0

    ms = h.getModelStatus()
    status = {highspy.HighsModelStatus.kOptimal: "optimal",
              highspy.HighsModelStatus.kTimeLimit: "time_limit",
              highspy.HighsModelStatus.kInfeasible: "infeasible"}.get(ms, str(ms))
    info = h.getInfo()
    if info.primal_solution_status == 0:   # no solution: fall back to the warm start
        xv = np.round(x0).astype(int)
        status = status + "_no_incumbent"
    else:
        xv = np.round(np.array(h.getSolution().col_value)).astype(int)
    layout = [int(np.argmax([xv[X(i, p, 0)] for p in range(m)])) for i in range(n)]
    swaps, t = [], 0
    for k in range(K):
        gap = []
        for _ in range(rounds[k]):
            gap.append([E[e] for e in range(len(E)) if xv[Sv(e, t)]])
            t += 1
        swaps.append(gap)
    return HardResult(status=status, objective=float(xv[nx_:].sum()),
                      dual_bound=float(info.mip_dual_bound), gap=float(info.mip_gap),
                      layout=layout, swaps=swaps, slot_of=slot_of,
                      warm_start=int(warm_swaps), warm_source=ws.source, rounds=rounds,
                      num_vars=nvar, num_constraints=r, seconds=secs)
