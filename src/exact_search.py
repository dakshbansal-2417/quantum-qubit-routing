"""Exact solver for the HARD problem by state-space search (no change to the MILP).

The MILP in hard_milp.py needs a slot structure and a round limit R before it can be
written down, and its LP bound is weak on larger circuits. This module solves the
*same* problem (same decisions, same objective, same rules) by breadth-first search,
which needs neither:

    state   = (where each logical qubit sits, how many gates each qubit has done)
    move    = one SWAP on a hardware edge with at least one occupied end (cost 1)
    free    = any gate whose two qubits are adjacent and whose predecessors are
              done is executed immediately (this never hurts)
    start   = every possible initial placement (cost 0), or one fixed placement
    goal    = all gates executed

Searching level by level (0 SWAPs, 1 SWAP, ...) the first goal found is the true
optimum over ALL slot structures and round counts. States that provably cannot beat
a known solution (cost so far + lower bound on the rest >= known value) are dropped.

Lower bound on the remaining SWAPs (admissible):
    h1 = max over each qubit's next gate of (distance - 1)
    h2 = ceil( sum over ready gates of (distance - 1) / 2 )
         (ready gates are qubit-disjoint and one SWAP moves two qubits one hop)

The optimal SWAP sequence is converted to slot form and given to the MILP as its
starting solution, so the MILP reaches the true optimum with its constraints
unchanged, and the search level reached is a proven lower bound.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import permutations

from hard_milp import WarmStart, pack_rounds
from load_data import Hardware


@dataclass
class ExactResult:
    status: str                 # optimal | limit | skipped
    value: int | None           # best SWAP count known (search result or the given ub)
    lower_bound: int            # proven lower bound on the optimum
    improved: bool              # True if the search found fewer SWAPs than `ub`
    layout: list[int] | None    # initial placement of the solution found (if improved)
    events: list | None         # ("gate", j) / ("swap", p, q) in execution order
    states: int
    seconds: float


def exact_route(gates, n: int, hw: Hardware, ub: int | None = None,
                roots: list[list[int]] | None = None, max_roots: int = 1_500_000,
                max_states: int = 4_000_000, time_limit: float = 600.0) -> ExactResult:
    """Minimum number of SWAPs to run `gates` on `hw`.

    ub:    SWAP count of a known feasible solution; only better ones are searched for.
    roots: allowed initial placements (default: all). One root = fixed layout.
    """
    t0 = time.time()
    m = hw.num_qubits
    dist = hw.dist.tolist()
    nbr = [sorted(hw.graph.neighbors(p)) for p in range(m)]
    queue = [[] for _ in range(n)]              # gate indices per qubit, in order
    for j, (a, b) in enumerate(gates):
        queue[a].append(j); queue[b].append(j)
    qlen = [len(q) for q in queue]
    total = 2 * len(gates)
    UB = math.inf if ub is None else ub

    if roots is None:
        if math.perm(m, n) > max_roots:
            return ExactResult("skipped", ub, 0, False, None, None, 0, 0.0)
        roots = permutations(range(m), n)

    def advance(where, ptr, start):
        """Execute every gate that became ready and adjacent. Returns new ptr tuple."""
        ptr = list(ptr)
        stack = list(start)
        while stack:
            a = stack.pop()
            if ptr[a] == qlen[a]:
                continue
            j = queue[a][ptr[a]]
            x, y = gates[j]
            b = y if a == x else x
            if queue[b][ptr[b]] == j and dist[where[x]][where[y]] == 1:
                ptr[a] += 1; ptr[b] += 1
                stack.append(a); stack.append(b)
        return tuple(ptr)

    def h(where, ptr):
        hmax, hsum = 0, 0
        for a in range(n):
            if ptr[a] == qlen[a]:
                continue
            j = queue[a][ptr[a]]
            x, y = gates[j]
            d = dist[where[x]][where[y]] - 1
            if d > hmax:
                hmax = d
            if a == x and queue[y][ptr[y]] == j:      # ready gate, counted once
                hsum += d
        return max(hmax, (hsum + 1) // 2)

    parent = {}                                  # state -> (previous state, p, q) | None

    def finish(state, g):
        path = []
        while parent[state] is not None:
            prev, p, q = parent[state]
            path.append((p, q)); state = prev
        path.reverse()
        layout = list(state[0])
        # replay to list events in execution order
        where, ptr, events = list(layout), [0] * n, []

        def run_ready():
            changed = True
            while changed:
                changed = False
                for j, (x, y) in enumerate(gates):
                    if (ptr[x] < qlen[x] and ptr[y] < qlen[y] and queue[x][ptr[x]] == j
                            and queue[y][ptr[y]] == j and dist[where[x]][where[y]] == 1):
                        ptr[x] += 1; ptr[y] += 1
                        events.append(("gate", j)); changed = True
        run_ready()
        for p, q in path:
            where = [q if v == p else p if v == q else v for v in where]
            events.append(("swap", min(p, q), max(p, q)))
            run_ready()
        assert sum(ptr) == total and len(path) == g
        return ExactResult("optimal", g, g, g < UB, layout, events, len(parent),
                           time.time() - t0)

    # ---- level 0: all initial placements
    zero = (0,) * n
    cur = []
    for r in roots:
        where = tuple(r)
        ptr = advance(where, zero, range(n))
        state = (where, ptr)
        if state in parent:
            continue
        if sum(ptr) == total:
            parent[state] = None
            return finish(state, 0)
        if h(where, ptr) >= UB:
            continue
        parent[state] = None
        cur.append(state)
    g = 0
    # ---- levels 1, 2, ...: one more SWAP each
    while cur and g + 1 < UB:
        nxt = []
        for state in cur:
            where, ptr = state
            occ = {p: i for i, p in enumerate(where)}
            for i, p in enumerate(where):
                for q in nbr[p]:
                    k = occ.get(q)
                    if k is not None and k < i:
                        continue                  # this SWAP was generated from k's side
                    w = list(where)
                    w[i] = q
                    moved = [i]
                    if k is not None:
                        w[k] = p; moved.append(k)
                    w = tuple(w)
                    np_ = advance(w, ptr, moved)
                    ns = (w, np_)
                    if ns in parent:
                        continue
                    if sum(np_) == total:
                        parent[ns] = (state, p, q)
                        return finish(ns, g + 1)
                    if g + 1 + h(w, np_) >= UB:
                        continue
                    parent[ns] = (state, p, q)
                    nxt.append(ns)
            if len(parent) > max_states or time.time() - t0 > time_limit:
                # levels <= g fully explored without a goal: optimum >= g + 1
                return ExactResult("limit", ub, g + 1, False, None, None, len(parent),
                                   time.time() - t0)
        cur = nxt
        g += 1
    # nothing cheaper than UB exists
    if ub is None:
        raise RuntimeError("search space exhausted without a solution")
    return ExactResult("optimal", ub, ub, False, None, None, len(parent), time.time() - t0)


def to_warm_start(res: ExactResult, gates) -> WarmStart:
    """Slot form of the search solution: gates between SWAPs share a slot."""
    slot_of = [None] * len(gates)
    gaps, cur, slot, open_slot = [], [], 0, False
    for ev in res.events:
        if ev[0] == "swap":
            if open_slot:
                slot += 1; open_slot = False
            cur.append((ev[1], ev[2]))
        else:
            if not open_slot:
                gaps.append(pack_rounds(cur)); cur = []; open_slot = True
            slot_of[ev[1]] = slot
    return WarmStart("exact_search", list(res.layout), slot_of, gaps)
