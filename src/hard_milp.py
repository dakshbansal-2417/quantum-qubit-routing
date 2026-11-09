"""HARD model: time-expanded integer routing (MILP).

Time is split into states tau = 0..S-1. Gate k (in circuit order) runs at state
tau_k = k * R, and between two consecutive states there is one round of
parallel SWAPs, so up to R rounds fit between consecutive gates.

Variables
    x[i,p,t] in {0,1}   logical qubit i sits on physical qubit p at state t
    s[e,t]   in {0,1}   SWAP on hardware edge e = (p,q) between state t and t+1

Objective
    min  sum_{e,t} s[e,t]                                   (total SWAPs)

Constraints
    (1) sum_p x[i,p,t] = 1                     each logical qubit in one place
    (2) sum_i x[i,p,t] <= 1                    each physical qubit holds <= 1
    (3) x[a,p,tau_k] <= sum_{q in N(p)} x[b,q,tau_k]   gate k=(a,b) is on an edge
    (4) sum_{e ni p} s[e,t] <= 1               SWAPs in one round are disjoint
    (5) x[i,q,t+1] >= x[i,p,t] + s[pq,t] - 1   a SWAP moves the qubit (both ways)
    (6) x[i,p,t+1] >= x[i,p,t] - sum_{e ni p} s[e,t]   no SWAP at p: it stays
Gate order is built into the time indexing.

R is a modelling parameter; raising it can only lower (or keep) the optimum. We
check that R and R+1 give the same value as part of the verification.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from load_data import Hardware


@dataclass
class HardResult:
    status: str
    objective: float | None     # SWAPs in best solution found
    dual_bound: float | None    # proven lower bound
    gap: float | None
    layout: list[int] | None    # initial placement (logical -> region index)
    swaps: list[list[tuple[int, int]]] | None   # swaps[t] = SWAPs in round t
    rounds_per_gate: int
    num_vars: int
    num_constraints: int
    seconds: float


def merge_repeats(gates):
    """Drop gates that repeat the previous pair: no SWAP can be needed between them."""
    out = []
    for g in gates:
        if out and set(out[-1]) == set(g):
            continue
        out.append(tuple(g))
    return out


def solve_hard(gates, n: int, hw: Hardware, rounds_per_gate: int = 2,
               fixed_layout: list[int] | None = None, time_limit: float = 120.0,
               verbose: bool = False) -> HardResult:
    t0 = time.time()
    G = merge_repeats(gates)
    K, m, R = len(G), hw.num_qubits, rounds_per_gate
    E = hw.edges
    nbr = {p: list(hw.graph.neighbors(p)) for p in range(m)}
    edges_at = {p: [k for k, (u, v) in enumerate(E) if p in (u, v)] for p in range(m)}
    S = (K - 1) * R + 1 if K else 1
    T = S - 1  # number of SWAP rounds

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
    for k, (a, b) in enumerate(G):                           # (3)
        t = k * R
        for p in range(m):
            add([(X(a, p, t), 1)] + [(X(b, q, t), -1) for q in nbr[p]], -np.inf, 0)
    for t in range(T):
        for p in range(m):                                   # (4)
            if len(edges_at[p]) > 1:
                add([(Sv(e, t), 1) for e in edges_at[p]], -np.inf, 1)
        for e, (p, q) in enumerate(E):                       # (5)
            for i in range(n):
                add([(X(i, q, t + 1), 1), (X(i, p, t), -1), (Sv(e, t), -1)], -1, np.inf)
                add([(X(i, p, t + 1), 1), (X(i, q, t), -1), (Sv(e, t), -1)], -1, np.inf)
        for p in range(m):                                   # (6)
            for i in range(n):
                add([(X(i, p, t + 1), 1), (X(i, p, t), -1)]
                    + [(Sv(e, t), 1) for e in edges_at[p]], 0, np.inf)

    A = csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    c = np.zeros(nvar)
    c[nx_:] = 1.0
    lb, ub = np.zeros(nvar), np.ones(nvar)
    if fixed_layout is not None:                             # hard routing, easy layout
        for i, p in enumerate(fixed_layout):
            lb[X(i, p, 0)] = 1

    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nvar),
               bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": verbose, "mip_rel_gap": 0})
    secs = time.time() - t0

    status = {0: "optimal", 1: "time_limit", 2: "infeasible"}.get(res.status, res.message)
    if res.x is None:
        return HardResult(status, None, getattr(res, "mip_dual_bound", None), None, None,
                          None, R, nvar, r, secs)
    xv = np.round(res.x).astype(int)
    layout = [int(np.argmax([xv[X(i, p, 0)] for p in range(m)])) for i in range(n)]
    swaps = [[E[e] for e in range(len(E)) if xv[Sv(e, t)]] for t in range(T)]
    return HardResult(status=status, objective=float(round(res.fun)),
                      dual_bound=float(getattr(res, "mip_dual_bound", res.fun)),
                      gap=float(getattr(res, "mip_gap", 0.0)), layout=layout, swaps=swaps,
                      rounds_per_gate=R, num_vars=nvar, num_constraints=r, seconds=secs)
