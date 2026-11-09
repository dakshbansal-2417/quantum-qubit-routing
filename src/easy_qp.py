"""EASY model: continuous (spectral) qubit placement.

Each logical qubit i gets a continuous position x_i in R^2. Qubits that interact
often should sit close together:

    min_X   sum_{i<j} w_ij ||x_i - x_j||^2  =  tr(X^T L X)     (L = Laplacian of w)
    s.t.    1^T X = 0        (centred: removes the trivial "everyone on one point")
            X^T X = I        (spread:  removes collapse onto a line / point)

Without the two constraints the optimum is X = 0 (all qubits on one spot, cost 0),
which is useless. With them, the global optimum is known in closed form
(Ky Fan / Rayleigh-Ritz): X = the eigenvectors of L for the 2nd and 3rd smallest
eigenvalues, with optimal value lambda_2 + lambda_3. That gives an exact
certificate of optimality.

The easy model ignores the wiring (any position is allowed) and the gate order.
To compare it with the hard model we snap the continuous positions onto the
physical qubits with a linear assignment problem (an LP with integral optimum).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from load_data import Hardware, Workload


@dataclass
class EasyResult:
    X: np.ndarray               # n x 2 continuous positions (the QP optimum)
    objective: float            # tr(X^T L X)
    eigenvalues: np.ndarray     # spectrum of L (for the certificate)
    layout: list[int]           # logical i -> physical qubit (index in region)
    static_cost: int            # sum over gates of (hop distance - 1): SWAP proxy
    certificate: dict           # verification numbers


def laplacian(w: np.ndarray) -> np.ndarray:
    return np.diag(w.sum(axis=1)) - w


def solve_spectral(w: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Global optimum of min tr(X^T L X) s.t. 1^T X = 0, X^T X = I."""
    n = w.shape[0]
    L = laplacian(w)
    vals, vecs = np.linalg.eigh(L)
    if n < 3:  # not enough eigenvectors for 2D; pad with zeros
        X = np.zeros((n, 2))
        X[:, : n - 1] = vecs[:, 1:n]
    else:
        X = vecs[:, 1:3]
    return X, float(np.trace(X.T @ L @ X)), vals


def certify(w: np.ndarray, X: np.ndarray, vals: np.ndarray) -> dict:
    """Check feasibility and that the objective equals the Ky Fan lower bound."""
    n = w.shape[0]
    L = laplacian(w)
    k = min(2, n - 1)
    obj = float(np.trace(X.T @ L @ X))
    bound = float(vals[1 : 1 + k].sum())
    return {
        "centred_err": float(np.abs(X.sum(axis=0)).max()),
        "orthonormal_err": float(np.abs(X[:, :k].T @ X[:, :k] - np.eye(k)).max()),
        "objective": obj,
        "kyfan_lower_bound": bound,
        "optimal": bool(abs(obj - bound) < 1e-8 * max(1.0, bound)),
    }


def static_cost(layout: list[int], gates: list[tuple[int, int]], dist: np.ndarray) -> int:
    """SWAPs a fixed layout would need if every gate were routed independently."""
    return int(sum(dist[layout[a], layout[b]] - 1 for a, b in gates))


def snap_to_hardware(X: np.ndarray, hw: Hardware, gates, n_angles: int = 24,
                     scales=(0.6, 0.8, 1.0, 1.25, 1.6)) -> list[int]:
    """Round continuous positions to distinct physical qubits.

    The spectral solution is only defined up to rotation/reflection and scale, so we
    try a grid of those and, for each, solve a linear assignment (squared distance
    from x_i to physical qubit p). We keep the assignment with the lowest static cost.
    """
    m = hw.num_qubits
    C = np.array([hw.pos[p] for p in range(m)])
    C = C - C.mean(axis=0)
    rms_c = np.sqrt((C ** 2).sum(axis=1).mean())
    Xc = X - X.mean(axis=0)
    rms_x = np.sqrt((Xc ** 2).sum(axis=1).mean()) or 1.0
    best, best_cost = None, None
    for flip in (1, -1):
        for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            Y = (Xc * [1, flip]) @ R.T / rms_x
            for s in scales:
                cost = ((Y[:, None, :] * s * rms_c * np.sqrt(X.shape[0] / m) - C[None]) ** 2).sum(-1)
                rows, cols = linear_sum_assignment(cost)
                layout = [0] * X.shape[0]
                for r, c in zip(rows, cols):
                    layout[r] = int(c)
                sc = static_cost(layout, gates, hw.dist)
                if best_cost is None or sc < best_cost:
                    best, best_cost = layout, sc
    return best


def solve_easy(work: Workload, hw: Hardware) -> EasyResult:
    w = work.weights
    X, obj, vals = solve_spectral(w)
    layout = snap_to_hardware(X, hw, work.gates)
    return EasyResult(X=X, objective=obj, eigenvalues=vals, layout=layout,
                      static_cost=static_cost(layout, work.gates, hw.dist),
                      certificate=certify(w, X, vals))
