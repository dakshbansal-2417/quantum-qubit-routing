"""Analysis figures and summary tables from the experiment results.

    python analysis/plots.py

Reads experiments/results/{results.csv, solutions.json, scale_results.csv} and writes
analysis/figures/*.png and analysis/summary.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "experiments" / "results"
FIG = ROOT / "analysis" / "figures"

C_EASY, C_HARD, C_SABRE, C_GREY = "#d1495b", "#2e86ab", "#7a7a7a", "#c8c8c8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- layouts
def _draw(ax, sub, layout, title, swaps=None, gates=None):
    pos = {p: sub.pos[p] for p in range(sub.num_qubits)}
    used = {p: i for i, p in enumerate(layout)}
    nx.draw_networkx_edges(sub.graph, pos, ax=ax, edge_color=C_GREY, width=2)
    if gates:
        pairs = {}
        for a, b in gates:
            k = tuple(sorted((layout[a], layout[b])))
            pairs[k] = pairs.get(k, 0) + 1
        for (p, q), c in pairs.items():
            if sub.dist[p, q] > 1:
                ax.plot(*zip(pos[p], pos[q]), ls=":", color=C_EASY, lw=0.8 + 0.4 * c, alpha=0.7)
    if swaps:
        for p, q in swaps:
            ax.plot(*zip(pos[p], pos[q]), color=C_HARD, lw=4, alpha=0.5, solid_capstyle="round")
    nx.draw_networkx_nodes(sub.graph, pos, ax=ax, node_size=260, nodelist=list(pos),
                           node_color=["#1f3b57" if p in used else "white" for p in pos],
                           edgecolors="#1f3b57")
    nx.draw_networkx_labels(sub.graph, pos, ax=ax, font_size=8, font_color="white",
                            labels={p: f"q{i}" for p, i in used.items()})
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_layouts(sub, nodes, work, easy_layout, hard, path):
    """Easy layout (dotted = gates that are not adjacent) vs hard layout + SWAPs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    _draw(axes[0], sub, easy_layout, f"Easy layout\n(dotted: interacting pairs not wired together)",
          gates=work.gates)
    sw = [e for gap in hard.swaps for rnd in gap for e in rnd]
    _draw(axes[1], sub, hard.layout, f"Hard model: {len(sw)} SWAPs\n(blue: edges where SWAPs happen)",
          swaps=sw)
    fig.suptitle(f"{work.name} on FakeBrisbane qubits {min(nodes)}–{max(nodes)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- figures
def fig_price_of_realism(df):
    d = df[(df.hard > 0) | (df.easy_milp > 0)].sort_values("easy_milp")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.27, d.easy_milp, 0.27, color=C_EASY, label="Easy layout + optimal routing")
    ax.bar(x, d.hard, 0.27, color=C_HARD, label="Hard model (placement + routing)")
    ax.bar(x + 0.27, d.sabre, 0.27, color=C_SABRE, label="SABRE (Qiskit)")
    for i, (_, r) in enumerate(d.iterrows()):
        if r.hard_status != "optimal":
            ax.plot([i - 0.13, i + 0.13], [r.hard_bound] * 2, color="k", lw=1.5)
            ax.text(i, r.hard + 0.3, f"≥{int(r.hard_bound)}", ha="center", fontsize=7)
        else:
            ax.text(i, r.hard + 0.3, "✓", ha="center", fontsize=8, color=C_HARD)
    ax.set_xticks(x, d.circuit, rotation=35, ha="right")
    ax.set_ylabel("SWAPs")
    ax.set_title("Price of realism: SWAPs needed on FakeBrisbane (the easy model believes 0)\n"
                 "✓ = proven optimal, ≥ k = proven lower bound (black line)", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "price_of_realism.png", dpi=160)
    plt.close(fig)


def fig_scaling(df, sc):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    opt = df.hard_status == "optimal"
    ax.scatter(df.hard_vars[opt], df.hard_seconds[opt], color=C_HARD, label="proven optimal")
    ax.scatter(df.hard_vars[~opt], df.hard_seconds[~opt], color=C_EASY, marker="x", s=50,
               label="time limit (gap open)")
    for _, r in df[~opt].iterrows():
        ax.annotate(r.circuit, (r.hard_vars, r.hard_seconds), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("MILP binary variables (chip block)")
    ax.set_ylabel("solve time (s)")
    ax.set_title("Exact MILP on chip blocks")
    ax.legend(frameon=False)
    ax = axes[1]
    s = sc.dropna(subset=["milp_vars"])
    for suite, mk in (("small", "o"), ("medium", "s")):
        t = s[s.suite == suite]
        ax.scatter(t.n, t.milp_vars, marker=mk, label=f"QASMBench {suite}",
                   color=C_HARD if suite == "small" else C_EASY)
    ax.axhline(df.hard_vars[opt].max(), color="k", ls="--", lw=1)
    ax.text(s.n.min(), df.hard_vars[opt].max() * 1.3, "largest model solved to optimality",
            fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("logical qubits n")
    ax.set_ylabel("MILP binary variables")
    ax.set_title("Same MILP on the full 127-qubit chip")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "scaling.png", dpi=160)
    plt.close(fig)


def fig_full_chip(sc):
    s = sc.dropna(subset=["sabre"]).sort_values(["suite", "n"])
    s = s[(s.sabre > 0) | (s.easy_sabre > 0)]
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, s.easy_sabre, 0.4, color=C_EASY, label="Easy layout + SABRE routing")
    ax.bar(x + 0.2, s.sabre, 0.4, color=C_SABRE, label="SABRE layout + routing")
    ax.set_xticks(x, [f"{c} (n={n})" for c, n in zip(s.circuit, s.n.astype(int))],
                  rotation=45, ha="right", fontsize=7)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("SWAPs (symlog)")
    ax.set_title("Full 127-qubit chip: wiring-blind placement vs wiring-aware heuristic")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "full_chip.png", dpi=160)
    plt.close(fig)


def example_layouts(df):
    from easy_qp import solve_easy
    from hard_milp import HardResult  # noqa: F401
    from load_data import QASMBENCH_SMALL, load_circuit, load_hardware, subgraph
    sols = json.loads((RES / "solutions.json").read_text())
    hw = load_hardware()
    for name in ("adder_n4", "error_correctiond3_n5"):
        if name not in sols:
            continue
        s = sols[name]
        nodes = s["region_nodes"]
        sub = subgraph(hw, nodes)
        work = load_circuit(QASMBENCH_SMALL / name / f"{name}.qasm")
        hard = type("H", (), {"layout": s["hard"]["layout"],
                              "swaps": [[[tuple(e) for e in rnd] for rnd in gap]
                                        for gap in s["hard"]["swaps"]]})
        plot_layouts(sub, nodes, work, s["easy"]["layout"], hard, FIG / f"layout_{name}.png")


# --------------------------------------------------------------------------- summary
def summary(df, sc):
    nz = df[df.hard > 0]
    ratio = (nz.easy_milp / nz.hard)
    lines = [
        "# Results summary", "",
        "Generated by `python analysis/plots.py` from `experiments/results/`.", "",
        "## Chip-block experiments (exact methods)", "",
        f"- Circuits: {len(df)} QASMBench small circuits, all verified: {bool(df.verified.all())}",
        f"- Easy model certified optimal (Ky Fan bound): {int(df.easy_certified.sum())}/{len(df)}",
        f"- Hard model proven optimal: {int((df.hard_status == 'optimal').sum())}/{len(df)} "
        f"(MILP bound: {int((df.hard_proof == 'milp').sum())}, exact search: "
        f"{int((df.hard_proof == 'exact_search').sum())})",
        f"- Easy layout + optimal routing proven optimal: "
        f"{int((df.easy_milp_status == 'optimal').sum())}/{len(df)}",
        f"- Price of realism (circuits that need SWAPs, n={len(nz)}): easy layout needs "
        f"{ratio.mean():.2f}x the hard optimum on average, up to {ratio.max():.1f}x "
        f"({nz.circuit[ratio.idxmax()]})",
        f"- Hard model vs SABRE: equal on {int((df.hard == df.sabre).sum())}/{len(df)}, "
        f"better on {int((df.hard < df.sabre).sum())}", "",
        "| circuit | n | easy ideal | easy layout + optimal routing | hard | proven bound | SABRE |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r.circuit} | {r.n} | 0 | {int(r.easy_milp)} | {int(r.hard)} | "
                     f"{'optimal' if r.hard_status == 'optimal' else '≥ ' + str(int(r.hard_bound))}"
                     f" | {int(r.sabre)} |")
    if sc is not None:
        ok = sc.dropna(subset=["sabre"])
        nzs = ok[ok.sabre > 0]
        lines += ["", "## Full 127-qubit chip", "",
                  f"- Circuits routed: {len(ok)} ({int((ok.suite == 'medium').sum())} medium); "
                  f"skipped: {', '.join(sc[sc.sabre.isna()].circuit)}",
                  f"- Easy layout + SABRE vs SABRE: {(nzs.easy_sabre / nzs.sabre).mean():.2f}x "
                  f"SWAPs on average",
                  f"- MILP size on the full chip: {int(ok.milp_vars.min()):,} to "
                  f"{int(ok.milp_vars.max()):,} binary variables "
                  f"(largest solved to optimality on a block: {int(df.hard_vars[df.hard_status == 'optimal'].max()):,})"]
        notes = []
        if "hard_full" in sc:
            f = sc.dropna(subset=["hard_full", "hard_block"])
            same = (f.hard_full == f.hard_block).sum()
            lines += [f"- Small circuits re-solved with the MILP on the full chip (n ≤ 5): {len(f)}; "
                      f"same SWAP count as on the block: {same}/{len(f)}; optimality proven on "
                      f"the full chip: {int((f.hard_full_status == 'optimal').sum())}/{len(f)} "
                      f"(the 127-qubit model is much larger, so its lower bound stays weak)"]
            for _, r in f[f.hard_full != f.hard_block].iterrows():
                notes.append(f"- {r.circuit}: the full-chip MILP stopped at {int(r.hard_full)} "
                             f"SWAPs within the time limit; the block model found and proved "
                             f"{int(r.hard_block)}. More qubits make the MILP harder, not better.")
        blk = df.set_index("circuit")
        for _, r in ok.iterrows():
            if r.circuit in blk.index and r.sabre < blk.loc[r.circuit, "hard"]:
                notes.append(f"- **{r.circuit}: the chip block limited the answer.** On the full "
                             f"chip SABRE needs {int(r.sabre)} SWAPs, below the block's best "
                             f"{int(blk.loc[r.circuit, 'hard'])} (proven ≥ "
                             f"{int(blk.loc[r.circuit, 'hard_bound'])} only on the block). Its "
                             f"interactions form a chain, and a BFS-shaped block has no straight "
                             f"path that long. Block results are optimal for the block, not for "
                             f"the whole chip.")
        lines += ["", "## Limitations found", ""] + (notes or ["- none"])
        lines += ["", "| circuit | n | CNOTs | easy layout + SABRE | SABRE | MILP vars (full chip) |",
                  "|---|---|---|---|---|---|"]
        for _, r in ok.iterrows():
            lines.append(f"| {r.circuit} | {int(r.n)} | {int(r.gates)} | {int(r.easy_sabre)} | "
                         f"{int(r.sabre)} | {int(r.milp_vars):,} |")
    (ROOT / "analysis" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "results.csv")
    sc = pd.read_csv(RES / "scale_results.csv") if (RES / "scale_results.csv").exists() else None
    fig_price_of_realism(df)
    example_layouts(df)
    if sc is not None:
        fig_scaling(df, sc)
        fig_full_chip(sc)
    summary(df, sc)
    print("Wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png"))), "and summary.md")
