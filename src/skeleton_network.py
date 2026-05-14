"""Top-k BLI skeleton network figures.

For each selected Congress, keep only the top-k highest-BLI legislators and
draw the cross-party edges among them (with same-party edges thinned to a
light backdrop). Nodes are positioned by DW-NOMINATE dim. 1 (x) and a small
vertical jitter (y) to match the rest of the paper's network panels.

Emits TikZ snippets (one per Congress) plus a 2x2 PDF preview.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR,
    DEM_COLOR, REP_COLOR, CROSS_COLOR,
)

PANEL_CONGRESSES = [100, 107, 112, 118]
TOP_K = 20
TIKZ_DIR = FIGURES_DIR / "skeleton_tikz"
TIKZ_DIR.mkdir(parents=True, exist_ok=True)


def ordinal(n: int) -> str:
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
    return f"{n}{suffix}"


def load_congress(c: int):
    bli_all = json.load(open(RESULTS_DIR / "bli_results.json"))
    data = np.load(PROCESSED_DIR / f"congress_{c}.npz", allow_pickle=True)
    bli_vals = np.asarray(bli_all[str(c)]["bli_values"], dtype=float)
    bli_ids = np.asarray(bli_all[str(c)]["member_ids"], dtype=int)
    npz_ids = np.asarray(data["member_ids"], dtype=int)
    # Align BLI vector to NPZ member order (BLI file may use different order).
    id_to_bli = dict(zip(bli_ids.tolist(), bli_vals.tolist()))
    bli = np.array([id_to_bli.get(int(m), 0.0) for m in npz_ids])
    return dict(
        adj=data["adjacency"],
        nom1=data["nominate_dim1"],
        party=data["party_codes"],
        names=data["member_names"],
        bli=bli,
    )


def top_k_indices(bli: np.ndarray, party: np.ndarray, k: int) -> np.ndarray:
    """Top-k by absolute BLI overall, but ensure at least k/2 from each major party."""
    half = k // 2
    idx_d = np.where(party == 100)[0]
    idx_r = np.where(party == 200)[0]
    top_d = idx_d[np.argsort(-np.abs(bli[idx_d]))[:half]]
    top_r = idx_r[np.argsort(-np.abs(bli[idx_r]))[:half]]
    return np.concatenate([top_d, top_r])


def shorten_name(s: str) -> str:
    return s.split(",")[0].title()


def emit_tikz(congress: int, sel: np.ndarray, ctx: dict) -> str:
    nom = ctx["nom1"][sel]
    bli = np.abs(ctx["bli"][sel])
    party = ctx["party"][sel]
    names = ctx["names"][sel]
    adj = ctx["adj"][np.ix_(sel, sel)]

    # Stable vertical placement: rank within party, scaled into [-0.4, 0.4].
    y = np.zeros_like(nom, dtype=float)
    for p in (100, 200):
        mask = party == p
        if mask.sum() == 0:
            continue
        order = np.argsort(-bli[mask])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(mask.sum())
        n = mask.sum()
        y[mask] = (ranks / max(n - 1, 1) - 0.5) * 0.8

    bli_max = bli.max() if bli.max() > 0 else 1.0
    sizes = 1.5 + 4.0 * (bli / bli_max)  # node radius in pt

    dem_hex = DEM_COLOR.lstrip("#")
    rep_hex = REP_COLOR.lstrip("#")
    cross_hex = CROSS_COLOR.lstrip("#")
    lines = [
        f"% Top-{TOP_K} BLI skeleton, {ordinal(congress)} Congress",
        f"% Requires: \\usepackage{{xcolor}} and \\usepackage{{tikz}}",
        f"\\definecolor{{demblue}}{{HTML}}{{{dem_hex}}}",
        f"\\definecolor{{repred}}{{HTML}}{{{rep_hex}}}",
        f"\\definecolor{{crossorange}}{{HTML}}{{{cross_hex}}}",
        "\\begin{tikzpicture}[x=4cm, y=2cm,",
        "    every node/.style={inner sep=0.5pt, font=\\tiny}]",
        "  % cross-party edges",
    ]
    n = len(sel)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] <= 0:
                continue
            if party[i] == party[j]:
                continue
            lines.append(
                f"  \\draw[crossorange, line width=0.3pt, opacity=0.55] "
                f"({nom[i]:.3f},{y[i]:.3f}) -- ({nom[j]:.3f},{y[j]:.3f});"
            )
    lines.append("  % same-party edges (backdrop)")
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] <= 0 or party[i] != party[j]:
                continue
            col = "demblue!25" if party[i] == 100 else "repred!25"
            lines.append(
                f"  \\draw[{col}, line width=0.15pt, opacity=0.4] "
                f"({nom[i]:.3f},{y[i]:.3f}) -- ({nom[j]:.3f},{y[j]:.3f});"
            )
    lines.append("  % nodes")
    for i in range(n):
        col = "demblue" if party[i] == 100 else "repred"
        r = sizes[i]
        label = shorten_name(str(names[i]))
        lines.append(
            f"  \\filldraw[fill={col}, draw=black, line width=0.15pt] "
            f"({nom[i]:.3f},{y[i]:.3f}) circle ({r:.2f}pt);"
        )
        lines.append(
            f"  \\node[anchor=south, font=\\tiny] at ({nom[i]:.3f},{y[i] + 0.05:.3f}) "
            f"{{\\textcolor{{black!70}}{{{label}}}}};"
        )
    lines.append("  \\node[anchor=north, font=\\small] at (0, -0.55) "
                 f"{{\\textbf{{{ordinal(congress)} Congress}}}};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def render_preview(panels: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, panel in zip(axes.ravel(), panels):
        sel = panel["sel"]
        ctx = panel["ctx"]
        nom = ctx["nom1"][sel]
        bli = np.abs(ctx["bli"][sel])
        party = ctx["party"][sel]
        adj = ctx["adj"][np.ix_(sel, sel)]

        y = np.zeros_like(nom, dtype=float)
        for p in (100, 200):
            mask = party == p
            if mask.sum() == 0:
                continue
            order = np.argsort(-bli[mask])
            ranks = np.empty_like(order)
            ranks[order] = np.arange(mask.sum())
            y[mask] = (ranks / max(mask.sum() - 1, 1) - 0.5) * 0.8

        same_segs, cross_segs = [], []
        n = len(sel)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] <= 0:
                    continue
                seg = [[nom[i], y[i]], [nom[j], y[j]]]
                if party[i] != party[j]:
                    cross_segs.append(seg)
                else:
                    same_segs.append(seg)
        if same_segs:
            ax.add_collection(LineCollection(same_segs, colors="#cccccc",
                                             linewidths=0.4, alpha=0.5, zorder=1))
        if cross_segs:
            ax.add_collection(LineCollection(cross_segs, colors=CROSS_COLOR,
                                             linewidths=0.9, alpha=0.8, zorder=2))

        bli_max = bli.max() if bli.max() > 0 else 1.0
        sizes = 20 + 120 * (bli / bli_max)
        dem = party == 100
        rep = party == 200
        ax.scatter(nom[dem], y[dem], s=sizes[dem], c=DEM_COLOR,
                   edgecolors="black", linewidths=0.3, zorder=3)
        ax.scatter(nom[rep], y[rep], s=sizes[rep], c=REP_COLOR,
                   edgecolors="black", linewidths=0.3, zorder=3)

        n_cross = len(cross_segs)
        ax.set_title(f"{ordinal(panel['congress'])} Congress  "
                     f"({n_cross} cross-party edges in top-{TOP_K})",
                     fontsize=10)
        ax.set_xlabel("DW-NOMINATE dim. 1")
        ax.set_ylabel("rank within party")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.6, 0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DEM_COLOR,
               markeredgecolor="black", markersize=6, label="Democrat"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REP_COLOR,
               markeredgecolor="black", markersize=6, label="Republican"),
        Line2D([0], [0], color=CROSS_COLOR, lw=1.2, label="Cross-party edge"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    panels = []
    for c in PANEL_CONGRESSES:
        ctx = load_congress(c)
        sel = top_k_indices(ctx["bli"], ctx["party"], TOP_K)
        tikz = emit_tikz(c, sel, ctx)
        out = TIKZ_DIR / f"skeleton_congress_{c}.tex"
        out.write_text(tikz)
        print(f"  wrote {out.name}")
        panels.append(dict(congress=c, sel=sel, ctx=ctx))

    preview = FIGURES_DIR / "skeleton_network_preview.pdf"
    render_preview(panels, preview)
    print(f"  wrote {preview.name}")


if __name__ == "__main__":
    main()
