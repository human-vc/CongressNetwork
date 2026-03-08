import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, CONGRESSES,
    DEM_COLOR, REP_COLOR,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_PURPLE = "#CC79A7"
NEUTRAL = "#999999"
DARK_TEXT = "#000000"
LIGHT_TEXT = "#000000"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def load_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fig_robustness_panels():
    null_path = RESULTS_DIR / "null_model_results.json"
    w_path = RESULTS_DIR / "weighted_spectral_results.json"
    if not null_path.exists() or not w_path.exists():
        print("  robustness_panels.pdf: missing data")
        return

    with open(null_path) as f:
        null_data = json.load(f)
    with open(w_path) as f:
        w_data = json.load(f)

    config = null_data["configuration_model"]
    temporal = null_data.get("temporal_null", {})

    # --- Panel A: Null models ---
    congresses_n, empirical, null_mean, ci_lo, ci_hi, temp_mean = [], [], [], [], [], []
    for c in CONGRESSES:
        cs = str(c)
        if cs not in config:
            continue
        congresses_n.append(c)
        empirical.append(config[cs]["empirical"])
        null_mean.append(config[cs]["null_mean"])
        ci_lo.append(config[cs]["null_ci_lo"])
        ci_hi.append(config[cs]["null_ci_hi"])
        temp_mean.append(temporal[cs]["mean"] if cs in temporal else None)

    # --- Panel B: Weighted comparison ---
    congresses_w, binary_vals, weighted_vals = [], [], []
    for c in CONGRESSES:
        cs = str(c)
        if cs not in w_data or not isinstance(w_data[cs], dict):
            continue
        congresses_w.append(c)
        binary_vals.append(w_data[cs]["binary_fiedler"])
        weighted_vals.append(w_data[cs]["weighted_fiedler"])
    corr = w_data.get("correlation", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel A
    ax1.fill_between(congresses_n, ci_lo, ci_hi, alpha=0.18, color=NEUTRAL,
                     linewidth=0, zorder=1)
    ax1.plot(congresses_n, null_mean, "--", color=NEUTRAL, linewidth=0.9,
             alpha=0.7, zorder=2)
    temp_c = [c for c, v in zip(congresses_n, temp_mean) if v is not None]
    temp_v = [v for v in temp_mean if v is not None]
    if temp_c:
        ax1.plot(temp_c, temp_v, ":", color=OI_ORANGE, linewidth=1.1,
                 alpha=0.85, zorder=2)
    ax1.plot(congresses_n, empirical, "o-", color=OI_BLUE, markersize=3,
             linewidth=1.2, zorder=4)

    above = [(c, e) for c, e, hi in zip(congresses_n, empirical, ci_hi) if e > hi]
    if above:
        ax1.scatter([c for c, _ in above], [e for _, e in above],
                    marker="^", s=35, facecolors="none", edgecolors=OI_VERMILLION,
                    linewidths=1.0, zorder=5)

    # Legend with symbols, center-right
    from matplotlib.lines import Line2D as _L2D
    leg_a = [
        _L2D([0], [0], marker="o", color=OI_BLUE, linewidth=1.2, markersize=3,
             label=r"Empirical $\lambda_2$"),
        _L2D([0], [0], linestyle="--", color=NEUTRAL, linewidth=0.9,
             label="Config. model mean"),
        _L2D([0], [0], linestyle=":", color=OI_ORANGE, linewidth=1.1,
             label="Linear-decline null"),
    ]
    ax1.legend(handles=leg_a, fontsize=6,
               handlelength=1.5, handletextpad=0.4, labelspacing=0.15,
               borderpad=0.1, borderaxespad=0.0,
               bbox_to_anchor=(1.02, 1.05), loc="upper right")

    ax1.set_xlabel("Congress")
    ax1.set_ylabel(r"Fiedler value ($\lambda_2$)")
    ax1.set_xlim(congresses_n[0] - 0.5, congresses_n[-1] + 0.5)
    remove_spines(ax1)
    panel_label(ax1, "A", x=-0.16, y=1.08)

    # Panel B
    ax2.plot(congresses_w, binary_vals, "o-", color=OI_BLUE, markersize=3,
             linewidth=1.2, zorder=3)
    ax2.plot(congresses_w, weighted_vals, "s--", color=OI_VERMILLION, markersize=3,
             linewidth=1.0, zorder=3)
    ax2.fill_between(congresses_w, binary_vals, weighted_vals, alpha=0.06,
                     color="#666666", linewidth=0, zorder=1)

    # Legend with symbols, top-right
    leg_b = [
        _L2D([0], [0], marker="o", color=OI_BLUE, linewidth=1.2, markersize=3,
             label=r"Binary ($\tau = 0.5$)"),
        _L2D([0], [0], marker="s", linestyle="--", color=OI_VERMILLION,
             linewidth=1.0, markersize=3, label="Weighted (continuous)"),
    ]
    ax2.legend(handles=leg_b, fontsize=6,
               handlelength=1.5, handletextpad=0.4, labelspacing=0.15,
               borderpad=0.1, borderaxespad=0.0,
               bbox_to_anchor=(1.0, 1.05), loc="upper right")

    ax2.set_xlabel("Congress")
    ax2.set_ylabel(r"Fiedler value ($\lambda_2$)")
    ax2.set_xlim(congresses_w[0] - 0.5, congresses_w[-1] + 0.5)
    remove_spines(ax2)
    panel_label(ax2, "B", x=-0.16, y=1.08)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES_DIR / "robustness_panels.pdf")
    plt.close()
    print("  robustness_panels.pdf")


def fig_fiedler_party_distance():
    spectral = load_json("spectral_results.json")
    nom_data = spectral.get("nominate_distance", {})

    congresses, house_fiedler, nom_dist = [], [], []
    for c in CONGRESSES:
        cs = str(c)
        if cs not in spectral or "fiedler" not in spectral[cs]:
            continue
        congresses.append(c)
        house_fiedler.append(spectral[cs]["fiedler"])
        nom_dist.append(nom_data.get(cs, None))

    fig, ax1 = plt.subplots(figsize=(6.5, 3.2))

    ax1.plot(congresses, house_fiedler, "o-", color=OI_BLUE, markersize=4,
             linewidth=1.4, label=r"Fiedler value ($\lambda_2$)", zorder=3)
    ax1.set_xlabel("Congress")
    ax1.set_ylabel(r"Fiedler value ($\lambda_2$)", color=OI_BLUE)
    ax1.tick_params(axis="y", labelcolor=OI_BLUE, colors=OI_BLUE)
    ax1.spines["left"].set_color(OI_BLUE)
    remove_spines(ax1)
    ax1.set_xlim(congresses[0] - 0.5, congresses[-1] + 0.5)

    ax2 = ax1.twinx()
    nom_c = [c for c, v in zip(congresses, nom_dist) if v is not None]
    nom_v = [v for v in nom_dist if v is not None]
    if nom_c:
        ax2.plot(nom_c, nom_v, "^--", color=OI_VERMILLION, markersize=3.5,
                 linewidth=1.0, alpha=0.85, label="NOMINATE distance")
    ax2.set_ylabel("Median party NOMINATE distance", color=OI_VERMILLION)
    ax2.tick_params(axis="y", labelcolor=OI_VERMILLION, colors=OI_VERMILLION)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(OI_VERMILLION)

    events = {
        104: "Contract w/ America",
        107: "Post-9/11",
        111: "Obama supermajority",
        112: "Tea Party wave",
    }
    ymin, ymax = ax1.get_ylim()
    for c_evt, label in events.items():
        if c_evt in congresses:
            ax1.axvline(c_evt, color="#bbbbbb", linewidth=0.4, linestyle=":", zorder=1)
            idx = congresses.index(c_evt)
            fval = house_fiedler[idx]
            ax1.annotate(
                label, xy=(c_evt, fval), xytext=(c_evt + 0.8, fval + (ymax - ymin) * 0.12),
                fontsize=6.5, color=LIGHT_TEXT,
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", linewidth=0.4),
                va="bottom", ha="left",
            )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fiedler_party_distance.pdf")
    plt.close()
    print("  fiedler_party_distance.pdf")


def fig_sri_bars():
    spectral = load_json("spectral_results.json")

    congresses_sri, sri_vals = [], []
    for c in CONGRESSES:
        cs = str(c)
        if cs in spectral and isinstance(spectral[cs], dict) and "sri" in spectral[cs]:
            val = spectral[cs]["sri"]
            if val > 0:
                congresses_sri.append(c)
                sri_vals.append(val)

    if not congresses_sri:
        print("  sri_bars.pdf: no data")
        return

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(congresses_sri))

    highlight_set = {104, 107, 111, 112}
    colors = [OI_VERMILLION if c in highlight_set else OI_SKY for c in congresses_sri]

    bars = ax.bar(x, sri_vals, width=0.7, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in congresses_sri], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Congress (transition from previous)")
    ax.set_ylabel("Structural Realignment Index")

    highlight = {104: "Contract w/\nAmerica", 107: "Post-9/11", 111: "Obama", 112: "Tea Party"}
    for c, label in highlight.items():
        if c in congresses_sri:
            idx = congresses_sri.index(c)
            ax.annotate(label, xy=(idx, sri_vals[idx]), xytext=(idx, sri_vals[idx] + 0.003),
                        ha="center", va="bottom", fontsize=6, color=DARK_TEXT)

    remove_spines(ax)
    ax.set_ylim(0, max(sri_vals) * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sri_bars.pdf")
    plt.close()
    print("  sri_bars.pdf")


def fig_network_comparison():
    early, late = 103, 114
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    ordinals = {103: "103rd", 114: "114th"}

    for panel_idx, (ax, congress_num) in enumerate(zip(axes, [early, late])):
        path = PROCESSED_DIR / f"congress_{congress_num}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        adj = data["adjacency"]
        nom1 = data["nominate_dim1"]
        party = data["party_codes"]
        nom2 = data["features"][:, 1]

        rng = np.random.RandomState(42)
        jx = rng.normal(0, 0.015, len(nom1))
        jy = rng.normal(0, 0.015, len(nom2))
        px = nom1 + jx
        py = nom2 + jy

        rows, cols = np.nonzero(adj)
        mask = rows < cols

        same_segs, cross_segs = [], []
        for k in np.where(mask)[0][::4]:
            i, j = rows[k], cols[k]
            seg = [[px[i], py[i]], [px[j], py[j]]]
            if party[i] != party[j]:
                cross_segs.append(seg)
            else:
                same_segs.append(seg)

        if same_segs:
            lc_same = LineCollection(same_segs, colors="#d0d0d0", linewidths=0.15,
                                     alpha=0.25, zorder=1, rasterized=True)
            ax.add_collection(lc_same)
        if cross_segs:
            lc_cross = LineCollection(cross_segs, colors=OI_ORANGE, linewidths=0.2,
                                      alpha=0.18, zorder=2, rasterized=True)
            ax.add_collection(lc_cross)

        dem_mask = party == 100
        rep_mask = party == 200
        ax.scatter(px[dem_mask], py[dem_mask], c=DEM_COLOR, s=8, alpha=0.8,
                   zorder=3, edgecolors="none", rasterized=True)
        ax.scatter(px[rep_mask], py[rep_mask], c=REP_COLOR, s=8, alpha=0.8,
                   zorder=3, edgecolors="none", rasterized=True)

        n_edges = int(mask.sum())
        n_cross = int(np.sum(party[rows[mask]] != party[cols[mask]]))
        cross_pct = 100 * n_cross / n_edges if n_edges > 0 else 0

        title = ordinals.get(congress_num, f"{congress_num}th") + " Congress"
        stats_str = f"{n_edges:,} edges, {n_cross:,} cross-party ({cross_pct:.0f}%)"
        ax.set_title(f"{title}\n{stats_str}", fontweight="normal", fontsize=9, pad=6,
                     linespacing=1.6)
        ax.title.set_multialignment("center")

        ax.set_xlabel("DW-NOMINATE dim. 1")
        if panel_idx == 0:
            ax.set_ylabel("DW-NOMINATE dim. 2")
        else:
            ax.set_ylabel("")
        remove_spines(ax)
        ax.autoscale_view()

        panel_label(ax, chr(65 + panel_idx), x=-0.14, y=1.14)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DEM_COLOR,
               markersize=4, label="Democrat"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REP_COLOR,
               markersize=4, label="Republican"),
        Line2D([0], [0], color=OI_ORANGE, linewidth=1, alpha=0.5, label="Cross-party edge"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=5.5,
               handletextpad=0.3, borderpad=0.2, handlelength=1.2,
               labelspacing=0.2, bbox_to_anchor=(0.98, 0.95))

    fig.tight_layout(w_pad=1.5, rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "network_comparison.pdf")
    plt.close()
    print("  network_comparison.pdf")


def fig_bli_over_time():
    bli_data = load_json("bli_results.json")

    congresses_bli, mean_abs_bli, max_bli = [], [], []
    for c in CONGRESSES:
        cs = str(c)
        if cs not in bli_data:
            continue
        vals = np.array(bli_data[cs]["bli_values"])
        congresses_bli.append(c)
        mean_abs_bli.append(np.mean(np.abs(vals)))
        max_bli.append(np.max(vals))

    fig, ax1 = plt.subplots(figsize=(6.5, 3.2))

    ln1 = ax1.plot(congresses_bli, mean_abs_bli, "o-", color=OI_BLUE, markersize=4,
                   linewidth=1.4, label=r"Mean $|\mathrm{BLI}|$", zorder=3)
    ax1.set_xlabel("Congress")
    ax1.set_ylabel(r"Mean $|\mathrm{BLI}|$", color=OI_BLUE)
    ax1.tick_params(axis="y", labelcolor=OI_BLUE, colors=OI_BLUE)
    ax1.spines["left"].set_color(OI_BLUE)
    ax1.set_ylim(0, max(mean_abs_bli) * 1.15)
    ax1.set_xlim(congresses_bli[0] - 0.5, congresses_bli[-1] + 0.5)
    remove_spines(ax1)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(congresses_bli, max_bli, "v--", color=OI_VERMILLION, markersize=3,
                   linewidth=0.9, alpha=0.8, label="Max BLI", zorder=2)
    ax2.set_ylabel("Max BLI", color=OI_VERMILLION)
    ax2.tick_params(axis="y", labelcolor=OI_VERMILLION, colors=OI_VERMILLION)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(OI_VERMILLION)
    ax2.set_ylim(0, max(max_bli) * 1.15)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bli_over_time.pdf")
    plt.close()
    print("  bli_over_time.pdf")


def fig_bli_regression_coefs():
    reg = load_json("bli_regression_results.json")

    with_bli = reg["with_bli"]
    params = with_bli["params"]
    pvals = with_bli["pvalues"]
    bse = with_bli["bse"]

    vars_to_plot = ["bli", "ideology_distance", "seniority", "is_republican"]
    labels = ["BLI", "Ideology\ndistance", "Seniority", "Republican"]

    coefs = [params[v] for v in vars_to_plot]
    errors = [1.96 * bse[v] for v in vars_to_plot]
    pvalues = [pvals[v] for v in vars_to_plot]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), gridspec_kw={"width_ratios": [1, 1.1]})

    ax = axes[0]
    y = np.arange(len(vars_to_plot))
    bar_colors = [OI_VERMILLION if p < 0.001 else OI_ORANGE if p < 0.05 else NEUTRAL for p in pvalues]

    ax.barh(y, coefs, xerr=errors, color=bar_colors, alpha=0.85,
            edgecolor="none", height=0.6, capsize=2.5,
            error_kw={"linewidth": 0.7, "capthick": 0.7})
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("GEE coefficient (log-odds)")

    for i, (c, p) in enumerate(zip(coefs, pvalues)):
        stars = sig_stars(p)
        if stars:
            offset = errors[i] + max(abs(c) * 0.04, 0.3)
            ax.text(c + offset if c >= 0 else c - offset, i, stars,
                    ha="left" if c >= 0 else "right", va="center",
                    fontsize=9, fontweight="bold", color=DARK_TEXT)

    remove_spines(ax)
    panel_label(ax, "A")

    ax2 = axes[1]
    eras = reg["era_splits"]
    era_names = ["early (100-106)", "middle (107-112)", "late (113-116)"]
    era_labels = ["100th--106th", "107th--112th", "113th--116th"]

    bli_coefs, bli_sig = [], []
    for era in era_names:
        if era in eras:
            bli_coefs.append(eras[era]["params"]["bli"])
            bli_sig.append(eras[era]["pvalues"]["bli"])
        else:
            bli_coefs.append(0)
            bli_sig.append(1)

    y2 = np.arange(len(era_labels))
    bar_colors2 = [OI_VERMILLION if p < 0.01 else OI_ORANGE if p < 0.05 else NEUTRAL for p in bli_sig]

    ax2.barh(y2, bli_coefs, color=bar_colors2, alpha=0.85, edgecolor="none", height=0.6)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(era_labels)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("BLI coefficient (log-odds)")

    for i, (c, p) in enumerate(zip(bli_coefs, bli_sig)):
        stars = sig_stars(p)
        p_str = f"$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"
        display = f"{p_str} {stars}" if stars else f"{p_str}"
        x_pos = max(c + 8, 15)
        ax2.text(x_pos, i, display, ha="left", va="center", fontsize=6.5, color=LIGHT_TEXT)

    remove_spines(ax2)
    panel_label(ax2, "B")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGURES_DIR / "bli_regression_coefs.pdf")
    plt.close()
    print("  bli_regression_coefs.pdf")


def fig_house_senate_fiedler():
    spectral = load_json("spectral_results.json")
    senate_data = spectral.get("senate_fiedler", {})

    congresses, house_vals, senate_vals = [], [], []
    for c in CONGRESSES:
        cs = str(c)
        if cs in spectral and isinstance(spectral[cs], dict) and "fiedler" in spectral[cs]:
            h = spectral[cs]["fiedler"]
            s = senate_data.get(cs, None)
            if s is not None:
                congresses.append(c)
                house_vals.append(h)
                senate_vals.append(s)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    ax.plot(congresses, house_vals, "o-", color=OI_BLUE, markersize=4,
            linewidth=1.4, label="House", zorder=3)
    ax.plot(congresses, senate_vals, "s--", color=OI_VERMILLION, markersize=3.5,
            linewidth=1.2, label="Senate", zorder=3)

    ax.fill_between(congresses, house_vals, senate_vals, alpha=0.06,
                    color="#666666", linewidth=0, zorder=1)

    ax.set_xlabel("Congress")
    ax.set_ylabel(r"Fiedler value ($\lambda_2$)")
    ax.legend(loc="upper right")
    ax.set_xlim(congresses[0] - 0.5, congresses[-1] + 0.5)
    remove_spines(ax)

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.12)
    events = {107: ("9/11", 0), 111: ("Obama", -1.0), 112: ("Tea Party", 1.0)}
    ymin, ymax = ax.get_ylim()
    for c_evt, (label, x_nudge) in events.items():
        if c_evt in congresses:
            ax.axvline(c_evt, color="#cccccc", linewidth=0.4, linestyle=":", zorder=1)
            ax.text(c_evt + x_nudge, ymax * 0.97, label,
                    fontsize=6.5, color=LIGHT_TEXT, rotation=0, ha="center", va="top")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "house_senate_fiedler.pdf")
    plt.close()
    print("  house_senate_fiedler.pdf")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")

    fig_robustness_panels()
    fig_fiedler_party_distance()
    fig_sri_bars()
    fig_network_comparison()
    fig_bli_over_time()
    fig_bli_regression_coefs()
    fig_house_senate_fiedler()

    print("All figures generated.")


if __name__ == "__main__":
    main()
