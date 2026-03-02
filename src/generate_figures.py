import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, CONGRESSES,
    TEST_CONGRESSES, ANALYSIS_CONGRESS,
    DEM_COLOR, REP_COLOR, CROSS_COLOR,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def fig_network_comparison():
    early, late = 104, 117
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, congress_num, title in zip(axes, [early, late], [f"{early}th Congress", f"{late}th Congress"]):
        path = PROCESSED_DIR / f"congress_{congress_num}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        adj = data["adjacency"]
        nom1 = data["nominate_dim1"]
        party = data["party_codes"]

        rng = np.random.RandomState(42)
        nom2 = data["features"][:, 1]
        jitter_x = rng.normal(0, 0.02, len(nom1))
        jitter_y = rng.normal(0, 0.02, len(nom2))

        rows, cols = np.nonzero(adj)
        for k in range(0, len(rows), 3):
            i, j = rows[k], cols[k]
            if i < j:
                cross = party[i] != party[j]
                ax.plot(
                    [nom1[i] + jitter_x[i], nom1[j] + jitter_x[j]],
                    [nom2[i] + jitter_y[i], nom2[j] + jitter_y[j]],
                    color=CROSS_COLOR if cross else "#cccccc",
                    alpha=0.15 if cross else 0.05,
                    linewidth=0.3,
                    zorder=1,
                )

        dem_mask = party == 100
        rep_mask = party == 200
        ax.scatter(nom1[dem_mask] + jitter_x[dem_mask], nom2[dem_mask] + jitter_y[dem_mask],
                   c=DEM_COLOR, s=12, alpha=0.7, zorder=2, edgecolors="none")
        ax.scatter(nom1[rep_mask] + jitter_x[rep_mask], nom2[rep_mask] + jitter_y[rep_mask],
                   c=REP_COLOR, s=12, alpha=0.7, zorder=2, edgecolors="none")

        ax.set_xlabel("NOMINATE Dim. 1")
        ax.set_ylabel("NOMINATE Dim. 2")
        ax.set_title(title, fontweight="normal")
        remove_spines(ax)

    plt.tight_layout(w_pad=2.0)
    fig.savefig(FIGURES_DIR / "network_comparison.pdf")
    plt.close()
    print("  network_comparison.pdf")


def fig_fiedler_trajectory():
    spectral = load_json("spectral_results.json")

    congresses = []
    house_fiedler = []
    senate_fiedler_vals = []
    nom_dist = []

    senate_data = spectral.get("senate_fiedler", {})
    nom_data = spectral.get("nominate_distance", {})

    for c in CONGRESSES:
        cs = str(c)
        if cs not in spectral or "fiedler" not in spectral[cs]:
            continue
        congresses.append(c)
        house_fiedler.append(spectral[cs]["fiedler"])
        senate_fiedler_vals.append(senate_data.get(cs, None))
        nom_dist.append(nom_data.get(cs, None))

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(congresses, house_fiedler, "o-", color="#1b7837", markersize=4, linewidth=1.5, label="House Fiedler")
    senate_c = [c for c, v in zip(congresses, senate_fiedler_vals) if v is not None]
    senate_v = [v for v in senate_fiedler_vals if v is not None]
    if senate_c:
        ax1.plot(senate_c, senate_v, "s--", color="#1b7837", markersize=3, linewidth=1.0, alpha=0.6, label="Senate Fiedler")

    ax1.set_xlabel("Congress")
    ax1.set_ylabel("Fiedler Value ($\\lambda_2$)", color="#1b7837")
    ax1.tick_params(axis="y", labelcolor="#1b7837")
    remove_spines(ax1)

    ax2 = ax1.twinx()
    nom_c = [c for c, v in zip(congresses, nom_dist) if v is not None]
    nom_v = [v for v in nom_dist if v is not None]
    if nom_c:
        ax2.plot(nom_c, nom_v, "^-", color="#762a83", markersize=3, linewidth=1.2, alpha=0.8, label="NOMINATE Distance")
    ax2.set_ylabel("Median Party NOMINATE Distance", color="#762a83")
    ax2.tick_params(axis="y", labelcolor="#762a83")
    ax2.spines["top"].set_visible(False)

    events = {104: "Contract w/ America", 110: "Dem Majority", 115: "Trump Era"}
    for c_evt, label in events.items():
        if c_evt in congresses:
            ax1.axvline(c_evt, color="#999999", linewidth=0.5, linestyle=":", alpha=0.7)
            ax1.text(c_evt + 0.3, ax1.get_ylim()[1] * 0.95, label,
                     fontsize=7, color="#666666", rotation=90, va="top")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fiedler_trajectory.pdf")
    plt.close()
    print("  fiedler_trajectory.pdf")


def fig_attention():
    eval_results = load_json("evaluation_results.json")
    attn = eval_results.get("attention", {})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    congresses_with_attn = sorted([int(c) for c in attn.keys()])
    same_means = [attn[str(c)]["same_party_mean"] for c in congresses_with_attn]
    cross_means = [attn[str(c)]["cross_party_mean"] for c in congresses_with_attn]

    x = np.arange(len(congresses_with_attn))
    w = 0.35
    axes[0].bar(x - w/2, same_means, w, color=DEM_COLOR, alpha=0.8, label="Same-party")
    axes[0].bar(x + w/2, cross_means, w, color=CROSS_COLOR, alpha=0.8, label="Cross-party")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(c) for c in congresses_with_attn], fontsize=8)
    axes[0].set_xlabel("Congress")
    axes[0].set_ylabel("Mean Attention Weight")
    axes[0].set_title("GAT Attention by Party Alignment", fontweight="normal")
    axes[0].legend(fontsize=8, frameon=False)
    remove_spines(axes[0])

    analysis_c = str(ANALYSIS_CONGRESS)
    if analysis_c in attn:
        path = PROCESSED_DIR / f"congress_{ANALYSIS_CONGRESS}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            nom1 = data["nominate_dim1"]
            sorted_idx = np.argsort(nom1)
            party = data["party_codes"]

            n = len(nom1)
            heatmap = np.zeros((n, n))
            adj = data["adjacency"]
            agreement = data["agreement"]
            heatmap = agreement[np.ix_(sorted_idx, sorted_idx)]

            im = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=0.3, vmax=0.9, aspect="auto")
            axes[1].set_xlabel("Member (sorted by ideology)")
            axes[1].set_ylabel("Member (sorted by ideology)")
            axes[1].set_title(f"Agreement Matrix, {ANALYSIS_CONGRESS}th", fontweight="normal")
            plt.colorbar(im, ax=axes[1], shrink=0.8, label="Agreement")

    plt.tight_layout(w_pad=2.5)
    fig.savefig(FIGURES_DIR / "attention_analysis.pdf")
    plt.close()
    print("  attention_analysis.pdf")


def fig_roc_curves():
    eval_results = load_json("evaluation_results.json")
    baseline_path = RESULTS_DIR / "baseline_results.json"
    has_baselines = baseline_path.exists()
    if has_baselines:
        baselines = load_json("baseline_results.json")

    n_test = len(TEST_CONGRESSES)
    fig, axes = plt.subplots(1, n_test, figsize=(4 * n_test, 4))
    if n_test == 1:
        axes = [axes]

    colors = {"GAT": "#e66101", "GCN": "#5e3c99", "RF": "#1b9e77", "LR": "#d95f02"}

    for idx, c in enumerate(TEST_CONGRESSES):
        ax = axes[idx]
        cs = str(c)

        for model_key, label, color in [("gat", "GAT", colors["GAT"]), ("gcn", "GCN", colors["GCN"])]:
            if cs in eval_results.get(model_key, {}):
                fpr = eval_results[model_key][cs]["roc_fpr"]
                tpr = eval_results[model_key][cs]["roc_tpr"]
                auc = eval_results[model_key][cs]["auc"]
                ax.plot(fpr, tpr, color=color, linewidth=1.3, label=f"{label} (AUC={auc:.2f})")

        if has_baselines:
            for model_key, label, color in [("rf", "RF", colors["RF"]), ("lr", "LR", colors["LR"])]:
                if cs in baselines.get(model_key, {}):
                    probs = np.array(baselines[model_key][cs]["probabilities"])
                    path_npz = PROCESSED_DIR / f"congress_{c}.npz"
                    if path_npz.exists():
                        y_true = np.load(path_npz, allow_pickle=True)["labels"]
                        from sklearn.metrics import roc_curve as sk_roc
                        fpr, tpr, _ = sk_roc(y_true, probs)
                        auc = baselines[model_key][cs]["auc"]
                        ax.plot(fpr, tpr, color=color, linewidth=1.0, linestyle="--", label=f"{label} (AUC={auc:.2f})")

        ax.plot([0, 1], [0, 1], "k:", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        if idx == 0:
            ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{c}th Congress", fontweight="normal")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        remove_spines(ax)

    plt.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES_DIR / "roc_curves.pdf")
    plt.close()
    print("  roc_curves.pdf")


def fig_bli_scatter():
    bli_data = load_json("bli_results.json")

    target_congresses = [110, 114, ANALYSIS_CONGRESS]
    available = [c for c in target_congresses if str(c) in bli_data]
    if not available:
        print("  bli_scatter.pdf: no data")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, c in enumerate(available):
        ax = axes[idx]
        cs = str(c)
        path = PROCESSED_DIR / f"congress_{c}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        nom1 = data["nominate_dim1"]
        party = data["party_codes"]
        member_names = data["member_names"]
        bli = np.array(bli_data[cs]["bli_values"])

        dem_mask = party == 100
        rep_mask = party == 200
        ax.scatter(nom1[dem_mask], bli[dem_mask], c=DEM_COLOR, s=15, alpha=0.6, edgecolors="none", label="Democrat")
        ax.scatter(nom1[rep_mask], bli[rep_mask], c=REP_COLOR, s=15, alpha=0.6, edgecolors="none", label="Republican")

        top_k = np.argsort(-np.abs(bli))[:3]
        for k in top_k:
            name = str(member_names[k]).split(",")[0]
            ax.annotate(name, (nom1[k], bli[k]), fontsize=6, alpha=0.8,
                       xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("NOMINATE Dim. 1")
        if idx == 0:
            ax.set_ylabel("Bridging Legislator Index")
        ax.set_title(f"{c}th Congress", fontweight="normal")
        ax.legend(fontsize=7, frameon=False)
        ax.axhline(0, color="#999999", linewidth=0.5, linestyle=":")
        remove_spines(ax)

    plt.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES_DIR / "bli_scatter.pdf")
    plt.close()
    print("  bli_scatter.pdf")


def fig_counterfactual():
    spectral = load_json("spectral_results.json")

    congresses_cf = []
    delta_bli = []
    delta_ideo = []
    delta_rand = []

    for c in CONGRESSES:
        cs = str(c)
        if cs in spectral and "counterfactual" in spectral[cs]:
            cf = spectral[cs]["counterfactual"]
            congresses_cf.append(c)
            delta_bli.append(cf["delta_bli"])
            delta_ideo.append(cf["delta_ideology"])
            delta_rand.append(cf["delta_random"])

    if not congresses_cf:
        print("  counterfactual_bars.pdf: no data")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(congresses_cf))
    w = 0.25

    ax.bar(x - w, delta_bli, w, color="#e66101", alpha=0.85, label="Remove Top-BLI")
    ax.bar(x, delta_ideo, w, color="#5e3c99", alpha=0.85, label="Remove Top-Ideology")
    ax.bar(x + w, delta_rand, w, color="#999999", alpha=0.7, label="Remove Random")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in congresses_cf], fontsize=8, rotation=45)
    ax.set_xlabel("Congress")
    ax.set_ylabel("$\\Delta$ Fiedler Value")
    ax.set_title("Counterfactual: Fiedler Change After Removing Top-10 Members", fontweight="normal")
    ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="black", linewidth=0.5)
    remove_spines(ax)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "counterfactual_bars.pdf")
    plt.close()
    print("  counterfactual_bars.pdf")


def fig_model_comparison():
    eval_results = load_json("evaluation_results.json")
    baseline_path = RESULTS_DIR / "baseline_results.json"
    has_baselines = baseline_path.exists()
    if has_baselines:
        baselines = load_json("baseline_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = ["GAT", "GCN"]
    model_keys = ["gat", "gcn"]
    colors = ["#e66101", "#5e3c99", "#1b9e77", "#d95f02"]

    if has_baselines:
        models += ["RF", "LR"]
        model_keys += ["rf", "lr"]

    metrics = [("auc", "AUC"), ("f1", "F1 Score")]
    for metric_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[metric_idx]
        x = np.arange(len(TEST_CONGRESSES))
        w = 0.8 / len(models)

        for m_idx, (model_name, model_key) in enumerate(zip(models, model_keys)):
            vals = []
            for c in TEST_CONGRESSES:
                cs = str(c)
                if model_key in ("gat", "gcn"):
                    source = eval_results.get(model_key, {})
                else:
                    source = baselines.get(model_key, {}) if has_baselines else {}
                vals.append(source.get(cs, {}).get(metric_key, 0))

            offset = (m_idx - len(models)/2 + 0.5) * w
            ax.bar(x + offset, vals, w * 0.9, color=colors[m_idx], alpha=0.85, label=model_name)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}th" for c in TEST_CONGRESSES])
        ax.set_ylabel(metric_label)
        ax.set_title(f"Test Set {metric_label} by Congress", fontweight="normal")
        ax.legend(fontsize=8, frameon=False)
        remove_spines(ax)

    plt.tight_layout(w_pad=2.0)
    fig.savefig(FIGURES_DIR / "model_comparison.pdf")
    plt.close()
    print("  model_comparison.pdf")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")

    fig_network_comparison()
    fig_fiedler_trajectory()
    fig_attention()
    fig_roc_curves()
    fig_bli_scatter()
    fig_counterfactual()
    fig_model_comparison()

    print("All figures generated.")


if __name__ == "__main__":
    main()
