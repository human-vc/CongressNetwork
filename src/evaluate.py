import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, SEED, CONGRESSES,
    TRAIN_CONGRESSES, VAL_CONGRESS, TEST_CONGRESSES, ANALYSIS_CONGRESS,
    HIDDEN_DIM, N_HEADS, N_TEMPORAL_HEADS, DROPOUT,
    MCCARTHY_HOLDOUTS,
)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from model import CongressGAT, CongressGCN, load_congress_data, load_fiedler_targets, set_seed


def calibrate_threshold(model, val_data, device):
    model.eval()
    with torch.no_grad():
        h = model.encode(val_data)
        probs = model.predict_defection(h, val_data.x).cpu().numpy()
    y_true = val_data.y.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[min(best_idx, len(thresholds) - 1)])


def evaluate_model(model, congress_nums, threshold, device):
    model.eval()
    results = {}
    for c in congress_nums:
        data = load_congress_data(c)
        if data is None:
            continue
        data = data.to(device)
        with torch.no_grad():
            h = model.encode(data)
            probs = model.predict_defection(h, data.x).cpu().numpy()
        y_true = data.y.cpu().numpy()
        preds = (probs >= threshold).astype(int)
        try:
            auc = float(roc_auc_score(y_true, probs))
        except ValueError:
            auc = 0.0
        f1 = float(f1_score(y_true, preds, zero_division=0))
        fpr, tpr, _ = roc_curve(y_true, probs)
        results[str(c)] = {
            "auc": auc,
            "f1": f1,
            "threshold": threshold,
            "n_flagged": int(preds.sum()),
            "n_actual": int(y_true.sum()),
            "probabilities": probs.tolist(),
            "predictions": preds.tolist(),
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
        }
    return results


def extract_attention(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        edge_index, attn_weights = model.get_attention_weights(data)

    edge_index = edge_index.cpu().numpy()
    attn_weights = attn_weights.cpu().numpy()
    party = data.party_codes.cpu().numpy()

    same_party_attn = []
    cross_party_attn = []

    for k in range(edge_index.shape[1]):
        src, dst = edge_index[0, k], edge_index[1, k]
        w = float(attn_weights[k].mean())
        if party[src] == party[dst]:
            same_party_attn.append(w)
        else:
            cross_party_attn.append(w)

    return {
        "same_party_mean": float(np.mean(same_party_attn)) if same_party_attn else 0.0,
        "cross_party_mean": float(np.mean(cross_party_attn)) if cross_party_attn else 0.0,
        "same_party_std": float(np.std(same_party_attn)) if same_party_attn else 0.0,
        "cross_party_std": float(np.std(cross_party_attn)) if cross_party_attn else 0.0,
        "all_weights": (same_party_attn + cross_party_attn),
        "same_party_weights": same_party_attn,
        "cross_party_weights": cross_party_attn,
    }


def compute_overlap(gat_preds, baseline_preds):
    gat_set = set(np.where(np.array(gat_preds) == 1)[0])
    baseline_set = set(np.where(np.array(baseline_preds) == 1)[0])
    if len(gat_set) == 0 and len(baseline_set) == 0:
        return {"jaccard": 1.0, "gat_unique_pct": 0.0, "n_gat": 0, "n_baseline": 0, "n_shared": 0}
    jaccard = len(gat_set & baseline_set) / len(gat_set | baseline_set) if len(gat_set | baseline_set) > 0 else 0.0
    gat_only = gat_set - baseline_set
    gat_unique_pct = len(gat_only) / len(gat_set) if len(gat_set) > 0 else 0.0
    return {
        "jaccard": jaccard,
        "gat_unique_pct": gat_unique_pct,
        "n_gat": len(gat_set),
        "n_baseline": len(baseline_set),
        "n_shared": len(gat_set & baseline_set),
    }


def mccarthy_analysis(model, threshold, device):
    data = load_congress_data(ANALYSIS_CONGRESS)
    if data is None:
        return None

    npz = np.load(PROCESSED_DIR / f"congress_{ANALYSIS_CONGRESS}.npz", allow_pickle=True)
    member_names = npz["member_names"]
    party_codes = npz["party_codes"]
    nominate_dim1 = npz["nominate_dim1"]
    state_abbrev = npz["state_abbrev"]
    member_ids = npz["member_ids"]

    data = data.to(device)
    model.eval()
    with torch.no_grad():
        h = model.encode(data)
        probs = model.predict_defection(h, data.x).cpu().numpy()

    rep_mask = party_codes == 200
    rep_indices = np.where(rep_mask)[0]
    rep_probs = probs[rep_indices]
    rep_nom = nominate_dim1[rep_mask]
    rep_median = np.median(rep_nom)
    rep_std = np.std(rep_nom)
    ranked = np.argsort(-rep_probs)

    top_rebels = []
    for rank, idx in enumerate(ranked[:20]):
        orig_idx = rep_indices[idx]
        top_rebels.append({
            "rank": rank + 1,
            "name": str(member_names[orig_idx]),
            "state": str(state_abbrev[orig_idx]),
            "icpsr": int(member_ids[orig_idx]),
            "defection_prob": float(probs[orig_idx]),
            "flagged": bool(probs[orig_idx] >= threshold),
            "nominate_dim1": float(nominate_dim1[orig_idx]),
        })

    holdout_results = []
    holdouts_caught = 0
    holdouts_ideological_outlier = 0
    for i in range(len(member_names)):
        surname = str(member_names[i]).split(",")[0].strip().upper()
        if surname in MCCARTHY_HOLDOUTS and party_codes[i] == 200:
            caught = bool(probs[i] >= threshold)
            dist_from_median = abs(nominate_dim1[i] - rep_median)
            is_outlier = dist_from_median > rep_std
            if caught:
                holdouts_caught += 1
            if caught and is_outlier:
                holdouts_ideological_outlier += 1
            holdout_results.append({
                "name": str(member_names[i]),
                "surname": surname,
                "defection_prob": float(probs[i]),
                "caught": caught,
                "nominate_dim1": float(nominate_dim1[i]),
                "dist_from_median": float(dist_from_median),
                "ideological_outlier": bool(is_outlier),
            })

    return {
        "congress": ANALYSIS_CONGRESS,
        "threshold": threshold,
        "n_republicans": int(rep_mask.sum()),
        "n_flagged_republicans": int((probs[rep_mask] >= threshold).sum()),
        "top_rebels": top_rebels,
        "holdout_matching": {
            "n_holdouts_total": len(MCCARTHY_HOLDOUTS),
            "n_holdouts_found": len(holdout_results),
            "n_holdouts_caught": holdouts_caught,
            "n_caught_ideological_outlier": holdouts_ideological_outlier,
            "details": holdout_results,
        },
    }


def evaluate_polarization(model, device):
    fiedler_targets = load_fiedler_targets()
    if not fiedler_targets:
        return None

    all_congresses = sorted(c for c in CONGRESSES if load_congress_data(c) is not None)
    if len(all_congresses) < 2:
        return None

    model.eval()
    with torch.no_grad():
        graph_embeddings = []
        for c in all_congresses:
            data = load_congress_data(c).to(device)
            h = model.encode(data)
            graph_embeddings.append(h.mean(dim=0))

        temporal_out = model.temporal_forward(graph_embeddings)

        predictions = {}
        for t_idx, c in enumerate(all_congresses):
            pred = model.predict_polarization(temporal_out[t_idx]).item()
            predictions[c] = pred

    test_results = {}
    for c in TEST_CONGRESSES:
        if c in predictions and c in fiedler_targets:
            test_results[str(c)] = {
                "predicted": predictions[c],
                "actual": fiedler_targets[c],
                "error": abs(predictions[c] - fiedler_targets[c]),
            }

    test_mse = np.mean([r["error"] ** 2 for r in test_results.values()]) if test_results else 0.0
    test_mae = np.mean([r["error"] for r in test_results.values()]) if test_results else 0.0

    drift_errors = []
    for c in TEST_CONGRESSES:
        prev_c = c - 1
        if prev_c in fiedler_targets and c in fiedler_targets:
            drift_errors.append(abs(fiedler_targets[prev_c] - fiedler_targets[c]))
    drift_mse = np.mean([e ** 2 for e in drift_errors]) if drift_errors else 0.0
    drift_mae = np.mean(drift_errors) if drift_errors else 0.0

    projection_119 = None
    last_c = max(all_congresses)
    if last_c == ANALYSIS_CONGRESS:
        with torch.no_grad():
            graph_embeddings_ext = list(graph_embeddings)
            graph_embeddings_ext.append(graph_embeddings[-1])
            temporal_out_ext = model.temporal_forward(graph_embeddings_ext)
            projection_119 = model.predict_polarization(temporal_out_ext[-1]).item()

    return {
        "test_congresses": test_results,
        "model_mse": float(test_mse),
        "model_mae": float(test_mae),
        "drift_mse": float(drift_mse),
        "drift_mae": float(drift_mae),
        "projection_119": projection_119,
        "all_predictions": {str(c): float(v) for c, v in predictions.items()},
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gat = CongressGAT()
    gat.load_state_dict(torch.load(RESULTS_DIR / "gat_model.pt", map_location=device, weights_only=True))
    gat = gat.to(device)

    gcn = CongressGCN()
    gcn.load_state_dict(torch.load(RESULTS_DIR / "gcn_model.pt", map_location=device, weights_only=True))
    gcn = gcn.to(device)

    val_data = load_congress_data(VAL_CONGRESS)
    if val_data is None:
        print("No validation data found")
        return
    val_data = val_data.to(device)

    print("Calibrating thresholds on congress 114...")
    gat_threshold = calibrate_threshold(gat, val_data, device)
    gcn_threshold = calibrate_threshold(gcn, val_data, device)
    print(f"  GAT threshold: {gat_threshold:.3f}")
    print(f"  GCN threshold: {gcn_threshold:.3f}")

    print("\nEvaluating on test congresses...")
    gat_results = evaluate_model(gat, TEST_CONGRESSES, gat_threshold, device)
    gcn_results = evaluate_model(gcn, TEST_CONGRESSES, gcn_threshold, device)

    for c in TEST_CONGRESSES:
        cs = str(c)
        if cs in gat_results and cs in gcn_results:
            print(f"  Congress {c}: GAT AUC={gat_results[cs]['auc']:.3f} F1={gat_results[cs]['f1']:.3f} | "
                  f"GCN AUC={gcn_results[cs]['auc']:.3f} F1={gcn_results[cs]['f1']:.3f}")

    baseline_path = RESULTS_DIR / "baseline_results.json"
    overlap_results = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_results = json.load(f)

        for c in TEST_CONGRESSES:
            cs = str(c)
            if cs in gat_results and cs in baseline_results.get("rf", {}):
                rf_probs = np.array(baseline_results["rf"][cs]["probabilities"])
                rf_threshold = baseline_results["rf_threshold"]
                rf_preds = (rf_probs >= rf_threshold).astype(int).tolist()
                overlap = compute_overlap(gat_results[cs]["predictions"], rf_preds)
                overlap_results[cs] = overlap
                print(f"  Congress {c}: GAT-RF Jaccard={overlap['jaccard']:.3f}, "
                      f"GAT-unique={overlap['gat_unique_pct']:.1%} ({overlap['n_gat']-overlap['n_shared']}/{overlap['n_gat']})")

    print("\nExtracting attention weights...")
    attention_results = {}
    for c in TEST_CONGRESSES + [ANALYSIS_CONGRESS]:
        data = load_congress_data(c)
        if data is None:
            continue
        attn = extract_attention(gat, data, device)
        attention_results[str(c)] = {
            "same_party_mean": attn["same_party_mean"],
            "cross_party_mean": attn["cross_party_mean"],
            "same_party_std": attn["same_party_std"],
            "cross_party_std": attn["cross_party_std"],
        }
        print(f"  Congress {c}: same={attn['same_party_mean']:.4f}, cross={attn['cross_party_mean']:.4f}")

    print("\nMcCarthy 118th analysis...")
    mccarthy = mccarthy_analysis(gat, gat_threshold, device)
    if mccarthy:
        print(f"  Flagged {mccarthy['n_flagged_republicans']}/{mccarthy['n_republicans']} Republicans")
        for r in mccarthy["top_rebels"][:5]:
            flag = "*" if r["flagged"] else ""
            print(f"    {r['rank']}. {r['name']} ({r['state']}) p={r['defection_prob']:.3f} {flag}")
        hm = mccarthy["holdout_matching"]
        print(f"  Holdout matching: {hm['n_holdouts_caught']}/{hm['n_holdouts_found']} caught "
              f"({hm['n_caught_ideological_outlier']} ideological outliers)")
        for h in hm["details"]:
            tag = "CAUGHT" if h["caught"] else "missed"
            print(f"    {h['surname']}: p={h['defection_prob']:.3f} [{tag}]")

    print("\nEvaluating polarization predictions...")
    gat_polarization = evaluate_polarization(gat, device)
    gcn_polarization = evaluate_polarization(gcn, device)
    if gat_polarization:
        print(f"  GAT: MSE={gat_polarization['model_mse']:.6f}, MAE={gat_polarization['model_mae']:.4f}")
        print(f"  Naive drift: MSE={gat_polarization['drift_mse']:.6f}, MAE={gat_polarization['drift_mae']:.4f}")
        for cs, r in gat_polarization["test_congresses"].items():
            print(f"    Congress {cs}: predicted={r['predicted']:.4f}, actual={r['actual']:.4f}")
        if gat_polarization["projection_119"] is not None:
            print(f"  119th projection: {gat_polarization['projection_119']:.4f}")

    eval_output = {
        "gat": gat_results,
        "gcn": gcn_results,
        "gat_threshold": gat_threshold,
        "gcn_threshold": gcn_threshold,
        "attention": attention_results,
        "overlap": overlap_results,
        "mccarthy": mccarthy,
        "polarization": {
            "gat": gat_polarization,
            "gcn": gcn_polarization,
        },
    }

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_output, f, indent=2)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
