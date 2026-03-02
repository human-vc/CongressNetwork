import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, SEED,
    TRAIN_CONGRESSES, TEST_CONGRESSES, VAL_CONGRESS,
)

import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


def load_congress_arrays(congress_num):
    path = PROCESSED_DIR / f"congress_{congress_num}.npz"
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"]


def pool_train_data():
    X_list, y_list = [], []
    for c in TRAIN_CONGRESSES:
        X, y = load_congress_arrays(c)
        if X is not None:
            X_list.append(X)
            y_list.append(y)
    return np.concatenate(X_list), np.concatenate(y_list)


def calibrate_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[min(best_idx, len(thresholds) - 1)])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    X_train, y_train = pool_train_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"Training set: {len(X_train)} members, {y_train.sum()} defectors")

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=SEED,
    )
    rf.fit(X_train_scaled, y_train)

    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )
    lr.fit(X_train_scaled, y_train)

    val_X, val_y = load_congress_arrays(VAL_CONGRESS)
    if val_X is not None:
        val_X_scaled = scaler.transform(val_X)
        rf_val_probs = rf.predict_proba(val_X_scaled)[:, 1]
        lr_val_probs = lr.predict_proba(val_X_scaled)[:, 1]
        rf_threshold = calibrate_threshold(val_y, rf_val_probs)
        lr_threshold = calibrate_threshold(val_y, lr_val_probs)
    else:
        rf_threshold = 0.5
        lr_threshold = 0.5

    results = {"rf": {}, "lr": {}}

    for c in TEST_CONGRESSES:
        X_test, y_test = load_congress_arrays(c)
        if X_test is None:
            continue
        X_test_scaled = scaler.transform(X_test)

        rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

        rf_preds = (rf_probs >= rf_threshold).astype(int)
        lr_preds = (lr_probs >= lr_threshold).astype(int)

        results["rf"][str(c)] = {
            "auc": float(roc_auc_score(y_test, rf_probs)),
            "f1": float(f1_score(y_test, rf_preds)),
            "threshold": rf_threshold,
            "probabilities": rf_probs.tolist(),
        }

        results["lr"][str(c)] = {
            "auc": float(roc_auc_score(y_test, lr_probs)),
            "f1": float(f1_score(y_test, lr_preds)),
            "threshold": lr_threshold,
            "probabilities": lr_probs.tolist(),
        }

        print(f"Congress {c}: RF AUC={results['rf'][str(c)]['auc']:.3f} F1={results['rf'][str(c)]['f1']:.3f} | "
              f"LR AUC={results['lr'][str(c)]['auc']:.3f} F1={results['lr'][str(c)]['f1']:.3f}")

    results["rf_threshold"] = rf_threshold
    results["lr_threshold"] = lr_threshold
    results["feature_importance"] = dict(zip(
        ["nom1", "nom2", "party", "participation", "yea_rate", "mean_agreement", "cross_party", "within_party"],
        rf.feature_importances_.tolist(),
    ))

    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Baseline training complete.")


if __name__ == "__main__":
    main()
