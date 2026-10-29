"""
train.py - Offline training for the Customer Segmentation project.

Steps
1. Clean transactions and build customer-level RFM features
2. Scale features (StandardScaler)
3. Compare clustering algorithms (KMeans, Agglomerative, GMM, DBSCAN)
4. Use K-Means (K=3) to create segment labels
5. Compare classifiers that predict the segment from RFM, with GridSearchCV tuning
6. Final evaluation on a held-out test set: accuracy, macro-F1, confusion matrix
7. Save model artifacts + result tables + plots (used by app.py)

Run:  python train.py
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from pipeline import (
    FEATURES,
    RANDOM_STATE,
    build_rfm,
    compare_clustering,
    get_classifiers,
    name_segments,
    standardize_transactions,
)

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "cleaned_retail_transactions.csv"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2025-07-01")  # day after last transaction (30 Jun 2025)
K = 3


def main():
    # 1. cleaning + RFM
    raw = pd.read_csv(DATA_FILE)
    df, notes = standardize_transactions(raw)
    print("Cleaning notes:", *notes, sep="\n  - ")
    rfm = build_rfm(df, SNAPSHOT_DATE)
    rfm.to_csv(BASE / "data" / "rfm_features.csv", index=False)
    print(f"\nRFM table: {rfm.shape[0]} customers")

    # 2. scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[FEATURES])

    # 3. clustering comparison
    clus_cmp = compare_clustering(X, K)
    clus_cmp.to_csv(RESULT_DIR / "clustering_comparison.csv", index=False)
    print("\n=== Clustering algorithm comparison ===")
    print(clus_cmp.to_string(index=False))

    # 4. final segments from K-Means
    kmeans = KMeans(K, random_state=RANDOM_STATE, n_init=10).fit(X)
    rfm["Cluster"] = kmeans.labels_
    seg_names = name_segments(rfm)
    rfm["Segment"] = rfm["Cluster"].map(seg_names)
    rfm.to_csv(RESULT_DIR / "rfm_with_segments.csv", index=False)
    profile = rfm.groupby("Segment")[FEATURES].mean().round(1)
    profile["Customers"] = rfm["Segment"].value_counts()
    profile.to_csv(RESULT_DIR / "segment_profile.csv")
    print("\n=== Segment profile ===")
    print(profile)

    # 5. train/test split + CV comparison + tuning
    y = rfm["Cluster"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, tuned = [], {}
    for name, (model, grid) in get_classifiers().items():
        base_cv = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        search = GridSearchCV(model, grid, cv=cv, scoring="accuracy").fit(X_train, y_train)
        pred = search.predict(X_test)
        tuned[name] = search
        rows.append(
            {
                "Model": name,
                "CV acc (before tuning)": round(base_cv.mean(), 4),
                "CV acc (after tuning)": round(search.best_score_, 4),
                "Best params": search.best_params_,
                "Test accuracy": round(accuracy_score(y_test, pred), 4),
                "Test macro-F1": round(f1_score(y_test, pred, average="macro"), 4),
            }
        )
    model_cmp = pd.DataFrame(rows)
    model_cmp.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    print("\n=== Classifier comparison (before vs after tuning) ===")
    print(model_cmp.drop(columns=["Best params"]).to_string(index=False))

    # 6. best model by tuned CV accuracy (tie-break: test macro-F1)
    best_name = model_cmp.sort_values(
        ["CV acc (after tuning)", "Test macro-F1"], ascending=False
    ).iloc[0]["Model"]
    best = tuned[best_name]
    print(f"\nBest model: {best_name}  params={best.best_params_}")

    label_names = [seg_names[i] for i in sorted(seg_names)]
    pred = best.predict(X_test)
    report = classification_report(
        y_test, pred, target_names=label_names, zero_division=0, output_dict=True
    )
    pd.DataFrame(report).T.round(3).to_csv(RESULT_DIR / "classification_report.csv")
    print("\n=== Classification report (best model, test set) ===")
    print(classification_report(y_test, pred, target_names=label_names, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=label_names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {best_name} (test set)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(RESULT_DIR / "confusion_matrix_best.png", dpi=130)
    plt.close(fig)

    # confusion matrix of every model: 5-fold CV predictions on all customers
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (name, search) in zip(axes, tuned.items()):
        cv_pred = cross_val_predict(search.best_estimator_, X, y, cv=cv)
        ConfusionMatrixDisplay(confusion_matrix(y, cv_pred), display_labels=label_names).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"{name}\n5-fold CV accuracy = {accuracy_score(y, cv_pred):.3f}")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULT_DIR / "confusion_matrices_all_models.png", dpi=120)
    plt.close(fig)

    # segment scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg, g in rfm.groupby("Segment"):
        ax.scatter(g["Recency"], g["Monetary"] / 1e8, label=seg, s=45, alpha=0.8)
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Monetary (x 10^8)")
    ax.set_title(f"Customer segments (K-Means, K={K})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULT_DIR / "segments_scatter.png", dpi=130)
    plt.close(fig)

    # 7. artifacts for the web app
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "classifier": best.best_estimator_,
            "classifier_name": best_name,
            "segment_names": seg_names,
            "features": FEATURES,
        },
        MODEL_DIR / "segmentation_model.joblib",
    )
    print("\nSaved: models/segmentation_model.joblib and results/*")


if __name__ == "__main__":
    main()
