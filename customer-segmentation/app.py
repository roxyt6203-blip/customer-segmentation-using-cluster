"""
app.py - Streamlit demo for Customer Segmentation (RFM + K-Means)

Tab 1  Full pipeline   : upload transactions -> cleaning -> RFM -> scaling ->
                         elbow/silhouette -> K-Means segments -> profile ->
                         (optional) classifiers + confusion matrix
Tab 2  New Data Predict: upload ANY supported file (CSV, XLSX, PDF, DOCX) and
                         see it cleaned, RFM-built, and predicted with the
                         saved trained model - full results shown
Tab 3  Predict one     : quick single-customer prediction from R, F, M
Tab 4  Saved results   : results of the offline training run (train.py)

Supported upload formats: CSV, TSV, TXT, XLSX, XLS, PDF (table-based), DOCX

Run:  streamlit run app.py     (run `python train.py` once first)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler

from pipeline import (
    FEATURES,
    RANDOM_STATE,
    build_rfm,
    compare_classifiers_cv,
    load_uploaded_file,
    name_segments,
    select_k,
    standardize_transactions,
)

BASE       = Path(__file__).parent
MODEL_PATH = BASE / "models" / "segmentation_model.joblib"
RESULT_DIR = BASE / "results"
SAMPLE_DATA = BASE / "data" / "cleaned_retail_transactions.csv"
ONLINE_RETAIL = BASE / "data" / "online_retail.xlsx"

ACCEPTED_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "pdf", "docx"]

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
)

ACTIONS = {
    "High-Value Loyal": (
        "Top priority. Reward with loyalty perks, early access and premium offers. "
        "Protect this group from churn."
    ),
    "Active Customers": (
        "Healthy base. Nurture with cross-sell/up-sell offers to move them "
        "towards the High-Value group."
    ),
    "At-Risk / Dormant": (
        "Have not purchased for a long time. Run win-back campaigns: "
        "personalised discounts, reminders, feedback survey."
    ),
}

SEG_COLORS = {
    "High-Value Loyal":  "#2ecc71",
    "Active Customers":  "#3498db",
    "At-Risk / Dormant": "#e74c3c",
}


def action_for(segment: str) -> str:
    if "Loyal"   in segment: return ACTIONS["High-Value Loyal"]
    if "Risk"    in segment or "Dormant" in segment: return ACTIONS["At-Risk / Dormant"]
    if "Active"  in segment: return ACTIONS["Active Customers"]
    return "Review this segment's RFM profile and design a targeted campaign."


def seg_color(segment: str) -> str:
    for k, v in SEG_COLORS.items():
        if k in segment:
            return v
    return "#888"


# ---------------------------------------------------------------- cached load
@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_saved_rfm():
    return pd.read_csv(RESULT_DIR / "rfm_with_segments.csv")


@st.cache_data
def run_classifier_comparison(X: np.ndarray, y: np.ndarray):
    return compare_classifiers_cv(X, y)


# ---------------------------------------------------------------- plot helpers
def plot_k_selection(k_table: pd.DataFrame, k_used: int):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(k_table["K"], k_table["Inertia"],    marker="o")
    ax[0].axvline(k_used, ls="--", color="gray")
    ax[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
    ax[1].plot(k_table["K"], k_table["Silhouette"], marker="o", color="green")
    ax[1].axvline(k_used, ls="--", color="gray")
    ax[1].set(title="Silhouette score", xlabel="K", ylabel="Score")
    for a in ax:
        a.set_xticks(k_table["K"])
    fig.tight_layout()
    return fig


def plot_segments(rfm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seg, g in rfm.groupby("Segment"):
        ax.scatter(g["Recency"], g["Monetary"], label=seg,
                   color=seg_color(seg), s=55, alpha=0.85, edgecolors="k", linewidths=0.3)
    ax.set(xlabel="Recency (days since last purchase)",
           ylabel="Monetary (total spend)",
           title="Customer segments (K-Means)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusions(y, preds: dict, label_names):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.3))
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, preds.items()):
        ConfusionMatrixDisplay(
            confusion_matrix(y, p, labels=range(len(label_names))),
            display_labels=label_names,
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    return fig


def predict_saved(art, df_rfm: pd.DataFrame) -> pd.DataFrame:
    X    = art["scaler"].transform(df_rfm[art["features"]])
    clf  = art["classifier"]
    ids  = clf.predict(X)
    out  = df_rfm.copy()
    out["Segment"] = [art["segment_names"][int(i)] for i in ids]
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        out["Confidence_%"] = (proba.max(axis=1) * 100).round(1)
    return out


def segment_bar_chart(rfm: pd.DataFrame):
    counts = rfm["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(
        counts["Segment"], counts["Count"],
        color=[seg_color(s) for s in counts["Segment"]],
        edgecolor="white",
    )
    ax.bar_label(bars, padding=3)
    ax.set(title="Customer count per segment", xlabel="Segment", ylabel="Count")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- guard
if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train.py` first.")
    st.stop()

art       = load_artifacts()
saved_rfm = load_saved_rfm()

st.title("📊 Customer Segmentation using RFM + Clustering")
st.caption(
    "Upload retail transactions and see the full pipeline: "
    "cleaning → RFM → K-selection → K-Means segments → classifier comparison."
)

tab_full, tab_new, tab_one, tab_saved = st.tabs(
    ["Full pipeline (upload data)",
     "New Data Prediction",
     "Predict one customer",
     "Saved training results"]
)


# ======================================================================= TAB 1
def render_full_pipeline():
    st.markdown(
        "**Supported formats:** CSV, TSV, TXT, XLSX/XLS, PDF (table-based), DOCX  \n"
        "**Required columns:** a customer id, a date, and an amount "
        "(or Quantity + UnitPrice — UCI Online Retail format auto-detected)."
    )
    c_up, c_sample = st.columns([3, 1])
    uploaded   = c_up.file_uploader(
        "Upload transactions file",
        type=ACCEPTED_TYPES,
    )
    use_sample = c_sample.checkbox(
        "Use sample data",
        value=False,
        help="Cleaned dataset shipped with the project (100 customers, Jan-Jun 2025)",
    )

    raw = None
    try:
        if uploaded is not None:
            raw = load_uploaded_file(uploaded)
        elif use_sample and SAMPLE_DATA.exists():
            raw = pd.read_csv(SAMPLE_DATA)
    except Exception as e:
        st.error(f"Could not read the file: {e}")

    if raw is None:
        st.info("Upload a file or tick 'Use sample data' to start.")
        return

    try:
        df, notes = standardize_transactions(raw)
    except ValueError as e:
        st.error(str(e))
        return

    # ---- 1. Data understanding & cleaning
    st.header("1. Data understanding and cleaning")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw rows",           f"{len(raw):,}")
    m2.metric("Clean transactions", f"{len(df):,}")
    m3.metric("Customers",          f"{df['Customer_ID'].nunique():,}")
    m4.metric("Date range",
              f"{df['Date'].min():%d %b %Y} - {df['Date'].max():%d %b %Y}")
    st.write("**Cleaning steps applied:**")
    for n in notes:
        st.write(f"- {n}")
    st.dataframe(df.head(20))

    # ---- 2. RFM
    rfm = build_rfm(df)
    n_cust = len(rfm)
    if n_cust < 10:
        st.error(f"Only {n_cust} customers found. Clustering needs at least 10.")
        return
    if n_cust < 30:
        st.warning(f"Only {n_cust} customers — clustering results may be unstable.")

    st.header("2. RFM feature engineering")
    snap = df["Date"].max() + pd.Timedelta(days=1)
    st.caption(
        f"Recency = days from snapshot ({snap:%d %b %Y}) to last purchase; "
        "Frequency = transactions; Monetary = total spend."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Recency (days)",  f"{rfm['Recency'].mean():.1f}")
    c2.metric("Avg Frequency",       f"{rfm['Frequency'].mean():.1f}")
    c3.metric("Avg Monetary",        f"{rfm['Monetary'].mean():,.0f}")
    st.dataframe(rfm)

    # ---- 3. K selection
    st.header("3. Scaling and choosing K")
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[FEATURES])
    k_table, best_k = select_k(X)
    k_max  = int(k_table["K"].max())
    k_used = st.number_input(
        f"Number of clusters K (silhouette recommends K={best_k})",
        min_value=2, max_value=k_max, value=best_k, step=1,
    )
    st.pyplot(plot_k_selection(k_table, int(k_used)))
    st.dataframe(k_table.round(4))

    # ---- 4. K-Means segments
    st.header("4. K-Means customer segments")
    km = KMeans(int(k_used), random_state=RANDOM_STATE, n_init=10).fit(X)
    rfm["Cluster"]  = km.labels_
    seg_names       = name_segments(rfm)
    rfm["Segment"]  = rfm["Cluster"].map(seg_names)
    profile = rfm.groupby("Segment")[FEATURES].mean().round(1)
    profile["Customers"] = rfm["Segment"].value_counts()
    st.subheader("Segment profile (average RFM)")
    st.dataframe(profile)
    cA, cB = st.columns(2)
    cA.pyplot(plot_segments(rfm))
    cB.pyplot(segment_bar_chart(rfm))
    st.subheader("Business actions per segment")
    for seg in profile.index:
        n = int(profile.loc[seg, "Customers"])
        st.markdown(f"**{seg}** ({n} customers): {action_for(seg)}")
    st.download_button(
        "Download customers with segments (CSV)",
        rfm.to_csv(index=False), "customer_segments.csv", "text/csv",
    )

    # ---- 5. Classifier comparison (optional)
    st.header("5. Model comparison and confusion matrix (optional)")
    st.caption(
        "Trains 4 classifiers to predict the K-Means segment from RFM, "
        "with GridSearchCV tuning and stratified CV."
    )
    if n_cust < 30:
        st.info("Needs at least 30 customers.")
    elif st.checkbox("Run classifier comparison"):
        try:
            with st.spinner("Tuning classifiers... (may take a minute)"):
                y = rfm["Cluster"].to_numpy()
                cmp_table, preds, n_splits = run_classifier_comparison(X, y)
            label_names = [seg_names[i] for i in range(int(k_used))]
            st.write(
                f"Stratified {n_splits}-fold CV on all {n_cust} customers."
            )
            st.dataframe(cmp_table)
            st.pyplot(plot_confusions(y, preds, label_names))
            best_row = cmp_table.sort_values(
                "CV acc (after tuning)", ascending=False
            ).iloc[0]
            st.success(
                f"Best model: **{best_row['Model']}** "
                f"(tuned CV acc = {best_row['CV acc (after tuning)']:.3f})"
            )
            st.caption(
                "Labels come from K-Means on the same RFM features — high accuracy "
                "shows the segments are well separated, not accuracy vs. external truth."
            )
        except ValueError as e:
            st.error(str(e))


with tab_full:
    render_full_pipeline()


# ======================================================================= TAB 2
with tab_new:
    st.header("New Data Prediction")
    st.markdown(
        "Upload a new dataset in **any supported format** (CSV, XLSX, PDF, DOCX).  \n"
        "The app will clean it, compute RFM, and use the **saved trained model** "
        "to predict the customer segment for each customer.  \n"
        "You can also click **Use Online Retail sample** to try with the UCI dataset."
    )

    col_up, col_sample = st.columns([3, 1])
    new_file   = col_up.file_uploader(
        "Upload new data file",
        type=ACCEPTED_TYPES,
        key="new_data_uploader",
    )
    use_online = col_sample.checkbox(
        "Use Online Retail sample",
        value=False,
        help="UCI Online Retail dataset (online_retail.xlsx) shipped with the project",
    )

    new_raw = None
    try:
        if new_file is not None:
            new_raw = load_uploaded_file(new_file)
        elif use_online and ONLINE_RETAIL.exists():
            with st.spinner("Loading Online Retail dataset (541k rows)..."):
                new_raw = pd.read_excel(str(ONLINE_RETAIL), engine="openpyxl")
        elif use_online:
            st.warning(
                "online_retail.xlsx not found in data/. "
                "Please copy it there or upload it directly."
            )
    except Exception as e:
        st.error(f"Could not read file: {e}")

    if new_raw is None:
        st.info("Upload a file or tick 'Use Online Retail sample' to start.")
        st.stop()

    try:
        with st.spinner("Cleaning data..."):
            new_df, new_notes = standardize_transactions(new_raw)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # ---- Cleaning summary
    st.subheader("1. Cleaning summary")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Raw rows",           f"{len(new_raw):,}")
    r2.metric("Clean transactions", f"{len(new_df):,}")
    r3.metric("Customers",          f"{new_df['Customer_ID'].nunique():,}")
    r4.metric("Date range",
              f"{new_df['Date'].min():%d %b %Y} - {new_df['Date'].max():%d %b %Y}")
    for n in new_notes:
        st.write(f"- {n}")

    # ---- Build RFM
    st.subheader("2. RFM features")
    with st.spinner("Building RFM..."):
        new_rfm = build_rfm(new_df)
    n_new = len(new_rfm)
    if n_new < 3:
        st.error(
            f"Only {n_new} customer(s) found after cleaning. "
            "Need at least 3 for meaningful prediction."
        )
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers",        n_new)
    c2.metric("Avg Recency",      f"{new_rfm['Recency'].mean():.1f} days")
    c3.metric("Avg Monetary",     f"{new_rfm['Monetary'].mean():,.0f}")
    st.dataframe(new_rfm.head(20))

    # ---- Predict with saved model
    st.subheader("3. Segment predictions (saved model)")
    st.caption(
        f"Model: **{art['classifier_name']}**, trained on the project dataset (100 customers).  \n"
        "Segments are assigned using the trained K-Means scaler + Logistic Regression classifier."
    )

    with st.spinner("Predicting segments..."):
        new_rfm_pred = predict_saved(art, new_rfm)

    # Count per segment
    seg_counts = new_rfm_pred["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    seg_counts["Percentage"] = (seg_counts["Count"] / n_new * 100).round(1)

    st.dataframe(seg_counts)

    cA, cB = st.columns(2)
    with cA:
        st.pyplot(segment_bar_chart(new_rfm_pred))
    with cB:
        # Scatter: Recency vs Monetary
        fig, ax = plt.subplots(figsize=(6, 4))
        for seg, g in new_rfm_pred.groupby("Segment"):
            ax.scatter(
                g["Recency"], g["Monetary"],
                label=seg, color=seg_color(seg),
                s=20, alpha=0.6, edgecolors="none",
            )
        ax.set(
            xlabel="Recency (days)", ylabel="Monetary",
            title="RFM segments (new data)",
        )
        ax.legend(markerscale=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Segment actions
    st.subheader("4. Business recommendations")
    for _, row in seg_counts.iterrows():
        seg = row["Segment"]
        st.markdown(
            f"**{seg}** — {row['Count']} customers ({row['Percentage']}%):  \n"
            f"{action_for(seg)}"
        )

    # Show full table
    st.subheader("5. Full prediction table")
    cols_show = ["Customer_ID", "Recency", "Frequency", "Monetary", "Segment"]
    if "Confidence_%" in new_rfm_pred.columns:
        cols_show.append("Confidence_%")
    st.dataframe(new_rfm_pred[cols_show].sort_values("Segment"))

    # Download
    csv_out = new_rfm_pred.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_out,
        file_name="new_data_predictions.csv",
        mime="text/csv",
    )


# ======================================================================= TAB 3
with tab_one:
    st.subheader("Enter customer RFM values")
    st.caption(
        f"Uses the saved model: **{art['classifier_name']}** "
        f"trained on the project dataset."
    )
    cols  = ["Recency", "Frequency", "Monetary"]
    lo    = saved_rfm[cols].min()
    hi    = saved_rfm[cols].max()
    med   = saved_rfm[cols].median()

    c1, c2, c3 = st.columns(3)
    r = c1.number_input("Recency (days since last purchase)", 0, 1000,
                        int(med["Recency"]),   1)
    f = c2.number_input("Frequency (number of transactions)", 0, 5000,
                        int(med["Frequency"]), 1)
    m = c3.number_input("Monetary (total amount spent)", 0.0, 1e11,
                        float(med["Monetary"]), 1_000_000.0, format="%.0f")

    if st.button("Predict segment", type="primary"):
        res = predict_saved(
            art, pd.DataFrame({"Recency": [r], "Frequency": [f], "Monetary": [m]})
        )
        seg = res.loc[0, "Segment"]
        color = seg_color(seg)
        st.markdown(
            f"<div style='background:{color};padding:14px;border-radius:8px;"
            f"color:white;font-size:1.3rem;font-weight:bold;text-align:center'>"
            f"Predicted Segment: {seg}</div>",
            unsafe_allow_html=True,
        )
        if "Confidence_%" in res.columns:
            st.metric("Model confidence", f"{res.loc[0, 'Confidence_%']}%")
        st.info(action_for(seg))

        outside = [n for n, v in zip(cols, [r, f, m]) if v < lo[n] * 0.5 or v > hi[n] * 1.5]
        if outside:
            st.warning(
                "Input is far outside the training data range for: "
                + ", ".join(outside)
                + ". Prediction may be unreliable."
            )

    with st.expander("Training data range (for reference)"):
        st.dataframe(saved_rfm[cols].describe().loc[["min", "50%", "max"]])


# ======================================================================= TAB 4
with tab_saved:
    st.subheader("Segment profile (average RFM per segment)")
    st.dataframe(pd.read_csv(RESULT_DIR / "segment_profile.csv"))

    st.subheader("Clustering algorithms compared")
    st.dataframe(pd.read_csv(RESULT_DIR / "clustering_comparison.csv"))

    st.subheader("Classifiers compared (before vs after hyperparameter tuning)")
    st.dataframe(pd.read_csv(RESULT_DIR / "model_comparison.csv"))

    cA, cB = st.columns(2)
    cA.image(str(RESULT_DIR / "confusion_matrix_best.png"),
             caption="Confusion matrix - best model (20 test customers)")
    cB.image(str(RESULT_DIR / "segments_scatter.png"),
             caption="Customer segments (training data)")
    st.image(
        str(RESULT_DIR / "confusion_matrices_all_models.png"),
        caption="All models - 5-fold CV predictions on all 100 customers",
    )
    st.caption(
        "Note: segment labels come from K-Means on the same RFM features, so the "
        "classifiers learn to reproduce the clustering boundaries. High accuracy "
        "shows the segments are well-defined and learnable; it is not accuracy "
        "against external ground truth."
    )
