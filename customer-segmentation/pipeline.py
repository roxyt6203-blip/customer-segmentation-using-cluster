"""
pipeline.py - reusable functions for the RFM customer-segmentation pipeline.
Used by both train.py (offline training) and app.py (live "upload and run" demo).

Supports file formats: CSV, TSV, TXT, XLSX, XLS, PDF (tables), DOCX
Supports datasets:
  - Project dataset (Customer_ID, Date, Amount)
  - UCI Online Retail (InvoiceNo, CustomerID, InvoiceDate, Quantity, UnitPrice -> Amount computed)
  - Raw block-format Kaggle file
  - Any file with a customer id, a date and an amount column (column aliases handled)
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Recency", "Frequency", "Monetary"]
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
#  Column-name aliases so different datasets work automatically                #
# --------------------------------------------------------------------------- #
_ALIASES = {
    "Customer_ID": [
        "customerid", "customer_id", "customer id", "customer",
        "cust_id", "custid", "customer_no", "client_id",
    ],
    "Date": [
        "invoicedate", "invoice_date", "invoice date",
        "date", "order_date", "order date",
        "transaction_date", "transaction date", "purchase_date",
    ],
    "Amount": [
        "amount", "sales", "total", "revenue",
        "total_amount", "total amount", "sale_amount", "lineamount",
    ],
}

# Online Retail dataset computes Amount = Quantity * UnitPrice
_QTY_COLS   = ["quantity", "qty"]
_PRICE_COLS = ["unitprice", "unit_price", "price", "unit price"]


# --------------------------------------------------------------------------- #
#  File loader (multi-format)                                                  #
# --------------------------------------------------------------------------- #
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load data from various file types:
      CSV / TXT / TSV  -> pandas read_csv
      XLSX / XLS       -> pandas read_excel  (openpyxl)
      PDF              -> pdfplumber table extraction
      DOCX             -> python-docx table extraction
      DOC              -> raises a clear error (not supported)

    Works with both a Streamlit UploadedFile object and a plain file path (str / Path).
    """
    # Resolve the file name for extension detection
    if hasattr(uploaded_file, "name"):
        name = uploaded_file.name.lower()
    else:
        name = str(uploaded_file).lower()

    if name.endswith(".csv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file)

    elif name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")

    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    elif name.endswith(".pdf"):
        import pdfplumber  # noqa: PLC0415

        # pdfplumber needs a seekable bytes buffer
        if hasattr(uploaded_file, "read"):
            buf = io.BytesIO(uploaded_file.read())
        else:
            buf = uploaded_file

        all_rows: list[list] = []
        with pdfplumber.open(buf) as pdf:
            for page in pdf.pages:
                for table in (page.extract_tables() or []):
                    for row in table:
                        if any(cell is not None and str(cell).strip() for cell in row):
                            all_rows.append(row)

        if not all_rows:
            raise ValueError(
                "No tabular data could be extracted from the uploaded PDF. "
                "Make sure the PDF contains actual tables (not scanned images)."
            )
        header = [str(c).strip() if c else f"Col_{i}" for i, c in enumerate(all_rows[0])]
        return pd.DataFrame(all_rows[1:], columns=header)

    elif name.endswith(".docx"):
        import docx  # noqa: PLC0415

        if hasattr(uploaded_file, "read"):
            buf = io.BytesIO(uploaded_file.read())
        else:
            buf = uploaded_file

        doc = docx.Document(buf)
        all_rows = []
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    all_rows.append(cells)
        if not all_rows:
            raise ValueError(
                "No tabular data found inside the uploaded Word document. "
                "Make sure the data is inside a table (not just plain text)."
            )
        return pd.DataFrame(all_rows[1:], columns=all_rows[0])

    elif name.endswith(".doc"):
        raise ValueError(
            "Legacy .doc format is not supported. "
            "Please open the file in Word and save as .docx or .xlsx."
        )

    else:
        # Fallback: try CSV
        return pd.read_csv(uploaded_file)


# --------------------------------------------------------------------------- #
#  Transaction cleaner                                                         #
# --------------------------------------------------------------------------- #
def standardize_transactions(df: pd.DataFrame):
    """
    Turn any uploaded transactions table into a clean table with:
      Customer_ID  (str)
      Date         (datetime64)
      Amount       (float, > 0)
    plus any other columns that are present.

    Handles:
      - UCI Online Retail format (computes Amount = Quantity * UnitPrice)
      - Raw block-structured Kaggle file (header row per customer)
      - Already-cleaned project dataset
      - Any file with aliased column names

    Returns (clean_df, notes) where notes is a list of human-readable strings.
    """
    notes: list[str] = []
    df = df.copy()

    lower_map = {c.lower().strip(): c for c in df.columns}

    # ------------------------------------------------------------------
    # 1.  UCI Online Retail: compute Amount = Quantity * UnitPrice
    #     if Amount (or alias) is missing but Quantity and UnitPrice exist
    # ------------------------------------------------------------------
    has_amount = any(
        al in lower_map for al in (_ALIASES["Amount"] + ["Amount"])
    ) or "Amount" in df.columns

    if not has_amount:
        qty_col   = next((lower_map[c] for c in _QTY_COLS   if c in lower_map), None)
        price_col = next((lower_map[c] for c in _PRICE_COLS if c in lower_map), None)
        if qty_col and price_col:
            df["Amount"] = (
                pd.to_numeric(df[qty_col],   errors="coerce").fillna(0)
                * pd.to_numeric(df[price_col], errors="coerce").fillna(0)
            )
            notes.append(
                f"Computed Amount = {qty_col} x {price_col} "
                f"(UCI Online Retail / similar format)."
            )
            lower_map["amount"] = "Amount"   # so alias loop below won't re-map

    # ------------------------------------------------------------------
    # 2.  Raw block format: Customer ID in 'Transaction ID' on header rows
    # ------------------------------------------------------------------
    if "Customer_ID" not in df.columns and {"Transaction ID", "Product ID"}.issubset(df.columns):
        is_header = df["Product ID"].isna()
        if is_header.any():
            df["Customer_ID"] = df["Transaction ID"].where(is_header).ffill()
            if "Date" in df.columns and "Region" not in df.columns:
                df["Region"] = df["Date"].where(is_header).ffill()
            n_h = int(is_header.sum())
            df = df[~is_header].copy()
            notes.append(
                f"Raw block format detected: forward-filled Customer ID/Region "
                f"and removed {n_h} header rows."
            )
            lower_map = {c.lower().strip(): c for c in df.columns}  # refresh

    # ------------------------------------------------------------------
    # 3.  Column-name aliasing
    # ------------------------------------------------------------------
    lower_map = {c.lower().strip(): c for c in df.columns}  # refresh after block-fix
    for target, aliases in _ALIASES.items():
        if target in df.columns:
            continue
        for al in aliases:
            if al in lower_map:
                df = df.rename(columns={lower_map[al]: target})
                notes.append(f"Renamed column '{lower_map[al]}' -> '{target}'.")
                lower_map = {c.lower().strip(): c for c in df.columns}
                break

    missing = [c for c in ("Customer_ID", "Date", "Amount") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            "The file must have a customer id column, a date column, and an amount "
            "(or Quantity + UnitPrice) column."
        )

    # ------------------------------------------------------------------
    # 4.  Type coercion
    # ------------------------------------------------------------------
    if not pd.api.types.is_numeric_dtype(df["Amount"]):
        df["Amount"] = pd.to_numeric(
            df["Amount"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        notes.append("Converted Amount from text to numbers.")

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        s = df["Date"].astype(str)
        if s.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}$").mean() > 0.9:
            df["Date"] = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        notes.append("Parsed Date column to datetime.")

    df["Amount"]      = df["Amount"].astype(float)
    df["Customer_ID"] = df["Customer_ID"].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 5.  Drop bad rows
    # ------------------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=["Customer_ID", "Date", "Amount"])
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} rows with missing customer/date/amount.")

    before = len(df)
    df = df[df["Amount"] > 0]
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} rows with zero or negative amount.")

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} duplicate rows.")

    # Drop CustomerID == "nan" (string) which can come from float NaN -> str
    before = len(df)
    df = df[df["Customer_ID"].str.lower() != "nan"]
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} rows with null customer ID.")

    if not notes:
        notes.append("Data was already clean - no changes needed.")

    return df.reset_index(drop=True), notes


# --------------------------------------------------------------------------- #
#  RFM feature engineering                                                     #
# --------------------------------------------------------------------------- #
def build_rfm(df: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Customer-level Recency / Frequency / Monetary. Snapshot defaults to last date + 1 day."""
    if snapshot is None:
        snapshot = df["Date"].max() + pd.Timedelta(days=1)
    return (
        df.groupby("Customer_ID")
        .agg(
            Recency=("Date",   lambda x: (snapshot - x.max()).days),
            Frequency=("Date", "size"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------- #
#  Clustering helpers                                                          #
# --------------------------------------------------------------------------- #
def select_k(X: np.ndarray, k_min: int = 2, k_max: int = 8):
    """Elbow (inertia) + silhouette for K in [k_min, k_max]. Returns (table, best_k)."""
    k_max = min(k_max, len(X) - 1)
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(k, random_state=RANDOM_STATE, n_init=10).fit(X)
        rows.append(
            {"K": k, "Inertia": round(km.inertia_, 2),
             "Silhouette": round(silhouette_score(X, km.labels_), 4)}
        )
    table = pd.DataFrame(rows)
    return table, int(table.loc[table["Silhouette"].idxmax(), "K"])


def name_segments(rfm_labeled: pd.DataFrame) -> dict:
    """Give each cluster id a business name from its average RFM profile."""
    prof = rfm_labeled.groupby("Cluster")[FEATURES].mean()
    names = {int(prof["Recency"].idxmax()): "At-Risk / Dormant"}
    rest = prof.drop(index=prof["Recency"].idxmax())
    if len(rest) >= 2:
        loyal = rest["Monetary"].idxmax()
        names[int(loyal)] = "High-Value Loyal"
        rest = rest.drop(index=loyal)
    rest = rest.sort_values("Recency")
    for i, c in enumerate(rest.index, start=1):
        names[int(c)] = "Active Customers" if len(rest) == 1 else f"Active Customers {i}"
    return names


def compare_clustering(X: np.ndarray, k: int = 3) -> pd.DataFrame:
    algos = {
        f"K-Means (K={k})":           KMeans(k, random_state=RANDOM_STATE, n_init=10),
        f"Agglomerative (Ward, K={k})": AgglomerativeClustering(n_clusters=k),
        f"Gaussian Mixture (K={k})":  GaussianMixture(k, random_state=RANDOM_STATE),
        "DBSCAN (eps=0.8)":            DBSCAN(eps=0.8, min_samples=5),
    }
    rows = []
    for name, algo in algos.items():
        labels = algo.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        row = {"Algorithm": name, "Clusters": n_clusters,
               "Noise points": int((labels == -1).sum())}
        if n_clusters > 1:
            row["Silhouette"]         = round(silhouette_score(X, labels), 4)
            row["Davies-Bouldin"]     = round(davies_bouldin_score(X, labels), 4)
            row["Calinski-Harabasz"]  = round(calinski_harabasz_score(X, labels), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Classifier helpers                                                          #
# --------------------------------------------------------------------------- #
def get_classifiers():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [1, 3, 5, 7, 9], "weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [2, 3, 4, 5, None], "min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5]},
        ),
    }


def compare_classifiers_cv(X: np.ndarray, y: np.ndarray):
    """
    Stratified K-fold CV over ALL customers (used by the live app).
    Returns (comparison_df, {model_name: cv_predictions}, n_splits).
    """
    counts = np.bincount(y)
    min_class = int(counts[counts > 0].min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        raise ValueError(
            "Every segment needs at least 2 customers for cross-validation. "
            f"Smallest segment has only {min_class} customer(s)."
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows, preds = [], {}
    for name, (model, grid) in get_classifiers().items():
        base   = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        search = GridSearchCV(model, grid, cv=cv, scoring="accuracy").fit(X, y)
        preds[name] = cross_val_predict(search.best_estimator_, X, y, cv=cv)
        rows.append(
            {
                "Model":                   name,
                "CV acc (before tuning)":  round(base, 4),
                "CV acc (after tuning)":   round(search.best_score_, 4),
                "Best params":             str(search.best_params_),
                "CV-prediction accuracy":  round(accuracy_score(y, preds[name]), 4),
            }
        )
    return pd.DataFrame(rows), preds, n_splits
