# Customer Segmentation of Retail Customers Using Clustering Techniques

ML (BIC703) Mini Project - East West Institute of Technology

**Team:** Chinmai M (1EW23IC013), Dharmendar Choudhari (1EW23IC014), Harshal S Govi (1EW23IC015), Vikash Kumar (1EW23IC049)

## Problem
Retail businesses treat all customers alike. This project groups customers into behaviour-based
segments using **RFM analysis** (Recency, Frequency, Monetary) and unsupervised clustering, and
provides an interactive web app supporting multiple file formats where anyone can upload transactions,
run the full pipeline, or test new datasets with the trained model.

## Datasets
1. **Project Dataset (Training / Phase 2):**
   - 30,100 raw rows from Kaggle (block-structured: a header row per customer followed by transactions).
   - After cleaning: 30,000 clean transactions, 100 unique customers, Jan-Jun 2025 (`data/cleaned_retail_transactions.csv`).
2. **Phase 1 Demonstration Raw Data:**
   - Raw Kaggle file with 100 customer header rows (`raw_rfm_sales_transactions_30000.csv`).
3. **New External Evaluation Dataset:**
   - UCI Online Retail dataset (`data/online_retail.xlsx`): 541,909 raw transactions, 4,372 unique customers across international retail operations. Automatically detects `Quantity` and `UnitPrice` to compute `Amount`.

## Supported Upload File Formats
The app and pipeline accept any of the following file types:
- **CSV, TSV, TXT** (standard delimited tables)
- **Excel (.xlsx, .xls)** (via `openpyxl`)
- **PDF (.pdf)** (extracts tabular transaction data via `pdfplumber`)
- **Word (.docx)** (extracts tables via `python-docx`)

## Pipeline
1. **Cleaning:** Forward-fill Customer ID/Region, drop 100 header rows (raw block format), convert comma-separated values to numbers, parse dates to datetime, drop nulls and non-positive transactions.
2. **Feature Engineering:** Customer-level Recency (days since last purchase), Frequency (total orders), Monetary (total spend).
3. **Feature Scaling:** `StandardScaler` to normalize RFM distributions.
4. **K Selection:** Elbow method (inertia) + Silhouette score ($K = 2 \dots 8$).
5. **Clustering Comparison:** K-Means, Agglomerative (Ward), Gaussian Mixture, and DBSCAN evaluated on Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
   - **Best Algorithm:** K-Means with $K=3$ (Silhouette score = **0.3584**, Davies-Bouldin = **0.9750**, Calinski-Harabasz = **62.4**).
6. **Segment Identification:**
   - **Active Customers:** 38 customers
   - **At-Risk / Dormant:** 32 customers
   - **High-Value Loyal:** 30 customers
7. **Supervised Classification:** Using K-Means segments as labels, models are trained and tuned via 5-fold Stratified CV with `GridSearchCV`:
   - Logistic Regression ($C=0.1$) achieves **1.00** test accuracy and **1.0000** macro-F1 on the 20-sample test set.
8. **Deployment:** Streamlit web application with 4 tabs and browser test suite.

## Web App (4 Tabs)
- **Tab 1: Full pipeline (upload data):** Upload any supported transaction file (or tick "Use sample data") to run the end-to-end pipeline: data summary, cleaning notes, RFM metrics, elbow/silhouette plots, K-Means clustering, segment profiles, business actions, and classifier comparison with confusion matrices.
- **Tab 2: New Data Prediction:** Upload new real-world data (e.g., CSV, XLSX like Online Retail, PDF, DOCX) to clean, build RFM, and predict customer segments using the trained model.
- **Tab 3: Predict one customer:** Interactive form to input Recency, Frequency, and Monetary values to get an immediate segment prediction, confidence score, and business strategy.
- **Tab 4: Saved training results:** Offline training results, comparison tables, and confusion matrix visualizations.

## Actual Results (see `results/`)
| Model | CV Acc (Before Tuning) | CV Acc (After Tuning) | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---|
| **Logistic Regression (best, C=0.1)** | **0.9750** | **0.9750** | **1.0000** | **1.0000** |
| **KNN** | 0.9250 | 0.9500 | 0.9500 | 0.9521 |
| **Decision Tree** | 0.9000 | 0.9000 | 0.9500 | 0.9521 |
| **Random Forest** | 0.9250 | 0.9375 | 0.9500 | 0.9521 |

> **Caveats & Transparency:**
> Segment labels are derived from K-Means on the same RFM features; classifier accuracy demonstrates that the segment boundaries are cleanly learnable and mathematically well-separated in RFM space, rather than accuracy against an external ground truth. The primary project dataset has 100 customers (80 train / 20 test).

## Verified Screenshots (in `docs/screenshots/`)
- `01_cleaning.png` - Data understanding and automated cleaning steps
- `02_rfm.png` - Customer-level RFM metrics and summary statistics
- `03_k_selection.png` - Elbow curve and Silhouette score vs K
- `04_segments.png` - K-Means customer segment profiles and scatter plot
- `05_confusion_matrix.png` - Cross-validated confusion matrices across models
- `06_predict_one.png` - Single customer prediction interface with business action
- `07_saved_results.png` - Saved offline training benchmarks and matrices
- `08_raw_upload.png` - Automated block format detection on raw Kaggle transactions
- `09_online_retail_prediction.png` - End-to-end evaluation on new Online Retail dataset

## Run Locally (PowerShell)
```powershell
# 1. Activate environment
.venv\Scripts\Activate.ps1
# (If execution policy blocks: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run offline training
python train.py

# 4. Launch the web app
streamlit run app.py
```

## Project Structure
```
customer-segmentation/
│
├── app.py                     # Streamlit multi-tab web application
├── pipeline.py                # Reusable ETL, multi-format loaders, RFM, & clustering
├── train.py                   # Offline model training & benchmark script
├── capture_screenshots.py     # Playwright automated browser test suite
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore file (excludes .venv, caches)
│
├── data/
│   ├── cleaned_retail_transactions.csv   # 30,000 clean transactions (100 customers)
│   ├── rfm_features.csv                  # RFM feature dataset
│   └── online_retail.xlsx                # New evaluation dataset (541k rows)
│
├── models/
│   └── segmentation_model.joblib         # Serialized scaler, kmeans, classifier
│
├── results/
│   ├── clustering_comparison.csv         # K-Means, Agglomerative, GMM, DBSCAN
│   ├── model_comparison.csv              # LR, KNN, DT, RF cross-validation metrics
│   ├── segment_profile.csv               # Mean RFM by segment
│   ├── classification_report.csv         # Precision, recall, F1 per class
│   └── *.png                             # Confusion matrices and scatter plots
│
└── docs/
    └── screenshots/                      # 9 verified browser screenshots (01 to 09)
```
