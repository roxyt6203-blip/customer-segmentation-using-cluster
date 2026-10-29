"""
capture_screenshots.py - Automated browser testing and screenshot capture
using Playwright and Streamlit.
"""
import os
import subprocess
import time
import pathlib
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).parent.resolve()
SCREENSHOTS_DIR = BASE / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Start Streamlit server in background
env = os.environ.copy()
env["MPLCONFIGDIR"] = str((BASE / ".mpl_cache").resolve())

server_proc = subprocess.Popen(
    [
        str(BASE / ".venv" / "Scripts" / "streamlit.exe"),
        "run",
        str(BASE / "app.py"),
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ],
    cwd=str(BASE),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("Starting Streamlit server on port 8501...")
time.sleep(5)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # -------------------------------------------------------------
        # Test 3a: Tab "Full pipeline", tick "Use sample data"
        # -------------------------------------------------------------
        print("Testing 3a: Sample data pipeline...")
        # Check 'Use sample data'
        sample_cb = page.locator("text=Use sample data").first
        sample_cb.click()
        page.wait_for_timeout(4000)

        # 01_cleaning.png
        sec1 = page.locator("text=1. Data understanding and cleaning")
        sec1.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_cleaning.png"))
        print("Captured 01_cleaning.png")

        # 02_rfm.png
        sec2 = page.locator("text=2. RFM feature engineering")
        sec2.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "02_rfm.png"))
        print("Captured 02_rfm.png")

        # 03_k_selection.png
        sec3 = page.locator("text=3. Scaling and choosing K")
        sec3.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "03_k_selection.png"))
        print("Captured 03_k_selection.png")

        # 04_segments.png
        sec4 = page.locator("text=4. K-Means customer segments")
        sec4.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "04_segments.png"))
        print("Captured 04_segments.png")

        # -------------------------------------------------------------
        # Test 3d: Change K to 4, tick "Run classifier comparison"
        # -------------------------------------------------------------
        print("Testing 3d: Classifier comparison...")
        sec5 = page.locator("text=5. Model comparison and confusion matrix")
        sec5.scroll_into_view_if_needed()
        page.wait_for_timeout(1000)

        clf_cb = page.locator("text=Run classifier comparison").first
        clf_cb.click()
        # Wait for model tuning to complete (up to 30s)
        page.wait_for_timeout(15000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "05_confusion_matrix.png"))
        print("Captured 05_confusion_matrix.png")

        # -------------------------------------------------------------
        # Test 3e: Tab "Predict one customer"
        # -------------------------------------------------------------
        print("Testing 3e: Predict one customer...")
        page.locator("text=Predict one customer").first.click()
        page.wait_for_timeout(2000)

        # Test dormant: Recency ~ 125
        num_inputs = page.locator('input[type="number"]')
        if num_inputs.count() >= 3:
            num_inputs.nth(0).fill("125")
            num_inputs.nth(1).fill("298")
            num_inputs.nth(2).fill("736000000")
            page.locator("text=Predict segment").first.click()
            page.wait_for_timeout(2000)

        page.screenshot(path=str(SCREENSHOTS_DIR / "06_predict_one.png"))
        print("Captured 06_predict_one.png")

        # -------------------------------------------------------------
        # Test 3f: Tab "Saved training results"
        # -------------------------------------------------------------
        print("Testing 3f: Saved training results...")
        page.locator("text=Saved training results").first.click()
        page.wait_for_timeout(2000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "07_saved_results.png"))
        print("Captured 07_saved_results.png")

        # -------------------------------------------------------------
        # Test 3b: Upload test with raw_rfm_sales_transactions_30000.csv
        # -------------------------------------------------------------
        print("Testing 3b: Upload raw block file...")
        page.locator("text=Full pipeline (upload data)").first.click()
        page.wait_for_timeout(1500)
        # Uncheck use sample data
        sample_cb = page.locator("text=Use sample data").first
        sample_cb.click()
        page.wait_for_timeout(1500)

        raw_csv_path = r"C:\Users\raj vikash\Desktop\7th sem\ML project\ml dataset\raw_rfm_sales_transactions_30000.csv"
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(raw_csv_path)
        page.wait_for_timeout(5000)

        sec1 = page.locator("text=1. Data understanding and cleaning")
        sec1.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "08_raw_upload.png"))
        print("Captured 08_raw_upload.png")

        # -------------------------------------------------------------
        # New Tab: New Data Prediction with Online Retail
        # -------------------------------------------------------------
        print("Testing New Data Prediction tab...")
        page.locator("text=New Data Prediction").first.click()
        page.wait_for_timeout(2000)
        online_cb = page.locator("text=Use Online Retail sample").first
        online_cb.click()
        page.wait_for_timeout(10000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "09_online_retail_prediction.png"))
        print("Captured 09_online_retail_prediction.png")

        browser.close()
        print("All browser tests and screenshots completed successfully!")

finally:
    print("Terminating Streamlit server...")
    server_proc.terminate()
    server_proc.wait()
