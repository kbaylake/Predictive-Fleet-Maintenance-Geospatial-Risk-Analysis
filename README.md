# Predictive Fleet Maintenance & Geospatial Risk Analysis

This project is an end-to-end data science solution designed to solve a high-value business problem: **predicting and visualizing commercial vehicle failures before they happen.**

The system ingests raw truck sensor data, uses a machine learning model to predict the probability of a component failure, and plots high-risk vehicles on an interactive map.

This project was built to demonstrate a practical, hands-on understanding of the full data science lifecycle, from data cleaning and EDA to model versioning with MLOps tools.

## The Business Problem

In the logistics and supply chain industry, unplanned vehicle downtime is one of the largest and most unpredictable costs. A single vehicle failure can halt a delivery, strand a driver, and cause cascading delays across the entire supply chain, costing thousands.

## The Solution

This project builds a system to provide actionable intelligence to a fleet manager:

1.  **Predicts Failures:** A LightGBM classification model trained on a real-world (Scania) sensor dataset predicts which vehicles are at high risk of an imminent component failure.

2.  **Identifies "Where":** The system simulates fleet location data and plots the exact "failure hotspots" on an interactive map (`failure_hotspot_map.html`), allowing managers to see *where* their most at-risk assets are operating.

This allows a business to move from *reactive* maintenance (fixing what's broken) to *predictive* maintenance (fixing what's *about* to break), saving time and money.

## Key Skills & Technologies Demonstrated

This project provides direct, verifiable proof of proficiency in:

* **MLOps (MLflow):** Used to manage the complete experiment lifecycle, including logging parameters, tracking performance metrics (F1-score, Precision/Recall), and versioning models.
* **End-to-End Workflow:** Demonstrates ownership of the entire data science process from data ingestion and cleaning to a final, business-ready visualization.
* **Machine Learning (Scikit-learn, LightGBM):** Implemented a high-performance gradient-boosting model.
* **Geospatial Analysis (Folium):** Translated raw predictions into a valuable location intelligence tool (the interactive risk map).
* **Data-Driven Problem Solving:** Identified and solved critical real-world data issues, such as severe class imbalance (using SMOTE) and widespread missing data.

## Key Results & Deliverables

This project produces three key deliverables:

1.  **The Predictive Model:** A trained `LGBM` model logged and versioned in the `mlruns` directory via MLflow.
2.  **The MLflow Dashboard:** A complete dashboard showing all experiment runs, parameters, and metrics.
    * *(To add: Insert a screenshot of your `mlflow ui` here)*
3.  **The Failure Hotspot Map:** A self-contained `failure_hotspot_map.html` file that visually identifies high-risk vehicles.
    * *(To add: Insert a screenshot of your interactive map here)*

## How to Run This Project

There are two ways to explore this project:

### Option 1: Run the Python Script (Quick Run)

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy lightgbm mlflow folium matplotlib seaborn scikit-learn imbalanced-learn
    ```
2.  **Place Data:** Ensure `aps_failure_training_set.csv` and `aps_failure_test_set.csv` are in the same directory.
3.  **Run the Script:**
    ```bash
    python Predictive_Fleet_Maintenance.py
    ```
4.  **View Results:**
    * Open the generated `failure_hotspot_map.html` in your browser.
    * Run `mlflow ui` in your terminal and navigate to `http://127.0.0.1:5000` to see the experiment logs.

### Option 2: Explore the Jupyter Notebook (Detailed Explanations)

For a more detailed, step-by-step walkthrough of the data analysis, feature engineering, and modeling process (with better explanations), you can run the Jupyter Notebook.

1.  **Install Dependencies:**
    Ensure you have all the dependencies from Option 1, plus Jupyter:
    ```bash
    pip install jupyterlab
    ```
2.  **Launch Jupyter:**
    ```bash
    jupyter lab
    ```
3.  **Open the Notebook:**
    Open the `Predictive_Fleet_Maintenance.ipynb` (or your named `.ipynb` file) and run the cells sequentially to see the detailed process.
