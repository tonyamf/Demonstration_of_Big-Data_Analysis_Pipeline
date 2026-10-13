# House Price Prediction using Hadoop and Scikit-Learn

## Abstract

This project demonstrates a comprehensive big data analytics pipeline for predicting house prices in India. The process begins with data cleaning and preprocessing using Java-based MapReduce programs on the Apache Hadoop framework. Following the initial cleaning, the project leverages Python's scientific computing stack (Pandas, Matplotlib, Scikit-learn) for in-depth exploratory data analysis (EDA), visualization, and the development of a predictive Multiple Linear Regression model. The analysis reveals key features influencing house prices and evaluates the performance of the predictive model.

---

## Project Overview

The primary goal of this project is to analyze the factors affecting house prices in India and to build a model capable of predicting these prices.

The methodology is broken down into two main stages:

1.  **Data Cleaning with Hadoop MapReduce:**
    * The raw dataset is processed in a distributed environment using a series of Java MapReduce jobs.
    * Tasks include: identifying and removing duplicate records, checking for empty features, removing constant or irrelevant columns (`BHK_OR_RK`), and converting units (Square Feet to Square Meters, INR Lacs to GBP).

2.  **Analysis and Modeling with Python:**
    * The cleaned data is loaded into a Jupyter Notebook for analysis.
    * **Exploratory Data Analysis (EDA):** Visualizations like histograms, pie charts, and scatter plots are used to understand data distributions and relationships. A correlation matrix is generated to quantify the relationships between numerical features.
    * **Predictive Modeling:** A Multiple Linear Regression model is implemented using Scikit-learn to predict the `TARGET(PRICE_IN_BGP)`. The model is trained and evaluated, and the impact of different features, including the seller type (`POSTED_BY`), is assessed.

---

## Data Origin

The dataset used is from the Kaggle competition **"Predict the house prices in India"** uploaded by user **Anmol Kumar**.

- **Link:** `https://www.kaggle.com/anmolkumar/house-price-prediction-challenge/`

---

## Key Findings

* **Feature Correlation:** The house's area (`SQUARE_M`) has the highest positive correlation with price (0.40 in the run of `house_price_analysis.py`; the original write-up said 0.41). The `RESALE` status has the strongest negative correlation (-0.21 in that run; the original write-up said -0.20).
* **Model Performance:** The linear regression models achieved a low Coefficient of Determination (R²) ranging from 0.17 to 0.34 in the re-run of `house_price_analysis.py` (the original write-up said 0.19 to 0.36; the difference is most likely due to library versions). This indicates that a linear model alone is not sufficient to accurately capture the complex factors influencing house prices in this dataset. In that re-run the Owner-only model had the highest R² (0.34); the all-listings model with `POSTED_BY` as a numeric code scored 0.24.

---

## Code Structure

### Java MapReduce (Data Cleaning)

Located in `src/DataFilfer/`, this code is intended for a Hadoop environment. It contains several MapReduce jobs:

* `DublicateChecker` / `DublicateCounter` / `DublicateRemover`: Finds, counts, and removes duplicate entries. (class and file names keep the original "Dublicate" spelling).
* `FeatureRemover`: Removes the `BHK_OR_RK` column and converts units.
* `EmptyFeatues`: Scans for rows with missing values.

The Java sources were not compiled or run in the HubRefine check (no Hadoop cluster; Maven Central was rate-limiting at the time). They import `org.apache.hadoop.*` and need the Hadoop client libraries (e.g. 3.3.x) on the classpath, e.g. `javac -cp "$(hadoop classpath)" -d build src/DataFilfer/*.java`.

### Python Script (Analysis & Prediction)

The `house_price_analysis.py` script contains the complete workflow for analysis and modeling.

---

## How to Run the Python Analysis

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
MPLBACKEND=Agg python house_price_analysis.py   # Agg = no GUI windows; omit to see the plots
```

`train.csv` (the raw Kaggle data) is read from the script's own folder. The script converts units itself (sq ft to m², INR lacs to GBP at 94.94), so it does not need the Hadoop output. It prints the correlations and the four regression results.

### Known limitations

* `house_price_analysis_notebook.ipynb` expects the **MapReduce-cleaned** CSV (columns `SQUARE_M`, `TARGET(PRICE_IN_BGP)`), which is not in the repo, and it needs `unicodecsv`. It fails with `KeyError: 'SQUARE_M'` on the raw `train.csv`.
* The Java MapReduce jobs were not run (they need a Hadoop installation).
* `DataFilter.jar` was committed in the first commit and still sits in git history (about 45 MB). It was removed from the working tree but history was not rewritten.

## Tests

```bash
pip install -r requirements-dev.txt && pytest      # Python (uses tests/fixtures, plus train.csv for one end-to-end test)
mvn test                                           # Java: mapper/reducer logic with a mocked Hadoop Context, no cluster
```
Both run in GitHub Actions (`.github/workflows/tests.yml`).
