#  Customer Credit Risk Prediction

### Holistic Data Preprocessing & Feature Engineering Project

##  Project Overview

This project focuses on building a **complete end-to-end data preprocessing pipeline** for a **Customer Credit Risk dataset**. The goal is to prepare raw data from multiple sources into a clean, structured format suitable for **Machine Learning modeling**.

As described in the project brief (Page 1), the objective is to perform:

* Data Understanding
* Data Cleaning
* Missing Value Imputation
* Outlier Handling
* Encoding
* Feature Scaling
* Feature Engineering

##  Problem Statement

You are working as a **Junior Data Scientist** in a fintech company. The task is to preprocess customer data to predict:

 **Whether a customer will default on a loan (0 = No, 1 = Yes)**

The dataset includes:

* Demographics (age, gender, region, education)
* Financial data (income, loan amount, credit score)
* Behavioral data (transactions, spending habits)

---

##  Data Sources

The project integrates data from multiple sources:

*  CSV → Customer dataset
*  JSON → External financial API data
*  SQL → Transaction history
*  API → Economic indicators

##  Tech Stack

* Python 
* Pandas, NumPy
* Scikit-learn
* SciPy
* SQLite
* YData Profiling

## Project Workflow

### 1️ Data Acquisition

* Load CSV, JSON, SQL, and API data
* Merge datasets using `customer_id`

### 2️ Data Understanding

* `.info()` and `.describe()`
* Generate **data quality report** using profiling

---

### 3️ Data Cleaning

* Convert date columns
* Extract:

  * Year
  * Month
  * Weekday

### 4️ Missing Value Handling

Applied multiple techniques (Page 3):

* Simple Imputer (mean/median/mode)
* KNN Imputer
* MICE (Iterative Imputer)
* Missing Indicator
* Complete Case Analysis

### 5️ Outlier Handling

Techniques used (Page 5):

* Z-score method
* IQR method
* Percentile capping
* Winsorization


### 6️ Feature Engineering

#### Encoding

* Ordinal Encoding → education_level
* Label Encoding → gender
* One-Hot Encoding → region, loan_purpose

####  Numerical Transformations

* Binning
* Quantile Binning
* K-Means Binning
* Binary flags (credit_score > 700)

---

### 7️ Feature Scaling

Multiple scaling techniques applied:

* StandardScaler
* MinMaxScaler
* MaxAbsScaler
* RobustScaler

### 8️ Feature Construction

New features created:

* Debt-to-Income Ratio
* Average Monthly Transactions
* Spending-to-Income Ratio

### 9️ Transformations

* Log transformation
* Reciprocal transformation
* Square root transformation
* Yeo-Johnson transformation

###  Pipeline Building

Used **ColumnTransformer + Pipeline** to apply:

* Scaling on numerical features
* Encoding on categorical features

---

##  Final Output

*  Cleaned dataset:
  `final_cleaned_credit_risk_dataset.csv`

*  Target variable:
  `default_flag`

*  Features ready for ML model training

##  Key Features of Project

Multi-source data integration
 Advanced missing value techniques
 Multiple outlier detection methods
 End-to-end feature engineering
 Scalable preprocessing pipeline

---

##  Expected Outcome (From Project)

As per the question paper (Page 6):

* Understand complete preprocessing workflow
* Apply data cleaning techniques
* Perform encoding & scaling
* Build ML-ready dataset
##  How to Run

```bash
pip install pandas numpy scikit-learn scipy ydata-profiling
```

```bash
python customer_credit_risk.py
```

##  Deliverables

*  Jupyter Notebook / Python Script
*  Final Clean Dataset
*  Data Quality Report (HTML)
*  README.md

##  Future Improvements

* Add ML models (Logistic Regression, Random Forest)
* Hyperparameter tuning
* Model evaluation (Accuracy, ROC-AUC)
* Deployment using Flask / Streamlit

