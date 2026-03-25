# ============================================================
# COMPLETE DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE
# Customer Credit Risk Prediction Project
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import requests

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder,
    StandardScaler, MinMaxScaler,
    MaxAbsScaler, RobustScaler,
    PowerTransformer
)

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from scipy.stats import zscore
from scipy.stats.mstats import winsorize

from ydata_profiling import ProfileReport

# ============================================================
# PART B – DATA ACQUISITION
# ============================================================

# CSV
customer_df = pd.read_csv("customer_credit_risk_dataset.csv")

# JSON
api_df = pd.read_json("external_financial_api_data.json")

# SQL
conn = sqlite3.connect(":memory:")

with open("customer_transactions.sql","r") as f:
    sql_script = f.read()

conn.executescript(sql_script)

transactions_df = pd.read_sql(
    "SELECT * FROM customer_transactions",
    conn
)

print("Customer dataset:",customer_df.shape)
print("Transactions dataset:",transactions_df.shape)
print("API dataset:",api_df.shape)

# ============================================================
# PART C – DATA UNDERSTANDING
# ============================================================

print(customer_df.info())
print(customer_df.describe())

profile = ProfileReport(customer_df)
profile.to_file("data_quality_report.html")

# ============================================================
# DATE HANDLING
# ============================================================

customer_df["join_date"] = pd.to_datetime(customer_df["join_date"])
api_df["date"] = pd.to_datetime(api_df["date"],unit="ms")
transactions_df["transaction_date"] = pd.to_datetime(
    transactions_df["transaction_date"]
)

# Extract date features
customer_df["join_year"] = customer_df["join_date"].dt.year
customer_df["join_month"] = customer_df["join_date"].dt.month
customer_df["join_weekday"] = customer_df["join_date"].dt.weekday

# ============================================================
# TRANSACTION AGGREGATION
# ============================================================

txn_features = transactions_df.groupby("customer_id").agg(

total_transaction_amount=("transaction_amount","sum"),
avg_transaction_amount=("transaction_amount","mean"),
transaction_count_db=("transaction_id","count")

).reset_index()

customer_df = customer_df.merge(
txn_features,
on="customer_id",
how="left"
)

# ============================================================
# MERGE ECONOMIC INDICATORS
# ============================================================

latest_macro = api_df.tail(1)

customer_df["interest_rate"] = latest_macro["interest_rate"].values[0]
customer_df["inflation_rate"] = latest_macro["inflation_rate"].values[0]
customer_df["usd_to_inr"] = latest_macro["usd_to_inr"].values[0]
customer_df["gdp_growth_rate"] = latest_macro["gdp_growth_rate"].values[0]

# ============================================================
# MISSING VALUE HANDLING
# ============================================================

num_cols = customer_df.select_dtypes(include=np.number).columns
cat_cols = customer_df.select_dtypes(include="object").columns

# Missing indicator
customer_df["income_missing"] = customer_df["annual_income"].isnull().astype(int)

# Simple Imputer
num_imputer = SimpleImputer(strategy="median")
customer_df[num_cols] = num_imputer.fit_transform(customer_df[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
customer_df[cat_cols] = cat_imputer.fit_transform(customer_df[cat_cols])

# KNN Imputer
knn = KNNImputer(n_neighbors=5)
customer_df[num_cols] = knn.fit_transform(customer_df[num_cols])

# MICE Imputer
mice = IterativeImputer()
customer_df[num_cols] = mice.fit_transform(customer_df[num_cols])

# Complete case analysis
customer_df = customer_df.dropna()

# ============================================================
# PART D – OUTLIER HANDLING
# ============================================================

# Z-score
z = np.abs(zscore(customer_df[num_cols]))
customer_df = customer_df[(z < 3).all(axis=1)]

# IQR
Q1 = customer_df["annual_income"].quantile(0.25)
Q3 = customer_df["annual_income"].quantile(0.75)

IQR = Q3 - Q1

customer_df = customer_df[
(customer_df["annual_income"] >= Q1 - 1.5*IQR) &
(customer_df["annual_income"] <= Q3 + 1.5*IQR)
]

# Percentile method
lower = customer_df["annual_income"].quantile(0.01)
upper = customer_df["annual_income"].quantile(0.99)

customer_df = customer_df[
(customer_df["annual_income"] >= lower) &
(customer_df["annual_income"] <= upper)
]

# Winsorization
customer_df["annual_income"] = winsorize(
customer_df["annual_income"],
limits=[0.05,0.05]
)

# ============================================================
# PART E – ENCODING
# ============================================================

# Ordinal encoding
ord_cols = ["education_level"]

ord_encoder = OrdinalEncoder()

customer_df[ord_cols] = ord_encoder.fit_transform(
customer_df[ord_cols]
)

# Label encoding
le = LabelEncoder()

customer_df["gender"] = le.fit_transform(customer_df["gender"])

# One hot encoding
customer_df = pd.get_dummies(
customer_df,
columns=["region","loan_purpose"]
)

# ============================================================
# NUMERICAL ENCODING
# ============================================================

# Binning
customer_df["income_group"] = pd.cut(
customer_df["annual_income"],
bins=4
)

# Quantile binning
customer_df["income_quantile"] = pd.qcut(
customer_df["annual_income"],
4
)

# Binarization
customer_df["good_credit"] = np.where(
customer_df["credit_score"] > 700,
1,
0
)

# K-Means binning
kmeans = KMeans(n_clusters=4)

customer_df["txn_cluster"] = kmeans.fit_predict(
customer_df[["transaction_count"]]
)

# ============================================================
# FEATURE SCALING
# ============================================================

# Standardization
std_scaler = StandardScaler()

customer_df[["annual_income","loan_amount"]] = std_scaler.fit_transform(
customer_df[["annual_income","loan_amount"]]
)

# MinMax
minmax = MinMaxScaler()

customer_df["spending_ratio"] = minmax.fit_transform(
customer_df[["spending_ratio"]]
)

# MaxAbs
maxabs = MaxAbsScaler()

customer_df["transaction_count"] = maxabs.fit_transform(
customer_df[["transaction_count"]]
)

# Robust scaling
robust = RobustScaler()

customer_df["credit_score"] = robust.fit_transform(
customer_df[["credit_score"]]
)

# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

customer_df["debt_to_income"] = (
customer_df["loan_amount"] /
customer_df["annual_income"]
)

customer_df["avg_monthly_txn"] = (
customer_df["transaction_count"] / 6
)

customer_df["spending_to_income"] = (
customer_df["spending_ratio"] /
customer_df["annual_income"]
)

# ============================================================
# TRANSFORMATIONS
# ============================================================

# Log transform
customer_df["log_spending"] = np.log1p(
customer_df["spending_ratio"]
)

# Reciprocal
customer_df["reciprocal_spending"] = 1/(customer_df["spending_ratio"]+1)

# Square root
customer_df["sqrt_spending"] = np.sqrt(customer_df["spending_ratio"])

# Yeo-Johnson
pt = PowerTransformer(method="yeo-johnson")

customer_df[["loan_amount"]] = pt.fit_transform(
customer_df[["loan_amount"]]
)

# ============================================================
# COLUMN TRANSFORMER PIPELINE
# ============================================================

num_features = customer_df.select_dtypes(include=np.number).columns

cat_features = customer_df.select_dtypes(include="object").columns

preprocessor = ColumnTransformer(

transformers=[

("num",StandardScaler(),num_features),
("cat",OrdinalEncoder(),cat_features)

]

)

# ============================================================
# FINAL DATASET
# ============================================================

X = customer_df.drop(
["default_flag","customer_id","join_date"],
axis=1
)

y = customer_df["default_flag"]

customer_df.to_csv(
"final_cleaned_credit_risk_dataset.csv",
index=False
)

print("Pipeline completed successfully")
print("Final dataset shape:",customer_df.shape)