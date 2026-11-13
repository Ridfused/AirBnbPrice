
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load dataset
DATA_PATH = Path("data/raw/airbnb_india_top_500.csv")
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

# 2. Price analysis
price_col = "pricing/rate/amount"
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

print("\n========== PRICE STATISTICS ==========")
print(df[price_col].describe())

print("\n========== PRICE OUTLIERS ==========")
q1 = df[price_col].quantile(0.25)
q3 = df[price_col].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df[price_col] < lower) |
    (df[price_col] > upper)
]

print("Potential outliers:", len(outliers))
print("Lower boundary:", lower)
print("Upper boundary:", upper)

# 3. Room type analysis
print("\n========== ROOM TYPE DISTRIBUTION ==========")
print(df["roomType"].value_counts())

print("\n========== AVERAGE PRICE BY ROOM TYPE ==========")
print(
    df.groupby("roomType")[price_col]
    .agg(["count", "mean", "median", "min", "max"])
    .sort_values("median", ascending=False)
)

# 4. Location analysis
print("\n========== TOP 20 LOCATIONS ==========")
print(df["address"].value_counts().head(20))

print("\n========== AVERAGE PRICE BY LOCATION ==========")
print(
    df.groupby("address")[price_col]
    .agg(["count", "mean", "median"])
    .sort_values("count", ascending=False)
    .head(20)
)

# 5. Guest capacity
print("\n========== GUEST CAPACITY ==========")
print(df["numberOfGuests"].describe())

# 6. Geographic coverage
print("\n========== GEOGRAPHICAL COVERAGE ==========")
print("Unique locations:", df["address"].nunique())
print("Latitude range:",
      df["location/lat"].min(),
      "to",
      df["location/lat"].max())

print("Longitude range:",
      df["location/lng"].min(),
      "to",
      df["location/lng"].max())

print("\n========== EDA COMPLETED ==========")
