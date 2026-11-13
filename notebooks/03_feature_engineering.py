
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
INPUT_FILE = Path("data/processed/airbnb_india_clean.csv")
OUTPUT_FILE = Path("data/processed/airbnb_india_features.csv")

print("=" * 50)
print("AIRBNB INDIA - FEATURE ENGINEERING")
print("=" * 50)

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:", df.shape)

# 1. Remove duplicate records
df = df.drop_duplicates()

# 2. Convert numerical columns
numeric_cols = [
    "latitude",
    "longitude",
    "numberOfGuests",
    "price",
    "stars"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Remove invalid prices
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]

# 4. Keep geographically valid Indian listings
before_geo = len(df)

df = df[
    df["latitude"].between(6, 38) &
    df["longitude"].between(68, 98)
].copy()

print("\nListings removed due to invalid coordinates:",
      before_geo - len(df))

# 5. Extract city and state from address
address_parts = (
    df["address"]
    .fillna("")
    .str.split(",")
)

df["city"] = address_parts.str[0].str.strip()
df["state"] = address_parts.str[1].str.strip()

# 6. Clean room type
df["roomType"] = (
    df["roomType"]
    .fillna("Unknown")
    .str.strip()
)

# 7. Convert Superhost into binary feature
def convert_superhost(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["true", "yes", "1"]:
        return 1
    if value in ["false", "no", "0"]:
        return 0

    return np.nan

df["is_superhost"] = (
    df["isHostedBySuperhost"]
    .apply(convert_superhost)
)

# 8. Remove unnecessary listing identifiers/text
df = df.drop(
    columns=["name", "isHostedBySuperhost"],
    errors="ignore"
)

# 9. Rename columns consistently
df = df.rename(columns={
    "numberOfGuests": "max_guests",
    "roomType": "room_type"
})

# 10. Keep model features and target
features = [
    "city",
    "state",
    "latitude",
    "longitude",
    "max_guests",
    "room_type",
    "stars",
    "is_superhost",
    "price"
]

df = df[features]

# 11. Save feature-engineered dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# 12. Display results
print("\nFEATURE ENGINEERING COMPLETED")
print("=" * 50)

print("\nFinal dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nRoom type distribution:")
print(df["room_type"].value_counts())

print("\nCity distribution:")
print(df["city"].value_counts().head(15))

print("\nPrice statistics:")
print(df["price"].describe())

print("\nSaved to:", OUTPUT_FILE)

