
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data/raw/airbnb_india_top_500.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/airbnb_india_clean.csv"

# Load dataset
df = pd.read_csv(RAW_FILE)

print("INITIAL DATA:", df.shape)

# Convert numeric columns
numeric_cols = [
    "pricing/rate/amount",
    "numberOfGuests",
    "location/lat",
    "location/lng",
    "stars"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicate records
df = df.drop_duplicates()

# Remove records without a valid target price
df = df.dropna(subset=["pricing/rate/amount"])

# Keep only positive prices
df = df[df["pricing/rate/amount"] > 0]

# Keep valid guest capacities
df = df[df["numberOfGuests"].between(1, 50)]

# Validate Indian geographical coordinates
valid_coordinates = (
    df["location/lat"].between(6, 38.5)
    & df["location/lng"].between(68, 98.5)
)

print("INVALID COORDINATES:", (~valid_coordinates).sum())

df = df[valid_coordinates].copy()

# Clean text columns
for col in ["address", "roomType", "name"]:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

# Create useful geographical features
df["latitude"] = df["location/lat"]
df["longitude"] = df["location/lng"]

# Rename target column
df = df.rename(
    columns={"pricing/rate/amount": "price"}
)

# Create processed directory
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n========== CLEANING COMPLETED ==========")
print("Final dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Price statistics:\n", df["price"].describe())
print("\nSaved to:", OUTPUT_FILE)
