
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/airbnb_india_features.csv")
OUTPUT = Path("data/processed/airbnb_india_training.csv")

df = pd.read_csv(INPUT)

print("Original dataset:", df.shape)

# Fill missing state
df["state"] = df["state"].fillna("Unknown")

# Fill missing ratings with a neutral placeholder.
# Preserve information about whether a rating existed.
df["has_rating"] = df["stars"].notna().astype(int)
df["stars"] = df["stars"].fillna(0)

# Remove invalid target values
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]

# Remove duplicate rows
df = df.drop_duplicates()

# Check categorical values
df["city"] = df["city"].fillna("Unknown")
df["room_type"] = df["room_type"].fillna("Unknown")

# Save
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nTRAINING DATA PREPARATION COMPLETE")
print("Final shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSaved:", OUTPUT)
