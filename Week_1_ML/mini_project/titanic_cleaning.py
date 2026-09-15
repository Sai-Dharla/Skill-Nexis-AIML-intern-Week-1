"""
WEEK 1 - MINI PROJECT
Titanic Survival Prediction - Data Cleaning Project

DATASET:
    data/titanic.csv  (the classic Kaggle Titanic passenger dataset,
    891 rows). The original file is NEVER modified - all outputs are
    written as NEW files.

WHAT THIS PROGRAM DOES:
    Task 1: Inspect and clean missing values
              - Age     -> filled with the MEDIAN (robust to outliers/skew)
              - Embarked-> filled with the MODE (most common port, only 2 gaps)
              - Cabin   -> column DROPPED (~77% missing, too sparse to fill)
    Task 2: Encode Sex (LabelEncoder) and Embarked (OneHotEncoder)
    Task 3: Plot the Age distribution (Seaborn histogram) -> age_distribution.png
    Task 4: Save the cleaned dataset as a NEW CSV -> titanic_cleaned.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # save figures to files without a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> Week_1_ML/
DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"
OUTPUT_DIR = Path(__file__).resolve().parent         # -> mini_project/
CLEANED_CSV = OUTPUT_DIR / "titanic_cleaned.csv"
AGE_PLOT_PNG = OUTPUT_DIR / "age_distribution.png"


def load_dataset():
    """Load the Titanic CSV and explain clearly if it is missing."""
    if not DATA_PATH.exists():
        print("=" * 60)
        print("DATASET NOT FOUND")
        print("=" * 60)
        print(f"The program expects the Titanic dataset at:\n  {DATA_PATH}")
        print("\nWhat to do:")
        print("  1. Download 'titanic.csv' from Kaggle (Titanic competition).")
        print("  2. Place it inside the 'data' folder of Week_1_ML.")
        print("  3. Keep the filename exactly: titanic.csv")
        print("  4. Run this program again.")
        return None
    return pd.read_csv(DATA_PATH)


def main():
    # ========================================================================
    # TASK 0 - LOAD AND INSPECT THE DATASET
    # ========================================================================
    df = load_dataset()
    if df is None:
        return  # cannot continue without the dataset

    print("=" * 60)
    print("TITANIC DATA CLEANING PROJECT")
    print("=" * 60)
    print(f"\nLoaded dataset shape (rows, columns): {df.shape}")
    print("\nColumns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3))

    # --- Inspect missing values ----------------------------------------------
    # df.isnull().sum() counts missing (NaN) entries per column.
    missing = df.isnull().sum()
    print("\n--- TASK 1: MISSING VALUE INSPECTION ---")
    print("Missing values per column:")
    print(missing)
    print("\nMissing values as % of rows:")
    print((missing / len(df) * 100).round(1))

    # ========================================================================
    # TASK 1 - CLEAN THE MISSING DATA
    # ========================================================================
    cleaned = df.copy()   # never touch the original DataFrame

    # --- Age: fill with MEDIAN ------------------------------------------------
    # Age is right-skewed (many young passengers, few very old ones) and the
    # median is robust to that skew and to outliers, so it represents the
    # 'typical passenger' better than the mean would.
    median_age = cleaned["Age"].median()
    cleaned["Age"] = cleaned["Age"].fillna(median_age)
    print(f"\nAge      -> filled missing values with median = {median_age:.1f}")

    # --- Embarked: fill with MODE ---------------------------------------------
    # Embarked (port of boarding: S/C/Q) is a TEXT column, so a mean/median
    # is impossible. The mode (most frequent port) fills its 2 missing
    # values with the most likely answer.
    mode_embarked = cleaned["Embarked"].mode()[0]
    cleaned["Embarked"] = cleaned["Embarked"].fillna(mode_embarked)
    print(f"Embarked -> filled missing values with mode = '{mode_embarked}'")

    # --- Cabin: DROP the column -----------------------------------------------
    # ~77% of Cabin values are missing - guessing so many values would
    # fabricate data, so dropping the column is the honest choice.
    cleaned = cleaned.drop(columns=["Cabin"])
    print("Cabin    -> column DROPPED (too many missing values to impute)")

    # Verify the cleaning worked.
    print("\nMissing values AFTER cleaning:")
    print(cleaned.isnull().sum())
    print(f"Cleaned shape: {cleaned.shape}")

    # ========================================================================
    # TASK 2 - ENCODE Sex AND Embarked
    # ========================================================================
    print("\n--- TASK 2: ENCODING Sex AND Embarked ---")

    # --- Sex: LabelEncoder (binary column -> 2 codes is fine) ------------------
    sex_encoder = LabelEncoder()
    cleaned["Sex_encoded"] = sex_encoder.fit_transform(cleaned["Sex"])
    print(f"Sex mapping: "
          f"{dict(zip(sex_encoder.classes_, range(len(sex_encoder.classes_))))}")

    # --- Embarked: OneHotEncoder ----------------------------------------------
    # S, C, Q are NOMINAL ports - no true order exists, so one-hot columns
    # avoid inventing a misleading ranking like S(2) > Q(1) > C(0).
    embarked_encoder = OneHotEncoder(sparse_output=False)
    embarked_codes = embarked_encoder.fit_transform(cleaned[["Embarked"]])
    embarked_columns = embarked_encoder.get_feature_names_out(["Embarked"])
    embarked_df = pd.DataFrame(embarked_codes,
                               columns=embarked_columns,
                               index=cleaned.index)

    # Join the new 0/1 columns and remove the raw text columns.
    cleaned = pd.concat([cleaned, embarked_df], axis=1)
    cleaned = cleaned.drop(columns=["Sex", "Embarked"])

    print("Embarked -> one-hot columns created:", list(embarked_columns))
    print("\nCleaned + encoded dataset (first 5 rows):")
    print(cleaned.head())

    # ========================================================================
    # TASK 3 - VISUALIZE THE AGE DISTRIBUTION
    # ========================================================================
    print("\n--- TASK 3: AGE DISTRIBUTION VISUALIZATION ---")
    plt.figure(figsize=(9, 5))
    # Histogram of ages with a smooth KDE curve overlaid for readability.
    sns.histplot(cleaned["Age"], bins=30, kde=True, color="steelblue")
    plt.title("Distribution of Passenger Ages - Titanic Dataset", fontsize=13)
    plt.xlabel("Age (years)")
    plt.ylabel("Number of Passengers")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(AGE_PLOT_PNG, dpi=150)   # save the figure as a PNG file
    print(f"Age distribution chart saved to: {AGE_PLOT_PNG.name}")

    # ========================================================================
    # TASK 4 - SAVE THE CLEANED DATASET AS A NEW CSV
    # ========================================================================
    print("\n--- TASK 4: SAVE CLEANED DATASET ---")
    cleaned.to_csv(CLEANED_CSV, index=False)  # index=False: no extra numbering

    # Verify the output file exists and report its size.
    if CLEANED_CSV.exists():
        size_kb = CLEANED_CSV.stat().st_size / 1024
        print(f"SUCCESS: '{CLEANED_CSV.name}' created "
              f"({size_kb:.1f} KB, {cleaned.shape[0]} rows, "
              f"{cleaned.shape[1]} columns).")
    else:
        print("ERROR: the cleaned CSV was not created - check permissions.")

    print("\nPROJECT COMPLETE - original titanic.csv was NOT modified.")


if __name__ == "__main__":
    main()
