"""
WEEK 1 - ASSIGNMENT 1
Load a dataset using Pandas and summarize basic statistics.

WHAT THIS PROGRAM DOES:
    1. Loads 'data/sample_dataset.csv' with pandas.
    2. Shows dataset structure with df.info()  (dtypes, non-null counts).
    3. Shows descriptive statistics with df.describe() (mean, std, min...).
    4. Runs a few extra useful exploration commands.

TOPICS DEMONSTRATED: Dataset exploration, .info(), .describe().
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> Week_1_ML/
CSV_PATH = PROJECT_ROOT / "data" / "sample_dataset.csv"


def main():
    # --- Step 1: Load the dataset -------------------------------------------
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find the CSV file at:\n  {CSV_PATH}")
        return

    print("=" * 60)
    print("ASSIGNMENT 1: DATASET EXPLORATION WITH PANDAS")
    print("=" * 60)

    # --- Step 2: First look at the data --------------------------------------
    print("\n--- First 5 rows (df.head()) ---")
    print(df.head())

    print(f"\nDataset shape (rows, columns): {df.shape}")

    # --- Step 3: df.info() - STRUCTURE of the dataset ------------------------
    # .info() prints one line per column: how many values are NOT null
    # and the data type (int64, float64, object=text, ...).
    # It is the fastest way to spot columns that have missing values
    # (count < total rows) or wrongly-typed columns.
    print("\n--- df.info() - column names, non-null counts, dtypes ---")
    df.info()

    # --- Step 4: df.describe() - STATISTICS of numeric columns ---------------
    # .describe() summarises every numeric column with:
    #   count, mean, std (spread), min, 25%, 50% (median), 75%, max.
    print("\n--- df.describe() - descriptive statistics ---")
    print(df.describe())

    # describe() on text columns instead gives counts of unique values,
    # the most frequent value, etc. (pandas 2 accepts "object", pandas 3
    # also accepts "str" - we pass both, filtering what exists).
    print("\n--- df.describe() on text columns ---")
    print(df.describe(include=["object"]))

    # --- Step 5: A few extra useful exploration commands ---------------------
    print("\n--- Extra exploration ---")
    print(f"Column names        : {list(df.columns)}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    print(f"\nAverage exam score  : {df['ExamScore'].mean():.2f}")
    print(f"Highest exam score  : {df['ExamScore'].max():.0f}")
    print(f"Lowest exam score   : {df['ExamScore'].min():.0f}")
    print(f"Students per city   :\n{df['City'].value_counts()}")


if __name__ == "__main__":
    main()
