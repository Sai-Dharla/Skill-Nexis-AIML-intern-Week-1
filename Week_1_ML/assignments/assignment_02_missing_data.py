"""
WEEK 1 - ASSIGNMENT 2
Handle missing data using mean / median imputation.

WHAT THIS PROGRAM DOES:
    1. Loads 'data/sample_dataset.csv' (it intentionally contains gaps).
    2. Identifies exactly which columns have missing values.
    3. Fills numeric gaps with the column MEAN (for fairly balanced data).
    4. Fills numeric gaps with the column MEDIAN (for skewed data with
       outliers) - demonstrated on ExamScore, which has an outlier (100).
    5. Verifies that no missing values remain.

MEAN vs MEDIAN IMPUTATION - the difference:
    MEAN   = add all values, divide by count. Every value pulls it, so ONE
             extreme value (an outlier) can drag it far away.
    MEDIAN = the middle value when sorted. Outliers barely affect it.
    RULE OF THUMB:
      - Data roughly symmetric, no big outliers -> MEAN is fine.
      - Data skewed or containing outliers      -> MEDIAN is safer.
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
    print("ASSIGNMENT 2: MISSING DATA - MEAN / MEDIAN IMPUTATION")
    print("=" * 60)

    # --- Step 2: Identify columns containing missing values ------------------
    # df.isnull() gives True/False per cell; .sum() counts Trues per column.
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    print(f"\nTotal rows in dataset: {len(df)}")
    print("\nMissing values per column:")
    print(missing_counts)

    if columns_with_missing.empty:
        print("\nNo missing values found - nothing to impute.")
        return

    print("\nColumns that contain missing values:")
    for col in columns_with_missing.index:
        print(f"  - {col}: {columns_with_missing[col]} missing "
              f"({columns_with_missing[col] / len(df):.1%} of rows)")

    # We impute NUMERIC columns only. 'City' has no gaps here; if a text
    # column did have gaps, a common approach is filling with its mode
    # (most frequent value) - shown but not needed for this dataset.
    numeric_columns = df.select_dtypes(include="number").columns
    numeric_missing = [c for c in numeric_columns if df[c].isnull().any()]

    # Work on a copy so the original loaded data stays untouched.
    df_filled = df.copy()

    # --- Step 3: MEAN imputation --------------------------------------------
    # Used for StudyHours and SleepHours: their distributions are fairly
    # balanced (no extreme outliers), so the mean is a representative
    # "typical student" value.
    if "StudyHours" in numeric_missing:
        mean_value = df["StudyHours"].mean()
        df_filled["StudyHours"] = df["StudyHours"].fillna(mean_value)
        print("\n--- MEAN imputation ---")
        print(f"Column 'StudyHours': filled missing entries with mean = "
              f"{mean_value:.2f}")

    if "SleepHours" in numeric_missing:
        mean_value = df["SleepHours"].mean()
        df_filled["SleepHours"] = df["SleepHours"].fillna(mean_value)
        print(f"Column 'SleepHours': filled missing entries with mean = "
              f"{mean_value:.2f}")

    # --- Step 4: MEDIAN imputation -------------------------------------------
    # Used for ExamScore: its mean (71.6) sits well below its median (77.0)
    # because several very low scores pull the mean towards them. The
    # median is robust to such unbalanced values, so it represents the
    # "typical" score better than the mean.
    if "ExamScore" in numeric_missing:
        mean_value = df["ExamScore"].mean()
        median_value = df["ExamScore"].median()
        df_filled["ExamScore"] = df["ExamScore"].fillna(median_value)
        print("\n--- MEDIAN imputation ---")
        print(f"Column 'ExamScore': mean = {mean_value:.2f} vs "
              f"median = {median_value:.2f}")
        print("Low scores pull the mean down; the median stays near the")
        print("typical score, so the MEDIAN is the safer fill here.")

    # Also demonstrate Age (mean - fairly balanced column).
    if "Age" in numeric_missing:
        mean_age = df["Age"].mean()
        df_filled["Age"] = df["Age"].fillna(mean_age)
        print(f"\nColumn 'Age': filled missing entries with mean = "
              f"{mean_age:.2f}")

    # --- Step 5: Verify that missing values were handled ---------------------
    print("\n--- Verification ---")
    remaining = df_filled.isnull().sum()
    print("Missing values AFTER imputation:")
    print(remaining)

    if int(remaining.sum()) == 0:
        print("\nSUCCESS: no missing values remain in the numeric columns.")
    else:
        print("\nNOTE: some missing values remain (see above).")

    # --- Side-by-side proof of what changed ----------------------------------
    rows_that_had_gaps = df[df[numeric_missing].isnull().any(axis=1)]
    print("\n--- Rows that originally had gaps (BEFORE -> AFTER) ---")
    for idx in rows_that_had_gaps.index:
        for col in numeric_missing:
            if pd.isnull(df.loc[idx, col]):
                print(f"  Row {idx}: {col} was NaN -> now "
                      f"{df_filled.loc[idx, col]:.2f}")


if __name__ == "__main__":
    main()
