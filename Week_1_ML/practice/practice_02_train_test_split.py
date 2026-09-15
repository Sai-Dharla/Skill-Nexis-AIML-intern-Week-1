"""
WEEK 1 - PRACTICE QUESTION 2
Split a dataset into train/test using sklearn.

WHAT THIS PROGRAM DOES:
    1. Loads 'data/sample_dataset.csv' using pandas.
    2. Separates the features (inputs) from the target (output to predict).
    3. Splits them into a training set and a testing set using
       sklearn.model_selection.train_test_split.
    4. Prints the shapes/sizes so the split is easy to understand.

TOPIC DEMONSTRATED: Train/Test Split.
"""

from pathlib import Path

import pandas as pd
# train_test_split randomly divides rows into two groups:
# one group for TRAINING the model, one group for TESTING it on unseen data.
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> Week_1_ML/
CSV_PATH = PROJECT_ROOT / "data" / "sample_dataset.csv"


def main():
    # --- Step 1: Load the dataset ------------------------------------------
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find the CSV file at:\n  {CSV_PATH}")
        return

    print("=" * 60)
    print("PRACTICE 2: TRAIN / TEST SPLIT WITH SCIKIT-LEARN")
    print("=" * 60)
    print(f"\nOriginal dataset shape (rows, columns): {df.shape}")

    # --- Step 2: Prepare a clean copy for splitting -------------------------
    # This sample dataset has a few missing values. For this simple split
    # demo we remove those rows so sklearn receives complete numbers only.
    # (Proper missing-value handling is covered in Assignment 2.)
    clean_df = df.dropna(subset=["StudyHours", "SleepHours", "ExamScore"])
    print(f"After dropping rows with missing values  : {clean_df.shape}")

    # --- Step 3: Separate features (X) and target (y) -----------------------
    # X = the input columns the model will learn FROM.
    # y = the answer column the model wants to PREDICT.
    feature_columns = ["StudyHours", "SleepHours"]
    X = clean_df[feature_columns]          # DataFrame of inputs
    y = clean_df["ExamScore"]              # Series of answers

    # --- Step 4: Perform the train/test split -------------------------------
    # test_size=0.2  -> 20% of rows go to the test set, 80% to training.
    # random_state=42 -> makes the random split the same every run
    #                    so results are reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Step 5: Display the resulting shapes so the split is clear ---------
    total_rows = len(X)
    print("\n--- Split results (80% train / 20% test) ---")
    print(f"X (features)   shape: {X.shape}   <- all {total_rows} rows")
    print(f"y (target)     shape: {y.shape}   <- all {total_rows} rows")
    print(f"X_train shape: {X_train.shape}  <- {len(X_train)} rows "
          f"({len(X_train) / total_rows:.0%} of data)")
    print(f"X_test  shape: {X_test.shape}   <- {len(X_test)} rows "
          f"({len(X_test) / total_rows:.0%} of data)")
    print(f"y_train shape: {y_train.shape}  <- {len(y_train)} answers")
    print(f"y_test  shape: {y_test.shape}   <- {len(y_test)} answers")

    # --- Step 6: Peek at the actual split rows ------------------------------
    print("\n--- X_train (first 5 rows) ---")
    print(X_train.head())
    print("\n--- X_test (all rows, held out for testing) ---")
    print(X_test)

    # WHY SPLIT? A model that is graded on data it has already seen can look
    # better than it really is. The test set simulates brand-new, unseen data.


if __name__ == "__main__":
    main()
