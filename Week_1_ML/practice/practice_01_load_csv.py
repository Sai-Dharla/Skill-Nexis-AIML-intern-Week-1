"""
WEEK 1 - PRACTICE QUESTION 1
Load a CSV using Pandas and print the first 10 rows.

WHAT THIS PROGRAM DOES:
    1. Loads 'data/sample_dataset.csv' using the pandas library.
    2. Displays the first 10 rows using df.head(10).
    3. Prints basic information (shape, columns) for context.

TOPIC DEMONSTRATED: Importing and exploring datasets.
"""

# --- Imports -------------------------------------------------------------
# pandas is THE library for loading and working with tabular data in Python.
import pandas as pd
from pathlib import Path

# --- Paths ---------------------------------------------------------------
# We anchor the path to THIS file's location so the program runs correctly
# no matter which folder you launch it from inside the Week_1_ML project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> Week_1_ML/
CSV_PATH = PROJECT_ROOT / "data" / "sample_dataset.csv"


def main():
    # --- Step 1: Load the CSV file ----------------------------------------
    # pd.read_csv() reads a comma-separated file into a DataFrame
    # (a DataFrame is like an in-memory spreadsheet with rows and columns).
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find the CSV file at:\n  {CSV_PATH}")
        print("Make sure 'sample_dataset.csv' exists inside the 'data' folder.")
        return

    # --- Step 2: Basic context about the dataset ---------------------------
    print("=" * 60)
    print("PRACTICE 1: LOAD A CSV AND SHOW THE FIRST 10 ROWS")
    print("=" * 60)
    print(f"\nFile loaded : {CSV_PATH.name}")
    print(f"Dataset shape (rows, columns) : {df.shape}")
    print(f"Column names                  : {list(df.columns)}")

    # --- Step 3: Show the first 10 rows (the actual requirement) -----------
    # df.head(10) returns the first 10 rows of the DataFrame.
    # By default head() shows 5 rows; passing 10 shows exactly ten.
    print("\n--- First 10 rows (df.head(10)) ---")
    print(df.head(10))


# This standard Python pattern lets the file be run directly
# with 'python practice_01_load_csv.py' and also imported safely.
if __name__ == "__main__":
    main()
