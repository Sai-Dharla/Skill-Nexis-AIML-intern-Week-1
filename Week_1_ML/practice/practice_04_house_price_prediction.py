"""
WEEK 1 - PRACTICE QUESTION 4
Predict house price based on an area input from the user.

WHAT THIS PROGRAM DOES:
    1. Loads 'data/house_prices.csv' (Area_sqft -> Price_thousand).
    2. Trains a Linear Regression model on ALL of the data
       (using every row gives the best final predictor).
    3. Asks the user to type an area in square feet.
    4. Predicts and displays the corresponding house price.

HOW TO RUN (from the Week_1_ML folder in the VS Code terminal):
    python practice/practice_04_house_price_prediction.py
    ...then type an area when asked, e.g. 1500

    You may also pass the area directly as an argument (no prompt):
    python practice/practice_04_house_price_prediction.py 1500
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> Week_1_ML/
CSV_PATH = PROJECT_ROOT / "data" / "house_prices.csv"


def predict_price_for_area(area_value, model, feature_name):
    """Return the model's price prediction for one area value."""
    # sklearn expects a 2-D input (rows x features), even for ONE row.
    # We build a one-row DataFrame (with the same column name used in
    # training) so sklearn recognises the feature correctly.
    return model.predict(pd.DataFrame({feature_name: [area_value]}))[0]


def main():
    # --- Step 1: Load the dataset -------------------------------------------
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find the CSV file at:\n  {CSV_PATH}")
        return

    print("=" * 60)
    print("PRACTICE 4: HOUSE PRICE PREDICTION FROM AREA INPUT")
    print("=" * 60)
    print(f"\nTraining on {len(df)} house records from {CSV_PATH.name}...")

    # --- Step 2: Features (area) and target (price) --------------------------
    X = df[["Area_sqft"]]        # input feature  (2-D DataFrame)
    y = df["Price_thousand"]     # target         (1-D Series)

    # --- Step 3: Train the model on ALL available data -----------------------
    model = LinearRegression()
    model.fit(X, y)
    print(f"Learned relationship: price = {model.coef_[0]:.4f} * area "
          f"+ {model.intercept_:.4f}")

    # --- Step 4: Get the area from the user ----------------------------------
    # If an area was passed on the command line, use it; otherwise prompt.
    # (The argument option also lets the script run without typing live.)
    if len(sys.argv) > 1:
        area_text = sys.argv[1]
    else:
        area_text = input("\nEnter the area of the house in square feet: ")

    # --- Step 5: Validate the input and predict ------------------------------
    try:
        area_value = float(area_text)
        if area_value <= 0:
            print("ERROR: Area must be a positive number, e.g. 1500")
            return
    except ValueError:
        print(f"ERROR: '{area_text}' is not a valid number. "
              "Please enter digits, e.g. 1500")
        return

    predicted_price = predict_price_for_area(area_value, model, "Area_sqft")

    # --- Step 6: Display the result clearly ----------------------------------
    print("\n" + "-" * 60)
    print(f"Area entered      : {area_value:,.0f} sq ft")
    print(f"PREDICTED PRICE   : {predicted_price:,.2f} thousand")
    print(f"                  = Rs. {predicted_price * 1000:,.0f}")
    print("-" * 60)
    print("(Prices are in thousands, matching the training data's "
          "'Price_thousand' column. Prediction is a straight-line "
          "estimate, not a guarantee.)")


if __name__ == "__main__":
    main()
