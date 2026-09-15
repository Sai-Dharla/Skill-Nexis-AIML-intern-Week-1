"""
WEEK 1 - PRACTICE QUESTION 3
Train a Linear Regression model and check its accuracy.

WHAT THIS PROGRAM DOES:
    1. Loads 'data/house_prices.csv' (Area_sqft -> Price_thousand).
    2. Splits the data into train and test sets.
    3. Trains a sklearn LinearRegression model on the training data.
    4. Predicts on the test data and evaluates the model with REGRESSION
       metrics: R^2 score, MAE, MSE and RMSE.

WHY THESE METRICS (and NOT classification accuracy):
    Linear Regression predicts a CONTINUOUS number (a price like 187.4),
    not a category. "Accuracy" (percent of exactly-correct predictions)
    only makes sense for classification. A regression prediction will
    almost NEVER match the true price to the last decimal, so accuracy
    would always be 0% and would tell us nothing.
    Instead we measure HOW CLOSE the predictions are:
      - R^2  : fraction of price variation the model explains (1.0 = perfect)
      - MAE  : average error in price units (easy to read)
      - MSE  : average of squared errors (punishes big mistakes)
      - RMSE : square root of MSE, back in price units
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> Week_1_ML/
CSV_PATH = PROJECT_ROOT / "data" / "house_prices.csv"


def main():
    # --- Step 1: Load the dataset -------------------------------------------
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find the CSV file at:\n  {CSV_PATH}")
        return

    print("=" * 60)
    print("PRACTICE 3: LINEAR REGRESSION + REGRESSION EVALUATION")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(df.head())

    # --- Step 2: Separate features and target -------------------------------
    # X must be 2-D (a DataFrame), y must be 1-D (a Series).
    # Double brackets [[...]] keep X as a DataFrame for sklearn.
    X = df[["Area_sqft"]]          # feature(s)  - input
    y = df["Price_thousand"]       # target      - output to predict

    # --- Step 3: Train/test split -------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining rows: {len(X_train)} | Testing rows: {len(X_test)}")

    # --- Step 4: Create and train the model ---------------------------------
    # LinearRegression finds the best-fitting straight line:
    #     price = (slope * area) + intercept
    model = LinearRegression()
    model.fit(X_train, y_train)   # .fit() = "learn from the training data"

    print(f"\nLearned line: price = {model.coef_[0]:.4f} * area "
          f"+ {model.intercept_:.4f}")
    print("(slope means: each extra sq ft adds this much to the price)")

    # --- Step 5: Predict on the unseen test data ----------------------------
    y_pred = model.predict(X_test)

    # Show a few real vs predicted values side by side.
    results = pd.DataFrame({"Actual": y_test.values,
                            "Predicted": y_pred.round(2)})
    print("\n--- Sample predictions on the test set ---")
    print(results.to_string(index=False))

    # --- Step 6: Evaluate with REGRESSION metrics ---------------------------
    r2 = r2_score(y_test, y_pred)                     # 1.0 is perfect
    mae = mean_absolute_error(y_test, y_pred)         # avg absolute error
    mse = mean_squared_error(y_test, y_pred)          # avg squared error
    rmse = mse ** 0.5                                 # back to price units

    print("\n--- Model evaluation (on the test set) ---")
    print(f"R^2 Score : {r2:.4f}   (closer to 1.0 is better)")
    print(f"MAE       : {mae:.2f}    (average miss, in 1000s)")
    print(f"MSE       : {mse:.2f}   (squared error units)")
    print(f"RMSE      : {rmse:.2f}    (typical miss, in 1000s)")

    # --- Step 7: Explain the metric choice -----------------------------------
    print("\nWHY THESE METRICS?")
    print("This is a REGRESSION problem (predicting a continuous price),")
    print("so classification 'accuracy' would be meaningless here - a")
    print("predicted price of 187.40 vs an actual 187.39 is an excellent")
    print("guess but counts as 'wrong' under accuracy. R^2 / MAE / RMSE")
    print("measure how CLOSE predictions are, which is the right question.")
    print(f"R^2 = {r2:.4f} means the model explains about "
          f"{r2 * 100:.1f}% of the price variation in the test data.")


if __name__ == "__main__":
    main()
