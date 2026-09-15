"""
WEEK 1 - ASSIGNMENT 3
Encode categorical variables using LabelEncoder and OneHotEncoder.

WHAT THIS PROGRAM DOES:
    1. Builds a small clearly-labelled categorical dataset (no CSV needed).
    2. Shows the data BEFORE encoding.
    3. Demonstrates LabelEncoder  on the 'Gender' column (binary).
    4. Demonstrates OneHotEncoder on the 'City' column (nominal, 3+ values).
    5. Shows the data AFTER each encoding.
    6. Explains WHEN to use which encoder.

LABEL ENCODER vs ONE-HOT ENCODER - when to use which:
    LabelEncoder turns each category into one integer (Male->1, Female->0).
      USE IT for: binary columns (only 2 categories) or truly ordered
      categories (Small < Medium < Large).
      AVOID IT for nominal categories with 3+ values: the model would
      invent a false ranking, e.g. Delhi(0) < Mumbai(1) < Chennai(2),
      which is misleading - no city is "greater" than another.

    OneHotEncoder creates one new 0/1 column per category.
      USE IT for: nominal (unordered) categories with 3+ values, because
      no artificial ordering is created.
      TRADE-OFF: it adds columns (one per category).
"""

from pathlib import Path

import pandas as pd
# OneHotEncoder produces a sparse matrix by default; sparse_output=False
# returns a normal dense array that is easy to wrap back into a DataFrame.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # -> Week_1_ML/


def build_sample_data():
    """Small categorical dataset for demonstrating both encoders."""
    return pd.DataFrame({
        "StudentID": [1, 2, 3, 4, 5, 6, 7, 8],
        "Gender": ["Male", "Female", "Female", "Male",
                   "Female", "Male", "Female", "Male"],
        "City": ["Chennai", "Delhi", "Mumbai", "Delhi",
                 "Chennai", "Mumbai", "Delhi", "Chennai"],
        "Enrolled": ["Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes"],
        "ExamScore": [78, 85, 55, 92, 48, 88, 61, 74],
    })


def main():
    print("=" * 60)
    print("ASSIGNMENT 3: ENCODING - LABEL ENCODER & ONE-HOT ENCODER")
    print("=" * 60)

    df = build_sample_data()

    # --- Step 1: Show the data BEFORE encoding -------------------------------
    print("\n--- BEFORE encoding (raw categorical data) ---")
    print(df)

    # =========================================================================
    # PART A - LABEL ENCODER (on 'Gender', a binary column)
    # =========================================================================
    print("\n" + "-" * 60)
    print("PART A: LABEL ENCODER on 'Gender' (binary column)")
    print("-" * 60)

    label_encoder = LabelEncoder()
    df["Gender_encoded"] = label_encoder.fit_transform(df["Gender"])

    print(f"Learned mapping: "
          f"{dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    print("\nGender column AFTER LabelEncoding:")
    print(df[["Gender", "Gender_encoded"]])

    # =========================================================================
    # PART B - ONE-HOT ENCODER (on 'City', a nominal column with 3 values)
    # =========================================================================
    print("\n" + "-" * 60)
    print("PART B: ONE-HOT ENCODER on 'City' (nominal column, 3 categories)")
    print("-" * 60)

    # OneHotEncoder expects a 2-D input, hence double brackets.
    onehot_encoder = OneHotEncoder(sparse_output=False)
    city_encoded = onehot_encoder.fit_transform(df[["City"]])

    # The encoder tells us the new column names, e.g. 'City_Chennai'.
    encoded_columns = onehot_encoder.get_feature_names_out(["City"])
    city_encoded_df = pd.DataFrame(city_encoded,
                                   columns=encoded_columns,
                                   index=df.index)

    print(f"Categories found: {list(onehot_encoder.categories_[0])}")
    print("Each category became its own 0/1 column:")
    print(city_encoded_df)

    # --- Step 2: Final result - raw + encoded side by side -------------------
    df_final = pd.concat([df, city_encoded_df], axis=1)
    print("\n--- AFTER encoding (full view) ---")
    print(df_final)

    # =========================================================================
    # PART C - WHY 'City' MUST NOT BE LABEL-ENCODED
    # =========================================================================
    print("\n--- WHY LabelEncoder on 'City' would be WRONG ---")
    demo = LabelEncoder()
    wrong_codes = demo.fit_transform(df["City"])
    print(f"LabelEncoder would give: "
          f"{dict(zip(demo.classes_, range(len(demo.classes_))))}")
    print("The model would read this as Delhi(0) < Chennai(1) < Mumbai(2),")
    print("a ranking that does not exist in reality. OneHotEncoder avoids")
    print("this by using separate 0/1 columns with no implied order.")

    print("\nSUMMARY")
    print("  LabelEncoder  -> binary or ordered categories (e.g. Gender).")
    print("  OneHotEncoder -> nominal categories with 3+ values (e.g. City).")

    # Optional: save the encoded result so the output is inspectable.
    out_path = PROJECT_ROOT / "assignments" / "encoded_output.csv"
    df_final.to_csv(out_path, index=False)
    print(f"\nEncoded dataset saved to: {out_path.name}")


if __name__ == "__main__":
    main()
