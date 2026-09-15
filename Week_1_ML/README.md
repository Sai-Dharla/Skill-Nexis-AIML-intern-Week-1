# Week 1 – Machine Learning & AI

## Topics Covered
- Machine Learning fundamentals
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
- Dataset exploration
- Missing-data handling
- Encoding
- Scaling
- Train/Test Split
- Linear Regression

## Project Tree

```
Week_1_ML/
│
├── README.md
│
├── data/
│   ├── sample_dataset.csv      # student dataset (has intentional gaps)
│   ├── house_prices.csv        # area vs price (for regression practice)
│   └── titanic.csv             # original Kaggle Titanic dataset (UNCHANGED)
│
├── practice/
│   ├── practice_01_load_csv.py
│   ├── practice_02_train_test_split.py
│   ├── practice_03_linear_regression.py
│   └── practice_04_house_price_prediction.py
│
├── assignments/
│   ├── assignment_01_dataset_exploration.py
│   ├── assignment_02_missing_data.py
│   └── assignment_03_encoding.py
│
└── mini_project/
    ├── titanic_cleaning.py
    ├── age_distribution.png    # generated output
    └── titanic_cleaned.csv     # generated output
```

## Practice Set
1. **Practice 1 – Load a CSV and print the first 10 rows** — loads
   `data/sample_dataset.csv` with pandas and shows `df.head(10)`.
2. **Practice 2 – Train/Test split** — separates features and target and
   splits them with `sklearn.model_selection.train_test_split`
   (80% train / 20% test), printing every resulting shape.
3. **Practice 3 – Linear Regression + evaluation** — trains
   `sklearn.linear_model.LinearRegression` and evaluates with **R²,
   MAE, MSE, RMSE** (regression metrics — classification accuracy is
   not used because the target is a continuous price).
4. **Practice 4 – House price prediction** — trains on
   `data/house_prices.csv`, asks the user for an area in square feet
   and predicts the price.

## Assignments
1. **Assignment 1 – Dataset exploration** — loads a dataset with pandas
   and summarizes it using `df.info()`, `df.describe()` plus shape,
   missing counts and value counts.
2. **Assignment 2 – Missing data handling** — identifies columns with
   missing values, demonstrates **mean imputation** (balanced columns)
   and **median imputation** (outlier-safe), explains the difference
   and verifies no gaps remain.
3. **Assignment 3 – Encoding** — demonstrates **LabelEncoder** on a
   binary column and **OneHotEncoder** on a nominal column, shows
   before/after views and explains when each encoder is appropriate.

## Mini Project
**Titanic Survival Prediction – Data Cleaning Project**

### Objective
Practice the full Week 1 preprocessing pipeline on a real dataset:
inspect it, clean its missing values, encode its categorical columns,
visualize a numeric column and export a machine-ready CSV.

### Dataset
- File: `data/titanic.csv` (classic Kaggle Titanic dataset, 891 rows).
- The original file is **never modified**.

### Tasks performed
1. **Clean missing data** — Age filled with the **median** (robust to
   skew/outliers), Embarked filled with the **mode** (most common
   port), Cabin column **dropped** (~77% missing values).
2. **Encoding** — `Sex` encoded with **LabelEncoder** (binary column),
   `Embarked` encoded with **OneHotEncoder** (nominal ports, avoids a
   misleading fake ordering).
3. **Visualization** — Age distribution plotted as a Seaborn histogram
   with a KDE curve, saved as `mini_project/age_distribution.png`.
4. **Export** — cleaned dataset saved as a NEW file
   `mini_project/titanic_cleaned.csv`.

### Libraries used
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- scikit-learn (encoding, split, regression)

### How to run the programs
Open a terminal in the `Week_1_ML` folder in VS Code, then:

```powershell
# one-time setup (if packages are missing)
python -m pip install pandas numpy matplotlib seaborn scikit-learn

# practice set
python practice/practice_01_load_csv.py
python practice/practice_02_train_test_split.py
python practice/practice_03_linear_regression.py
python practice/practice_04_house_price_prediction.py   # interactive: type an area
python practice/practice_04_house_price_prediction.py 1500   # or pass the area directly

# assignments
python assignments/assignment_01_dataset_exploration.py
python assignments/assignment_02_missing_data.py
python assignments/assignment_03_encoding.py

# mini project
python mini_project/titanic_cleaning.py
```

### Output files
| File | Created by |
|------|------------|
| `mini_project/age_distribution.png` | Age histogram figure |
| `mini_project/titanic_cleaned.csv` | Cleaned + encoded Titanic dataset |
| `assignments/encoded_output.csv` | Encoded dataset from Assignment 3 |
