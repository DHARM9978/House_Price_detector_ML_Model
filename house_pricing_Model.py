"""
STEP 1 : Training a model to predict house prices (Ames housing) and saving the
model pipeline and RMSE in a .pkl file

This script has 3 sections :
    1. Data loading
    2. Model development code (Pipeline: impute -> clip outliers -> scale ->
       one-hot encode (rare categories grouped automatically) -> Linear Regression)
    3. Saving the Pipeline and RMSE in a .pkl file

Command to run the complete script : python train_house_price_model.py

Single file, only built-in scikit-learn / numpy components are used, so the .pkl
can be loaded anywhere without extra helper files.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


#----------------------------------------------------------------------
#                     01 - Data loading
#----------------------------------------------------------------------

DATA_PATH = "housing_data.csv"      # <- change to your csv path

house_data = pd.read_csv(DATA_PATH)

X = house_data.drop(columns=["Order", "PID", "SalePrice"])
y = house_data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(f"X_train --> {X_train.shape} | X_test --> {X_test.shape}")


#----------------------------------------------------------------------
#                     02 - Model development code
#----------------------------------------------------------------------

# ---- Missing value rules (same as your notebook) ----
NA_Values_List = ['Alley', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1',
                  'BsmtFin Type 2', 'Electrical', 'Fireplace Qu', 'Garage Type',
                  'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence',
                  'Misc Feature']
None_Value_List = ['Mas Vnr Type']
Average_Value_List = ['Lot Frontage', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
                      'Total Bsmt SF', 'Garage Yr Blt', 'Bsmt Unf SF', 'Garage Area']
Zero_Value_List = ['Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Cars']

# ---- Column groups ----
numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

other_numeric = [c for c in numeric_features
                 if c not in Average_Value_List and c not in Zero_Value_List]
other_categorical = [c for c in categorical_features
                     if c not in NA_Values_List and c not in None_Value_List]

# Categories rarer than this share of the training rows are merged into one
# "infrequent" group automatically (replaces your hand-written 'Other' lists).
RARE_CATEGORY_THRESHOLD = 0.01


def numeric_pipe(imputer, cols):
    """impute -> clip to train 1%/99% quantiles -> standardize"""
    imputed_train = imputer.fit_transform(X_train[cols])
    lower = np.quantile(imputed_train, 0.01, axis=0)
    upper = np.quantile(imputed_train, 0.99, axis=0)
    return Pipeline([
        ("impute", clone(imputer)),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": lower, "a_max": upper})),
        ("scale", StandardScaler()),
    ])


def categorical_pipe(imputer):
    """impute -> one-hot (drop first, rare categories grouped, unseen values handled)"""
    return Pipeline([
        ("impute", imputer),
        ("onehot", OneHotEncoder(drop="first",
                                 min_frequency=RARE_CATEGORY_THRESHOLD,
                                 handle_unknown="infrequent_if_exist")),
    ])


preprocessor = ColumnTransformer([
    ("num_avg",   numeric_pipe(SimpleImputer(strategy="mean"), Average_Value_List), Average_Value_List),
    ("num_zero",  numeric_pipe(SimpleImputer(strategy="constant", fill_value=0), Zero_Value_List), Zero_Value_List),
    ("num_other", numeric_pipe(SimpleImputer(strategy="mean"), other_numeric), other_numeric),
    ("cat_na",    categorical_pipe(SimpleImputer(strategy="constant", fill_value="NA")), NA_Values_List),
    ("cat_none",  categorical_pipe(SimpleImputer(strategy="constant", fill_value="None")), None_Value_List),
    ("cat_other", categorical_pipe(SimpleImputer(strategy="most_frequent")), other_categorical),
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)

# ---- Evaluation ----
train_preds = pipeline.predict(X_train)
test_preds = pipeline.predict(X_test)

print("\n--- Training performance ---")
print(f"MSE  : {mean_squared_error(y_train, train_preds):,.0f}")
print(f"RMSE : {mean_squared_error(y_train, train_preds) ** 0.5:,.0f}")
print(f"R^2  : {r2_score(y_train, train_preds):.4f}")

mae = mean_absolute_error(y_test, test_preds)
rmse = mean_squared_error(y_test, test_preds) ** 0.5
r2 = r2_score(y_test, test_preds)

print("\n--- Testing performance ---")
print(f"Test MAE  : {mae:,.0f}")
print(f"Test RMSE : {rmse:,.0f}")
print(f"Test R^2  : {r2:.4f}")


#----------------------------------------------------------------------
#               03 - Saving the Pipeline and RMSE in a .pkl file
#---------------------------------------------------------------------

joblib.dump({"pipeline": pipeline, "rmse": rmse}, "house_price_pipeline.pkl")
print("\nSaved -> house_price_pipeline.pkl (pipeline + rmse)")