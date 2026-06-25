import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(
    "dataset/cleaned_study_planner_dataset.csv"
)

# ==================================================
# SELECT FEATURES
# ==================================================

features = [

    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "motivation_level",
    "stress_level",
    "time_management_score",
    "exam_anxiety_score",
    "previous_gpa"
]

target = "exam_score"

# Keep only available columns

features = [
    col for col in features
    if col in df.columns
]

X = df[features]
y = df[target]

# ==================================================
# ENCODE CATEGORICAL DATA
# ==================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# MODEL TRAINING
# ==================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==================================================
# PREDICTION
# ==================================================

predictions = model.predict(X_test)

# ==================================================
# EVALUATION
# ==================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nMODEL PERFORMANCE")
print("=" * 40)

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("R²  :", round(r2, 4))

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(
    model,
    "models/performance_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/model_columns.pkl"
)

print("\nModel Saved Successfully")