import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/cleaned_study_planner_dataset.csv"
)

# ==========================================
# CREATE PRIORITY COLUMN
# ==========================================

df["priority"] = pd.cut(

    df["exam_score"],

    bins=[0, 60, 80, 100],

    labels=[
        "High",
        "Medium",
        "Low"
    ]
)

# ==========================================
# FEATURES
# ==========================================

features = [

    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "stress_level",
    "motivation_level",
    "time_management_score",
    "exam_anxiety_score",
    "previous_gpa"
]

features = [
    col
    for col in features
    if col in df.columns
]

X = df[features]

y = df["priority"]

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test
)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/priority_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/priority_columns.pkl"
)

print("\nPriority Model Saved")