import pandas as pd
import joblib

model = joblib.load(
    "models/priority_model.pkl"
)

columns = joblib.load(
    "models/priority_columns.pkl"
)

student = {

    "study_hours_per_day": 2,
    "attendance_percentage": 60,
    "sleep_hours": 5,
    "stress_level": 8,
    "motivation_level": 4,
    "time_management_score": 3,
    "exam_anxiety_score": 8,
    "previous_gpa": 5.2
}

df = pd.DataFrame([student])

df = pd.get_dummies(df)

df = df.reindex(
    columns=columns,
    fill_value=0
)

prediction = model.predict(df)

print(
    "Priority:",
    prediction[0]
)