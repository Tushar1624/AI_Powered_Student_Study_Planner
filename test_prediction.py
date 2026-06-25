import joblib
import pandas as pd

model = joblib.load(
    "models/performance_model.pkl"
)

columns = joblib.load(
    "models/model_columns.pkl"
)

sample = pd.DataFrame([{

    "study_hours_per_day": 5,
    "attendance_percentage": 85,
    "sleep_hours": 7,
    "motivation_level": 8,
    "stress_level": 3,
    "time_management_score": 8,
    "exam_anxiety_score": 3,
    "previous_gpa": 8.5

}])

sample = sample.reindex(
    columns=columns,
    fill_value=0
)

prediction = model.predict(sample)

print(prediction)