import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load(
    "models/performance_model.pkl"
)

columns = joblib.load(
    "models/model_columns.pkl"
)

importance = pd.DataFrame({

    "Feature": columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")

plt.tight_layout()

plt.show()